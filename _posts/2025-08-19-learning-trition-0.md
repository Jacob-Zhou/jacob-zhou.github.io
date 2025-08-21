---
layout:     post
title:      "Triton 学习手记 （一）：基本概念"
date:       2025-08-19 19:00:0+0800
author:     "Houquan Zhou"
header-img: "/assets/img/post-bg.jpg"
mathjax: true
catalog: true
published: false
tags:
    - Triton
---

# 前言

什么是 Triton？
Triton 的[官网文档](https://triton-lang.org/main/index.html) 中是这样介绍的：
> Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.

简单来说 Triton 是一个允许你在 Python 中自定义如何在 GPU 上并行计算的包。

我们知道在神经网络经常涉及大量的运算，其中有非常多的运算是可以并行处理的。例如对一个 $1024 \times 1024$ 的矩阵进行 ReLU 运算，由于 ReLU 运算对每个元素都是独立的，因此可以将这个矩阵切分为数个小的矩阵，然后并行地对这些小矩阵进行 ReLU 运算。

事实上 pytorch、tensorflow 等深度学习框架提供了大量的函数，这些函数的背后都使用了 CUDA 及其类似的技术来实现并行加速。但是他们的并行化细节对用户隐藏，用户只能使用这些函数，而不能控制并行化细节。

假设我们现在构思出了一个全新的函数，我们希望并行优化这个函数，但是这个函数又不是 pytorch 或 tensorflow 中的函数，我们该怎么办？

这时候 Triton 就派上用场了。他提供了一个相对简单的语言（Python + 类 numpy 的语法），让用户可以在这个语言中控制 GPU 的并行化和运算细节，充分挖掘 GPU 的潜力。

这篇 Blog 是我学习 Triton 的笔记。作为初学者，我不可避免地会犯一些错误，欢迎大家指正。

# Triton 的基本概念

## 单程序多数据
第一个概念是**单程序多数据**。

Triton 使用单程序多数据（Single Program Multiple Data, SPMD）的编程模型。
也就是说在并发的时候，在同一时间运行的是**同一份程序**。这份代码在运行时通过通过某些系统提供的标识符来区分自己要**处理哪份数据**。

在 Triton 中，这份程序通常被称作**kernel**。下面是一个简单的例子：

<span id="zero-kernel"></span>

```python
import triton.language as tl
import triton
import torch

# 修饰符，表示这是一个 Triton Kernel
@triton.jit
def zero_kernel(x_ptr):
    # 获取当前程序的编号
    pid = tl.program_id(0)
    # 将当前程序的编号对应的元素设置为 0
    offset = pid
    tl.store(x_ptr + offset, 0)

if __name__ == "__main__":
    n = 128
    # 创建一个长度为 n 的空 tensor，并将其移动到 GPU 上
    x = torch.empty(n).to("cuda")
    print(x) # 打印未初始化前的 x
    grid = (n, )
    zero_kernel[grid](x)
    print(x) # 打印初始化后的 x
```

这个例子展示了 SPMD 的核心思想：在运行时，调度程序会根据 `grid` 启动 $n$ 个 `zero_kernel`，第 $i$ 个实例通过 `tl.program_id(0)` 知道自己的编号是 $i$，然后去设置第 $i$ 个元素。

如果觉得还是很难理解，我们不妨换个思路：

```python
@triton.jit                         #
def zero_kernel(x_ptr):             # def zero(x, pid):
    pid = tl.program_id(0)          #
    pass                            #     pass

if __name__ == "__main__":          # if __name__ == "__main__":
    n = 128                         #     n = 128
    x = torch.empty(n).to("cuda")   #     x = torch.empty(n).to("cuda")
    grid = (n, )                    #     for pid in range(n):
    zero_kernel[grid](x)            #         zero(x, pid)
```

我们还是依照单线程循环的思路来写代码，只不过在 Triton 中，编译器自动根据 `grid` 来帮我们执行了“循环”。

Triton 允许我们设置最多三维的 `grid`，例如：
```python
grid = (n, m, l)                    # for pid_n in range(n):
                                    #     for pid_m in range(m):
                                    #         for pid_l in range(l):
kernel[grid](*args)                 #             func(*args, pid_n, pid_m, pid_l)
```

**注**：调研了一些资料，目前暂时没有找到 `grid` 维度设置的最佳实践，似乎高维只是为了方便用户理解。

## Tensor 的读取和存储
在阅读前面的[zero-kernel](#zero-kernel)时，你可能已经注意到，在`zero`函数中，我们传递向 `zero_kernel` 传递了一个 tensor，但是为什么 `zero_kernel` 的参数是 `x_ptr`？
为什么 `x_ptr + offset` 表示第 `offset` 个元素？
为什么不是像 Pytorch 那样直接使用 `x[offset]`？

下面我们介绍 Triton 中的 Tensor 的读取和存储。

和 Pytorch 编程不太一样，Tensor 是以**指针**的形式传递给 Triton Kernel 的。这一设计的初衷是为了**方便用户可以更精确地控制读取哪些元素**。

**指针**可以理解为一个整数，它指向了显存中的一个位置。我们可以通过对指针进行加减运算来访问显存中的不同位置。

TODO

**输入连续性**：（如果输入不是连续的，那么在 kernel 中读取的时候就会出现问题。）

**安全**：在 Kernel 中，我们可以通过指针访问到整个显存空间。如果不多加小心，例如没有检查边界时，计算错误的偏移量，可能会导致 Kernel 访问到不属于它的数据，甚至修改其他数据。因此在读写，尤其是写的时候，需要格外注意检查边界。