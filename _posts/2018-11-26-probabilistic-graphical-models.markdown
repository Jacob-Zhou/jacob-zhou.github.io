---
layout:     post
title:      "概率图模型"
date:       2018-11-26 10:32:00 +0800
author:     "Zo"
header-img: "assets/img/probabilistic-graphical-models/post-bg.jpg"
mathjax: true
catalog: true
tags:
    - 概率图
---

# 前言
11月22号的时候鼓起勇气来将HMM和CRF里面重要的几个公式推了一下，似乎明白了些什么。最近发现有些东西光是读和看，会以为自己已经懂了，但实际上这只是假象。要真自己动手就才会发现自己其实什么都不懂。所以试着用这种方法巩固知识和分享自己的一点见解。

# 概率图模型
#### 概率模型的意义
概率图模型是
> a general-purpose framework for constructing and using probabilistic models of complex systems.  
> 一种能够构建和利用复杂系统概率模型的通用框架。[^1]  

在概率模型中我们往往要处理数量巨大的随机变量，并对他们的联合分布进行建模  
*知道了联合分布我们就可以通过计算得到任意的边缘分布与条件分布*

$$\begin{equation}P(A_1, A_2, A_3, \dotsc, A_i, \dotsc,A_n)\end{equation}$$

我们假设 $k_i=\|Val(A_i)\|$ 为每个随机变量可能的取值个数，那么包含 $n$ 个随机变量的联合分布就可能有 $\prod_i^n k_i$ 种可能取值。我们取最极端的情况，所有的随机变量都只有 $2$ 种类取值，那么 $n$ 个随机变量的联合分布的取值也会是 $2^n$ 种，这对我们训练概率模型和使用概率模型来推理是十分不利的。  
既然我们无法一次性处理那么多随机变量的联合分布，那么我们有没有什么办法将其中的随机变量进行分割，将这个大问题化成许多个小问题呢？或者说，我们如何描述这些随机变量之间的（独立）关系呢？答案肯定是有的，两种概率图模型为我们提供了这种可能。
#### 图论与概率论的几个重要概念
**图中的亲属关系**{: #graphic-relatives}

![graphic-relatives](assets/img/probabilistic-graphical-models/graphic-relatives.png)

**链式法则**{: #chain-rule}  

$$\begin{equation}P(A_1, \dotsc, A_n)=P(A_1)P(A_2|A_1)P(A_3|A_2,A_1)\dotsm P(A_n|A_{n-1},\dotsc,A_1)\end{equation}$$

#### 概率的图化
首先我们需要了解如何使用图来表示概率。我们通过一个例子来说明：**T公司**打算招应届生，他们希望招到的员工有很强的**代码能力** $C$，所以他们想到了通过在线笔试来获得**笔试成绩** $G$，以此来确定学生的代码能力。
于是他们首先构造了一个只有两个随机变量 $C,G$ 的模型。首先我们将每个随机变量作为一个结点。接着我们要确定随机变量间的关系。根据直觉，我们知道是**代码能力**在很大程度上决定了**笔试成绩**。我们将这样一个模型 $P(C,G)$ 用概率图模型表现出来如下

![code-to-grade](assets/img/probabilistic-graphical-models/code-to-grade.png)

这个简单的有向图为我们给出了概率图模型的第一个好处：让我们可以将一个关于 $n$ 个随机变量的联合分布可以分解为多个紧凑的因子

$$ \begin{equation}P(C,G)=P(C)P(G|C)\end{equation} $$

在前面的例子中，**T公司**发现单纯使用**笔试成绩**就来判断**代码能力**的高低有点过于武断。他们在应聘者的简历上给出了他们技术博客的地址，在博客中可以看到**点赞数量** $L$，我们假设**点赞数量**只和**博客质量** $Q$ 有关，而要写出一篇高质量的博客不但要求**代码能力**，还要求他们的**表达能力** $E$。为此面试官对模型进行升级改造：

![recruit-model](assets/img/probabilistic-graphical-models/recruit-model.svg){: #recruit-model}

我们使用[链式法则](#chain-rule)可以轻易地将模型的联合分布变成多个条件概率的乘积的形式

$$ \begin{equation}P(C,G,E,Q,L)=P(C)P(G|C)P(E|C,G)P(Q|C,G,E)P(L|C,G,E,Q)\end{equation} $$

但是我们更具概率图可以写出更加简单的式子

$$ \require{cancel}
\begin{equation}\begin{split}
P(C,G,E,Q,L)&=P(C)P(G|C)P(E|\cancel{C,G})P(Q|C,E\cancel{,G})P(L|\cancel{C,E,G,}Q) \\
&=P(C)P(G|C)P(E)P(Q|C,E)P(L|Q)
\end{split}\end{equation}
$$

我们很容易可以在概率图中找到每一项乘积的对应

![factor-of-model](assets/img/probabilistic-graphical-models/factor-of-model.png)

我们可以看到一般地，我们可以得到这样的公式

$$ \begin{equation}P(A_1,\dotsc,A_n)=\prod_{i=1}^n P(X_i|\mathbf{Pa}_{X_i}^\mathcal{G})\end{equation} $$

其中因子 $P(X_i\|\mathbf{Pa}_{X_i}^\mathcal{G})$ 就是我们常说的*条件概率分布*，我们通过模型将链式法则以一种更加清晰、可理解的方式呈现出来。这正是因为概率图模型为我们提供的第二个好处：明确地给出了模型中所蕴含的独立关系假设。
我们把形如[上图](#recruit-model)的有向无圈图（DAG）$\mathcal{G}$，叫做贝叶斯网，结点为随机变量，边表示一个结点对另外一个结点的直观影响。

>可以以两种截，然不同的方式看待图 $\mathcal{G}$:
> 1. 它是提供了以因子分解的方式紧凑表示联合分布骨架的数据结构  
> 2. 它是关于一个分布的一系列条件独立性假设的紧凑表示  
>
>在严格意义上，这两种观点是等价的。[^1]

前面介绍了有向图下面我们介绍如何使用无向图来描述概率。  
并不是所有的独立性描述都可以使用贝叶斯网来描述，一方面是因为贝叶斯网中不允许环的存在，从直观上来说是因为贝叶斯网为每一种影响都指定了因果方向，因此当我们在描述两个随机变量相互影响的时候就会出现问题  

如下面这个例子  

![markov-model-exmple](assets/img/probabilistic-graphical-models/markov-model-exmple.svg)

一个矩形水缸被划分为 $A, B, C, D$ 四个区域，区域被玻璃板隔开。但相邻两个区域间的玻璃板上有缺口，允许分子自由地穿过。这就意味着如果发现一个区域被污染了，那么因为分子的自由扩散，它相邻的区域也有可能会受到污染。  
在这样的例子中存在着这样的独立假设，在知道 $B,D$ 区域是否受污染的时候， $A,C$ 之间不会有影响；同理在知道 $A,C$ 区域是否受污染的时候， $B,D$ 之间不会有影响，除了这两个独立假设外不再存在别的独立假设即

$$ \begin{equation}\begin{cases}
    & (A\bot C|\{B,D\})\\
    & (B\bot D|\{A,C\})
\end{cases}\end{equation} $$ {: #markov-independence}

但是使用贝叶斯网是无法表示这样的独立关系而不带入额外的独立关系。因此我们需要使用无向图来表示这种独立假设

![markov-model](assets/img/probabilistic-graphical-models/markov-model.svg){: #markov-model}

我们把形如[上图](#markov-model)的无向图$\mathcal{H}$，叫做马尔可夫网，结点为随机变量，边表示两个结点间具有亲密关系。

与贝叶斯网类似，通过马尔可夫网可以将联合分布转换为紧凑的因子 $\phi$ 的分解，同时提供一系列马尔可夫独立性假设。  
但与贝叶斯网不同的是，他们因子之间的含义。在有向图中因子的物理含义为条件概率分布；但在无向图中，由于因子之间的连接是没有方向的，因此我们无法将因子表达为条件概率。在无向图中我们使用因子来表达随机变量之间的兼容度，或者说是随机变量之间的**密切程度**。此外，在因子的取值上两种模型也不相同，在有向图中因子的定义为条件概率分布因此它的取值要符合概率分布的所有要求，例如取值范围是 $[0,1]$ 、$P(\Omega)=1$ 和 $P(A + B) = P(A) + P(B)$ 等。但在无向图中因子不再具有这种要求，因子的取值范围甚至可以为负值，正因为缺少这种约束，无向图模型的联合分布并不能之间表示为，所有因子的乘积形式，应为它并没有被归一化，因此我们引入一个全局归一化常数 $Z$ 来将其变成一个合法的分布。

这样，我们可以为求[例子](#markov-model)中的 $P(A, B, C, D)$ 了

$$ \begin{equation}
    P(A, B, C, D)=\frac{1}{Z}\phi_1(A, B)\phi_2(B, C)\phi_3(C, D)\phi_4(D, A)\\
    Z=\sum_{A, B, C, D}\phi_1(A, B)\phi_2(B, C)\phi_3(C, D)\phi_4(D, A)
\end{equation} $$

更一般地我们有计算无向图模型联合分布的公式

$$ \begin{equation}
    P_\Phi(A_1,\dotsc,A_n)=\frac{1}{Z}\tilde{P}_\Phi(A_1,\dotsc,A_n)\\
    \tilde{P}_\Phi(A_1,\dotsc,A_n)=\prod_{i=1}^K\phi_i(\mathbf{D}_i)\\
    Z = \sum_{A_1,\dotsc,A_n}\tilde{P}_\Phi(A_1,\dotsc,A_n)
\end{equation} $$

其中 $\Phi$ 为因子集 $\Phi = \{\phi_1(\mathbf{D}_1),\dotsc,\phi_K(\mathbf{D}_K)}$ ，公式中每个 $\mathbf{D}$ 都是无向图 $\mathcal{H}$ 中的一个完全子图即**团**  
那么描述**密切程度**的 $\phi(\mathbf{D})$ 应该如何求呢？ 我们通常将因子改写为对数线性模型的形式，即

$$ \begin{equation}\phi(\mathbf{D})=\mathrm{exp}(\omega f(\mathbf{D}))\end{equation} $$

其中 $f(\mathbf{D})$ 为从子图 $\mathbf{D}$ 中提取的特征，很多时候我们会让特征的取值为 $0$ 或 $1$ ；即，取值为 $0$ 表示子图中没有对应的特征存在，反之取 $1$，$\omega $ 为特征对应的权重，这样一来联合分布就可以改写为对数线性模型的形式了

$$ \begin{equation}
    P(A_1,\dotsc,A_n)=\frac{1}{Z}\mathrm{exp}\left[\sum_{i=1}^k\omega_i f_i(\mathbf{D}_i)\right]
\end{equation} $$

我们知道，使用贝叶斯网网络无法表示[前面例子](#markov-independence)所描述的独立性；那么有没有什么独立性是贝叶斯网可以描述的而马尔可夫网是不能描述的呢？  
我们考虑贝叶斯网中一个经典的结构：v-结构。在这个图 $\mathcal{G}$ 中，存在着边缘独立 $(I\bot D)$ 并且不存在着条件独立 $(I\bot D|G)$ ，但我们试图构建满足这样独立性的马尔可夫网的时候，我们会发现并不能找到满足这样独立性的图 $\mathcal{H}$ 

![markov-v](assets/img/probabilistic-graphical-models/markov-v.png)

# 最后
本来打算在这篇文章中就吧HMM和CRF都讲了的，结果发现还是高估自己了。光是概率图的部分就写了3天，所以果断将概率图模型的部分截出来，单独作为一篇文章。

[^1]: Koller D, Friedman N, Bach F. Probabilistic graphical models: principles and techniques\[M\]. MIT press, 2009.
