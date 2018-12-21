---
layout:     post
title:      "Tensorflow中的一个小坑"
date:       2018-12-21 10:19:00 +0800
author:     "Zo"
header-img: "img/post-bg.jpg"
catalog: true
tags:
    - Tensorflow
---


Tensorflow的文档真是特别迷，有些组件空有一个页面，而没有任何介绍。最近在跑别人的模型发现了一些小坑

# 迷一样的 DEFINE_bool

在运行别人写好的 shell 的时候发现，在 tf.flags.DEFINE_bool 默认为 True 的时候没法赋值为 False  
比如

```tf.flags.DEFINE_bool('is_train', True, "Train or predict")``` 

使用

```--is_train False```

是没有效果的，在网上很多教程中也就只有这一种方法，别人的代码中也是只有这样。  

今天去查了一下代码发现了其中的玄机  

>Import router for absl.flags. See https://github.com/abseil/abseil-py.

于是我进入链接，在其中找到了解决之道  

>**DEFINE_bool or DEFINE_boolean**: typically does not take an argument: pass *--myflag* to set FLAGS.myflag to True, or *--nomyflag* to set FLAGS.myflag to False. *--myflag=true* and *--myflag=false* are also supported, but **not recommended**.

现在 bool flag 显式设置为 false 需要通过

```--nois_train``` 或者 ```--is_train=False``` 来进行了。