---
layout:     post
title:      "Triton 学习手记 （二）：实现 Flash Attention 2"
date:       2025-08-29 19:00:0+0800
author:     "Houquan Zhou"
header-img: "/assets/img/post-bg.jpg"
mathjax: true
catalog: true
hidden: true
tags:
    - Triton
    - 手记
---

## 前言

在[之前的手记](/2025/08/19/learning-trition-0)中，我们介绍了 Triton 的基本概念，在这篇手记中我们开始实践，实现 Flash Attention 2。

## 什么是 Flash Attention 2？

在进入实现环节之前，我们先来回顾一下 Flash Attention 2。
如果读者熟悉 Flash Attention 2，可以跳过这一节。([直达下一节](#flash-attention-2-的-triton-实现))

## Flash Attention 2 的 Triton 实现
