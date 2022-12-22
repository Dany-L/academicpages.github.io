---
title: Robust System Identification
date: 2022-11-30
tags:
    - research
    - system identification
    - robust recurrent neural network
---

*Abstract*
Recurrent neural networks are capable of learning the dynamics of an unknown nonlinear system purely from input-output measurements. However, the resulting models do not provide any stability guarantees on the input-output mapping. In this work, we represent a recurrent neural network as a linear time-invariant system with nonlinear disturbances. By introducing constraints on the parameters, we can guarantee finite gain stability and incremental finite gain stability. We apply this identification method to learn the motion of a four-degrees-of-freedom ship that is moving in open water and compare it against other purely learning-based approaches with unconstrained parameters. Our analysis shows that the constrained recurrent neural network has a lower prediction accuracy on the test set, but it achieves comparable results on an out-of-distribution set and respects stability conditions.

[Download paper here](https://dany-l.github.io/spaghetti/files/paper1.pdf)
[arXiv](https://arxiv.org/abs/2212.05781)
