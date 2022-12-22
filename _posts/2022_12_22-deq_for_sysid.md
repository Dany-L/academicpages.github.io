---
title: Equilibrium Models for System Identification
date: 2022-12-22
tags:
    - research
    - system identification
    - equilibrium models
---

some sample text

# Introduction

# Background

## Inverted pendulum
The discretized inverted pendulum can be described by the state space representation
<!-- $$
\begin{align} 
    x^{k+1} & = 
    \begin{pmatrix}
        1 & \delta \\
        \frac{g \delta}{l} & 1 - \frac{\mu \delta}{m l^2}
    \end{pmatrix}
    x^k
    \begin{pmatrix}
        0 \\
        -\frac{g\delta}{l}
    \end{pmatrix}
    u^k
    \begin{pmatrix}
        0 \\
        \frac{\delta}{ml^2}
    \end{pmatrix}
    w^k \\
    y^k & = 
    \begin{pmatrix}
        1 & 0
    \end{pmatrix} x^k \\
    z^k & = 
    \begin{pmatrix}
        1 & 0
    \end{pmatrix}
    x^k
\end{align}
w^k = \Delta(z^k) = z^k - \sin(z^k)
$$ -->
<!-- where $\delta = 0.001$ is the sampling time, $g$ is the gravitational constant, $l$ the length of the rod and $m$ the mass. -->
## Discrete linear time-invariant systems with nonlinear disturbance

<!-- $$
    \begin{align}
        x^{k+1} & = A x^k + B_1 u^k + B_2 w^k \\
        y^k & = C_1 x^k + D_{11} u^k + D_{12} w^k \\
        z^k & = C_2 x^k + D_{21} u^k + D_{22} w^k
    \end{align}
    w^k = \Delta(z^k)
$$ -->
