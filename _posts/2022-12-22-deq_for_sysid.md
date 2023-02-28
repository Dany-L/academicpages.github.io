---
title: Equilibrium Models for System Identification
date: 2022-12-22
tags:
    - research
    - system identification
    - equilibrium models
---
Deep equilibrium networks and their relation to system theory, part of the seminar *Machine Learning in the Sciences by [Mathias Niepert](http://www.matlog.net)*. The code for the examples shown is available on [GitHub](https://github.com/Dany-L/RenForSysId)

# Motivation
Equilibrium network were introduced at [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html) with the main benefit being their memory efficiency. Compared to state-of-the-art networks deep equilibrium networks could reach the same level of accuracy without storing the output of each layer to do backpropagation. In this post the goal is to stress the connection between deep equilibrium networks and how they can be applied to system identification and control. This link is also seen in a [CDC 2022](https://ieeexplore.ieee.org/abstract/document/9992684/) and [CDC 2021](https://ieeexplore.ieee.org/abstract/document/9683054/) paper.

To appreciate that connection let us assume an unknown nonlinear dynamical system that can be described by a discrete differential equation 

$$
\begin{equation}
    \begin{aligned}
    x^{k+1} & = f_{\text{true}}(x^k, u^k) \\
    y^{k} & = g_{\text{true}}(x^k, u^k)
    \end{aligned}
\label{eq:nl_system}
\end{equation}
$$

with given initial condition $x^0$. The state is denoted by $x^k$, the input by $u^k$ and the output by $y^k$, the superscript indicates the time step of the sequence $k=1, \ldots, N$. The goal in system identification is to learn the functions $g_{\text{true}}: \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \mapsto \mathbb{R}^{n_y}$ and $f_{\text{true}}: \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \mapsto \mathbb{R}^{n_x}$ from a set of input-output measurements $\mathcal{D} = \left\lbrace (u, y)_i \right\rbrace_{i=1}^K$.

The system \eqref{eq:nl_system} maps an input sequence $u$ to an output sequence $y$, recurrent neural networks are a natural fit to model sequence-to-sequence maps. From a system theoretic perspective recurrent neural networks are a discrete, linear, time-invariant system interconnected with a static nonlinearity known as the activation function, a very general formulation therefore follows as

$$
\begin{equation}
    \begin{pmatrix}
        x^{k+1} \\
        \hat{y} \\
        z^k 
    \end{pmatrix} =
    \begin{pmatrix}
        A & \tilde{B}_1 & \tilde{B}_2 \\
        C_1 & D_{11} & D_{12} \\
        C_2 & D_{21} & D_{22} 
    \end{pmatrix}
    \begin{pmatrix}
        x^k \\
        u^k \\
        w^k
    \end{pmatrix}
    \label{eq:rnn_linear}
\end{equation}
$$

with $w^k = \Delta(z^k)$, the standard recurrent neural network (See [Equation 10](https://www.deeplearningbook.org/contents/rnn.html)) results as a special case of this more general description, this can be seen by choosing the hidden state $h^{k} = x^{k+1}$, $\Delta(z^k) = \tanh(z^k)$ and the following parameters:

$$
\begin{equation*}
    \begin{pmatrix}
        x^{k+1} \\
        \hat{y}^k \\
        z^k 
    \end{pmatrix}=
    \begin{pmatrix}
        0 & 0 & I \\
        0 & 0 & W_y \\
        W_h & U_h & 0
    \end{pmatrix}
    \begin{pmatrix}
        x^k \\
        u^k \\
        w^k
    \end{pmatrix}
\end{equation*}
$$

Note that we neglected the bias terms.

The problem of learning the system \eqref{eq:nl_system} can now be made more formal. Given a dataset $\mathcal{D}$, find a parameter set $\theta = \lbrace A, B_1, B_2, C_1, D_{11}, D_{12}, C_2, D_{21}, D_{22} \rbrace$ such that the error between the prediction and the output measurement is small $\min_{\theta} \sum_{k=1}^{N} \|\hat{y}^k - y^k \|$.

Before diving into deep equilibrium networks let us shortly recap the motivation. Recurrent neural networks are a good fit to model unknown dynamical systems. The parameters are tuned by looking at the difference between the prediction of the recurrent neural network and the output measurements. A more general description of a recurrent neural network is given by a general discrete LTI system interconnected with a static nonlinearity.

In the next section the basic concept of deep equilibrium networks will be explained, this naturally leads to monotone operator equilibrium networks. In section 3 the motivation is revisited and followed by a conclusion.

The focus of this post is to highlight th link between deep equilibrium networks and their application to problems in system and control. Details on how to calculate the gradient and monotone operator theory are only referenced.

# Deep equilibrium networks
Consider a input sequence $u$ that is fed through a neural network with $L$ layers, on each layer $f_{\theta}^{[0]}(x^0, u), \ldots, f_{\theta}^{[L-1]}(x^{L-1}, u)$, where $x$ represents the hidden state and $f_{\theta}^{[i]}$ the activation function on each layer.

![Deep forward model](/spaghetti/images/fwd_deep.png)

The first step towards deep equilibrium networks is to tie the weights $f_{\theta}^{0}(x^0, u) = $f_{\theta}^{i}(x^0, u)$ for all $i=0, \ldots, L-1$. It turns out that this restriction does not hurt the prediction accuracy of the network, since any deep neural network can be replaced by a single layer by increasing the size of the weight (See [Appendix C](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html) for details).

![Weight tied network](/spaghetti/images/fwd_tied.png)

In a next step the number of layer is increased $L \to \infty$. The forward pass can now also be formulated as finding a fixed point $x^*$, which can be solved by a number of root fining algorithm as illustrated next.

![Deep equilibrium model](/spaghetti/images/fwd_deq.png)

## Backward pass
To train the deep equilibrium network the gradient with respect to the parameters $\theta$ needs to be calculated from the forward pass. Traditionally this is achieved by stepping trough the forward pass of the deep neural network. For deep equilibrium models however this is not desired, since the gradient should be independent of the root finding algorithm.

The loss function follows as

$$
\ell=\mathcal{L}\left(h\left(\operatorname{RootFind}\left(g_0 ; u\right)\right), y\right),
$$

with the output layer $h:\mathbb{R}^{n_z} \mapsto \mathbb{R}^{n_y}$, which can be any differentiable function (e.g. linear), $y$ is the ground-truth sequence and $\mathcal{L}:\mathbb{R}^{n_y}\times\mathbb{R}^{n_y} \mapsto \mathbb{R}$ is the loss function.

The gradient with respect to $(\cdot)$ (e.g. $\theta$) can now be calculated by implicit differentiation

$$
\frac{\partial \ell}{\partial(\cdot)}=-\frac{\partial \ell}{\partial h} \frac{\partial h}{\partial x}^{\star}\left(\left.J_{g_\theta}^{-1}\right|_{x^*}\right) \frac{\partial f_\theta\left(x^{\star} ; u\right)}{\partial(\cdot)},
$$

were $\left.J_{g_\theta}^{-1}\right|_{x^*}$ is the inverse Jacobian of $g_{\theta}$ evaluated at $x^*$

For details the gradient and how it can be calculated see [Chapter 4](http://implicit-layers-tutorial.org/deep_equilibrium_models/) of the implicit layer tutorial.

## Example
Lets make a simple example to compare a fixed layer neural network with a deep equilibrium model. We assume sequence length $T=3$, size of hidden state $n_x = 10$, input and output size $n_y = n_u = 1$. The weight are randomly initialized and the initial hidden state is set to zero $x^0 = 0$, $W_x \in \mathbb{R}^{n_x \times n_x}$, $U_x\in \mathbb{R}^{n_x \times T}$ and we take a linear output layer with $W_y \in \mathbb{R}^{n_y \times n_x}$, the biases are accordingly.

The forward pass for $L$ layers sequence-to-sequence model in PyTorch:
```python
# forward pass for fixed number of layers
x = torch.zeros(size=(1, n_x))
u = torch.tensor(u).reshape(1, n_u)
for l in range(L):
    x = nl(W_x(z) + U_x(u) + b_x)
y_hat = W_y(x) + b_y
```
The forward pass for the deep equilibrium model:
```python
# DEQ
def g_theta(x):
    x = x.reshape(n_x,1)
    return np.squeeze(np.tanh(W_x_numpy @ z + U_x_numpy @ x + b_x_numpy) - z)

x_star, infodict, ier, mesg = fsolve(g_theta, x0=x_0, full_output=True)
x_star = z_star.reshape(n_z, 1)
y_hat_eq = W_y_numpy @ x_star + b_y_numpy
```
Note that the code are only small snippets that should give an idea on how to implement the models, the code is not supposed to run without further adjustment, for the root finding algorithm [scipy.optimize.fsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html) is used.

The results for different values of $L$ are compared
```python
Number of finite layers: 0       || x^L - x^* ||^2: 0.7032
Number of finite layers: 1       || x^L - x^* ||^2: 0.3898
Number of finite layers: 2       || x^L - x^* ||^2: 0.2898
Number of finite layers: 3       || x^L - x^* ||^2: 0.1621
Number of finite layers: 4       || x^L - x^* ||^2: 0.09451
Number of finite layers: 10      || x^L - x^* ||^2: 0.001685
Number of finite layers: 20      || x^L - x^* ||^2: 7.595e-06
Number of finite layers: 30      || x^L - x^* ||^2: 7.069e-08
```
The result shows that a feed forward neural network converges to the same result as the equilibrium network if the layer size increases.

# Monotone operator equilibrium networks
Looking at the results of the comparison a natural question to ask is whether the deep neural network always converges to a fixed point for sufficient large $L$? Lets play with the example a little bit
```python
W_z = torch.nn.Linear(in_features=n_z, out_features=n_z, bias=True)
torch.nn.init.normal_(W_z.weight)
U_z = torch.nn.Linear(in_features=n_x, out_features=n_z, bias=False)
W_y = torch.nn.Linear(in_features=n_z, out_features=n_x, bias=True)
```

# System identification with equilibrium networks


# Background

<!-- 
## Cart pole example
The discretized inverted pendulum can be described by the state space representation
$$
\begin{align}
P & \left\{
\begin{aligned} 
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
\end{aligned} \right. \label{eq:linear_inv_pend}\\
w^k & = \Delta(z^k) = z^k - \sin(z^k) \label{eq:nonlinear_inv_pend}
\end{align}
$$
where $\delta = 0.001$ is the sampling time, $g$ is the gravitational constant, $l$ the length of the rod and $m$ the mass. Common nonlinearities \eqref{eq:nonlinear_inv_pend} for neural networks are $\tanh(\cdot)$, $\operatorname{ReLU}(\cdot)$ or $\operatorname{LeakyReLU}(\cdot)$.
## Discrete linear time-invariant systems with nonlinear disturbance

$$
\begin{align}
    G & \left\{ \begin{aligned}
        x^{k+1} & = A x^k + B_1 u^k + B_2 w^k \\
        y^k & = C_1 x^k + D_{11} u^k + D_{12} w^k \\
        z^k & = C_2 x^k + D_{21} u^k + D_{22} w^k
    \end{aligned} \right.\\
    w^k & = \Delta(z^k)
\end{align}
$$ -->
