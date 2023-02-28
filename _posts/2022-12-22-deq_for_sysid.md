---
title: Equilibrium Models for System Identification
date: 2022-12-22
tags:
    - research
    - system identification
    - equilibrium models
---
Deep equilibrium networks and their relation to system theory, part of the seminar *Machine Learning in the Sciences by [Mathias Niepert](http://www.matlog.net)*. 

<!-- The code for the examples shown is available on [GitHub](https://github.com/Dany-L/RenForSysId) -->

# Motivation
Equilibrium network were introduced at [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html) with the main benefit being their memory efficiency. Compared to state-of-the-art networks deep equilibrium networks could reach the same level of accuracy without storing the output of each layer to do backpropagation. In this post the goal is to stress the connection between deep equilibrium networks and how they can be applied to system identification and control. This link is also seen in a [CDC 2022](https://ieeexplore.ieee.org/abstract/document/9992684/) and [CDC 2021](https://ieeexplore.ieee.org/abstract/document/9683054/) paper.
TODO: add references

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

with given initial condition $x^0$. The state is denoted by $x^k$, the input by $u^k$ and the output by $y^k$, the superscript indicates the time step of the sequence $k=1, \ldots, N$. The goal in system identification is to learn the functions $g_{\text{true}}: \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \mapsto \mathbb{R}^{n_y}$ and $f_{\text{true}}: \mathbb{R}^{n_x} \times \mathbb{R}^{n_u} \mapsto \mathbb{R}^{n_x}$ from a set of input-output measurements $\mathcal{D} = \lbrace (u, y)_i \rbrace_{i=1}^K$.

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

with $w^k = \Delta(z^k)$, the standard recurrent neural network results as a special case of this more general description, this can be seen by choosing the hidden state $h^{k} = x^{k+1}$, $\Delta(z^k) = \tanh(z^k)$ and the following parameters:

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
Consider a input sequence $u$ that is fed through a neural network with $L$ layers, on each layer $f_{\theta}^{0}(x^0, u), \ldots, f_{\theta}^{L-1}(x^{L-1}, u)$, where $x$ represents the hidden state and $f_{\theta}^i$ the activation function on each layer, the network is shown in Figure 

<script type="text/tikz">
  \begin{tikzpicture}[align=center]
    \draw (0,0) circle (1in);
  \end{tikzpicture}
</script>

test python code block

```python
# forward pass for fixed number of layers
z = torch.zeros(size=(1, n_z))
x = torch.tensor(u).reshape(1, n_x)
for l in range(L):
    z = nl(W_z(z) + U_z(x))
y_hat = W_y(z)
```

<script type="text/tikz">
\begin{tikzpicture}[
    node distance = 0.25cm and 0.5cm, 
    auto, 
    align=center,
    block/.style={
        draw,
        rectangle,
        rounded corners,
        minimum height=2em,
        minimum width=2em
    }
]    
    % blocks
    \node[] (input) {};
    \node[block, right= of input] (G) {$G$};
\end{tikzpicture}
</script>

<!-- <script type="text/tikz">
\begin{tikzpicture}[
    node distance = 0.25cm and 0.5cm, 
    auto, 
    align=center,
    block/.style={
        draw,
        rectangle,
        rounded corners,
        minimum height=2em,
        minimum width=2em
    }
]    
    % blocks
    \node[] (input) {};
    \node[block, right= of input] (G) {
        \begin{tikzpicture}[
            node distance = 0.25cm and 0.5cm, 
            auto, 
            align=center,
            block/.style={
                draw,
                rectangle,
                rounded corners,
                minimum height=2em,
                minimum width=2em
            }
        ]   
            \node[] (inL1) {};
            \node[block, right= of inL1] (L1) {$f_{\theta}^{[0]}(z_{1:T}^0; x_{1:T})$};
            \node[right= of L1] (outL1) {};
            \node[above= of L1] (inX) {};

            \node[right= of outL1] (dots) {$\cdots$};

            \node[right= of dots] (inLL) {};
            \node[block, right= of inLL] (LL) {$f_{\theta}^{[L-1]}(z_{1:T}^{L-1}; x_{1:T})$};
            \node[right= of LL] (outLL) {};
            \node[above= of LL] (inXL) {};
            
            
            % Input and outputs coordinates
            
            % lines
            \draw[->] (inX) node[right] {$x_{1:T}$} -- (L1.north);
            \draw[->] (inL1) node[above] {$z_{1:T}^0$} -- (L1);
            \draw[->] (L1)  --  (outL1) node[above] {$z^1_{1:T}$};
            \draw[->] (inXL) node[right] {$x_{1:T}$} -- (LL.north);
            \draw[->] (inLL) node[above] {$z_{1:T}^{L-1}$} -- (LL);
            \draw[->] (LL) -- (outLL) node[above] {$z_{1:T}^L$};  
        \end{tikzpicture}
    };
    \node at (G.north) [above] {$\mathcal{S}_{\operatorname{DEQ}}$};
    \node[right= of G] (output) {};
    
    % Input and outputs coordinates
    
    % lines
    \draw[->] (input)  node[above] {$x_{1:T}, z_{1:T}^0$} -- (G);
    \draw[->] (G) -- (output) node[above] {$z_{1:T}^L$} ;    
\end{tikzpicture}
</script> -->

TODO: add figure. 

Note that such a network matches the system \eqref{eq:nl_system}.

The first step towards deep equilibrium networks is to tie the weights $f_{\theta}^{0}(x^0, u) = $f_{\theta}^{i}(x^0, u)$ for all $i=0, \ldots, L-1$. It turns out that this restriction does not hurt the prediction accuracy of the network, since any deep neural network can be replaced by a single layer by increasing the size of the weight (See TODO for details).

The weight tied network is shown in Figure TODO.

In a next step the number of layer is increased $L \to \infty$. The forward pass can now also be formulated as finding a fixed point $z^*$, which can be solved by a number of root fining algorithm as illustrated in Figure TODO

# Monotone operator equilibrium networks

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
