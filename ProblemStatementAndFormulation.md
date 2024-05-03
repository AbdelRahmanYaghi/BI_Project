## **Problem**
We train a large feed forward neural network, larger than the size normally needed for the problem at hand, we formulate an MILP such that we prune as much of the network as possible while maintaining a large accuracy.

## **Notation**
![alt text](image.png)

Lets say we have layer $L_1$ and layer $L_2$, $L_1$ has $n$ neurons and $L_2$ has $m$ neurons.

then the weights connecting $L_1, L_2$ are represented by the matrix $\textbf{W}_{mn}$ where each element $i,j$ is the weight from neuron $j$ in the first layer to neuron $i$ in the second.


$$\begin{bmatrix}
w_{11} & w_{12} & w_{13} & \dots & w_{1n}\\
w_{21} & w_{22} & w_{23} & \dots & w_{2n}\\
\vdots & \vdots & \vdots & \ddots & w_{3n}\\
w_{m1} & w_{m2} & w_{m3} & \dots & w_{mn}\\
\end{bmatrix}$$

We say that $\textbf{W}^{L_2}$ is the Matrix of weights conntecting $L_1$ to $L_2$.

Now lets look at how our network functions:

$$
\begin{bmatrix}
w_{11} & w_{12} & w_{13} & w_{14} & b_1 \\
w_{21} & w_{22} & w_{23} & w_{24} & b_2 \\
w_{31} & w_{32} & w_{33} & w_{34} & b_3 
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4 \\ 1
\end{bmatrix} = 
\begin{bmatrix}
x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1\\
x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2\\
x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3
\end{bmatrix} 
$$

This can be written as a shorthand like this:

$$\textbf{W}^{L_2} \textbf{X} = \textbf{A}$$

## **Formulation**

Now for our MILP formulation, we need to have a binary variable for each node in each level, this variable indicates whether to this neuron in the network or leave it out.

so for each level $L_{i}$ with $m$ neurons, we have a vector of binary vars.

$$Z_{i} =
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_m
\end{bmatrix}$$

We are also going to say that removing a neuron corrsiponds to zeroing out all incoming and outgoing weights,