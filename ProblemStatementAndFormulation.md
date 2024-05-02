## **Problem**
We train a large feed forward neural network, larger than the size normally needed for the problem at hand, we formulate an MILP such that we prune as much of the network as possible while maintaining a large accuracy.

## **Notation**
![alt text](image.png)

Lets say we have layer $L_1$ and layer $L_2$, $L_1$ has $n$ neurons and $L_2$ has $m$ neurons.

then the weights connecting $L_1, L_2$ are represented by the matrix $\textbf{W}_{mn}$ where each element $i,j$ is the weight from neuron $j$ in the first layer to neuron $i$ in the second.


$\begin{bmatrix}
w_{11} & w_{12} & w_{13} & \dots & w_{1n}\\
w_{21} & w_{22} & w_{23} & \dots & w_{2n}\\
\vdots & \vdots & \vdots & \ddots & w_{3n}\\
w_{m1} & w_{m2} & w_{m3} & \dots & w_{mn}\\
\end{bmatrix}$


