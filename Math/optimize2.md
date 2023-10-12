# Optimization

[TOC]

---

## 线性规划概述

---

### 15.1

introduction

---

超平面

$$
H = \{\boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{u}^{T} \boldsymbol{x} = v\}
$$

超平面将 $\mathbb{R}^{n}$ 空间分成两半

$$
\begin{align*}
    & H_{+} = \{\boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{u}^{T} \boldsymbol{x} > v\} \\
    & H_{-} = \{\boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{u}^{T} \boldsymbol{x} < v\} \\
\end{align*}
$$

---

求解矩阵方程就是求解超平面的交集。

$$
\begin{align*}
    & \boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}\\
    & \left\{\begin{aligned}
        \boldsymbol{a}_{1}^{T} \boldsymbol{x} &= b_{1} \\
        \boldsymbol{a}_{2}^{T} \boldsymbol{x} &= b_{2} \\
        \boldsymbol{a}_{m}^{T} \boldsymbol{x} &= b_{m}
    \end{aligned}\right.
\end{align*}
$$

---

多面体和多胞体：多个半空间的交集。

---

### 15.2

线性规划的简单例子

---

标准模型

$$
\begin{align*}
    \text{minimize } & \boldsymbol{c}^{T} \boldsymbol{x} \\
    \text{subject to }& \boldsymbol{A} \boldsymbol{x} = \boldsymbol{b} \\
    & \boldsymbol{A} \in \mathbb{R}^{m \times n}, \boldsymbol{x},\boldsymbol{c} \in \mathbb{R}^{n}, \boldsymbol{b} \in \mathbb{R}^{m}
\end{align*}
$$

不定矩阵 $\boldsymbol{A}$ 满足：$m > n$.

---

### 15.3

二维线性规划

---

### 15.4

凸多面体和线性规划

---

令 $\Theta$ 是一个凸集，$\boldsymbol{y}$ 是凸集 $\Theta$ 的边界点。某个经过 $\boldsymbol{y}$ 的超平面将 $\mathbb{R}^{n}$ 空间分成两个半空间，如果 $\Theta$ 完全位于一个半空间中，定义该超平面为**支撑超平面**。

支撑超平面的方程

$$
\begin{align*}
    & \boldsymbol{c}^{T} \boldsymbol{x} = \beta \\
    & \forall \boldsymbol{x} \in M, \boldsymbol{c}^{T} \boldsymbol{x} \geq \beta
\end{align*}
$$

---

### 15.5

线性规划的标准型

---

$$
\begin{align*}
    & \text{minimize} && \boldsymbol{c}^{T} \boldsymbol{x} \\
    & \text{subject to} && \boldsymbol{A} \boldsymbol{x} = \boldsymbol{b} \\
    & && \boldsymbol{x} \geq 0 \\
    & && \mathbb{A} \in \mathbb{R}^{m \times n}, \mathrm{rank} A = m
\end{align*}
$$

对矩阵秩的规定是为了保证方程有解。

---

对于非等式约束，如何转化为标准型？

引入**松弛变量**。

$$
\begin{align*}
    & \boldsymbol{A} \boldsymbol{x} \geq \boldsymbol{b} \\
    \Rightarrow & \boldsymbol{A}\boldsymbol{x} - \boldsymbol{y} = \boldsymbol{b} \\
    & \boldsymbol{x}, \boldsymbol{y} \geq \boldsymbol{0}
\end{align*}
$$

---

这样就转化成

$$
\begin{align*}
    &\text{minimize} && \boldsymbol{c}^{T} \boldsymbol{x} \\
    &\text{subject to} && \begin{bmatrix} A & -I_{m} \end{bmatrix}
        \begin{bmatrix} \boldsymbol{x} \\ \boldsymbol{y} \end{bmatrix} = 0 \\
    & && \begin{bmatrix} \boldsymbol{x} \\ \boldsymbol{y} \end{bmatrix} \geq \boldsymbol{0}
\end{align*}
$$

---

example:

$$
\begin{align*}
    & \text{minimize} && x_{2} - x_{1} \\
    & \text{subject to} && 3x_{1} = x_{2} - 5 \\
    & && |x_{2}| \leq 2 \\
    & && x_{1} \leq 0
\end{align*}
$$

---

$$
\begin{align*}
    & x_{1}' = -x_{1} \geq 0 \\
    & x_{2} = u - v && u, v \geq 0 \\
    & x_{3} = 2 - x_{2} \geq 0 \\
    & x_{4} = x_{2} + 2 \geq 0
\end{align*}
$$

其中第二个式子是对于不非负的决策变量进行转化的技巧。

---

重写得到：

$$
\begin{align*}
    & \text{minimize} && x_{1}' + u - v \\
    & \text{subject to} && 3x_{1}' + u - v = 5 \\
    & && u - v + x_{3} = 2 \\
    & && -u + v + x_{4} = 2 \\
    & && x_{1}', u, v, x_{3}, x_{4} \geq 0
\end{align*}
$$

---

### 15.6

基本解

---

对于矩阵方程

$$
\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}, \boldsymbol{A} \in \mathbb{R}^{m \times n}, \mathrm{rank}A = m
$$

对矩阵 $\boldsymbol{A}, \boldsymbol{x}$ 进行重排，将 $\boldsymbol{A}$ 中线性无关的前 $m$ 个向量放在前 $m$ 列，对 $\boldsymbol{x}$ 做对应的变换。构成新的线性方程组

$$
\begin{align*}
    & \boldsymbol{A}\boldsymbol{E} = \begin{bmatrix} \boldsymbol{B} & \boldsymbol{D} \end{bmatrix} \\
    & |\boldsymbol{B}| \neq 0 \\
    & \begin{bmatrix} \boldsymbol{B} & \boldsymbol{D} \end{bmatrix} \boldsymbol{x}' = \boldsymbol{b} \\
    & \boldsymbol{B} \boldsymbol{x}_{B} = \boldsymbol{b} \Rightarrow \boldsymbol{x}_{B} = \boldsymbol{B}^{-1} \boldsymbol{b} \\
    & \boldsymbol{x}' = \begin{bmatrix} \boldsymbol{x}_{B} \\ \boldsymbol{0} \end{bmatrix} \\
    & \boldsymbol{x} = E^{-1} \begin{bmatrix} \boldsymbol{x}_{B} \\ \boldsymbol{0} \end{bmatrix}
\end{align*}
$$

对 $\boldsymbol{x}'$ 进行恢复，就能得到**基本解** $\boldsymbol{x}$.

如何找到所有的基本解？从 $n$ 个列向量中选取 $m$ 个组成新的系数矩阵，共有 $\binom{n}{m}$ 种选取方式。如果矩阵可逆，求解再恢复就能得到基本解。

---

一些定义

- 基变量：向量 $\boldsymbol{x}_{B}$ 中的元素
- 基本列向量：矩阵 $\boldsymbol{B}$ 中的列向量
- 退化的基本解：如果基本解中的某些基变量为零，基本解退化
- 可行解：满足约束 $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}, \boldsymbol{x} \geq 0$，意思就是在可行集中
- 基本可行解：既基本又可行
- 退化的基本可行解
- 最优可行解：使得目标函数取得最小值时的可行解
- 最优基本可行解：“基本”的最优可行解

---

如何从一组基本可行解得到另一组？

- 入基向量
- 出基向量

详见下一节

---

### 15.7

基本解的性质

---

线性规划的基本定理

- 存在可行解，则一定存在基本可行解
- 存在最优可行解，则一定存在最优基本可行解

---

这个定理的直觉很好理解，因为可行集必定存在边界，这个边界就是基本可行解；而对于最优可行解，一定在边界上：要么是顶点，顶点一定是最优基本可行解；要么是边上，而此时整个边一定都是最优可行解，边的端点（顶点）一定是最优基本可行解。

更加严谨的证明就是对矩阵进行了重排，假设（最优）可行解的前 $p$ 个坐标是正值，其余的都是 0. 对重排以后矩阵的前 $p$ 个列向量进行两种情况的讨论：

1. 这 $p$ 个列向量线性无关，那么这个（最优）可行解必然是基本的。
2. 这 $p$ 个列向量线性相关，则可以将这个（最优）可行解进行 $\boldsymbol{y}$ 方向的移动。需要满足条件：$\boldsymbol{y}$ 只有前 $p$ 个值为非零值，而且可以保证至少有一个正值，需要满足：

$$
\begin{align*}
    & \boldsymbol{A} \cdot \boldsymbol{y} = \boldsymbol{0} \iff \sum_{i=1}^{p} y_{i} \boldsymbol{a}_{i} = \boldsymbol{0} \\
    & \boldsymbol{x} \rightarrow \boldsymbol{x} - \varepsilon \boldsymbol{y} \in D
\end{align*}
$$

对于命题一，可以不断地移动直到成为基本可行解。

对于命题二，可以根据 $\boldsymbol{x}$ 已经是最优可行解证明沿着 $\boldsymbol{y}$ 方向的移动也是最优的，可以不断地移动直到成为最优基本可行解。

Q.E.D

---

### 15.8

几何视角下的线性规划

---

凸集 $\Theta$

$$
\begin{align*}
    \forall \boldsymbol{x}, \boldsymbol{y} \in \Theta, \forall \alpha \in [0, 1], \alpha \boldsymbol{x} + (1 - \alpha) \boldsymbol{y} \in \Theta
\end{align*}
$$

凸集 $\Theta$ 的极点 $\boldsymbol{x}$

$$
\begin{align*}
    \forall \boldsymbol{x}_{1}, \boldsymbol{x}_{2} \in \Theta, \forall \alpha \in [0, 1], \exists \boldsymbol{x} \in \Theta, \boldsymbol{x} \neq \alpha \boldsymbol{x}_{1} + (1 - \alpha) \boldsymbol{x}_{2} 
\end{align*}
$$

---

线性规划的极点定理

**可行集**（凸集）的极点**等于**约束的**基本可行解**。

几何直观：只要查看多边形的顶点是不是最优解就好了。

---

## 单纯形法

---

### 16.1

利用初等行变换求解线性方程组

---

三类**初等变换**对应三类左乘**初等矩阵**。

矩阵的求逆算法

$$
\begin{bmatrix} A  & I \end{bmatrix} \overset{r}{\sim} \begin{bmatrix} I & A^{-1} \end{bmatrix}
$$

---

线性规划的第一约束

$$
\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}, \boldsymbol{A} \in \mathbb{R}^{m \times n}, \mathrm{rank} \boldsymbol{A} = m
$$

不失一般性，假设 $\boldsymbol{A}$ 的前 $m$ 列线性无关

$$
\begin{align*}
    &\begin{bmatrix} \boldsymbol{A} & \boldsymbol{b} \end{bmatrix} \sim 
    \begin{bmatrix} \boldsymbol{I} & \boldsymbol{D} & \boldsymbol{b}' \end{bmatrix} \\
    & \boldsymbol{x} = \begin{bmatrix} \boldsymbol{x}_{B} \\ \boldsymbol{x}_{D}\end{bmatrix} \\
    & \boldsymbol{x}_{B} + \boldsymbol{D}\boldsymbol{x}_{D} = \boldsymbol{b}' \\
    & \boldsymbol{x} = \begin{bmatrix} \boldsymbol{b}' \\ \boldsymbol{0} \end{bmatrix} + \begin{bmatrix} -\boldsymbol{D} \boldsymbol{x}_{D} \\ \boldsymbol{x}_{D} \end{bmatrix}
\end{align*}
$$

约束转化为基本解加上一个通解的形式。

---

### 16.2

增广矩阵的规范形

---

**典式**：对方程 $\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}$，进行初等行变换以后，存在 $m$ 个变量，每个变量都各自对应一个方程，且仅仅在该方程中出现，系数为 1.

选择不同的列向量作为 $\mathbb{R}^{m}$ 空间的基，得到的典式也不同。考虑在基 $\boldsymbol{a}_{1}, \cdots, \boldsymbol{a}_{m}$ 上的==增广矩阵规范形==

$$
\begin{bmatrix} \boldsymbol{I}_{m} & \boldsymbol{Y}_{m,n-m} & \boldsymbol{y}_{0} \end{bmatrix} \sim \begin{bmatrix} \boldsymbol{A}  & \boldsymbol{b} \end{bmatrix}
$$

- 基变量：和作为基的列向量对应的变量
- 非基变量

---

上面的问题的可行解为

$$
\boldsymbol{x} = \begin{bmatrix} \boldsymbol{y}_{0} \\ \boldsymbol{0} \end{bmatrix}
$$

存在关系

$$
\boldsymbol{b} = \sum_{i=1}^{m} y_{i0} \boldsymbol{a}_{i}
$$

增广矩阵规范形最后一列的各个元素是 $\boldsymbol{b}$ 在基 $\{\boldsymbol{a}_{1}, \cdots, \boldsymbol{a}_{m}\}$ 中的坐标。

增广矩阵规范形第 j 列的各个元素是 $\boldsymbol{a}_{j}$ 在基 $\{\boldsymbol{a}_{1}, \cdots, \boldsymbol{a}_{m}\}$ 中的坐标。

---

### 16.3

更新增广矩阵

---

用 $\boldsymbol{a}_{q}(q>m)$ 替代 $\boldsymbol{a}_{p}(p \leq m)$ ，形成一组新的基。下面计算各个列向量在新的基下的坐标：

已有：

$$
\boldsymbol{a}_{q} = \sum_{i=1}^{m} Y_{iq}\boldsymbol{a}_{i}
= \sum_{i=1, i \neq p}^{m} Y_{iq} \boldsymbol{a}_{i} + Y_{pq} \boldsymbol{a}_{p}
$$

如果 $Y_{pq} = 0$，那么 $\boldsymbol{a}_{q}$ 和其他的基向量线性相关，不能构成基，该情况不应该存在。

$$
\boldsymbol{a}_{p} = \frac{1}{Y_{pq}} \boldsymbol{a}_{q} - \sum_{i=1,i\neq p}^{m} \frac{Y_{iq}}{Y_{pq}} \boldsymbol{a}_{i}
$$

---

对于其他的向量 $\boldsymbol{a}_{j} (m < j \leq n)$

$$
\begin{align*}
    \boldsymbol{a}_{j} &= \sum_{i=1, i \neq p}^{m} Y_{ij}\boldsymbol{a}_{i} + Y_{pj}\boldsymbol{a}_{p} \\
    &= \sum_{\substack{i=1 \\ i \neq p}}^{m} \left(Y_{i j}-\frac{Y_{p j}}{Y_{p q}} Y_{i q}\right) \boldsymbol{a}_{i}+\frac{Y_{p j}}{Y_{p q}} \boldsymbol{a}_{q}
\end{align*}
$$

在更新的矩阵中，第 $j$ 行就变成了

$$
\begin{align*}
    & Y_{ij}' = Y_{ij} - \frac{Y_{pj}}{Y_{pq}} Y_{iq} && (i \neq p) \\
    & Y_{pj}' = \frac{Y_{pj}}{Y_{pq}}
\end{align*}
$$

这就是枢轴变换，本质上是将第 $p$ 行作为被倍加的行进行第三类初等行变换，使得第 $q$ 列成为只有 $Y_{pq} = 1$，其余都是 0 的向量。

增广矩阵规范型的最后一列 $\boldsymbol{y}_{0}$ 就是基变量。

---

### 16.4

单纯形法

---

思路：从一个**基本可行解**转化到另一个**基本可行解**。

假设存在一组基本可行解

$$
\boldsymbol{x} = \begin{bmatrix} \boldsymbol{y}_{0} \\ \boldsymbol{0} \end{bmatrix}, \boldsymbol{y}_{0} > 0
$$

???

---

步骤

1. 寻找进基向量，使得目标函数值变小
2. 更新增广矩阵

---

## 仅含等式约束的优化问题

---

### 20.1

introduction

---

$$
\begin{align*}
    & \text{minimize} && f(\boldsymbol{x}) \\
    & \text{subject to} && \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0} \\
    & && \boldsymbol{g}(\boldsymbol{x}) \leq \boldsymbol{0} \\
    & && \boldsymbol{h} : \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}, \boldsymbol{g} : \mathbb{R}^{n} \rightarrow \mathbb{R}^{p}
\end{align*}
$$

线性规划问题可以看作这种问题的特例。

可行点和可行集

---

### 20.2

问题描述

---

$$
\begin{align*}
    & \text{minimize} && f(\boldsymbol{x}) \\
    & \text{subject to} && \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0} \\
    & && f: \mathbb{R}^{n} \rightarrow \mathbb{R} \\
    & && \boldsymbol{h}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m} \\
    & && m \leq n \\
    & && \boldsymbol{h} = \begin{bmatrix} h_{1} & \cdots & h_{m} \end{bmatrix}^{T}
\end{align*}
$$

导数矩阵

$$
D \boldsymbol{h}(\boldsymbol{x}) = \begin{bmatrix} Dh_{1}(\boldsymbol{x}) \\ \vdots \\ D h_{m} (\boldsymbol{x}) \end{bmatrix} = \begin{bmatrix} \nabla h_{1}(\boldsymbol{x})^{T} \\ \vdots \\ \nabla h_{m}(\boldsymbol{x})^{T} \end{bmatrix}
$$

正则点：梯度向量全部线性无关，导数矩阵行满秩。

---

如果可行集曲面

$$
S = \{ \boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0} \}
$$

上的所有点都是正则点，那么曲面 $S$ 的维数是 $n - m$.

（$n$ 个未知数，$m$ 个约束）

---

### 20.3

切线空间和法线空间

---

切线空间：对于曲面 $S$ 上的一点 $\boldsymbol{x}^{*}$ 处的切线空间为

$$
T(\boldsymbol{x}^{*}) = \{ \boldsymbol{y}: D \boldsymbol{h}(\boldsymbol{x}^{*}) \boldsymbol{y} = 0 \} = \mathrm{Nul} [D \boldsymbol{h}(\boldsymbol{x}^{*})]
$$

如果 $\boldsymbol{x}^{*}$ 是正则点，则该处导数矩阵是行满秩的，零空间的维数等于 $n$ - 导数矩阵维数 = $n - m$.

切平面由切线空间平移得到

$$
TP(\boldsymbol{x}^{*}) = T(\boldsymbol{x}^{*}) + \boldsymbol{x}^{*}
$$

这样就能使切线空间过点 $\boldsymbol{x}^{*}$ 了。

---

如何理解切线空间是导数矩阵的零空间？

导数矩阵是由梯度向量转置形成的，切线空间中的任何向量都和梯度向量垂直。而梯度向量是和可行集空间垂直的，因为可行集空间是==“水平集”==，而梯度向量是==“这些水平集之间增大最快的方向”==。

根据这一直觉，可以得到曲线某个点处的切线空间应该是该点处所有切向量的集合。

定理：假设 $\boldsymbol{x}^{*} \in S$ 是正则点，切线空间 $T(\boldsymbol{x}^{*})$。当且仅当存在一条经过 $\boldsymbol{x}^{*}$ 的可微曲线（$S$ 中），在 $\boldsymbol{x}^{*}$ 处的导数为 $\boldsymbol{y}$ 时，$\boldsymbol{y} \in T(\boldsymbol{x}^{*})$.

---

充分性的证明：

$$
\begin{align*}
    & \boldsymbol{x}^{*} = \boldsymbol{x}(t^{*}), t \in [a, b] \\
    & \boldsymbol{h}(\boldsymbol{x}^{*}) = \boldsymbol{0} \Rightarrow \boldsymbol{h}(\boldsymbol{x}(t)) = \boldsymbol{0}
\end{align*}
$$

根据水平集的推导方法

$$
\begin{align*}
    & D \boldsymbol{h}(\boldsymbol{x}) \dot{\boldsymbol{x}}(t) = \boldsymbol{0} \\
    & D \boldsymbol{h}(\boldsymbol{x}) \boldsymbol{y} = \boldsymbol{0}
\end{align*}
$$

Q.E.D

---

法线空间是导数矩阵的行空间。由梯度向量张成。

$$
N(\boldsymbol{x}^{*}) = \mathrm{Col} (D \boldsymbol{h}(\boldsymbol{x}^{*})^{T})
$$

行空间和零空间正交，因此法线空间和切线空间也正交。正则点处法线空间的维数为 $m$.

法平面

$$
NP(\boldsymbol{x}^{*}) = \boldsymbol{x}^{*} + N(\boldsymbol{x}^{*})
$$

---

$$
\forall \boldsymbol{v} \in \mathbb{R}^{n}, \exists \boldsymbol{x} \in T(\boldsymbol{x}^{*}), \boldsymbol{y} \in N(\boldsymbol{x}^{*}), \boldsymbol{v} = \boldsymbol{x} + \boldsymbol{y}
$$

---

### 20.4

拉格朗日条件

---

一阶必要条件

前提：

1. $\boldsymbol{x}^{*}$ 是正则点
2. $h: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}, m \leq n$
3. $\boldsymbol{\lambda}^{*} \in \mathbb{R}^{m}$

当满足下面的条件时，$\boldsymbol{x}^{*}$ 是局部==驻点==。

$$
\begin{align*}
    & D f(\boldsymbol{x}^{*}) + \boldsymbol{\lambda}^{*T} \cdot D \boldsymbol{h}(\boldsymbol{x}^{*}) = \boldsymbol{0}^{T} \\
    \iff & \nabla f(\boldsymbol{x}^{*}) + D \boldsymbol{h}(\boldsymbol{x}^{*})^{T} \cdot \boldsymbol{\lambda}^{*} = \boldsymbol{0} \\
    \iff & \nabla f(\boldsymbol{x}^{*}) \in N(\boldsymbol{x}^{*})
\end{align*}
$$

换句话说就是 $\nabla f(\boldsymbol{x}^{*})$ ==在约束的法线空间（导数矩阵的行空间）中==。

---

根据拉格朗日定理，可以构造拉格朗日函数

$$
\begin{align*}
    & l(\boldsymbol{x}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) + \lambda^{T} \boldsymbol{h}(\boldsymbol{x}) \\
    & f(\boldsymbol{x}^{*}) \in N(\boldsymbol{x}^{*}) \Rightarrow \exists \lambda \in \mathbb{R}^{m}, Dl(\boldsymbol{x}^{*}, \boldsymbol{\lambda}) = \boldsymbol{0}^{T}
\end{align*}
$$

求导运算 $D$ 指的是对联合向量 $\begin{bmatrix} \boldsymbol{x}^{*T} & \boldsymbol{\lambda}^{T} \end{bmatrix}^{T}$ 的求导

==这样，就将约束优化问题的一阶必要条件转化成了无约束优化问题的一阶必要条件==。

---

为什么要对联合向量求导呢？

因为两个向量 $\boldsymbol{x}^{*}, \boldsymbol{\lambda}$ 都是未知数，将它们联合起来作为单一未知数求解更容易。

---

为什么能对联合向量求导，这一条件和原来的条件等价吗？

记

$$
\begin{align*}
    & D_{x} l, D_{\lambda} l \\
    & D l = \begin{bmatrix} D_{x} l & D_{\lambda} l \end{bmatrix} \\
    & D_{x} l = D f(\boldsymbol{x}) + \lambda^{T} D \boldsymbol{h}(\boldsymbol{x}) \\
    & D_{\lambda} l = \boldsymbol{h}(\boldsymbol{x})^{T} = \boldsymbol{0}^{T} \\
    & Dl = \begin{bmatrix} Df(\boldsymbol{x}) + \lambda^{T} D \boldsymbol{h}(\boldsymbol{x}) & \boldsymbol{0}^{T} \end{bmatrix} \\
    & Dl = \boldsymbol{0}^{T} \iff D f(\boldsymbol{x}) + \lambda^{T} D \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}^{T}
\end{align*}
$$

---

example

对于方阵 $\boldsymbol{P}, \boldsymbol{Q} \in \mathbb{R}^{n \times n}, \boldsymbol{Q} \geq 0, \boldsymbol{P} > 0$. 求解优化问题。

$$
\begin{align*}
    & \text{maximize} && \frac{\boldsymbol{x}^{T}\boldsymbol{Q}\boldsymbol{x}}{\boldsymbol{x}^{T}\boldsymbol{P}\boldsymbol{x}}
\end{align*}
$$

可以转化成问题

$$
\begin{align*}
    & \text{maximize} && \boldsymbol{x}^{T} \boldsymbol{Q} \boldsymbol{x} \\
    & \text{subject to} && 1 - \boldsymbol{x}^{T} \boldsymbol{P} \boldsymbol{x} = 0
\end{align*}
$$

---

$$
\begin{align*}
    & l(\boldsymbol{x}, \lambda) = \boldsymbol{x}^{T}\boldsymbol{Q}\boldsymbol{x} + \lambda(1 - \boldsymbol{x}^{T}\boldsymbol{P}\boldsymbol{x}) \\
    & Dl(\boldsymbol{x}, \lambda) = \begin{bmatrix} 2 \boldsymbol{x}^{T} (\boldsymbol{Q} - \lambda \boldsymbol{P}) & 1 - \boldsymbol{x}^{T} \boldsymbol{P} \boldsymbol{x} \end{bmatrix} \\
    & Dl = \boldsymbol{0}^{T} \Rightarrow (\lambda \boldsymbol{P} - \boldsymbol{Q}) \boldsymbol{x} = \boldsymbol{0} \\
    & \boldsymbol{P}^{-1} (\lambda \boldsymbol{P} - \boldsymbol{Q}) \boldsymbol{x} = \boldsymbol{0} \\
    & \boldsymbol{P}^{-1} \boldsymbol{Q} \boldsymbol{x} = \lambda \boldsymbol{x}
\end{align*}
$$

$\lambda$ 是 $\boldsymbol{P}^{-1} \boldsymbol{Q}$ 的特征值。

---

$$
\begin{align*}
    & (\lambda \boldsymbol{P} - \boldsymbol{Q}) \boldsymbol{x} = \boldsymbol{0} \\
    & \boldsymbol{x}^{T} (\lambda \boldsymbol{P} - \boldsymbol{Q}) \boldsymbol{x} = \boldsymbol{0} \\
    & \boldsymbol{x}^{T} \boldsymbol{Q} \boldsymbol{x}^{T} = \lambda
\end{align*}
$$

因此取 $\boldsymbol{P}^{-1} \boldsymbol{Q}$ 的最大特征值和最小特征值，就可以得到 $\boldsymbol{x}^{T}\boldsymbol{Q}\boldsymbol{x}$ 的最大值和最小值。

$$
\begin{align*}
    & \lambda_{min} \Vert \boldsymbol{x} \Vert \leq \boldsymbol{x}^{T} \boldsymbol{Q} \boldsymbol{x} \leq \lambda_{max} \Vert \boldsymbol{x} \Vert
\end{align*}
$$

取 $\boldsymbol{P} = I_{n}$，就可以证明瑞利不等式。

---

### 20.5

二阶条件

---

拉格朗日函数

$$
\begin{align*}
    & l(\boldsymbol{x}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) + \boldsymbol{\lambda}^{T} \boldsymbol{h}(\boldsymbol{x}) \\
    & = f(\boldsymbol{x}) + \sum_{i=1}^{m} \lambda_{i} h_{i} (\boldsymbol{x})
\end{align*}
$$

关于 $\boldsymbol{x}$ 的黑塞矩阵

$$
\begin{align*}
    & \boldsymbol{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = D_{x}(D_{x} l) = \boldsymbol{F}(\boldsymbol{x}) + \sum_{i=1}^{m} \lambda_{i} \boldsymbol{H}_{i} (\boldsymbol{x})
\end{align*}
$$

---

二阶必要条件

前提：

- 满足一阶必要条件
- $f, \boldsymbol{h} \in C^{2}$
- $\boldsymbol{x}^{*}$ 是正则点

结论：

$$
\exists \lambda^{*} \in \boldsymbol{R}^{m}, \forall \boldsymbol{y} \in T(\boldsymbol{x}^{*}), \boldsymbol{y}^{T} \boldsymbol{L}(\boldsymbol{x}^{*}, \boldsymbol{\lambda}^{*}) \boldsymbol{y} \geq 0
$$

直觉：所有切向的方向都不会引起增加。

---

Proof：因为 $\boldsymbol{x}^{*}$ 是局部极小点，对于函数（表示曲面上的一根曲线）$\boldsymbol{x} = \boldsymbol{x}(t)$ 对应的自变量 $\boldsymbol{t}^{*}$ 也是局部极小点。根据无约束优化问题的二阶必要条件：

$$
\begin{align*}
    & \left. \frac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} f(\boldsymbol{x}(t)) \right|_{t=t^{*}} \geq 0 \\
    & \frac{\mathrm{d}}{\mathrm{d}t} f(\boldsymbol{x}(t)) = Df(\boldsymbol{x}(t)) \dot{\boldsymbol{x}}(t) \\
    & \frac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} f(\boldsymbol{x}(t)) = \frac{\mathrm{d}}{\mathrm{d}t} \left[ Df(\boldsymbol{x}(t)) \dot{\boldsymbol{x}}(t) \right] \\
    & = \dot{\boldsymbol{x}}(t)^{T} \frac{\mathrm{d}}{\mathrm{d}t} [\nabla f(\boldsymbol{x}(t))] + Df(\boldsymbol{x}(t)) \ddot{\boldsymbol{x}}(t) \\
    & = \dot{\boldsymbol{x}}(t)^{T} \boldsymbol{F}(\boldsymbol{x}(t)) \dot{\boldsymbol{x}}(t)+ Df(\boldsymbol{x}(t)) \ddot{\boldsymbol{x}}(t) \\
\end{align*}
$$

---

根据切向空间的性质，可以选择一个函数（曲线）$\boldsymbol{x} = \boldsymbol{x} (t)$ 使得 $\boldsymbol{y} = \dot{\boldsymbol{x}}(t^{*})$

$$
\begin{align*}
    & \boldsymbol{y}^{T} \boldsymbol{F}(\boldsymbol{x}^{*}) \boldsymbol{y} + Df(\boldsymbol{x}^{*})\ddot{\boldsymbol{x}}(t^{*}) \geq 0
\end{align*}
$$

---

水平集

$$
\begin{align*}
    &0 = \left.\frac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} \boldsymbol{\lambda}^{*} \boldsymbol{h}(\boldsymbol{x}(t))\right|_{x=x^{*}} \\
    & = \cdots \\
    & = \boldsymbol{y}^{T} \left(\sum_{i=1}^{m} \lambda^{*}_{i} \boldsymbol{H}_{i}\right)\boldsymbol{y} + \boldsymbol{\lambda}^{*T} D \boldsymbol{h}(\boldsymbol{x}(t^{*})) \ddot{\boldsymbol{x}}(t^{*})
\end{align*}
$$

与上一片中的不等式相加，利用一阶必要条件，得到

$$
\begin{align*}
    & \boldsymbol{y}^{T} \left(\boldsymbol{F}(\boldsymbol{x}^{*}) + \sum_{i=1}^{m} \lambda_{i}^{*} \boldsymbol{H}_{i} \right) \boldsymbol{y} \geq 0 \\
    & \boldsymbol{y}^{T} \boldsymbol{L}(\boldsymbol{x}^{*}, \boldsymbol{\lambda}^{*}) \boldsymbol{y} \geq 0
\end{align*}
$$

Q.E.D

---

二阶充分条件

Condition:

- $f, \boldsymbol{h} \in C^{2}$
- 一阶必要条件

$$
\exists \boldsymbol{\lambda}^{*} \in \boldsymbol{R}^{m}, \forall \boldsymbol{y} \in T(\boldsymbol{x}^{*}) \text{\\} \{\boldsymbol{0}\}, \boldsymbol{y}^{T} \boldsymbol{L}(\boldsymbol{x}^{*}, \boldsymbol{\lambda}^{*}) \boldsymbol{y} > 0
$$

Conclusion:

严格局部极小点

---

### 20.6

线性约束下二次型函数的极小化

---

$$
\begin{align*}
    & \text{minimize} && \frac{1}{2} \boldsymbol{x}^{T} \boldsymbol{Q} \boldsymbol{x} \\
    & \text{subject to} && \boldsymbol{A}\boldsymbol{x} = \boldsymbol{b} \\
    & && \boldsymbol{Q} \in \mathbb{R}^{n} > 0 \\
    & && \boldsymbol{A} \in \boldsymbol{R}^{m \times n}, \mathrm{rank} \boldsymbol{A} = m \leq n
\end{align*}
$$

一般形式的二次规划还包括了条件：$\boldsymbol{x} \geq \boldsymbol{0}$.

---

利用一阶必要条件计算得到唯一的

$$
\boldsymbol{x}^{*} = \boldsymbol{Q}^{-1} \boldsymbol{A}^{T} (\boldsymbol{A} \boldsymbol{Q}^{-1} \boldsymbol{A}^{T})^{-1} \boldsymbol{b}
$$

黑塞矩阵

$$
\boldsymbol{L} = \boldsymbol{Q} > 0
$$

因此求解了唯一极小值。

---

取 $\boldsymbol{Q} = \boldsymbol{I}_{n}$，得到了线性方程 $\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}$ 的[最小范数解](optimize1.md#12.3)

$$
\boldsymbol{x}^{*} = \boldsymbol{A}^{T} (\boldsymbol{A} \boldsymbol{A}^{T})^{-1} \boldsymbol{b}
$$

example：线性二次型调节器

$$
\begin{align*}
    & \text{minimize} && \frac{1}{2} \sum_{i=1}^{N} (q x_{i}^{2} + r u_{i}^{2}) \\
    & \text{subject to} && x_{k} = a x_{k -1} + b u_{k} , (k = 1, \cdots, N)
\end{align*}
$$

加权求和法

---

## 含不等式约束的优化问题

---

### 21.1

Karush-Kuhn-Tucker 条件

---

考虑一般形式的优化问题

$$
\begin{align*}
    & \text{minimize} && f(\boldsymbol{x}) \\
    & \text{subject to} && \boldsymbol{h}(\boldsymbol{x}) = 0 \\
    & && \boldsymbol{g}(\boldsymbol{x}) \leq 0
\end{align*}
$$

其中

$$
\begin{align*}
    & f: \mathbb{R}^{n} \rightarrow \mathbb{R} \\
    & \boldsymbol{h}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m} \\
    & \boldsymbol{g}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{p}
\end{align*}
$$

---

- 不起作用约束 $g_{j}(\boldsymbol{x}^{*}) < 0$
- 起作用约束 $g_{j}(\boldsymbol{x}^{*}) = 0$

等式约束总是起作用约束

正则点：所有起作用约束在此处的梯度向量线性无关。

---

一阶必要条件（KKT）

CONDITION:

- $f, \boldsymbol{h}, \boldsymbol{g} \in C^{1}$
- $\boldsymbol{x}^{*}$ 是正则点和局部极小点

CONCLUSION:

$\exists \boldsymbol{\lambda}^{*} \in \mathbb{R}^{m}, \boldsymbol{\mu}^{*} \in \mathbb{R}^{p}$

$$
\begin{align*}
    & \boldsymbol{\mu}^{*} \geq \boldsymbol{0} \\
    & D[f(\boldsymbol{x}^{*}) + \boldsymbol{\lambda}^{*T} h(\boldsymbol{x}^{*}) + \boldsymbol{\mu}^{*T} g(\boldsymbol{x}^{*})] = \boldsymbol{0}^{T} \\
    & \boldsymbol{\mu}^{*T} \boldsymbol{g}(\boldsymbol{x}^{*}) = 0
\end{align*}
$$

$\boldsymbol{\mu}^{*}$ 为 KKT 乘子。

---

起作用约束的下标集

$$
\begin{align*}
    & g_{j}(\boldsymbol{x}^{*}) = 0 \\
    & j \in J(\boldsymbol{x}^{*})
\end{align*}
$$

由于 $\boldsymbol{\mu}^{*T} \boldsymbol{g}(\boldsymbol{x}^{*}) = \boldsymbol{0}$

$$
\begin{align*}
    & \sum_{j \notin J(\boldsymbol{x}^{*})} \mu_{j} g_{j}(\boldsymbol{x}^{*}) = 0 \\
    & \forall j \notin J(\boldsymbol{x}^{*}), g_{j}(\boldsymbol{x}^{*}) < 0 \\
    & \Rightarrow \forall j \notin J(\boldsymbol{x}^{*}) , \mu_{j} = 0
\end{align*}
$$

---

几何解释

![images/Pasted image 20220611111107.png|500](images/Pasted%20image%2020220611111107.png)

$$
\begin{align*}
    & \nabla f(\boldsymbol{x}^{*}) = - (\mu_{1} \nabla g_{1}(\boldsymbol{x}^{*}) + \mu_{2} \nabla g_{2}(\boldsymbol{x}^{*})) \\
    & \mu_{1}, \mu_{2} > 0
\end{align*}
$$

具体证明过程省略。

---

当问题转化为：

- 求解目标函数的最大值
- 不等式约束为 $\geq$

只需要改变为：

$$
\mu^{*} \leq 0
$$

当上面两种转化同时发生，则 $\mu^{*} \geq 0$

---

在求解问题的最优解时，可以将条件一并列写如下：

$$
\begin{align*}
    & 1. && \boldsymbol{\mu}^{*} \geq (\leq) \boldsymbol{0} \\
    & 2. && D \boldsymbol{f}(\boldsymbol{x}^{*}) + \boldsymbol{\lambda}^{*T} D \boldsymbol{h}(\boldsymbol{x}^{*}) + \boldsymbol{\mu}^{*T} D \boldsymbol{g}(\boldsymbol{x}^{*}) = \boldsymbol{0}^{T} \\
    & 3. && \boldsymbol{\mu}^{*T} g(\boldsymbol{x}^{*}) = 0 \\
    & 4. && \boldsymbol{h}(\boldsymbol{x}^{*}) = \boldsymbol{0} \\
    & 5. && \boldsymbol{g}(\boldsymbol{x}^{*}) \leq (\geq) \boldsymbol{0}
\end{align*}
$$

---

如果 $\boldsymbol{g}(\boldsymbol{x}) = \boldsymbol{x} \geq 0$，不存在等式约束 $\boldsymbol{h}(\boldsymbol{x})$，求目标函数极小值。

$$
\begin{align*}
    & 1. && \boldsymbol{\mu}^{*} \leq 0 \\
    & 2. && D \boldsymbol{f}(\boldsymbol{x}^{*}) + \boldsymbol{\mu}^{*T} = \boldsymbol{0}^{T} \\
    & 3. && \boldsymbol{\mu}^{*T} \boldsymbol{x} = 0 \\
    & 4. && \boldsymbol{x}^{*} \geq 0 \\
\end{align*}
$$

可以消去中间变量 $\boldsymbol{\mu}^{*}$

$$
\begin{align*}
    & 1. && \nabla \boldsymbol{f}(\boldsymbol{x}^{*}) \geq \boldsymbol{0}  \\
    & 2. && \boldsymbol{x}^{*T} D \boldsymbol{f}(\boldsymbol{x}^{*}) = 0 \\
    & 3. && \boldsymbol{x}^{*} \geq \boldsymbol{0} \\
\end{align*}
$$

---

### 21.2

二阶条件

---

同样定义 Lagrange 矩阵

$$
\begin{align*}
    & \boldsymbol{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = \boldsymbol{F}(\boldsymbol{x}) + \sum_{i=1}^{m} \lambda_{i} \boldsymbol{H}(\boldsymbol{x}) + \sum_{i=1}^{p} \mu_{i} \boldsymbol{G}(\boldsymbol{x})
\end{align*}
$$

切线空间

$$
\begin{align*}
    & T(\boldsymbol{x}^{*}) = \{\boldsymbol{y} \in \mathbb{R}^{n}: D \boldsymbol{h}(\boldsymbol{x}^{*}) \boldsymbol{y} = \boldsymbol{0}, D g_{j}(\boldsymbol{x}^{*}) \boldsymbol{y} = 0, j \in J(\boldsymbol{x}^{*}) \}
\end{align*}
$$

参与线性组合的起作用约束的下标集

$$
\tilde{J}(\boldsymbol{x}^{*}, \boldsymbol{\mu}^{*}) = \{ j: g_{j}(\boldsymbol{x}^{*}) = 0, \mu_{j} \geq 0 \}
$$

新的切线空间

$$
\tilde{T}(\boldsymbol{x}^{*}, \boldsymbol{\mu}^{*}) = \{ \boldsymbol{y} \in \mathbb{R}^{n}: D \boldsymbol{h}(\boldsymbol{x}^{*}) \boldsymbol{y} = \boldsymbol{0}, \boldsymbol{g}_{j}(\boldsymbol{x}^{*}) \boldsymbol{y} = 0, j \in \tilde{J}(\boldsymbol{x}^{*}, \mu^{*}) \}
$$

---

二阶必要条件和仅含等式约束的二阶条件形式一致

前提：

- 满足一阶必要条件
- $f, \boldsymbol{h}, \boldsymbol{g} \in C^{2}$
- $\boldsymbol{x}^{*}$ 是正则点

结论：

$$
\exists \lambda^{*} \in \boldsymbol{R}^{m}, \mu^{*} \in \mathbb{R}^{p},  \forall \boldsymbol{y} \in T(\boldsymbol{x}^{*}), \boldsymbol{y}^{T} \boldsymbol{L}(\boldsymbol{x}^{*}, \boldsymbol{\lambda}^{*}, \boldsymbol{\mu}^{*}) \boldsymbol{y} \geq 0
$$

---

二阶充分条件

Condition:

- $\boldsymbol{x}^{*}$ 是正则点
- $f, \boldsymbol{h}, \boldsymbol{g} \in C^{2}$
- 满足一阶必要条件，而且：

$$
\exists \boldsymbol{\lambda}^{*} \in \mathbb{R}^{m}, \exists \boldsymbol{\mu}^{*} \in \mathbb{R}^{p}, \forall \boldsymbol{y} \in \tilde{T}(\boldsymbol{x}^{*}, \boldsymbol{\mu}^{*}), \boldsymbol{y}^{T} \boldsymbol{L}(\boldsymbol{x}^{*}, \boldsymbol{\lambda}^{*}, \boldsymbol{\mu}^{*}) \boldsymbol{y} > 0
$$

Conclusion:

$\boldsymbol{x}^{*}$ 是严格局部极小点。

---

## 凸优化问题

---

### 22.1

introduction

---

### 22.2

凸函数

---

实值函数图像

$$
\begin{align*}
    & f: \Omega \rightarrow \mathbb{R}, \Omega \subset \mathbb{R}^{n} \\
    & \text{graph: } \left\{ \begin{bmatrix} \boldsymbol{x} \\ f(\boldsymbol{x}) \end{bmatrix}: \boldsymbol{x} \in \Omega \right\}
\end{align*}
$$

实值函数上图

$$
\begin{align*}
    & \mathrm{epi}(f) = \left\{ \begin{bmatrix} \boldsymbol{x} \\ \beta \end{bmatrix}: \boldsymbol{x} \in \Omega, \beta \geq f(\boldsymbol{x}) \right\}
\end{align*}
$$

凸函数：上图是凸集

$$
\begin{align*}
    & \forall \boldsymbol{x}_{1}, \boldsymbol{x}_{2} \in \mathrm{epi}(f), \forall \alpha \in [0, 1], \alpha \boldsymbol{x}_{1} + (1 - \alpha) \boldsymbol{x}_{2} \in \mathrm{epi}(f)
\end{align*}
$$

定理：凸函数的定义域是凸集

---

凸函数的另外定义（等价条件）

$$
\begin{align*}
    & f: \Omega \rightarrow \mathbb{R}, \Omega \subset \mathbb{R}^{n} \\
    & \forall \boldsymbol{x}, \boldsymbol{y} \in \Omega, \forall \alpha \in (0, 1)\\
    & f(\alpha \boldsymbol{x} + ( 1 - \alpha ) \boldsymbol{y}) \leq \alpha f(\boldsymbol{x}) + (1 - \alpha) f(\boldsymbol{y})
\end{align*}
$$

凸函数具有

- 可加性：两个凸函数相加仍然是凸函数
- 倍乘性：一个凸函数乘任意一个非负数仍然是凸函数
- 线性组合：一组凸函数以一组非负实数为权线性组合的结果仍然是凸函数
- 取最大值：一组凸函数取最大值得到的函数仍然是凸函数

---

严格凸函数

$$
\begin{align*}
    & f: \Omega \rightarrow \mathbb{R}, \Omega \subset \mathbb{R}^{n} \\
    & \forall \boldsymbol{x}, \boldsymbol{y} \in \Omega, \forall \alpha \in (0, 1)\\
    & f(\alpha \boldsymbol{x} + ( 1 - \alpha ) \boldsymbol{y}) < \alpha f(\boldsymbol{x}) + (1 - \alpha) f(\boldsymbol{y})
\end{align*}
$$

当 $-f$ 是严格凸函数的时候，$f$ 是严格凹函数。

这一点和凹凸的汉字象形是相反的。

---

判断一个二次型函数是否为凸函数？

对于二次型函数

$$
\begin{align*}
    & f(\boldsymbol{x}) = \boldsymbol{x}^{T} \boldsymbol{Q} \boldsymbol{x}, \boldsymbol{Q}^{T} = \boldsymbol{Q}, \boldsymbol{x} \in \Omega \subset \mathbb{R}^{n}
\end{align*}
$$

它是凸函数的等价条件为

$$
\begin{align*}
    & \forall \boldsymbol{x}, \boldsymbol{y} \in \Omega \\
    & (\boldsymbol{x} - \boldsymbol{y})^{T} \boldsymbol{Q} (\boldsymbol{x} - \boldsymbol{y}) \geq 0
\end{align*}
$$

证明需要利用计算结果

$$
\begin{align*}
    & \alpha \boldsymbol{x}^{T} \boldsymbol{Q} \boldsymbol{x} +(1 - \alpha) \boldsymbol{y}^{T} \boldsymbol{Q} \boldsymbol{y} - [\alpha \boldsymbol{x} + ( 1 - \alpha) \boldsymbol{y}]^{T} \boldsymbol{Q} [\alpha \boldsymbol{x} + (1 - \alpha) \boldsymbol{y}] \\
    = & \alpha( 1 - \alpha ) (\boldsymbol{x} - \boldsymbol{y})^{T} \boldsymbol{Q} (\boldsymbol{x} - \boldsymbol{y})
\end{align*}
$$

---

凸函数与多元 Lagrange 终值定理

Condition:

1. $f: \Omega \rightarrow \mathbb{R}, f \in C^{1}$
2. 凸集 $\Omega \subset \mathbb{R}^{n}$

Conclusion:

$$
\begin{align*}
    & f(\boldsymbol{y}) \geq f(\boldsymbol{x}) + D f(\boldsymbol{x}) (\boldsymbol{y} - \boldsymbol{x})
\end{align*}
$$

等价于 $f$ 是凸函数。

---

二阶条件

Condition:

1. $f: \Omega \rightarrow \mathbb{R}, f \in C^{2}$
2. 凸集 $\Omega \subset \mathbb{R}^{n}$

凸函数的等价条件是

$$
\begin{align*}
    & \forall \boldsymbol{x} \in  \Omega, \\
    & \boldsymbol{F}(\boldsymbol{x}) \geq 0
\end{align*}
$$

证明方法：利用 Taylor 定理将它转化成一阶条件，证明充分性时利用反证法。

$$
\begin{align*}
    & f(\boldsymbol{y}) = f(\boldsymbol{x}) + Df(x) (\boldsymbol{y} - \boldsymbol{x}) + \frac{1}{2} (\boldsymbol{y} - \boldsymbol{x})^{T} \boldsymbol{F}(\boldsymbol{x} + \alpha(\boldsymbol{y} - \boldsymbol{x})) (\boldsymbol{y} - \boldsymbol{x}) \\
    & \boldsymbol{x}, \boldsymbol{y} \in \Omega \Rightarrow \boldsymbol{x} + \alpha(\boldsymbol{y} - \boldsymbol{x}) \in \Omega \Rightarrow \boldsymbol{F}(\boldsymbol{x} + \alpha (\boldsymbol{y} - \boldsymbol{x})) \geq 0
\end{align*}
$$

---

### 22.3

凸优化问题

目标函数是凸函数，约束集是凸集的规划问题

---

凸优化问题中的局部极小值就是全局极小值

不严谨的说明：如果不是的话，那么从局部极小值到全局极小值的直线路径上任何一点都比局部极小值要小，这就不再是局部极小值了。

---

全局极值点的集合也是凸集。

证明路径：

- 对于凸集 $\Omega$ 上的单值函数 $g$，$\Gamma_{c} = \{ \boldsymbol{x} \in \Omega: g(\boldsymbol{x}) \leq c \}$ 也是凸集（证明直接套用定义，利用两个不等式传递）
- $g$ 在 $\Omega$ 上的全局极小点构成的集合是凸集（令 $c = \ \mathrm{min}_{\boldsymbol{x} \in \Omega} f(\boldsymbol{x})$）

---

凸优化问题的特殊性

- 对于连续可微的凸函数在无约束条件下的优化问题，一阶必要条件是充分条件；
- 对于凸函数在等式约束下的优化问题，如果约束集是凸集，那么 Lagrange 条件就是充分条件；
- 对于凸函数在等式和不等式约束下的优化问题，如果约束集是凸集，那么 KKT 条件就是充分条件。
