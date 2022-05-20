# Optimization

线性规划

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
    & \boldsymbol{A} = \begin{bmatrix} \boldsymbol{B} & \boldsymbol{D} \end{bmatrix} \\
    & |\boldsymbol{B}| \neq 0 \\
    & \begin{bmatrix} \boldsymbol{B} & \boldsymbol{D} \end{bmatrix} \boldsymbol{x}' = \boldsymbol{b} \\
    & \boldsymbol{B} \boldsymbol{x}_{B} = \boldsymbol{b} \Rightarrow \boldsymbol{x}_{B} = \boldsymbol{B}^{-1} \boldsymbol{b} \\
    & \boldsymbol{x}' = \begin{bmatrix} \boldsymbol{x}_{B} & \boldsymbol{0} \end{bmatrix}
\end{align*}
$$

对 $\boldsymbol{x}'$ 进行恢复，就能得到**基本解** $\boldsymbol{x}$.

---

一些定义

- 基变量：向量 $\boldsymbol{x}_{B}$ 中的元素
- 基本列向量：矩阵 $\boldsymbol{B}$ 中的列向量
- 退化的基本解：如果基本解中的某些基变量为零，基本解退化
- 可行解：满足约束 $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}, \boldsymbol{x} \geq 0$
- 基本可行解：既基本又可行
- 退化的基本可行解

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

PROOF1


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

考虑增广矩阵规范形

$$
\begin{bmatrix} \boldsymbol{I}_{m} & \boldsymbol{Y}_{m,n-m} & \boldsymbol{y}_{0} \end{bmatrix} \sim \begin{bmatrix} \boldsymbol{A}  & \boldsymbol{b} \end{bmatrix}
$$

---

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

对于其他的向量

$$
\begin{align*}
    \boldsymbol{a}_{j} &= \sum_{i=1, i \neq q}^{m} Y_{ij}\boldsymbol{a}_{i} + Y_{pj}\boldsymbol{a}_{p} \\
    \boldsymbol{a}_{j} &= \sum_{\substack{i=1 \\ i \neq p}}^{m} \left(Y_{i j}-\frac{Y_{p j}}{Y_{p q}} Y_{i q}\right) \boldsymbol{a}_{i}+\frac{Y_{p j}}{Y_{p q}} \boldsymbol{a}_{q}
\end{align*}
$$

???

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

（$m$ 个未知数，$n$ 个约束）

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

法线空间是导数矩阵的行空间。

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
\exists \boldsymbol{\lambda}^{*} \in \boldsymbol{R}^{m}, \forall \boldsymbol{y} \in T(\boldsymbol{x}^{*}) \text{\\} \{\boldsymbol{0}\}, \boldsymbol{y}^{T} \boldsymbol{L}(\boldsymbol{x}^{*}, \boldsymbol{\lambda}^{*}) \boldsymbol{y} \geq 0
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

取 $\boldsymbol{Q} = \boldsymbol{I}_{n}$，得到了线性方程 $\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}$ 的[最小范数解](optimize1#12.3)

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

