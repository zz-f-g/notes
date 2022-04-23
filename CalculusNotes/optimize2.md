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