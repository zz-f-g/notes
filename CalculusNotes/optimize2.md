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

如何从一组基本可行解得到另一组？

- 入集向量
- 出集向量

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

