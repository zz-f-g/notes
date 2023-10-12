# 线性代数及其应用（二）

[TOC]

## 四、向量空间

### 4.1_向量空间和子空间

#### 向量空间

定义：向量的非空矩阵

- 封闭性：加法完整，数乘完整
- 各种“律”：交换律、结合律、分配律（向量、标量）
- 0 向量、负向量、1 的数乘

1. $\forall u, v \in V, u+v \in V$
2. $u + v = v + u$
3. $(u+v)+w = u+(v+w)$
4. $\exists \vec{0} \in V, \forall u \in V, u + \vec{0} = u$
5. $\forall u \in V, \exists -u \in V, u + (-u) = 0, \forall v \neq -\vec{u} \in V, u + v \neq 0$
6. $\forall c \in R, \forall \vec{u} \in V, c \vec{u} \in V$
7. $c(\vec{u} + \vec{y}) = c \vec{u} + c \vec{v}$
8. $(c + d) \vec{u} = c \vec{u} + d \vec{u}$
9. $c(d \vec{u}) = (cd) \vec{u}$
10. $1 \vec{u} = \vec{u}$

***Attention***: The DIFFERENCE between the zero entry $\vec{0}$ and scalar 0.

证明一个变换是线性变换：

1. ==变换后的元素仍然在原来的空间中==
    1. 加法封闭
    2. 乘法封闭
2. 线性 $L(a \boldsymbol{u} + b \boldsymbol{v}) = a L(\boldsymbol{u}) + b L(\boldsymbol{v})$

线性空间举例：

- $R^n$
- 物理向量的集合
- 双无穷数列（doubly infinite sequences）的集合
- 有限多项式（次数小于 $n$）的集合，但是次数恰好等于 n 的多项式不构成线性空间，因为不存在 0.
- 实值一元函数
- $m \times n$ 矩阵

#### 子空间

定义：向量空间的子集

- 包含 0
- 加法和乘法的封闭性

举例：

- 零空间 $\{ \vec{0} \}$
- 确定最高次幂的多项式集合
- 注意：$R^2$ 不是 $R^3$ 的子空间，以为他们的元素不一样

#### 集合张成的子空间

可以表示为集合中向量的线性组合的向量的集合。

$$
H = \text{Span} \{ \vec{v_1}, \cdots, \vec{v_p} \}
$$

#### 定理一：向量张成的空间是子空间

### 4.2_零空间，列空间和线性变换

表示子空间的两种方式

- 齐次线性方程组的解集
  - 零空间：满足矩阵方程 $A \vec{x} = \vec{0}$ 的向量的集合。$\text{Nul} A = \{ x: x \in \mathbb{R}^n \text{ and } A \vec{x} = \vec{0} \}$
- 向量集合的线性组合
  - 列空间：给出矩阵 $A$ 的列向量张成的空间。$\text{Col} A = \text{Span} \{\vec{a_1}, \cdots, \vec{a_n} \}$

#### 定理二：零空间是子空间

#### 零空间的显式描述

在解线性方程组的过程中，将系数矩阵化为简化阶梯型。

某些情况下发现解集可以表示为线性组合，其中线性组合的权为自由变量的值（通解）。

两个注意点

- 通过上述方法求出的零空间中的向量自动是**线性无关**的。
- 向量的个数等于自由变量的个数。

例如：

$$
\begin{aligned}
A &= \begin{bmatrix}
-3 & 6 & -1 & 1 & -7\\
1 & -2 & 2 & 3 & -1\\
2 & -4 & 5 & 8 & -4
\end{bmatrix} \\
A &\sim \begin{bmatrix}
0 & 0 & 5 & 10 & -10\\
1 & -2 & 2 & 3 & -1\\
0 & 0 & 1 & 2 & -2
\end{bmatrix} \\
& \sim \begin{bmatrix}
0 & 0 & 1 & 2 & -2\\
1 & -2 & 2 & 3 & -1\\
0 & 0 & 0 & 0 & 0
\end{bmatrix} \\
& \sim \begin{bmatrix}
0 & 0 & 1 & 2 & -2\\
1 & -2 & 0 & -1 & 3\\
0 & 0 & 0 & 0 & 0
\end{bmatrix} \\
\boldsymbol{x} &= x_{2} \begin{bmatrix}
2\\ 1\\ 0\\ 0\\ 0
\end{bmatrix} + x_{4} \begin{bmatrix}
1\\ 0\\ -2\\ 1\\ 0
\end{bmatrix} + x_{5} \begin{bmatrix}
-3\\ 0\\ 2\\ 0\\ 1
\end{bmatrix}
\end{aligned}
$$

这样就将零空间转化成了列空间。

$$
\mathrm{Nul}A = \mathrm{Col} \begin{bmatrix}
2 & 1 & -3 \\ 
1 & 0 & 0 \\ 
0 & -2 & 2 \\ 
0 & 1 & 0 \\ 
0 & 0 & 1
\end{bmatrix}
$$

#### 定理三：列空间是子空间

$$
\text{Col}A = \{ \vec{b}: \vec{b} = A \vec{x} \text{ and } \vec{x} \in \mathbb{R}^n \}
$$

### 4.3_线性无关集和基

#### 定理四：线性相关的充要条件

对于首个向量不为零向量的向量集合（顺序已经排列好），存在一个向量可以表示为前面的向量的线性组合。

第一章中有[类似的证明](#定理七：线性相关与线性组合)。

#### 子空间的基

张成给定子空间的一组线性无关的向量集合。

$R^n$ 空间的**标准基**：单位矩阵 $I_n$ 的列构成的集合。

#### 定理五：生成基定理

CONDITION:

- $S = \{v_1, \cdots, v_p\}$
- $H = \text{Span} \{v_1, \cdots, v_p\}$
- $v_k$ is a linear conbination of other vectors in $S$.

CONCLUSION:

- $H = \text{Span} \{v_1, \cdots, v_p\}$ without $v_k$
- Some subset of $S$ can span $H$.

#### 定理六：列空间和主元列

矩阵 $A$ 的列空间的基是 $A$ 的主元列的集合。

The pivot columns of the matrix $A$ form a basis of $\text{Col} A$.

==矩阵的行等价矩阵的列空间和其本身的列空间并不一定相同。==

- 初等行变换不改变矩阵的行空间
- 初等列变换不改变矩阵的列空间

### 4.4_坐标系

#### 定理七：唯一表示定理

CONDITION:

- $\{\vec{b_1}, \cdots, \vec{b_n}\}$ is a basis of a vector space $V$.
- $\{c_1, \cdots, c_n\}$ is a scalar set.
- $B \vec{c} = \vec{x}$

CONCLUSION:

- $\{c_1, \cdots, c_n\}$ is unique.

基于定理七的表示符号，定义坐标系：
$$
[x]_B = \begin{bmatrix}
c_1 \\ \cdots \\ c_n
\end{bmatrix}
$$
定义坐标映射：
$$
x \rightarrow [x]_B
$$

#### 定理八：坐标映射是单射线性变换

从向量空间 $V$ 到向量空间 $K$ 的一个单射线性变换被称为一个**同构**。

### 4.5_向量空间的维数

#### 定理九：基的数量上限

在向量空间中，如果一组基有 n 个向量，那么任意多于 n 个向量的集合中的向量必然线性相关。

#### 定理十：基的数量唯一性

如果向量空间的一组基有 n 个向量，那么任意一组基都必须刚好有 n 个向量。

PROOF:

同一个向量空间的两组基：$B_1, B_2$.

根据定理九，$B_1$ 的基的数量不会多于 $B_2$，同样地，$B_2$ 的基的数量不会多于 $B_1$。因此两组基中向量的数量相等。

**向量空间的维数**：向量空间的一组基中向量的数量。（前提是向量空间是有限维的，即被一个有限向量集合张成）
$$
\text{dim} R^n = n
$$

#### 定理十一：子空间的维数

有限维的向量空间 $V$ 存在子空间 $H$，则：
$$
\text{dim} H \leq \text{dim} V
$$

#### 定理十二：基定理

任意维数 $p \geq 1$ 的向量空间 $V$ 中，任意有 $p$ 个向量，且线性无关（或者能够张成 $V$）的向量集合是向量空间 $V$ 的一组基、

证明需要使用定理十。

#### 零空间和列空间的维数

- 零空间的维数等于自由变量的数量；
- 列空间的维数等于主元列的数量，等于行空间的维数。

因此：

$$
\forall \boldsymbol{A} \in \mathbb{R}^{m \times n}, \mathrm{Nul} \boldsymbol{A} + \mathrm{Col} \boldsymbol{A} = n
$$

### 4.6_秩

矩阵中和秩相等的量：

- 主元数量
- 线性无关的列的最大数量（列空间的维数）
- 线性无关的行的最大数量（行空间的维数）

#### 行空间

$$
\text{Row} A = \text{Col} A^T
$$

#### 定理十三：行等价则行空间相同

如果两个矩阵行等价，则两个矩阵的行空间相同。如果其中一个矩阵是阶梯型，那么这个矩阵的非零行构成了两个矩阵的行空间的一组基。

PROOF:

- 行化简以后，每一行是原来一行的线性组合。
- 行化简可逆

在**行空间、列空间、零空间**三种空间中：

- 行空间、列空间可以直接通过矩阵中的元素表示（找到主元行（通过列等价变换）、主元列（通过行等价变换））
- 零空间不可以直接通过矩阵中的元素表示（化简为最简阶梯型，找自由变量）

#### 秩的定义

矩阵列空间的维数。

行列式定义：在 $m \times n$ 矩阵 $A$ 中，任取 $k$ 行 $k$ 列，位于这些行列交叉处的元素不改变相对顺序构成的新行列式称为矩阵 $A$ 的 $k$ 阶子式。矩阵的秩定义为矩阵最高阶非零子式的阶数。

阶数高于 $k$ 的所有子式都是 0.

#### 定理十四：秩定理

对于 $m \times n$ 矩阵 $A$：

$$
\text{rank} A = \text{dim} \text{Col} A = \text{dim} \text{Row} A = n - \text{dim} \text{Nul} A
$$

PROOF：

主元列的数量 + 非主元列的数量（自由变量的数量）= 列的数量

后续会证明：行空间和零空间正交

#### 秩和逆矩阵定理

下面的命题等价：（对于 n 阶矩阵 $A$ ）

- 矩阵 $A$ 可逆
- $A$ 的列是 $\mathbb{R}^{n}$ 的一组基
- $A$ 的列空间是 $\mathbb{R}^n$
- $A$ 的列空间的维数是 n
- $A$ 的秩是 n
- $A$ 的零空间是 $\{0\}$（只有平凡解）
- $A$ 的零空间的秩是 0

#### 比较显然的秩的性质

- $\max \{ \mathrm{rank} A, \mathrm{rank} B \} \leq \mathrm{rank} ([A, B]) \leq \mathrm{rank} (A) + \mathrm{rank} (B)$
- $AX = B$ 有解 $\iff \mathrm{rank}A = \mathrm{rank} [A, B]$

#### 不那么显然的秩的性质

- $\mathrm{rank} (A + B) \leq \mathrm{rank} (A) + \mathrm{rank} (B)$

PROOF:

$$
\begin{aligned}
& \begin{bmatrix}
A & A+B\\
O & B
\end{bmatrix} = \begin{bmatrix}
I & I\\
O & I
\end{bmatrix} \begin{bmatrix}
A & O\\
O & B
\end{bmatrix} \begin{bmatrix}
I & I\\
O & I
\end{bmatrix} \\
& \mathrm{rank}(A+B) \leq \mathrm{rank} \begin{bmatrix} A & A+B \\ O & B  \end{bmatrix} = \mathrm{rank} \begin{bmatrix} A & O \\ O & B  \end{bmatrix} = \mathrm{rank}A+\mathrm{rank}B
\end{aligned}
$$

- $\mathrm{rank} (AB) \leq \min \{ \mathrm{rank} A, \mathrm{rank} B\}$

PROOF:

$$
\begin{aligned}
&\begin{bmatrix} AB & A \end{bmatrix} \begin{bmatrix} I & 0 \\ -B & I \end{bmatrix} = \begin{bmatrix} O & A \end{bmatrix} \\
&\begin{bmatrix} I & -A \\ O & I \end{bmatrix} \begin{bmatrix} AB \\ B \end{bmatrix} = \begin{bmatrix} O \\ B \end{bmatrix} \\
\end{aligned}
$$

- $A_{m \times n} B_{n \times l} = O \Rightarrow \mathrm{rank} A + \mathrm{rank} B \leq n$（利用了[秩定理](#定理十四：秩定理)，B 的列向量属于 A 的零空间，B 的维数不可能大于 A 的零空间的维数）

- $A \in R^{m \times n}, B \in R^{n \times p},\mathrm{rank}A + \mathrm{rank}B \leq n + \mathrm{rank}AB$ （作为上面第三条性质的推广）

PROOF:
$$
\begin{aligned}
n + \mathrm{rank}AB &= \mathrm{rank} \left( \begin{bmatrix} AB & A \\ O & I_n \end{bmatrix} \begin{bmatrix} - I_p & O \\ B & I_n \end{bmatrix} \right) \\
&= \mathrm{rank}\begin{bmatrix} O & A \\ B & I_n \end{bmatrix} \\
&\geq \mathrm{rank} \begin{bmatrix} O & A \\ B & O \end{bmatrix} \\
&= \mathrm{rank}A + \mathrm{rank}B
\end{aligned}
$$

上面的性质的证明方法普遍都构造了分块矩阵，探讨了他们之间的等价关系。

不那么显然的秩的性质中，第一条和第四条在一些情况下可以夹逼出两个矩阵秩的和的值。

#### 秩的性质的应用：消去律

1. 如果 A 列满秩，$AB = AC \Rightarrow B = C$
2. 如果 A 行满秩，$BA = CA \Rightarrow B = C$

PROOF:

$$
\begin{aligned}
AB &= AC \\
A(B-C) &= O \\
\mathrm{rank}A + \mathrm{rank}(B-C) &\leq n + \mathrm{rank}O = n \\
\mathrm{rank}(B-C) &\leq n - \mathrm{A} = 0 \\
B &= C
\end{aligned}
$$

行满秩的情况可以用转置证明，也可以用类似的方法。

这些性质看起来奇技淫巧，实际上都有几何本质：$A$ 列满秩，意味着 $A \boldsymbol{x} = \boldsymbol{b}$ 存在解时仅有唯一解。

### 4.7_基变换

#### 定理十五：基变换的唯一性

向量空间 $V$ 存在两组基
$$
\begin{align*}
B = \{\vec{b_1}, \cdots, \vec{b_2}\} \\
C = \{\vec{c_1}, \cdots, \vec{c_2}\}
\end{align*}
$$
存在唯一的坐标变换矩阵
$$
P = \begin{bmatrix}
[\vec{b_1}]_C, \cdots, [\vec{b_2}]_C
\end{bmatrix}
$$
使得
$$
[\vec{x}]_C = P [\vec{x}]_B
$$

$$
B = C P \Rightarrow P = C^{-1}B
$$

### 4.8_差分方程中的应用

信号空间 $\mathbb{S}$ 中的线性无关：

$$
\forall k \in \mathbb{Z}, \sum_{i=1}^{n} c_{i} u_{i} = 0 \Rightarrow \forall i, c_{i} = 0
$$

Casorati Matrix

$$
\begin{bmatrix}
u_{k} & v_{k} & w_{k} \\ 
u_{k+1} & v_{k+1} & w_{k+1} \\ 
u_{k+2} & v_{k+2} & w_{k+2}
\end{bmatrix} \begin{bmatrix}
c_{1} \\ c_{2} \\ c_{3}
\end{bmatrix} = \boldsymbol{0}
$$

对至少一个 $k$，Casorati 矩阵可逆，则信号线性无关。

$n$ 阶线性差分方程

$$
a_{0} y_{k+n} + a_{1} y_{k+n-1} + \cdots + a_{n} y_{k} = z_{k}
$$

一般取 $a_{0} = 1$. 若 $z_{k} = 0$，则称齐次线性差分方程。

#### 定理十六：初始条件给定的差分方程解唯一

线性空间视角下的差分信号：

$$
\begin{aligned}
T &: \mathbb{S} \rightarrow \mathbb{S} \\
w_{k} &= y_{k+n} + a_{1} y_{k+n-1} + \cdots + a_{n} y_{k}
\end{aligned}
$$

在不考虑特征方程重根的情况下，

$$
\begin{aligned}
0 &= y_{k+n} + a_{1} y_{k+n-1} + \cdots + a_{n} y_{k} \\
y_{k} &= r^{k}
\end{aligned}
$$

构成线性变换 $T$ 的零空间（核），零空间的维数是 $n$，每个特征根构成一个基。

#### 定理十七：线性差分方程的零空间的维数是方程的阶数

和矩阵方程 $\boldsymbol{A} \boldsymbol{x} = \boldsymbol{b}$ 类似，差分方程的非齐次方程也是特解加零空间基的线性组合的形式。

齐次线性差分方程组可以改写成一阶形式

$$
\boldsymbol{x}_{k+1} = \boldsymbol{A} \boldsymbol{x}_{k}
$$

$$
\begin{aligned}
0 &= y_{k+n} + a_{1}y_{k+n-1} + \cdots + a_{n} y_{k} \\
\boldsymbol{x}_{k} &= \begin{bmatrix} y_{k} & y_{k+1} & \cdots & y_{k+n-1} \end{bmatrix}^{T}\\
\boldsymbol{x}_{k+1} &= \begin{bmatrix} y_{k+1} & y_{k+2} & \cdots & y_{k+n} \end{bmatrix}^{T} \\
&= \begin{bmatrix} y_{k+1} & y_{k+2} & \cdots & -a_{n} y_{k} - a_{n-1} y_{k+1} - \cdots -a_{1} y_{k+n-1} \end{bmatrix}^{T} \\
&= \begin{bmatrix}
0 & 1 & 0 & \cdots & 0 & 0 \\
0 & 0 & 1 & \cdots & 0 & 0 \\
0 & 0 & 0 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots  & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & 0 & 1 \\
-a_{n} & -a_{n-1} & -a_{n-2} & \cdots & -a_{2} & -a_{1}
\end{bmatrix} \boldsymbol{x}_{k}
\end{aligned}
$$

其实这个矩阵被称为多项式 $p_{n}(t) = t^{n} + a_{1}t^{n-1} + \cdots + a_{n}$ 的**友矩阵**。友矩阵的重要性质是，友矩阵的特征值集合等于多项式的解集。

要证明这个结论，需要求解友矩阵的特征多项式，这里应用数学归纳法证明：

$$
\mathrm{det} (\boldsymbol{C}_{p_{n}} - \lambda \boldsymbol{I}_{n}) = (-1)^{n} p_{n}(\lambda)
$$

归纳奠基很显然，这里只写归纳递推。

$$
\begin{aligned}
C_{p_{n+1}} - \lambda I_{n+1} &= 
\begin{bmatrix}
-\lambda & 1 & 0 & \cdots & 0 \\
0 & -\lambda & 1 & \cdots & 0 \\
\vdots & & & & \vdots \\
0 & 0 & 0 & \cdots & 1\\
-a_{n+1} & -a_{n} & -a_{n-1} & \cdots & -a_{1} -\lambda
\end{bmatrix} \\
|\boldsymbol{C}_{p_{n+1}} - \lambda \boldsymbol{I}_{n+1} | &= -\lambda |\boldsymbol{C}_{p_{n}} - \lambda I_{n}| - a_{n+1} \\
&= -\lambda p_{n}(\lambda) - a_{n+1} \\
&= - p_{n+1}(\lambda)
\end{aligned}
$$

求行列式按第一行展开。证毕。

$\boldsymbol{C}_{p}$ 的特征值是 $\left\{ \lambda_{1}, \cdots, \lambda_n \right\}$，那么特征向量呢？发现特征向量刚好构成了范德蒙德矩阵。

$$
\boldsymbol{V} = \begin{bmatrix}
1 & 1 & \cdots & 1 \\ 
\lambda_{1} & \lambda_{2} & \cdots & \lambda_{n} \\ 
\vdots & & & \vdots \\ 
\lambda_{1}^{n} & \lambda_{2}^{n} & \cdots & \lambda_{n}^{n}
\end{bmatrix} = \begin{bmatrix} \boldsymbol{v}_{1} & \cdots & \boldsymbol{v}_{n} \end{bmatrix}
$$

因为

$$
\begin{aligned}
\boldsymbol{C}_{p} \boldsymbol{v}_{i} &= \begin{bmatrix}
\lambda_{i} & \lambda_{i}^{2} & \cdots & \lambda_{i}^{n} & -a_{1} \lambda_{i}^{n} - a_{2} \lambda_{i}^{n-1} - \cdots - a_{n}\lambda_{i}
\end{bmatrix}^{T} \\
&= \lambda_{i} \begin{bmatrix}
1 & \lambda_{i} & \cdots & \lambda_{i}^{n} & -a_{1} \lambda_{i}^{n-1} - a_{2} \lambda_{i}^{n-2} - \cdots - a_{n}
\end{bmatrix}^{T} \\
&= \lambda_{i} \boldsymbol{v}_{i}
\end{aligned}
$$

这个结论说明了为什么一个齐次线性差分方程组解空间的基总是由特征多项式的根的幂次组成。因为对这个矩阵作相似对角化以后，对角矩阵正是由特征值（多项式解）构成。

$$
\boldsymbol{V}^{-1} \boldsymbol{x}_{k+1} = \boldsymbol{\Lambda} \boldsymbol{V}^{-1} \boldsymbol{x}_{k}
$$

### 4.9_马尔科夫链中的应用

$$
\boldsymbol{x}_{k+1} = \boldsymbol{P} \boldsymbol{x}_{k}
$$

- $\boldsymbol{x}_{k}$ 是状态向量，和为 1
- $\boldsymbol{P}$ 是随机矩阵，每一列的和为 1

$$
\boldsymbol{1}_{n}^{T} = \boldsymbol{1}_{n}^{T} P
$$

稳态向量 $\boldsymbol{q}: \boldsymbol{q} = \boldsymbol{P} \boldsymbol{q}$

#### 定理十八：马尔可夫链收敛

## 五、特征值和特征向量

### 5.1_特征向量和特征值

定义：对于 n 阶方阵 $A$，如果
$$
\exists \lambda \in R, \exists \vec{x} \in R^n \text{ and } \vec{x} \neq \vec{0}, A \vec{x} = \lambda x
$$
则称 $\lambda$ 为矩阵 $A$ 的特征值，$\vec{x}$ 为矩阵 $A$ 的特征向量。

检验矩阵 $A$ 是否有特征值 $\lambda$：判断矩阵 $A - \lambda I$ 是否可逆。该矩阵的零空间就是特征向量的子空间（定义为特征空间）。==同一个特征值可以对应线性无关的一组特征向量，取决于零空间的维数。==

0 是 $A$ 的特征值当且仅当 $A$ 不可逆。

#### 定理一：三角矩阵的特征值

三角矩阵的对角元素是其特征值。

#### 定理二：特征向量之间线性无关

PROOF：

假设特征向量

$$
\{\vec{v_1}, \cdots, \vec{v_r}\}
$$

之间线性相关，则由于 $\vec{v_1} \neq 0$

$$
\exists p < r - 1, \vec{v}_{p+1} = \sum_{i=1}^p c_i \vec{v_i}
$$

而 $\{\vec{v}_1, \cdots, \vec{v}_p\}$ 线性无关。则：

$$
\begin{align}
A \vec{v}_{p+1} &= \sum_{i=1}^p c_i A \vec{v}_i \\
\lambda_{p+1} \vec{v}_{p+1} &= \sum_{i=1}^p c_i \lambda_i \vec{v}_i
\end{align}
$$

式（15）乘以 $\lambda_{p+1}$，减去式（17）：

$$
\sum_{i=1}^p c_i (\lambda_{p+1} - \lambda_i) \vec{v}_i = 0
$$

与线性无关的假设相矛盾，证毕。

#### 特征值的线性

$$
A \vec{x} = \lambda \vec{x} \iff f(A) \vec{x} = f(\lambda) \vec{x}
$$

其中，

$$
f(x) = \sum_i c_i x^i
$$

对于 i 取负数的情况，需要先证明：

$$
A \vec{x} = \lambda \vec{x} \Rightarrow A^{-1} \vec{x} = \frac{1}{\lambda} \vec{x}
$$

如果 $\{\lambda_n\}$ 是 n 阶方阵 $A$ 的全部特征值，那么 $\{f(\lambda_n)\}$ 是方阵 $A$ 的全部特征值。

如果：

$$
f(A) = O
$$

则：

$$
\vec{0} = f(\lambda) \vec{x} \Rightarrow f(\lambda) = 0
$$

方阵 $A$ 的特征值是方程 $f(\lambda) = 0$ 的解。

但是定理反过来不成立。

通过上面的定理也可以说明 Cayley-Hamilton 定理：

如果 $p(\lambda)$ 是矩阵 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ 的特征多项式，则 $p(\boldsymbol{A}) = \boldsymbol{O}$.

证明方法：对于 $\boldsymbol{A}$ 的任意特征值 $\lambda$，矩阵 $p(\boldsymbol{A})$ 的对应特征值是 $p(\lambda) = 0$.

如果方阵的任意一行的和都等于 $\lambda$，那么 $\lambda$ 是该方阵的特征值，特征向量是 $[1,1,1]^T$.

例题：求矩阵 $A = \begin{bmatrix} a & b & \cdots & b \\ b & a & \cdots & b \\ \vdots & \vdots & \ddots & \vdots \\ b & b & \cdots & a \end{bmatrix}$ 的特征值。

解：[回顾行列式技巧](LinearAlgebra1.md#3.4_行列式求解技巧简介)

$$
\begin{aligned}
\boldsymbol{A} &= b \boldsymbol{J}_{n} + (a-b) \boldsymbol{I}_{n} \\
\boldsymbol{A} - \lambda \boldsymbol{I}_{n} &= (a -b-\lambda)I_{n} + b \boldsymbol{J}_{n} \\
\mathrm{det}\boldsymbol{A} &= (a-b)^{n-1} [a + (n-1)b] && (a \rightarrow a - \lambda) \\
\mathrm{det}(\boldsymbol{A} - \lambda \boldsymbol{I}_{n}) &= (a - b - \lambda)^{n-1} [a - \lambda + (n-1)b]
\end{aligned}
$$

### 5.2_特征方程

#### 定理三：特征值与可逆矩阵定理

$A$ 是可逆的 $\iff$ 0 不是 $A$ 的特征值。

$$
\det(A - \lambda I) = 0
$$

**特征方程**的解是矩阵 $A$ 的**特征根**。如果 $A$ 是 $n \times n$ 方阵，那么特征方程是**特征根** $\lambda$ 的 $n$ 次多项式，称为**特征多项式**。

特征根 $\lambda$ 作为特征方程根的重数定义为 $\lambda$ 的**代数重数**。

$$
\sum_i n_i = n
$$

几何重数 $d_i$ 指的是方程 $A - \lambda I = 0$ 的解空间的维数。满足条件：

$$
1 \leq d_i \leq n_i
$$

PROPERTIES:

- 方阵的所有特征根的和等于方阵主对角线上元素的和，定义为矩阵的**迹**。
- 方阵的所有特征根的积等于方阵的行列式。

PROOF：

$$
\begin{aligned}
    |A - \lambda I| &= c \prod_{i=1}^{n} (\lambda - \lambda_{i}) \\
    &= c \left[ \lambda^{n} - \left( \sum_{i=1}^{n} \lambda_{i} \right) \lambda^{n-1} + \cdots + (-1)^n \prod_{i=1}^n \lambda_i \right]
\end{aligned}
$$

根据行列式的性质可知，该多项式的最高项和次高项都来源于矩阵 $A - \lambda I$ 的主对角线元素的积。

对于次高项可以得到特征根的和等于方阵主对角线元素之和——**迹**。

对于最高项可以得到 $c = (-1)^n$

取 $\lambda = 0$，$|A| = c (-1)^n \Pi_{i=1}^n \lambda_i = \Pi_{i=1}^n \lambda_i$。证明完毕。

这也推导出了另一个分块矩阵的结论：

如果 $\boldsymbol{A} \in \mathbb{R}^{m \times m}, \boldsymbol{B} \in \mathbb{R}^{n \times n}$

$\begin{bmatrix} \boldsymbol{A} & \boldsymbol{X} \\  \boldsymbol{O} & \boldsymbol{B} \end{bmatrix}$ 的特征多项式 $p(\lambda) = p_{A}(\lambda)p_{B}(\lambda)$，其特征值集合由 $\boldsymbol{A}, \boldsymbol{B}$ 特征值集合叠加构成。$\begin{vmatrix} \boldsymbol{A} & \boldsymbol{X} \\ \boldsymbol{O} & \boldsymbol{B} \end{vmatrix} = \mathrm{det}(\boldsymbol{A}) \cdot \mathrm{det}(\boldsymbol{B})$

#### 相似性

对于 $n \times n$ 方阵 $A$ 和 $B$，如果存在可逆矩阵 $P$：

$$
P^{-1} A P = B
$$

则定义 $A$ 相似于 $B$.

相似矩阵运算具有线性

$$
\begin{aligned}
A &= PBP^{-1} \Rightarrow f(A) = P f(B) P^{-1} \\
f(A) &= \sum_i c_i A^i
\end{aligned}
$$

#### 定理四：相似矩阵的特征多项式

相似矩阵的特征多项式相同，具有相同的特征值和重数。

PROOF：

$$
\begin{align}
\det(B - \lambda I) &= \det(P^{-1} B P - P^{-1} (\lambda I) P) \\
&= \det(P^{-1} (A - \lambda I) P) \\
&= \det(P) \det(A - \lambda I) \det(P) \\
&= \det(A - \lambda I)
\end{align}
$$

ATTENTION:

1. 有相同的特征值不代表相似；
2. 相似性和行等价不同，行等价通常都会改变特征值。

求解差分方程 $\vec{x}_{k+1} = A \vec{x}_k$，已知 $\vec{x}_0$。

- 求解 $A$ 的所有特征值 $\lambda_i$ 和对应特征空间 $S_i$ 的基。
- 对于所有特征空间均为一维的情况：
  - 求解 $[\vec{v}_1 \cdots \vec{v}_i] \vec{c} = \vec{x}_0$
  - $\vec{x}_k = [\lambda_1^k \vec{v}_1 \cdots \lambda_i^k \vec{v}_i] \vec{c}$

#### 求特征值的QR算法

把待求矩阵 $A$ 分解为 $A = QR$，其中 $Q$ 是正交矩阵，$R$ 是上三角矩阵。

- $A = Q_1 R_1, A_1 = R_1 Q_1$
- $A_1 = Q_2 R_2, A_2 = R_2 Q_2$
- ……

证明矩阵集合 $\{A_i\}$ 两两相似。

PROOF：

$$
Q_1^{-1}AQ_1 = Q_1^{-1} Q_1 R_1 Q_1 = A_1
$$

其他同理。

由于相似矩阵的特征值相同，因此可以通过求解较好求解（？）的矩阵 $A_n$ 来求解 $A$。

### 5.3_对角化

如果方阵 $A$ 相似于对角矩阵，即存在可逆矩阵 $P$ 和对角矩阵 $D$ 使得 $A = PDP^{-1}$，则称 $A$ **可对角化**。

对角化使得计算一般矩阵的幂更加容易。

$$
A^{n} = (P D P^{-1})^{n} = P D^{n} (P{-1})^{n}
$$

#### 定理五和七：对角化定理

一组充分必要条件：

- $n \times n$ 矩阵 $A$ 可对角化。此时对于 $A = PDP^{-1}$，可逆矩阵 $P$ 的每一列都是线性无关的特征向量，对角矩阵 $D$ 的对角元素是对应的特征值。
- $A$ 有 $n$ 个线性无关的特征向量（但不是 $n$ 个不同的特征值）。
- $\sum_i d_i = n \iff \forall i, d_i = n_i$ 任意特征值的几何重数和代数重数相等，几何重数的和等于矩阵的阶数。

PROOF:

$$
\begin{aligned}
AP &= \begin{bmatrix} A\vec{v}_1 \cdots A \vec{v}_n \end{bmatrix} \\
PD &= \begin{bmatrix} \lambda_1 \vec{v}_1 \cdots \lambda_n \vec{v}_n \end{bmatrix}
\end{aligned}
$$

如果 $A$ 可以对角化，那么 $P$ 可逆，特征方程成立。

如果有 $n$ 个特征向量，可以构造这样的矩阵 $P$ 和 $D$，使得 $A$ 可对角化。

#### 定理六：对角化的充分条件

有 $n$ 个相异的特征值，则可以对角化。

对角化的步骤

- 求特征值 $\{ \lambda_i \}$
- 求出和矩阵同阶的线性无关特征向量组 $\{ \vec{v}_i \}$，如果找不到则不能对角化。
- 用特征向量组构造过渡矩阵 $P$
- 用特征值构造对角矩阵 $D$
- 验证 $AP = PD$

不可对角化的矩阵在几何意义上是作切变（shear）。

### 5.4_特征向量与线性变换

线性变换 $T: V \rightarrow M$，线性空间 $V$ 维度为 $n$，线性空间 $W$ 维度为 $m$. $V, W$ 各有一组基 $B, C$，求 $[T(\boldsymbol{x})]_{C}$ 和 $[\boldsymbol{x}]_{B}$ 之间的关系。

$$
\begin{aligned}
\boldsymbol{x} &= \sum_{i=1}^{n} ([\boldsymbol{x}]_{B})_{i} \boldsymbol{b}_{i} \\
T(\boldsymbol{x}) &= \sum_{i=1}^{n} \left( [x]_{B} \right)_{i} T(\boldsymbol{b}_{i}) \\
[T(\boldsymbol{x})]_{C} &= \sum_{i=1}^{n} ([\boldsymbol{x}]_{B})_{i} [T(\boldsymbol{b}_{i})]_{C} \\
&= \begin{bmatrix} [T(\boldsymbol{b}_{1})]_{C} & \cdots & [T(\boldsymbol{b}_{n})]_{C} \end{bmatrix} \cdot [\boldsymbol{x}]_{B} \\
&= \boldsymbol{M} [\boldsymbol{x}]_{B}
\end{aligned}
$$

#### 定理八：对角矩阵表示

对于可以相似对角化的 $n$ 阶矩阵 $A = PDP^{-1}$，以过渡矩阵 $P$ 的列向量组为线性空间 $R^n$ 的一组基 $B$，则对角矩阵 $D$ 为在基 $B$ 表示下的线性变换 $x \rightarrow Ax$ 的标准矩阵。

PROOF:

$$
\begin{aligned}
\vec{x} &= P [\vec{x}]_P \\
A \vec{x} &= P [A \vec{x}]_P \\
& = A P [\vec{x}]_P \\
[A \vec{x}]_P &= P^{-1} A P [\vec{x}]_P \\
&= D [\vec{x}]_P
\end{aligned}
$$

### 5.5_复特征值

有复数特征值和复数元素特征向量的矩阵对应的线性变换存在旋转。

复数的共轭运算在复数线性代数中的推广：

- $\overline{rx} = \overline{r} \ \overline{x}$
- $\overline{Bx} = \overline{B} \overline{x}$
- $\overline{BC} = \overline{B} \ \overline{C}$
- $\overline{rB} = \overline{r} \overline{B}$

当 $A$ 是实数矩阵时，其特征值必然以共轭复数对的形式出现。

PROOF:

$$
\begin{aligned}
A &= \overline{A} \\
A \overline{x} &= \overline{A} \overline{x} \\
&= \overline{Ax} \\
&= \overline{\lambda} \overline{x}
\end{aligned}
$$

Both $\lambda$ and $\overline{\lambda}$ are eigenvalues with the base of its eigenspace are conjugate eigenvectors.

#### 定理九：二阶旋转方阵与复特征值

对于 $2 \times 2$ 矩阵的简单情况。

设 $A \in \mathbb{R}^{2 \times 2}$，有复特征值 $\lambda=a - ib (b \neq 0)$ 及对应的 $\mathbb{C}^{2}$ 中的复特征向量 $\boldsymbol{v}$，那么

$$
\begin{aligned}
A&=P C P^{-1} \\
P&=\begin{bmatrix} \Re \boldsymbol{v} & \Im \boldsymbol{v}\end{bmatrix}\\
C &= \begin{bmatrix} a & -b \\
b & a
\end{bmatrix}
\end{aligned}
$$

Proof:

$$
\begin{aligned}
\boldsymbol{A} \boldsymbol{P} &= \boldsymbol{A} \begin{bmatrix} \Re  \boldsymbol{v} & \Im \boldsymbol{v} \end{bmatrix} \\
&= \begin{bmatrix} \boldsymbol{A} \Re \boldsymbol{v} & \boldsymbol{A} \Im \boldsymbol{v} \end{bmatrix} \\
&= \begin{bmatrix} \Re (\boldsymbol{A} \boldsymbol{v}) & \Im  (\boldsymbol{A} \boldsymbol{v}) \end{bmatrix} \\
&= \begin{bmatrix} \Re (\lambda \boldsymbol{v}) & \Im (\lambda \boldsymbol{v}) \end{bmatrix} \\
&= \begin{bmatrix} a \Re \boldsymbol{v} + b \Im \boldsymbol{v} & a \Im  \boldsymbol{v} - b \Re \boldsymbol{v} \end{bmatrix} \\
&= \begin{bmatrix} \Re \boldsymbol{v} & \Im \boldsymbol{v} \end{bmatrix} \begin{bmatrix} a & -b \\ b & a \end{bmatrix} \\
&= \boldsymbol{P} \boldsymbol{C}
\end{aligned}
$$

另外还要证明 $\boldsymbol{P}$ 可逆。

$C$ can be rewrite as:

$$
C = \begin{bmatrix}
r & 0 \\
0 & r
\end{bmatrix}
\begin{bmatrix}
\cos \varphi & -\sin \varphi \\
\sin \varphi & \cos \varphi
\end{bmatrix}
$$

其中矩阵 $C$ 可以表示为一个旋转的线性变换和伸缩的线性变换的组合。

### 5.6_离散动力系统

$$
\begin{aligned}
\boldsymbol{x}_{k} &= \boldsymbol{A} \boldsymbol{x}_{k-1} \\
\boldsymbol{x}_{0} &= \sum_{i=1}^{n} c_{i} \boldsymbol{v}_{i} \\
\boldsymbol{x}_{k} &= \sum_{i=1}^{n} c_{i} \lambda_{i}^{k} \boldsymbol{v}_{i}
\end{aligned}
$$

离散动力系统状态轨迹

- 吸引子
- 排斥子
- 鞍点

在线性系统中，上面这些点只能是原点。

### 5.7_微分方程中的应用

$$
\begin{aligned}
&\boldsymbol{x}' = \boldsymbol{A} \boldsymbol{x} \\
&\left\{ \begin{aligned}
x'_{1} &= a_{11}x_{1} + a_{12} x_{2} + \cdots + a_{1n} x_{n} \\
\cdots \\
x'_{n}&= a_{n1} x_{1} + a_{n2} x_{2} + \cdots + a_{nn} x_{n}
\end{aligned} \right.
\end{aligned}
$$

如果 $A$ 是对角矩阵，那么微分方程组是解耦的，可以直接求解。

$$
x_{i} = A_{i} e^{a_{11}t}
$$

如果 $A$ 可以相似对角化，那么可以通过将方程组解耦求解：

$$
\begin{aligned}
\boldsymbol{x}' &= \boldsymbol{P} \boldsymbol{\Lambda} \boldsymbol{P}^{-1} \boldsymbol{x} \\
\boldsymbol{P}^{-1} \boldsymbol{x}' &= \boldsymbol{\Lambda} \boldsymbol{P}^{-1} \boldsymbol{x} \\
\boldsymbol{x} &= \boldsymbol{P} \begin{bmatrix} c_{1} e^{\lambda_{1}t} & \cdots & c_{n}e^{\lambda_{n}t} \end{bmatrix}^{T} \\
&= c_{1} \boldsymbol{v}_{1} e^{\lambda_{1}t} + \cdots + c_{n} \boldsymbol{v}_{n} e^{\lambda_{n}t}
\end{aligned}
$$

$\boldsymbol{v}_{i} e^{\lambda_{i}t}$ 可以视为原系统解空间的基。

如何求 $\boldsymbol{c}$ 呢？带入初始条件。

$$
\begin{aligned}
\boldsymbol{x}_{0} &= \boldsymbol{P} \boldsymbol{c} \\
\boldsymbol{c} &= \boldsymbol{P}^{-1} \boldsymbol{x}_{0} \\
\begin{bmatrix} \boldsymbol{P} & \boldsymbol{x}_{0} \end{bmatrix} &\sim \begin{bmatrix}\boldsymbol{I} & \boldsymbol{c}\end{bmatrix}
\end{aligned}
$$

当存在复特征值时，可以仿照上面的方法硬求，也可以利用实数矩阵的特征值和特征向量互为共轭且解空间一定为实空间的特点，改变两组特征向量基：

$$
\boldsymbol{v} e^{\lambda t}, \overline{{\boldsymbol{v}}} e^{\overline{\lambda} t}
$$

变成

$$
\begin{aligned}
\Re (\boldsymbol{v} e^{ibt}) e^{at}, \Im \left( \boldsymbol{v}e^{ibt} \right) e^{at} && (\lambda = a + ib)
\end{aligned}
$$

求解之前首先将 $\boldsymbol{v} e^{ibt}$ 转换为 $a + ib$ 的形式，可以省去很多复数运算。

解的几何特征：

- 吸引子
- 排斥子
- 鞍点
- 螺旋吸引子、排斥子（复特征值）
- 椭圆（虚特征值）

### 5.8_特征值的迭代估计

#### 幂算法

只能求绝对值最大的**主特征值** $\lambda_{1}$。

$$
|\lambda_{1}| > |\lambda_{2}| \geq \cdots
$$

```python
def main_eigval(A:np.matrix, iter:int=100)->float:
    assert A.shape[0] == A.shape[1]
    x = np.matrix(np.random.random(A.shape[0])).reshape(-1, 1)
    for _ in range(iter):
        x = A * x
        l = np.max(np.abs(x))
        x = x / l
    return l
```

上面求出的是绝对值。

#### 逆算法

本质上还是幂算法，但是可以找到任意特征值：找 $(\boldsymbol{A} - \alpha \boldsymbol{I})^{-1}$ 的特征值 $\frac{1}{\lambda-\alpha}$，如果初始选择的 $\alpha$ 和任意一个特征值比较接近，就可以通过幂算法找 $(\boldsymbol{A} - \alpha \boldsymbol{I})^{-1}$ 的主特征值 $\mu = \frac{1}{\lambda-\alpha}$ 来反求 $\lambda=\alpha + \frac{1}{\mu}$.

更通用的方法是 [QR 算法](#求特征值的QR算法)。

## 六、正交性和最小二乘法

### 6.1_内积长度和正交性

$$
\vec{u} \cdot \vec{v} = \vec{u}^T \vec{v}
$$

#### 定理一：内积的线性

1. $\vec{u} \cdot \vec{v} = \vec{v} \cdot \vec{u}$
2. $(u + v) \cdot w = u \cdot w + v \cdot w$
3. $(c \vec{u}) \cdot \vec{v} = c(\vec{u} \cdot \vec{v})$
4. $\vec{u} \cdot \vec{u} \geq 0, \vec{u} \cdot \vec{u} = 0 \iff \vec{u}=\vec{0}$

范数：

$$
||\vec{v}|| = \sqrt{\vec{v} \cdot \vec{v}}
$$

向量的单位化：

$$
\vec{e}_v = \frac{1}{||\vec{v}||} \vec{v}
$$

#### 定理二：毕达哥拉斯定理

正交补：如果空间 $R^n$ 中的向量 $\vec{z}$ 与子空间 $W$ 中的任意向量正交，则称 $z$ 正交于 $W$。全体向量 $\vec{z}$ 的集合称为子空间 $W$ 的正交补。

$$
\{\vec{z}\} = W^{\bot}
$$

#### 定理三

$$
\begin{aligned}
\mathrm{Nul} A &= (\mathrm{Row} A)^{\bot} \\
\mathrm{Nul} A^{T} &= (\mathrm{Col} A)^{\bot}
\end{aligned}
$$

PROOF:
$$
\begin{aligned}
&\forall \vec{x} \in \mathrm{Nul} A, A \vec{x} = \vec{0} \\
&\vec{x} \cdot (A^T \vec{y}) = (A \vec{x})^T \vec{y} = 0 \\
&\forall \vec{y} \in R^m, A^T \vec{y} \in \mathrm{Row} A \\
&\mathrm{Nul} A = (\mathrm{Row}A)^{\bot}
\end{aligned}
$$

### 6.2_正交集

#### 定理四：一组正交向量集合必然线性无关

方法一：

$$
\begin{align}
\sum_{i=1}^p c_i \vec{a}_i &= \vec{0} \\
\vec{a}_k \cdot \sum_{i=1}^p c_i \vec{a}_i &= 0 \\
c_k ||\vec{v}_k||^2 & = 0 \\
c_k &= 0
\end{align}
$$

方法二：

$$
\begin{align}
A &= [\vec{a}_1 \cdots \vec{a}_p] \\
B &= A^T A = [||a_1||^2 \vec{e}_1, \cdots, ||a_p||^2 \vec{e}_p] \\
\mathrm{rank} B &= p \\
\mathrm{rank} A &= p
\end{align}
$$

证明完毕。

#### 定理五：正交基的优越性

对于子空间 $W$ 的正交基 $\{\vec{u}_1, \cdots, \vec{u}_p\}$

$$
\forall \vec{y} \in W, \vec{y} = \sum_{i=1}^p \frac{\vec{u}_i \cdot \vec{y}}{\vec{u}_i \cdot \vec{u}_i} \vec{u}_i
$$

**正交矩阵**：列向量是单位正交集

$$
A^T A = E \text{ or } A^T = A^{-1}
$$

PROPERTIES:

- 正交矩阵的转置矩阵、逆矩阵也是正交矩阵。
- 正交矩阵，等价于，矩阵的列向量的模为 1 且两两正交。
- 正交矩阵的积也是正交矩阵。$(PQ)^T PQ = Q^T P^T P Q = E$

任何一个三维空间中的旋转都可以用一个唯一的正交矩阵表示。

#### 定理六：正交列矩阵的性质

一个 $m \times n$ 的矩阵具有单位正交列向量的等价条件是 $U^T U = I$.

PROOF:
$$
U^T U = \begin{bmatrix}
u_1^T \\
u_2^T \\
\vdots \\
u_n^T
\end{bmatrix}
\begin{bmatrix}
u_1 & u_2 & \cdots & u_n
\end{bmatrix} = \begin{bmatrix}
u_1^T \cdot u_1 & u_1^T \cdot u_2 & \cdots & u_1^T \cdot u_n \\
u_2^T \cdot u_1 & u_2^T \cdot u_2 & \cdots & u_2^T \cdot u_n \\
\vdots & \vdots & \ddots & \vdots \\
u_n^T \cdot u_1 & u_n^T \cdot u_2 & \cdots & u_n^T \cdot u_n
\end{bmatrix} = I
$$

**正交变换**：正交矩阵对应的线性变换

#### 定理七：正交列矩阵对应变换的性质

对于列向量正交且模均为 1 的 $m \times n$ 矩阵 $U$，$\forall \vec{x}, \vec{y} \in R^n$

- $\Vert U \vec{x} \Vert = \Vert \vec{x} \Vert$
- $(U \vec{x}) \cdot (U \vec{y}) = \vec{x} \cdot \vec{y}$
- $(U \vec{x}) \cdot (U \vec{y}) = 0 \iff \vec{x} \cdot \vec{y} = 0$

正交变换不改变向量的内积，因此也不改变向量的长度和向量之间的夹角。

PROOF:

$$
(Q \vec{x}) \cdot (Q \vec{y}) = (Q \vec{x})^T (Q \vec{y}) = \vec{x}^T Q^T Q \vec{y} = \vec{x} \cdot \vec{y}
$$

### 6.3_正交投影

#### 定理八：正交分解定理

如果 $W$ 是 $R^n$ 中的子空间，$\boldsymbol{W}^{\bot}$ 为所有与 $\boldsymbol{W}$ 中任意向量正交的向量的集合。

$$
\forall \vec{y} \in R^n, \exists \vec{\hat{y}} \in W, \exists \vec{z} \in W^{\bot}, \vec{y} = \vec{\hat{y}} + \vec{z}
$$

#### 定理九和十：最佳逼近定理

$$
\forall v \neq \hat{y} \in W, \Vert v - y \Vert > \Vert \hat{y} - y \Vert
$$

如果：$W = \mathrm{Span} \{u_1, u_2, \cdots, u_p\}$

$$
\hat{y} = \sum_{i=1}^{p} \frac{y \cdot u_i}{u_i \cdot u_i} u_i
$$

对所有的基单位化，$\Vert \boldsymbol{u}_{i} \Vert = 1, \boldsymbol{U} = \begin{bmatrix} \boldsymbol{u}_{i} \end{bmatrix}$

$$
\hat{\boldsymbol{y}} = \sum_{i=1}^{p} (\boldsymbol{y} \cdot \boldsymbol{u}_{i}) \boldsymbol{u}_{i} = \sum_{i=1}^{p} \boldsymbol{u}_{i} \boldsymbol{u}_{i}^{T} \boldsymbol{y} = \boldsymbol{U} \boldsymbol{U}^{T} \boldsymbol{y}
$$

注意到当 $\boldsymbol{U}$ 为方阵时，$\boldsymbol{\hat{y}} = \boldsymbol{y}$

### 6.4_格拉姆——施密特方法

#### 定理十一：格拉姆——施密特方法得到的基是子空间的正交基

**施密特正交化**：已知向量集合 $A$，求正交向量集合 $B$ 与原来的向量集合等价。

方法：加强条件，集合 $B$ 中的任意前 k 个向量与 $A$ 中前 k 个向量等价。

- $b_1 = a_1$
- $b_2 = a_2 + c_{21} b_1$
- $b_3 = a_3 + c_{31} b_1 + c_{32} b_2$
- ……

这样的取法使得向量集合 $A$ 与 $B$ 等价。下面解决正交的问题：

利用数学归纳法的思路，对于前 k-1 个向量，假设已经设置好都是两两正交的，现在添加第 k 个向量：

$$
\begin{align}
\vec{b}_i \cdot\vec{b}_k &= \vec{b}_i \cdot \vec{a}_k + c_{k,i} ||\vec{b}_i||^2 \\
c_{k,i} &= - \frac{\vec{a}_k \cdot \vec{b}_i}{||\vec{b}_i||^2} \\
\vec{b}_k &= \vec{a}_k - \sum_{i=1}^{k-1} \frac{\vec{a}_k \cdot \vec{b}_i}{||\vec{b}_i||^2} \vec{b}_i
\end{align}
$$

这样通过递归完成了施密特正交化。

#### 定理十二：QR decomposition

CONDITION: $m \times n$ matrix $A$ has linearly independent column vectors

CONCLUSION:

$$
A = QR
$$

- $Q$ is an $m \times n$ matrix whose orthonormal column vectors span $\mathrm{Col} A$
- $R$ is an invertible upper triangle matrix with positive entries on its diagonal

算法：

1. 通过格拉姆——施密特方法计算线性无关列构成的矩阵 $A$ 对应的单位正交列向量构成的矩阵 $Q$
2. 注意到 $Q Q^T = I \Rightarrow A = Q Q^T A \Rightarrow R = Q^T A$，根据这一点计算上三角可逆矩阵 $R$.

### 6.5_最小二乘问题

$$
\begin{aligned}
& \text{min} && \Vert \boldsymbol{A} \boldsymbol{x} - \boldsymbol{b} \Vert \\
& \text{s.t.} && \boldsymbol{x} \in \mathbb{R}^{n}
\end{aligned}
$$

根据勾股定理，等价于使得 $\boldsymbol{b} - \boldsymbol{A} \boldsymbol{x} \in \mathrm{Col}\boldsymbol{A}^{\bot}$

$$
\begin{aligned}
0 &= \boldsymbol{A}^{T} (\boldsymbol{b} - \boldsymbol{A}\boldsymbol{x}) \\
\boldsymbol{A}^{T} \boldsymbol{A} \boldsymbol{x} &= \boldsymbol{A}^{T} \boldsymbol{b}
\end{aligned}
$$

最后一个方程被称为法方程。

#### 定理十三：求解最小二乘解集等价于求解法方程非空解集

可以通过正交分解的唯一性证明。

#### 定理十四：最小二乘解唯一的充要条件

$\boldsymbol{A}$ 列满秩 $\iff$ $\boldsymbol{A}^{T} \boldsymbol{A}$ 可逆 $\iff$ 最小二乘解唯一

#### 定理十五：QR分解求解最小二乘问题

当然也可以使用 QR 分解来求解，这种方法得到的数值解更稳定。这种方法必须满足 $\boldsymbol{A}$ 列满秩。

$$
\begin{aligned}
\boldsymbol{x} &= (\boldsymbol{A}^{T} \boldsymbol{A})^{-1} \boldsymbol{A}^{T} \boldsymbol{b} \\
&= (\boldsymbol{R}^{T} \boldsymbol{R})^{-1} \boldsymbol{R}^{T} \boldsymbol{Q}^{T} \boldsymbol{b} \\
&= \boldsymbol{R}^{-1} \boldsymbol{Q}^{T} \boldsymbol{b}
\end{aligned}
$$

### 6.6_线性模型中的应用

- 最小二乘直线
- 一般线性模型
- 其他曲线的最小二乘拟合
- 多重回归

### 6.7_内积空间

作为范数的基础，可以定义内积

向量空间 $V$，$\boldsymbol{u}, \boldsymbol{v}, \boldsymbol{w} \in V, \left<\boldsymbol{u}, \boldsymbol{v}\right> \in \mathbb{R}$

- 交换性 $\left< \boldsymbol{u}, \boldsymbol{v} \right> = \left< \boldsymbol{v}, \boldsymbol{u} \right>$
- 分配律 $\left< \boldsymbol{u} + \boldsymbol{v}, \boldsymbol{w} \right> = \left< \boldsymbol{u}, \boldsymbol{w} \right> + \left< \boldsymbol{v}, \boldsymbol{w} \right>$
- 结合律 $\left< c \boldsymbol{u}, \boldsymbol{v} \right> = c \left< \boldsymbol{u}, \boldsymbol{v} \right>$
- 正定性 $\left< \boldsymbol{u}, \boldsymbol{u} \right> \geq 0$ 当且仅当 $\boldsymbol{u} = \boldsymbol{0}$ 时取等

在内积的基础上，定义长度、距离和正交性。

$$
\Vert \boldsymbol{v} \Vert = \sqrt{\left< \boldsymbol{v}, \boldsymbol{v} \right>}
$$

多项式内积空间：用 $\mathbb{P}_{n}$ 表示最高次方不超过 $n$ 的多项式。给定实数序列 $\left\{ t_{0}, \cdots, t_{n} \right\}$，任意两个实数不相等。定义两个多项式的内积是：

$$
\left< p, q \right> = \sum_{i=0}^{n} p(t_{i})q(t_{i})
$$

这个定义满足内积的四个条件。

如果仅仅使用有限次幂的多项式来拟合一段范围（包含上述给定实数序列）的未知函数 $p$（但是已知实数序列上的取值）。可以使用投影的方法：先找到有限次幂多项式空间中的正交基 $\begin{bmatrix} p_{i} \end{bmatrix}$（维数等于最高次幂），然后利用上面的内积定义和下面的投影公式获得对未知函数的最佳多项式逼近 $\hat{p}$。

$$
\hat{p} = \sum_{i=1}^{n} \frac{\left< p, p_{i} \right>}{\left< p_{i}, p_{i} \right>} p_{i}
$$

#### 定理十六：柯西施瓦茨不等式

$$
\forall \boldsymbol{u}, \boldsymbol{v} \in V, | \left< \boldsymbol{u}, \boldsymbol{v} \right> | \leq \Vert \boldsymbol{u} \Vert \cdot \Vert \boldsymbol{v} \Vert
$$

证明利用了投影小于原长。

$$
\Vert \mathrm{proj}_{\boldsymbol{u}} \boldsymbol{v} \Vert \leq \Vert \boldsymbol{v} \Vert
$$

#### 定理十七：三角不等式

$$
\forall \boldsymbol{u}, \boldsymbol{v} \in V, \Vert \boldsymbol{u} + \boldsymbol{v} \Vert \leq \Vert \boldsymbol{u} \Vert + \Vert \boldsymbol{v} \Vert
$$

对多项式内积的扩展：积分内积。

想象将多项式内积中的给定实数序列无限增加，间隔无限减小，如果是等间隔，则转化为：

$$
\begin{aligned}
\left< p, q \right> &= \frac{1}{n+1} \sum_{i=0}^{n} p(t_{i}) q(t_{i}) \\
& \approx \int_{t_{0}}^{t_{n}} p(t) q(t) \mathrm{d} t
\end{aligned}
$$

这个定义很容易让人想到[傅里叶级数与变换](SS1.md#3.2)，将函数转化为傅里叶级数的过程实际上就是一种正交分解。

### 6.8_内积空间的应用

#### 加权最小二乘法

经典的最小二乘法：

$$
SSE = \Vert \boldsymbol{y} - \boldsymbol{\hat{y}} \Vert
$$

加权最小二乘法：

$$
SSE = \sum_{i=1}^{n} w_{i}^{2} (y_{i} - \hat{y}_{i})^{2}
$$

比如说用一个模型去逼近测量的结果，如果每次测量的环境和手段不同，可能测量结果的方差不同。这时就可以取 $w_{i} = \dfrac{1}{\sigma_{i}}$.

可以定义新的内积：

$$
\begin{aligned}
\left< \boldsymbol{x}, \boldsymbol{y} \right> &= \sum_{i=1}^{n} w_{i}^{2} x_{i} y_{i} = (\boldsymbol{W} \boldsymbol{x})^{T} (\boldsymbol{W} \boldsymbol{y}) \\
\boldsymbol{W} &= \mathrm{diag} \begin{bmatrix} w_{1} & \cdots & w_{n} \end{bmatrix}
\end{aligned}
$$

转化成了经典最小二乘问题

$$
\begin{aligned}
\boldsymbol{W}\boldsymbol{A}\boldsymbol{x} &= \boldsymbol{W} \boldsymbol{y} \\
\boldsymbol{x} &= (\boldsymbol{A}^{T} \boldsymbol{W}^{T} \boldsymbol{W} \boldsymbol{A})^{-1} \boldsymbol{A}^{T} \boldsymbol{W}^{T} \boldsymbol{W} \boldsymbol{y}
\end{aligned}
$$

#### 数据趋势分析

用有限次幂的多项式拟合一定范围内的未知函数（已知函数在离散点上的取值）。

#### 傅里叶级数

傅里叶正交基：$\left\{ 1, \cos t, \cos 2t, \cdots, \cos nt, \sin t, \sin 2t, \cdots, \sin nt \right\}$

在区间 $[-\pi, \pi]$ 上的积分内积满足正交条件。

单位正交基：$\left\{ \frac{1}{\sqrt{2\pi}}, \frac{1}{\sqrt{\pi}} \cos t, \cdots, \frac{1}{\sqrt{\pi}} \sin nt \right\}$

## 七、二次型

### 7.1_对称矩阵的对角化

#### 定理一：对称矩阵的特征向量

CONDITION: $A$ is a symmetric matrix.

CONCLUSION:

$$
\lambda_{i} \neq \lambda_{j} \Rightarrow \vec{v}_{i} \cdot \vec{v}_{j} = 0
$$

PROOF:

$$
\begin{aligned}
x^{T} A y &= x^{T} \lambda_{2} y = \lambda_{2} (x \cdot y) \\
x^{T} A y &= x^{T} A^{T} y = (A x)^{T} y = (\lambda_{1} x)^{T} y = \lambda_{1} (x \cdot y) \\
(\lambda_{1} - \lambda_{2}) (x \cdot y) &= 0 \\
x \cdot y &= 0
\end{aligned}
$$

实对称阵只有实数特征值。

PROOF:

共轭矩阵的共轭特征值：

$$
A \overline{x} = \overline{A} \overline{x} = \overline{\lambda} \overline{x}
$$

如果 $\lambda \neq \overline{\lambda}$，则根据上面的定理：

$$
\begin{aligned}
&\boldsymbol{x} \cdot \overline{\boldsymbol{x}} = 0 \\
&\sum ||x_i||^2 = 0
\end{aligned}
$$

和特征向量不能为 0 的定义矛盾。因此，

$$
\lambda = \overline{\lambda} \in R
$$

#### 定理二与三：对称阵与正交相似

实对称阵 $\iff$ 正交相似于对角阵。（正交相似：过渡矩阵是正交阵）

必要性易于证明。充分性证明用到了舒尔因子分解。

衍生结论：

- 实对称阵必然可以对角化，有 $n$ 个包含重根的特征值。
- 实对称阵的任意特征值的几何重数和代数重数相等。
- 对不相等的特征值，对应的特征空间正交。

谱分解：限定于实对称阵。

$$
\begin{aligned}
\boldsymbol{A} &= \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{T} \\
&= \begin{bmatrix} \boldsymbol{q}_{1} & \cdots & \boldsymbol{q}_{n} \end{bmatrix} \boldsymbol{\Lambda} \begin{bmatrix}
\boldsymbol{q}_{1}^{T} \\
\vdots \\
\boldsymbol{q}_{n}^{T}
\end{bmatrix} \\
&= \sum_{i=1}^{n} \lambda_{i} \boldsymbol{u}_{i} \boldsymbol{u}_{i}^{T}
\end{aligned}
$$

谱分解将一个实对称阵分解为它的谱（特征值）对应的小块。

另一方面，对于 $\boldsymbol{x} \in \mathbb{R}^{n}$

$$
\begin{aligned}
\boldsymbol{A} \boldsymbol{x} &= \sum_{i=1}^{n} \lambda_{i} \boldsymbol{u}_{i} \boldsymbol{u}_{i}^{T} \boldsymbol{x} \\
&= \sum_{i=1}^{n} \lambda_{i} \left( \frac{\boldsymbol{u}_{i} \boldsymbol{x}}{\Vert \boldsymbol{u}_{i} \Vert} \right) \boldsymbol{u}_{i}
\end{aligned}
$$

将一个线性变换转换成了若干个向特征向量投影然后再加权求和。

当然也有不限定于实对称阵的谱分解，但是就不再是投影加权求和了。

### 7.2_二次型

二次型的一般形式：

$$
f(\vec{x}) = \sum_{i=1}^n a_{ii}x_i^2 + \sum_{i<j} 2 a_{ij} x_i x_j
$$

矩阵表示

$$
\begin{aligned}
f(\vec{x}) &= \vec{x}^T A \vec{x} \\
A &= \begin{bmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{bmatrix}
\end{aligned}
$$

其中 $A$ 是对称矩阵，即 $a_{ij} = a_{ji} \ (i \neq j)$.

一种描述

$$
\begin{aligned}
f(\vec{x}) = \begin{bmatrix} 1 & \cdots & 1 \end{bmatrix}
\begin{bmatrix}
x_1 & 0 & \cdots & 0 \\
0 & x_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & x_n
\end{bmatrix}
\begin{bmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{bmatrix}
\begin{bmatrix}
x_1 & 0 & \cdots & 0 \\
0 & x_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & x_n
\end{bmatrix}
\begin{bmatrix}
1 \\
1 \\
\vdots \\
1
\end{bmatrix}
\end{aligned}
$$

将左右两个对角矩阵视为初等矩阵，分别对 $\boldsymbol{A}$ 进行行倍乘和列倍乘。最左最右的 1 向量对倍乘后的矩阵进行逐元素求和。

**合同关系**：如果 $C$ 是可逆（正交）矩阵，称之为 $A$ 和 $B$ （正交）合同。

$$
B = C^T A C
$$

对矩阵 $A$ 做行列同步变换，得到的矩阵是合同矩阵。

$$
B = EAE^T
$$

其中 $E$ 是对应的初等矩阵。

合同变换本质上是一种变量代换，对于原来的二次型

$$
\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}
$$

使用可逆矩阵 $\boldsymbol{P}, \boldsymbol{x} = \boldsymbol{P} \boldsymbol{y}$ 作变量代换，那么原来的二次型就变成：

$$
\boldsymbol{y}^{T} \boldsymbol{P}^{T} \boldsymbol{A} \boldsymbol{P} \boldsymbol{y}
$$

从几何的角度，对于 $\boldsymbol{x} \in \mathbb{R}^{2}$，二次型表征了平面二次曲线簇，使用可逆矩阵去变化一个二次型等效于采用不同的基去看待同一个平面二次曲线簇。特别地，当可逆矩阵还是一个单位正交矩阵时，在新的基下看原来的二次曲线，只经过了旋转操作。

- **标准二次型**：只含有平方项，标准矩阵是对角矩阵的二次型。
- **规范二次型**：标准二次型矩阵的元素仅有 1，-1，0.

#### 定理四：主轴定理

$\boldsymbol{A} \in \mathbb{R}^{n \times n}, \exists \boldsymbol{P} \in \mathbb{R}^{n \times n}: \boldsymbol{x} = \boldsymbol{P} y$，将二次型 $\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}$ 转换成不含交叉项的 $\boldsymbol{y}^{T} \boldsymbol{D} \boldsymbol{y}$.

将二次型标准化：

- 写出二次型对应的对称矩阵 $A$
- 将矩阵相似对角化 $A = Q D Q^T$，过渡矩阵 $Q$ 是规范正交矩阵。
- 写出对应的正交变换 $x = Q y$ 和新的标准型 $f(y)$

在 2 维情况下，主轴表示椭圆或双曲线的长轴或短轴。

#### 惯性定理

基于坐标变换的方法，总可以将任意的二次型变成规范二次型。

$$
\begin{aligned}
f(\boldsymbol{x}) &= \boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x} \\
&= z_{1}^{2} + \cdots + z_{p}^{2} - z_{p+1}^{2} - \cdots - z_{r}^{2} && (r = \mathrm{rank} \boldsymbol{A})
\end{aligned}
$$

这个定理背后的几何原则是：坐标变换不会改变二次型代表的曲线的形状。例如对于二维情形，不可能存在一种坐标变换将椭圆变成双曲线。

**正定矩阵和正定二次型**：

$$
\forall \vec{x} \neq 0 \in R^n, f(\vec{x}) = \vec{x}^T A \vec{x} > 0
$$

负定二次型同理。

对矩阵进行合同变换不改变矩阵的正定性。

#### 定理五：正定性与特征值

对于实对称阵 $A$，下面的命题等价：

- $A$ 为正定矩阵
- $A$ 的所有特征值都是正值
- $A = C^T C$，其中 $C$ 是可逆矩阵

负定二次型同理。

**Chelosky Decomposition**

对于正定阵 $\boldsymbol{A}$，可以分解为两个上三角矩阵的乘积。

基于定理五的第三点，再利用 QR 分解。

$$
\boldsymbol{A} = \boldsymbol{C}^{T} \boldsymbol{C} = (\boldsymbol{Q} \boldsymbol{R})^{T} (\boldsymbol{Q} \boldsymbol{R}) = \boldsymbol{R}^{T} \boldsymbol{R}
$$

#### 霍尔维茨定理

对称矩阵 $A$ 是正定矩阵当且仅当 $A$ 的 $n$ 个顺序主子式全大于 0.

顺序主子式：

$$
\det{\Delta_k} = \det \begin{bmatrix}
a_{11} & \cdots & a_{1k} \\
\vdots & \ddots & \vdots \\
a_{k1} & \cdots & a_{kk}
\end{bmatrix}
$$

推论：对称矩阵 $A$ 是负定矩阵当且仅当 $A$ 的 $n$ 个顺序主子式中，奇数阶均为负值，偶数阶均为正值。

### 7.3_条件优化

Rayleigh 商

$$
\frac{\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{T} \boldsymbol{x}}
$$

#### 定理六：二次型取值范围

$$
\lambda_{min} \leq \frac{\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{T} \boldsymbol{x}} \leq \lambda_{max}
$$

证明：正交相似对角化

$$
\begin{aligned}
\frac{\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{T} \boldsymbol{x}} &= \frac{\boldsymbol{x}^{T} \boldsymbol{Q}^{T} \boldsymbol{\Lambda} \boldsymbol{Q} \boldsymbol{x}}{\boldsymbol{x}^{T} \boldsymbol{Q}^{T} \boldsymbol{Q} \boldsymbol{x}} \\
&= \frac{\boldsymbol{y}^{T} \boldsymbol{\Lambda} \boldsymbol{y}}{\boldsymbol{y}^{T} \boldsymbol{y}} && \boldsymbol{y} = \boldsymbol{Q} \boldsymbol{x}
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol{y}^{T} \boldsymbol{\Lambda} \boldsymbol{y} &= \sum_{i=1}^{n} \lambda_{i} y_{i}^{2}\\
\boldsymbol{y}^{T} \boldsymbol{\Lambda} \boldsymbol{y} &\leq \lambda_{max} \sum_{i=1}^{n} y_{i}^{2} = \lambda_{max} \boldsymbol{y}^{T} \boldsymbol{y}\\
\boldsymbol{y}^{T} \boldsymbol{\Lambda} \boldsymbol{y} &\geq \lambda_{min} \sum_{i=1}^{n} y_{i}^{2} = \lambda_{max} \boldsymbol{y}^{T} \boldsymbol{y}
\end{aligned}
$$

取等时，$\boldsymbol{y}$ 为对应的单位向量，$\boldsymbol{x}$ 为对应的特征向量。

#### 定理七和八：特征向量正交空间中二次型最大值

Condition:

$$
\boldsymbol{A} = \boldsymbol{A}^{T} = \boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^{-1}
$$

对 $\boldsymbol{\Lambda}$ 进行重排，$\lambda_{1} \geq \lambda_{2} \geq \cdots \geq \lambda_{n}$，$\boldsymbol{Q} = \begin{bmatrix} \boldsymbol{u}_{1} & \cdots & \boldsymbol{u}_{n} \end{bmatrix}$

Conclusion:

$$
\begin{aligned}
& \text{min} && \frac{\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{T} \boldsymbol{x}} \\
& \text{s.t.} && \boldsymbol{x}^{T} \begin{bmatrix}\boldsymbol{u}_{1} & \cdots & \boldsymbol{u}_{k}\end{bmatrix} = \boldsymbol{0}^{T}
\end{aligned}
$$

上述优化问题的解为：$\boldsymbol{x}^{*} = \boldsymbol{u}_{k+1}, \dfrac{\boldsymbol{x}^{T} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{T} \boldsymbol{x}} = \lambda_{k+1}$.

Proof:

$$
\begin{aligned}
\boldsymbol{x} &= \boldsymbol{Q} \boldsymbol{y} \\
& \text{min} && \frac{\boldsymbol{y}^{T} \boldsymbol{\Lambda} \boldsymbol{y}}{\boldsymbol{y}^{T} \boldsymbol{y}} \\
& \text{s.t.} && \boldsymbol{y}^{T} \boldsymbol{Q}^{T} \begin{bmatrix}\boldsymbol{u}_{1} & \cdots & \boldsymbol{u}_{k}\end{bmatrix} = \boldsymbol{0}^{T} \\
\end{aligned}
$$

$$
\begin{aligned}
& \boldsymbol{Q}^{T} \begin{bmatrix} \boldsymbol{u}_{1} & \cdots & \boldsymbol{u}_{k}\end{bmatrix} = \begin{bmatrix} \boldsymbol{I}_{k} \\ \boldsymbol{O}\end{bmatrix} \\
& y_{1} = \cdots = y_{k} = 0 \\
& \frac{\boldsymbol{y}^{T} \boldsymbol{\Lambda} \boldsymbol{y}}{\boldsymbol{y}^{T} \boldsymbol{y}} = \frac{\sum_{i=k+1}^{n} \lambda_{i} y_{i}^{2}}{\sum_{i=k+1 }^{n} y_{i}^{2}} \leq \lambda_{k+1}
\end{aligned}
$$

### 7.4_奇异值分解

$\boldsymbol{A} \in \mathbb{R}^{m \times n}$，考察线性变换 $\boldsymbol{A}$ 对 $\mathbb{R}^{n}$ 空间中的单位超球的变换。求变换后 $\Vert \boldsymbol{A} \boldsymbol{x} \Vert$ 的最大值。

$$
\begin{aligned}
\Vert \boldsymbol{A} \boldsymbol{x} \Vert^{2} &= \boldsymbol{x}^{T} (\boldsymbol{A}^{T} \boldsymbol{A}) \boldsymbol{x} \\
&\leq \lambda_{max}(\boldsymbol{A}^{T} \boldsymbol{A}) \Vert \boldsymbol{x} \Vert^{2}
\end{aligned}
$$

设对称矩阵 $\boldsymbol{A}^{T} \boldsymbol{A}$ 的递减特征值为 $\lambda_{1} \geq \cdots \geq \lambda_{n}$，则定义 $\boldsymbol{A}$ 的**奇异值** $\sigma_{i} = \sqrt{\lambda_{i}}$，奇异值的几何意义是：当 $\boldsymbol{x} = \boldsymbol{u}_{i}$（$\boldsymbol{A}^{T} \boldsymbol{A}$ 的对应单位特征向量）时，$\Vert \boldsymbol{A} \boldsymbol{x} \Vert = \sigma_{i}$.

上面阐述了 $\boldsymbol{A}\boldsymbol{u}_{i}$ 的长度关系，定理九则阐述了 $\boldsymbol{A} \boldsymbol{u}_{i}$ 之间的几何关系：正交。

#### 定理九：Gram 矩阵特征向量变换后正交

Condition:

- Gram Matrix $\boldsymbol{A}^{T} \boldsymbol{A}$ 的标准正交基（按照特征值从大到小排序）$\left\{\boldsymbol{u}_{1}, \cdots, \boldsymbol{u}_{n}\right\}$ 
- $\boldsymbol{A}$ 有 $r$ 个非零奇异值

Conclusion:

- $\mathrm{Col}\boldsymbol{A}$ 的标准正交基 $\left\{ \frac{1}{\sigma_{1}} \boldsymbol{A} \boldsymbol{u}_{1}, \cdots, \frac{1}{\sigma_{r}}\boldsymbol{A} \boldsymbol{u}_{r} \right\}$，$\mathrm{rank}\boldsymbol{A} = r$

Proof:

首先证明正交：

$$
\begin{aligned}
\left< \boldsymbol{A} \boldsymbol{u}_{i}, \boldsymbol{A} \boldsymbol{u}_{j} \right> &= \boldsymbol{u}_{i}^{T} \boldsymbol{A}^{T} \boldsymbol{A} \boldsymbol{u_{j}} \\
&= \lambda_{i} \boldsymbol{u}_{i}^{T} \boldsymbol{u}_{j} \\
&= \lambda_{j} \boldsymbol{u}_{i}^{T} \boldsymbol{u}_{j} \\
&= 0
\end{aligned}
$$

利用了对称阵特征向量正交。

再证明维度：

$$
\begin{aligned}
\boldsymbol{x} &= c_{1} \boldsymbol{u}_{1} + \cdots + c_{n} \boldsymbol{u}_{n} \\
\boldsymbol{A} \boldsymbol{x} &= c_{1} \boldsymbol{A} \boldsymbol{u}_{1} + \cdots + c_{r} \boldsymbol{A} \boldsymbol{u}_{r} + c_{r+1} \boldsymbol{A} \boldsymbol{u}_{r+1} + \cdots + c_{n} \boldsymbol{A} \boldsymbol{u}_{n} \\
\Vert \boldsymbol{A} \boldsymbol{u}_{k} \Vert &= \sigma_{k} = 0 (k \geq r) \\
\boldsymbol{A} \boldsymbol{x} &= \begin{bmatrix}\boldsymbol{A} \boldsymbol{u}_{1} & \cdots & \boldsymbol{A} \boldsymbol{u}_{r}\end{bmatrix} \boldsymbol{c}
\end{aligned}
$$

因此列空间的基就是 $\boldsymbol{A} \boldsymbol{u}_{i}(i \leq r)$.

#### 定理十：奇异值分解

$$
\begin{aligned}
\boldsymbol{A} &= \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{T} \\
\boldsymbol{U} &\in \mathbb{R}^{m \times m}, \boldsymbol{U}^{T} = \boldsymbol{U}^{-1} \\
\boldsymbol{V} &\in \mathbb{R}^{n \times n}, \boldsymbol{V}^{T} = \boldsymbol{V}^{-1} \\
\boldsymbol{\Sigma} &= \begin{bmatrix}
\boldsymbol{D} & \boldsymbol{O}_{r \times (n-r)} \\
\boldsymbol{O}_{(m-r) \times r} & \boldsymbol{O}_{(m-r)\times(n-r)}
\end{bmatrix} \in \mathbb{R}^{m \times n}
\end{aligned}
$$

首先对 $\boldsymbol{U}, \boldsymbol{V}, \boldsymbol{D}$ 中的元素进行说明：

- $\boldsymbol{v}_{i}$ 是 $\boldsymbol{R}^{n}$ 的一组标准正交基，其中前 $r$ 个是 $\boldsymbol{A}^{T} \boldsymbol{A}$ 的特征向量。
- $\boldsymbol{u}_{i}$ 是 $\mathbb{R}^{m}$ 的一组标准正交基，其中前 $r$ 个满足对应关系 $\boldsymbol{u}_{i} = \dfrac{1}{\sigma_{i}} \boldsymbol{A} \boldsymbol{v}_{i}$
- $\boldsymbol{D}$ 是由不为 0 的奇异值 $\left\{ \sigma_{1}, \cdots, \sigma_{r} \right\}$ 构成的对角矩阵。

$$
\begin{aligned}
\boldsymbol{A} \boldsymbol{V} &= \begin{bmatrix} \boldsymbol{A} \boldsymbol{v}_{1} & \cdots & \boldsymbol{A}\boldsymbol{v}_{r} &  \boldsymbol{0} & \cdots & \boldsymbol{0}\end{bmatrix} \\
&= \begin{bmatrix}\sigma_{1} \boldsymbol{u}_{1} & \cdots & \sigma_{r}\boldsymbol{u}_{r} & \boldsymbol{0} & \cdots & \boldsymbol{0}\end{bmatrix} \\
&= \begin{bmatrix}\boldsymbol{u}_{1} & \cdots & \boldsymbol{u}_{m}\end{bmatrix} \begin{bmatrix}
\boldsymbol{D} & \boldsymbol{O} \\
\boldsymbol{O} & \boldsymbol{O}
\end{bmatrix} \\
&= \boldsymbol{U} \boldsymbol{\Sigma}
\end{aligned}
$$

由于 $\boldsymbol{u}_{i}(i \leq r)$ 是 $\boldsymbol{A}$ 的列空间的基，因此与之正交的、一起张成 $\boldsymbol{R}^{m}$ 的 $\boldsymbol{u}_{i}(i > r)$ 是 $(\mathrm{Col}\boldsymbol{A})^{\wedge}=\mathrm{Nul} \boldsymbol{A}^{T}$ 的基。由对称性，$\boldsymbol{v}_{i}(i\leq r)$ 是 $\mathrm{Row}\boldsymbol{A}$ 的基，而与之正交的、一起张成 $\boldsymbol{R}^{n}$ 的 $\boldsymbol{v}_{i}(i>r)$ 是 $\mathrm{Nul}\boldsymbol{A}$ 的基。关于零空间基的结论在上面的第一行公式中也可以得到。

利用奇异值分解解线性方程组

$$
\begin{aligned}
\boldsymbol{A} \boldsymbol{x} &= \boldsymbol{b} \\
\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{T} \boldsymbol{x} &= \boldsymbol{b} \\
\boldsymbol{\Sigma} (\boldsymbol{V}^{T} \boldsymbol{x}) &= \boldsymbol{U}^{T} \boldsymbol{b} \\
\boldsymbol{\Sigma} \widetilde{\boldsymbol{x}} &= \widetilde{\boldsymbol{b}}
\end{aligned}
$$

正交变换具有长度不变性。因此数值求解的误差仅仅来源于 $\Sigma$ 中的元素。可以使用条件数来衡量线性变换 $\boldsymbol{A}$ 的变形程度。

$$
\text{Condition Number} = \frac{\sigma_{1}}{\sigma_{r}}
$$

奇异值分解的另一种形式：

$$
\boldsymbol{A} = \boldsymbol{U}_{r} \boldsymbol{D} \boldsymbol{V}_{r}^{T}
$$

用奇异值分解求矩阵的伪逆

$$
\begin{aligned}
\boldsymbol{A}^{\dagger} = \boldsymbol{V}_{r} \boldsymbol{D}^{-1} \boldsymbol{U}_{r}^{T}
\end{aligned}
$$

### 7.5_图像处理和统计学中的应用

观测矩阵：从系统中采样 $N$ 次，每次采集 $p$ 个维度的指标，构成 $\boldsymbol{X} \in \mathbb{R}^{p \times N}$.

均值

$$
\boldsymbol{m} = \boldsymbol{X} \cdot \boldsymbol{1}_{N \times 1}
$$

平均偏差形式

$$
\boldsymbol{B} = \begin{bmatrix} \boldsymbol{x}_{1} - \boldsymbol{m} & \cdots & \boldsymbol{x}_{N} - \boldsymbol{m} \end{bmatrix}
$$

协方差矩阵

$$
\boldsymbol{S} = \frac{1}{N-1} \boldsymbol{B} \boldsymbol{B}^{T} \geq 0
$$

协方差矩阵中对角线元素是某一个维度的方差，非对角线元素是两个维度之间的协方差，如果后者是 0. 则称两个维度无关。样本的总方差为 $\mathrm{tr} (\boldsymbol{S})$.

主成分分析

寻找正交矩阵 $\boldsymbol{P}: \boldsymbol{x} = \boldsymbol{P}\boldsymbol{y}$

$$
\begin{aligned}
\boldsymbol{D} &= \frac{1}{N-1} \boldsymbol{P} \boldsymbol{B}_{y} \boldsymbol{B}_{y}^{T} \boldsymbol{P}^{T} \\
&= \boldsymbol{P} \boldsymbol{S} \boldsymbol{P}^{T} \\
\boldsymbol{S} &= \boldsymbol{P}^{T} \boldsymbol{D} \boldsymbol{P}
\end{aligned}
$$

中间的 $\boldsymbol{D}$ 是正交矩阵。这个正交变换不改变方差，即 $\mathrm{tr} \boldsymbol{D} = \mathrm{tr} \boldsymbol{S}$

第一主成分：最大特征值对应的特征向量

## 八、向量空间几何学

### 8.1_仿射组合

**仿射组合**：

$$
\begin{aligned}
\boldsymbol{y} &= \sum_{i=1}^{n} c_{i} \boldsymbol{v}_{i} \\
1 &= \sum_{i=1}^{n} c_{i}
\end{aligned}
$$

整合成矩阵相乘：

$$
\begin{aligned}
\begin{bmatrix} \boldsymbol{y} \\ 1\end{bmatrix} = \begin{bmatrix}
\boldsymbol{v}_{1} & \cdots & \boldsymbol{v}_{n} \\
1 & \cdots & 1
\end{bmatrix} \boldsymbol{c}
\end{aligned}
$$

**仿射包**（仿射生成集）$\mathrm{aff} S$：集合 $S$ 中所有向量（点）的仿射组合的集合。

#### 定理一：仿射组合与平移

$\boldsymbol{y}$ 是 $\left\{ \boldsymbol{v}_{1}, \cdots, \boldsymbol{v}_{n} \right\}$ 的仿射组合 $\iff$ $\boldsymbol{y} - \boldsymbol{v}_{1}$ 是 $\left\{ \boldsymbol{v}_{2}, \cdots, \boldsymbol{v}_{n} \right\}$ 的仿射组合。

几何角度很好想到，证明也比较容易。

**仿射**的集合：$\forall \boldsymbol{q}, \boldsymbol{q} \in S, \forall t \in \mathbb{R}, t \boldsymbol{p} + (1-t) \boldsymbol{q} \in S$.

#### 定理二：仿射集合与仿射包

$S$ 是放射的 $\iff S = \mathrm{aff} S$

## 九、优化

### 9.1_博弈矩阵

零和博弈才有博弈矩阵。玩家 R 有 $m$ 种决策，玩家 C 有 $n$ 种决策，**Payoff Matrix** $\boldsymbol{A} \in \mathbb{R}^{m \times n}$，其中元素 $a_{ij}$ 表示了玩家 R 从玩家 C 那里得到的收益。

举例：

$$
\begin{bmatrix}
-1 & -1 & -1 \\ 
1 & -5 & -5 \\ 
1 & 5 & 10
\end{bmatrix}
$$

- R 的选择 $\mathrm{max}_{i} \left( \mathrm{min}_{j} a_{ij} \right)$
- C 的选择 $\mathrm{min}_{j} \left( \mathrm{max}_{i} a_{ij} \right)$

很像最小最大算法。
