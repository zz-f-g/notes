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
4. $\exist \vec{0} \in V, \forall u \in V, u + \vec{0} = u$
5. $\forall u \in V, \exist -u \in V, u + (-u) = 0, \forall v \neq -\vec{u} \in V, u + v \neq 0$
6. $\forall c \in R, \forall \vec{u} \in V, c \vec{u} \in V$
7. $c(\vec{u} + \vec{y}) = c \vec{u} + c \vec{v}$
8. $(c + d) \vec{u} = c \vec{u} + d \vec{u}$
9. $c(d \vec{u}) = (cd) \vec{u}$
10. $1 \vec{u} = \vec{u}$

***Attention***: The DIFFERENCE between the zero entry $\vec{0}$ and scalar 0.

证明一个变换是线性变换：

1. ==变换后的元素仍然在原来的空间中==
2. 加法封闭
3. 乘法封闭

举例：

- $\R^n$
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
- 注意：$\R^2$ 不是 $\R^3$ 的子空间，以为他们的元素不一样

#### 集合张成的子空间

可以表示为集合中向量的线性组合的向量的集合。
$$
H = \text{Span} \{ \vec{v_1}, \cdots, \vec{v_p} \}
$$

#### 定理一：向量张成的空间是子空间

### 4.2_零空间，列空间和线性变换

表示子空间的两种方式

- 齐次线性方程组的解集
  - 零空间：满足矩阵方程 $A \vec{x} = \vec{0}$ 的向量的集合。$\text{Nul} A = \{ x: x \in \R^n \text{ and } A \vec{x} = \vec{0} \}$
- 向量集合的线性组合
  - 列空间：给出矩阵 $A$ 的列向量张成的空间。$\text{Col} A = \text{Span} \{\vec{a_1}, \cdots, \vec{a_n} \}$

#### 定理二：零空间是子空间

#### 零空间的显式描述

在解线性方程组的过程中，将系数矩阵化为简化阶梯型。

某些情况下发现解集可以表示为线性组合，其中线性组合的权为自由变量的值。

两个注意点

- 通过上述方法求出的零空间中的向量自动是**线性无关**的。
- 向量的个数等于自由变量的个数。

#### 定理三：列空间是子空间

$$
\text{Col}A = \{ \vec{b}: \vec{b} = A \vec{x} \text{ and } \vec{x} \in \R^n \}
$$

### 4.3_线性无关集和基

#### 定理四：线性相关的充要条件

对于首个向量不为零向量的向量集合（顺序已经排列好），存在一个向量可以表示为前面的向量的线性组合。

第一章中有[类似的证明](#定理七：线性相关与线性组合)。

#### 子空间的基

张成给定子空间的一组线性无关的向量集合。

$\R^n$ 空间的**标准基**：单位矩阵 $I_n$ 的列构成的集合。

#### 定理五：生成基定理

CONDITION:

- $S = \{v_1, \cdots, v_p\} \\$
- $H = \text{Span} \{v_1, \cdots, v_p\}$
- $v_k$ is a linear conbination of other vectors in $S$.

CONCLUSION:

- $H = \text{Span} \{v_1, \cdots, v_p\}$ without $v_k$
- Some subset of $S$ can span $H$.

#### 定理六：列空间和主元列

矩阵 $A$ 的列空间的基是 $A$ 的主元列的集合。

The pivot columns of the matrix $A$ form a basis of $\text{Col} A$.

==矩阵的行等价矩阵的列空间和其本身的列空间并不一定相同。==

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
\text{dim} \R^n = n
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
- 列空间的维数等于主元列的数量

### 4.6_秩

矩阵中和秩相等的量：

- 主元数量
- 线性无关的列的最大数量（列空间的维数）
- 线性无关的行的最大数量

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

#### 定理十四：秩定理

对于 $m \times n$ 矩阵 $A$：
$$
\text{rank} A = \text{dim} \text{Col} A = \text{dim} \text{Row} A = n - \text{dim} \text{Nul} A
$$
PROOF：

主元列的数量 + 非主元列的数量（自由变量的数量）= 列的数量

后续会证明：行空间和零空间垂直

#### 秩和逆矩阵定理

下面的命题等价：（对于 n 阶矩阵 $A$ ）

- 矩阵 $A$ 可逆
- $A$ 的列是 $\R^n$ 的一组基
- $A$ 的列空间是 $\R^n$
- $A$ 的列空间的维数是 n
- $A$ 的秩是 n
- $A$ 的零空间是 $\{0\}$
- $A$ 的零空间的秩是 0

#### 比较显然的秩的性质

- $\max \{ \mathrm{rank} A, \mathrm{rank} B \} \leq \mathrm{rank} ([A, B]) \leq \mathrm{rank} (A) + \mathrm{rank} (B)$
- $AX = B$ 有解 $\iff \mathrm{rank}A = \mathrm{rank} [A, B]$

#### 不那么显然的秩的性质

- $\mathrm{rank} (A + B) \leq \mathrm{rank} (A) + \mathrm{rank} (B)$

PROOF:
$$
\mathrm{rank}(A+B) < \mathrm{rank} \begin{bmatrix} A & A+B \\ O & B  \end{bmatrix} = \mathrm{rank} \begin{bmatrix} A & O \\ O & B  \end{bmatrix} = \mathrm{rank}A+\mathrm{rank}B
$$

- $\mathrm{rank} (AB) \leq \min \{ \mathrm{rank} A, \mathrm{rank} B\}$

PROOF:
$$
\begin{bmatrix} AB & A \end{bmatrix} \begin{bmatrix} I & 0 \\ -B & I \end{bmatrix} = \begin{bmatrix} O & A \end{bmatrix} \\
\begin{bmatrix} I & -A \\ O & I \end{bmatrix} \begin{bmatrix} AB \\ B \end{bmatrix} = \begin{bmatrix} O \\ B \end{bmatrix} \\
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

### 4.7_基变换

#### 定理十五：基变换的唯一性

向量空间 $V$ 存在两组基
$$
B = \{\vec{b_1}, \cdots, \vec{b_2}\} \\
C = \{\vec{c_1}, \cdots, \vec{c_2}\}
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

### 4.9_马尔科夫链中的应用

## 五、特征值和特征向量

### 5.1_特征向量和特征值

定义：对于 n 阶方阵 $A$，如果
$$
\exists \lambda \in R, \exists \vec{x} \in \R^n \text{ and } \vec{x} \neq \vec{0}, A \vec{x} = \lambda x
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

如果方阵的任意一行的和都等于 $\lambda$，那么 $\lambda$ 是该方阵的特征值，特征向量是 $[1,1,1]^T$.

例题：求矩阵 $A = \begin{bmatrix} a & b & \cdots & b \\ b & a & \cdots & b \\ \vdots & \vdots & \ddots & \vdots \\ b & b & \cdots & a \end{bmatrix}$ 的特征值。

解：

$$
A = b I_n + (a-b) J_n
$$

### 5.2_特征方程

#### 定理三：特征值与可逆矩阵定理

$A$ 是可逆的 $\iff$ 0 不是 $A$ 的特征值。
$$
\det(A - \lambda I) = 0
$$
**特征方程**的解是矩阵 $A$ 的**特征根**。如果 $A$ 是 $n \times n$ 方阵，那么特征方程是**特征根** $\lambda$ 的 $n$ 次多项式，称为**特征多项式**。

特征根 $\lambda$ 作为特征方程根的重数定义为 $\lambda$ 的**（代数）重数**。
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
\begin{align}
|A - \lambda I| = c \Pi_{i=1}^n (\lambda - \lambda_i) = c \left[ \lambda^n - (\sum_{i=1}^n \lambda_i) \lambda^{n-1} + \cdots + (-1)^n \Pi_{i=1}^n \lambda_i \right]
\end{align}
$$
根据行列式的性质可知，该多项式的最高项和次高项都来源于矩阵 $A - \lambda I$ 的主对角线元素的积。

对于次高项可以得到特征根的和等于方阵主对角线元素之和——**迹**。

对于最高项可以得到 $c = (-1)^n$

取 $\lambda = 0$，$|A| = c (-1)^n \Pi_{i=1}^n \lambda_i = \Pi_{i=1}^n \lambda_i$。证明完毕。

#### 相似性

对于 $n \times n$ 方阵 $A$ 和 $B$，如果存在可逆矩阵 $P$：
$$
P^{-1} A P = B
$$
则定义 $A$ 相似于 $B$.

相似矩阵运算具有线性
$$
A = PBP^{-1} \Rightarrow f(A) = P f(B) P^{-1} \\
f(A) = \sum_i c_i A^i
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

#### 求特征值的 QR 算法

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

如果方阵 $A$ 相似与对角矩阵，即存在可逆矩阵 $P$ 和对角矩阵 $D$ 使得 $A = PDP^{-1}$，则称 $A$ **可对角化**。

对角化使得计算一般矩阵的幂更加容易。
$$
A^n = (P D P^{-1})^n = P D^n (P{-1})^n
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

### 5.4_特征向量与线性变换

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

对于 $2 \times 2$ 矩阵的简单情况。

设 $A$ 2 $\times$ 2 实矩阵，有复特征值 $\lambda=a-b \mathrm{i}(b \neq 0)$ 及对应的 $\mathbb{C}^{2}$ 中的复特征向量 $\boldsymbol{v}$，那么

$$
A=P C P^{-1} \text {, 其中 } P=[\operatorname{Re} \boldsymbol{v} \operatorname{Im} \boldsymbol{v}], C=\left[\begin{array}{rr}
a & -b \\
b & a
\end{array}\right]
$$

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

#### 5.6_离散动力系统

#### 5.7_微分方程中的应用

#### 5.8_特征值的迭代估计

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
\mathrm{Nul} A = (\mathrm{Row} A)^{\bot} \\
\mathrm{Nul} A = (\mathrm{Col} A^T)^{\bot}
$$

PROOF:
$$
\forall \vec{x} \in \mathrm{Nul} A, A \vec{x} = \vec{0} \\
\vec{x} \cdot (A^T \vec{y}) = (A \vec{x})^T \vec{y} = 0 \\
\forall \vec{y} \in R^m, A^T \vec{y} \in \mathrm{Row} A \\
\mathrm{Nul} A = (\mathrm{Row}A)^T
$$

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

PROOF:
$$
\vec{y} \cdot \vec{u}_i = c_i ||\vec{u}_i||^2
$$

**正交矩阵**：
$$
A^T A = E \text{ or } A^T = A^{-1}
$$
PROPERTIES:

- 正交矩阵的转置矩阵、逆矩阵也是正交矩阵。
- 正交矩阵，等价于，矩阵的列向量的模为 1 且两两正交。
- 正交矩阵的积也是正交矩阵。$(PQ)^T PQ = Q^T P^T P Q = E$

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

如果 $W$ 是 $R^n$ 中的子空间

$$
\forall \vec{y} \in R^n, \exist \vec{\hat{y}} \in W, \exist \vec{z} \in W^{\bot}, \vec{y} = \vec{\hat{y}} + \vec{z}
$$

#### 定理九和十：最佳逼近定理

$$
\forall v \neq \hat{y} \in W, \Vert v - y \Vert > \Vert \hat{y} - y \Vert
$$

如果：$W = \mathrm{Span} \{u_1, u_2, \cdots, u_p\}$

$$
\hat{y} = \sum_{i=1}^{p} \frac{y \cdot u_i}{u_i \cdot u_i} u_i
$$

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

## 七、二次型

### 7.1_对称矩阵的对角化

#### 定理一：对称矩阵的特征向量

CONDITION: $A$ is a symmetric matrix.

CONCLUSION:
$$
\lambda_i \neq \lambda_j \Rightarrow \vec{v}_i \cdot \vec{v}_j = 0
$$

PROOF:
$$
x^T A y = x^T \lambda_2 y = \lambda_2 (x \cdot y) \\
x^T A y = x^T A^T y = (A x)^T y = (\lambda_1 x)^T y = \lambda_1 (x \cdot y)
(\lambda_1 - \lambda_2) (x \cdot y) = 0 \\
x \cdot y = 0
$$

- 实对称阵只有实数特征值。

PROOF:
共轭矩阵的共轭特征值：
$$
A \overline{x} = \overline{A} \overline{x} = \overline{\lambda} \overline{x}
$$
如果 $\lambda \neq \overline{\lambda}$，则根据上面的定理：
$$
\lambda \cdot \overline{\lambda} = 0 \\
\sum ||x_i||^2 = 0
$$
和特征向量不能为 0 的定义矛盾。因此，
$$
\lambda = \overline{\lambda} \in R
$$

- 实对称阵的任意特征值的几何重数和代数重数相等。
- 实对称阵必然可以对角化。
- 任意实对称阵必然正交相似于对角阵。（正交相似：过渡矩阵是正交阵）

二次型的一般形式：

$$
f(\vec{x}) = \sum_{i=1}^n a_{ii}x_i^2 + \sum_{i<j} 2 a_{ij} x_i x_j
$$

矩阵表示

$$
f(\vec{x}) = \vec{x}^T A \vec{x} \\
A = \begin{bmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{bmatrix}
$$

其中 $A$ 是对称矩阵，即 $a_{ij} = a_{ji} \ (i \neq j)$.

一种描述

$$
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
$$

**合同关系**：如果 $C$ 是可逆（正交）矩阵，称之为 $A$ 和 $B$ （正交）合同。

$$
B = C^T A C
$$

对矩阵 $A$ 做行列同步变换，得到的矩阵是合同矩阵。

$$
B = EAE^T
$$

其中 $E$ 是对应的初等矩阵。

**标准二次型**：只含有平方项，标准矩阵是对角矩阵的二次型。

**规范二次型**：标准二次型矩阵的元素仅有 1，-1，0.

将二次型标准化：

- 写出二次型对应的对称矩阵 $A$
- 将矩阵相似对角化 $A = Q D Q^T$，过渡矩阵 $Q$ 是规范正交矩阵。
- 写出对应的正交变换 $x = Q y$ 和新的标准型 $f(y)$

**正定矩阵和正定二次型**：

$$
\forall \vec{x} \neq 0 \in R^n, f(\vec{x}) = \vec{x}^T A \vec{x} > 0
$$
负定二次型同理。

对矩阵进行合同变换不改变矩阵的正定性。?

惯性定理?

$$

$$

对于实对称阵 $A$，下面的命题等价：

- $A$ 为正定矩阵
- $A$ 的所有特征值都是正值
- $A = C^T C$，其中 $C$ 是可逆矩阵

霍尔维茨定理

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
