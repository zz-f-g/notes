# 线性代数及其应用（一）

[toc]

## 一、线性代数中的线性方程组

### 本章内容简介

| 研究对象   | 存在性                       | 唯一性                         |
| ---------- | ---------------------------- | ------------------------------ |
| 线性方程组 | 增广矩阵的最后一列不为主元列 | 增广矩阵除最后一列外均为主元列 |
| 矩阵方程   | 矩阵每一行均为主元位置       | 齐次线性方程组无非平凡解       |
| 线性变换   | 满射（标准矩阵列张成上域）   | 单射（标准矩阵列线性无关）     |

### 一小结

本章的研究对象是：线性系统

聚焦于解的两个方面：

- **存在性 Consistence**
- **唯一性 Uniqueness**

求解的方法归根到底常用的只有一种，不要管 Crammer's Rule.

- 对增广矩阵(Augmented Matrix)，初等行变换(Elementary Row Operations)化为（简化）阶梯型(Reduced Echelon Form)
- 对应于求解多元一次方程组的算法

从解线性方程组的方法出发，定义对矩阵的三种变换：

1. 倍加变换
2. 对换变换
3. 倍乘变换

这些变换成为矩阵的**初等行变换**。

#### 定理一：任何矩阵仅有唯一的等价简化阶梯型矩阵

引入术语：**主元位置 (Pivot Position)** 和**主元列 (Pivot Column)**

主元位置

矩阵中的**主元位置**是其对应最简阶梯型中 1 的位置。

**存在性**看系数矩阵每行是否都有主元位置，如果增广矩阵在系数矩阵的没有主元位置的行上有主元位置，则不存在解。

**唯一性**看系数矩阵每列是否都有主元位置，如果**在存在的基础上**每一列不都有主元位置，则存在无穷多解。

#### 定理二：针对线性方程组的解

- **存在性**：增广矩阵的最右列不是主元列
- **唯一性**：在==存在==的前提下，没有**自由变量 (Free Variables)**

引入：向量及其运算（加法、数乘）：为了**代数和几何的联系**

引入术语：**线性组合 (Linear Combination)**

- 线性组合和线性方程组、增广矩阵的关系。

引入术语：张成**子空间 (Subset of $\mathbb{R}^{n}$ Spanned by …)**

- 定义：任意可写作给定向量的线性组合的向量集合

引入：矩阵和向量的乘法（作为线性组合的简化记号）

- **矩阵乘法**和**线性组合**、**线性方程组**的解的等价性

#### 定理四：矩阵方程解的存在性

当且仅当矩阵 $A$ 每一行均有主元位置，矩阵方程 $A\boldsymbol{x} = \boldsymbol{b}$ 解总存在

- 证明归结到讨论线性方程组**增广矩阵**的情形

#### 定理五：矩阵乘法的线性

引入：**齐次 (Homogeneous)** 线性方程组**平凡解 (Trivial Solution)** 和**非平凡解 (Nontrivial Solution)**：为了讨论任意矩阵方程解的**唯一性**

当且仅当齐次线性方程组存在自由变量，即不含有主元的列时，存在非平凡解。

引入：解的**参数向量形式 (Parametric Vector Form)**：为了更好地刻画解集的空间形状，并为了讨论非齐次

#### 定理六：矩阵方程解的唯一性

- 当矩阵方程 $A\vec{x} = \vec{b}$ 解存在时，解集为 $\vec{x} = \vec{p} + \vec{v_h}$
- $\vec{p}$ 是特解，$\vec{v_h}$ 是 $A\vec{x} = \vec{0}$ 的任意解

引入：向量的**线性无关 (Linearly Independence)**/相关：从向量视角研究解的**唯一性**

- 当且仅当 $\vec{x} = \vec{0}, A\vec{x} = \vec{0}$, $A$ 的各列线性无关。

#### 定理七：线性相关与线性组合

- 当且仅当向量集线性相关时，向量集中至少有一个向量可以表示为其他所有向量的线性组合

#### 定理八、定理九：线性相关的必要条件

- 定理八：向量数量多于向量中元素的数量
- 定理九：集合中存在零向量

引入：矩阵变换（对矩阵乘法几何意义的诠释）

引入：**线性变换 (Linear Transformation)**：“线性”的变换

证明一个变换是线性变换：

1. ==变换后的元素仍然在原来的空间中==
2. 加法封闭
3. 乘法封闭

#### 定理十：每个线性变换对应唯一的标准矩阵

- 标准矩阵中的每一列是**单位矩阵 (Identity Matrix)** 对应列线性变换后的结果

引入：**满射 (Onto)**、**单射 (One-to-one)**（对线性变换的**存在性**和**唯一性**讨论）

- 满射：每个因变量对应最少一个自变量
- 单射：每个因变量对应最多一个自变量

#### 定理十一、十二：满射&单射

- 满射等价于标准矩阵的列张成上域 (codomain)；单射等价于线性变换后结果为零仅有平凡解，等价于标准矩阵的列**线性无关**。

---

## 二、矩阵代数

### 2.1_矩阵操作

#### 矩阵中的元素

- 第 i 行第 j 列记为 $a_{ij}$
- 第 i 行第 i 列元素 $a_{ii}$ 为**对角线元素 (diagonal entries)**，他们构成**主对角线 (main diagonal)**
- **对角矩阵 (diagonal matrix)**：非对角线元素均为 0
- 零矩阵 (zero matrix)：矩阵元素均为零

```python
a = np.diag([1, 2, 3])
b = np.zero((2, 2))
```

单位矩阵（identity matrix）

~~~python
'''
1_{i \times i}
'''
from sympy import Matrix
Ei = Matrix(n,n,lambda a,b: 1 if a==i and b==i else 0)
~~~

- 矩阵相等：大小相等且对应元素相等
- 矩阵的和：大小相等的矩阵对应元素相加
- 矩阵的数乘：对应元素都乘给定的标量

#### 定理一：矩阵线性的基本运算法则

1. $A+B=B+A$
2. $A+(B+C)=(A+B)+C$
3. $A+0=A$
4. $r(A+B)=rA+rB$
5. $(r+s)A=rA+sA$
6. $r(sA)=(rs)A$

#### 矩阵相乘

- 右边矩阵的每一列与左边矩阵相乘，构成结果矩阵的每一列。
- 需满足条件：左边矩阵的列数等于右边矩阵的行数
- 结果矩阵的行数等于左边矩阵的行数，结果矩阵的列数等于右边矩阵的列数。
- 计算方法：
  - $AB=A[b_1 \ b_2 \ \cdots \ b_p]=[Ab_1 \ Ab_2 \ \cdots \ Ab_p]$
  - $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$
- 便于计算的公式：

$$
(ab)_{ij}=\sum_{k=1}^{n}{a_{ik}b_{kj}}
$$

```python
np.dot(A, B)
```

**矩阵相乘的意义就是求复合函数**

另有：

$$
\begin{aligned}
row_{i}{(AB)}&=row_{i}A \cdot B \\
column_{j}{(AB)}&=A \cdot column_{j}B
\end{aligned}
$$

#### 定理二：矩阵乘法的性质

$$
\left\{\begin{aligned}
A(BC) &= (AB)C \\
A(B + C)&= AB + AC \\
(B + C)A &= BA + BC \\
r(AB) &= (rA)B = A(rB) \\
I_{m} A &= A = AI_{n}
\end{aligned}\right.
$$

$AB$ 可**交换**（commute），当 $AB=BA, A, B \in \mathbb{R}^{n \times n}$.

$$
\forall B \in R^{n \times n}, AB = BA \iff A = k I_n \ (k \in R)
$$

PROOF:

B 可以表示为 $n \times n$ 个基的线性组合。对每一个基都可交换，那么对 B 就可交换。比如基

$$
\begin{aligned}
E_{ij} &= \left\{ a_{mn} \right\} && a_{mn} = \begin{cases}
1 && m = i, n = j \\
0 && \text{else}
\end{cases} \\
B &= \sum_{i, j}^{n, n} b_{ij} E_{ij}
\end{aligned}
$$

对基可交换

$$
\begin{aligned}
A E_{ij} &= \begin{bmatrix} 0 & \cdots & \text{col}_{i} A & \cdots & 0 \end{bmatrix} && (j \text{ th colomn}) \\
E_{ij} A &= \begin{bmatrix} 0\\ \vdots\\ \text{row}_{j} A\\ \vdots\\ 0\end{bmatrix} && (i \text{ th row})
\end{aligned}
$$

因此除了 $(AE_{ij})_{ij}$ 不为 0，其余都是 0.

$$
\begin{aligned}
\forall i, j, a_{ii} &= (AE_{ij})_{ij} = (E_{ij}A)_{ij} = a_{jj} \Rightarrow A = k I_{n}
\end{aligned}
$$

几个值得注意的地方：

1. 一般来说，$AB \neq BA$
2. $\exists B \neq C, \ \exists A, \ AB=AC$
3. $\exists A \neq 0 \ B \neq 0 , \ AB=0$

矩阵的幂

矩阵的转置

#### 定理三：矩阵转置

a. $(A^T)^T = A$
b. $(A+B)^T=A^T+B^T$
c. $\forall r \in R, \ (rA)^T=rA^T$
d. $(AB)^T=B^T A^T$

```python3
a = np.array([
        [1, 2],
        [3, 4]
    ])
a.transpose()
```

矩阵的积的转置的转置矩阵逆序求积

练习题：
    
$$
\begin{aligned}
    S &= \begin{bmatrix}
    0 & 1 & 0 & 0 & 0\\
    0 & 0 & 1 & 0 & 0\\
    0 & 0 & 0 & 1 & 0\\
    0 & 0 & 0 & 0 & 1\\
    0 & 0 & 0 & 0 & 0
    \end{bmatrix} \\
    S^{2} &= \begin{bmatrix}
    0 & 0 & 1 & 0 & 0\\
    0 & 0 & 0 & 1 & 0\\
    0 & 0 & 0 & 0 & 1\\
    0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0
    \end{bmatrix} \\
    S^{3} &= \begin{bmatrix}
    0 & 0 & 0 & 1 & 0\\
    0 & 0 & 0 & 0 & 1\\
    0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0
    \end{bmatrix} \\
    S^{4} &= \begin{bmatrix}
    0 & 0 & 0 & 0 & 1\\
    0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0\\
    0 & 0 & 0 & 0 & 0
    \end{bmatrix}
\end{aligned}
$$

---

### 2.2_矩阵的逆

#### 可逆矩阵

- if $\exists C \in \mathbb{R}^{n\times n},CA=I_n \text{ and } AC=I_n, \text{ then }A_{n \times n}$ is **invertible（可逆的）**.
- $C$ is an inverse of $A$. $C=A^{-1}$
- **可逆矩阵**有时被称为**非奇异矩阵 (nonsingular matrix)**，不可逆的也被称为**奇异矩阵 (singular matrix)**

#### 定理四：求解矩阵的逆

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

行列式定义：$\mathrm{det}(A)=ad-bc$. If $\mathrm{det}(A) = 0$，$A$ 不可逆。

$$
A^{-1} = \frac{1}{\mathrm{det}(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

#### 定理五：解矩阵方程

$$
\begin{aligned}
A \in \mathbb{R}^{m \times m}, & A \text{ is invertible.} \Rightarrow \\
\forall \vec{b} \in R^n ,& A\vec{x}=\vec{b} \ \text{ has a unique solution }  \vec{x}=A^{-1}\vec{b}
\end{aligned}
$$

可逆同时说明解的存在性和唯一性。

#### 定理六：可逆矩阵的性质

1. If $A$ is invertible, then the inverse of $A$ is also invertible. $(A^{-1})^{-1}=A$
2. If $A$, ***B*** and ***AB*** are both invertible and equal in shape, then $(AB)^{-1}=B^{-1}A^{-1}$
3. If $A$ is invertible, then the transpose of A is also invertible, $(A^T)^{-1}=(A^{-1})^T$

证明通过矩阵逆的定义可以很快完成。

对于第 2 条定理的推广：对于任意几个大小相同的正方形矩阵，其**乘积的逆等于逆的逆序乘积**。

引入**初等矩阵 (Elementary Matrices)**（为了计算矩阵的逆）

- 单位矩阵经过**一次**初等行变换后得到的矩阵定义为：初等矩阵。
- 对$A \in \mathbb{R}^{m \times n}$ 矩阵进行一次初等行变换操作等效于 $EA$，其中 $E \in \mathbb{R}^{m \times m}$ 矩阵是对单位矩阵进行**相同初等行变换**的结果。
- **所有初等矩阵均可逆**，其逆矩阵也是初等矩阵，代表着逆的初等行变换。

```python
from sympy import eye


def Interchange_EleMat(n, row1, row2):
    I_n = eye(n)
    temp = I_n[row1-1, :]
    I_n[row1-1, :] = I_n[row2-1, :]
    I_n[row2-1, :] = temp
    return I_n


def Scaling_EleMat(n, row, time):
    I_n = eye(n)
    I_n[row-1, :] *= time
    return I_n


def Replacement_EleMat(n, row1, row2, time):
    I_n = eye(n)
    I_n[row1-1, :] += time*I_n[row2-1, :]
    return I_n
```

#### 定理七：矩阵求逆

1. 矩阵可逆，当且仅当，矩阵行等价于大小相同的单位矩阵。
2. 将矩阵还原为等大单位矩阵的初等行变换操作集合，同时将单位矩阵变为该矩阵的逆。

定理七必要性的证明：

$$
\begin{aligned}
& A \sim I \Rightarrow A = E_{1} E_{2} \cdots E_{n} \\
& \exists C=E_n^{-1}E_{n-1}^{-1} \cdots E_{1^{-1}}, CA=I, AC=I
\end{aligned}
$$

事实上，对于奇异矩阵，也有相似的公式：

$$
\begin{aligned}
&\forall \boldsymbol{A} \in \mathbb{R}^{m \times n}, \exists \boldsymbol{P} \in \mathbb{R}^{m \times m}, Q \in \mathbb{R}^{n \times n} \\
&\boldsymbol{A} = \boldsymbol{P} \begin{bmatrix} I_{r} & \boldsymbol{O} \\ \boldsymbol{O} & \boldsymbol{O} \end{bmatrix} \boldsymbol{Q}
\end{aligned}
$$

其中 $\boldsymbol{P}, \boldsymbol{Q}$ 均可逆。可以理解为先对矩阵作初等行变换 $P = E_{1} E_{2} \cdots E_{a}$，将矩阵变为等价行最简阶梯型。再对矩阵作初等列变换 $\boldsymbol{Q} = E'_{1} E'_{2} \cdots E'_{b}$，让主元列集中到一起称为单位阵。

#### 求逆算法

- 将矩阵 $\begin{bmatrix}A & I\end{bmatrix}$ 化简为简化阶梯型

> 当然从另一种角度来看，这个化简过程是在求解一系列矩阵方程 $A\vec{x}=\vec{e_{i}}$，**每一个解对应 $A^{-1}$ 的对应列**。
>
> 这种观点在求逆矩阵的某几列时可以减少计算量。

---

### 2.3_可逆矩阵的特征

#### 定理八：可逆矩阵定理

对 $A \in \mathbb{R}^{n \times n}$ ，下面所有命题均等价

1. $A$ 可逆
2. $A$ 行等价于等大的单位矩阵
3. $A$ 有 n 个主元位置
4. 唯一性
   - $A\vec{x}=\vec{0}$ 仅有平凡解
   - $A$ 的各列线性无关
   - $T(\vec{x})=A\vec{x}$ 是单射
5. 存在性
   - $\forall \vec{b}\in R^n ,\ A\vec{x}=\vec{b}$ 有解
   - $A$ 的各列张成 $R^n$
   - $T(\vec{x})=A\vec{x}$ 是满射
6. $\exists C \in \mathbb{R}^{n \times n}, CA=I_n$
7. $\exists D \in \mathbb{R}^{n \times n}, AD=I_n$
8. $A^T$ 可逆

逻辑关系梳理如下：

~~~mermaid
graph TD
T1((A可逆))
T2(("A~I"))
T3((A有n个主元位置))
T4((Uniqueness))
T5((Consistence))
T6((CA=I))
T7((AD=I))
T8((A的转置矩阵可逆))

T1-.定理七.->T2--定理七-->T1
T1--定义-->T6--详细见下-->T4
T1--定义-->T7--详细见下-->T5
T3--定理七-->T8-.定理七.->T3
T2-->T3-->T2
T3-->T4-->T3
T3-->T5-->T3
~~~

- CA=I $\Rightarrow$ Uniqueness

$$
\begin{aligned}
& \boldsymbol{x} \in \mathrm{Nul} A, (A \boldsymbol{x} = \boldsymbol{0}) \\
& \boldsymbol{x} = I \boldsymbol{x} = CA \boldsymbol{x} = \boldsymbol{0}
\end{aligned}
$$

因此，$A \boldsymbol{x} = \boldsymbol{0}$ 只有 trival solution.

- AD=I $\Rightarrow$ Consistence

$$
\begin{aligned}
&\forall \vec{b}\in R^n, \exists \vec{x}=D\vec{b}\in R^n\\
&A\vec{x}=AD\vec{b}=\vec{b}
\end{aligned}
$$

#### 关于矩阵可逆性的注解

如果 $AB=I$，$A$ 不一定可逆。==一定要认识到，可逆的前提是方阵。==

#### 定理九：可逆线性变换

定义：与可逆矩阵相似

线性变换可逆，当且仅当其标准矩阵可逆。

#### 矩阵的条件数（condition_number）

矩阵的条件数可以用来反映其所对应的线性变换的变形程度，条件数越大，变形越厉害，对误差的敏感程度越高。

sympy 模块中得到条件数的方法：

~~~python
import sympy
m=[[4.5,3.1],[1.6,1.1]]
m=sympy.Matrix(m)
m.condition_number() #m为方阵矩阵
~~~

用条件数估计矩阵方程 $A\vec{x}=\vec{b}$ 解的精确度：

如果 $A$ 和 $\vec{b}$ 的元素最少有 r 位有效数字，$conditionNumber(A)\approx10^k$，则 $\vec{x}$ 的有效数字为 $r-k$ 位。

---

### 2.4_分块矩阵

分块矩阵就是将原来的矩阵沿一些水平线或竖直线分成若干分块。**将矩阵视为大矩阵的元素。**

#### 分块矩阵的乘法

**首先分块矩阵的对应关系要能保证他们可以相乘**。

将矩阵视为数，方法同经典矩阵乘法。注意[矩阵加法的定义](#矩阵中的元素)。

#### 定理十：矩阵乘法的行列展开（column-row_expand）

$$
\begin{aligned}
AB&=\left[\begin{matrix}col_1(A) & col_2(A) & \cdots & col_n(A)\end{matrix}\right]\left[\begin{matrix}row_1(B)\\row_2(B)\\ \cdots\\row_n(B)\end{matrix}\right] \\
&=col_1(A)row_1(B) + col_2(A)row_2(B) + \cdots + col_n(A)row_n(B)
\end{aligned}
$$

#### 分块矩阵的逆

形如 $A=\left[\begin{matrix}A_{11} & A_{12} \\ 0 & A_{22}\end{matrix}\right]$ 的矩阵是分块上三角矩阵，其中 $A_{11} \in \mathbb{R}^{p\times p}, A_{22} \in \mathbb{R}^{q\times q}$ 均可逆。

设 $B=\left[\begin{matrix}B_{11} & B_{12} \\ B_{21} & B_{22}\end{matrix}\right]=A^{-1}$，则 $\left[\begin{matrix}A_{11} & A_{12} \\ 0 & A_{22}\end{matrix}\right]\left[\begin{matrix}B_{11} & B_{12} \\ B_{21} & B_{22}\end{matrix}\right]=I$.

$$
\begin{aligned}
& A_{11}B_{11}+A_{12}B_{21}=I_p \\
& A_{11}B_{12}+A_{12}B_{22}=0 \\
& A_{22}B_{21}=0 \\
& A_{22}B_{22}=I_q
\end{aligned}
$$

根据可逆矩阵定理，$B_{22}=A_{22}^{-1},  B_{21}=0$.

得到 $B_{12}=-A_{11}^{-1}A_{12}A_{22}^{-1},\ B_{11}=A_{11}^{-1}$

$$
A^{-1}=\left[\begin{matrix}A_{11}^{-1} & -A_{11}^{-1}A_{12}A_{22}^{-1} \\ 0 & A_{22}^{-1}\end{matrix}\right]
$$

**分块对角矩阵**（partitioned diagonal matrix）指的是除了主对角线上的分块矩阵不为零，其余分块矩阵均为零的矩阵。

当且仅当其分块矩阵均可逆时分块对角矩阵才可逆。

#### 舒尔补

对于方阵矩阵 $A=\left[\begin{matrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{matrix}\right]$ 称

$$
S=A/A_{11}=A_{22}-A_{21}A_{11}^{-1}A_{12}
$$

为 $A_{11}$ 的**舒尔补（schur complement）**，称

$$
S=A/A_{22}=A_{11}-A_{12}A_{22}^{-1}A_{21}
$$

为 $A_{22}$ 的**舒尔补（schur complement）**。

前提是 $A_{11},\ A_{22}$ 均可逆。

**记忆方法：顺时针顺序，或者看它能不能乘**

对于第一种情况：

$$
\begin{aligned}
&\text{Suppose}
\left[\begin{matrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{matrix}\right]=
\left[\begin{matrix}I & 0 \\ X & I\end{matrix}\right]
\left[\begin{matrix}A_{11} & 0 \\ 0 & S\end{matrix}\right]
\left[\begin{matrix}I & Y \\ 0 & I\end{matrix}\right]\\
&X=A_{21}A_{11}^{-1},\ Y=A_{11}^{-1}A_{12} \\
&\text{So}
\left[\begin{matrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{matrix}\right]=
\left[\begin{matrix}I & 0 \\ A_{21}A_{11}^{-1} & I\end{matrix}\right]
\left[\begin{matrix}A_{11} & 0 \\ 0 & S\end{matrix}\right]
\left[\begin{matrix}I & A_{11}^{-1}A_{12} \\ 0 & I\end{matrix}\right]\\
&\text{To vertify:}\
XA_{11}Y+S=A_{22}
\end{aligned}
$$

$\left[\begin{matrix}I & 0 \\ A_{21}A_{11}^{-1} & I\end{matrix}\right],\ \left[\begin{matrix}I & A_{11}^{-1}A_{12} \\ 0 & I\end{matrix}\right]$ 均可看作以分块矩阵形式出现的初等矩阵。从这个意义上讲，舒尔补就是经过类似高斯消元的方法得到的结果。

此概念常应用于系统工程。

舒尔补也可以用来求逆：
$$
\begin{align*}
    & A = \begin{bmatrix}
        A_{11} & A_{12} \\ 
        A_{21} & A_{22}
    \end{bmatrix}=
    \begin{bmatrix}
        I & 0 \\ 
        A_{21}A_{11}^{-1} & I
    \end{bmatrix}
    \begin{bmatrix}
        A_{11} & 0 \\ 
        0 & S
    \end{bmatrix}
    \begin{bmatrix}
        I & A_{11}^{-1}A_{12} \\ 
        0 & I
    \end{bmatrix}\\
\end{align*}
$$

$$
\begin{align*}
    A^{-1} &=
    \begin{bmatrix}I & A_{11}^{-1}A_{12} \\ 0 & I\end{bmatrix}^{-1}
    \begin{bmatrix}A_{11}^{-1} & 0 \\ 0 & S^{-1}\end{bmatrix}
    \begin{bmatrix}I & 0 \\ A_{21}A_{11}^{-1} & I\end{bmatrix}^{-1} \\
    &=
    \begin{bmatrix}I & -A_{11}^{-1}A_{12} \\ 0 & I\end{bmatrix}
    \begin{bmatrix}A_{11}^{-1} & 0 \\ 0 & S^{-1}\end{bmatrix}
    \begin{bmatrix}I & 0 \\ -A_{21}A_{11}^{-1} & I\end{bmatrix}
\end{align*}
$$

### 2.5_矩阵的因式分解

- 矩阵乘法：数据的综合
- 矩阵的因式分解：数据的分析

#### 三角因式分解

假设矩阵 $A$（$m\times n$ 矩阵）可以在**没有行交换**的初等行变换下化简为阶梯型。则：

$$
\begin{aligned}
&A=LU\\
&L=\left[\begin{matrix}
1&0&0&0\\
l_{21}&1&0&0\\
l_{31}&l_{32}&1&0\\
l_{41}&l_{42}&l_{43}&1
\end{matrix}\right]\\
&U=\left[\begin{matrix}
u_{11}&u_{12}&u_{13}&u_{14}&u_{15}\\
0&u_{22}&u_{23}&u_{24}&u_{25}\\
0&0&0&u_{34}&u_{35}\\
0&0&0&0&u_{45}
\end{matrix}\right]
\end{aligned}
$$

其中 $L$ 为下三角方阵，且对角线均为 1，定义这样的矩阵为**单位下三角矩阵**。

~~~python
from sympy import *
A=Matrix([[2,-4,-1,5,-2],[-4,-5,3,-8,1],[2,-5,-4,1,8],[-6,0,7,-3,1]])
A.LUdecomposition()
~~~

这样就可以把解方程 $A\vec{x}=\vec{b}$ 的过程分为两步：

$$
\begin{aligned}
&L\vec{y}=\vec{b}\\
&U\vec{x}=\vec{y}
\end{aligned}
$$

由于两个矩阵都是三角矩阵，因此比较好解。

其实三角分解的优势在算一个单独的矩阵方程时并不能体现出优势。其优势体现在计算相同的系数矩阵的矩阵方程组时，相比直接行化简，可以节约许多共同的步骤。

#### 三角因式分解算法

设 $A$ 可以只用倍加行变换转化为阶梯型矩阵 $U$，则

$$
\begin{aligned}
&E_p\cdots E_1A=U\\
&A=LU, L=(E_p\cdots E_1)^{-1}
\end{aligned}
$$

其中 $E_i$ 均为单位下三角矩阵。（思考一下平时解多元一次线性方程组时的方法）

可以从此证明 $L$ 也为单位下三角矩阵。

由于 $E_p\cdots E_1\cdot L=E_p\cdots E_1\cdot (E_p\cdots E_1)^{-1}=I$，因此将 $A$ 变换为 $U$ 的倍加行变换的步骤也将待求的 $L$ 转化成了单位矩阵。求法见下面的[练习一](#练习)。

对于必须要用行交换才能转化为阶梯型的矩阵，其对应的 $L$ 将不再是单位下三角矩阵，而是**置换单位下三角矩阵**（指经过行交换以后可以成为单位下三角矩阵）。

$$
\begin{align*}
&L=PL'\\
&A=PL'U
\end{align*}
$$

#### 三角因式分解练习

一、求 $A=\left[\begin{matrix}2&4&-1&5&-2\\-4&-5&3&-8&1\\2&-5&-4&1&8\\-6&0&7&-3&1\end{matrix}\right]$ 的 LU 分解。

首先消去第一列的非主元：

$$
A\sim A_{1}=\left[\begin{matrix}2 & 4 & -1 & 5 & -2\\0 & 3 & 1 & 2 & -3\\0 & -9 & -3 & -4 & 10\\0 & 12 & 4 & 12 & -5\end{matrix}\right]
$$

注意到消去的过程就是将相应的行减去第一行乘第一行加上该行第一个元素除以第一行第一个元素（主元）。

$$
row_i(A_1)=row_i(A)-\frac{A_{i1}}{A_{11}}row_1(A),\ \text{when}\ i\neq 1
$$

那么可以找到在这样的变换下将 $L$ 的第一列转化为单位矩阵的第一列的 $L$ 的第一列。

$$
\begin{aligned}
L&=\left[\begin{matrix}1&0&0&0\\-2&1&0&0\\1&0&1&0\\-3&0&0&1\end{matrix}\right]\\
L_{i1}&=\frac{A_{i1}}{A_{11}}\\
A &=  L A_{1}
\end{aligned}
$$

这样经过相同的变换：

$$
L_{1,i1}=L_{i1}-\frac{A_{i1}}{A_{11}}L_{11}=0,\ \text{when}\ i\neq 1
$$

而其他元素不变。

相似地，可以根据 $A_{1}$ 求出 $L$ 的第二列，以此类推。同过对 $A$ 的行化简可以求出 $L$.

$$
L=\left[\begin{matrix}1&0&0&0\\-2&1&0&0\\1&-3&1&0\\-3&4&2&1\end{matrix}\right]\\
$$

---

二、求 $A=\left[\begin{matrix}2&-4&-2&3\\6&-9&-5&8\\2&-7&-3&9\\4&-2&-2&-1\\-6&3&3&4\end{matrix}\right]$ 的 LU 分解。

$$
\begin{align*}
    &A\sim A_1=\left[\begin{matrix}2&-4&-2&3\\0&3&1&-1\\0&-3&-1&6\\0&6&2&-7\\0&-9&-3&13\end{matrix}\right] \\
    &\sim A_2=\left[\begin{matrix}2&-4&-2&3\\0&3&1&-1\\0&0&0&5\\0&0&0&-5\\0&0&0&10\end{matrix}\right] \\
    &\sim A_3=\left[\begin{matrix}2&-4&-2&3\\0&3&1&-1\\0&0&0&5\\0&0&0&0\\0&0&0&0\end{matrix}\right]
\end{align*}
$$

由于该矩阵并不是每一列都有主元列，因此从基本的定义出发，重新探寻新的简单方法。

$$
\begin{align*}
&A_3=E_3E_2E_1A\\
&E_1=\left[\begin{matrix}1&0&0&0&0\\-3&1&0&0&0\\-1&0&1&0&0\\-2&0&0&1&0\\3&0&0&0&1\end{matrix}\right]\\
&E_2=\left[\begin{matrix}1&0&0&0&0\\0&1&0&0&0\\0&1&1&0&0\\0&-2&0&1&0\\0&3&0&0&1\end{matrix}\right]\\
&E_3=\left[\begin{matrix}1&0&0&0&0\\0&1&0&0&0\\0&0&1&0&0\\0&0&1&1&0\\0&0&-2&0&1\end{matrix}\right]
\end{align*}
$$

因此在做没有主元的一列时，对于 $L$，直接补齐下一列即可。

$$
L=\left[\begin{matrix}
1&0&0&0&0\\
3&1&0&0&0\\
1&-1&1&0&0\\
2&2&-1&1&0\\
-3&-3&2&0&1
\end{matrix}\right]
$$

#### 简化三角因式分解

事实上，对于练习二，我们还可以对矩阵 $L, U$ 进行简化，去掉多余的 0.

$$
\begin{align*}
&A=\left[\begin{matrix}
2&-4&-2&3\\
6&-9&-5&8\\
2&-7&-3&9\\
4&-2&-2&-1\\
-6&3&3&4
\end{matrix}\right]=LU\\
&L=\left[\begin{matrix}
1&0&0\\
3&1&0\\
1&-1&1\\
2&2&-1\\
-3&-3&2
\end{matrix}\right]\\
&U=\begin{bmatrix}
2&-4&-2&3\\
0&3&1&-1\\
0&0&0&5
\end{bmatrix}
\end{align*}
$$

对于这样的因式分解，首先要满足必要条件 $A$ 不可逆。一个 $m\times n$ 的矩阵乘 $n\times m$ 矩阵（$m>n$）的结果必然不可逆。因为后一个矩阵的各列必然线性相关，得到的结果矩阵的各列必然也线性相关。

#### 秩分解

基础知识参见第四章中有关秩的内容。

$$
\begin{aligned}
&A \in R^{m \times n}, \mathrm{rank}A = r
\Rightarrow A = P E_r Q \\
&E_r = \begin{bmatrix} I_r & O \\ O & O \end{bmatrix} = \sum_{i=1}^r 1_{i \times i} \\
A &= \sum_{i=1}^{r} \vec{p}_{i} \vec{q}_{i}^{T} \\
&= \begin{bmatrix}\boldsymbol{p}_{1} & \cdots & \boldsymbol{p}_{r}\end{bmatrix} \begin{bmatrix}
\boldsymbol{q}_{1}^{T} \\
\vdots \\
\boldsymbol{q}_{r}^{T}
\end{bmatrix}
\end{aligned}
$$

上面的推导表明：对于矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times n}, \mathrm{rank} \boldsymbol{A} = r$，则矩阵可以拆成一个列满秩矩阵 $\boldsymbol{B} \in \mathbb{R}^{m \times r}$ 和一个行满秩矩阵 $\boldsymbol{C} \in \mathbb{R}^{r \times n}$ 的乘积。

满秩分解是求任意矩阵伪逆过程中需要的第一步。

和 LU 分解一样，满秩分解也可以通过行化简来实现。其实现过程甚至比 LU 分解更为简单。不失普遍性，假设矩阵 $\boldsymbol{A}$ 的前几列 $\left\{ \boldsymbol{a}_{1}, \boldsymbol{a}_{2}, \cdots, \boldsymbol{a}_{r} \right\}$ 线性无关。后面的几列都可以表示为前面 $r$ 列的线性组合。

$$
\begin{aligned}
\boldsymbol{a}_{k} &= \begin{bmatrix}\boldsymbol{a}_{1} & \cdots & \boldsymbol{a}_{r}\end{bmatrix} \begin{bmatrix}c_{1k} \\ \vdots \\ c_{rk}\end{bmatrix} && (r < k \leq n) \\
\boldsymbol{A} & \sim \begin{bmatrix}I_{O} & \boldsymbol{c}_{k} & \boldsymbol{c}_{k+1} & \cdots & \boldsymbol{c}_{n}\end{bmatrix}
\end{aligned}
$$

其中用 $\boldsymbol{I}_{O} \in \mathbb{R}^{m \times r}$ 表示在 $I_{r}$ 的中间插入 $(m-r)$ 行零行的矩阵。

摘取行化简以后的矩阵，删除零行，将矩阵行交换重拍得到新矩阵

$$
\begin{aligned}
\boldsymbol{C} &= \begin{bmatrix} \boldsymbol{I}_{r} & \boldsymbol{c}'_{k} & \cdots & \boldsymbol{c}'_{n} \end{bmatrix} \\
\boldsymbol{B} &= \begin{bmatrix}\boldsymbol{a}_{1} & \cdots & \boldsymbol{a}_{r}\end{bmatrix}
\end{aligned}
$$

注意到

$$
\begin{aligned}
\boldsymbol{a}_{k} &= \boldsymbol{B} \boldsymbol{c}'_{k}
\end{aligned}
$$

因此上面的 $\boldsymbol{B}, \boldsymbol{C}$ 就是满秩分解的结果。

举例：

$$
\begin{aligned}
\boldsymbol{A} &= \begin{bmatrix}
2 & 1 & -2 & 5 \\
1 & 0 & -3 & 2 \\
3 & 1 & -13 & 5
\end{bmatrix} \sim \begin{bmatrix}
0 & 1 & 4 & 1 \\
1 & 0 & -3 & 2 \\
0 & 0 & 0 & 0
\end{bmatrix} \\
\boldsymbol{B} &= \begin{bmatrix}
2 & 1 \\
1 & 0 \\
3 & 1
\end{bmatrix} \\
\boldsymbol{C} &= \begin{bmatrix}
1 & 0 & -3 & 2 \\
0 & 1 & 4 & 1
\end{bmatrix}
\end{aligned}
$$

#### qr_分解

一种特殊的 LU 分解。基础知识参见第六章 GramSchmidt 方法。

$$
\begin{aligned}
&A=QR\\
&Q,R \in R^{n \times n}\\
&Q^TQ=I_n\\
&R\ \text{is invertible upper triangular matrix.}
\end{aligned}
$$

***Q*** 也叫正交矩阵。

$$
\begin{aligned}
Q&=\begin{bmatrix}
\vec{q_1}&\vec{q_2}&\vec{q_3}&\cdots&\vec{q_n}
\end{bmatrix}\\
q_{i}^{T}\cdot q_{j}&=
\left\{
\begin{array}{lr}
1&i=j\\
0&i\neq j
\end{array}
\right.\\
Q^{T}Q&=\begin{bmatrix}
\vec{q_1}^T\\
\vec{q_2}^T\\
\vec{q_3}^T\\
\cdots\\
\vec{q_n}^T
\end{bmatrix}
\begin{bmatrix}
\vec{q_1}&\vec{q_2}&\vec{q_3}&\cdots&\vec{q_n}
\end{bmatrix}=I_n
\end{aligned}
$$

#### 奇异值分解

详见第七章。

#### 谱分解

$$
A=PDP^{-1}
$$

其中 ***D*** 为对角矩阵。这种矩阵的因式分解在求矩阵的幂时非常有用。

$$
A^n=PD^nP^{-1}
$$

求解算法需要用到特征值。

### 2.6_leontief_投入产出模型

#### 消耗矩阵

$$
\vec{x}=C\vec{x}+\vec{d}
$$

其中 $\vec{x}$ 指的是总生产量，$C\vec{x}$ 指的是生产成本，$\vec{d}$ 指的是最终的消费需求。

$$
\vec{x}=(I-C)^{-1}\vec{d},\ \text{when}\ (I-C)\ \text{is invertible.}
$$

而中间需求（生产成本）也可已通过迭代的方式理解。

$$
\begin{aligned}
&\vec{x}=\vec{d}+C\vec{d}+C^2\vec{d}+C^3\vec{d}+\cdots+C^m\vec{d}\\
&(I-C)(I+C+C^2+C^3+\cdots+C^m)=I-C^{m+1}\\
&\text{Considering }\Sigma_{i=1}^nC_{ij}<1,\ lim_{m\rightarrow\infty}C^m=0\\
&\vec{x}=(I-C)^{-1}\vec{d}
\end{aligned}
$$

### 2.7_计算机图形学中的应用

如何储存一个二维图形？

将二维图形转化为小线段的集合，首先要记录线段端点的信息。

设有 n 个端点，用一个 $2\times n$ 的矩阵来记录端点的坐标信息。每一列表示原点到端点的向量。

这样将一个二维方阵左乘端点坐标矩阵，就可以得到对应的线性变换后的结果。

#### 齐次坐标

引入齐次坐标来实现点的平移的矩阵描述。

对于一点，其对应坐标为 $\begin{bmatrix}x\\y\end{bmatrix}$，引入其齐次坐标 $\begin{bmatrix}x\\y\\1\end{bmatrix}$.

这样通过如下矩阵乘法就可以描述**平移**：

$$
\begin{bmatrix}
1&0&h\\
0&1&k\\
0&0&1
\end{bmatrix}\cdot
\begin{bmatrix}x\\y\\1\end{bmatrix}=
\begin{bmatrix}x+h\\y+k\\1\end{bmatrix}
$$

**这种引入方法解决了线性变换中原点不变的问题，通过升高维度的方法使二维平面的原点不再是原点。**

描述**旋转**：

$$
\begin{bmatrix}
cos(\varphi)&sin(\varphi)&0\\
-sin(\varphi)&cos(\varphi)&0\\
0&0&1
\end{bmatrix}
$$

描述关于直线 $y=x$ **对称**：

$$
\begin{bmatrix}
0&1&0\\
1&0&0\\
0&0&1
\end{bmatrix}
$$

描述**伸缩变换**：

$$
\begin{bmatrix}
a&0&0\\
0&b&0\\
0&0&1
\end{bmatrix}
$$

更高维度的齐次坐标同理。

组合变换可以通过上面矩阵的依次左乘实现，也可以通过矩阵的乘积一次实现。

对于三维，还要考虑**中心投影**。

假设眼睛位于坐标 $(0,0,d)$ 处，对于坐标 $(x,y,z)$ 处的点，根据相似三角形，可以得到投影到画面 $z=0$ 上的坐标 $(x^*,y^*,0)$.

$$
\begin{aligned}
x^*=\frac{x}{1-\frac{z}{d}}\\
y^*=\frac{y}{1-\frac{z}{d}}
\end{aligned}
$$

应用齐次坐标。

$$
\vec{r^*}=\begin{bmatrix}
x\\y\\0\\1-\frac{z}{d}
\end{bmatrix}
=P\vec{r}
$$

这里对坐标同时乘了一个 $1-\frac{z}{d}$，便于计算。

$$
P=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&0\\
0&0&-\frac{1}{d}&1
\end{bmatrix}
$$

这就是表示投影的矩阵。

算出 $\vec{r^*}$ 以后，把每个坐标除以第四行的同一列元素就得到了投影以后的坐标（注意 ***z*** 坐标为 0）。

#### 投影变换与初等反射变换

对于平面 $A$ 的单位法向量 $\vec{u}$，可以构造矩阵：

$$
\begin{aligned}
P=\vec{u}\cdot \vec{u}^T\\
Q=I-2P
\end{aligned}
$$

使得 $P\vec{x}$ 表示将 $\vec{x}$ 投影到平面 $A$ 的法向上，$(I-P)\vec{x}$ 表示将 $\vec{x}$ 投影到平面 $A$ 上，$Q\vec{x}$ 表示将 $\vec{x}$ 关于平面 $A$ 反射。

$P, Q$ 有如下性质：

$$
\begin{aligned}
& P^{2}= \boldsymbol{u} (\boldsymbol{u}^{T} \boldsymbol{u}) \boldsymbol{u}^{T} = P\\
& P^{T}=P\\
& Q^{2}= (I - 2P) (I - 2P) = I - 4P + 4P^{2} =I
\end{aligned}
$$

### 2.8_子空间

#### 子空间定义

$\mathbb{R}^{n}$ 的子空间（$H\subseteq \mathbb{R}^{n}$）：

- 包含原点（$\vec{0}\in H$）
- 包含所有内部的和（$\forall \vec{x},\vec{y}\in H,\vec{x}+\vec{y}\in H$）
- 包含所有内部的数乘（$\forall\vec{x}\in H,\forall a\in R,a\vec{x}\in H$）

### 2.9_维数与秩

### 二小结

本小结对矩阵进行了承接、规范以及深化。

1. 承接

   在可逆矩阵定理一章中，通过对第一章中**存在性、唯一性**的引用，直接将这些性质套用在了可逆矩阵上：

   - $A$ 行等价于等大的单位矩阵
   - $A$ 有 n 个主元位置
   - 唯一性
     - $A\vec{x}=\vec{0}$ 仅有平凡解
     - $A$ 的各列线性无关
     - $T(\vec{x})=A\vec{x}$ 是单射
   - 存在性
     - $\forall \vec{b}\in R^n ,\ A\vec{x}=\vec{b}$ 有解
     - $A$ 的各列张成 $R^n$
     - $T(\vec{x})=A\vec{x}$ 是满射

2. 规范

   规范建立在完善的定义定理体系上。

   - 首先定义加法和数乘的定义，得到相应的[基本公式](#定理一：矩阵线性的基本运算法则)。
   - 再定义转置，得到相应的[基本公式](#定理三：矩阵转置)。
   - 再从矩阵对向量的乘法引出矩阵之间的乘法：本质上是一种复合函数。
   - 再引出矩阵的逆，与第一章中的内容进行了承接。

   还有对一些概念的矩阵描述。

   - 在[求逆的过程](#求逆算法)中，对初等行变换进行了矩阵描述。
   - 在[计算机图形学中的应用](#2.7_计算机图形学中的应用)一章中，对图形的线性变换进行了描述和拓展（平移、中心投影）。

3. 深化

   [分块矩阵](#2.4_分块矩阵)一章中，指出矩阵的元素可以是矩阵，这便利了很多归纳法的证明。

一些重要的算法：

- [矩阵求逆算法](#求逆算法)
- [舒尔补的算法](#舒尔补)
- [三角因式分解算法](#三角因式分解算法)

---

## 三、行列式

### 3.1_行列式介绍

对于 $2\times 2$ 的矩阵而言，行列式是否为零是判断矩阵可逆不可逆的标志。对于更高阶的矩阵，扩展该定义。

观察高阶矩阵行化简的过程。

#### 行列式定义

余子式：对于矩阵 $A$ ，其余子式 $A_{ij}$ 表示删去第 i 行和第 j 列的子矩阵。

行列式（对于矩阵阶数 $n>2$）：

$$
\begin{aligned}
&\mathrm{det}A=\sum_{j=1}^{n} (-1)^{1+j} a_{1j} \cdot \mathrm{det}A_{1j}\\
&\text{Suppose } C_{ij}=(-1)^{1+j}\cdot detA_{1j}\\
&\text{then } \mathrm{det}A=\sum_{j=1}^na_{1j}\cdot C_{1j}\\
\end{aligned}
$$

称作为矩阵第一行的余子式展开式。

#### 定理一：矩阵可以按任意行或列展开

因此计算时最好找 0 最多的一行或者一列。

#### 定理二：三角矩阵的行列式为对角矩阵的乘积

### 3.2_行列式的性质

#### 定理三：初等行变换与行列式

- 倍加变换：行列式不变
- 倍乘变换：行列式乘相应的数
- 交换变换：行列式乘 -1

证明：

定理转述为：

$$
\begin{aligned}
&\mathrm{det}(EA)=\mathrm{det}E\cdot \mathrm{det}A\\
&\mathrm{det}E=
\begin{cases}
1&\text{replacement}\\
r&\text{scalar}\\
-1&\text{interchange}
\end{cases}
\end{aligned}
$$

证明采用数学归纳法。当为 2 阶行列式时显然成立。

对于任意的高阶矩阵，由于任何初等矩阵变换最多涉及 2 行。因此取没有被变换的一行进行行列式的展开，将变成低一阶的矩阵，从而证明了归纳递推。

#### 定理四：行列式与可逆性

矩阵 $A$ 可逆，当且仅当 $\mathrm{det}(A)\neq 0$.

基于定理三和可逆矩阵定理，这一点易于证明（通过化简为阶梯型），同时也是对矩阵可逆定理的扩充。

#### 定理五：转置矩阵的行列式

$$
\mathrm{det}(A^{T})=\mathrm{det}(A)
$$

基于定理一，对 $A^{T}$ 的第一列的展开等于对 $A$ 的第一行的展开（数学归纳法）。

#### 定理六：乘积的行列式

$$
\mathrm{det}(AB)=\mathrm{det}(A)\cdot \mathrm{det}(B)
$$

进行可逆或不可逆的分类讨论，应用定理三。

#### 从变换的角度看行列式

$$
\begin{aligned}
& T: \mathbb{R}^{n} \rightarrow \mathbb{R}\\
& T(\boldsymbol{x}) = \mathrm{det} A = \mathrm{det} \begin{bmatrix} \boldsymbol{a}_{1} & \cdots & \boldsymbol{a}_{i-1} & \boldsymbol{x} & \boldsymbol{a}_{i+1} & \cdots & \boldsymbol{a}_{n} \end{bmatrix}
\end{aligned}
$$

可以证明它是线性变换。

$$
\begin{aligned}
T(c \boldsymbol{x}) &= \mathrm{det} \begin{bmatrix}\boldsymbol{a}_{1} \cdots & \mathrm{a}_{i-1} & c \boldsymbol{x} & \boldsymbol{a}_{i+1} & \cdots & \boldsymbol{a}_{n} \end{bmatrix} \\
&= \mathrm{det} (A E_{c, i}) \\
&= c T(\boldsymbol{x}) \\
T(\boldsymbol{x} + \boldsymbol{y}) &= \sum_{j=1}^{n} (x_{j} + y_{j}) C_{ij} \\
&= \sum_{j=1}^{n} x_{j} C_{ij} + \sum_{j=1}^{n} y_{j} C_{ij} \\
&= T(\boldsymbol{x}) + T(\boldsymbol{y})
\end{aligned}
$$

#### 范德蒙德矩阵与插值多项式

Vandermonde Matrix

$$
\begin{aligned}
V &= [a_{ij}]_{n \times n} \\
a_{ij} &= x_i^{j-1}
\end{aligned}
$$

define the polynomial:

$$
p(x)=\sum_{i=1}^{n} c_{i-1} x^{i-1}
$$

define the vector:

$$
\begin{aligned}
\vec{c} &= [c_0, c_1, \cdots, c_{n-1}]^{T} \\
\vec{y} &= [y_1, \cdots, y_{n}]^{T}
\end{aligned}
$$

if $(x_i, y_i) \ i = 1,2,\cdots, n$ are n points on the graph of polynomial $p(x)$
then solve $V \vec{c} = \vec{y}$ to get the coefficients in $p(x)$

证明范德蒙德矩阵可逆：

令 $\vec{y} = \vec{0}$，则 $\{x_n\}$ 构成多项式的零点集合子集。对于最高次数为 $(n-1)$ 的一元方程，最多有 $n-1$ 个不同的零点。对于多出来的某一个值 $x_{k}: p(x_{k}) = 0 \Rightarrow \forall i, c_{i} = 0$，多项式的系数均为 0. 方程 $V\vec{c}=\vec{0}$ 仅有平凡解，范德蒙德矩阵可逆。

#### 消去非对角元素的技巧

$$
\begin{aligned}
& A = |a_{ij}|  \\
& a_{ij} = i a_{j} \text{ if } i \neq j \text{ else } b
\end{aligned}
$$

### 3.3_克莱默法则、体积和线性变换

#### 定理七：克莱默法则

$$
\begin{aligned}
&A \vec{x} = \vec{b} \\
&x_{i}=\frac{\det{A_i(\vec{b})}}{\det{A}} \\
&\text{where, }A_i(\vec{b})=\begin{bmatrix} \vec{a_1} & \cdots & \vec{a_{i-1}} & \vec{b} & \vec{a_{i+1}} & \cdots & \vec{a_n} \end{bmatrix}
\end{aligned}
$$

证明：

$$
\begin{aligned}
& X_{i} = E_{i}(\boldsymbol{x}) = \begin{bmatrix}
1 & \cdots & 0 & x_1 & 0 & \cdots & 0 \\
\cdots \\
0 & \cdots & 1 & x_{i-1} & 0 & \cdots & 0 \\
0 & \cdots & 0 & x_i & 0 & \cdots & 0 \\
0 & \cdots & 0 & x_{i+1} & 1 & \cdots & 0 \\
\cdots \\
0 & \cdots & 0 & x_n & 0 & \cdots & 1
\end{bmatrix} \\
&AX_{i} = \begin{bmatrix} \vec{a_1} \cdots \vec{a_{i-1}} \ \sum_{i=1}^n x_i \vec{a_i} \ \vec{a_{i+1}} \cdots \vec{a_n} \end{bmatrix}
=\begin{bmatrix} \vec{a_1} \cdots \vec{a_{i-1}} \ \vec{b} \ \vec{a_{i+1}} \cdots \vec{a_n} \end{bmatrix}
=A_i(\vec{b}) \\
& x_{i} = \det{X_i} = \det{A_i(\vec{b})} / \det{A}
\end{aligned}
$$

#### 定理八：伴随矩阵与逆矩阵

通过克莱默法则求矩阵的逆。

$$
\begin{aligned}
& A A^{-1} = I \Rightarrow A \mathrm{col}_{j} (A^{-1}) = \mathrm{col}_{j} I = \boldsymbol{e}_{j} \\
& A^{-1}_{ij} = \frac{\det A_i(e_j)}{\det A} = \frac{\det (-1)^{i+j} \det A_{ji}}{\det A} = \frac{Cji}{\det A}\\
& A^{-1} = \frac{1}{\det A} \begin{bmatrix}
C_{11} & \cdots & C_{n1} \\
\cdots \\
C_{1n} & \cdots & C_{nn}
\end{bmatrix}\\
& \mathrm{adj } A = \begin{bmatrix}
C_{11} & \cdots & C_{n1} \\
\cdots \\
C_{1n} & \cdots & C_{nn}
\end{bmatrix}
\end{aligned}
$$

**注意，伴随矩阵的余子式的下标相当于正常下标的转置。**

#### 定理九：行列式与体积

矩阵 $A$ 的列构成的图形的广义体积等于 $|\det A|$.

#### 定理十：线性变换与体积

$$
V(T(S)) = |\mathrm{det}(A)| V(S)
$$

其中 $S$ 为定义域的子集，$A$ 为线性变换 $T$ $V$ 表示广义体积。

### 3.4_行列式求解技巧简介

行列式的另外一种定义：

$$
\begin{aligned}
& A=(a_{ij})_{n \times n} \\
& \det A = \sum_{j_1j_2\cdots j_n} (-1)^{\tau(j_1j_2\cdots j_n)} a_{1,j_1} a_{2,j_2} \cdots a_{n, j_n}
\end{aligned}
$$

其中 $\tau(j_1j_2\cdots j_n)$ 指的是对于排列 $j_1j_2 \cdots j_n$ 的逆序数（逆序对的数目）。

一个生成逆序数的代码便于加深理解：

~~~python
def getInvCount(arr, n):
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count
~~~

例题

$$
A=aI+b(J-I)=\begin{bmatrix}
a & b & b & \cdots & b \\
b & a & b & \cdots & b \\
b & b & a & \cdots & b \\
\cdots \\
b & b & b & \cdots & b
\end{bmatrix}
$$

方法一：加一行特殊行。

$$
\begin{aligned}
|A| &= \begin{vmatrix}
1 & b & \cdots & b & b \\
0 & a & \cdots & b & b \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & b & \cdots & a & b \\
0 & b & \cdots & b & a
\end{vmatrix} \\
&= \begin{vmatrix}
1 & b & \cdots & b & b \\
-1 & a-b & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
-1 & 0 & \cdots & a-b & 0 \\
-1 & 0 & \cdots & 0 & a-b
\end{vmatrix} \\
&= \begin{vmatrix}
1 + \frac{nb}{a-b} & b & \cdots & b \\
0 & a-b & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a-b
\end{vmatrix} \\
&= \left(1 + \frac{nb}{a-b}\right) (a-b)^{n} \\
&= (a-b)^{n-1} [a + (n-1)b]
\end{aligned}
$$

方法二：利用[行列式可以看作线性变换](#从变换的角度看行列式)这一特点来进行拆分。

方法三：传统方法，逐行消去。
