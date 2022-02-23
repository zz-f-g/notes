# Optimization

答疑：周五晚上 17：00 ~ 19：00，线上预约线下答疑，（智信馆 611）。

---

什么是最优化？

> 用数学的方式，借助计算机工具**描述和找出**优化问题最优解的一门学科。

机器学习 = 模型 + 优化方法

---

简单例子：回归分析

- 建立模型
$$
y = b_0 + b_1 t
$$
- 优化问题
$$
\min_{b_0, b_1} \left \Vert \begin{bmatrix}
1 & t_1 \\
\vdots & \vdots \\
1 & t_n
\end{bmatrix} \begin{bmatrix}
b_0 \\ b_1
\end{bmatrix} - \begin{bmatrix}
y_1 \\ \vdots \\ y_n
\end{bmatrix} \right \Vert
$$
---

最优化的应用：
1. 智能制造
2. 生产调度
3. 机器人
4. 运动规划（模型预测控制）

---

## 0 Introduction

---

### 0.1

什么是最优化？

三要素
- 优化目标
- 决策变量
- 约束条件

---

研究的问题
- 数学性质（线性？凸问题？存在？唯一？灵敏度？）
-  构造寻求最优解的计算方法
- 分析计算方法的：１.理论性质；２.实际计算表现

---

一些求解的方法
- 图解法
- 解析法
- 计算机＋最优化理论：迭代法
	- 搜索方向 $\lambda_k$
	- 搜索步长 $p^{(k)}$
$$
x^{(k+1)} = x^{(k)} + \lambda_k p^{(k)}
$$

---

难点：局部最优解和全局最优解不统一。

鞍点：在某一个维度上达到极值，但在另一个维度上不是。

---

### 0.2

课程的主要内容
1. 数学知识
2. 无约束优化
3. 线性规划
4. 非线性规划

---

## 5 Calculus Basic

### 5.1

序列、极限、函数连续性

微积分的基本理念：利用仿射函数对函数进行局部近似。
$$
\begin{aligned}
f: R^n \rightarrow R^m \\
L \in R^{m \times n}, y \in R^m \\
A(x) = L(x) + y \\
\end{aligned}
$$

---

### 5.2

differentiable

---

### 5.3

derivative matrix

---

### 5.4

principle of derivative

---

### 5.5

水平集和参数化

---

### 5.6

Taylor series

---

## 6 集合约束和无约束优化问题

---

### 6.1

intoduction

---

$$
\begin{align*}
    &\text{minimize } f(x)\\
    &\text{subject to } x \in \Omega
\end{align*}
$$
从可行集合中找到使得目标函数最小的点。
$$
\begin{align*}
    &f:\mathbb{R}^{n} \rightarrow \mathbb{R}, \text{domain:} \Omega \subset \mathbb{R}^{n}\\
    &\exists \epsilon > 0, \forall x: 0 < \Vert x - x^{*} \Vert < \epsilon, f(x) > f(x^{*})
\end{align*}
$$

- 局部极小点
- 全局最小点

---

### 6.2

局部极小点的条件

---

可行方向

所有的可行方向构成了可行基在某一个点附近的逼近。

方向导数

$$
\frac{\partial f}{\partial d}(x) = \lim_{\alpha \rightarrow 0} \frac{f(x+\alpha d)-f(x)}{\alpha}
$$
计算简化
$$
\begin{align*}
    &\phi(\alpha) = f(x+\alpha d) \\
    &\frac{\partial f}{\partial d}(x) = \lim_{\alpha \rightarrow 0} \frac{\phi(\alpha)-\phi(0)}{\alpha} \\
    &= \phi'(0) = \nabla f(x)^{T} d = d^{T}\nabla f(x)
\end{align*}
$$

---

定理一：局部极小点的一阶必要条件

这是**排除法**

$$
\frac{\partial f}{\partial d}(x^{*}) = d^{T} \nabla f(x^{*}) \geq 0
$$

如果 $x^{*}$ 在 $\Omega$ 内部，则：
$$
\nabla f(x^{*}) = 0
$$

---

定理二：局部极小点的二阶必要条件

前提：满足一阶必要条件
$$
\phi''(0) = d^{T} D^{2}f(x^{*}) d \geq 0
$$
黑塞矩阵半正定。

---

当然，满足两个条件也不一定就是极小值（必要性）。例如：

- $x=0, f(x) = x^{3}$

但是只要有一个必要条件不满足，就一定不是极小值点。例如：

- $x = \begin{bmatrix}0 \\ 0\end{bmatrix}, f(x) = x_{1}^{2} - x_{2}^{2}$

---

定理三：局部极小点的二阶充分条件

前提：内点

$$
\begin{align*}
    &\nabla f(x^{*}) = 0\\
    &F(x^{*}) = D^{2}f(x^{*}) > 0
\end{align*}
$$
第二条条件指的是矩阵正定。
