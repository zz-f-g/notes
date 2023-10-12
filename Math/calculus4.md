# Calculus4

[TOC]

---

## Concepts

---

### 曲线积分和曲面积分

---

曲线积分

- 第一类：标量积分

$$
\begin{aligned}
    I &:= \int_{L} \lambda (\boldsymbol{r})  \mathrm{d} s
\end{aligned}
$$

做法：同一变量化成定积分，积分限**从小到大**。

- 第二类：矢量积分

$$
\begin{aligned}
    I&:= \int_{L} \boldsymbol{F}(\boldsymbol{r}) \cdot  \mathrm{d} \boldsymbol{r}
\end{aligned}
$$

做法：同一变量化成定积分，积分限**从起点到终点**。

曲面积分

- 第一类：标量积分

$$
\begin{aligned}
    I&:= \iint_{\Sigma} \sigma(\boldsymbol{r})  \mathrm{d} S
\end{aligned}
$$

- 第二类：矢量积分

$$
\begin{aligned}
    I&:= \iint_{\Sigma} \boldsymbol{E} \cdot  \mathrm{d} \boldsymbol{S}
\end{aligned}
$$

这些积分之间的关系：

- 格林公式：矢量线积、二重积分

Condition:

1. 闭区域 $D$ 的边界 $\partial D$ 为连续光滑曲线
2. $P(x, y), Q(x, y)$ 在 $D$ 上有一阶连续偏导数

$$
\begin{aligned}
    & \iint_{D} \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)  \mathrm{d} \sigma = \oint_{\partial D^{+}} P  \mathrm{d} x + Q  \mathrm{d} y
\end{aligned}
$$

$\partial D^{+}$ 表示曲线的正方向（曲线的左侧为内部区域）。

- 高斯公式：矢量面积、三重积分

Condition:

1. 闭区域 $\Omega$ 的边界 $\partial \Omega$ 为分片光滑闭曲面
2. $P(x, y, z), Q(x, y, z), R(x, y, z)$ 在 $\Omega$ 上有一阶连续偏导数

$$
\begin{aligned}
    & \iiint_{\Omega}  \left( \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial Q}{\partial z} \right)  \mathrm{d}v = \iint_{\partial \Omega^{+}} P  \mathrm{d} y  \mathrm{d} z + Q  \mathrm{d}z  \mathrm{d}x + R  \mathrm{d}x  \mathrm{d} y \\
    & \iiint_{\Omega} (\nabla \cdot \boldsymbol{E}) \mathrm{d} v = \iint_{\partial \Omega^{+}} \boldsymbol{E} \cdot  \mathrm{d} \boldsymbol{S}
\end{aligned}
$$

$\partial \Omega^{+}$ 表示曲面的外侧。

直观证明：考虑一个轴向和 $z$ 轴平行的柱状物上下曲面为
$$
\begin{align*}
    &\Sigma_{2}: z_{2} = z_{2} (x, y)  \\
    &\Sigma_{1}: z_{1} = z_{1} (x, y) 
\end{align*}
$$

- 斯托克斯公式：矢量面积、矢量线积

Condition:

1. 分片光滑有向曲面 $\Sigma$，其边界 $\partial \Sigma^{+}$ 表示方向和 $\Sigma$ 满足右手螺旋法则的分片光滑有向曲线
2. $P, Q, R$ 在 $\Sigma, \partial \Sigma$ 上有连续一阶偏导数

$$
\begin{aligned}
    & \iint_{\Sigma} (\nabla \times \boldsymbol{E}) \cdot  \mathrm{d} \boldsymbol{S} = \oint_{\partial \Sigma^{+}} \boldsymbol{E} \cdot  \mathrm{d} \boldsymbol{l} \\
    & (\nabla \times \boldsymbol{E}) \cdot  \mathrm{d} \boldsymbol{S} =  \mathrm{det} \begin{bmatrix}
        \frac{\partial}{\partial x} & P &  \mathrm{d}y  \mathrm{d}z \\
        \frac{\partial}{\partial y} & Q &  \mathrm{d}z  \mathrm{d}x \\
        \frac{\partial}{\partial z} & R &  \mathrm{d}x  \mathrm{d}y
    \end{bmatrix}
\end{aligned}
$$

---

两类曲面积分之间的联系

计算方法一：分面投影法，计算三次二重积分

计算方法二：（转化为第一类曲面积分）

$$
\begin{align*}
    & \iint P \mathrm{d}y \mathrm{dz} + Q \mathrm{d}z \mathrm{d}x + R \mathrm{d}x \mathrm{d}y \\
    & = \iint P \cos \alpha \mathrm{d} S \\
    & + \iint Q \cos \beta \mathrm{d} S \\
    & + \iint R \cos \gamma \mathrm{d} S
\end{align*}
$$

$(\cos \alpha, \cos \beta, \cos \gamma)$ 是曲面在 $(x, y, x)$ 处与面指向一致的法向量。

$$
\begin{aligned}
    & \begin{bmatrix}
        \cos \alpha \\
        \cos \beta \\
        \cos \gamma \\
    \end{bmatrix} = k \begin{bmatrix} 
        1 \\
        0 \\
        z_{x} \\
    \end{bmatrix} \times \begin{bmatrix}
        0 \\
        1 \\
        z_{y}
    \end{bmatrix} = k \begin{bmatrix}
        -z_{x} \\
        -z_{y} \\
        1
    \end{bmatrix}
\end{aligned}
$$

$$
\left\{\begin{align*}
    & \cos \alpha = \mp \frac{z_{x}}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
    & \cos \beta = \mp \frac{z_{y}}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
    & \cos \gamma = \pm \frac{1}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
\end{align*}\right.
$$

取第一个符号表示平面向上，否则表示平面向下。

计算方法三：合一投影法，把三元积分转化成一个二元积分
$$
\iint P \mathrm{d}y \mathrm{d}z + Q \mathrm{d}z \mathrm{d}x + R \mathrm{d}x \mathrm{d}y = \iint [P(- z_{x}) + Q(- z_{y}) + R] \mathrm{d}x \mathrm{d}y
$$

example:

$$
\begin{align*}
    & \Sigma = \{(x, y, z): z = \frac{1}{2} (x^{2} + y^{2}), z \in [0, 2]\} \\
    & ? \iint_{\Sigma} (z^{2} + x) \mathrm{d}y \mathrm{d}z - z \mathrm{d}x \mathrm{d}y
\end{align*}
$$

$$
\begin{align*}
    & \iint_{\Sigma} (z^{2} + x) \mathrm{d}y \mathrm{d}z - z \mathrm{d}x \mathrm{d}y \\
    & = \iint_{\Sigma} [(z^{2} + x)(-x) -z] \mathrm{d}x \mathrm{d}y \\
    & = -\iint_{\Sigma} [([\frac{1}{2} (x^{2} + y^{2})]^{2} + x)(-x) -\frac{1}{2} (x^{2} + y^{2})] \mathrm{d}x \mathrm{d}y \\
\end{align*}
$$

负号因为积分的曲面是下侧。

---

曲线积分是否与路径有关？

**单连通区域** $D$ 中和路径无关时的等价条件：

1. 和路径无关
2. 任意分段光滑闭曲线积分为 0
3. $\forall (x, y) \in D, \frac{\partial P}{\partial y} = \frac{\partial Q}{\partial x}$
4. $\exists \text{ continuous } u(x, y) \text{ on } D, u_{x} = P, u_{y} = Q,  \mathrm{d} u = P  \mathrm{d} x + Q  \mathrm{d} y$

注意此处 $D$ 一定要是单连通区域，区域内 $P, Q$ 一阶偏导数处处连续，不能有极点。

---

### 无穷级数

---

级数收敛的必要条件

$$
\sum_{n=1}^{\infty} u_{n} < \infty \Rightarrow \lim_{n \rightarrow \infty} u_{n} = 0
$$

Proof:

$$
\lim u_{n} = \lim (s_{n+1} - s_{n}) = \lim s_{n+1} - \lim s_{n} = 0
$$

---

调和级数不收敛的证明

$$
\begin{align*}
    & s_{n} := \sum_{i=1}^{n} \frac{1}{i} \\
    & s_{2n} - s_{n} = \sum_{i=1}^{n} \frac{1}{n+i} > \sum_{i=1}^{n} \frac{1}{2n} > \frac{1}{2} \\
    & \lim_{n \rightarrow \infty} (s_{2n} - s_{n}) > \frac{1}{2}
\end{align*}
$$

收敛的级数子列应当也收敛，而调和级数子列不满足收敛级数的必要条件。因此调和级数不收敛。

Q.E.D

---

柯西审敛原理：充要条件

$$
\begin{align*}
    &\forall \varepsilon > 0, \exists N \in \mathbb{N}, \forall n > N, \forall p \in \mathbb{N}_{+}, \left| \sum_{i=1}^{p} u_{n+i} \right| < \varepsilon
\end{align*}
$$

证明用到了 Cauchy 审敛原理。

---

Fibbonacci Array: $a_{0} = a_{1} = 1$

$$
\begin{align*}
    &\sum_{k=1}^{\infty} \frac{1}{a_{k-1}a_{k+1}} \\
    & = \sum_{k=1}^{\infty} \frac{1}{a_{k}}\cdot \left(\frac{1}{a_{k-1}}- \frac{1}{a_{k+1}}\right) \\
    & = \sum_{k=1}^{\infty} \left(\frac{1}{a_{k-1}a_{k}} - \frac{1}{a_{k}a_{k+1}}\right) \\
    &= \lim_{k \rightarrow \infty} \left(\frac{1}{a_{0}a_{1}}- \frac{1}{a_{k}a_{k+1}}\right) \\
    & = 1 \\
    &\sum_{k=1}^{\infty} \frac{a_{k}}{a_{k-1}a_{k+1}} \\
    & = \sum_{k=1}^{\infty} \left(\frac{1}{a_{k-1}} - \frac{1}{a_{k+1}}\right) \\
    &= \lim_{k=1} \left(\frac{1}{a_{0}} + \frac{1}{a_{1}} - \frac{1}{a_{k}} - \frac{1}{a_{k+1}}\right) \\
    &= 2
\end{align*}
$$

---

正向级数审敛法

正向级数：$\forall n \in \mathbb{N}, a_{n} \geq 0$

==正项级数==收敛的充要条件：部分和数列有界。

证明使用单调有界数列必有极限。

---

证明下面的级数收敛

$$
\begin{align*}
    \sum_{k=1}^{\infty} \frac{1}{k^{p}} (p > 1)
\end{align*}
$$

方法：

- 积分放缩
- 代数放缩

---

$$
\begin{align*}
    & \frac{1}{k^{p}} \leq \int_{k-1}^{k} \frac{1}{x^{p}} \mathrm{d}x \\
    & S_{n} \leq 1 + \int_{1}^{k} \frac{1}{x^{p}}\mathrm{d}x = \frac{1}{1-p} (k^{1-p} - 1^{1-p}) = \frac{k^{1-p}-1}{1-p}
\end{align*}
$$

部分和数列有界所以收敛。

---

$$
\begin{align*}
    & n \leq 2^{n} - 1 \Rightarrow S_{n} \leq S_{2^{n}-1} \\
    & S_{2^{n}-1} - S_{2^{n-1}-1} = \frac{1}{(2^{n-1})^{p}} + \frac{1}{(2^{n-1} + 1)^{p}} + \cdots + \frac{1}{(2^{n}-1)^{p}} \\
    & \leq \frac{2^{n-1}}{(2^{n-1})^{p}} = \frac{1}{(2^{n-1})^{p-1}} = \frac{1}{t^{n-1}}
\end{align*}
$$

其中令 $t = 2^{p-1} > 1$

$$
\begin{align*}
    & \sum_{k=1}^{n} S_{2^{k}-1} - S_{2^{k-1}-1} \leq \sum_{k=1}^{n} \frac{1}{t^{k-1}} \\
    & S_{2^{n}-1} - S_{0} \leq \frac{1 - \frac{1}{t^{n}}}{1 - \frac{1}{t}} \\
    & S_{n} \leq S_{2^{n}-1} \leq \frac{1 - \frac{1}{t^{n}}}{1 - \frac{1}{t}} \\
\end{align*}
$$

部分和数列有界所以收敛。

---

比较审敛法：放缩

推论放松了条件：

1. 只要当 $n > N$ 时满足
2. 可以比较两个级数项数之比的值的范围

可以将这两个条件整合成极限的形式

$$
\lim_{n \rightarrow \infty} \frac{u_{n}}{v_{n}}= l \in (0, \infty)
$$

- 如果 $l = 0, \sum v_{n}< \infty \Rightarrow \sum u_{n} < \infty$
- 如果 $l > 0 \Rightarrow \sum v_{n}, \sum u_{n}$ 具有相同的敛散性
- 如果 $l = +\infty, \sum v_{n} = \infty \Rightarrow \sum u_{n} = \infty$

---

比值审敛法：d'Alembert

和复变函数[ComplexFunction](ComplexFunction.md#Cauchy-Hardmard formula)中的结论类似

$$
\lim_{n \rightarrow \infty} \frac{u_{n+1}}{u_{n}} = \rho
$$

- $\rho < 1 \Rightarrow \sum u_{n} < \infty$
- $\rho > 1 \Rightarrow \sum u_{n} = \infty$

证明方法就是和等比级数比较应用比较审敛法。

---

根值审敛法：Cauchy

和复变函数[ComplexFunction](ComplexFunction.md#Cauchy-Hardmard formula)中的结论类似

$$
\lim_{n \rightarrow \infty} \sqrt[n]{a_{n}} = \rho
$$

- $\rho < 1 \Rightarrow \sum u_{n} < \infty$
- $\rho > 1 \Rightarrow \sum u_{n} = \infty$

---

极限审敛法

$$
\begin{align*}
    & \lim_{n \rightarrow \infty} nu_{n} = l > 0 \Rightarrow \sum u_{n} = \infty \\
    & \lim_{n \rightarrow \infty} n^{p} u_{n} = l, p > 1 \Rightarrow \sum u_{n} < \infty
\end{align*}
$$

证明方法利用了之前证明的级数。

---

交错级数审敛法

交错级数指的是各项正负交错的级数，可以通过正项级数 $u_{n}$ 来表示

$$
\sum_{n=1}^{\infty} (-1)^{n-1} u_{n}
$$

Lebniz 定理

Condition:

- $u_{n} \geq u_{n+1}$
- $\lim_{n \rightarrow \infty} u_{n} = 0$

Conclusion:

- 级数收敛
- $s \leq u_{1}$
- $r_{n} := s - s_{n}, |r_{n}| \leq u_{n+1}$

---

幂级数的和函数的性质

$$
s(x) = \sum_{n=1}^{\infty} a_{n} x^{n}
$$

1. $s(x)$ 在收敛域上连续。
2. $s(x)$ 在收敛域上可积，而且可以交换积分和求和的顺序。

$$
\int_{0}^{x} s(x) \mathrm{d}x = \sum_{n=1}^{\infty} a_{n} \int_{0}^{x} x^{n} \mathrm{d}x = \sum_{n=1}^{\infty} \frac{a_{n}}{n+1} x^{n+1}
$$

积分函数的收敛域和原来的函数的相同。

并不是所有的函数都可以交换积分和求和的顺序。

3. 也可以交换求导和求和的顺序。

$$
\frac{\mathrm{d}}{\mathrm{d}x} s(x) = \sum_{n=1}^{\infty} na_{n} x^{n-1}
$$

收敛半径也是不变的。

---

## Exercise

---

### 曲面积分和曲线积分练习

---

I

已知

$$
\Gamma:  \left\{ \begin{aligned}
    & x^{2}+y^{2}+z^{2}= \frac{9}{2} \\
    & x + z = 1
\end{aligned}\right.
$$

求

$$
\begin{aligned}
    I &:= \int_{\Gamma} (x^{2} + y^{2} + z^{2})  \mathrm{d} s
\end{aligned}
$$

*Sol*:

如何统一变量？参数方程

$$
\begin{aligned}
    & x^{2} + y^{2} + (1-x)^{2} = \frac{9}{2} \\
    & \frac{\left(x- \frac{1}{2}\right)^{2}}{2} + \frac{y^{2}}{4} = 1 \\
    &  \left\{ \begin{aligned}
        & x = \frac{1}{2} + \sqrt{2} \cos t \\
        & y = 2 \sin t \\
        & z = \frac{1}{2} - \sqrt{2} \cos t
        \end{aligned} \right. && t \in [0, 2\pi) \\
\end{aligned}
$$

$$
\begin{aligned}
    I & = \int_{0}^{2\pi}  \left[  \left( \frac{1}{2} + \sqrt{2} \cos t \right)^{2} + (2\sin t)^{2} +  \left( \frac{1}{2} - \sqrt{2} \cos t \right)^{2} \right] \sqrt{(\sqrt{2} \sin t)^{2} + (2 \cos t)^{2} + (\sqrt{2} \sin t)^{2}}  \mathrm{d} t \\
    & = \frac{9}{2} \int_{0}^{2\pi} 2  \mathrm{d} t \\
    & = 18 \pi
\end{aligned}
$$

***Attention***:

或者直接看出原积分只需要计算出曲线 $\Gamma$ 的长度就可以了。

---

II 单连通区域

已知路径 $L$ 是抛物线 $y = 2-2x^{2}$ 上从 $A(-1, 0)$ 到 $B(1,0)$ 的弧线。求：

$$
\begin{aligned}
    &I = \int_{L} \frac{(x-y) \mathrm{d}x + (x+y) \mathrm{d} y}{x^{2}+y^{2}}
\end{aligned}
$$

*Sol*:

$$
\begin{aligned}
    & Q_{x} - P_{y} = 0
\end{aligned}
$$

构造以原点为圆心半径为 1 的圆

$$
\begin{aligned}
    & I = \int_{L} = \int_{\overset{\LARGE{\frown}}{AB}} \\
\end{aligned}
$$

参数方程

$$
 \left\{ \begin{aligned}
    & x = - \cos \theta \\
    & y = \sin \theta
\end{aligned} \right.
$$

$$
\begin{aligned}
    I&= \int_{0}^{\pi} (- \cos \theta - \sin \theta) \sin \theta  \mathrm{d} \theta + (\sin \theta - \cos \theta) \cos \theta  \mathrm{d} \theta \\
    &= - \pi
\end{aligned}
$$

***Attention***:

一定要注意，单连通区域内部 $P, Q$ 一定要存在，所以不能直接

$$
\begin{aligned}
    & \int_{L} = \int_{ \overline{AB}}
\end{aligned}
$$

因为原点是极点。

---

III

平面曲线

$$
C_{r} : x^{2} + xy + y^{2} = r^{2}
$$

逆时针方向的环路积分

$$
I_{a}(r) = \oint_{C_{r}} \frac{y \mathrm{d} x - x \mathrm{d} y}{(x^{2} + y^{2})^{a}}
$$

求：

$$
\lim_{r \rightarrow \infty} I_{a}(r)
$$

*Sol*:

$$
\begin{aligned}
\begin{bmatrix}
\xi \\
\eta
\end{bmatrix} &= \begin{bmatrix}
\cos \theta & \sin \theta \\
- \sin \theta & \cos \theta
\end{bmatrix} \cdot \begin{bmatrix}
x\\
y
\end{bmatrix} \\
\begin{bmatrix}
x\\
y
\end{bmatrix} &= \begin{bmatrix}
\cos \theta  & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix} \cdot \begin{bmatrix}
\xi \\
\eta
\end{bmatrix} \\
r^{2} &= (\xi \cos \theta - \eta \sin \theta)^{2} \\
&+ (\xi \cos \theta - \eta \sin \theta)(\xi \sin \theta + \eta \cos \theta) \\
&+ (\xi \sin \theta + \eta \cos \theta)^{2} \\
&= (1 + \cos \theta \sin \theta)\xi^{2} + (1 - \cos \theta \sin \theta) \eta^{2} + \xi \eta \cos 2 \theta \\
\end{aligned}
$$

令交叉项为 0，$\theta = \dfrac{\pi}{4}$

$$
\begin{aligned}
r^{2} &= \frac{3}{2} \xi^{2} + \frac{1}{2} \eta^{2} \\
& \left\{ \begin{aligned}
\xi &= \sqrt{\frac{2}{3}} r \cos \alpha \\
\eta &= \sqrt{2} r \sin \alpha
\end{aligned} \right. && \alpha \in [0, \pi) \\
& \left\{ \begin{aligned}
x &= \sqrt{\frac{1}{3}} r \cos \alpha - r \sin \alpha \\
y &= \sqrt{\frac{1}{3}} r \cos \alpha + r \sin \alpha
\end{aligned} \right.
\end{aligned}
$$

$$
\begin{aligned}
I_{a}(r) &= \lim_{r \rightarrow \infty} \int_{0}^{2\pi} \frac{- \frac{2}{\sqrt{3}} r^{2} \mathrm{d} \alpha}{r^{2a}(\frac{2}{3} \cos^{2} \alpha + 2 \sin^{2} \alpha)^{a}}
\end{aligned}
$$

- 当 $a > 1$ 时，$I_{a}(r) = 0$
- 当 $a < 1$ 时，$I_{a}(r) \rightarrow \infty$
- 当 $a = 1$ 时

$$
\begin{aligned}
I_{a}(r) &= \int_{0}^{2\pi} - \frac{1}{\sqrt{3}} \frac{\mathrm{d} \alpha}{1 - \frac{2}{3} \cos^{2} \alpha} \\
&= - \frac{1}{\sqrt{3}} \int_{0}^{2\pi} \frac{\mathrm{d} \tan \alpha}{\frac{1}{3} + \tan^{2} \alpha} \\
&= - (3 - \sqrt{3}) \pi
\end{aligned}
$$

---

### 无穷级数练习
