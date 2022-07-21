# Calculus4

## 两类曲面积分之间的联系

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
\left\{\begin{align*}
    & \cos \alpha = \mp \frac{z_{x}}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
    & \cos \beta = \mp \frac{z_{y}}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
    & \cos \gamma = \pm \frac{1}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
\end{align*}\right.
$$

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

高斯公式

将第二类曲面积分转化成三重积分

闭空间区域 $\Omega$ 由分片光滑闭曲面 $\Sigma$ 围成，函数具有一阶连续偏导数。
$$
\iiint_{\Omega} \left(\frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}\right)\mathrm{d}v = \oint_{\Sigma} P \mathrm{d}y \mathrm{d}z + Q \mathrm{d}z \mathrm{d}x + R \mathrm{d}x \mathrm{d}y
$$

直观证明：考虑一个轴向和 $z$ 轴平行的柱状物上下曲面为
$$
\begin{align*}
    &\Sigma_{2}: z_{2} = z_{2} (x, y)  \\
    &\Sigma_{1}: z_{1} = z_{1} (x, y) 
\end{align*}
$$

---

## 级数收敛条件

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

## 正向级数审敛法

---

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

## 交错级数审敛法

---

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
