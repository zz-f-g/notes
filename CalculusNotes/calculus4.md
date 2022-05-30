# Calculus4

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

