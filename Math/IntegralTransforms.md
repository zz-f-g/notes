# Integral Transforms

A Integral Transform is a **mapping** which maps a function $f(x)$ to another function $g(x)$:

$$
g(x) = \int_a^b K(x, y) f(y) \mathrm{d} y
$$

通过黎曼和离散化

$$
g(x_j) = \int_a^b K(x_j, y) f(y) \mathrm{d} y \approx \sum_{k=0}^{N-1} K(x_j, y_k) f(y_k) \Delta y_k
$$

是矩阵—向量相乘的形式。因此可以将积分变换理解为有限维度的矩阵向量相乘的形式。

## Fourier_Transform

### FT_Def

$$
\hat{f}(w) =  \int_{-\infty}^{\infty} f(x) e^{-i \omega x} \mathrm{d} x
$$

Transform $f(x)$ to $\hat{f}(\omega)$.

注意：$\mathrm{max} \{ \left|\Re f(x) e^{-i \omega t}\right|, \left|\Im f(x) e^{-i \omega t} \right| \} \leq \left|f(x)\right|$，因此仅当满足下面条件时，傅里叶变换才有意义。

$$
\int_{-\infty}^{\infty} \left|f(x)\right| \mathrm{d} x < \infty
$$

此时称 $f(x)$ 为**绝对可积**。

Multivariate case:

$$
\begin{aligned}
    & f: \mathbb{R}^{d} \rightarrow \mathbb{C} \\
    & \boldsymbol{\omega} = \begin{bmatrix} \omega_{1} & \omega_{2} & \cdots & \omega_{n} \end{bmatrix}^{T} \in \mathbb{R}^{d} \\
    & \hat{f}(\boldsymbol{\omega}) = \mathscr{F}(f)(\boldsymbol{\omega}) = \int_{R^d} f(\boldsymbol{x}) e^{-i \boldsymbol{\omega} \cdot \boldsymbol{x}} \mathrm{d}x_1 \mathrm{d}x_2 \dots \mathrm{d}x_n
\end{aligned}
$$

### FT_Properties

- Linearity
- Translation

$$
\mathscr{F}(f(x-x_0))(\omega) = e^{-i x_0 \omega} \mathscr{F}(f)(w)
$$

- Modulation

$$
\mathscr{F}(e^{i \omega_0 x} f(x))(\omega) = \hat{f}(\omega - \omega_0)
$$

- Scaling

$$
\mathscr{F}(f(a^{-1} x))(\omega) = |a| \hat{f}(a \omega)
$$

换元证明时，需要注意：上下限

- Derivative in $x$

$$
\mathscr{F}(f^{(n)}(x))(\omega) = (i \omega)^n \hat{f}(\omega)
$$

CONDITION:

$f, f',f^{(2)}, \cdots, f^{(n)}$ are absolutely integrable.

PROOF:

$$
\begin{align}
\mathscr{F}(f'(x))(\omega) &= \int_{-\infty}^{\infty} f'(x) e^{-i \omega x} \mathrm{d}x \\
&= f(x)e^{-i \omega x}|_{-\infty}^{\infty} + i \omega \int_{-\infty}^{\infty} f(x) e^{- i \omega x} \mathrm{d}x \\
&= i \omega \hat{f}(\omega)
\end{align}
$$

对于高阶导数，递归即可。注意到绝对可积包含着函数在无穷远点处极限为 0 的条件，因此

$$
\left[ f(x) e^{-i \omega x} \right]_{-\infty}^{\infty} = 0
$$

- Derivative in $\omega$

$$
\frac{\mathrm{d}^n \hat{f}}{\mathrm{d} \omega^n} = \mathscr{F}((-ix)^n f(x))
$$

CONDITION:

$x^i f(n), i= 0,1,2,\cdots, n$ are absolutely integrable.

PROOF:

$$
\begin{align}
\frac{\mathrm{d} \hat{f}}{\mathrm{d} \omega} &= \frac{\mathrm{d}}{\mathrm{d} \omega} \int_{-\infty}^{\infty} f(x) e^{-i \omega x} \mathrm{d}x \\
&= \int_{-\infty}^{\infty} f(x) \frac{\partial}{\partial \omega}(e^{- i \omega x}) \mathrm{d} x \\
&= \int_{-\infty}^{\infty} - i x f(x) e^{- i \omega x} \mathrm{d}x \\
&= \mathscr{F}((-ix)f(x))(\omega)
\end{align}
$$

- Integration

$$
\mathscr{F} \left[ \int_{-\infty}^x f(y) \mathrm{d} y \right] = \frac{1}{i \omega} \hat{f}(\omega) + \pi \hat{f}(0) \delta(\omega)
$$

PROOF in [Heaviside Function](#Heaviside_Step).

Example:

$$
\mathscr{F}\left(f(x) = e^{-\frac{x^{2}}{2}}\right)(\omega)
$$

$$
\begin{align}
\frac{\mathrm{d} \hat{f}}{\mathrm{d} w} &= \int_{-\infty}^{\infty} - i x f(x) e^{- i \omega x} \mathrm{d}x = i \mathscr{F}(f') = - \omega \hat{f} \\
\hat{f} &= C e^{-\frac{\omega^2}{2}} \\
C &= \hat{f}(0) = \int_{-\infty}^{\infty} f(x) \mathrm{d}x = \int_{-\infty}^{\infty} e^{-\frac{x^2}{2}} \mathrm{d}x = \sqrt{2 \pi} \\
\hat{f} &= \sqrt{2 \pi} e^{-\frac{\omega^2}{2}}
\end{align}
$$

首先根据傅里叶变换后函数的导数关系构建变换后函数的微分方程，再根据 $\omega=0$ 的处置条件求出常量。当然也可以参考 [Gauss Integration](ComplexFunction.md#Gauss_integral) 的内容。

### Convolution

$$
h*f(x) = \int_{-\infty}^{\infty} h(x-y) f(y) \mathrm{d} y
$$

卷积满足和乘法相似的规律

- 交换律
- 结合律
- 分配律
- translation invariance

$$
\begin{aligned}
    & \tau_{z}: \tau_{z} f(x) = f(x-z) \\
    & (\tau_{z} h)*f = h*(\tau_{z} f) = \tau_{z}(h*f)
\end{aligned}
$$

Convolution Theorem

$$
\mathscr{F}(h*f) = \mathscr{F}(h)\mathscr{F}(f)
$$

PROOF:

$$
\begin{align*}
\mathscr{F}(h*f) &= \int_{-\infty}^{\infty} e^{-i \omega x} \left[\int_{-\infty}^{\infty} h(x-y)f(y)\mathrm{d} y \right] \mathrm{d}x \\
&= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-i \omega (x - y)} h(x-y) e^{-i \omega y} f(y) \mathrm{d}x \mathrm{d} y \\
&= \int_{-\infty}^{\infty} e^{-i \omega y} f(y) \left[ \int_{-\infty}^{\infty} h(x - y) e^{-i \omega (x-y)} \mathrm{d} x \right] \mathrm{d} y \\
&= \mathscr{F}(h) \int_{-\infty}^{\infty} e^{-i \omega y} f(y) \mathrm{d} y \\
&= \mathscr{F}(h) \mathscr{F}(f)
\end{align*}
$$

### IFT

$$
\mathscr{F}^{-1}(f)(x) = \frac{1}{2 \pi} \int_{-\infty}^{\infty} f(\omega) e^{i \omega x} \mathrm{d} \omega = \frac{1}{2 \pi} \hat{f}(-x)
$$

如何理解这个形式正是 IFT？

$$
\begin{aligned}
    f& : \mathbb{R} \rightarrow \mathbb{R} \\
    f_{L}(x) & = \left\{ \begin{aligned}
        & f(x) && -L < x \leq L \\
        & f(x - L) && x > L \\
        & f(x + L) && x < -L
    \end{aligned} \right.\\
    f(x) &= \lim_{L \rightarrow \infty} f_{L}(x)
\end{aligned}
$$

根据周期信号的 FT

$$
\begin{aligned}
    f_{L}(x) &= \sum_{n \in \mathbb{Z}} c_{n} e^{i \omega_{n}x} \\
    c_{n} &= \frac{1}{2L} \int_{-L}^{L} f_{L}(y) e^{-i \omega_{n}y} \mathrm{d} y \\
    \omega_{n} &= n \cdot \frac{\pi}{L} \\
    & L \rightarrow \infty \Rightarrow c_{n} \rightarrow 0 \\
    \Delta \omega&:= \lim_{L \rightarrow \infty} \frac{\pi}{L} \\
    c_{n} &= \frac{\Delta \omega}{2\pi} \int_{-\infty}^{\infty} f_{L}(y) e^{-i \omega_{n} y} \mathrm{d} y \\
    f(x) &:= \sum_{n \in \mathbb{Z}} \frac{1}{2\pi} \left[ \int_{-\infty}^{\infty} f(y) e^{-i \omega y} \mathrm{d} y \right] e^{i \omega x}\Delta \omega \\
    &= \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\omega) e^{i \omega x} \mathrm{d} \omega
\end{aligned}
$$

在周期函数的 FT 中：

$$
\begin{aligned}
    \frac{f(x^{+}) + f(x^{-})}{2} = \sum_{n=-\infty}^{\infty} c_{n} e^{i \frac{n \pi x}{L}}
\end{aligned}
$$

得到反演定理。

### Fourier_Inverse_Theorem

CONDITION:

$f(x)$ 逐段连续可导且

$$
\begin{aligned}
    & \int_{-\infty}^{\infty} |f(x)| \mathrm{d} x < \infty
\end{aligned}
$$

CONCLUSION:

$$
\begin{aligned}
    &  \mathscr{F}^{-1}\left[  \mathscr{F}(f) \right](x) = \frac{f(x^{+}) + f(x^{-})}{2}
\end{aligned}
$$

## Dirac_Delta

$$
f_k(x - a) = \left\{
\begin{aligned}
& \frac{1}{k} && (a \leq x \leq a+k) \\
& 0 && \text{else}
\end{aligned}
\right.
$$

The Dirac Delta or Short Impulse:

$$
\delta(x-a) = \lim_{k \rightarrow 0} f_k(x-a)
$$

With the property:

$$
\int_{-\infty}^{\infty} \delta(x - a) \mathrm{d} x = 1
$$

Dirac delta function is **not** a function in the ordinary sense, but a so-called generalized function.

$$
\delta(x-a) = \left\{
\begin{aligned}
& \infty && (x = a) \\
& 0 && (x \neq a)
\end{aligned}
\right.
$$

### Dirac_Delta_Properties

Sifting property（筛选性质）:

$$
\int_{-\infty}^{\infty} \delta(x-a) \varphi(x) \mathrm{d} x = \varphi(a)
$$

PROOF:
$$
\begin{aligned}
    \int_{-\infty}^{\infty} f_k(x-a) \varphi(x) \mathrm{d} x &= \frac{1}{k} \int_{a}^{a+k} \varphi(x) \mathrm{d} x \\
    &= \frac{\Phi(a+k) - \Phi(a)}{k} \\
    \int_{-\infty}^{\infty} \delta(x - a) \phi(x) \mathrm{d} x &= \lim_{k = 0} \frac{\Phi(a+k) - \Phi(a)}{k} \\
    &= \Phi'(a) \\
    &= \varphi(a)
\end{aligned}
$$

$$
f(x) \delta(x - a) = f(a) \delta(x - a)
$$

Dirac Delta Function activates the information at $x = a$.

$$
\hat{\delta}(x - a) = \int_{-\infty}^{\infty} \delta(x) e^{-i \omega x} \mathrm{d} x = 1
$$

$$
F(\delta(x - a))(\omega) = e^{- i \omega a}
$$

According to Fourier Inversion Theorem:

$$
\begin{aligned}
& \delta(x) = \frac{1}{2 \pi} \int_{-\infty}^{\infty} e^{i \omega x} \mathrm{d} \omega = \frac{1}{2 \pi} \int_{-\infty}^{\infty} e^{-i \omega x} \mathrm{d} \omega \\
& \hat{1}(\omega) = \int_{-\infty}^{\infty} e^{-i \omega x} \mathrm{d} x = 2 \pi \delta(\omega)
\end{aligned}
$$

identity of convolution

$$
\delta * f(x) = f(x)
$$

上面引入 Dirac delta 是使用了函数

$$
f_{k}(x) = \left\{ \begin{aligned}
    & \frac{1}{k} && a < x \leq a + k \\
    & 0 && \text{else}
\end{aligned} \right.
$$

### General_Function

事实上，对于任意满足绝对可积条件的函数 $\varphi(x)$，都可以作为引入

$$
\begin{aligned}
    c &:= \int_{-\infty}^{\infty} \varphi(x) \mathrm{d} x = \hat{\varphi}(0) \neq 0 \\
    \varphi_{a}(x) &:= \frac{1}{a} \varphi\left(\frac{x}{a}\right) && (a > 0) \\
    \hat{\varphi}_{a}(\omega) &:= \int_{-\infty}^{\infty} \frac{1}{a} \varphi\left(\frac{x}{a}\right) e^{-i \omega x} \mathrm{d}x \\
        &= \int_{-\infty}^{\infty} \varphi(y) e^{-i a \omega y} \mathrm{d} y && \left(y = \frac{x}{a}\right) \\
        &= \hat{\varphi}(a \omega) \\
    \hat{\varphi}_{a}(0) &= \hat{\varphi}(0) = c \\
    \delta(x) & := \frac{1}{c} \lim_{a \rightarrow 0^{+}} \hat{\varphi}_{a}(x)
\end{aligned}
$$

直观感觉上讲，$a \rightarrow 0$ 的过程就是 $\varphi(x)$ 变窄变高的过程。

### Heaviside_Step

$$
u(x-a) = H(x-a) = \left\{
\begin{aligned}
1 && (x \geq a) \\
0 && (x < a)
\end{aligned}
\right.
$$

FT of heaviside function

$$
\begin{aligned}
    u(x) & = \lim_{a \rightarrow 0^{+}} e^{-ax} u(x) \\
     \mathscr{F}[e^{-ax}u(x)] &= \int_{0}^{\infty} e^{-ax} e^{-i \omega x} \mathrm{d} x = \frac{1}{a + i \omega} = \frac{a}{a^{2}+\omega^{2}} - i \frac{\omega}{a^{2}+\omega^{2}} \\
    \hat{u}(\omega) & = \lim_{a \rightarrow 0^{+}} \mathscr{F}[e^{-ax} u(x)] = \lim_{a \rightarrow 0^{+}} \left(\frac{a}{a^{2}+\omega^{2}}\right) + \frac{1}{i \omega}
\end{aligned}
$$

下面分析

$$
\begin{aligned}
    \xi(\omega) & := \lim_{a \rightarrow 0^{+}} \left(\frac{a}{a^{2}+\omega^{2}}\right) \\
    \xi(0) & \rightarrow +\infty \\
    \xi(\omega) &= 0 && (\omega \neq 0)\\
    \int_{-\infty}^{\infty} \xi(\omega) \mathrm{d} \omega &= \int_{-\infty}^{\infty} \frac{\mathrm{d} x}{1 + x^{2}} = \pi \\
    \Rightarrow \xi(\omega) &= \pi \delta(\omega)
\end{aligned}
$$

这一点和[上面](#General_Function)相似。最终得到：

$$
\hat{u}(\omega) = \pi \delta(\omega) + \frac{1}{i \omega}
$$

从这个性质可以证明 FT [积分性质](#FT_Properties)

$$
\begin{aligned}
    \int_{-\infty}^{x} f(y) \mathrm{d} y & = \int_{-\infty}^{\infty} u(x-y) f(y) \mathrm{d} y = u(x) * f(x) \\
    \mathscr{F}\left[\int_{-\infty}^{x}f(y) \mathrm{d} y\right] &= \hat{u}(\omega) \hat{f}(\omega) = \frac{\hat{f}(\omega)}{i \omega} + \pi \hat{f}(\omega) \delta(\omega) \\
        &= \frac{\hat{f}(\omega)}{i \omega} + \pi \hat{f}(0) \delta(\omega)
\end{aligned}
$$

### Discrete_Revolution

离散型和周期性的内在联系

Poisson Summation Formula

$$
\sum_{k \in \mathbb{Z}} f(x + kT) = \frac{1}{T} \sum_{n \in \mathbb{Z}} \hat{f} \left(\frac{2\pi n}{T}\right) e^{i \frac{2 \pi n}{T} x}
$$

PROOF:

$$
\begin{aligned}
    g(x) :& = \sum_{k \in \mathbb{Z}} f(x + kT) \\
    g(x + T) &= g(x) \\
    \Rightarrow g(x) &= \sum_{n \in \mathbb{Z}} c_{n} e^{i \frac{2\pi n}{T} x} \\
    c_{n} &= \frac{1}{T} \int_{- \frac{T}{2}}^{\frac{T}{2}} g(x) e^{-i \frac{2\pi n}{T} x} \mathrm{d} x \\
        &= \frac{1}{T} \sum_{k \in \mathbb{Z}} \int_{- \frac{T}{2}}^{\frac{T}{2}} f(x + kT) e^{-i \frac{2\pi n}{T} x} \mathrm{d} x \\
        &= \frac{1}{T} \int_{-\infty}^{\infty} f(x) e^{-i \frac{2\pi n}{T}x} \mathrm{d} x \\
        &= \frac{1}{T} \hat{f}\left(\frac{2\pi n}{T}\right)
\end{aligned}
$$

Dirac comb

$$
\begin{aligned}
    & c_{T}(x) = \sum_{k \in \mathbb{Z}} \delta(x - k T) \\
    & \hat{c}_{T}(\omega) = \sum_{k \in \mathbb{Z}} e^{-ikT \omega}
\end{aligned}
$$

对于上一片的结论

$$
\sum_{k \in \mathbb{Z}} f(x - kT) = \frac{1}{T} \sum_{n \in \mathbb{Z}} \hat{f}\left(\frac{2\pi n}{T}\right) e^{i \frac{2\pi n}{T} x}
$$

两边同时 FT

$$
\begin{aligned}
    \sum_{k \in \mathbb{Z}} \hat{f}(\omega) e^{-i kT \omega} &= \frac{1}{T} \sum_{n \in \mathbb{Z}} \hat{f}\left(\frac{2\pi n}{T}\right) 2\pi \delta\left(\omega - \frac{2\pi n}{T}\right) \\
&= \frac{2\pi}{T} \sum_{n \in \mathbb{Z}} \hat{f}(\omega) \delta\left(\omega - \frac{2\pi n}{T}\right) \\
    \Rightarrow \sum_{k \in \mathbb{Z}} e^{-i k T \omega} &= \frac{2\pi}{T} \sum_{n \in \mathbb{Z}} \delta\left(\omega - \frac{2\pi n}{T}\right) \\
    \hat{c}_{T}(\omega) & = \frac{2\pi}{T} \sum_{n \in \mathbb{Z}} \delta\left(\omega - \frac{2 \pi n}{T}\right)
\end{aligned}
$$

这个结论说明了 Dirac comb 的频谱仍然是 Dirac comb.

**Sampling** with Dirac Comb

$$
\begin{aligned}
    f_{d}(x) :&= f(x) c_{T}(x) = \sum_{k \in \mathbb{Z}} f(kT) \delta(x - kT)
\end{aligned}
$$

FT of $f_{d}(x)$

$$
\begin{aligned}
    \hat{f}_{d}(\omega) :&= \mathscr{F} \left[ \sum_{k \in \mathbb{Z}}f(kT) \delta(x - kT) \right] \\
        &= \sum_{k \in \mathbb{Z}} f(kT) e^{-ikT \omega}
\end{aligned}
$$

另一方面，利用时域乘积等于频域卷积：

$$
\begin{aligned}
    \mathscr{F}^{-1} (\hat{f}*\hat{g}) &= \frac{1}{2\pi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \hat{f}(\omega - \Omega) \hat{g}(\Omega) \mathrm{d} \Omega e^{i \omega x} \mathrm{d} \omega \\
        &= \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\omega - \Omega) e^{i (\omega - \Omega) x} \mathrm{d} (\omega - \Omega) \int_{-\infty}^{\infty} \hat{g}(\Omega) e^{i \Omega x} \mathrm{d} \Omega \\
        &= 2\pi f(x) g(x) \\
\end{aligned}
$$

得到

$$
\begin{aligned}
    \hat{f}_{d}(\omega) :&= \frac{1}{2\pi} \hat{f}(\omega) * \hat{c}_{T}(\omega) \\
        &= \frac{1}{T} \left[ \hat{f}(\omega) * \sum_{k \in \mathbb{Z}}\delta\left(\omega - \frac{2k\pi}{T}\right) \right] \\
        &= \frac{1}{T} \sum_{k \in \mathbb{Z}} \hat{f}\left(\omega - \frac{2\pi k}{T}\right) 
\end{aligned}
$$

总结

$$
\hat{f}_{d}(\omega) = \sum_{k \in \mathbb{Z}} f(kT) e^{-i kT \omega} = \frac{1}{T} \sum_{k \in \mathbb{Z}} \hat{f}\left(\omega - \frac{2\pi k}{T}\right)
$$

换句话说，通过间隔采样，让频谱图具有了周期性。加入理想选频网络

$$
1_{\left[- \frac{\pi}{T}, \frac{\pi}{T}\right]}(\omega) = \left\{ \begin{aligned}
    & 1 && \left(|\omega| \leq \frac{\pi}{T}\right) \\
    & 0 && \left( |\omega| > \frac{\pi}{T} \right)
\end{aligned} \right.
$$

作为系统传递函数，就可以恢复出原来的函数

$$
\begin{aligned}
    & \hat{f}_{d}(\omega) 1_{\left[ - \frac{\pi}{T}, \frac{\pi}{T} \right]}(\omega) = \frac{1}{T} \hat{f}(\omega)
\end{aligned}
$$

据此再做变换

$$
\begin{aligned}
    & \hat{f}(\omega) = T \hat{f}_{d}(\omega) 1_{\left[ - \frac{\pi}{T}, \frac{\pi}{T} \right]} \\
    & f(x) = T \mathscr{F}^{-1} \left\{ \hat{f}_{d}(\omega) 1_{\left[ - \frac{\pi}{T}, \frac{\pi}{T} \right]}(\omega) \right\}
\end{aligned}
$$

## Laplace Transformation

For $f(t)$ that $\forall t \geq 0, \exists f(t)$, define **Laplace Transformation** as:
$$
F(s) = L(f)(s) = \int_0^{\infty} e^{-st} f(t) \mathrm{d} t
$$
Conversely,
$$
f(t) = L^{-1} (F)(t)
$$
CONDITIONS:

- $\mathrm{Re}(s) > 0 \Rightarrow |F(s)| = |\int_0^{\infty} e^{-st} f(t) \mathrm{d} t| \leq \int_0^{\infty} |f(t)| \mathrm{d} t$
- 上面条件利用不充分，实际上，当 $|f(t)| \leq M e^{kt} \ (s > k)$ 时，

$$
\left| F(s) \right| \leq \int_0^{\infty} e^{-st} \left| f(t) \right| \mathrm{d} t \leq M \int_0^{\infty} e^{(k-s)t} \mathrm{d} t = \frac{M}{s-k} < \infty
$$

EXAMPLE:

- $L(1)(s > 0) = \frac{1}{s}$
- $L(e^{at})(s > a) = \int_0^{\infty} e^{(a-s)t} \mathrm{d} t = \frac{1}{s - a}$

PROPERTIES:

- Linearity: $L(af + bg) = aL(f) + bL(g)$
- s-Shifting: $F(s) = L(f(t)) \Rightarrow L(e^{at} f(t)) = F(s - a)$

Hyperbolic Fuctions:
$$
\begin{align}
L(\cosh at) &= \frac{1}{2} L(e^{at}) + \frac{1}{2} L(e^{-at}) \\
&= \frac{1}{2} \left( \frac{1}{s - a} + \frac{1}{s + a} \right) \\
&= \frac{s}{s^2 - a^2} \\
L(\sinh at) &= \frac{a}{s^2 - a^2}
\end{align}
$$
Trigonometric Functions:
$$
\begin{align}
L_c &= L(\cos \omega t) \\
&= \int_0^{\infty}e^{-st} \cos \omega t \mathrm{d}t \\
&= \left. -\frac{e^{st}}{s} \cos \omega t\right|^{\infty}_0 - \frac{\omega}{s} \int_0^{\infty} e^{-st} \sin \omega t \mathrm{d}t \\
&= \frac{1}{s} - \frac{\omega}{s} L_s\\
L_s &= L(\sin \omega t) \\
&= \frac{\omega}{s} L_c
\end{align}
$$
To solve the equations:
$$
L_c = \frac{s}{s^2 + \omega^2} \\
L_s = \frac{\omega}{s^2 + \omega^2}
$$
 三角函数和指数函数的统一性质：
$$
L(e^{i \omega t}) = \frac{1}{s - i \omega} = \frac{s}{s^2 + \omega^2} + i \frac{\omega}{s^2 + \omega^2}
$$
三角函数和双曲函数的统一性质：
$$
\begin{align}
\cos \omega t &= \cosh i \omega t \Rightarrow &L(\cos \omega t) &= \frac{s}{s^2 - (i \omega)^2} = \frac{s}{s^2 + \omega^2} \\
\sin \omega t &= -i \sinh i \omega t \Rightarrow &L(\sin \omega t) &= -i \frac{i \omega}{s^2 - (i \omega)^2} = \frac{\omega}{s^2 + \omega^2}
\end{align}
$$
Basic Transform

|      $f(t)$       |           $L(f)(s)$           |
| :---------------: | :---------------------------: |
|        $1$        |         $\frac{1}{s}$         |
|        $t$        |        $\frac{1}{s^2}$        |
| $t^n \ (n \in N)$ |     $\frac{n!}{s^{n+1}}$      |
|   $t^a \ (a>0$)   | $\frac{\Gamma(a+1)}{s^{a+1}}$ |
|$e^{at}$|$\frac{1}{s-a}$|
|$\cos \omega t$|$\frac{s}{s^2 + \omega^2}$|
|$\sin \omega t$|$\frac{\omega}{s^2 + \omega^2}$|
|$\cosh at$|$\frac{s}{s^2 - a^2}$|
|$\sinh at$|$\frac{a}{s^2 - a^2}$|
|$e^{at} \cos \omega t$|$\frac{s-a}{(s-a)^2 + \omega^2}$|
|$e^{at} \sin \omega t$|$\frac{\omega}{(s-a)^2 + \omega^2}$|

数学归纳法求整数幂函数 Laplace Transformation：

Assume that:
$$
L(t^n) = \frac{n!}{s^{n+1}}
$$
Then:
$$
L(t^{n+1}) = \int_0^{\infty} e^{-st} t^{n+1} \mathrm{d}t = - \frac{e^{-st}}{s} \left. t^{n+2} \right|^{\infty}_0 + \frac{n+1}{s} \int_0^{\infty} e^{-st} t^n \mathrm{d}t = \frac{(n+1)!}{s^{n+2}}
$$
Done!

对于正数幂函数，首先定义：
$$
\Gamma(\nu) = \int_0^{\infty} t^{\nu-1} e^{-t} \mathrm{d}t \ (\nu > 0)
$$
Then:
$$
L(t^a) = \int_0^{\infty} e^{-st} t^a \mathrm{d}t = \int_0^{\infty} e^{-x} \left(\frac{x}{s} \right)^a \frac{1}{s} \mathrm{d}x = \frac{\Gamma(a+1)}{s^{a+1}}
$$
事实上，$\Gamma(a+1) = a! \ (a \in Z)$.

Derivations and Integrations:
$$
\begin{align}
L(f')(s) &= s L(f)(s) - f(0) \\
L(f^{(n)})(s) &= s^n L(f)(s) - \sum_{i=1}^{n} s^{n-i} f^{(i-1)}(0)
\end{align}
$$
PROOF:
$$
\begin{align}
L(f') &= \int_0^{\infty} e^{-st} f'(t) \mathrm{d}t \\
&= \left. e^{-st} f(t) \right|^{\infty}_0 + s \int_0^{\infty} e^{-st} f(t) \mathrm{d}t \\
&= s L(f) - f(0)
\end{align}
$$
其他阶数可以使用数学归纳法普遍证明。

重新应用于三角函数：
$$
\begin{align}
\omega L_c &= s L_s \\
\omega L_s &= - sL_c + 1
\end{align}
$$
得到了相同的二元一次方程组。

EXAMPLE:

求 $f(t) = t \sin \omega t$ 的 Laplace 变换。

$$
L \left( \int_0^{t} f(\tau) \mathrm{d} \tau \right) = \frac{1}{s} L \left( f(t) \right) \\
L^{-1}\left(\frac{1}{s} L(f(t)) \right) = \int_0^{\tau} f(\tau) \mathrm{d} \tau
$$
EXAMPLE:

求解 $L^{-1}(\frac{1}{s^2(s^2 + \omega^2)})$.

方法：使用三次上式。
$$
\begin{align}
L^{-1} \left( \frac{1}{s^2(s^2 + \omega^2)} \right) &= \int_0^t L^{-1} \left( \frac{1}{s(s^2 + \omega^2)} \right) \mathrm{d}t \\
&= \int_0^t \int_0^{\tau} \frac{\sin \omega \tau}{\omega} \mathrm{d}\tau \\
&= \frac{1}{\omega^2} \int_0^t (1 - \cos \omega t) \mathrm{d}t \\
&= \frac{t}{\omega^2} - \frac{\sin \omega t}{\omega^3}
\end{align}
$$

Heaviside function
$$
\begin{equation}
u(t-a) = \left\{
\begin{array}{rlc}
1 && (t>a) \\
0 && (t<a)
\end{array}
\right.
\end{equation}
$$
$$
L(u(t-a)) = \int_a^{\infty} e^{-st} \mathrm{d}t = \frac{e^{-sa}}{s}
$$
Heaviside 阶跃函数可以用来描述：

- 开机
- 关机
- 延迟开机

t-shifting(Compared with s_shifting: $L(e^{at} f(t)) = F(s-a)$)
$F(s) = L(f)$
$$
L[f(t-a)u(t-a)] = e^{-sa} F(s) \iff L^{-1}(e^{-sa}F(s)) = f(t-a) u(t-a)
$$

求下面函数的 Laplace 变换：
$$
\begin{equation}
f(t) = \left\{
\begin{array}{rlc}
2 && (0<t<1) \\
\frac{t^2}{2} &&(1<t<\frac{\pi}{2}) \\
\cos t && (t > \frac{\pi}{2})
\end{array}
\right.
\end{equation}
$$
$$
\begin{align*}
f(t) &= 2[u(t)-u(t-1)] + \frac{t^2}{2} [u(t-1)-u(t-\frac{\pi}{2})] + \cos t u(t - \frac{\pi}{2}) \\
&= 2u(t) - 2u(t-1) + \frac{1}{2} u(t-1) + (t-1) u(t-1) +\frac{(t-1)^2}{2} u(t-1) \\
&- \frac{\pi^2}{8} u(t-\frac{\pi}{2}) - \frac{\pi}{2} (t - \frac{\pi}{2}) u(t - \frac{\pi}{2}) - \frac{(t - \frac{\pi}{2})^2}{2} u(t - \frac{\pi}{2}) \\
&- \sin (t - \frac{\pi}{2}) u(t-\frac{\pi}{2}) \\
F(s)&= \frac{2}{s} - \frac{2 e^{-s}}{s} + \frac{e^{-s}}{2s} + \frac{e^{-s}}{s^2} + \frac{e^{-s}}{s^3} - \frac{\pi^2 e^{-\frac{\pi}{2}s}}{8s} - \frac{\pi e^{-\frac{\pi}{2}s}}{2s^2} - \frac{e^{-\frac{\pi}{2}s}}{s^3} - \frac{e^{-\frac{\pi}{2}s}}{1 + s^2}
\end{align*}
$$

求 Laplace 逆变换：
$$
F(s) = \frac{e^{-s}}{s^2 + \pi^2} + \frac{e^{-2s}}{s^2 + \pi^2} + \frac{e^{-3s}}{(s+2)^2}
$$

Dirac delta 的 Laplace 变换：
$$
\begin{equation}
f_k(x - a) = \left\{
\begin{array}{rlc}
\frac{1}{k} && a \leq x \leq a+k \\
0 && \text{otherwise}
\end{array}
\right.
\end{equation}
$$
$$
\begin{align*}
L^{-1}\left(\frac{1}{s^2 + \pi^2}\right) &= \frac{1}{\pi} \sin \pi t \\
L^{-1} \left(\frac{1}{(s+2)^2}\right) &= e^{-2t} t \\
f(t) &= \frac{1}{\pi} \sin \pi (t-1) \cdot u(t-1) \\
&+ \frac{1}{\pi} \sin \pi (t-2) \cdot u(t-2) \\
&+ e^{-2(t-3)}(t-3) u(t-3)
\end{align*}
$$

The Dirac Delta or Short Impulse:
$$
\begin{align}
\delta(t-a) &= \lim_{k \rightarrow 0} f_k(x-a) \\
&=\lim_{k \rightarrow 0} \frac{1}{k} \left(u(t-a) - u(t - a - k)\right) \\
L(\delta(t-a)) &= \lim_{k \rightarrow 0} \frac{1}{k} \frac{e^{-sa} - e^{-s(a+k)}}{s} \\
&= \lim_{k \rightarrow 0} e^{-sa} \frac{1 - e^{-sk}}{sk} \\
&= e^{-sa}
\end{align}
$$

卷积与 Laplace 变换：

Causal Convolution

因果性指的是卷积中的积分只依赖于过去的状态。
$$
g(t) = (h*f)(t) = \int_0^t h(\tau)f(t - \tau) \mathrm{d}\tau
$$
和原来定义的卷积有积分上下限的区别。

Causal Convolution Thereom

拉普拉斯变换的因果卷积等于因果卷积的拉普拉斯变换。

PROOF:
$$
\begin{align*}
G(s) &= \int_0^{\infty} e^{-st} \int_0^t h(\tau) f(t - \tau) \mathrm{d} \tau \mathrm{d} t \\
&= \int_0^{\infty} \int_0^t e^{-s \tau} h(\tau) e^{-s(t - \tau)} f(t - \tau) \mathrm{d} \tau \mathrm{d} t \\
&= \int_0^{\infty} e^{-s \tau} h(\tau) \mathrm{d} \tau \int_0^{\infty} e^{-s (t - \tau)} f(t - \tau) \mathrm{d}(t - \tau) \\
&=H(s)F(s)
\end{align*}
$$
这是一种求解拉普拉斯逆变换的方法
$$
L^{-1}(G(s)) = L^{-1}(F(s)) *L^{-1}(H(s)) = \int_0^t L^{-1}(F(s))(\tau) \cdot L^{-1}(H(s))(t - \tau) \mathrm{d} \tau
$$
求解 $L^{-1}(\frac{1}{s(s - a)})$

方法一：
$$
\begin{align*}
L^{-1}(\frac{1}{s(s - a)}) &= \frac{1}{a} L^{-1} (\frac{1}{s-a} - \frac{1}{s}) \\
&= \frac{e^{at}}{a} - \frac{1}{a}
\end{align*}
$$
方法二：
$$
L^{-1}(\frac{1}{s(s - a)}) = e^{at}*1 = \int_0^t e^{a\tau} \mathrm{d} \tau = \frac{e^{at} -1}{a}
$$

Inverse Laplace 变换与导数
$$
F(s) = L(f(t))(s) \Rightarrow F'(s) = -L(tf(t))(s) \iff L^{-1}(F'(s)) = -tf(t)
$$
这样就又得到了一种求 Inverse Laplace Transformation 的方法
$$
L^{-1}(F(s)) = - \frac{1}{s} L^{-1}(F'(s))
$$

同样的，对于积分：（注意上下限）
$$
\begin{align*}
L^{-1}(\int_s^{\infty} F(\sigma) \mathrm{d} \sigma) &= L^{-1}(\int_s^{\infty}\int_0^{\infty}e^{-\sigma t}f(t) \mathrm{d}t \mathrm{d}\sigma) \\
&= L^{-1}(\int_0^{\infty} \frac{e^{-st}}{t} f(t) \mathrm{d}t) \\
&= \frac{f(t)}{t}
\end{align*}
$$

## Laplace Transform in ODE

Consider an initial value problem:
$$
y'' + ay' + by = r(t) \ y(0) = K_0 \ y'(0) = K_1
$$
$r(t)$ is the driving forces.

To solve the ODE by Laplace Transformation:

- Set up the subsidiary equation.

$$
s^2 L(y) - s y(0) - y'(0) + a (sL(y) - y(0)) + b L(y) = L(r(t))
$$

- Solve the equation with $L(y)$

$$
\begin{align*}
(s^2 + as + b)L(y) &= L(r(t)) + (s+a)y(0) + y'(0) \\
L(y) &= \frac{L(r(t)) + (s+a)y(0) + y'(0)}{(s^2 + as + b)}
\end{align*}
$$

- Inverse Laplace Transformation

$$
y = L^{-1}(L(y))
$$

A Volterra Integral of the second kind

$$
y(t) + \int_0^t y(\tau) k(t - \tau)\mathrm{d} \tau = r(t)
$$

*Example 1*
$$
y(t) - \int_0^t y(\tau) \sin(t - \tau)\mathrm{d} \tau = t \\
Y(s) - Y(s) \frac{1}{s^2 + 1} = \frac{1}{s^2} \\
Y(s) = \frac{1}{s^2} + \frac{1}{s^4} \\
y(t) = L^{-1}(Y) = t + \frac{t^3}{6}
$$

*Example 2*
$$
y(t) - \int_0^t (1+ \tau) y(t- \tau) \mathrm{d} \tau = 1 - \sinh t \\
Y(s) - (\frac{1}{s} + \frac{1}{s^2}) Y(s) = \frac{1}{s} - \frac{1}{s^2 - 1} \\
Y(s) = \frac{s}{s^2 - 1} \\
y(t) = \cosh t
$$

Linear ODEs with variable coefficients

不考

System of ODEs

$$
\frac{\mathrm{d}}{\mathrm{d}t} \vec{y} = A \vec{y} + \vec{g}(t) \\
s \vec{Y} - \vec{y}_0 = A \vec{Y} + \vec{G} \\
(s I - A) \vec{Y} = \vec{y}_0 + \vec{G}
$$

## Fourier Transform in PDE

homogeneous wave equation
$$
u_{tt} - c^2 u_{xx} = 0 \\
u(x, 0) = \phi(x) \\
u_t(x, 0) = \psi(x)
$$

Let $\hat{u}(\omega, t) = F(u(\cdot, t))(\omega)$.
$$
\hat{u}_{tt} + c^2 \omega^2 \hat{u} = 0 \\
\hat{u}(\omega, 0) = \hat{\phi}(\omega) \\
\hat{u}'(\omega, 0) = \hat{\psi}(\omega)
$$

The solution:
$$
\hat{u} = \hat{\phi}(\omega) \cos c \omega t + \frac{\hat{\psi}(\omega)}{c \omega} \sin c \omega t
$$

Inverse Fourier Transform
$$
u(x, t) = \frac{1}{2} \left[\phi(x+ct)-\phi(x-ct)\right] + \frac{1}{2c} \int_{x - ct}^{x+ct} \psi(y) \mathrm{d} y
$$

复变函数

到复积分

到 Laplace Transform 的应用

积分方程

变量系数不考

Fourier Transform 应用不考

Fourier Transform DFT FFT 不考

Commons and Differences between two transfroms

---

Direct Proof by Definition

$$
F(e^{i \omega_0 x} f(x)) = \hat{f}(\omega - \omega_0) \\
L(e^{at} f(t)) = F(s-a)
$$

---

Integration by Steps

$$
F(f^{(n)}(x)) = (i \omega)^n \hat{f}(\omega) \\
L(f^{(n)}(x)) = s^n F(s) - \sum_{i=1}^{n} s^{n-i} f^{(n-1)}(0)
$$

---

Direct Proof by Definition

$$
\hat{f}^{(n)}(\omega) = F((-i x)^n f(x)) \\
F^{(n)}(s) = L((-t)^n f(t)) \\
L^{-1}(F'(s)) = - t f(t)
$$

---

Double Integration

$$
F\left[ \int_{- \infty}^{x} f(y) \mathrm{d}y \right] = \frac{1}{i \omega} \hat{f}(\omega) + \pi \hat{f}(0) \delta(\omega) \\
L\left[ \int_0^t f(\tau) \mathrm{d}\tau \right] = \frac{1}{s} F(s)
$$

---

Convolution: $h*f = \int_{- \infty}^{+\infty} h(x-y)f(y) \mathrm{d}y$

$$
F(h*f) = F(h) \cdot F(f)
$$

Causal Convolution: $h*f = \int_{- \infty}^{x} h(x-y)f(y) \mathrm{d}y$

$$
L(h*f) = H \cdot F
$$
