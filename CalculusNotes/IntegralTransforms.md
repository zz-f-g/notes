# Integral Transforms

A Integral Transform is a **mapping** which maps a function $f(x)$ to another function $g(x)$:
$$
g(x) = \int_a^b K(x, y) f(y) \mathrm{d} y
$$
通过黎曼和离散化
$$
g(x_j) = \int_a^b K(x_i, y) f(y) \mathrm{d} y = \sum_{k=0}^{N-1} K(x_i, y_k) f(y_k) \Delta y_k
$$
是矩阵—向量相乘的形式。因此可以将积分变换理解为有限维度的矩阵向量相乘的形式。

## Fourier Transform

$$
\hat{f}(w) =  \int_{-\infty}^{\infty} f(x) e^{-i \omega x} \mathrm{d} x
$$

Transform $f(x)$ to $\hat{f}(\omega)$.

注意：$\mathrm{max} \{ \left|\mathrm{Re} f(x) e^{-i \omega t}\right|, \left|\mathrm{Im} f(x) e^{-i \omega} t\right| \} \leq \left|f(x)\right|$，因此仅当满足下面条件时，傅里叶变换才有意义。
$$
\int_{-\infty}^{\infty} \left|f(x)\right| \mathrm{d} x < \infty
$$
此时称 $f(x)$ 为**绝对可积**。

Multivariate case:
$$
\hat{f}(\vec{\omega}) = F(f)(\vec{\omega}) = \int_{R^d} f(\vec{x}) e^{-i \vec{\omega} \cdot \vec{x}} \mathrm{d}x_1 \mathrm{d}x_2 \dots \mathrm{d}x_n
$$
Properties:

- Linearity
- Translation

$$
F(f(x-x_0))(\omega) = e^{-i x_0 \omega} F(f)(w)
$$

- Modulation

$$
F(e^{i \omega_0 x} f(x))(\omega) = \hat{f}(\omega - \omega_0)
$$

- Scaling

$$
F(f(a^{-1} x))(\omega) = |a| \hat{f}(a \omega)
$$
换元证明时，需要注意：
$$
y = \frac{x}{a}, \mathrm{d}y = \left| a \right|^{-1} \mathrm{d}x
$$

- Derivative in $x$

$$
F(f^{(n)}(x))(\omega) = (i \omega)^n \hat{f}(\omega)
$$

CONDITION:

- $f, f',f^{(2)}, \cdots, f^{(n)}$ are absolutely integrable.

PROOF:
$$
\begin{align}
F(f'(x))(\omega) &= \int_{-\infty}^{\infty} f'(x) e^{-i \omega x} \mathrm{d}x \\
&= f(x)e^{-i \omega x}|_{-\infty}^{\infty} + i \omega \int_{-\infty}^{\infty} f(x) e^{- i \omega x} \mathrm{d}x \\
&= i \omega \hat{f}(\omega)
\end{align}
$$
对于高阶导数，递归即可。注意到绝对可积包含着函数在无穷远点处极限为 0 的条件，因此 $\left. f(x)e^{-i \omega x} \right|^{\infty}_{\infty} = 0$.

- Derivative in $\omega$

$$
\frac{\mathrm{d}^n \hat{f}}{\mathrm{d} \omega^n} = F((-ix)^n f(x))
$$

CONDITION:

- $x^i f(n), i= 0,1,2,\cdots, n$ are absolutely integrable.

PROOF:
$$
\begin{align}
\frac{\mathrm{d} \hat{f}}{\mathrm{d} \omega} &= \frac{\mathrm{d}}{\mathrm{d} \omega} \int_{-\infty}^{\infty} f(x) e^{-i \omega x} \mathrm{d}x \\
&= \int_{-\infty}^{\infty} f(x) \frac{\partial}{\partial \omega}(e^{- i \omega x}) \mathrm{d} x \\
&= \int_{-\infty}^{\infty} - i x f(x) e^{- i \omega x} \mathrm{d}x \\
&= F((-ix)f(x))(\omega)
\end{align}
$$

- Integration

$$
F \left[ \int_{-\infty}^x f(y) \mathrm{d} y \right] = \frac{1}{i \omega} \hat{f}(\omega) + \pi \hat{f}(0) \delta(\omega)
$$
PROOF in [Dirac delta](#Dirac-Delta)

Example: find $F(f(x) = e^{-\frac{x^2}{2}})(\omega)$
$$
\begin{align}
\frac{\mathrm{d} \hat{f}}{\mathrm{d} w} &= \int_{-\infty}^{\infty} - i x f(x) e^{- i \omega x} \mathrm{d}x = i F(f') = - \omega \hat{f} \\
\hat{f} &= C e^{-\frac{\omega^2}{2}} \\
C &= \hat{f}(0) = \int_{-\infty}^{\infty} f(x) \mathrm{d}x = \int_{-\infty}^{\infty} e^{-\frac{x^2}{2}} \mathrm{d}x = \sqrt{2 \pi} \\
\hat{f} &= \sqrt{2 \pi} e^{-\frac{\omega^2}{2}}
\end{align}
$$

==首先根据傅里叶变换后函数的导数关系构建变换后函数的微分方程，再根据 $\omega=0$ 的处置条件求出常量。==

Convolution
$$
h*f(x) = \int_{-\infty}^{\infty} h(x-y) f(y) \mathrm{d} y
$$
卷积满足和乘法相似的规律

- 交换律
- 结合律
- 分配律
- translation invariance

$$
\tau_z f(x) = f(x-z) \\
(\tau_z h)*f = h*(\tau_z f) = \tau_z(h*f)
$$

Convolution Theorem
$$
F(h*f) = F(h)F(f)
$$

PROOF:
$$
\begin{align*}
F(h*f) &= \int_{-\infty}^{\infty} e^{-i \omega x} \left[\int_{-\infty}^{\infty} h(x-y)f(y)\mathrm{d} y \right] \mathrm{d}x \\
&= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-i \omega (x - y)} h(x-y) e^{-i \omega y} f(y) \mathrm{d}x \mathrm{d} y \\
&= \int_{-\infty}^{\infty} e^{-i \omega y} f(y) \left[ \int_{-\infty}^{\infty} h(x - y) e^{-i \omega (x-y)} \mathrm{d} x \right] \mathrm{d} y \\
&= F(h) \int_{-\infty}^{\infty} e^{-i \omega y} f(y) \mathrm{d} y \\
&= F(h) F(f)
\end{align*}
$$

## Inverse Fourier Transform

$$
F^{-1}(f) = \frac{1}{2 \pi} \int_{-\infty}^{\infty} f(\omega) e^{i \omega x} \mathrm{d} \omega = \frac{1}{2 \pi} \hat{f}(-x)
$$

## Dirac Delta

Heaviside Function
$$
\begin{equation}
v(x-a) = H(x-a) = \left\{
\begin{array}{rlc}
1 && x \geq a \\
0 && x < a
\end{array}
\right.
\end{equation}
$$

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
\begin{equation}
\delta(x-a) = \left\{
\begin{array}{rlc}
\infty && x = a \\
0 && x \neq a
\end{array}
\right.
\end{equation}
$$
Sifting property（筛选性质）:
$$
\int_{-\infty}^{\infty} \delta(x-a) \varphi(x) \mathrm{d} x = \varphi(a)
$$
PROOF:
$$
\begin{array}{rlc}
\int_{-\infty}^{\infty} f_k(x-a) \varphi(x) \mathrm{d} x &= \frac{1}{k} \int_{a}^{a+k} \varphi(x) \mathrm{d} x \\
&= \frac{\Phi(a+k) - \Phi(a)}{k} \\
\int_{-\infty}^{\infty} \delta(x - a) \phi(x) \mathrm{d} x &= \lim_{k = 0} \frac{\Phi(a+k) - \Phi(a)}{k} \\
&= \Phi'(a) \\
&= \varphi(a)
\end{array}
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
\delta(x) = \frac{1}{2 \pi} \int_{-\infty}^{\infty} e^{i \omega x} \mathrm{d} \omega = \frac{1}{2 \pi} \int_{-\infty}^{\infty} e^{-i \omega x} \mathrm{d} \omega \\
\hat{1}(\omega) = \int_{-\infty}^{\infty} e^{-i \omega x} \mathrm{d} x = 2 \pi \delta(\omega)
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
