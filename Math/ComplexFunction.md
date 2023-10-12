# Complex Function

崔宰珪；宁静楼 313；jaycjk@tongji.edu.cn；微信 jaycjk

## Complex Numbers

### Conjugate_Complex_Numbers

$\text{For }z = x + i y,\ \text{the conjugate } \overline{z} = x - i y$.

$$
\begin{align*}
    & \Re z = \frac{1}{2} (z + \overline{z})\\
    & \Im z = \frac{1}{2} (z - \overline{z})\\
    & |z| = \sqrt{z \cdot \overline{z}}\\
    & \overline{z_{1} + z_{2}} = \overline{z_{1}} + \overline{z_{2}}\\
    & \overline{z_{1} z_{2}} = \overline{z_{1}} \overline{z_{2}}\\
    & \overline{\left(\frac{z_{1}}{z_{2}}\right)}=\frac{\overline{z_{1}}}{\overline{z_{2}}}
\end{align*}
$$

### Argument_of_Complex_Numbers

$\theta = \arg z = \arctan {\frac{y}{x}}$

Principle value of argument（辐角主值）

$\theta = \text{Arg } z = \arctan \frac{y}{x}$

$$
\begin{align*}
    & -\pi < \text{Arg} \leq \pi\\
    & \arg z = \text{Arg } z + 2n\pi
\end{align*}
$$

### Triangle_Inequation

$|z_1| + |z_2| \geq |z_1 + z_2|$

By induction,

$$
\sum_{i=1}^n |z_{n}| \geq \left|\sum_{i=1}^{n} z_{n}\right|
$$

### De_Moivre's_Formula

$$
(\cos \theta + i \sin \theta)^n=\cos n \theta + i \sin n \theta
$$

### Root

$$
\begin{align*}
    & \omega ^{n} = z = r e ^{i \theta}\\
    & \omega = \sqrt[n]{r} e ^{i \varphi}\\
\end{align*}
$$
where $n \varphi = \theta +2k\pi,\ k = 0,1,\cdots,n-1$, suppose $w_{1}= \sqrt[n]{r} e ^{\frac{\theta}{n}} ,\ \omega = e ^{\frac{2\pi}{n}}$

Then the nth roots can be written as $w_{1} \omega ^{k}$, where $k = 0,1,\cdots,n-1$, where $\omega$ is one of the nth roots of unity.

$$
\sqrt[n]{1}=e ^{i \cdot \frac{2\pi}{n}}
$$

所有解构成一个 n 边形。

## Complex_function

$$
\begin{aligned}
    & f: \mathbb{C} \rightarrow \mathbb{C}
\end{aligned}
$$

where domain and codomain / range are both complex set.

$z$ is called a **complex variable**.

$$
\begin{align*}
    & w=u+iv \\
    & z=x+iy \\
    & w = f(z) = u(x, y) + i v(x, y)
\end{align*}
$$

Thus we transform complex function to real function.

### Limit_of_Complex_Function

$$
\begin{aligned}
    & \lim_{z \rightarrow z_0}f(z)=l \\
    \iff & \forall \epsilon > 0,\exists \delta > 0, \forall z \in  \left\{ z\in \mathbb{C}:  |z-z_0|<\delta \right\}, |f(z)-l| < \epsilon
\end{aligned}
$$

**Direction is irrelevant.**

### Continuity

1. $\exists f(z_0)$
2. $\lim_{z\rightarrow z_o}f(z)=f(z_0)$

### Derivative

$$
f'(z_0)=\lim_{z\rightarrow z_0} \frac{f(z)-f(z_0)}{z-z_0}
$$

Similarly, **direction is irrelevant.**

二元函数的极限存在，则必然和趋向的方向无关。

***Example***: $f(z)=\overline{z}$
$$
\lim_{z\rightarrow z_0} \frac{f(z)-f(z_0)}{z-z_0}=\lim_{z\rightarrow z_0} \frac{\overline{z}-\overline{z_0}}{z-z_0} = \lim_{\Delta z \rightarrow 0} \frac{\overline{\Delta z}}{\Delta z} = \lim_{(\Delta x,\Delta y) \rightarrow (0,0)} \frac{\Delta x - i \Delta y}{\Delta x + i \Delta y} \text{ isn't existing.}
$$

### Analytic_Function

- Analyticity in Domain $\forall z \in D, \exists f, \exists f'$
- Analyticity at Point $\forall z \in U(z_0, \delta), \exists f, \exists f'.$

Entire Function: analytic in whole complex plane.

一般来说，在某一个点上可导是没有实际意义的。

### Cauchy-Riemann_Equation

Complex Function $w = f(z) = u(x, y) + i v(x, y)$ is analytic, if and only if it satisfies:
$$
\begin{aligned}
    & \frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \\
    & \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} = 0
\end{aligned}
$$
A criterion to test if a function is analytic.

关于其必要性的证明：将复变函数的导数分别按照两种最简单的路径（平行于坐标轴）求极限，实部和虚部对应相等。

$f'(z)$ 存在。

$$
\begin{equation}
\begin{array}{rlc}
f'(z) &= \lim_{\Delta x \rightarrow 0, \Delta y \rightarrow 0} \frac{u(x + \Delta x, y + \Delta y) - u(x, y) + i[v(x + \Delta x, y + \Delta y) - v(x, y)]}{\Delta x + i \Delta y}\\
& = \lim_{\Delta x \rightarrow 0} \left\{ \lim_{\Delta y \rightarrow 0} \frac{u(x + \Delta x, y + \Delta y) - u(x, y) + i[v(x + \Delta x, y + \Delta y) - v(x, y)]}{\Delta x + i \Delta y} \right\} \\
&= \lim_{\Delta x \rightarrow 0} \frac{u(x + \Delta x, y) - u(x, y) + i[v(x + \Delta x, x) - v(x, y)]}{\Delta x} \\
&= u_x + i v_x
\end{array}
\end{equation}
$$

交换求极限的顺序可以得到：

$$
f'(z) = v_y - i u_y
$$

因此：

$$
\left\{ \begin{aligned}
    & u_x = v_y \\
    & v_x = -u_y
\end{aligned} \right.
$$

关于其充分性的证明：

这里，完整的证明需要首先说明：如果 $u_{x}, u_{y}, v_{x}, v_{y}$ 存在，那么 $u, v$ 必然可微，即：

$$
\begin{aligned}
    & \Delta u = u_x \Delta x + u_y \Delta y + \omicron(\left| \Delta z \right|) \\
    & \Delta v = v_x \Delta x + v_y \Delta y + \omicron(\left| \Delta z \right|)
\end{aligned}
$$

根据柯西黎曼方程，设 $a = u_x = v_y, b = v_x = - u_y$

$$
f(z + \Delta z) - f(z) = (a + i b)\Delta z + \omicron(\Delta z)
$$

从而

$$
f'(z) = \lim_{\Delta z \rightarrow 0} \frac{f(z + \Delta z) - f(z)}{\Delta z} = a + i b
$$

证明完毕。

Cauchy-Riemann Equation in Polar System

$$
\begin{aligned}
    & \frac{\partial u}{\partial r} = \frac{1}{r} \frac{\partial v}{\partial \theta} \\
    & \frac{\partial v}{\partial r} = - \frac{1}{r} \frac{\partial u}{\partial \theta}
\end{aligned}
$$

*Prove*:

$$
\begin{aligned}
    & u = u(r, \theta) =  \widetilde{u}(x, y) \\
    &  \left\{ \begin{aligned}
        & x = r \cos \theta \\
        & y = r \sin \theta
    \end{aligned} \right. \\
    u_{r} & = u_{x} \cdot x_{r} + u_{y} \cdot y_{r} \\
        & = u_{x} \cos \theta + u_{y} \sin \theta \\
    v_{\theta} &= v_{x} \cdot x_{\theta} + v_{y} \cdot y_{\theta} \\
        &= -u_{y} \cdot (-r \sin \theta) + u_{x} \cdot (r \cos \theta) \\
        &= r u_{r} \\
    v_{r} &= v_{x} \cdot x_{r} + v_{y} \cdot y_{r} \\
        &= v_{x} \cos \theta + v_{y} \sin \theta \\
    u_{\theta} & = u_{x} \cdot x_{\theta} + u_{y} \cdot y_{\theta} \\
        &= u_{x} (-r \sin \theta) + u_{y} r \cos \theta \\
        &= -r v_{r}
\end{aligned}
$$

### Laplace's_Equation

$$
\begin{aligned}
    & \Delta u = u_{xx} + u_{yy} = 0 \\
    & \Delta v = v_{xx} + v_{yy} = 0
\end{aligned}
$$

证明就把 Cauchy-Riemann 方程分别对 x，y 求偏导。过程需要用到：$u_{xy} = u_{yx}$，即 $u, v$ 都具有一阶连续偏导数。

下面证明极坐标下的 Laplace's Equation:

Cauchy-Riemann Equation in Polar System:

$$
\begin{bmatrix} u_r \\ v_r \end{bmatrix} = \frac{1}{r} \begin{bmatrix} 0&1 \\ -1&0 \end{bmatrix}  \begin{bmatrix} u_{\theta} \\ v_{\theta} \end{bmatrix}
$$

Take the derivation of r:

$$
\begin{bmatrix} u_{rr} \\ v_{rr} \end{bmatrix} = \frac{\partial}{\partial r} \left(\frac{1}{r} \begin{bmatrix} 0&1 \\ -1&0 \end{bmatrix}  \begin{bmatrix} u_{\theta} \\ v_{\theta} \end{bmatrix}\right) = - \frac{1}{r} \begin{bmatrix} u_r \\ v_r \end{bmatrix} + \frac{1}{r} \begin{bmatrix} 0&1 \\ -1&0 \end{bmatrix}  \begin{bmatrix} u_{\theta r} \\ v_{\theta r} \end{bmatrix}
$$

Take the derivation of $\theta$:

$$
\begin{bmatrix} u_{r \theta} \\ v_{r \theta} \end{bmatrix} = \frac{1}{r} \begin{bmatrix} 0&1 \\ -1&0 \end{bmatrix} \begin{bmatrix} u_{\theta \theta} \\ v_{\theta \theta} \end{bmatrix}
$$

$$
\begin{aligned}
u_{r \theta} = u_{\theta r} \\
v_{r \theta} = v_{\theta r}
\end{aligned}
$$

So:

$$
\begin{bmatrix} u_{rr} \\ v_{rr} \end{bmatrix} = -\frac{1}{r} \begin{bmatrix} u_r \\ v_r \end{bmatrix} + \frac{1}{r^2} \begin{bmatrix} -1&0 \\ 0&-1 \end{bmatrix} \begin{bmatrix} u_{\theta \theta} \\ v_{\theta \theta} \end{bmatrix} = -\frac{1}{r} \begin{bmatrix} u_r \\ v_r \end{bmatrix} - \frac{1}{r^2} \begin{bmatrix} u_{\theta \theta} \\ v_{\theta \theta} \end{bmatrix}
$$

### Harmonic_Function

Functions that satisfy Laplace' Equation and have continuous second derivation.

A pair of function that satisfy Cauchy-Riemann Equation is called **Harmonic Conjugate Funcion**, they are real part and imaginary part of an analytic function $w = f(z)$.

如果已知一个函数满足拉普拉斯方程，那么可以找到它的共轭调和函数，其中含有一个任意的常数。例如对于函数 $u = x^2 - y^2 - y$.

$$
v_x = -u_y = 2y + 1
$$

$$
v_y = u_x = 2x
$$

$$
\begin{aligned}
    & v(x,y) = 2xy + x + h(y) \\
    & v_y = 2x + \frac{\partial h(y)}{\partial y}
\end{aligned}
$$

$$
\begin{aligned}
    & \frac{\partial h(y)}{\partial y} = 0 \\
    & h(y) = c \\
    & v(x,y) = 2xy + x + c \\
    & w = u + iv = x^2 - y^2 - y + i(2xy + x + c) = z^2 + iz + ic
\end{aligned}
$$

### Elementary_Functions

#### Expotential_Function

Definition: $e^z = e^x (\cos y + i \sin y)$.

Properties:

- Degeneration to $e^x$ when $y=0$.
- Entire Function
- $(e^z)' = e^z$
- $e^{z_1 + z_2} = e^{z_1} \cdot e^{z_2}$
- $|e^z| = e^x, \mathrm{Arg} e^z = y + 2 \pi n (n \in Z)$
- $e^z = \sum_{i=0}^{\infty} \frac{z^i}{z!}$
- Periodicity: $e^{z + i2 \pi} = e^z$

Thus we can gain the entire information of an expotenial function by its value in $\{z = x + i y ,\ - 2 \pi < y < 2 \pi\}$, which is called **fundamental region**.

Thereom from Periodicity

Suppose two complex number whose imaginary $|\Im z_1 - \Im z_2| < 2 \pi$, then $e^{z_1} \neq e^{z_2}$.

#### Trigonometric_Hyperbolic_function

According to the definition of expotential function:

$$
\begin{aligned}
    & \cos z = \frac{e^{iz} + e^{-iz}}{2} = \cosh iz \\
    & \sin z = \frac{e^{iz} - e^{-iz}}{i2} = -i \sinh iz
\end{aligned}
$$

Properties

$$
\begin{aligned}
& \cos z = \cos x \cosh y - i \sin x \sinh y \\
& \sin z = \sin x \cosh y + i \cos x \sinh y \\
& |\cos z|^2 = \cos^2 x + \sinh^2 y \\
& |\sin z|^2 = \sin^2 x + \sinh^2 y
\end{aligned}
$$

$\sin x, \cos x$ 的零点仍然只存在于实轴上。它们变成了无界函数，值域为全体复数 $C$.

Expand the definition in real ret:
$$
\begin{aligned}
\cosh z = \frac{e^{z} + e^{z}}{2} \\
\sinh z = \frac{e^{z} - e^{z}}{2}
\end{aligned}
$$

Properties

$$
\begin{aligned}
& \cosh iz = \cos z \\
& \sinh iz = i \sin z \\
& \cos iz = \cosh z \\
& \sin iz = \sinh z
\end{aligned}
$$

#### Logarithm

Definition by the inverse of exponential:

$$
\begin{aligned}
& w = \exp{z} \Rightarrow w = \mathrm{Ln} z \\
& \mathrm{Ln} z = \ln |z| + i \mathrm{Arg} z = \ln|z| + \arg z + i 2 \pi n (n \in \ \mathbb{Z})
\end{aligned}
$$
Principle Value:
$$
\ln z = \ln |z| + i \text{arg} z
$$
每一个 n 对应一个分支，主值（$n=0$）称为主分支。 每一个分支构成一个单叶区域（该区域上任意两个不同的复数经过函数映射以后的值不相等），在沿着**原点和负半轴割破的复平面**内解析。

对于通值函数仍然满足实变函数的性质：

$$
\begin{aligned}
& \mathrm{Ln} (z_1 z_2) = \mathrm{Ln} z_1 + \mathrm{Ln} z_2 \\
& \mathrm{Ln} \frac{z_1}{z_2} = \mathrm{Ln} z_1 - \mathrm{Ln} z_2
\end{aligned}
$$

The Analycity of Logarithm

$$
\begin{aligned}
&(x \neq 0, x^{2} + y^{2} \neq 0, n \in \mathbb{Z}) \\
\mathrm{Ln} z &= \ln \sqrt{x^2 + y^2} + i \arctan \frac{y}{x} + i 2 \pi n &  \\
u_x &= \frac{x}{\sqrt{x^2 + y^2}} =v_y \\
u_y &= \frac{y}{\sqrt{x^2 + y^2}} = -v_x
\end{aligned}
$$

The logarithm satisfies the Cauchy-Riemann Equation when $r \neq 0$ and $x \neq 0$.

$$
(\mathrm{Ln} z)' = u_x + i v_x = \frac{x - i y}{\sqrt{x^2 + y^2}} = \frac{1}{z}
$$

The Analycity of Principle Logarithm?

Branch cut: 0 + negative real axis.

#### General_Powers

$$
z^c = e^{c \ln z}
$$

Principle value:

$$
z^c = e^{c \text{Ln} z}
$$

一般来说，对于底数为实数的幂函数，习惯上只考虑它的主值。这样就可以统一之前关于指数函数的唯一定义。

## Complex Integration

### Complex_Line_Integrals

$$
\begin{aligned}
S &= \int_C f(z) \text{ d}z \\
C: z(t) &= x(t) + i y(t) & t \in [a, b]
\end{aligned}
$$

Assume ***C*** to be a smooth curve:

$$
\dot{z} (t) = \frac{\text{d}z}{\text{d}t} = \dot{x}(t) + i \dot{y}(t)
$$

==Continuous and nonzero derivative== at each point.

定义和实变函数的定积分类似，只是把实数轴换成了逐段光滑的平面曲线。

$$
\begin{aligned}
\{t_i\} &= \{t_0 = a, t_1, \cdots, t_{n-1}, t_n = b\} \\
\{z_i\} &= \{z(t_i)\} \\
\{\zeta_i\} &= \{z(t)\} & \forall t \in [t_{i-1}, t_i] \\
\int_C f(z) \mathrm{d} z &= \lim_{n \rightarrow \infty, \Delta z_m \rightarrow 0} \sum_{i=1}^{n} f(\zeta_i) \Delta z_m & \Delta z_m = z(t_m) - z(t_{m-1})
\end{aligned}
$$

Properties implied by definition:

- Linearity $\int_C [k_1 f_1(x) + k_2 f_2(x)] \text{ d}x = k_1 \int_C f_1(x) \text{ d}x + k_2 \int_C f_2(x) \text{ d}x$
- Sense reversal $\int_C f(z) \text{ d}z = - \int_{C^-} f(z) \text{ d}z$
- Partition of Path $\int_{C_1 \cup C_2} f(z) \mathrm{d} z = \int_{C_1} f(z) \mathrm{d} z + \int_{C_2} f(z) \mathrm{d}z$

复数线积分的存在性：

- 被积分函数是连续函数
- 积分路径是光滑曲线

证明方法：将复数积分转化为实数积分

#### ML_Inequation

$$
\left| \int_C f(z) \text{ d}z \right| \leq ML \\
\forall z \text{ on }C,M \geq |f(z)| \\
L = \int_C |z| \text{ d}z \text{ is the length of curve }C
$$

#### Evaluation-Method

- Use a Representation of a Path

$$
\int_C f(z) \text{ d}z = \int_a^b f(z(t)) \dot{z}(t) \text{ d}t
$$

where $z = z(t), \ t \in [a, b] \ f(z)$ is continuous.

求解 $\int_C \frac{\mathrm{d}z}{z}$，其中 $C$ 是单位逆时针圆周。

$$
\begin{align}
\oint_C \frac{\mathrm{d}z}{z} &= \int_0^{2 \pi} \frac{\dot{z} \mathrm{d}\theta}{z} \\
&= i \int_0^{2 \pi} \mathrm{d} \theta \\
&= i 2 \pi
\end{align}
$$

相似的做法得到：

$$
\oint_C (z - z_0)^m \mathrm{d}z = \left\{
\begin{aligned}
& i 2 \pi && (m = -1) \\
& 0 && (m \neq -1 \text{ and } m \in \mathbb{Z})
\end{aligned}
\right.
$$

- Indefinite Integration of Analytic Functions

$f(z)$ is analytic in simply connected domain $D$, there exists an ==indefinite integral== of $f(z)$ in $D$:

$$
\int_{z_0}^{z_1} f(z) \text{ d}z = F(z_1) - F(z_0)
$$

where $F'(z) = f(z)$.

### Cauchy's_Integral_Thereom

本定理给出了关于不定积分法求线积分方法的证明。

#### Simply_Closed_Path

A close Path that does not intersect or touch itself.

$z(t) = z$ 的根的数量小于等于 1.

#### Simply_Connected_Domain

$\forall \text{ simple closed path } C \in D, \forall \text{ points enclosed by } C \in D$.

#### p-fold_connected

由 p 个没有交集的单闭合集合（可以是曲线、线段甚至点）作为边界的有界领域。

#### Content_of_Cauchy's_Integral_Thereom

CONDITION: (Sufficient rather than Necessary)

- $f(z)$ is analytic in ==simply connected domain== D
- C is a simply closed path in D

CONCLUSION:
$$
\oint_C f(z) \text{ d}z = 0
$$
PROOF: (Cauchy's proof assuming that $f'(z)$ is continuous.)

$$
\begin{aligned}
\oint_C f(z) \text{ d}z &= \oint_C (u + i v) (\text{d}x + i  \text{d}y) \\
&= \oint_C (u \text{d}x - v \text{d}y) + i \oint_C (v \text{d}x + u \text{d} y) \\
&= \iint_R(- \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y})\text{d}x \text{d}y + i \iint_R (\frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}) \text{d}x \text{d}y \text{ (by Green's Thereom)} \\
&=0 \text{ (by Cauchy-Riemann Equation)}
\end{aligned}
$$

Goursat's proof without assumption that $f'(z)$ is continuous is far more complicated.

INFERENCE:

- CONDITION: $f(z)$ is analytic in simply connected domain D
- CONCLUSION: the integral of $f(z)$ is independent of path in D

$$
\begin{aligned}
& \oint _C (z - z_0)^m \text{d}z = 0 \ (m \neq -1) \\
& \oint _C \frac{\text{d}z}{z - z_0} = i 2 \pi
\end{aligned}
$$

计算的结果是通过按照参数方程计算单位圆得到的，通过路径变形原理可以得到==对任意包含 $z_0$ 在内的路径==环路积分结果都相等。

#### Cauchy's_Integral_Thereom_for_double_connected_domain

双连通区域由外边界和内边界构成。

![image-20211031104400092](images/ComplexFunction-double-connected.png)

转化成单连通区域

$$
\begin{aligned}
& \oint_{C_1} f(z) \text{d}z + \int_{\widetilde{C}} f(z)\text{d}z + \oint_{C_2^{-1}} f(z)\text{d}z + \int_{\widetilde{C}^-} f(z) \text{d}z = 0 \\
& \oint_{C_1} f(z) \text{d}z = \oint_{C_2} f(z) \text{d}z
\end{aligned}
$$

### Cauchy's_Integral_Formula

CONDITION:

- $f(z)$ is analytic in simply connected domain $D$
- $z_0 \in D$
- Simply close path $C$ encloses $z_0$

CONCLUSION:
$$
\oint_C \frac{f(z)}{z - z_0} \text{d}z = i 2 \pi f(z_0)
$$
where the integration is taken ==counterclockwise==.

PROOF:
$$
\begin{align}
\oint_C \frac{f(z)}{z - z_0} \text{d}z &= f(z_0) \oint_C \frac{\text{d}z}{z - z_0} + \oint_C \frac{f(z) - f(z_0)}{z - z_0} \text{d}z \\
&= i 2 \pi f(z_0) + \oint_K \frac{f(z) - f(z_0)}{z - z_0} \text{d}z
\end{align}
$$
where $K$ is a circle of radius $\rho$ and center $z_0$.

利用了[之前结论](#content_of_cauchy's_integral_thereom)和解析函数线积分路径无关的性质。

$$
\forall \varepsilon > 0, \exists \delta > 0,\forall z \in \{z \in C: \left|z - z_0\right| < \delta\}, \left|f(z) - f(z_0)\right| < \varepsilon \\
$$

choosing $\rho < \delta$:

$$
\left|\frac{f(z) - f(z_0)}{z - z_0}\right| < \frac{\varepsilon}{\rho}
$$

According to [ML-Inequation](#ML_Inequation):

$$
\begin{align}
\oint_K \frac{f(z) - f(z_0)}{z - z_0} \text{d}z &< \frac{\varepsilon}{\rho} \cdot 2 \pi \rho = 2 \pi \varepsilon \\
\oint_K \frac{f(z) - f(z_0)}{z - z_0} \text{d}z &= 0 \\
\oint_C \frac{f(z)}{z - z_0} \text{d}z &= i 2 \pi f(z_0)
\end{align}
$$
证明完毕。

这个定理指出了一个与静电场理论十分相似的性质：

如果两个函数均在有界区域 $D$ 内解析，在 $\overline{D}$ 上连续且 $\forall z \in \partial D, f(z) = g(z)$，则：

$$
\forall z \in D, f(z) = g(z)
$$

PROOF:

$$
\forall z \in D, f(z) = \frac{1}{i 2 \pi} \oint_{\partial D} \frac{f(\zeta)}{\zeta - z} \mathrm{d} \zeta = \frac{1}{i 2 \pi} \oint_{\partial D} \frac{g(\zeta)}{\zeta - z} \mathrm{d} \zeta = g(z)
$$

For ==Multiply Connected domains==, we use similar method as proof in [Cauchy's Integral Thereom](#Cauchy's_Integral_Thereom_for_double_connected_domain).

$$
\oint_{C_{out}} \frac{f(z)}{z - z_0} \text{d}z - \oint_{C_{in}} \frac{f(z)}{z - z_0} \text{d}z = i 2 \pi f(z_0)
$$

![image-20211031140634455](images/ComplexFunction-double-connected-cauchy-integral-formula.png)

### Derivatives_of_Analytic_Functions

#### Relation between Derivatives and Contour Integral

Cauchy Integration Formula proves that ==analytic functions have derivatives of all orders==, which establishes Taylor series representations.

CONDITION:

- $f(z)$ is analytic in domain $D$

CONCLUDION:

- $f(z)$ has derivatives of all orders.（这一点与实变函数很不同）

CONDITION:

- Simply closed path $C$ in domain $D$ encloses $z_0$.
- The interior of $C$ $\subset$ $D$

CONCLUSION:

$$
f^{(n)}(z_0) = \frac{n!}{i 2 \pi} \oint_C \frac{f(z)}{(z - z_0)^{n+1}} \text{d}z
$$

[Cauchy's Integral Formula](#Cauchy's_Integral_Formula) is a unique form of it.

一种记忆方法：

$$
\begin{aligned}
f^{(n)}(z_0) &= \frac{\partial^n}{\partial z_0^n} \left[\frac{1}{i 2 \pi} \oint_C \frac{f(z)}{z - z_0} \mathrm{d}z\right] \\
&= \frac{1}{i 2 \pi} \oint_C \frac{\partial^n}{\partial z_0^n} \frac{f(z)}{z - z_0} \mathrm{d}z \\
&= \frac{n!}{i 2 \pi} \oint_C \frac{f(z)}{(z - z_0)^{n+1}} \mathrm{d}z
\end{aligned}
$$

PROOF: (to 1st order)

$$
\begin{aligned}
\frac{f(z_0 + \Delta z_0) - f(z_0)}{\Delta z_0} &= \frac{1}{i 2 \pi \Delta z_0} \left[\oint_C \frac{f(z)}{z - z_0 - \Delta z_0} \text{d}z - \oint_C \frac{f(z)}{z - z_0} \text{d}z\right] \\
&= \frac{1}{i 2 \pi} \oint \frac{f(z)}{(z - z_0)(z - z_0 - \Delta z_0)} \text{d}z \\
&= \frac{1}{i 2 \pi} \left[\oint_C \frac{f(z)}{(z - z_0)^2} \text{d}z + \oint_C \left(\frac{f(z)}{(z - z_0)(z - z_0 - \Delta z_0)} - \frac{f(z)}{(z - z_0)^2}\right)\text{d}z\right] \\
&= \frac{1}{i 2 \pi} \left[\oint_C \frac{f(z)}{(z - z_0)^2} \text{d}z + \oint_C \frac{f(z) \Delta z_0}{(z - z_0)^2 (z - z_0 - \Delta z_0)}\text{d}z\right]
\end{aligned}
$$

Continuouos $f(z)$ $\Rightarrow$ $\left|f(z)\right| < K$, where $K \in \mathbb{R}$.

随便找一些基准点

$$
\begin{align*}
&2 \Delta z_0 \leq d \leq \left|z - z_0\right| \\
&\frac{1}{\left|z-z_0\right|^2} \leq \frac{1}{d^2} \\
&d \leq \left|z - z_0 - \Delta z_0 + \Delta z_0\right| \leq \left|z - z_0 - \Delta z_0\right| + \left|\Delta z_0\right| \\
&\left|z - z_0 - \Delta z_0\right| \geq d - \Delta z_0 \geq \frac{d}{2} \\
&\oint_C \frac{f(z) \Delta z_0}{(z - z_0)^2 (z - z_0 - \Delta z_0)} \text{d}z \leq \frac{2K}{d^3} \Delta z_0 L \\
&\lim_{\Delta z_0 \rightarrow 0} \oint_C \frac{f(z) \Delta z_0}{(z - z_0)^2 (z - z_0 - \Delta z_0)} \text{d}z = 0
\end{align*}
$$

其中利用了 [ML 不等式](#ML_Inequation)。最终：

$$
f'(z_0) = \lim_{\Delta z_0 \rightarrow 0} \frac{f(z_0 + \Delta z_0) - f(z_0)}{\Delta z_0} = \frac{1}{i 2 \pi} \oint_C \frac{f(z)}{(z - z_0)^2} \text{d}z
$$

以此类推可以得到任意阶导数的公式。$n!$ 来源于微分过程。

$$
\begin{aligned}
    & f''(z_{0}) = \lim_{\Delta z_{0} \rightarrow \infty} \frac{f(z_{0} + 2 \Delta z_{0}) - 2 f(z_{0} + \Delta z_{0}) + f(z_{0})}{\Delta z_{0}^{2}} \\
    & ......
\end{aligned}
$$

#### Cauchy's_Inequality

Choose $C$ as circle with center $z_0$ and radius $r$. Analytic Function $f(z)$ satisfies $f(z) < M, z \in C$. Then apply the ML Inequality to it:

$$
\begin{aligned}
    & f^{(n)}(z_0) = \frac{n!}{i 2 \pi} \oint_C \frac{f(z)}{r^{n+1}} \text{d}z \\
    & \left|f^{(n)}(z_0)\right| \leq \frac{n!}{2 \pi} \left|\frac{M}{r^{n+1}}\right| (2 \pi r)= \frac{n! M}{r^{n}}
\end{aligned}
$$

That is Cauchy's Inequality:

$$
\left|f^{(n)}(z_0)\right| \leq \frac{n! M}{r^{n}}
$$

#### Liouville's_Theorem

CONDITION:

- entire function $f(z)$
- $\forall z \in C, \left|f(z)\right| \leq M$. $f(z)$ is bounded.

CONCLUSION:
$$
f(z) = Const
$$
PROOF:

Apply Cauchy's Inequality to it:

$$
\begin{aligned}
    & \left|f^{(1)}(z_0)\right| \leq \frac{M}{r} \\
    & \forall \varepsilon > 0, \exists r > \frac{M}{\varepsilon}, \forall z_{0} \in \mathbb{C}, |f^{(1)}(z_{0}) | < \varepsilon \\
    \Rightarrow & f^{(1)}(z_{0}) = 0
\end{aligned}
$$

According to $f'(z_0) = u_x + i v_x = u_y - i v_y$ in Cauchy-Riemann Equation.

$$
u_x = v_x = u_y = v_y \Rightarrow u = Const, v = Const \Rightarrow f(z) = Const
$$

#### Morera's_Theoreom

CONDITION:

- $f(z)$ is continuous in $D$
- $\forall C \text{ a closed path } \subset \text{ simple connected domain } D, \oint_C f(z) \mathrm{d}z = 0$

CONCLUSION:

$f(z)$ is analytic.

PROOF:

因为环路积分等于 0，所以从 $z_0$ 到 $z$ 的积分的值和路径无关。
$$
F(z) := \int_{z_0}^{z} f(z) \mathrm{d} z
$$
$F(z)$ is analytic.

According to Relation between Derivatives and Contour Integral, $f(z) = F'(z)$ is analytic.

## Taylor Series Laurent Series

### Convergence

定义和实数级数相同

复数序列收敛，当且仅当，实部序列和虚部序列均收敛。

复数级数收敛，当且仅当，实部级数和虚部级数均收敛。

$$
\lim_{n \rightarrow \infty} z_n = a + i b \iff \lim_{n \rightarrow \infty} x_n = a \text{ and } \lim_{n \rightarrow \infty} y_n = b
$$

PROOF:

Based on the geometric fact that:

$$
\begin{aligned}
    & \mathrm{max} \{ \left|\mathrm{Re}z\right|, \left| \mathrm{Im}z \right| \} \leq \left|z\right| \leq \left|\mathrm{Re}z\right| + \left| \mathrm{Im}z \right| \\
    & \mathrm{max}\left\{ |\Re [z_{n} - (a + ib)]|, |\Im [z_{n} - (a + ib)] \right\} \leq |z_{n} - (a + ib)| \\
    & \forall \varepsilon > 0, \exists N \in \mathbb{N}, \forall n > N, |z_{n} - (a + ib)| < \varepsilon \\
    \Rightarrow & |\Re z_{n} - a| < \varepsilon,  |\Im z_{n} - b| < \varepsilon \\
    \Rightarrow &\left\{ \begin{aligned}
        & \lim_{n \rightarrow \infty} \Re z_{n} = a \\
        & \lim_{n \rightarrow \infty} \Im z_{n} = b
\end{aligned} \right. 
\end{aligned}
$$

An obvious conclusion:

$$
s = \sum_{i=1}^{\infty} z_{i} \text{ is convergent}, \Rightarrow \lim_{n \rightarrow \infty} z_{n} = 0
$$

如果给定的无限级数收敛，则作为序列的极限为 0.（充分条件但不是必要条件）

Cauchy Principle

$$
\begin{aligned}
    & s = \sum_{i=1}^{\infty} z_{i} \text{ is convergent} \\
    \iff & \forall \varepsilon > 0, \exists N, p \in \mathbb{N}, \forall n > N, \left| \sum_{i=1}^{p} z_{n + i} \right| < \varepsilon
\end{aligned}
$$

两种收敛：

- 绝对收敛：绝对值级数收敛。
- 条件收敛：级数收敛但绝对值级数不收敛。

例如交错调和级数：

$$
\sum_{m=1}^{inf} \frac{(-1)^{m}}{m}
$$

#### Comparison_Test

如果给定级数的每一项的绝对值都小于等于已知的收敛级数的对应项，则给定级数收敛。

常用的例子：几何级数

$$
\begin{aligned}
    & z_n = q^n && (|q| < 1) \\
    & \sum_{m=1}^\infty q^n = \frac{q}{1-q} && (|q| < 1) \\
    & \exists N \in N, \forall m > N, \left|\frac{z_{m+1}}{z_m}\right| \leq q < 1 && \Rightarrow \{z_m\} \text{ is absolutely convergent.}
\end{aligned}
$$

#### Ratio_Test

CONDITION:（注意其中 $q$ 的存在，**调和级数**）

- $\exists N \in N, \forall n > N$

$$
\left|\frac{z_{n+1}}{z_n}\right| \leq q < 1
$$

**注意不是** $\left| \frac{z_{n+1}}{z_{n}} \right| < 1$.

CONCLUSION:

- $\{z_n\}$ converge absolutely.

一个便于使用的推论（但它的判别能力不够上述定理强）：

$$
L = \lim_{n \rightarrow \infty} \left|\frac{z_{n+1}}{z_n}\right|
$$

- $L < 1$, absolute converge
- $L > 1$, diverge
- $L = 1$, no conclusion

#### Root_Test

$$
\begin{aligned}
    & \exists N \in N, \forall m > N, \sqrt[m]{z_m} \leq q < 1 \Rightarrow \{z_m\} \text{ is absolutely convergent.} \\
\end{aligned}
$$

$$
L = \lim_{n \rightarrow \infty} \sqrt[n]{z_n}
$$

- $L < 1$, absolute converge
- $L > 1$, diverge
- $L = 1$, no conclusion

### Power_Series

$$
\sum_{n=0}^{\infty} a_n (z - z_0)^n
$$

#### Abel's_Convergence_Theorem_for_Power_Series

- Power series converges at $z_0$
- If power series converges at $z_1$, then $\forall  z \in \{z: \left|z - z_0\right| < \left|z_1 - z_0\right|\}$, the power series converges.
- If power series diverges at $z_2$, then $\forall  z \in \{z: \left|z - z_0\right| > \left|z_2 - z_0\right|\}$, the power series diverges.

This Theorem indicates the existence of **Convergence Radius** $R$.

$$
\begin{aligned}
& \forall z \in \{z: \left|z - z_0\right| < R\}, \sum_{n=0}^{\infty} a_n (z - z_0)^n \text{ converges.} \\
& \forall z \in \{z: \left|z - z_0\right| > R\}, \sum_{n=0}^{\infty} a_n (z - z_0)^n \text{ diverges.}
\end{aligned}
$$

然而，就像和比较判别法和根判别法中一样，在收敛半径上的点无法判断是否收敛。

#### Cauchy-Hardmard_formula

$$
\begin{aligned}
& R = \frac{1}{L^{*}}, L^{*} = \lim_{n \rightarrow \infty} \left|\frac{a_{n+1}}{a_n}\right| \\
& R = \frac{1}{\tilde{L}}, \tilde{L} = \lim_{n \rightarrow \infty} \sqrt[n]{a_n} \\
& R = \frac{1}{\tilde{l}}, \tilde{l} = \lim_{n \rightarrow \infty} \mathrm{sup} \sqrt[n]{a_n}
\end{aligned}
$$

This is a method to get the convergence radius by the coefficients.

其中，$\lim_{n \rightarrow \infty} \mathrm{sup} \sqrt[n]{a_n}$ 表示上极限（存在极限的子列中极限最大的）。

### Functions_Given_by_Power_Series

Any power series with the convergence radius $R > 0$ can represented an analytic function

$$
f(z) = \sum_{n=0}^{\infty} a_n z^n
$$

This representation is unique to a certain function as in the followed theorem.

#### Identity_Theorem_for_Power_Series

Let the power series $\sum_{n=0}^{\infty} a_n z^n$ and $\sum_{n=0}^{\infty} b_n z^n$  both be convergent for $\left|z\right| < R$, where $R > 0$, and let them both have the same sum for all these $z$.

Then the series are identical, that is, $a_0 = b_0, a_1 = b_1, a_2 = b_2, \cdots$.

Hence, if a function $f(z)$ can be represented by a power series with any center $z_0$, this representation is unique.

证明方法：令 $z = 0$，可以得到 $a_0 = b_0$.

再用数学归纳法。就可以逐项证明。

#### Termwise_Multiplication_of_Power_Series

**Cauchy product**

$$
\left(\lim_{n \rightarrow \infty} \sum_{n=0}^{\infty} a_n z^n\right) \cdot \left(\lim_{n \rightarrow \infty} \sum_{n=0}^{\infty} b_n z^n\right) = \lim_{n \rightarrow \infty} \sum_{i = 0}^{n} a_i b_{n-i} z^n
$$

很像卷积是吧？这就是 ZT.

级数的乘积也收敛（在给定的收敛半径中）。

#### Termwise_Differentiation_of_Power_Series

$$
\left(\sum_{n=0}^{\infty} a_n z^n\right)' = \sum_{n=0}^{\infty} (n+1) a_{n+1} z^n
$$

Derivation has the same convergence radius as the origin series.

$$
\lim_{n \rightarrow \infty} \frac{(n+1)a_{n+1}}{n a_n} = \lim_{n \rightarrow \infty} \frac{a_{n+1}}{a_n}
$$

#### Termwise_Integration_of_Power_Series

$$
\lim_{n \rightarrow \infty} \sum_{n=0}^{\infty} a_n z^n = \left(\lim_{n \rightarrow \infty} \sum_{n=0}^{\infty} \frac{a_{n}}{n+1} z^{n+1}\right)'
$$

==Every Power Series at every point interior to its circle of convergence can represented an Analytic Function.==

Their differentiations has the same convergence radius(domain).

### Taylor_and_Maclaurin_Series

#### Taylor's_Theorem

CONDITION:

- $f(z)$ is analytic in $D$, $z_0 \in D$

CONCLUSION:

- Only one Taylor Series with  its center $z_0$ with convergence circle the largest disk with center $z_0$ where $f(z)$ is analytic.

$$
\begin{aligned}
& f(z) = \sum_{m = 0}^{n} a_m (z - z_0)^m + R_n(z) \\
& a_m = \frac{f^{(m)}(z_0)}{m!} = \frac{1}{i 2 \pi} \oint_C \frac{f(\zeta)}{(\zeta - z_0)^{m+1}} \mathrm{d} \zeta \\
& R_n(z) = \frac{1}{i 2 \pi}\left( \oint_C \frac{f(\zeta)}{(\zeta - z_0)^{n+1} (\zeta - z)} \mathrm{d} \zeta \right) (z - z_0)^{n + 1}
\end{aligned}
$$

CONDITION:

- On the circle $\left|z - z_0\right| = r$ with the point on and inside it in $D$, $\forall z \in \{z: \left|z - z_0\right| = r\}, \left|f(z)\right| \leq M$

CONCLUSION:

$$
\left|a_n\right| \leq \frac{M}{r^n}
$$

PROOF:

选择一个以 $z_{0}$ 为圆心的逆时针圆周 $C$ 其中包含了 $z$. 利用 Cauchy 积分公式：

$$
\begin{aligned}
    f(z) &= \frac{1}{i 2 \pi} \oint_C \frac{f(\zeta)}{\zeta - z} \mathrm{d} \zeta \\
        &= \frac{1}{i 2 \pi} \oint_C \frac{f(\zeta)}{\zeta - z_0 - (z - z_0)} \mathrm{d} \zeta \\
    \frac{1}{\zeta - z_{0} - (z - z_{0})} &= \frac{1}{\zeta-z_{0}} \left( \frac{1}{1-\frac{z-z_{0}}{\zeta-z_{0}}} \right) \\
\end{aligned}
$$

由于 $|z - z_{0}| < |\zeta - z_{0}| \Rightarrow \left| \dfrac{z-z_{0}}{\zeta-z_{0}} \right| < 1$

$$
\begin{aligned}
    \frac{1}{1-\frac{z-z_{0}}{\zeta-z_{0}}} &= \frac{1-\left(\frac{z-z_{0}}{\zeta-z_{0}}\right)^{n}}{1-\frac{z-z_{0}}{\zeta-z_{0}}} + \left( \frac{z-z_{0}}{\zeta-z_{0}} \right)^{n} \cdot \frac{1}{1-\frac{z-z_{0}}{\zeta-z_{0}}} \\
        &= 1 + \left( \frac{z-z_{0}}{\zeta-z_{0}} \right) + \left( \frac{z-z_{0}}{\zeta-z_{0}} \right)^{2} + \cdots + \left( \frac{z-z_{0}}{\zeta-z_{0}} \right)^{n-1} + \frac{(z-z_{0})^{n}}{(\zeta-z)(\zeta-z_{0})^{n-1}} \\
    f(z) &= \frac{1}{i2\pi} \sum_{k=1}^{n} \oint_{C} \frac{f(\zeta)}{(\zeta-z_{0})^{k}} \mathrm{d} \zeta (z-z_{0})^{k-1} + \frac{1}{i2\pi} \oint_{C} \frac{f(\zeta)}{(\zeta-z)(\zeta-z_{0})^{n-1}} \mathrm{d} \zeta (z-z_{0})^{n} \\
        &= \frac{1}{i2\pi} \sum_{k=0}^{n} \left[ \oint_{C} \frac{f(\zeta)}{(\zeta - z_{0})^{k+1}} \mathrm{d} \zeta \right] (z - z_{0})^{k} + \frac{1}{i2\pi} \oint_{C} \frac{f(\zeta)}{(\zeta - z)(\zeta - z_{0})^{n}} \mathrm{d} \zeta (z - z_{0})^{n+1}
\end{aligned}
$$

利用 Cauchy 高阶导数公式：

$$
\begin{aligned}
    f(z) &= \sum_{k=0}^{n} \frac{f^{(k)}(z_{0})}{k!} + R_{n}(z) \\
    R_{n}(z) &= \frac{1}{i2\pi} \left[ \oint_{C} \frac{f(\zeta)}{(\zeta-z)(\zeta-z_{0})^{n+1}} \mathrm{d} \zeta\right] (z - z_{0})^{n+1}
\end{aligned}
$$

由于 $f(\zeta)$ 解析，因此有界：

$$
\begin{aligned}
    & \frac{f(\zeta)}{(\zeta - z)} \leq M \\
    |R_{n}(z)| &= \frac{|z-z_{0}|^{n+1}}{2\pi} \left| \oint_{C} \frac{f(\zeta)}{(\zeta - z)(\zeta - z_{0})^{n+1}} \mathrm{d} \zeta \right| \\
        &\leq \frac{\left| z-z_{0} \right|^{n+1}}{2\pi} \left| \oint_{C} \frac{M}{\rho^{n+1}} \mathrm{d} \zeta \right| && \rho = |\zeta - z_{0}| \\
    &= M \rho \left( \frac{|z-z_{0}|}{\rho} \right)^{n+1} \\
    \lim_{n \rightarrow \infty} |R_{n}(z)| &= 0 && \left(\frac{|z-z_{0}|}{\rho} < 1\right)
\end{aligned}
$$

### Laurent_Series

#### Laurent's_Theorem

![image-20211107115005487](images/ComplexFunction-Laurent-Theorem.png)

特例：如果曲线 $C_{2}$ 内部没有极点，那么根据 Cauchy 积分定理：

$$
\begin{aligned}
    & \oint_{C_{2}} \frac{f(\zeta)}{\zeta - z} \mathrm{d} \zeta = 0
\end{aligned}
$$

转化为之前的 Taylor 级数。

对于普遍的情况（$C_{2}$ 内部存在极点），要考虑内环路积分所对应的负数次幂，这部分被称为 Laurent 级数的**主要部分**。

CONDITION:

- $f(z)$ is analytic in the annulus between two concentric circle $C_1, C_2$ with center $z_0$.

CONCLUSION:

$$
\begin{aligned}
    & f(z) = \sum_{n = - \infty}^{\infty} a_n (z - z_0)^n \\
    & a_n = \frac{1}{i 2 \pi} \oint_C \frac{f(\zeta)}{(\zeta - z_0)^n} \mathrm{d} \zeta \text{ where } n \in \mathbb{Z}
\end{aligned}
$$

PROOF:

对环域应用 Cauchy 积分定理：

$$
\begin{aligned}
    i2\pi f(z) &= \oint_{C_{1}} \frac{f(\zeta)}{\zeta - z} \mathrm{d} \zeta + \oint_{C_{2}^{-}} \frac{f(\zeta)}{\zeta - z} \mathrm{d} \zeta \\
    g(z) &:= \oint_{C_{1}} \frac{f(\zeta)}{\zeta-z} \mathrm{d} \zeta = \sum_{n=0}^{\infty} \frac{1}{i2\pi} \left[ \oint_{C} \frac{f(\zeta)}{(\zeta - z_{0})^{n}} \mathrm{d} \zeta \right] (z-z_{0})^{n} \\
    h(z) &:= \oint_{C_{2}^{-}} \frac{f(\zeta)}{\zeta -z} \mathrm{d} \zeta
\end{aligned}
$$

其中 $g(z)$ 的展开方法和证明 Taylor 定理的方法一致。

$$
\begin{aligned}
    h(z) &= - \oint_{C_{2}} \frac{f(\zeta)}{(\zeta - z)} \mathrm{d} \zeta \\
        &= - \oint_{C_{2}} \frac{f(\zeta)}{\zeta - z_{0} - (z - z_{0})} \mathrm{d} \zeta \\
        &= - \oint_{C_{2}} \frac{f(\zeta)}{(z-z_{0})\left(\frac{\zeta-z_{0}}{z-z_{0}}-1\right)} \mathrm{d} \zeta \\
        &= \oint_{C_{2}} \frac{f(\zeta)}{(z - z_{0})} \left[ 1 + \frac{\zeta-z_{0}}{z-z_{0}} + \left(\frac{\zeta-z_{0}}{z-z_{0}}\right)^{2} + \cdots + \left( \frac{\zeta-z_{0}}{z-z_{0}} \right)^{n} + \frac{(\zeta - z_{0})^{n+1}}{(z-\zeta)(z - z_{0})^{n}} \right] \mathrm{d} \zeta \\
        &= \sum_{k=0}^{n} \left[ \oint_{C_{2}} f(\zeta) (\zeta - z_{0})^{k} \mathrm{d} \zeta \right] \frac{1}{(z-z_{0})^{k+1}} + \left[ \oint_{C_{2}} f(\zeta) \frac{(\zeta -z_{0})^{n+1}}{z-\zeta} \mathrm{d} \zeta \right] \frac{1}{(z-z_{0})^{n+1}}
\end{aligned}
$$

得到了余项：

$$
\begin{aligned}
    R_{n}^{*}(z) &= \frac{1}{i2\pi} \left[ \oint_{C_{2}} f(\zeta) \frac{(\zeta - z_{0})^{n+1}}{z - \zeta} \mathrm{d} \zeta \right] \frac{1}{(z - z_{0})^{n+1}} \\
        &\leq \frac{1}{2\pi |z-z_{0}|^{n+1}} M \rho^{n+1} 2\pi \rho \\
        &= M \rho \left( \frac{\rho}{|z-z_{0}|} \right)^{n+1} \\
    \lim_{n \rightarrow \infty} R_{n}^{*} (z) &= 0
\end{aligned}
$$

事实上，一般来说 $h(z)$ 展开的项数是有限的，这种情况可以直接证明结束。在无限的情况下才会使用上面的证明。

同一个函数在不同的收敛域展开成不同形式的函数项级数：

$$
\begin{aligned}
    f(z) &= \frac{1}{1-z} \\
    f(z) &= 1 + z + z^{2} + \cdots && (|z| < 1) \\
    f(z) &= - \frac{1}{z(1- \frac{1}{z})} \\
        &= - \frac{1}{z} \left( 1 + \frac{1}{z} + \left(\frac{1}{z}\right)^{2} + \cdots \right) && (|z| > 1) \\
        &= - \frac{1}{z} - \left( \frac{1}{z} \right)^{2} - \left( \frac{1}{z} \right)^{3} - \cdots && (|z| > 1)
\end{aligned}
$$

负数次幂可以有限，也可以无限：

$$
\begin{aligned}
    \frac{\sin z}{z^{5}} &= \sum_{k=0}^{\infty} z^{-5} \frac{(-1)^{k}}{(2k+1)!} z^{2k+1} \\
        &= \sum_{k=0}^{\infty} \frac{(-1)^{k}}{(2k+1)!} z^{2k-4} \\
        &= \frac{1}{z^{4}} - \frac{1}{6z^{2}} + \frac{1}{120} - \frac{1}{5040} z^{2} + \cdots && (|z| > 0)
\end{aligned}
$$

$$
\begin{aligned}
    z^{2}e^{\frac{1}{z}} &= z^{2} \sum_{k=0}^{\infty} \frac{1}{k!} \left(\frac{1}{z}\right)^{k} \\
        &= \sum_{k=0}^{\infty} \frac{1}{k!} \frac{1}{z^{k-2}} \\
        &= z^{2} + z + \frac{1}{2} + \frac{1}{3!z} + \frac{1}{4!z^{2}}+ \cdots && (|z| > 0)
\end{aligned}
$$

### Singularities_and Zeros,Infinity

#### Singularity

$f(z)$ ceases to be analytic at $z = z_0$.

- Isolated Singularity: no other singularity in the neighborhood of $z_0$
  - Pole: Laurent 级数的最低负幂 $m$
  - Simple Poles: $m = -1$
  - Isolated Essential Singularity: $m \rightarrow - \infty$
  - Removable Singularity: 可以通过补充定义的方式使得函数在该点解析，Laurent 级数的负数次幂为 0
- Nonisolated Singularity

**Pole Theorem**

CONDITION:

$f(z)$ is analytic and has a pole at $z_{0}$.

CONCLUSION:

$$
\lim_{z \rightarrow z_{0}} |f(z)| = \infty
$$

举例：$f(z) = \frac{1}{z^{2}}$ 在 $z = 0$

**Picard 大定理**

CONDITION:

$f(z)$ is analytic and has an essential isolated singularity at $z_{0}$.

cONCLUSION:

$f(z)$ 在 $z_{0}$ 的任何一个去心领域内的取值，至多有一个复数取不到。（稠密）

举例：$f(z) = e^{\frac{1}{z}}$ 在 $z = 0$.

$$
f(z) = \sum_{n=0}^{\infty} \frac{1}{n!} \frac{1}{z^{n}}
$$

Removable Singularity: $f(z) = \dfrac{\sin z}{z}$

$$
\begin{aligned}
    f(z) &= \frac{1}{z} \sum_{n=0}^{\infty} \frac{(-1)^{n}}{(2n+1)!} z^{2n+1} \\
        &= \sum_{n=0}^{\infty} \frac{(-1)^{n}}{(2n+1 )! } z^{2n}
\end{aligned}
$$

#### Zeros_of_Analytic_Function

n-order zero:

$$
\forall i \in \{0, 1, \cdots, n-1\}, f^{(i)}(z_0) = 0 \text{ and } f^{(n)}(z_0) \neq 0
$$

#### Zeros_are_Isolated

CONDITION: $f(z)$ is an analytic function ***except*** that $f(z) \equiv 0$.

CONCLUDION: zeros of $f(z)$ are isolated, that is, no other zeros in the neighborhood of a zero.

PROOF:

$f(z)$ has n-order zero at $z_{0}$

$$
\begin{aligned}
    f(z) &= \sum_{k=n}^{\infty} a_{k}(z-z_{0})^{k} \\
        &= (z-z_{0})^{n} \sum_{k=0}^{\infty} a_{n+k} (z-z_{0})^{k} \\
    g(z) &:= \sum_{k=0}^{\infty} a_{n+k} (z-z_{0})^{k} \\
    g(z_{0}) &= a_{n} \neq 0
\end{aligned}
$$

$(z-z_{0})^{k}$ 仅在 $z_{0}$ 处取 0，而 $g(z)$ 为解析函数。因此在领域内不会存在零点。

**Poles and Zeros**

CONDITION:

- $f(z)$ has n-th order zero at $z_{0}$
- $h(z)$ is analytic at $z_{0}$, and $h(z_{0}) \neq 0$

CONCLUSION:

$\dfrac{h(z)}{f(z)}$ has n-th order pole at $z_{0}$.

Especially, $\dfrac{1}{f(z)}$ has n-th order pole at $z_{0}$.

#### Infinity_and_Riemann_Sphere

![images/ComplexFunction-Riemann-sphere.png](images/ComplexFunction-Riemann-sphere.png)

- 复平面 $\mathbb{C}$ 上的点和 Riemann sphere 上除去北极点 $N$ 以外的所有点一一映射。
- 扩充复平面 $\overline{\mathbb{C}}$ 上的点和 Riemann sphere 上的所有点一一映射。

研究较大的 $\left|z\right|$ 在 $f(z)$ 的特性：

$$
g(w) := f\left(\frac{1}{w}\right) = f(z)
$$

- $f(z)$ is analytic/singular at infinity $\iff$ $g(w)$ is analytic/sigular at $w = 0$.
- $f(z)$ is zero at infinity $\iff$ $g(w)$ is zero at $w = 0$.
- $f(z)$ is poles at infinity $\iff$ $g(w)$ is poles at $w = 0$.
- $f(z)$ is essential singularity at infinity $\iff$ $g(w)$ is essential singularity at $w = 0$.

### Residue_Integration_Method

To solve the line integral $\oint_C f(z) \mathrm{d}z$ when there is a singularity interior to the path $C$. We use Laurent Series:

$$
f(z) = \sum_{z=0}^{\infty} a_n (z - z_0)^n + \sum_{n=1}^{\infty} b_n (z - z_0)^{-n}
$$

[Integral Formula given by Laurent Series](#Laurent's_Theorem)

$$
\oint_C f(z) \text{d}z = i2 \pi b_1
$$

Definition

$$
b_1 = \mathrm{Res} (f(z), z_0)
$$

e.g.

$$
\begin{aligned}
    & \oint_{C} \frac{1}{z^{3}-z^{4}} \mathrm{d} z
\end{aligned}
$$

首先得到 Laurent 级数：

$$
\begin{aligned}
    f(z) &= \frac{1}{z^{3}} \cdot \frac{1}{1-z}
\end{aligned}
$$

When $|z| < 1$

$$
\begin{aligned}
    f(z) &= \frac{1}{z^{3}} (1 + z + z^{2} + z^{3} + \cdots) \\
        &= \frac{1}{z^{3}}+ \frac{1}{z^{2}} + \frac{1}{z} + 1 + \cdots
\end{aligned}
$$

When $|z| > 1$

$$
\begin{aligned}
    f(z) &= -\frac{1}{z^{4}} \cdot \frac{1}{1- \dfrac{1}{z}} \\
        &= - \frac{1}{z^{4}} \left( 1 + \frac{1}{z} + \frac{1}{z^{2}} + \frac{1}{z^{3}} + \cdots \right) \\
        &= -z^{-4} - z^{-5} - z^{-6} - \cdots
\end{aligned}
$$

对于不同的收敛域，有不同的 Laurent 级数，因此对于不同的路径，环路积分的值也不同。

When $C = \left\{ z \in \mathbb{C}: |z| = \dfrac{1}{2} \right\}$ clockwise

$$
\oint_{C} f(z) \mathrm{d} z = - i 2 \pi \mathrm{Res}[f(z), 0] = -i 2\pi
$$

When $C = \left\{ z \in \mathbb{C}: |z| = \dfrac{3}{2} \right\}$

$$
\oint_{C} f(z) \mathrm{d} z = 0
$$

#### Get_the_Residue

Get the Residue without Laurent Series when $z_0$ is:

- Simple Pole
  - Method 1: $\mathrm{Res} (f(z), z_0) = \lim_{z \rightarrow z_0} (z-z_0) f(z)$
  - Method 2: Assume $f(z) = \frac{p(z)}{q(z)} \text{ and } p(z_0) \neq 0$, then $z_0$ is a simple zero of $q(z)$:

$$
\begin{aligned}
    & \mathrm{Res}(f(z), z_{0}) = \lim_{z \rightarrow z_{0}} \frac{(z-z_{0})p(z)}{q(z)} = \frac{p(z_0)}{q'(z_0)}
\end{aligned}
$$

- Poles of any order
  - If $z_0$ is a m-th order pole of $f(z_0)$:

$$
\mathrm{Res}(f(z), z_0) = \frac{1}{(m - 1)!} \lim_{z \rightarrow z_0} \frac{\mathrm{d}^{m-1}}{\mathrm{d}z^{m-1}} \left[(z - z_0)^m f(z) \right]
$$

#### Residue_Theorem

CONDITION:

- $f(z)$ is analytic inside and on the simple closed path $C$, except for finite singular point $\{z_k\}$ inside $C$

CONCLUSION:

$$
\oint_C f(z) \mathrm{d}z = i 2 \pi \sum_{i = 1}^{k} \mathrm{Res}_{z = z_i} f(z)
$$

### Residue_Integration_of_Real_Integrals

#### Integrals_of_Trigonometric_Fuctions

$$
J = \int_0^{2 \pi} F(\cos \theta, \sin \theta) \mathrm{d} \theta
$$

$F(\cos \theta, \sin \theta)$ is a rational function of $\cos \theta$ and $\sin \theta$. Let $z = e^{i \theta}$:

$$
\begin{aligned}
    & \cos \theta = \frac{1}{2}(z + \frac{1}{z}) \\
    & \sin \theta = \frac{1}{2 i}(z - \frac{1}{z}) \\
    & \frac{\mathrm{d}z}{\mathrm{d} \theta} = i e^{i \theta} \Rightarrow \mathrm{d} \theta = \frac{\mathrm{d}z}{i z}
\end{aligned}
$$

Thus, $F(\cos \theta, \sin \theta)$ is transformed to a rational function of $z$, called $f(z)$.

$$
J = \oint_{C} F\left[\dfrac{1}{2}\left(z + \dfrac{1}{z}\right), \dfrac{1}{i2}\left(z- \dfrac{1}{z}\right)\right] \frac{\mathrm{d}z}{iz} 
$$

$C$ is the unit circle counterclockwise.

#### Cauchy_Principle_Value

$$
\mathrm{pv} \int_{-\infty}^{\infty} f(x) \mathrm{d}x = \lim_{R \rightarrow \infty} \int_{-R}^R f(x) \mathrm{d}x
$$

![image-20211107184142838](images/ComplexFunction-Cauchy-pv-integral.png)

$$
\mathrm{pv} \int_{-\infty}^{\infty} f(x) \mathrm{d}x + \lim_{R \rightarrow \infty}\int_S f(z) \mathrm{d}z = i 2 \pi \sum_{i = 1}^{k} \mathrm{Res}[f(z), z = z_{i}]
$$

where $\Im z_i > 0$.

当 $f(x)$ 的分母多项式的最高幂次至少比分子的最高幂次大 2 时

$$
\lim_{R \rightarrow \infty} \int_S f(z) \mathrm{d} z \leq \lim_{R \rightarrow \infty} \left| \int_S f(z) \mathrm{d} z \right| \leq \lim_{R \rightarrow \infty} \frac{C}{R^2} \pi R = \lim_{R \rightarrow \infty} \frac{\pi C}{R} = 0
$$

Thus,

$$
\mathrm{pv} \int_{-\infty}^{\infty} f(x) \mathrm{d}x = i 2 \pi \sum_{i = 1}^{k} \mathrm{Res}[f(z), z = z_{i}]
$$

其中，$z_{1} \cdots z_{k}$ 表示复平面中在 $x$ 轴上方的极点。（事实上也可以取下方的极点，因为分母的最高幂次比分子大，无穷远点不是极点）

#### Fourier_Integrals

对于同样满足分母多项式的最高幂次至少比分子的最高幂次大 2 的 $f(x)$，如果要求它的傅里叶级数：

$$
\begin{aligned}
    & \int_{-\infty}^{\infty} f(x) \cos sx \mathrm{d}x \\
    & \int_{-\infty}^{\infty} f(x) \sin sx \mathrm{d}x
\end{aligned}
$$

可以使用和[上面部分](#Cauchy_Principle_Value)同样的解法。

$$
\begin{aligned}
    & \left\{ \begin{aligned}
        & \int_{-\infty}^{\infty} f(z) e^{isz} \mathrm{d}z = \int_{-\infty}^{\infty} f(x) \cos sx \mathrm{d} x + i \int_{-\infty}^{\infty}f(x) \sin sx \mathrm{d} x \\
        & \int_{-\infty}^{\infty}f(z) e^{isz} \mathrm{d} z = i2\pi \sum_{i=1}^{k} \mathrm{Res}[f(z)e^{isz}, z = z_{i}]
    \end{aligned} \right. \\
    \Rightarrow & \left\{  \begin{aligned}
    & \int_{-\infty}^{\infty} f(x) \cos sx \mathrm{d}x = -2 \pi \sum_{i = 1}^{k} \Im \mathrm{Res}_{z = z_i} [f(z) e^{i s z}] \\
    & \int_{-\infty}^{\infty} f(x) \sin sx \mathrm{d}x =2 \pi \sum_{i = 1}^{k} \Re \mathrm{Res}_{z = z_i} [f(z) e^{i s z}]
    \end{aligned}  \right.
\end{aligned}
$$

易于证明在乘了傅里叶的项以后在上一节的证明方法仍然成立。

#### Another_Kind_of_Improper_Integral

$\int_A^B f(x) \mathrm{d}x$ that $\exists a \in [A, B], \lim_{x \rightarrow a} f(x) = \infty$

$$
\begin{aligned}
    & \int_{A}^{B} f(x) \mathrm{d} x \\
    & \exists a \in [A, B], \lim_{x \rightarrow a} f(x) = \infty
\end{aligned}
$$

By definition,

$$
\int_A^B f(x) \mathrm{d} x = \lim_{\varepsilon \rightarrow 0^+} \int_A^{a - \varepsilon} f(x) \mathrm{d} x + \lim_{\eta \rightarrow 0^+}\int_{a + \eta}^B f(x) \mathrm{d} x
$$

可能存在两个极限在各自的变量趋近于 0 时不存在但是下面的极限存在的情况：

$$
\mathrm{pv} \int_A^B f(x) \mathrm{d} x = \lim_{\varepsilon \rightarrow 0^+} \left[\int_A^{a - \varepsilon} f(x) \mathrm{d} x + \int_{a + \varepsilon}^B f(x) \mathrm{d} x \right]
$$

Similarly, we define this integral **Cauchy principle value**.

**Simple Poles on Real Axis**

![images/ComplexFunction-simple-pole-real-axis.png](images/ComplexFunction-simple-pole-real-axis.png)

CONDITION:

$f(z)$ has a simple pole $z=a$ on real axis

CONCLUSION:

$$
\lim_{r \rightarrow 0} \int_{C_{2}} f(z) \mathrm{d} z = i \pi \mathrm{Res}[f(z), z = a]
$$

PROOF:

simple pole:

$$
\begin{aligned}
    f(z) &= \frac{b_{1}}{z-a} + g(z) && (0 < |z-a| < R)
\end{aligned}
$$

where $g(z)$ is analytic on $a$

$$
\begin{aligned}
    C_{2} & = \left\{ z \in \mathbb{C} : z = a + r e^{i \theta}, \theta \in [0, \pi] \right\} \\
    \int_{C_{2}} f(z) \mathrm{d}z & = \int_{0}^{\pi} \frac{b_{1}}{re^{i \theta}} \cdot i r e^{i \theta} \mathrm{d} \theta + \int_{C_{2}} g(z) \mathrm{d} z = i \pi b_{1} + \int_{C_{2}} g(z) \mathrm{d} z \\
    \int_{C_{2}} g(z) \mathrm{d} z &\leq M \pi r \Rightarrow \lim_{r \rightarrow 0} \int_{C_{2}} g(z) \mathrm{d} z = 0 \\
    \lim_{r \rightarrow 0} \int_{C_{2}} g(z) \mathrm{d} z &= i \pi b_{1} = i \pi \mathrm{ Res} [f(z), z = a]
\end{aligned}
$$

关于 [pv-integral](#Cauchy_Principle_Value) 的推广：

![images/ComplexFunction-Cauchy-pv-integral-simple-pole-on-real-axis.png](images/ComplexFunction-Cauchy-pv-integral-simple-pole-on-real-axis.png)

$$
\begin{aligned}
    \text{pv} \int_{-\infty}^{\infty} f(x) \mathrm{d} x &= i2\pi \sum_{j=1}^{k} \mathrm{Res}[f(z), z = z_{j}] - \int_{S} - \int_{C_{2}^{-}}
\end{aligned}
$$

当满足条件：

1. $f(x)$ 分母的幂次至少比分子的幂次大 2 ($\int_{S} \rightarrow 0$);
2. $f(x)$ 在 $a$ 的阶数为 1.

则：

$$
\begin{aligned}
    \text{pv} \int_{-\infty}^{\infty} f(x) \mathrm{d} x = i 2\pi \sum_{j=1}^{k} \mathrm{Res}[f(z), z = z_{j}] + i \pi \sum_{m=1}^{n} \mathrm{Res} [f(z), z = z_{m}]
\end{aligned}
$$

#### Gauss_Integral

$$
\int_{-\infty}^{\infty} e^{- \frac{x^{2}}{2} - i \omega x} \mathrm{d} x = \sqrt{2\pi} e^{- \frac{\omega^{2}}{2}}
$$

![images/ComplexFunction-Gauss-integral.png](images/ComplexFunction-Gauss-integral.png)

PROOF:

根据 Cauchy Integral Theorem:

$$
\begin{aligned}
    0 &= \oint_{C} f(z) \mathrm{d} z \\
    f(z) &= e^{- \frac{z^{2}}{2}} \\
    \oint_{C} f(z) \mathrm{d} z &= \int_{-R}^{R} e^{- \frac{x^{2}}{2}} \mathrm{d} x + \int_{0}^{\omega} e^{- \frac{(R+it)^{2}}{2}} i \mathrm{d}t - \int_{-R}^{R} e^{- \frac{(x+i \omega)^{2}}{2}} \mathrm{d} x - \int_{0}^{\omega} e^{- \frac{(-R+it)^{2}}{2}} i \mathrm{d} t
\end{aligned}
$$

首先要证明平行于虚数轴的路径积分为 0.

$$
\begin{aligned}
    & e^{- \dfrac{(\pm R+it)^{2}}{2}} = e^{\dfrac{t^{2}-R^{2}}{2} \mp iRt} \\
    & \left| e^{- \dfrac{(\pm R+it)^{2}}{2}} \right| = e^{\dfrac{t^{2}-R^{2}}{2}} \leq e^{\dfrac{\omega^{2}-R^{2}}{2}} \\
    & \int_{0}^{\omega} e^{- \dfrac{(\pm R +it)^{2}}{2}} i \mathrm{d} t \leq \omega e^{\dfrac{\omega^{2}-R^{2}}{2}} \\
    & \lim_{R \rightarrow \infty} \int_{0}^{\omega} e^{- \dfrac{(\pm R +it)^{2}}{2}} i \mathrm{d} t= 0
\end{aligned}
$$

因此得到：

$$
\begin{aligned}
    & \int_{-R}^{R} e^{- \frac{(x+i\omega)^{2}}{2}} \mathrm{d} x = \int_{-R}^{R} e^{- \frac{x^{2}}{2}} \mathrm{d} x \\
    & e^{\frac{\omega^{2}}{2}} \int_{-R}^{R} e^{- \frac{x^{2}}{2} - i \omega x} \mathrm{d} x = \int_{-R}^{R} e^{- \frac{x^{2}}{2}} \mathrm{d} x \\
    & \int_{-\infty}^{\infty} e^{- \frac{x^{2}}{2} - i \omega x} \mathrm{d} x = e^{- \frac{\omega^{2}}{2}} \int_{-\infty}^{\infty} e^{- \frac{x^{2}}{2}} \mathrm{d} x = \sqrt{2\pi} e^{- \frac{\omega^{2}}{2}}
\end{aligned}
$$

#### Frenel_Integral

$$
\int_{0}^{\infty} \cos x^{2} \mathrm{d} x = \int_{0}^{\infty} \sin x^{2} \mathrm{d} x = \frac{\sqrt{2\pi}}{4}
$$

![images/ComplexFunction-Fresnel-integral.png](images/ComplexFunction-Fresnel-integral.png)

PROOF:

对于三角函数的积分，不要直接用三角函数的函数

$$
f(z) = \cos z^{2}
$$

而要用指数函数

$$
f(z) = e^{iz^{2}}
$$

同样利用 Cauchy 积分定理：

$$
\begin{aligned}
    0 &= \int_{0}^{R} e^{ix^{2}} \mathrm{d} x + \int_{S} + \int_{L^{-}} \\
    \left| \int_{S} \right| &= \left| \int_{0}^{\frac{\pi}{4}} e^{i R^{2} e^{i2\theta}} iRe^{i \theta}\mathrm{d} \theta  \right| \\
        &= \left| \frac{R}{2} \int_{0}^{\frac{\pi}{2}} e^{- R^{2} \sin \theta} \mathrm{d} \theta \right| \\
        &\leq \left| \frac{\pi R}{4} \int_{0}^{\frac{\pi}{2}} e^{- R^{2} \frac{2}{\pi} \theta} \mathrm{d} \left( \frac{2}{\pi} \theta \right) \right| && \left(\sin \theta > \frac{2}{\pi} \theta \right) \\
        &= \frac{\pi}{4R} (1-e^{-R^{2}}) \\
    \lim_{R \rightarrow \infty} \left| \int_{S} \right| &= 0 \\
\end{aligned}
$$

$$
\begin{aligned}
    \int_{0}^{\infty} e^{ix^{2}}\mathrm{d}x &= \int_{L} e^{ix^{2}} \mathrm{d} x \\
        &= \int_{0}^{\infty} e^{i \left(le^{i \frac{\pi}{4}}\right)^{2}} e^{i \frac{\pi}{4}} \mathrm{d}l \\
        &= e^{i \frac{\pi}{4}} \int_{0}^{\infty} e^{-l^{2}} \mathrm{d} l
\end{aligned}
$$

下面求积分

$$
\begin{aligned}
    I &= \int_{0}^{\infty} e^{-x^{2}} \mathrm{d} x \\
    I^{2} &= \int_{0}^{\infty} \int_{0}^{\infty} e^{-(x^{2}+y^{2})} \mathrm{d} x \mathrm{d} y \\
        &= \int_{0}^{\frac{\pi}{2}} \mathrm{d} \theta \int_{0}^{\infty} e^{-r^{2}} r \mathrm{d} r \\
        &= \frac{\pi}{4} \\
    I &= \frac{\sqrt{\pi}}{2}
\end{aligned}
$$

$$
\begin{aligned}
    \int_{0}^{\infty} e^{ix^{2}} \mathrm{d} x &= e^{i \frac{\pi}{4}} \frac{\sqrt{\pi}}{2} \\
    \int_{0}^{\infty} \cos x^{2} \mathrm{d} x &= \Re \int_{0}^{\infty}e^{ix^{2}} \mathrm{d} x = \frac{\sqrt{2\pi}}{4} \\
    \int_{0}^{\infty} \sin x^{2} \mathrm{d} x &= \Im \int_{0}^{\infty}e^{ix^{2}} \mathrm{d} x = \frac{\sqrt{2\pi}}{4} \\
\end{aligned}
$$

#### Dirichlet_integral

$$
\int_{0}^{\infty} \frac{\sin x}{x} \mathrm{d} x = \frac{\pi}{2}
$$

![images/ComplexFunction-Dirichlet-integral.png](images/ComplexFunction-Dirichlet-integral.png)

同样的思路，先转化成复变的指数函数

$$
\begin{aligned}
    f(z) &= \frac{e^{iz}}{z} \\
    0 &= 2 \int_{\varepsilon}^{R} \frac{f(z)}{z} \mathrm{d} z + \int_{S} + \int_{C_{2}^{-}} \\
    \left| \int_{S} \right| &= \left| \int_{0}^{\pi} \frac{e^{i R e^{i \theta}}}{R e^{i \theta}} i R e^{i \theta} \mathrm{d} \theta \right| \\
        &= \left| \int_{0}^{\pi} e^{- R \sin \theta + i R \cos \theta} \mathrm{d} \theta \right| \\
        &= \left| \int_{0}^{\pi} e^{- R \sin \theta} \mathrm{d} \theta \right| \\
        &= 2 \left| \int_{0}^{\frac{\pi}{2}} e^{- R \sin \theta} \mathrm{d} \theta \right| \\
        &\leq \frac{\pi}{R} \left| \int_{0}^{\frac{\pi}{2}} e^{- \frac{2R}{\pi} \theta} \mathrm{d} \left( \frac{2R}{\pi} \theta \right) \right| && \left(\sin \theta \geq \frac{2}{\pi} \theta\right)\\
        & = \frac{\pi}{R} \left( 1 - e^{-R} \right) \\
    \lim_{R \rightarrow \infty} \left| \int_{S} \right| = 0
\end{aligned}
$$

利用[实数轴上单极点的留数](#Another_Kind_of_Improper_Integral)的关系

$$
\begin{aligned}
    \int_{C_{2}} \frac{e^{iz}}{z} \mathrm{d} z &= i \pi \mathrm{Res}\left[\frac{e^{iz}}{z}, z = 0 \right] = i \pi \\
    \int_{0}^{\infty} \frac{e^{iz}}{z} \mathrm{d} z &= \frac{1}{2} \int_{C_{2}} \frac{e^{iz}}{z} \mathrm{d} z = i \frac{\pi}{2} \\
    \int_{0}^{\infty} \frac{\sin x}{x} \mathrm{d} x &= \Im \int_{0}^{\infty} \frac{e^{iz}}{z} \mathrm{d} z = \frac{\pi}{2}
\end{aligned}
$$

学废的老铁记得双击加关注！
