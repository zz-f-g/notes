# 李群和李代数

来自《视觉 SLAM 十四讲》

## Definitions

**Group**: $G = (A, \cdot)$

$$
\begin{aligned}
& \forall a_{1}, a_{2} \in A, a_{1} \cdot a_{2} \in A \\
& \forall a_{1}, a_{2}, a_{3} \in A, (a_{1} \cdot a_{2}) \cdot a_{3} = a_{1} \cdot (a_{2} \cdot a_{3})\\
& \exists a_{0} \in A, \text{ s.t. }\forall a \in A, a_{0} \cdot a = a \cdot a_{0} \\
& \forall a \in A, \exists a^{-1} \in A, \text{ s.t. } a \cdot a^{-1} = a_{0} \\
\end{aligned}
$$

Examples of Group:

- GL(n)
    - $A = \left\{ \boldsymbol{M} \in \mathbb{R}^{n \times n}: \mathrm{rank} \boldsymbol{M} = n \right\}$
    - $\boldsymbol{M}_{1} \cdot \boldsymbol{M}_{2} = \boldsymbol{M}_{1} \boldsymbol{M}_{2}$
- SO(n) $\subset$ GL(n)
    - $A = \left\{ \boldsymbol{R} \in \mathbb{R}^{n \times n}: \boldsymbol{R} \boldsymbol{R}^{T} = \boldsymbol{I}_{n}, \mathrm{det} \boldsymbol{R} = 1 \right\}$
- SE(n) $\subset$ GL(n+1)
    - $A = \left\{ \boldsymbol{T} = \begin{bmatrix} \boldsymbol{R} & \boldsymbol{t} \\ \boldsymbol{0}^{T} & 1 \end{bmatrix}: \boldsymbol{R} \in SO(n), \boldsymbol{t} \in \mathbb{R}^{n} \right\}$

**Lie Group**: continuous, smooth

**Lie Algebra**: $\mathfrak{g} = (\mathbb{V}, \mathbb{F}, [,])$, where $\mathbb{V}$ is a set and $\mathbb{F}$ is a number domain（数域）, $[,]: \mathbb{V}, \mathbb{V} \rightarrow \mathbb{F}$

$$
\begin{aligned}
\text{1. } &\forall \boldsymbol{X}, \boldsymbol{Y} \in \mathbb{V}, [\boldsymbol{X}, \boldsymbol{Y}] \in \mathbb{V} \\
\text{2. } &\forall \boldsymbol{X}, \boldsymbol{Y} \in \mathbb{Y}, \forall a, b \in \mathbb{F},  \\
&\mathrm{ s.t. } [a \boldsymbol{X} + b \boldsymbol{Y}, \boldsymbol{Z}] = a [\boldsymbol{X}, \boldsymbol{Z}] + b [\boldsymbol{Y}, \boldsymbol{Z}] \\
& [\boldsymbol{Z}, a \boldsymbol{X} + b \boldsymbol{Y}] = a [\boldsymbol{Z}, \boldsymbol{X}] + b [\boldsymbol{Z}, \boldsymbol{Y}] \\
\text{3. }& \forall \boldsymbol{X} \in \mathbb{V}, [\boldsymbol{X}, \boldsymbol{X}] = 0 \\
\text{4. }& \forall \boldsymbol{X}, \boldsymbol{Y}, \boldsymbol{Z} \in \mathbb{V}, \\
&\text{s.t. } [\boldsymbol{X}, [\boldsymbol{Y}, \boldsymbol{Z}]] + [\boldsymbol{Z}, [\boldsymbol{X}, \boldsymbol{Y}]] + [\boldsymbol{Y}, [\boldsymbol{Z}, \boldsymbol{X}]] = 0
\end{aligned}
$$

## Correspondence

SO(3) 的李代数的引出

$$
\begin{aligned}
\boldsymbol{I} &= \boldsymbol{R}(t) \boldsymbol{R}(t)^{T} \\
\boldsymbol{0} &= \dot{\boldsymbol{R}(t)} \boldsymbol{R}(t)^{T} + \boldsymbol{R}(t) \dot{\boldsymbol{R}}(t)^{T} \\
\dot{\boldsymbol{R}}(t) \boldsymbol{R}(t) ^{T} &= - \boldsymbol{R}(t) \dot{\boldsymbol{R}}(t)^{T} = - [\dot{\boldsymbol{R}}(t) \boldsymbol{R}(t)^{T}]^{T}
\end{aligned}
$$

这是一个反对称矩阵，不妨设 $\varphi(t) \in \mathbb{R}^{3},$

$$
\begin{aligned}
\varphi^{\wedge}(t) &= \dot{\boldsymbol{R}}(t) \boldsymbol{R}(t)^{T} \\
\dot{\boldsymbol{R}}(t) &= \varphi^{\wedge}(t) \boldsymbol{R}(t)
\end{aligned}
$$

设 $\boldsymbol{R}(0) = \boldsymbol{I}$

$$
\begin{aligned}
\boldsymbol{\dot{R}}(t) &= I + \varphi^{\wedge}(0) t
\end{aligned}
$$

$\varphi^{\wedge}$ 表征了切空间。

李代数 $\varphi \in \mathbb{R}^{3}$，定义二元运算为

$$
[\varphi_{1}, \varphi_{2}] = (\varphi_{1}^{\wedge} \varphi_{2}^{\wedge} - \varphi_{2}^{\wedge} \varphi_{1}^{\wedge})^{\vee}
$$

验证这个二元运算符合李代数的四条性质，第二和第三条性质比较显然。第一条性质验证：

$$
\begin{aligned}
(\varphi_{1}^{\wedge} \varphi_{2}^{\wedge} - \varphi_{2}^{\wedge} \varphi_{1}^{\wedge})^{T} &= (-\varphi_{2}^{\wedge})(-\varphi_{1}^{\wedge}) - (-\varphi_{1}^{\wedge})(-\varphi_{2}^{\wedge}) \\
&= -(\varphi_{1}^{\wedge} \varphi_{2}^{\wedge} - \varphi_{2}^{\wedge} \varphi_{1}^{\wedge})
\end{aligned}
$$

第四条性质验证：

$$
\begin{aligned}
\left[ \varphi_{1}, \left[ \varphi_{2}, \varphi_{3} \right] \right] &= \left[ \varphi_{1}, (\varphi_{2}^{\wedge} \varphi_{3}^{\wedge} - \varphi_{3}^{\wedge}\varphi_{2}^{\wedge})^{\vee}  \right] \\
&= \varphi_{1}^{\wedge} (\varphi_{2}^{\wedge} \varphi_{3}^{\wedge} - \varphi_{3}^{\wedge} \varphi_{2}^{\wedge}) - (\varphi_{2}^{\wedge} \varphi_{3}^{\wedge} - \varphi_{3}^{\wedge} \varphi_{2}^{\wedge})\varphi_{1}^{\wedge} \\
&= \varphi^{\wedge}_{1,2,3} - \varphi^{\wedge}_{1,3,2} - \varphi^{\wedge}_{2,3,1} + \varphi^{\wedge}_{3,2,1}
\end{aligned}
$$

相加后可消去所有项。

李代数 $\mathfrak{se}(3)$

$$
\begin{aligned}
\mathfrak{se}(3) &= \left\{ \xi = \begin{bmatrix}\boldsymbol{\rho} \\ \boldsymbol{\varphi} \end{bmatrix} \in \mathbb{R}^{6}: \boldsymbol{\rho} \in \mathbb{R}^{3}, \boldsymbol{\varphi} \in \mathfrak{so}(3), \xi^{\wedge} = \begin{bmatrix} \boldsymbol{\varphi}^{\wedge} & \boldsymbol{\rho} \\ \boldsymbol{0}^{T} & 0 \end{bmatrix} \in \mathbb{R}^{4 \times 4} \right\} \\
[\boldsymbol{\xi}_{1}, \boldsymbol{\xi}_{2}] &= \left( \boldsymbol{\xi}_{1}^{\wedge} \boldsymbol{\xi}_{2}^{\wedge} - \boldsymbol{\xi}_{2}^{\wedge} \boldsymbol{\xi}_{1}^{\wedge} \right)^{\vee}
\end{aligned}
$$

验证方法和 $\mathfrak{so}(3)$ 类似。

## Mapping

**$\mathfrak{so}(3)$ 上的指数映射**

$$
\begin{aligned}
e^{\varphi^{\wedge}} &= \sum_{n=0}^{\infty} \frac{1}{n!} (\varphi^{\wedge})^{n}
\end{aligned}
$$
