# Signals and Systems

---

## 4 拉普拉斯变换

---

## 4.1

introduction

---

Disadvantage of Fourier Transform

- 绝大部分的傅里叶反变换比较难求
- 不满足迪利克雷条件的信号无法求解

Solutions

- 引入广义函数 $u(t), \delta(t)$
- 利用 Laplace Transform
    - 求解简单，初始条件被自动计入
    - 但是物理概念不清晰

---

复频域分析的基本信号
$$
e^{st}, s = \sigma + j \omega
$$

$\sigma = 0$ 时退化为 Fourier Transform.

---

## 4.2

Definition of Laplace Transform

---

对于因果信号，令零时刻为起始时刻
$$
\begin{align*}
    &\forall t < 0, f(t) = 0 \\
    &\mathscr{F}[f(t)] = \int_{0}^{\infty} f(t) e^{-j \omega t} \mathrm{d} t
\end{align*}
$$
对于不满足迪利克雷条件的信号，引入衰减因子，使他满足绝对可积条件
$$
\begin{align*}
    \mathscr{\mathscr{L}}[f(t)] &= \int_{0}^{\infty} f(t) e^{-\sigma t} e^{-j \omega t} \mathrm{d} t \\
        &= \int_{0}^{\infty} f(t) e^{-st} \mathrm{d} t
\end{align*}
$$

---

Inverse Laplace Transform
$$
\begin{align*}
    f(t)e^{-\sigma t} &= \frac{1}{2\pi} \int_{-\infty}^{\infty} F(s) e^{j \omega t} \mathrm{d} \omega \\
    f(t) &= \frac{1}{2\pi} \int_{-\infty}^{\infty} F(s) e^{st} \mathrm{d} \omega \\
        &= \frac{1}{j2\pi} \int_{\sigma-j\infty}^{\sigma+j\infty} F(s) e^{st} \mathrm{d}s & (\mathrm{d}s = j \mathrm{d} \omega)
\end{align*}
$$

---

采用 $0_{-}$ 系统
$$
F(s) = \int_{0_{-}}^{\infty} f(t) e^{-st} \mathrm{d}t
$$

---

convergence of Laplace Transform
$$
\begin{align*}
    &\int_{-\infty}^{\infty} | e^{-\sigma t} f(t) | \mathrm{d} t < \infty \\
    & \lim_{t \rightarrow \infty } e^{-\sigma t} f(t) = 0 \\
    & \sigma > \sigma_{0} 
\end{align*}
$$
Region of Convergence: $\sigma > \sigma_{0}$.

---

$$
\begin{align*}
    & \lim_{t \rightarrow \infty} t^{n} e^{-\sigma t} = 0 & (\sigma > 0) \\
    & \lim_{t \rightarrow \infty} e^{\alpha t} e^{-\sigma t} = 0 & (\sigma > \alpha) \\
    & \lim_{t \rightarrow \infty} e^{t^{2}} e^{-\sigma t} = \infty 
\end{align*}
$$

---

$$
\begin{align*}
    \mathscr{L}[u(t)] &= \int_{0}^{\infty} e^{-st} \mathrm{d}t = \frac{1}{s} \\
    \mathscr{L}[\delta(t-t_{0})] &= \int_{0}^{\infty} \delta(t-t_{0}) e^{-st} \mathrm{d} t = e^{-st_{0}} \\
    \mathscr{L}(e^{\alpha t}) &= \int_{0}^{\infty} e^{-(s-a)t} \mathrm{d} t  = \frac{1}{s-a} \\
    \mathscr{L}(t^{n}) &= \frac{n!}{s^{n+1}}
\end{align*}
$$

---

$$
\begin{align*}
    \mathscr{L}(\cos \omega t) &= \frac{s}{s^{2}+\omega^{2}} \\
    \mathscr{L}(\sin \omega t) &= \frac{\omega}{s^{2}+\omega^{2}} \\
\end{align*}
$$

More Details in [[IntegralTransforms#Laplace Transformation]]

---

## 4.3

Basic Properties of Laplace Transformation

---

- Linearity
- Derivative & Integration
    - $t$ domain
    - $s$ domain
- Shifting
    - $t$ domain
    - $s$ domain
 - Scaling
     - $t$ domain

---

电容的 $s$ domain 模型
$$
\begin{align*}
    v_{C}(t) &= \frac{1}{C} \int_{-\infty}^{t} i_{C}(\tau) \mathrm{d} \tau \\
    v_{C}(t) &= 
\end{align*}
$$
???

---

初值定理

Condition: $f(t)$ and $f'(t)$ 可以进行 Laplace Transform

Conclusion
$$
f(0_{+}) = \lim_{t \rightarrow 0_{+}} f(t) = \lim_{s \rightarrow \infty} s F(s)
$$

---

Proof:
$$
\begin{align*}
    sF(s) - f(0) &= \mathscr{L}(f'(t)) \\
        &= \int_{0_{-}}^{0^{+}} f'(t) e^{-st} \mathrm{d}t + \int_{0^{+}}^{\infty} f'(t) e^{-st} \mathrm{d}t \\
        &= \int_{0_{-}}^{0^{+}} e^{-st} \mathrm{d}[f(t)] + \int_{0^{+}}^{\infty} f'(t) e^{-st} \mathrm{d}t \\
        &= f(0_{+}) - f(0_{-}) + \int_{0_{+}}^{\infty} f'(t) e^{-st}\mathrm{d}t \\
    sF(s) &= f(0_{+}) + \int_{0^{+}}^{\infty} f'(t) e^{-st} \mathrm{d} t \\
    \lim_{s \rightarrow \infty} sF(s) &= f(0_{+}) + \int_{0^{+}}^{\infty} f'(t) \lim_{s \rightarrow \infty} e^{-st} \mathrm{d} t \\
        &= f(0^{+})
\end{align*}
$$

---

修正？???

---

终值定理

Condition: $f(t)$ and $f'(t)$ 可以进行 Laplace Transform

Conclusion
$$
f(\infty) = \lim_{t \rightarrow \infty} f(t) = \lim_{s \rightarrow 0} s F(s)
$$

---

## 4.4

Inverse Laplace Transform

---

- 部分分式分解
    - 一阶实数极点
    - 重根情况
    - 共轭复数根
    - 假分式
- 留数法

---

一阶实数极点
$$
\begin{align*}
    F(s) &= A(s)/B(s)\\
        &= \frac{A(s)}{\prod_{i=1}^{n}(s-s_{i})} \\
        &= \sum_{i=1}^{n} \frac{k_{i}}{s-s_{i}} \\
    k_{i} &= \lim_{s \rightarrow s_{i}} (s - s_{i})F(s)
\end{align*}
$$

---

重根情况
$$
\begin{align*}
    F(s) &= \sum_{i=1}^{n} \frac{k_{i}}{(s-s_{0})^{i}} \\
    k_{i} &= \frac{1}{(n-i)!} \lim_{s \rightarrow s_{0}} \frac{\mathrm{d}}{\mathrm{d}s} [ (s-s_{0})^{n} F(s) ]
\end{align*}
$$

---

共轭复数根情况
$$
\begin{align*}
    F(s) &= \sum_{i=1}^{n} \frac{k_{i}}{s-s_{i}} \\
    s_{i} = s_{j}^{*} & \Rightarrow k_{i} = k_{j}^{*} \\
\end{align*}
$$

---

假分式
$$
\begin{align*}
    F(s) &= p_{m-n}(s) + F'(s) \\
    \mathscr{L}^{-1}(1) &= \delta(s) \\
    \mathscr{L}^{-1}(s) &= \delta'(s) \\
    & \cdots
\end{align*}
$$

---

含有 $e^{-\alpha s}$ 项的非有理式

使用 $t$-shifting 性质
$$
\begin{align*}
    F(s) &= e^{-\alpha s} G(s) \\
    f(t) &= g(t - \alpha) u(t - \alpha)
\end{align*}
$$

---

## 4.5

$s$ 域电路分析

---

分析步骤

1. 列 $s$ 域方程
    - 列 $t$ 域方程，用 Laplace Tranform 转化成 $s$ 域方程
    - 直接套用元件的 $s$ 域模型
2. 求解 $s$ 域方程
3. Inverse Laplace Transform

---

从 $t$ 域方程到 $s$ 方程，可以不考虑零点跳变。
$$
\begin{align*}
    \mathscr{L}\left[\frac{\mathrm{d}f(t)}{\mathrm{d}t}\right] &= s F(s) - f(0_{-}) \\
    \mathscr{L}\left[\frac{\mathrm{d}^{2}f(t)}{\mathrm{d}t^{2}}\right] &= s^{2} F(s) - sf(0_{-}) - f'(0_{-}) \\
\end{align*}
$$

---

电阻
$$
\begin{align*}
    U_{R}(t) &= R I_{R}(t) \\
    U_{R}(s) &= R I_{R}(s) \\
\end{align*}
$$

---

电感
$$
\begin{align*}
    u_{L}(t) &= L \frac{\mathrm{d}i_{L}(t)}{\mathrm{d}t} \\
    U_{L}(s) &= Ls I_{L}(s)  - Li_{L}(0_{-})
\end{align*}
$$

---

电容
$$
\begin{aligned}
    &v_{C}(t)=\frac{1}{C} \int_{-\infty}^{t} i_{C}(\tau) \mathrm{d} t \\
    &V_{C}(s)=I_{C}(s) \frac{1}{s C}+\frac{1}{s} v_{C}\left(0_{-}\right)
\end{aligned}
$$
