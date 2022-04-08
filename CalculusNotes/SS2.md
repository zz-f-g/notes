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
