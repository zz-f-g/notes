# Signals and Systems

时域、频域、复频域、离散域。

答疑：

Z622 Tue 11:45(12:15)~13:20

---

信号（确定性）

- 连续时间信号
- 离散时间信号

分析信号的描述、特性、运算。

- 时域分析
- 频域分析

---

系统（LTI）

- 连续时间系统
- 离散时间系统

分析系统的描述、特性、响应和求解。

- 复频域分析
- Z 域分析

---

平时成绩 40%

- 15% 考勤 + 课堂参与
- 10% 平时作业
- 15% 实验成绩（MATLAB）
- 60% 期中（连续信号与系统） + 期末

---

## 1 Introduction

---

### 1.1

信号分析的基本思想：把复杂信号分解成**简单基本单元信号**的线性组合。

系统分析：研究输入输出关系

系统综合：根据需求设计系统

```mermaid
graph LR;
E --> system --> R
```

---

### 1.2

信号的描述和分类

描述

1. 物理上：随时间变化的物理量
2. 数学上：一个或多个变量的函数
3. 形态上：波形

数学描述——表达式

---

MATLAB -- sound

```matlab
load chirp
sound(y, Fs)
```

---

信号的分类

- 确定性
- 随机性（干扰噪声）

---

- 连续信号(t)
- 离散信号(n)

---

- 周期信号
$$
\begin{aligned}
    &x(t+T) = x(t) \\
    &x(n+N) = x(n)
\end{aligned}
$$
- 非周期信号

---

```mermaid
graph LR;
A[analog]
S[sampling]
D[digital]
A --sample--> S --quantize--> D
```

抽样信号幅值连续，数字信号幅值不连续。它们的时间都不连续。

---

信号的特性

1. 时间特性：波形随时间变化
2. 频率特性：不同的频率分量（幅值，相位，信号带宽）
3. 能量特性：周期信号一定是功率信号，非周期不一定。

---

几种典型确定信号

- 指数信号
- 正弦信号
- 复指数信号
- 抽样信号
- 钟形脉冲函数

---

指数信号
$$
f(t) = Ke^{\alpha t}
$$
时间常数
$$
\tau = \frac{1}{|\alpha|}
$$

---

正弦信号

频域的自变量是角频率，不是频率。

衰减正弦信号
$$
f(t) =
\begin{cases}
K e^{-\alpha t} \cos (\omega t + \varphi) & (t > 0) \\
0 & (t < 0)
\end{cases}
$$

$t=0$ 是关键点。

---

复指数信号
$$
\begin{aligned}
    f(t) &= K e^{st} & (s = \sigma + j \omega) \\
    &= K e^{\sigma t} \cos \omega t + j K e^{\sigma t} \sin \omega t
\end{aligned}
$$

---

抽样信号

Sampling signal
$$
Sa(t) = \frac{\sin t}{t} = \mathrm{sinc} (t/\pi)
$$

Properties:

- $Sa(t) = Sa(-t)$
- $Sa(0) = 1$
- $Sa(t) = 0, t = \pm \pi, \pm 2 \pi, \cdots$
- $\lim_{t \rightarrow \infty} Sa(t) = 0$
- $\int_0^{\infty} Sa(t) dt = \frac{\pi}{2}$

---

钟形脉冲函数

Gauss Signal
$$
f(t) = E e^{- \left(\frac{t}{\tau}\right)^2}
$$

---

### 1.3

信号的运算

---

- 平移，反褶，尺度
- 微积分
- 两个信号的相加或相乘

---

平移
$$
f(t) \rightarrow f(t- \tau)
$$

$\tau > 0$ 右移（滞后）
$\tau < 0$ 左移（超前）

超前信号在物理上是有极限的。

---

翻转
$$
f(t) \rightarrow f(-t)
$$
没有物理实现的器件。

但是通过堆栈的“后进先出”可以在数字信号处理中实现。

---

展缩
$$
f(t) \rightarrow f(\alpha t)
$$
- $|\alpha| < 1$，坐标轴拉长
- $|\alpha| > 1$，坐标轴收缩

---

一般情况

$$
f(t) \rightarrow f(at+b)
$$

- 先平移 $b$，左加右减
- 标度变换，如果 $a<0$，再翻转

$$
f(mt+n) \rightarrow f(at+b)
$$

---

微分和积分

对阶跃信号做微分得到冲激信号。对于冲激信号进行识别，可以应用与自动驾驶的车道线识别。

---

相加和相乘

**同一时刻**对信号进行相加或相乘。

同等功率下，信号的频率越大，传播的距离越短。

信号不做除法，无意义。

---

### 1.4

阶跃信号和冲激信号

---

奇异信号和奇异函数

函数不连续或导数积分不连续

---

单位斜变信号
$$
R(t) =
\begin{cases}
0 & (t<0) \\
t & (t>0)
\end{cases}
$$

三角形脉冲
$$
f(t) =
\begin{cases}
\frac{K}{\tau}R(t) & (t \leq \tau) \\
0 & (t>\tau)
\end{cases}
$$

---

单位阶跃信号
$$
u(t) = R'(t) = 
\begin{cases}
0 & (t<0) \\
1 & (t>0)
\end{cases}
$$

No def or $\frac{1}{2}$ on $t = 0$.

---

用单位阶跃信号描述其他信号

门函数
$$
f(t) = u\left(t+ \frac{\tau}{2}\right)- u\left(t - \frac{\tau}{2}\right)
$$

与其他函数相乘，只保留 $\tau$ 时间内的部分。（卡门）

---

符号函数
$$
sgn(t) = 2u(t) - 1 = u(t) - u(-t)
$$

---

单位冲激函数

$$
\begin{align*}
&\delta(t) = \begin{cases}0 &(t \neq 0) \\ \infty & (t=0) \end{cases} \\
&\int_{-\infty}^{\infty} \delta(t) \mathrm{d}t = 1 \\
&\int_{-\infty}^{\infty} \delta(t) \mathrm{d}t = \int_{0-}^{0+} \delta(t) \mathrm{d}t \\
\end{align*}
$$

---

用阶跃函数定义（不严谨）

$$
\delta(t) = \frac{d}{dt} u(t)
$$

---

用极限定义
$$
\begin{align*}
&u_{\Delta}(t) =
\begin{cases}
\frac{1}{\tau} R(t) &(t\leq \tau) \\
1 & (t> \tau)
\end{cases}\\
&\delta(t) = \lim_{\tau \rightarrow 0} \frac{d}{dt} u_{\Delta}(t)
\end{align*}
$$

或者用门限函数。

---

冲激函数的性质

1. 抽样性
2. 奇偶性（$\delta(t) = \delta(-t)$）
3. 冲激偶
4. 标度变换

---

抽样性
$$
\int_{-\infty}^{\infty} \delta(t - t_{0}) f(t) \mathrm{d} t = f(t_{0})
$$

---

冲激偶

$$
\begin{align*}
&\delta'(t) = \frac{d}{dt}\delta(t)\\
&\int_{-\infty}^{\infty} \delta'(t) = 0\\
&\int_{-\infty}^{0} \delta'(t) = \delta(t)\\
&
\end{align*}
$$

---

### 1.5

信号的分解

---

