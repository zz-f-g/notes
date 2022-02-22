# Signals and Systems

时域、频域、复频域、离散域。

Z612

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

MATLAB -- spsound

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
S[sampled]
D[digital]
A --sample--> S --quantize--> D
```

抽样信号幅值连续，数字信号幅值不连续。

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
