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
& \int_{-\infty}^{\infty} \delta'(t) f(t) \mathrm{d}t \\
=& (\delta(t) f(t))|_{-\infty}^{\infty} - \int_{-\infty}^{\infty} f'(t) \delta(t) \mathrm{t}\\
=& -f'(0)
\end{align*}
$$

---

### 1.5

信号的分解

---

- 直流+交流
- 奇+偶
- 脉冲
- 实部+虚部
- 正交
- (分形理论)

---

DC+AC
$$
\begin{align*}
    &f(t) = f_{D}(t) + f_{A}(t) \\
    & f_{D}(t) = \frac{1}{T} \int_{t_{0}}^{t_{0}+T} f(t) \mathrm{d}t \\
    P &= \frac{1}{T} \int_{t_{0}}^{t_{0}+T} f^{2}(t) \mathrm{d}t \\
    &= \frac{1}{T} \left( \int_{t_{0}}^{t_{0}+T} [f_{D}(t) + f_{A}(t)]^{2} \mathrm{d}t \right) \\
    &= f_{D}^{2}(t) + \frac{1}{T} \left( \int_{t_{0}}^{t_{0}+T} f_{A}^{2}(t) \mathrm{d}t \right)
\end{align*}
$$
DC,AC 正交。

---

odd+even

$$
\begin{align*}
    & f_{o}= \frac{1}{2}(f(t)-f(-t))
    & f_{e}= \frac{1}{2}(f(t)+f(-t))
\end{align*}
$$

---

pulse

$$
\begin{align*}
    f(t) &= \sum_{\tau=-\infty}^{\infty}  f(\tau) [u(t-\tau) - u(t-\tau-\Delta \tau)] \\
    &= \sum_{\tau=-\infty}^{\infty}  f(\tau) \frac{[u(t-\tau) - u(t-\tau-\Delta \tau)]}{\Delta \tau} \Delta \tau \\
    &= \int_{-\infty}^{\infty} f(\tau) \delta(t-\tau) \mathrm{d}\tau
\end{align*}
$$

---

Real+Imag

$$
\begin{align*}
    & f(t) = f_{r}(t) + j f_{i}(t)\\
    & f_{r}(t) = \frac{1}{2} [ f(t) + f^{*}(t) ] \\
    & f_{i}(t) = \frac{1}{2j} [ f(t) - f^{*}(t) ] \\
\end{align*}
$$

---

正交

---

### 1.6

系统模型及其分类

---

系统模型的基本运算

- 加法器
- 乘法器
- 标量乘法器
- 微分器
- 积分器
- 延时器

---

系统的表示

1. 数学表达式
2. 系统图：形象地表示其功能

---

系统的互联

1. 级联 Cascade
2. 并联 Parallel
3. 反馈 Feedback

---

系统的分类

1. 即时系统（无记忆）：全电阻网络
    - 恒等系统
2. 动态系统（记忆）：输出不仅和当前输入有关，也和其他时刻的输入有关。

---

可逆性：单射

如果一个系统和另一个系统级联之后构成恒等系统，则两个系统互逆。

---

因果性

输出只与此前输入有关则为因果系统。

非因果系统：
$$
\begin{align*}
    &y(t) = x(t+1) \\
    &y(n) = x(n) + x(n+1) \\
    &y(n) = x(-n)
\end{align*}
$$

一切即时系统（无记忆）都是因果的。一切物理可实现的连续时间系统都是因果的。

---

稳定性

输入有界 ---> 输出有界

RC; RLC 均为稳定系统。

不稳定系统
$$
\begin{align*}
    & y(n) = \sum_{k = -\infty}^{n} x(k) \\
    & y(t) = \int_{-\infty}^{t} x(\tau) \mathrm{d}\tau
\end{align*}
$$

---

时变、时不变系统

时不变：输入产生时间平移，输出只产生相同的时间平移。time-invariant

---

线性

叠加性+齐次性

线性系统一定是线性方程，但反之不成立。

$$
\begin{align*}
    & y(t) = kx(t) + 2
\end{align*}
$$

去掉 (+2) 才是线性系统。

$$
\begin{align*}
    & y(t) = kx(t)
\end{align*}
$$

线性系统满足零输入时零输出。

---

增量线性系统

线性系统+与输入无关的响应（零输入响应）

---

重点：**确定性**信号作用下的**集总参数***线性***时不变**系统。

---

### 1.7

线性时不变系统

---

### 1.8

系统分析方法

---

```mermaid
graph LR
I --> |"x(t)"|LS[Linear System]
```

---

- 输入输出描述法
- 状态变量分析法（多输入输出）

数学模型解法：

- 时域分析
    - 经典方法
        - 连续系统：微分方程
        - 离散系统：差分方程
    - 卷积积分
- 变换域
    - FT
    - LT
    - ZT
    - (DFT)
    - (DWT)

---

## 2 Time Domain Analysis on Continuous System

---

### 2.1

---

### 2.2

系统数学模型的建立

---

1. 建立系统模型（写微分方程）：元件约束，拓扑约束。
2. 求系统响应（解方程）
    1. 经典法
    2. 双零法（零输入响应，零输出响应）
        1. 零输入：经典法
        2. 零状态：卷积积分

---

在 RLC 电路中，有几个**独立的**储能元件，微分方程就有几阶。

例：两个并联的电容存在关联，不独立。

---

### 2.3

用时域经典法求解微分方程‘

---

用高阶微分方程表示系统：
$$
\begin{align*}
    & C_{0} \frac{\mathrm{d}^{n}}{\mathrm{d}t^{n}} r(t) + \cdots +C_{n} r(t) \\
    =& E_{0} \frac{\mathrm{d}^{n}}{\mathrm{d}t^{n}} e(t) + \cdots +E_{n} e(t) \\
    \text{Homogeneous sol: }& C_{0} \frac{\mathrm{d}^{n}}{\mathrm{d}t^{n}} r(t) + \cdots +C_{n} r(t) = 0 \\
    & C_{0} \alpha^{n} + \cdots + C_{n} \alpha = 0
\end{align*}
$$

如果没有重根：
$$
r(t) = \sum_{k=0}^{n} A_{k} e^{\alpha_{k}t}
$$
如果有重根：
$$
\begin{align*}
    & \alpha_{1} = \cdots = \alpha_{k} \\
    & r(t) = \left( \sum_{i=1}^{k} A_{i} t^{k-i} \right) e^{\alpha_{1}t} + \sum_{i=k+1}^{n} A_{i}e^{\alpha_{i}t}
\end{align*}
$$

---

特解和激励的数学形式有关
$$
\begin{align*}
    e(t) = E && r_{p}(t) &= B \\
    e(t) = t^{n} && r_{p}(t) &= P_{n}(t) \\
    e(t) = \cos \omega t && r_{p}(t) &= B_{1} \cos \omega t + B_{2} \sin \omega t
\end{align*}
$$

---

借助初始条件确定 $A_{i}$

---

### 2.4

起始点的跳变

---

求解系统的齐次解以后
$$
\begin{align*}
    i_{h}(t) &= f(t) && (t \geq 0^{+}) \\
        &= f(t) u(t)
\end{align*}
$$

---

通过 $i(0_-)$ 求解 $i(0_{+})$ 

依据电路元件的性质：

- $i_L(0_{-)}= i_L(0_{+})$
- $u_C(0_{-}) = u_C(0_{+})$

$$
\begin{align*}
    & r^{(k)}(0_{-}) = \begin{bmatrix} r(0_{-}) & \frac{\mathrm{d}}{\mathrm{d}t} r(0_{-})  & \cdots & \frac{\mathrm{d}^{n-1}}{\mathrm{d}t^{n-1}}r(0_{-}) \end{bmatrix} \\
    \rightarrow & r^{(k)}(0_{+}) = \begin{bmatrix} r(0_{+}) & \frac{\mathrm{d}}{\mathrm{d}t} r(0_{+})  & \cdots & \frac{\mathrm{d}^{n-1}}{\mathrm{d}t^{n-1}}r(0_{+}) \end{bmatrix}
\end{align*}
$$

---

但是当有冲击激励强迫作用于系统，上述的换路定理不再成立。

$$
\begin{align*}
    v_{C}(t) &= v_{C}(0_{-}) \frac{1}{C} \int_{0_{-}}^{t} i(t) \mathrm{d}t \\
        &=v_{C}(0_{-}) \frac{1}{C} \int_{0_{-}}^{0^{+}} i(t) \mathrm{d}t + \frac{1}{C} \int_{0_{+}}^{t} i(t) \mathrm{d}t
\end{align*}
$$

---

冲激函数匹配法确定初始条件

$t=0$ 时刻等式左右的 $\delta(t)$ 以及各阶导数应该系数相等（平衡）。

？？？

---

### 2.5

零输入响应和零状态响应

---

$$
e(t) = H[r(t)] + H[x(0_{-})]
$$

可以将原始状态等效成激励源。
$$
H[x(0_{-})] = H[r_{0}(t)]
$$

---

- 全响应 = 齐次响应 + 特解
- 全响应 = 零输入响应 + 零状态响应
- 全响应 = 暂态响应 + 稳态响应

特解包含在零状态响应中，因为零输入显然只是自由的。零状态响应中也有自由的成分。

---

零输入响应不存在跳变
$$
r_{zi}^{(k)}(0_{+}) = r_{zi}^{(k)}(0_{-})
$$

---

将输入信号分解为简单信号的和，分别求解。
$$
\begin{align*}
    x_{i}(t) & \rightarrow y_{i}(t) \\
    \sum x_{i}(t) & \rightarrow \sum  y_{i}(t)
\end{align*}
$$

---

### 2.6

冲激响应和阶跃响应

---

将输入信号分解为一系列矩形脉冲信号。
$$
\begin{align*}
    x_{\Delta}(t) &= \sum_{k=-\infty}^{\infty} x(k \Delta)(t-k \Delta)\Delta \\
    x(t) &= \int x(\tau) \delta(t-\tau) \mathrm{d}\tau
\end{align*}
$$

---

单位冲激响应：单位冲激信号在**零状态**响应。
$$
\delta(t) \rightarrow h(t)
$$

---

一阶电路的冲激响应。
$$
RC \frac{\mathrm{d}v_{C}}{\mathrm{d}t} + v_{C} = \delta(t)
$$
$\delta(t)$ 在 $t=0_{+}$ 时刻转化为系统的储能。之后体现为在初始储能后齐次方程的解。
$$
\begin{align*}
    & RC \frac{\mathrm{d}v_{C}}{\mathrm{d}t} + v_{C} = 0 \\
    & \alpha = - \frac{1}{RC} \\
    & v_{C} = A e^{- \frac{t}{RC}} u(t)
\end{align*}
$$

---

求 $v_C(0_{+})$.

方法一：
$$
\begin{align*}
    & \frac{\mathrm{d}v_{C}}{\mathrm{d}t} = a \delta(t) + b \Delta u(t) \\
    & v_{C} = a \Delta u(t) \\
    & \Rightarrow a = \frac{1}{RC} \\
    & v_{C}= \frac{1}{RC} e^{- \frac{t}{RC}} u(t) \\
    & i = C \frac{\mathrm{d}v_{C}}{\mathrm{d} t} = - \frac{1}{R^{2}C} e^{- \frac{t}{RC}} u(t) + \frac{1}{R} \delta(t)
\end{align*}
$$

方法二：直接将解得形式代入解得常数。

---

阶跃响应：系统在单位阶跃信号下的**零状态**响应。

---

### 2.7

卷积

---

$$
f_{1}(t) * f_{2}(t) = \int_{-\infty}^{\infty} f_{1}(\tau) f_{2}(t - \tau) \mathrm{d} \tau
$$

---

求系统的零状态响应 $r_{zs} (t)$ .
$$
\begin{align*}
    r(t) &= H[e(t)] \\
        &= H\left[\int e(\tau) \delta(t-\tau) \mathrm{d} \tau\right]\\
        &= \int e(\tau) H[\delta(t - \tau)] \mathrm{d} \tau \\
        &= \int e(\tau) h(t - \tau) \mathrm{d} \tau \\
        &= e(t) * h(t)
\end{align*}
$$

运算过程的实质：一个信号不懂，另一个信号翻转以后平移 $t$.

用图解法求积分限，再用解析法。

$$
f_{1}(t) \text{ on } [a,b], f_{2}(t) \text{ on } [c,d] \Rightarrow g(t) = f_{1}(t) * f_{2}(t) \text{ on } [a+c,b+d]
$$

---

### 2.8

卷积性质

---

1. 交换律
2. 分配律（并联）
3. 结合律（级联）

---

微积分性质
$$
\begin{align*}
    & g'(t) = f'(t) * h(t) = f(t) * h'(t) \\
    & g^{(-1)} = f^{(-1)} * h(t) = f(t) * h^{(-1)}(t) & (Integral) \\
    & g^{(m-n)}= f^{(m)}* h^{(-n)} = f^{(-n)} * h^{(m)}
\end{align*}
$$

---

冲激函数和阶跃函数
$$
\begin{align*}
    & f(t) * \delta(t) = f(t) \\
    & f(t) * \delta^{(n)}(t) = f^{(n)}(t) \\
    & f(t) * u(t) = f^{(-1)}(t) = \int_{-\infty}^{t} f(\lambda) \mathrm{d} \lambda
\end{align*}
$$

**可导和可微的区别**：求导以后再积分是不是本身？

---

时移特性
$$
f(t-t_{1}) * h(t - t_{2}) = g(t-t_{1}-t_{2})
$$