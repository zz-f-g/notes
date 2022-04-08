# Physic

---

- 10% 每章节大卷子
- 20% 两门 spoc 课程的成绩
- 10% 考勤
- 60% 期末考试：力学电磁学

---

讨论：关于质量连续分布的质点系

$$
\begin{align}
    & \vec{r}_{c} = \frac{1}{M}\sum_{i=1}^{n} m_{i} \vec{r}_{i} \\
    & \vec{r}_{c} = \frac{1}{M}\int \vec{r} \mathrm{d}m
\end{align}
$$

这两个公式的推导都应用了微元法。第一个系统是离散的质量系统，也就是说每一个质量都是以线度为 0 的质点形式存在。这显然是一种理想化的情况。对于一般的系统，质量是连续分布的，将空间任意微分，取得的物体质量都是无穷小，即为下式中的 $\mathrm{d}m$.

那么我们将这无穷小质量的物体也视作在位矢 $\vec{r}$ 处的质点（虽然确切地说并不是），就可以套用上式的结论。将求和号改写为积分号，就推导出了下式。

---

## week 01

---

质点由静止开始作直线运动，初始加速度 $a_{0}$，以后加速度均匀增加，每经过 $r$ 秒增加 $a_{0}$，求经过 $t$ 秒后质点的速度和位移。

解：由题意
$$
\begin{align*}
    & \frac{\mathrm{d}a}{\mathrm{d}t}= \frac{a_{0}}{r} \\
    & a = a_{0} + \int_{0}^{t} \frac{a_{0}}{r} \mathrm{d}\tau = a_{0}\left(1 + \frac{t}{r}\right) \\
    & v = \int_{0}^{t} a \mathrm{d}\tau = a_{0}t \left(1 + \frac{t}{2r}\right) \\
    & x = \int_{0}^{t} v \mathrm{d} \tau = \frac{1}{2} a_{0} t^{2} + \frac{a_{0}}{6r} t^{3}
\end{align*}
$$

---

一飞机驾驶员想往正北方向航行，而风以 60 $\text{km}\cdot \text{h}^{-1}$ 的速都向西刮来，如果飞机的航速（在静止空气中的速率）为 180 $\text{km}\cdot \text{h}^{-1}$，试问驾驶员应取什么航向？飞机相对于地面的速率为多少？用矢量图说明。

航向和正北方向的夹角
$$
\alpha = \sin^{-1} \frac{60}{180} = 19.47 \degree
$$

相对地面的速率
$$
v = \sqrt{60^{2} + 180^{2}} \approx 190 \text{ m} \cdot \text{s}^{-1}
$$

矢量图
![[Pasted image 20220321141512.png]]

---

## week 02

一个质点沿半径 0.01m 的圆周运动，其角位置 $\theta = 2 + 4t^{3}$（SI），求：

1. 在 $t=2\mathrm{s}$ 时，它的速度、加速度的大小各为多少？
2. 当切向加速度的大小恰好是总加速度的一半时，$\theta$ 值为多少？
3. 在什么时刻，切向加速度的大小恰好和法向加速度相等？

解：记 $R = 0.01\mathrm{m}$.

$$
\begin{align*}
    & \theta = 2 + 4t^{3} \\
    & \frac{\mathrm{d}\theta}{\mathrm{d}t} = 12 t^{2} \\
    & \frac{\mathrm{d}^{2}\theta}{\mathrm{d}t^{2}}= 24 t \\
    & v = R \frac{\mathrm{d}\theta}{\mathrm{d}t} = 0.12 t^{2} \\
    & a_{n} = R \left(\frac{\mathrm{d}\theta}{\mathrm{d}t} \right)^{2} = 1.44 t^{4} \\
    & a_{\tau} = R \frac{\mathrm{d}^{2}\theta}{\mathrm{d}t^{2}} = 0.24 t
\end{align*}
$$

（1）代入 $t=2$，
$$
\begin{align*}
    & v = 0.48 \mathrm{m} \cdot \mathrm{s}^{-1} \\
    & a_{n} = 23.04 \mathrm{m} \cdot \mathrm{s}^{-2} \\
    & a_{\tau} = 0.48 \mathrm{m} \cdot \mathrm{s}^{-2} \\
    & a = \sqrt{a_{n}^{2} + a_{\tau}^{2}} \approx 23.045 \mathrm{m} \cdot \mathrm{s}^{-2}
\end{align*}
$$

（2）
$$
\begin{align*}
    & a_{\tau} = \frac{1}{2} a \Rightarrow a_{n} = \sqrt{3} a_{\tau} \\
    & 1.44 t^{4} = \sqrt{3} \cdot 0.24 t \\
    & t = \sqrt[3]{\frac{\sqrt{3}}{6}} = 0.66 \mathrm{s}
\end{align*}
$$

（3）
$$
\begin{align*}
    & a_{\tau} = a_{n} \\
    & 1.44 t^{4} = 0.24 t \\
    & t = \sqrt[3]{\frac{1}{6}} = 0.55 \mathrm{s}
\end{align*}
$$

一颗子弹在一定高度以水平初速度 $v_{0}$ 射出，忽略空气阻力。取枪口为坐标原点，沿 $v_{0}$ 方向为 $Ox$ 轴，竖直方向为 $Oy$ 轴，并取发射时刻为 $t=0$，试求：

1. 子弹在任意时刻 $t$ 的位置坐标及轨道方程。
2. 子弹在任意时刻 $t$ 的速度，切向加速度和法向加速度。

解：（1）

$$
\begin{align*}
    & x = v_{0} t \\
    & y = \frac{1}{2} g t^{2}
\end{align*}
$$

消去时间得到轨道方程

$$
y = \frac{gx^{2}}{2v_{0}^{2}}
$$

（2）

速度大小

$$
v = \sqrt{v_{0}^{2} + g^{2} t^{2}}
$$

速度方向（与 $Ox$ 轴夹角）

$$
\alpha = \mathrm{arctan} \frac{gt}{v_{0}}
$$

切向加速度

$$
a_{\tau} = g \sin \alpha = \frac{g^{2}t}{\sqrt{v_{0}^{2} + g^{2}t^{2}}}
$$
方向和速度方向相同

法向加速度
$$
a_{n} = g \cos \alpha = \frac{gv_{0}}{\sqrt{v_{0}^{2} + g^{2}t^{2}}}
$$

方向和速度方向垂直，和 $-Ox$ 方向夹角为

$$
\frac{\pi}{2} - \alpha = \arctan \frac{v_{0}}{gt}
$$

---

## week 03

一条轻绳跨过轻滑轮（滑轮和轴之间的摩擦力可以忽略），在绳的一段挂质量为 $m_{1}$ 的物体，另一端有质量为 $m_{2}$ 的环。当环以相对于绳为恒定的加速度 $a_{2}$ 向下滑动时，物体和环分别相对于地面的加速度是多少？环和绳之间的摩擦力是多少？

解：设环和绳之间的摩擦力为 $f$，物体和环相对于地面的加速度分别是 $a_{10},a_{20}$.

$$
\begin{align*}
    m_{2} a_{20} &= m_{2} g - f \\
    m_{1} a_{10} &= m_{1} g - f \\
    a_{2} &= a_{10} + a_{20}
\end{align*}
$$

解得

$$
\left\{
\begin{align*}
    a_{10} &= \frac{m_{2}a_{2}+(m_{1}-m_{2})g}{m_{1}+m_{2}} \\
    a_{20} &= \frac{m_{1}a_{2}-(m_{1}-m_{2})g}{m_{1}+m_{2}} \\
    f &= \frac{m_{1}m_{2}}{m_{1}+m_{2}} (2g-a_{2})
\end{align*}
\right.
$$

![[Pasted image 20220401101636.png]]

解：（１）在 AB 开始运动到 C 开始运动的这段时间，记两者的加速度为 $a$，记 $l = 0.4 \text{m}$

$$
\begin{align*}
    m_{A} g &= (m_{A}+m_{B}) a \\
    l &= \frac{1}{2} a t^{2} \\
    \Rightarrow t&= \sqrt{\frac{2l(m_{A}+m_{B})}{m_{A}g}} = 0.4 \text{s}
\end{align*}
$$

（２）设 C 开始运动前 AB 的速度为 $v_{1}$，要求的速度为 $v_{2}$。动量守恒：

$$
\begin{align*}
    2Mv_{1} &= 3M v_{2} \\
    v_{1} &= at \\
    \Rightarrow v_{2} &= \frac{2}{3} \sqrt{gl} = \frac{4}{3} \text{m/s}
\end{align*}
$$

---

## week 04

一颗子弹在抢通里前进时所受的合力大小为 $F = 400 - \frac{4 \times 10^{5}}{3} t$(SI)，子弹从枪口射出的速度为 300 $\mathrm{m} \cdot \mathrm{s}^{-1}$. 假设子弹离开枪口时合力刚好为零，求：

1. 子弹走完枪筒全长所用的时间；
2. 子弹在枪筒中所受的冲量；
3. 子弹的质量。

解：（1）离开枪口时合力为零
$$
\begin{align*}
    &400 - \frac{4\times 10^{5}}{3} t = 0 \\
    &t = 3 \times 10^{-3} \mathrm{s}
\end{align*}
$$
（2）
$$
\begin{align*}
    I &= \int_{0}^{t} F \mathrm{d}\tau \\
        &= \int_{0}^{3 \times 10^{-3}} \left(400 - \frac{4 \times 10^{5}}{3} \tau \right)\mathrm{d}\tau \\
        &= 0.6 \mathrm{N}\cdot \mathrm{s}
\end{align*}
$$
（3）
$$
m = \frac{I}{v} = 2 \times 10^{-3} \mathrm{kg}
$$

一质量均匀分布的柔软轻绳索竖直地悬挂着，绳子的下端刚好接触水平桌面。如果无初速度地释放绳索的上端，绳索将落在桌面上。证明：在绳索下落的过程中，任意时刻作用于桌面的压力，等于已经落到桌面上的绳索所受重力的三倍。

证明：绳索在下落过程中，其内部不存在张力，各质量元自由下落，加速度为 $g$. 记 $t$ 时刻落到桌面上的绳索长度为 $x$
$$
\begin{align*}
    x &= \frac{1}{2} gt^{2} \\
    v &= gt \\
    P &= \lambda (L-x)v \\
    \frac{\mathrm{d}P}{\mathrm{d}t} &= \lambda L g - F \\
    F &= \lambda gL - \frac{\mathrm{d}P}{\mathrm{d}t} \\
        &= 3 \lambda g x
\end{align*}
$$
Q.E.D
