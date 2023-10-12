# mechanism

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

---

## week 05

一根轻绳绕过一定滑轮，滑轮轴光滑，滑轮的质量为 $\frac{1}{4} M$，均匀分布在其边缘上，绳子的 A 端有一质量为 $M$ 的人抓住了绳子，而在绳子的另一端 B 系了一个质量为 $\frac{1}{2}M$ 的重物。设人从静止开始以相对绳子匀速向上爬时，绳子与滑轮之间没有相对滑动，求 B 端重物上升的加速度。（已知对过滑轮中心且垂直于轮面的转轴的转动惯量为 $J=\frac{1}{4}MR^{2}$.

解：人和重物的加速度大小相等，设为 $a$.
$$
\begin{align*}
    & Ma = Mg - T_{1} \\
    & \frac{1}{2}Ma = T_{2} - \frac{1}{2}Mg \\
    & a = R \beta
\end{align*}
$$
转动定理
$$
(T_{1} - T_{2})R = J \beta
$$
解得
$$
a = \frac{2}{7}g
$$

---

两个长度均为 $L$、质量分别为 $m_{1}, m_{2}$ 的均匀细杆首尾相连组成长直细杆，计算该长直细杆绕 $m_{1}$ 一侧端点旋转的转动惯量。

解：分别计算两端的转动惯量
$$
\begin{align*}
    & J_{1} = \frac{1}{3} m_{1}L^{2} \\
    & J_{2} = \frac{1}{12} m_{2}L^{2} + m_{2} \left(\frac{3}{2}L\right)^{2} = \frac{7}{3}m_{2}L^{2} \\
    & J = J_{1} + J_{2} = \frac{m_{1}+7m_{2}}{3} L^{2}
\end{align*}
$$

---

## week 06

一根质量为 $m$，长度为 $l$ 的均匀细棒和质量为 $m$ 的小球牢固连接在一端，杆可以绕另一端的水平轴在竖直平面内自由转动。将棒从水平位置自由释放，求：

1. 刚体绕水平轴的转动惯量；
2. 当下摆至 $\theta$ 角时，刚体的角速度。

解：1
$$
J = \frac{1}{3} m l^{2} + m l^{2} = \frac{4}{3} m l^{2}
$$

2 机械能守恒
$$
\begin{align*}
    mgl \sin \theta + mg \cdot \frac{1}{2} l \sin \theta &= \frac{1}{2} J \dot{\theta}^{2} \\
    \dot{\theta} &= \frac{3}{2} \sqrt{\frac{g \sin \theta}{l}}
\end{align*}
$$

---

如图，长为 $l$ ，质量为 $m$ 的均匀细杆可以绕水平光滑固定轴 $O$ 转动，开始时杆静止在竖直位置，另一质量也为 $m$ 的小球用长为 $l$ 的细绳系在转轴 $O$ 处。现在将小球在竖直平面内拉开至细绳与竖直方向夹角为 $\theta$，使得小球自由下落与杆端发生弹性碰撞，杆的最大偏角为 $\frac{\pi}{3}$，求角度 $\theta$.

![[Pasted image 20220426215835.png]]

解：设碰撞后杆的角速度为 $\omega$，小球的速度为 $v$ （向左为正方向）

机械能守恒
$$
mgl(1 - \cos \theta) = \frac{1}{2} \cdot \frac{1}{3} m l^{2} \omega^{2} + \frac{1}{2} m v^{2}
$$
绕转轴的角动量守恒
$$
ml \sqrt{2gl (1 - \cos \theta)} = \frac{1}{3} ml^{2} \omega + mlv
$$
解得
$$
\omega = \frac{3}{2} \sqrt{\frac{2g(1-\cos \theta)}{l}}
$$
机械能守恒
$$
mg \frac{l}{2} \left(1 - \cos \frac{\pi}{3}\right) = \frac{1}{2} \cdot \frac{1}{3} m l^{2} \omega^{2}
$$
解得
$$
\theta = \arccos \frac{2}{3} = 48.19 \degree
$$

---

## week 07

一事件在 $S'$ 系中发生在 $x' = 60 \text{m}, t' = 8 \times 10^{-8} \text{s}(y' = z' = 0)$ . $S'$ 相对于 $S$ 系的速度为 $\frac{3}{5}c$ 沿 $x$ 轴正方向。$S, S'$ 的原点在 $t = t' = 0$ 时对齐。求这一事件在 $S$ 系中的坐标。

解：洛伦兹变换
$$
\begin{align*}
    & x = \frac{x' + v t'}{\sqrt{1 - \left(\frac{v}{c}\right)^{2}}} = 93 \text{m} \\
    & t = \frac{t' + \frac{v}{c^{2}} x'}{\sqrt{1 - \left(\frac{v}{c}\right)^{2}}} = 9.375 \times 10^{-8} \text{s}
\end{align*}
$$

地球上的天文学家测定距离地球 $8 \times 10^{11} \text{m}$ 的木卫一火山爆发和墨西哥的火山爆发同时发生，以 $2.5 \times 10^{8} \text{m/s}$ 的速度从地球向木星运动的旅行者也观察到了这两个事件。则对于空间旅行者来说：

1. 这两个事件哪一个先发生？
2. 这两个事件的空间距离是多少？

解：（1）建立从地球向木星为 $x$ 轴的坐标系 $S$ （地球系）和 $S'$ （旅行者系）。设墨西哥火山爆发为事件 $(0, 0)$ ，则木卫一火山爆发在 $S$ 为事件 $(x = 8 \times 10^{11} \text{m}, 0 )$. 在 $S'$ 系中的时间坐标：
$$
t' = \frac{- \frac{v}{c^{2}} x}{\sqrt{1 - \left(\frac{v}{c}\right)^{2}}} < 0
$$
木卫一火山爆发先发生。

（2）
$$
x' = \frac{x}{\sqrt{1 - \left(\frac{v}{c}\right)^{2}}} = 1.447 \times 10^{9} \text{m}
$$

---

## week 08

一旅客在星际旅行中打了 5 min 的瞌睡，他乘坐的宇宙飞船相对太阳系的速度是 $0.99c$，那么太阳系中的观测者认为他睡了多长时间？

解：瞌睡开始和瞌睡结束，在旅客参考系中的空间间隔为 0
$$
\begin{align*}
    & \Delta t = \frac{\Delta t' + \frac{v}{c^{2}} \Delta x'}{\sqrt{1 - \frac{v^{2}}{c^{2}}}} = \frac{\Delta t'}{\sqrt{1 - \frac{v^{2}}{c^{2}}}} = 25 \text{min}
\end{align*}
$$

要使得电子从 $v_{1} = 1.2 \times 10^{8} \text{m} \cdot \text{s}^{-1}$ 加速到 $v_{2} = 2.4 \times 10^{8} \text{m} \cdot \text{s}^{-1}$，需要对它做功多少？

解：
$$
W = \left(\frac{1}{\sqrt{1 - \frac{v_{2}^{2}}{c^{2}}}} - \frac{1}{\sqrt{1 - \frac{v_{1}^{2}}{c^{2}}}}\right) m_{e} c^{2} = \frac{4}{71}\times10^{-14} \text{J}
$$

---
