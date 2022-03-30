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
