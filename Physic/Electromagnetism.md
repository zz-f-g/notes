# electronmagnetism

## week 01

绝缘细棒弯成半径为 $R$ 的半圆形，其上半段带电量为 $+q$，下半段带电量为 $-q$，均匀。求圆心处的电场强度。

![[Pasted image 20220502214829.png|300]]

解：取上下对称的电荷元，记电荷元与圆心的连线和水平线的夹角为 $\theta$. 电荷元的电场强度向下。

$$
\begin{align*}
    &\mathrm{d} E = \frac{1}{4\pi \varepsilon_{0}} \frac{\frac{q}{\frac{\pi}{2}R} \cdot R \mathrm{d} \theta}{R^{2}} = \frac{q}{2\pi^{2}\varepsilon_{0}R^{2}} \mathrm{d} \theta \\
    & E = \int_{0}^{\frac{\pi}{2}} \frac{q}{2\pi^{2}\varepsilon_{0}R^{2}} \mathrm{d} \theta = \frac{q}{4\pi \varepsilon_{0} R^{2}}
\end{align*}
$$
方向向下。

---

## week 02

![500](images/Pasted%20image%2020220506215518.png)

如图所示，求：

1. $O, D$ 两点电势 $U_{O}, U_{D}$
2. 把单位正电荷 $q_{0}$ 从 $O$ 点沿圆弧 $OCD$ 移动到 $D$ 点电场力做功
3. 把单位正电荷 $q_{0}$ 从 $D$ 点沿 $AB$ 延长线移动到无穷远电场力做功

解：（1）
$$
\begin{align*}
    & U_{O} = \frac{q}{4\pi \varepsilon_{0} R} + \frac{-q}{4 \pi \varepsilon_{0} R} = 0 \\
    & U_{D} = \frac{q}{4 \pi \varepsilon_{0} (3R)} + \frac{-q}{4 \pi \varepsilon_{0} R} = - \frac{q}{6 \pi \varepsilon_{0} R}
\end{align*}
$$
（2）
$$
W_{2} = q_{0} (U_{D} - U_{O}) = - \frac{q}{6\pi \varepsilon_{0} R}
$$
（3）
$$
W_{3} = q_{0} (0 - U_{D}) = \frac{q}{6\pi \varepsilon_{0} R}
$$

## week 03

半径分别为 $1 \text{cm}$ 和 $2 \text{cm}$ 的两个球形导体，各带电荷 $1.0 \times 10^{-8} \text{C}$，两球相距很远。若用细导线将两球连接，求：（取 $\frac{1}{4\pi \varepsilon_{0}} = 9.0 \times 10^{9} \text{N} \cdot \text{m}^{2} \cdot \text{C}^{-2}$）

1. 每个球各带电荷
2. 每球的电势

解：将两个导体球视为孤立导体球，电容为
$$
\begin{align*}
    & C_{1} = 4\pi \varepsilon_{0} R_{1} \\
    & C_{2} = 4\pi \varepsilon_{0} R_{2} \\
\end{align*}
$$

用细导线连接以后称为等电势系统。
$$
U = \frac{q_{1}}{C_{1}} = \frac{q_{2}}{C_{2}}
$$
电荷守恒
$$
q_{1} + q_{2} = 2 q_{0}
$$

解得
$$
\left\{\begin{align*}
    & q_{1} = \frac{R_{1}}{R_{1}+R_{2}} \cdot 2 q_{0} \\
    & q_{2} = \frac{R_{2}}{R_{1}+R_{2}} \cdot 2 q_{0} \\
    & U = \frac{q_{0}}{2\pi \varepsilon_{0}(R_{1} + R_{2})}
\end{align*}\right.
$$

带入数据
$$
\left\{\begin{align*}
    & q_{1} = 6.67 \times 10^{-9} \text{C} \\
    & q_{2} = 1.33 \times 10^{-8} \text{C} \\
    & U = 6 \times 10^{3} \text{V}
\end{align*}\right.
$$

## week 04

两金属球的半径之比为 $1:4$，带等量同号电荷。当两者之间的距离远大于两者半径时，有一定的电势能。若将两球用导线连接再撤去，则电势能变为原来的多少倍？

解：两者的电容为
$$
\begin{align*}
    &C_{1} = 4 \pi \varepsilon_{0} R_{1} \\
    &C_{2} = 4 \pi \varepsilon_{0} R_{2} \\
\end{align*}
$$
两者的电势能为
$$
W_{0} = \frac{1}{2} \frac{q_{0}^{2}}{4\pi \varepsilon_{0} r} = \frac{q_{0}^{2}}{8\pi \varepsilon_{0} r}
$$
用导线连接以后变成等势体
$$
U = \frac{q_{1}}{C_{1}}= \frac{q_{2}}{C_{2}}
$$
电荷守恒
$$
q_{1} + q_{2} = 2 q_{0}
$$
解得
$$
\left\{\begin{align*}
    & q_{1} = \frac{R_{1}}{R_{1}+R_{2}} \cdot 2 q_{0} = \frac{2}{5} q_{0} \\
    & q_{2} = \frac{R_{2}}{R_{1}+R_{2}} \cdot 2 q_{0} = \frac{8}{5} q_{0} \\
\end{align*}\right.
$$
两者的电势能为
$$
W = \frac{1}{2} \frac{q_{1}q_{2}}{4\pi \varepsilon_{0} r} = \frac{16}{25} \frac{q_{0}^{2}}{4\pi \varepsilon_{0} r}
$$
$$
\frac{W}{W_{0}}= \frac{16}{25}
$$

---

## week 05

一无限长的载流导线中部被弯成圆弧形，如图所示，圆弧形半径为 $R=3\text{cm}$，导线中的电流为 $I=2\text{A}$。求圆弧形中心 $O$ 点的磁感应强度。

![300](https://edu-image.nosdn.127.net/D87DCD24A05E26342EB40E68AE006DF5.jpg?imageView&thumbnail=890x0&quality=100)

解：半无限长直导线在 $O$ 点产生的磁感应强度垂直纸面向外，大小为

$$
B_{1} = \frac{\mu_{0}I}{4\pi R}
$$

3/4 圆弧导线在 $O$ 点产生的磁感应强度垂直纸面向内，大小为

$$
B_{2} = \int_{0}^{\frac{3}{2}\pi} \frac{\mu_{0} I R \mathrm{d} \theta}{4\pi R^{2}} = \frac{3}{8} \frac{\mu_{0}I}{R}
$$

总磁感应强度垂直纸面向内，大小为

$$
B = B_{2} - 2 B_{1} = \left(\frac{3}{8} - \frac{1}{2\pi}\right) \frac{\mu_{0}I}{R}
$$

---

## week 06

一半圆形线圈半径为 $R$ ，共有 $N$ 匝，所载电流为 $I$，线圈放在磁感强度为 $B$ 的均匀磁场中，$B$ 的方向始终与线圈的直边垂直。

1. 求线圈所受的最大磁力矩；
2. 如果磁力矩等于最大磁力矩的一半，线圈处于什么位置？
3. 线圈所受的力矩与转动轴位置是否有关？

![300](https://edu-image.nosdn.127.net/B70369EE8A1BC078A095BB150E40B756.jpg?imageView&thumbnail=890x0&quality=100)

解：(1) 磁矩

$$
\begin{align*}
    \mu & = N \cdot\frac{\pi}{2} R^{2} I \\
    \boldsymbol{M} & = \boldsymbol{\mu} \times \boldsymbol{B} \\
    M_{max} & = \mu B = \frac{\pi}{2} NIR^{2}B
\end{align*}
$$

(2) 设线圈平面和磁感应方向的夹角为 $\varphi$

$$
\begin{align*}
    & \boldsymbol{M} = \mu B \cos \varphi \\
    & \frac{M}{M_{max}} = \frac{1}{2} \Rightarrow \varphi = \frac{\pi}{3}
\end{align*}
$$

线圈平面和磁感应强度方向的夹角为 $60 \degree$.

(3) 无关。

---

## week 07

长直导线与矩形单匝线圈共面放置，导线与线圈的长边平行。矩形线圈的边长分别为 $a, b$，它到直导线的距离为 $c$（如图）。当直导线中通有电流 $I = I_{0} \sin \omega t$ 时，求矩形线圈中的感应电动势。

![images/Pasted image 20220601164454.png](images/Pasted%20image%2020220601164454.png)

解：需要求线圈的磁通

$$
\begin{align*}
    B &= \frac{\mu_{0}I}{2\pi x} \\
    \Phi &= \iint_{S} B \mathrm{d}S \\
    &= \int_{0}^{b} \mathrm{d} y \int_{c}^{c+a} B \mathrm{d}x \\
    &= \frac{\mu_{0}Ib}{2\pi} \ln \left(\frac{c + a}{c}\right) \\
    &= \frac{\mu_{0}I_{0}b}{2\pi} \ln \left(\frac{c + a}{c}\right) \sin \omega t \\
\end{align*}
$$

根据电磁感应定律

$$
\begin{align*}
    \varepsilon & = - \frac{\partial \Phi}{\partial t} = - \frac{\mu_{0}\omega I_{0}b}{2\pi} \ln \left(\frac{c+a}{a}\right) \cos \omega t
\end{align*}
$$

---

## week 08

由半径为 $R_{1}, R_{2}$ 的两个同轴薄圆筒形导体构成的电缆，在两个圆筒中填充磁导率为 $\mu$ 的均匀磁介质。电缆内层电流为 $I$，外层为电流的返回路径，求长度为 $l$ 的一段电缆的磁场能量。

![images/Pasted image 20220606093429.png](images/Pasted%20image%2020220606093429.png)

解：利用安培环路定理得到磁场仅在圆筒之间有分布：

$$
\begin{align*}
    & B = \frac{\mu I}{2\pi r} \\
    & w_{B} = \frac{B^{2}}{2\mu} = \frac{\mu I^{2}}{8\pi^{2}r^{2}} \\
\end{align*}
$$

对空间积分

$$
\begin{align*}
    E_{B} &= \iiint_{V} w_{B} \mathrm{d}V \\
    &= \int_{R_{1}}^{R_{2}} \frac{\mu I^{2}}{8 \pi^{2} r^{2}} l \cdot 2 \pi r \mathrm{d}r \\
    & = \frac{\mu I^{2} l}{4\pi} \ln \frac{R_{2}}{R_{1}}
\end{align*}
$$
