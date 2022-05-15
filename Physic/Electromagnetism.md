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
