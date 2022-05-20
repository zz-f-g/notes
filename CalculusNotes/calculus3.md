# Calculus3

## 两类曲面积分之间的联系

计算方法一：分面投影法，计算三次二重积分

计算方法二：（转化为第一类曲面积分）

$$
\begin{align*}
    & \iint P \mathrm{d}y \mathrm{dz} + Q \mathrm{d}z \mathrm{d}x + R \mathrm{d}x \mathrm{d}y \\
    & = \iint P \cos \alpha \mathrm{d} S \\
    & + \iint Q \cos \beta \mathrm{d} S \\
    & + \iint R \cos \gamma \mathrm{d} S
\end{align*}
$$

$(\cos \alpha, \cos \beta, \cos \gamma)$ 是曲面在 $(x, y, x)$ 处与面指向一致的法向量。

$$
\left\{\begin{align*}
    & \cos \alpha = \mp \frac{z_{x}}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
    & \cos \beta = \mp \frac{z_{y}}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
    & \cos \gamma = \pm \frac{1}{\sqrt{1 + z_{x}^{2} + z_{y}^{2}}} \\
\end{align*}\right.
$$

计算方法三：合一投影法，把三元积分转化成一个二元积分
$$
\iint P \mathrm{d}y \mathrm{d}z + Q \mathrm{d}z \mathrm{d}x + R \mathrm{d}x \mathrm{d}y = \iint [P(- z_{x}) + Q(- z_{y}) + R] \mathrm{d}x \mathrm{d}y
$$

example:
$$
\begin{align*}
    & \Sigma = \{(x, y, z): z = \frac{1}{2} (x^{2} + y^{2}), z \in [0, 2]\} \\
    & ? \iint_{\Sigma} (z^{2} + x) \mathrm{d}y \mathrm{d}z - z \mathrm{d}x \mathrm{d}y
\end{align*}
$$

$$
\begin{align*}
    & \iint_{\Sigma} (z^{2} + x) \mathrm{d}y \mathrm{d}z - z \mathrm{d}x \mathrm{d}y \\
    & = \iint_{\Sigma} [(z^{2} + x)(-x) -z] \mathrm{d}x \mathrm{d}y \\
    & = -\iint_{\Sigma} [([\frac{1}{2} (x^{2} + y^{2})]^{2} + x)(-x) -\frac{1}{2} (x^{2} + y^{2})] \mathrm{d}x \mathrm{d}y \\
\end{align*}
$$

负号因为积分的曲面是下侧。

高斯公式

将第二类曲面积分转化成三重积分

闭空间区域 $\Omega$ 由分片光滑闭曲面 $\Sigma$ 围成，函数具有一阶连续偏导数。
$$
\iiint_{\Omega} \left(\frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}\right)\mathrm{d}v = \oint_{\Sigma} P \mathrm{d}y \mathrm{d}z + Q \mathrm{d}z \mathrm{d}x + R \mathrm{d}x \mathrm{d}y
$$

直观证明：考虑一个轴向和 $z$ 轴平行的柱状物上下曲面为
$$
\begin{align*}
    &\Sigma_{2}: z_{2} = z_{2} (x, y)  \\
    &\Sigma_{1}: z_{1} = z_{1} (x, y) 
\end{align*}
$$
