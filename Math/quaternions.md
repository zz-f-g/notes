# 四元数

启发来自：[如何形象地理解四元数？ - Yang Eninala的回答 - 知乎](https://www.zhihu.com/question/23005815/answer/33971127)

## 从三维旋转到定义

四元数的常用应用是为了描述三维空间中的矢量 $\boldsymbol{x}$ 绕三维空间中的矢量 $\boldsymbol{v}$ 左旋转（符合右手螺旋定则：右手大拇指指向 $\boldsymbol{v}$，四指指向为转动方向）$\theta$ 角度之后得到的矢量 $\boldsymbol{y}$ 之间的关系的。

定义四元数

$$
\begin{aligned}
q &= (w + x \hat{i} + y \hat{j} + z \hat{k})
\end{aligned}
$$

相加和 4 维向量一致，相乘是向量叉积的扩展，即除了

$$
\begin{aligned}
\hat{j} \times \hat{k} &= \hat{i} = - \hat{k} \times \hat{j} \\
\hat{k} \times \hat{i} &= \hat{j} = - \hat{i} \times \hat{k} \\
\hat{i} \times \hat{j} &= \hat{k} = - \hat{j} \times \hat{i} \\
\end{aligned}
$$

之外，还统一了复数（为了实现向量系数 $x, y, z$ 和标量系数[^1] $w$ 之间的互相转化）：

[^1]: 姑且这么一叫。

$$
\begin{aligned}
\hat{i}^{2} = \hat{j}^{2} = \hat{k}^{2} = -1
\end{aligned}
$$

定义四元数的共轭

$$
q^{*} = (w - x \hat{i} - y \hat{j} - z \hat{k})
$$

对于开头所说的三维旋转问题，构造**旋转特征四元数**[^2]：

[^2]: 同样姑且这么一叫。

$$
q_{v} = \left( \cos \frac{\theta}{2} + \sin \frac{\theta}{2} v_{1} \hat{i} + \sin \frac{\theta}{2} v_{2} \hat{j} + \sin \frac{\theta}{2} v_{3} \hat{k} \right)
$$

同样将原来的三位矢量 $\boldsymbol{x}$ 扩展为四元数：

$$
q_{x} = (0 + x_{1} \hat{i} + x_{2} \hat{j} + x_{3} \hat{k})
$$

那么就有结论：

$$
q_{y} = q_{v} \cdot q_{x} \cdot q_{v}^{*}
$$

## 四元数旋转与三维几何的统一性

这里使用一种不严谨的记法：

$$
q_{v} = \cos \frac{\theta}{2} + \sin \frac{\theta}{2} \boldsymbol{v}
$$

那么根据四元数的运算法则，将向量点积叉积同四元数的乘积统一起来：

$$
\begin{aligned}
q_{v1} \cdot q_{v2} &= \left( \cos \frac{\theta}{2} + \sin \frac{\theta}{2} \boldsymbol{v}_{1} \right) \cdot \left( \cos \frac{\theta}{2} + \sin \frac{\theta}{2} \boldsymbol{v}_{2} \right) \\
&= \left[ \left( \cos^{2} \frac{\theta}{2} - \sin^{2} \frac{\theta}{2} \boldsymbol{v}_{1} \cdot \boldsymbol{v}_{2} \right) + \left( \sin \frac{\theta}{2} \cos \frac{\theta}{2} (\boldsymbol{v}_{1} + \boldsymbol{v}_{2}) + \sin^{2} \frac{\theta}{2} \boldsymbol{v}_{1} \times \boldsymbol{v}_{2} \right) \right]
\end{aligned}
$$

代入 $q_{y} = q_{v} \cdot q_{x} \cdot q_{v}^{*}$

$$
\begin{aligned}
q_{v} \cdot q_{x} &= \left(\cos \frac{\theta}{2} + \sin \frac{\theta}{2} \boldsymbol{v} \right) \cdot (0 + \boldsymbol{x}) \\
&= \left( - \sin \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{x}) + \cos \frac{\theta}{2} \boldsymbol{x} + \sin \frac{\theta}{2} \boldsymbol{v} \times \boldsymbol{x} \right) \\
q_{y} &= (q_{v} \cdot q_{x}) \cdot \left( \cos \frac{\theta}{2} - \sin \frac{\theta}{2} \boldsymbol{v} \right) \\
&= - \sin \frac{\theta}{2} \cos \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{x}) - \left(\cos \frac{\theta}{2} \boldsymbol{x} + \sin \frac{\theta}{2} \boldsymbol{v} \times \boldsymbol{x}\right) \cdot \left(- \sin \frac{\theta}{2} \boldsymbol{v}\right) \\
& + \cos \frac{\theta}{2} \left( \cos \frac{\theta}{2} \boldsymbol{x} + \sin \frac{\theta}{2} \boldsymbol{v} \times \boldsymbol{x} \right) + \left( - \sin \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{x}) \right) \left(  - \sin \frac{\theta}{2} \boldsymbol{v} \right) \\
&- \cos \frac{\theta}{2} \sin \frac{\theta}{2} \boldsymbol{x} \times \boldsymbol{v} - \sin^{2} \frac{\theta}{2} (\boldsymbol{v} \times \boldsymbol{x} ) \times \boldsymbol{v}
\end{aligned}
$$

“实部”：

$$
\Re q_{y} = - \sin \frac{\theta}{2} \cos \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{x}) + \cos \frac{\theta}{2} \sin \frac{\theta}{2} (\boldsymbol{x} \cdot \boldsymbol{v}) + \sin^{2} \frac{\theta}{2} (\boldsymbol{v} \times \boldsymbol{x}) \cdot \boldsymbol{v} = 0
$$

这一点部分说明了该方法的合理性，经过变换以后的四元数 $q_{y}$ 和之前的四元数 $q_{x}$ 一致，都是纯四元数（“实部”为 0，同构于三维向量）。不然三维向量经过旋转就变成四维了。下面计算重头戏“虚部”：

$$
\begin{aligned}
\boldsymbol{y} &= \Im q_{y} \\
&= \cos^{2} \frac{\theta}{2} \boldsymbol{x} + \cos \frac{\theta}{2} \sin \frac{\theta}{2} \boldsymbol{v} \times \boldsymbol{x} + \sin^{2} \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{x}) \boldsymbol{v} \\
&- \cos \frac{\theta}{2} \sin \frac{\theta}{2} \boldsymbol{x} \times \boldsymbol{v} - \sin^{2} \frac{\theta}{2} (\boldsymbol{v} \times \boldsymbol{x}) \times \boldsymbol{v}
\end{aligned}
$$

利用矢量三重积：

$$
\boldsymbol{a} \times (\boldsymbol{b} \times \boldsymbol{c}) = (\boldsymbol{a} \cdot \boldsymbol{c}) \boldsymbol{b} - (\boldsymbol{a} \cdot \boldsymbol{b}) \boldsymbol{c}
$$

$$
\begin{aligned}
\boldsymbol{y} &= \cos^{2} \frac{\theta}{2} \boldsymbol{x} + 2 \sin \frac{\theta}{2} \cos \frac{\theta}{2} \boldsymbol{v} \times \boldsymbol{x}+ 2 \sin^{2} \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{x}) \boldsymbol{v} - \sin^{2} \frac{\theta}{2} (\boldsymbol{v} \cdot \boldsymbol{v}) \boldsymbol{x} \\
&= \left(\cos^{2} \frac{\theta}{2} - \left| v \right|^{2} \sin^{2} \frac{\theta}{2}\right) \boldsymbol{x} + (1 - \cos \theta) (\boldsymbol{v} \cdot \boldsymbol{x}) \boldsymbol{v} + \sin \theta (\boldsymbol{v} \times \boldsymbol{x})
\end{aligned}
$$

转化成正交基

$$
\boldsymbol{A} = \begin{bmatrix} \boldsymbol{a}_{1} & \boldsymbol{a}_{2} & \boldsymbol{a}_{3} \end{bmatrix} = \begin{bmatrix} \boldsymbol{x} - \dfrac{\boldsymbol{x} \cdot \boldsymbol{v}}{\boldsymbol{v} \cdot \boldsymbol{v}}\boldsymbol{v} & \boldsymbol{v} & \boldsymbol{v} \times \boldsymbol{x} \end{bmatrix}
$$

的线性组合。

$$
\begin{aligned}
\boldsymbol{y} &= \left( \cos^{2} \frac{\theta}{2} - \boldsymbol{v} \cdot \boldsymbol{v} \sin^{2} \frac{\theta}{2} \right) \left( \boldsymbol{a}_{1} + \frac{\boldsymbol{x} \cdot \boldsymbol{v}}{\boldsymbol{v} \cdot \boldsymbol{v}} \boldsymbol{a}_{2} \right) + (1 - \cos \theta) (\boldsymbol{v} \cdot \boldsymbol{x}) \boldsymbol{a}_{2} + \sin \theta \boldsymbol{a}_{3} \\
&= \boldsymbol{A} \cdot \begin{bmatrix} \cos^{2} \dfrac{\theta}{2} - \boldsymbol{v} \cdot \boldsymbol{v} \sin^{2} \dfrac{\theta}{2} \\ (1 - \cos \theta)(\boldsymbol{v} \cdot \boldsymbol{x}) + \cos^{2} \dfrac{\theta}{2} \dfrac{\boldsymbol{x} \cdot \boldsymbol{v}}{\boldsymbol{v} \cdot \boldsymbol{v}} - \sin^{2} \dfrac{\theta}{2} \boldsymbol{x} \cdot \boldsymbol{v} \\ \sin \theta \end{bmatrix} \\
&= \boldsymbol{A} \cdot \begin{bmatrix} \cos^{2} \dfrac{\theta}{2} - \boldsymbol{v} \cdot \boldsymbol{v} \sin^{2} \dfrac{\theta}{2} \\ \cos^{2} \dfrac{\theta}{2} \dfrac{\boldsymbol{x} \cdot \boldsymbol{v}}{\boldsymbol{v} \cdot \boldsymbol{v}} + \sin^{2} \dfrac{\theta}{2} \boldsymbol{x} \cdot \boldsymbol{v} \\ \sin \theta \end{bmatrix} \\
&= \boldsymbol{A} \begin{bmatrix}
\cos^{2} \dfrac{\theta}{2} - \left| \boldsymbol{v} \right|^{2} \sin^{2} \dfrac{\theta}{2} \\
\dfrac{\boldsymbol{x} \cdot \boldsymbol{v}}{\left| \boldsymbol{v} \right|^{2}} \left( \cos^{2} \dfrac{\theta}{2} + \left| \boldsymbol{v} \right|^{2} \sin^{2} \dfrac{\theta}{2} \right) \\
\sin \theta
\end{bmatrix}
\end{aligned} 
$$
