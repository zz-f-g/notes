这是一个没有复杂的向量运算的证明过程，希望各位看得愉快。

这毕竟是一个物理问题，除了基本的微积分以外还需要一些物理知识，列举如下：

- 牛顿第二定律
- 万有引力定律
- 角动量守恒定理
- 极坐标下的加速度形式

对于前两点，有过高中教育经历的都明白，这里不再赘述。下面取行星的轨道平面为极坐标平面，讨论第四点和第三点。

以静止恒星为极坐标极点，某一个方向取为极轴，建立极坐标系，行星的位置可以表示为。

$$
\left\{\begin{aligned}
    & r = r(t) \\
    & \theta = \theta(t)
\end{aligned}\right.
$$

极坐标下的加速度可以通过求导得到，这里略去推导过程，详见北大出版社舒幼生《力学》。

$$
\begin{align*}
    & \boldsymbol{a}_{r} = (\ddot{r} - r^{2} \dot{\theta}) \hat{r} \\
\end{align*}
$$

角动量守恒定理可以在极坐标下表示为：

$$
\begin{align*}
    & h := r^{2} \dot{\theta} = C \\
    & \dot{\theta} = \frac{\partial \theta}{\partial t} = \frac{h}{r^{2}}
\end{align*}
$$

根据径向列写动力学方程

$$
\begin{align*}
    & - \frac{GMm}{r^{2}} \hat{r} = m \boldsymbol{a}_{r} \\
    & \ddot{r} - r \dot{\theta}^{2} + \frac{GM}{r^{2}} = 0
\end{align*}
$$

对 $\ddot{r}$ 做变换

$$
\begin{align*}
    \ddot{r} &= \frac{\partial}{\partial t} \left( \frac{\partial r}{\partial t}\right) = \dot{\theta} \cdot \frac{\partial}{\partial \theta}\left( \frac{\partial r}{\partial \theta} \cdot \dot{\theta}\right) \\
    & = \frac{h}{r^{2}} \frac{\partial}{\partial \theta} \left( \frac{\partial r}{\partial \theta} \cdot \frac{h}{r^{2}}\right) \\
    & = - \frac{h^{2}}{r^{2}} \frac{\partial^{2}}{\partial \theta^{2}} \left(\frac{1}{r}\right)
\end{align*}
$$

变量代换，设 $u = \frac{1}{r}$

$$
\begin{align*}
    & \ddot{r} = - h^{2} u^{2} \frac{\partial^{2}u}{\partial \theta^{2}} \\
    & -h^{2}u^{2} \frac{\partial^{2}u}{\partial \theta^{2}} - r \left(\frac{h}{r^{2}}\right)^{2} + \frac{GM}{r^{2}}= 0 \\
    & \frac{\partial^{2}u}{\partial \theta^{2}} + u = \frac{GM}{h^{2}}
\end{align*}
$$

可以看出这是一个简谐运动的微分方程，如果我们取 $\dot{r} = 0$ 时的 $\theta = 0$ 则可以得到一个形式简单的解

$$
\begin{align*}
    & u = \frac{GM}{h^{2}} + A \cos \theta \\
    & r = \frac{1}{\frac{GM}{h^{2}} + A \cos \theta} \\
    & r = \frac{\frac{h^{2}}{GM}}{1 + \frac{Ah^{2}}{GM} \cos \theta}
\end{align*}
$$

这正是双曲线的极坐标方程，对于离心率合适的情况，是椭圆轨道。

轨道参数的值可以由行星的能量和角动量唯一地确定。