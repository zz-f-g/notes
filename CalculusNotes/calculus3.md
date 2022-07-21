# Calculus3

---

## Concepts

---

### 空间解析几何与向量代数

---

关于向量积（叉积）概念的若干阐述（来自 [3B1B Video](https://www.bilibili.com/video/BV1ys411472E?p=12&vd_source=85e71cac8676a765a42642c92ce0cd1b)）

从来就不存在什么二元叉积，一切都来自于一个 3 阶行列式

$$
\boldsymbol{A} = \begin{bmatrix} \boldsymbol{w} & \boldsymbol{u} & \boldsymbol{v} \end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$

如果把 $\boldsymbol{w}$ 作为自变量，矩阵 $\boldsymbol{A}$ 的行列式的值作为因变量构建函数：

$$
f(\boldsymbol{w}) = \mathrm{det}\left(\begin{bmatrix} \boldsymbol{w} & \boldsymbol{u} & \boldsymbol{v} \end{bmatrix}\right)
$$

根据行列式的性质，可以证明这是一个线性变换，即

- $f(\boldsymbol{0}) = 0$
- $f(a \boldsymbol{x} + b \boldsymbol{y}) = a f(\boldsymbol{x}) + b f(\boldsymbol{y})$

根据线性变换和矩阵乘法的等价性：

$$
f(\boldsymbol{w}) = \boldsymbol{p}^{T} \boldsymbol{w}
$$

其中，$\boldsymbol{p}^{T} \in \mathbb{R}^{1 \times 3}$ 是一个线性算子，记：

$$
\boldsymbol{p} = \boldsymbol{u} \times \boldsymbol{v}
$$

叉积对应的几何意义是平行六面体的体积。

$$
\mathrm{det}(\begin{bmatrix} \boldsymbol{w} & \boldsymbol{u} & \boldsymbol{v} \end{bmatrix}) = (\boldsymbol{u} \times \boldsymbol{v}) \cdot \boldsymbol{w}
$$

那么如何计算 $\boldsymbol{p}$ 呢？

$$
\begin{align*}
    \boldsymbol{p} & = p_{1} \cdot \hat{i} + p_{2} \cdot \hat{j} + p_{3} \cdot \hat{k} \\
    & = \boldsymbol{p}^{T} \begin{bmatrix} \hat{i} \\ \hat{j} \\ \hat{k} \end{bmatrix} \\
    & = f \left( \begin{bmatrix} \hat{i} \\ \hat{j} \\ \hat{k} \end{bmatrix}\right) \\
    & = \mathrm{det} \left( \begin{bmatrix}
        \hat{i} & u_{1} & v_{1} \\ 
        \hat{j} & u_{2} & v_{2} \\ 
        \hat{k} & u_{3} & v_{3} \\ 
    \end{bmatrix}\right)
\end{align*}
$$

---

异面直线距离

$$
\begin{align*}
    & L_{1}: \frac{x - x_{1}}{m_{1}} = \frac{y - y_{1}}{n_{1}} = \frac{z - z_{1}}{l_{1}} && M_{1}: \begin{bmatrix} x_{1} & y_{1} & z_{1} \end{bmatrix}^{T} \\
    & L_{2}: \frac{x - x_{2}}{m_{2}} = \frac{y - y_{2}}{n_{2}} = \frac{z - z_{2}}{l_{2}} && M_{2}: \begin{bmatrix} x_{2} & y_{2} & z_{2} \end{bmatrix}^{T} \\
\end{align*}
$$

异面直线距离就是和两直线上的两点距离在两个直线都垂直的方向上的投影。

$$
\begin{align*}
    & \boldsymbol{n} = \boldsymbol{s}_{1} \times \boldsymbol{s}_{2} \\
    & d = \frac{1}{\Vert \boldsymbol{n} \Vert} \boldsymbol{n}^{T} (\boldsymbol{r}_{1} - \boldsymbol{r}_{2}) \\
    & d = \frac{1}{\Vert s_{1} \times s_{2} \Vert} \left| \begin{matrix} \boldsymbol{r}_{1} - \boldsymbol{r}_{2} & \boldsymbol{s}_{1} & \boldsymbol{s}_{2} \end{matrix} \right|
\end{align*}
$$

---


