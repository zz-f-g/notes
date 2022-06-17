我认为我们一般说的电容，都是电路中的电容。

![](https://megaeshop.pk/media/catalog/product/2/1/2193-00_1_.jpg)

从电路设计者的角度来说，电压是因，电荷量是果。我们用比值定义法，将结果作为分子，将原因作为分母，得到的比值作为电容。电容的值越大，反映出原因对结果的作用越大。

$$
C = \frac{Q}{U}
$$

我猜测题主可能是第一次接触到有关电容的定义。在国内的教科书中，电容常常是在静电学的相关板块引入。而在学习这部分的内容时，我们常常认为电荷是产生电场（电势）的原因，所以会有题主的疑问。其实这各概念也有相应的定义，有一种定义是**电势系数**。

对于固定位置的多个导体，分别用下标 1, 2, ..., n 表示。取无穷远处为电势零点，每一个导体上的电势和导体上的电荷量为线性关系：

$$
\begin{align*}
    & \begin{bmatrix}
        U_{1} \\ 
        U_{2} \\ 
        \vdots \\ 
        U_{n}
    \end{bmatrix} = \begin{bmatrix}
        p_{11} & p_{12} & \cdots & p_{1n} \\ 
        p_{21} & p_{22} & \cdots & p_{2n} \\ 
        \vdots & \vdots & \ddots & \vdots \\ 
        p_{n1} & p_{n2} & \cdots & p_{nn}
    \end{bmatrix} \begin{bmatrix}
        Q_{1} \\ 
        Q_{2} \\ 
        \vdots \\ 
        Q_{n}
    \end{bmatrix}
\end{align*}
$$

写成矩阵的形式，就有：

$$
\begin{align*}
    & \boldsymbol{U} = \boldsymbol{P} \boldsymbol{Q}
\end{align*}
$$
