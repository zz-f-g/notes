# Contrastive Loss

[Chopra et al. 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)

用来作为降维、聚类的损失函数。其中，降维需要用到编码器：

$$
G_{W}: \mathbb{R}^{D} \rightarrow \mathbb{R}^{d} ( D \gg d)
$$

在嵌入维度上的距离

$$
D_{W} (\boldsymbol{X}_{1}, \boldsymbol{X}_{2}) = \Vert G_{W}(\boldsymbol{X}_{1}) - G_{W}(\boldsymbol{X}_{2})\Vert_{2}
$$

当然也可以使用曼哈顿距离或 Cosine similarity（向量夹角）。

如何构造损失函数呢？

对于两个样本 $\boldsymbol{X}_{1}, \boldsymbol{X}_{2}$，用标签 $Y$ 表征它们是否属于同一类（离散值）或相似度（连续值）。越相似，$Y$ 越小。

$$
L[\boldsymbol{W}; (Y, \boldsymbol{X}_{1}, \boldsymbol{X}_{2})^{i}] = (1 - Y) L_{S} (D^{i}_{W}) + Y L_{D} (D^{i}_{W})
$$

其中 $L_{S}$ 为相似损失，$L_{D}$ 为不相似损失。

$$
\begin{aligned}
L_{S} &= \frac{1}{2} D_{W}^{2} \\
L_{D} &= \frac{1}{2} \left\{ \mathrm{max} (0, m - D_{W}) \right\}^{2}
\end{aligned}
$$

所谓的 $m$ 是 margin，期望所有同一类（相似）的点距离都小于 $m$ 而不同类（不相似）的点距离都大于 $m$. 在之后作预测的时候就可以将待预测点周围距离小于 $m$ 的点看作同类。

为什么要设置 margin？否则网络 $G_{W}$ 会将所有点都映射到同一个点来降低损失函数。设置 margin 可以在拉近同一类点的同时推远不同的点。
