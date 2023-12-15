# Deep Metric Learning

目标：评估数据样本之间的相似度（例如判断两张头像是不是同一个人）

- data: $\mathscr{X} = \begin{bmatrix}\boldsymbol{x}_{1} & \cdots & \boldsymbol{x}_{n} \end{bmatrix}$
- embedding neural model (or feature extractor): $f_{\theta}(\cdot) : \mathscr{X} \rightarrow \mathbb{R}^{n}$
- distance: $D: \mathbb{R}^{n}, \mathbb{R}^{n} \rightarrow \mathbb{R}$
- labels: $\mathscr{Y} = \begin{bmatrix} y_{1} & \cdots & y_{n}\end{bmatrix}$

如何选择损失函数使得

$$
y_{1} = y_{2}, D(f_{\theta}(\boldsymbol{x}_{1}), f_\theta(\boldsymbol{x}_{2})) \downarrow
$$

传统方法：

- [t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)
- [UMAP](https://arxiv.org/abs/1802.03426)

学习方法：

- [Contrastive Loss](./contrastive-loss.md)
