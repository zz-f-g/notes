# Self-discovered Mask

来自 [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video](https://arxiv.org/pdf/1908.10553.pdf) Contributions 2.

它利用了 [Contributions 1](./geometry-consistence-loss.md) 的 $D_{diff}$. 掩码 $M$ 和 $D_{diff}$ 负相关，所以直接：

$$
M = 1 - D_{diff}
$$

考虑到 $D_{diff} \in [0, 1]$，这样实际构建出来的是一个权重掩码（weight mask）。

怎么理解呢？在训练一段时间后，scale inconsistency 基本消除，那么引起 $D_{diff}$ 的区域就是不满足静态场景假设的 (1) 遮挡；(2) 动态物体；(3) 预测不准的地方。所以直接作为 Mask 就可以了。

引入掩码以后就要对 photometric loss 作修改：

$$
L_{p}^{M} = \frac{1}{\left| V \right|} \sum_{p \in V} (M(p) L_{p}(p))
$$
