# Geometry Consistency Loss

来自 [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video](https://arxiv.org/pdf/1908.10553.pdf) Contributions 1.

这个损失函数是为了解决无监督深度估计中的尺度不一致性问题（scale inconsistency）：由于单目相机缺乏距离量纲，因此在没有参照物和监督的情况下根本无法获得场景的绝对距离信息，只能获得相对的距离。如果每一帧的相对距离没有相同的基准，那么结果没有意义。它还会导致 PoseNet 得到的 $\boldsymbol{T}$ 的绝对值不一样。

这个论文采用了将深度差值作为损失函数的方法纠正相邻两帧相对距离基准的差值。

$$
\begin{aligned}
L_{GC} &= \frac{1}{\left| V \right|} \sum_{p \in V} D_{diff}(p) \\
&= \frac{1}{\left| V \right|} \sum_{p \in V} \frac{\left| D_{b}^{a}(p) - D'_{b}(p) \right|}{D_{b}^{a} (p) + D'_{b}(p)}
\end{aligned}
$$

使用两个深度的和作为归一化因子。
