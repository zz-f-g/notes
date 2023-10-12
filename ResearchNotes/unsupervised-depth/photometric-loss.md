# Photometric Loss

损失函数，用于比较两张图片的相似性。被比较的两张图片一般是单目多帧序列中的一张图片和被重投影（warp）的另一张相邻图片，采用 [grid sample](gridsample.md) 方法。

计算损失函数时，不但要考虑亮度的绝对误差，也要考虑照明条件不同下的问题。后者使用 [SSIM](SSIM.md) 选自 [Image quality assessment: from error visibility to structural similarity | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/1284395/) 损失处理。

$$
L_{p} = \frac{1}{\left| V \right|} \sum_{p \in V} \left( \lambda_{i} \Vert I_{a}(p) - I'_{a}(p) \Vert_{1} + \lambda_{s} \frac{1 - SSIM_{aa'}(p)}{2} \right)
$$

$V$ 表示从 $I_{a}$ 投影到 $I_{b}$ 平面的有效点（投影的方向问题可见 [grid sample](gridsample.md)）。采用 1 范数可以减小离群点的影响。
