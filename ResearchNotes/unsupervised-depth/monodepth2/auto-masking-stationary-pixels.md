# Auto-Masking Stationary Pixels

Second contribution from Monodepth2: a simple auto-masking method that filters out pixels which do not change appearance from one frame to the next in the sequence.

在[贡献一](per-pixel-minimum-loss.md)的基础上，叠加了掩码：

$$
\mu = \left[ \mathrm{min}_{r} L_{p}(I_{t}, I_{r \rightarrow t}) < \mathrm{min}_{r} L_{p}(I_{t}, I_{r}) \right]
$$

`[]` 是 Iverson 括号，当括号中条件满足时值为 1，否则为 0，即过滤掉。

$\mu = 0$ 的三种情况：

1. 相机静止不动
2. 相机和动态物体保持大约相对静止
3. 低纹理区域
