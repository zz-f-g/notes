# Grid Sample

torch 库函数：[torch.nn.functional.grid_sample — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)

```python
torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
```

In the spatial (4-D) case, for `input` with shape $(N,C,H_{in}​,W_{in}​)$ and `grid` with shape $(N,H_{out}​,W_{out}​,2)$, the output will have shape $(N,C,H_{out}​,W_{out}​)$.

grid 参数表示采样的方式，注意到它的 H, W 和输出一致。`grid[N, H, W]` 是一个二维向量，存储了输入图像的像素坐标 `[x, y]`，意义是输出图像在 `[H, W]` 处的值由输入图像在 `[x, y]` 的值决定。注意到 `[x, y]` 不一定刚好对应一个像素，计算输入图像在 `[x, y]` 处的值需要通过指定的采样模式来实现。`[x, y]` 的数据格式是归一化的，`[-1, -1]` 表示左上角，`[1, -1]` 表示右上角。

具体的采样模式默认是双线性插值 bilinear.

Grid sample 的目的是实现诸如光流和重投影任务中图像的重建过程。通过光流和重投影的公式可以得到 `grid` 参数，这样就能从上一帧图片（reference）重建下一帧（target），从而计算损失函数等。当然除了 RGB 的重建还可以作深度的重建，但是忽略相机的位移直接用上一帧的深度多少不太合理。

我认为可以在 grid sample 的基础上进一步根据位姿变化信息 $[\boldsymbol{R}, \boldsymbol{t}]$ （在 reference 看来 target 相机坐标轴的坐标）作进一步微调。

深度的微调方法比较简单，可以直接利用相机公式得到。

$$
z_{t}\widetilde{\boldsymbol{p}}_{t} = - \boldsymbol{R}^{-1} \boldsymbol{t} + \boldsymbol{R}^{-1} z_{r} \boldsymbol{K}^{-1} \widetilde{\boldsymbol{p}}_{r}
$$

其中 $z_{r}$ 表示上游任务得到的 reference 帧的深度，它被 grid sample，$z_{t}$ 表示微调过后的值，作为 target 中的结果。

RGB 的微调方法比较复杂，需要考虑光源方向，物体表面法向量的参数，结合位姿信息变化考虑。或许计算机图形学的部分概念有助于这一问题的解决。
