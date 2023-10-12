# Smoothness Loss

损失函数，用于平衡深度估计结果和 RGB 图像的关系。就是让深度的变化不要太剧烈而频繁，而是跟 RGB 图像的变化相关。损失函数的数学表达式并不具有逻辑上的意义，只是一种工程技巧。

$$
L_{s} = \sum_{p} \left( e^{-\nabla I_{a}(p)} \cdot \nabla D_{a}(p) \right)^{2}
$$

```python
# from git@github.com:JiawangBian/sc_depth_pl.git
def compute_smooth_loss(tgt_depth, tgt_img):
    def get_smooth_loss(disp, img):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth, tgt_img)

    return loss
```
