# Tensor 存储方式分析

## stride

密集矩阵的存储方式是相通的，存在一块连续的内存（显存）区域以随机存取的方式存储矩阵中的数据。Pytorch Tensor 和 C 高维数组存储方式类似：越靠后的维度变化引起的地址变化越小。这个性质可以由属性 ``torch.Tensor.stride`` （字面意义是步长，类似卷积中的概念）看出。

```python
a = torch.rand(4, 3, 192, 640)
a.stride()
# (368640, 122880, 640, 1)
```

对应维度的 stride 值表示该维度加一对应的地址变化值。（粗浅理解）

## metadata

除了矩阵中的数据之外，还会存储矩阵的信息作为 Metadata，它们决定了数据的组织方式。包括常见的比如（来自 ChatGPT）：

 1. **形状（Shape）：**
    - `tensor.shape`：返回张量的形状，是一个元组。
2. **数据类型（Data Type）：**
    - `tensor.dtype`：返回张量的数据类型，如 `torch.float32`。
3. **设备（Device）：**
    - `tensor.device`：返回张量所在的设备，如 `cuda:0` 表示在 GPU 0 上，`cpu` 表示在 CPU 上。
4. **布局（Layout）：**
    - `tensor.layout`：返回张量的布局（layout），通常是 `torch.strided`。
5. **存储（Storage）：**
    - `tensor.storage()`：返回张量底层存储的实体，可以通过存储直接访问张量的数据。
6. **梯度（Gradient）：**
    - `tensor.grad`：返回张量的梯度。如果张量是叶子张量（leaf tensor），它将具有梯度，否则为 `None`。
7. **版本号（Version）：**
    - `tensor.version`：张量的版本号，每次执行 in-place 操作时，版本号会增加。 ==（实测并无此属性）==
8. **是否需要梯度（Requires Gradient）：**
    - `tensor.requires_grad`：返回一个布尔值，表示张量是否需要梯度计算。

## continguous

使用 Tensor 时常常改变 Tensor 的形状，改变形状的方式可以依据改变后 Tensor 的 ``stride`` 属性分成两类：

1. ``stride`` 仍然递减且所有维度上 ``stride`` 的值等于该位置及以后维度值的乘积。``view``，``reshape`` 属于此类。
2. ``stride`` 不满足递减关系。``permute``，``transpose`` 属于此类。

``stride`` 的两种属性可以通过 Tensor 的属性 ``is_contiguous`` 判断，若 True 则为第一类，否则反之。对于第二类 Tensor，可以调用 ``contiguous`` 方法**深拷贝**一份副本使得存储是连续的。[^1]

[^1]: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch

## view vs reshape

``view`` 和 ``reshape`` 的差异[^2]：如果使用 ``view`` 对 Tensor 的某几个维度合并时，这几个维度必须是“连续的”，也就是这几个维度上 ``stride`` 值相邻。否则报错：

```
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

[^2]: https://discuss.pytorch.org/t/whats-the-difference-between-torch-reshape-vs-torch-view/159172/4

当使用 ``reshape`` 作类似的操作时，如果遇到维度“不连续”的情况，会**深拷贝**连续的副本再 ``reshape``。==尽量避免使用 ``reshape``==。

## permute vs transpose

``permute`` 将原来的维度顺序（输入的参数值）改变成新的顺序（输入参数的位置），功能比只能交换两个维度的 ``transpose`` 更强大。[^3]

[^3]: https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/#transpose-and-permute
