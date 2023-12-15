# 解决 Cuda 内存不足的问题

内存不足通常有三个方面的因素：

1. 网络参数太多
2. 显卡显存太小
3. 训练过程中的数据大小太大

前两个方面的因素往往取决于网络和硬件，不好更改。解决第三方面的因素可以部分减少内存，治标不治本。目前实践中有以下方法：

1. 调小 batch size. 这导致的算法问题可以通过[梯度累计](../algorithm/gdoptim.md#GradientAccumulation)的方法补偿；
2. 减小图片分辨率；
3. 观察网络结构，将 ReLU 模块 inplace 改为 True. 当这个数据没有被多个网络同时作为输入的时候可以采用这种方法，[参见](https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948)；
4. 优化数据 forward 方式，避免大数据 clone，不用就释放。
