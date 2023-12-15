# 梯度下降优化策略

本文总结了神经网络利用反向传播得到的梯度对网络参数进行优化的几种策略，探讨了不同策略的优劣以及适用条件。

神经网络模型

$$
y = f(\boldsymbol{x}, \theta)
$$

训练数据集 $\left\{ (\boldsymbol{x}_{i}, \boldsymbol{y}_{i}): i = 1, \cdots, N \right\}$

## Gradient Descent

$$
\theta_{t+1} = \theta_{t} - \alpha \frac{1}{N} \sum_{i=1}^{N} \frac{\partial L[\boldsymbol{y}_{i}, f(\boldsymbol{x}_{i}, \theta_{t})]}{ \partial \theta_{t}}
$$

劣势：每一次迭代都需要遍历整个数据集，对内存不友好。

## Stochastic Gradient Descent

每一次迭代随机选择 $k$

$$
\theta_{t+1} = \theta_{t} - \alpha \frac{\partial L[\boldsymbol{y}_{k}, f(\boldsymbol{x}_{k}, \theta_{t})]}{\partial \theta_{t}}
$$

- 优势：对内存友好，易于用计算机实现，参数更新快，增量式学习，引入随机性增加泛化性能。
- 劣势：每次随机选择一个数据，算法收敛不稳定。

## Mini-batch Stochastic Gradient Descent

每一次迭代随机选择数据集子集 $S_{t}$，大小为 $k$

$$
\theta_{t+1} = \theta_{t} - \alpha \frac{1}{k} \sum_{i \in S_{t}} \frac{\partial L[\boldsymbol{y}_{k}, f(\boldsymbol{x}_{k}, \theta_{t})]}{\partial \theta_{t}}
$$

```python
# loop through batches
for (inputs, labels) in data_loader:

    # extract inputs and labels
    inputs = inputs.to(device)
    labels = labels.to(device)

    # passes and weights update
    with torch.set_grad_enabled(True):
        
        # forward pass 
        preds = model(inputs)
        loss  = criterion(preds, labels)

        # backward pass
        loss.backward() 

        # weights update
        optimizer.step()
        optimizer.zero_grad()
```

优势：算法收敛更稳定更快，手动调整 batch size 使之符合并行计算设备内存可以增加并行化效率。

## GradientAccumulation

[来自 Nikita Kozodoi 博客的介绍](https://kozodoi.me/blog/20210219/gradient-accumulation)

```python
# batch accumulation parameter
accum_iter = 4  

# loop through enumaretad batches
for batch_idx, (inputs, labels) in enumerate(data_loader):

    # extract inputs and labels
    inputs = inputs.to(device)
    labels = labels.to(device)

    # passes and weights update
    with torch.set_grad_enabled(True):
        
        # forward pass 
        preds = model(inputs)
        loss  = criterion(preds, labels)

        # normalize loss to account for batch accumulation
        loss = loss / accum_iter 

        # backward pass
        loss.backward()

        # weights update
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
```

它利用了 ``loss.backward()`` 会将新的梯度叠加到已有梯度累计的性质。参数更新并不发生在每个 batch 中，而是每 ``accum_iter`` 才进行一次参数更新和优化器清零。

$$
\theta_{t+1} = \theta_{t} - \alpha \sum_{j=1}^{n} \left\{ \frac{1}{k} \sum_{i \in S_{t, j}} \frac{\partial L[\boldsymbol{y}_{k}, f(\boldsymbol{x}_{k}, \theta_{t})]}{\partial \theta_{t}} \right\}
$$

优势：模仿更大的 batch size，增加收敛速度，对于大模型和数据集中单个数据较大的情况较为有用。
