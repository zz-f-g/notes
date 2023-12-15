# Triplet Loss

- anchor sample $(\boldsymbol{x}_{a}, y_{a})$
- positive sample $(\boldsymbol{x}_{p}, y_{p}), y_{p} = y_{a}$
- negative sample $(\boldsymbol{x}_{n}, y_{n}), y_{n} \neq y_{a}$

Embedded space distance: $D_{f_{\theta}}$

$$
L_{tirplet} = \mathrm{max} \left( 0, D^{2}_{f_{\theta}}(\boldsymbol{x}_{a}, \boldsymbol{x}_{p}) - D^{2}_{f_{\theta}} (\boldsymbol{x}_{a}, \boldsymbol{x}_{n}) + \alpha \right)
$$

和 [contrastive loss](./contrastive-loss.md) 相似，设置 margin $\alpha$ 是为了防止 $f_{\theta}$ 将所有数据映射到同一个点。
