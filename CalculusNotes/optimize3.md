# optimize-experiment

---

## 线性方程组

---

列满秩矩阵作为系数矩阵，齐次解唯一，只有平凡解。

---

## MATLAB 求解线性方程组

---

```matlab
R = rand(m, n); % random in [0, 1]
A = linspace(x1, x2, n); % n-1 等分
dev(A); % determinant
inv(A): % A^{-1}
```

---

矩阵除法

```matlab
A\B; % AX = B => inv(A) * B
B/A; % XA = B => B * inv(A)
```

但是当 $A$ 不是方阵时，矩阵除法会有所变化。

- 方程无解，给出 $\Vert AX - B \Vert$ 最小的最小二乘解。
- 方程无穷解，给出非零元素最少的解。

---

## 最小二乘法

---

递推最小二乘算法（recursive least square）

按照得到的数据一个一个计算。

---

函数程序

```matlab

```

---

## Karzmarz

