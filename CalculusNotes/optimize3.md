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

---

## 模型预测控制算法

---

模型预测控制：求解一个开环最优的控制问题。

- 预测模型：根据历史信息归纳的规律，可以用来预测未来的值。
- 滚动优化：每一个时间单位，重新根据新的信息计算最优控制策略。
- 反馈校正：根据输出值和模型预测值的误差对模型进行修正。

每一个采样周期执行一个优化问题

- 目标函数：未来几步和理想状态的偏差最小，同时要求控制信号和标准信号的偏差最小

---

### 线性约束模型预测控制

---

$$
\begin{align*}
    & \boldsymbol{x}_{k+1} = \boldsymbol{A} \boldsymbol{x}_{k} + \boldsymbol{B} \boldsymbol{u}_{k} \\
    & \boldsymbol{y}_{k} = \boldsymbol{C} \boldsymbol{x}_{k}
\end{align*}
$$

$$
\begin{align*}
    & \text{minimize} && \boldsymbol{x}_{N}^{T} \boldsymbol{P} \boldsymbol{x}_{N} + \sum_{k=0}^{N-1}(\boldsymbol{x}_{k}^{T} \boldsymbol{Q} \boldsymbol{x}_{k} + \boldsymbol{u}_{k}^{T} \boldsymbol{R} \boldsymbol{u}_{k}) \\
    & \text{subject to} && \boldsymbol{u}_{min} \leq \boldsymbol{u}_{k} \leq \boldsymbol{u}_{max} \\
    & && \boldsymbol{y}_{min} \leq \boldsymbol{y}_{k} \leq \boldsymbol{y}_{max}
\end{align*}
$$

---

转换决策变量

$$
\begin{align*}
    & \boldsymbol{z} = \begin{bmatrix}
        \boldsymbol{u}_{0} \\ 
        \boldsymbol{u}_{1} \\ 
        \vdots \\ 
        \boldsymbol{u}_{N-1}
    \end{bmatrix}
\end{align*}
$$

变成新的优化问题

$$
\begin{align*}
    & \text{min} && \frac{1}{2} \boldsymbol{x}_{t}^{T} \boldsymbol{Y} \boldsymbol{x}_{t} + \boldsymbol{x}_{t}^{T} \boldsymbol{F}^{T} \boldsymbol{z} + \frac{1}{2} \boldsymbol{z}^{T} \boldsymbol{Q} \boldsymbol{z} \\
    & \text{s.t.} && \boldsymbol{G}\boldsymbol{z} \leq \boldsymbol{W} + \boldsymbol{S}\boldsymbol{x}_{t}
\end{align*}
$$

---

### 二次优化的求解

---

快速梯度法
