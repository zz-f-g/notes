# Embedded System

---

- 10% 平时成绩
- 40% 期末考试
- 10% 小作业
- 40% 大作业

---

学习目标：

- 可以重建工作的系统
- 掌握 CPU 的工作过程，搭建解决实际问题的系统
- 掌握计算机的工作过程

---

## 数字电路基础

---

 在模拟电子电路中，三极管工作在放大区。

 在数字电路中，三极管工作在截止区或饱和区。因为：
 
- 放大区工作的功耗较大
- 截止区和饱和区构成三极管的两种状态

> 能承载的信息量和信噪比有关。——香农

一段波形是数字信号还是模拟信号，不取决于它的波形，而取决于人如何理解它。

数字信号牺牲信息量来获取较高的抗干扰能力。还能延时，压缩等等。

---

计算机是一套独特的规律科学？开始人类根据自己的社会经验抽象出的规则定义的机器？

后者。揣摩当初计算机设计者的思想。

写程序本质上就是翻译。

```mermaid
graph LR
human_language --programer--> machine_language
```

---

一些硬件储存信息的方法

- 磁盘：用 N/S 磁性来表征二进制
- 光盘：光盘的反射方向，收到光代表 1，未收到代表 0.

---

## 门电路

---

NOT

![inverter](https://circuitglobe.com/wp-content/uploads/2015/12/NOT-GATE-FIG-5-compressor.jpg)

一个不驮载直流信号的基本放大电路就是非门。

- 当输入高电平，三极管导通处于饱和区，输出低电平。
- 当输入低电平，三极管截止，输出高电平。

---

![circuit](https://boxbase.org/entries/2016/may/2/robot-electronics/not-gate.jpg)

---

AND

![and gate](https://tse1-mm.cn.bing.net/th/id/OIP-C.Z1u2ndIUN8qLVjoO9OXqUgHaCs?pid=ImgDet&rs=1)

电路图：两个三极管串联。

---

OR

电路图：两个三极管并联

---

通过改接输出的位置，可以制造出与非门，或非门。

---

XOR

![xor](https://tse2-mm.cn.bing.net/th/id/OIP-C.UZQCEnZglOQtoz30EpMNiAAAAA?w=186&h=180&c=7&r=0&o=5&dpr=1.56&pid=1.7)

---

![circuit](https://gss0.baidu.com/94o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/8c1001e93901213fb831144d5ae736d12f2e9524.jpg)
