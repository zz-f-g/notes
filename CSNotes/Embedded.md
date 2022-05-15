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

---

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

---

逻辑运算是**状态**的运算，不是电压值的运算。

因此逻辑运算不是只能通过电路实现，其他物理量也可以。

---

数字信号

- 易于储存，压缩
- 损坏以后可以修复（中继器）
- 可以推算传播损失函数，进行整形，恢复为方波

---

加法器

![[images/image20220302102403.png]]

---

True Value Tablet

| A   | B   | $\Sigma$ | $C_{out}$ |
| --- | --- | -------- | --------- |
| 0   | 0   | 0        | 0         |
| 0   | 1   | 1        | 0         |
| 1   | 0   | 1        | 0         |
| 1   | 1   | 0        | 1         |

---

全加器

```mermaid
graph LR
Carry_bit --> adder2 --O--> Output
A --> adder1 --O--> adder2 --C--> OR --> Output_carry
B --> adder1 --C--> OR
```

---

所有的运算都是由元运算（加减乘除与或非）张成的空间。

$$
\sin x = \frac{x}{1!} - \frac{x^{3}}{3!} + \frac{x^{5}}{5!} - \cdots
$$

---

时钟信号：方波，和输入输出信号做与运算。

当时钟信号值为 0，输入输出都为 0.

---

## 赋值的物理实现

---

```cpp
char a = 1;
```

a 的存储单元的前七个改成 0，后一个改成 1.

---

```cpp
#include <iostream>
using namespace std;
int main()
{
    char a[] = {1,2,3,4};
    short *I = (int *)a;
    cout << hex << *I << endl; // 0x201
    return 0;
}
```

低地址低字节

```cpp
int a = 80;
```

内存中：``50 00 00 00``

---

``signed`` and ``unsigned``

两种数的比较电路不同。JLE,JBE

```cpp
#include <iostream>
using namespace std;
int main()
{
    int a = 0;
    if (a>10)
        a = 20;

    unsigned int a = 0;
    if (a>10)
        a = 20;
    return 0;
}
```

这两种比较的 Assembly Language 不同。

---

CPU：好几套组合逻辑

如何让 CPU 知道正在调用的逻辑单元？

采用地址，但是考虑到可读性，汇编语言将地址转译成英文字母。

---

浮点数虽然表示的范围大，但它的相对误差是一定的。

也就是说，浮点数在表示较大的整数时，精度远远不如整型。

例如：

1000 0000 和 1000 1000 如果是浮点数中相邻的两个数，1000 0001 就无法用这种浮点数表示。
```cpp
    a + 1 == a
```

---

## C 语言回顾

---

1. 指针和数组
2. 结构体
3. 函数指针和结构体
4. 函数参数传递
5. 宏和预编译

---

数组天然配合循环。

硬件开发时轻易不要用 ``while`` 循环。否则极易进入死循环。

高维数组的转译：``arr[m][n]`` => ``arr[m*n]``

``arr[i][j]`` => ``arr[i*n+j]``

高维数组是计算机程序设计语言的福利：增强代码可读性。其物理存储还是一维形式连续的。

二维数组的折算使得其代码执行效率不如一维数组。

---

```cpp
#include <iostream>
using namespace std;
int main()
{
    int arr[m][n] = {{0}}; // ?
    int *p;
    for (int i=0; i<m; i++)
    {
        p = arr[i];
        for (int j=0; j<n; j++)
            func(p[j]); // 这种代码的执行效率高，不用每次计算乘法
    }
    return 0;
}
```

计算机的维度是组织数据的方式，不是真实的空间维度。

---

``int *p`` & ``int* p&

``p`` 和 ``*p`` 都是变量。

```cpp
    int a = 0;
    int* p1 = &a;
    char b = 'b';
    char* p2 = &b;
    p1++; // +4
    p2++; // +1
```

指针类型的计算根据指针所指的数据类型而确定。

直接给指针赋值非常危险。

```cpp
    int a = 0;
    int* p = &a;
    *(++p) = 1; // 病毒行为
```

---

```cpp
    char bytebuffer[] = { 0x23, 0x34, 0x6d, 0xca };
    int* p = (int *)bytebuffer;
```

---

```cpp
struct XX{
    ...;
}

int main()
{
    int bytebuffer[] = { 1,2,3,4,5,6,7,8 };
    XX* p = (XX*)bytebuffer;
    XX p = *(XX*)bytebuffer;
    return 0;
}
```

对于普遍情形而言，传输地址比传输数据更加轻量化。

---

结构成员对齐

结构体中每个成员分配的最小空间，32 位空间为 4 bytes，64 位空间为 8 bytes。

访问结构体成员，被编译器翻译为地址的**偏移量**。这一行为和访问数组元素的行为本质是一样的。

---

内存泄漏

```cpp
    char * p = new char[8];
    char a = 'a';
    p = &a; // 动态申请的 char[8] 不能再被访问，引起 8 bytes 的内存泄漏
```

有些软件运行时间过长需要重启，就是因为内存泄漏。

```cpp
    delete[] p; // 释放空间，不是删除变量
    p = &a; // 合理合法
```

但是可以这样

```cpp
    char * p = new char[8];
    char * q = p;
    p = &a;
    delete[] q;
```

```cpp
    char * p = new char[8];
    char * q = p;
    delete[] q;
    delete[] p; // 但是这样又会报错，不能再次释放
```

---

变量的生命期

从定义开始，到同级的 ``}`` 结束（全局变量到文件末尾结束）。

每一对 ``{}`` 中是一个栈。

---

函数参数传递

讲变量转入函数，将实参的数据**拷贝**一份放入函数的栈顶。

函数中的形参从栈顶取得相应的数据，形参本身不开辟新的空间。

因此函数如果修改形参的话，改变的是**拷贝的数据**，不改变外部的数据。

当然如果传输的类型是指针，情况就不同了。但是本质都是传递数据，只是数据是地址罢了。

一般来说推荐传入指针，可以添加 ``const`` 关键字防止误修改。

```cpp
int func(const int * a, int * b);
```

---

函数结果返回

```cpp
XX* func()
{
    XX xx = ...;
    return &xx;
}
```

如何理解这个==非常隐蔽的错误==？

> 基本常识：同一个文件中的所有函数都公用同一个栈。

在执行完函数 ``func()`` 后，虽然返回值指针所指的地址还是目标值，但是随着==后续函数的执行，栈区的值被覆盖==，将出现非常隐蔽的错误。

---

``new`` 和 ``delete``  处理的是堆区的数据，因此如果在函数中执行这两个语句，可以。

```cpp
XX* func()
{
    XX* ab = new XX();
    return ab;
}
```

这样返回指针值，不会被后续函数执行的结果所覆盖。

---

函数指针

```cpp
#include <iostream>

struct YY {
    int (*funP)(int a, int b);
};

int Minus(int a, int b)
{
    return a - b;
}
int Multiple(int a, int b)
{
    return a * b;
}

int main()
{
    YY yy;
    yy.funP = Minus; // or &Minus
    std::cout << yy.funP(1, 2) << std::endl;
    yy.funP = Multiple; // or Multiple
    std::cout << yy.funP(1, 2) << std::endl;
    return 0;
}
```

---

位段

```cpp
struct ABC{
    char a : 1; // a: 1bit 0/1
    char g : 7; // g: 7bits
}
```

---

联合

```cpp
#include <iostream>
using namespace std;
typedef union
{
    char a;
    char b;
}utype;

int main()
{
    utype c;
    c.a = 'A';
    cout << c.b << endl;
    return 0;
}
```

---

```cpp
#include <iostream>
using namespace std;
struct bits
{
    unsigned char bit0 : 1;
    unsigned char bit1 : 1;
    unsigned char bit2 : 1;
    unsigned char bit3 : 1;
    unsigned char bit4 : 1;
    unsigned char bit5 : 1;
    unsigned char bit6 : 1;
    unsigned char bit7 : 1;
};

typedef union
{
    char a;
    bits b;
}utype;

int main()
{
    utype c;
    c.a = 1;
    c.b.bit0; // \x01
    return 0;
}
```

先定义的 ``bit0``，位数较低。

---

## 计算机的工作原理

1. 指令系统的构建
2. 内存读写

---

指令系统设计

1. 假设 1Mb 内存：20 bit 地址编码
2. 8 个寄存器：3 bit 地址编码
3. 支持 4 条指令：2 bit 地址编码

读写加减乘除

```
// int a = 5;
write 0xccdde 05
// memory address 0xccdde

// a *= 5;
mul 0xccdde 05
```

---

设计方案二：引入寄存器

```
// a = 10;
write register011 0a // 1+3+8=12 bit
write 0xccdde register001 // 1+20+3=24 bit

// a++;
add register011 // 2+3=5 bit

// a *= 5;
mul register001 5 // 2+3+8=13 bit
write 0xccdde register001 // 1+20+3=24 bit
```

引入寄存器以后，增加了代码执行的效率。在对数据进行运算的时候，将数据暂存到 CPU 内部的寄存器中，减少了**编码的复杂度**和**速度**。（因为 CPU 中的频率高于外部）

==Mov==

---

![[images/image20220330105110.png]]

- ALU：计算单元（加减乘除与或非）
- 通用寄存器：临时存储中间数据，经常和内存进行互相读写
- 指令队列缓冲器：缓冲外部输入的指令数据
- 执行部分控制电路：读取指令队列缓冲器的指令并转译
- 总线：并行传送数据，一根总线传送 1 bit 数据
- IP：始终指向下一条指令的地址

---

CPU 和内存的数据读写角色可以交换。

CPU 和内存之间有两种角色的总线：

- 地址总线（传递指令的地址）
- 数据总线（传递数据）

这两种总线实际上可以用一条实现。

---

在电脑刚刚开机的时候，IP 指向 0 地址，执行第一条指令

以后每次将 IP 中送入地址加法器后==立刻==将 IP 移动到下一指令

---

编程中四种存储空间

1. 指令存储（只读一般）
2. 全局变量
3. 栈区：存储函数中变量
4. 堆区：存储 ``new, new[]`` 开辟的变量

---

![[images/image20220406100930.png]]

- AD15 - AD0: Address & Data
- RD: CPU 等待内存写入数据
- DT/R: Data Transmit and Receive
- READY: 反馈信号，内存（外设）给 CPU 的信号
- ALE: 输入地址有效
- DEN: 输入数据有效

后面两种是为了应对地址总线和数据总线复用而设计的区分信号。

---

CPU 读取数据的过程（MOV 指令）

1. DT/R = 0，表示开始进行数据交换。
2. ALE = 1，表示地址总线有效，CPU 向总线发送地址数据。
3. 发送完毕以后，RD = 0，表示 CPU 准备好读数据，等待速度较慢的外设。
4. 外设发送反馈信号 READY = 1，CPU 发送 DEN = 0 信号，读入内存发送到总线的数据。

在 CPU 内部的读写过程，也有相似的时序。

---

内存和外存

- 内存
    - RAM: random access memory 挥发性，随机存取
    - NOR Flash 非挥发性，随机存取，写之前必须全部恢复为 1 状态
- 外存
    - NAND Flash 块读写
    - ROM 块读写
    - U 盘
    - SSD

---

扩展存储芯片

---

## 芯片和系统

---

设备实际上是由多个芯片构成。

在 CPU 实际向设备发送的地址中，一部分是给芯片编码使用（片选），另一部分才是芯片中的地址（片内地址）。

片选地址根据数字逻辑运算，可以决定是否使能相应的芯片。

设备都是并联在总线上的，但是 CPU 发送的地址数据不同，片选译码器得到的结果不同，有且仅有一个设备被激活。

具体的编码可以在 Windows OS Device Manager IO Range 中查看。

---

对于外设来说，存在低速外设和高速外设的差别，例如内存和 USB。

为了效率，不能将这两种外设共享总线，而应该使用一个芯片处理低速外设的数据，再接入高速总线，这就是**南桥**。

不同的外设形成不同的层次，构成计算机组成的**树状体系**。

---

## 单片机概论

---

单片机的最小系统

给定单片机，需要的工作条件

- 电源
- 时钟系统
- RESET 系统

---

| 指令系统   | 制造商                       |
| ---------- | ---------------------------- |
| x86(AMD64) | Intel, AMD                   |
| ARM        | ...（买 ARM 核，自造单片机） |
| RISC-V     | OpenSource                             |

这些指令系统的 CPU 完全不可通用，因为指令的二进制编码不同。

底层的不通用也导致了软件的不通用。

---

操作系统不兼容：不同操作系统之间的软件也是不可移植的，即使 CPU 相同。

```mermaid
flowchart TD
hardware --> OS --> software1 & software2 & softwares["..."]
hardware --> OS' --> software1' & software2' & softwares'["..."]
```

不同的软件根据操作系统开发。

---

但是通过虚拟解释器，可以让不同的编程语言（java, Python）在不同的 OS 上运行。

```mermaid
flowchart TD
hardware --> OS1 --> env1["virtual environment for OS1"] --> Python3
hardware --> OS2 --> env2["virtual environment for OS2"] --> Python3
```

---

如何点亮 LED 灯？

