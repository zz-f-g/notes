# Linux_Learning

——《鸟哥的 Linux 私房菜》

## 0. 计算机概论

### CPU（CENTER PROCESSING UNIT）分类

根据 CPU 内含**微指令**的设计理念分为两种 CPU 架构：**精简指令集**和**复杂指令集**

#### 精简指令集（REDUCED INSTRUCTION SET COMPUTER, RISC）

微指令集较为精简，执行时间短。

例如：

- 甲骨文（Oracle）公司的 SPARC 系列
  - 常用于学术领域的大型工作站中，包括银行金融体系的主要服务器
- IBM 公司的 Powerful Architecture （包括 PowerfulPC）系列
  - ，例如索尼（Sony）公司出产的Play Station 3（PS3）
- 安谋公司（ARM Holdings）的 ARM CPU 系列
  - 各厂牌手机、PDA、导航系统、网络设备（交换器、路由器等）等（目前世界上使用范围最广）

#### 复杂指令集（COMPLEX INSTRUCTION SET COMPUTER, CISC）

每个小指令可以执行一些较低阶的硬件操作，指令数目多而且复杂， 每条指令的长度并不相同。花费时间长但处理工作丰富。

例如：AMD、Intel、VIA 等的 x86 [^1]架构的 CPU

[^1]: 那为何称为 x86 架构呢？这是因为最早的那颗Intel发展出来的CPU代号称为8086，后来依此架构又开发出80286，80386……，因此这种架构的CPU就被称为x86架构了。

64 位[^2]的个人电脑 CPU 又被统称为 x86_64 的架构。

[^2]: 所谓的位指的是CPU一次数据读取的最大量！64位CPU代表CPU一次可以读写64bits这么多的数据，32位CPU则是CPU一次只能读取32位的意思。

### Intel 芯片架构

#### 连接器：北桥，南桥

- 北桥：连接 CPU、内存、显卡接口
- 南桥：硬盘、USB、网卡

#### CPU

评价性能优劣：

1. 微指令集
2. 频率：每秒钟 CPU 的工作次数

前端总线速度（FRONT SIDE BUS, FSB）

CPU 从内存中取得的最快带宽 = CPU 内置的内存控制芯片对内存的工作频率 × “位数”

例如：1600MHz × 64bit = 1600MHz × 8 Bytes = 12.8GByte/s

字组大小（word size）：CPU 每次能够处理的数据量。CPU 的位数和上面说到的 FSB 正是根据这个确定的。

#### 内存

不论是数据还是程序，只有读入内存才能被 CPU 利用。

##### 动态随机存取内存（DYNAMIC RANDOM ACCESS MEMORY, DRAM）

断电后数据消失，也叫做**挥发性内存**

SDRAM 和 DDR RAM 的区别：

DDR RAM 可以“双倍传输数据”：double data rate

在一次工作周期中进行两次数据传输，频率速度 = 内部频率 * 2 （或者更高）

##### 静态随机存取内存（STATIC RANDOM ACCESS MEMORY, SRAM）

整合到 CPU 内成为第二层高速缓存（L2 cache），频率必须与 CPU 相同。

~~~mermaid
graph LR;
a[CPU]
b[CPU L2 cache]
c[主内存]
d[外部存储设备]
a --> b;
b --> a;
c --> b;
d --IO--> c;
~~~

##### 只读存储器（READ ONLY MEMORY, ROM）

非挥发性：没有通电时也能将数据存储下来。

例如 BIOS（固件，firmware）， 控制开机各项硬件参数的取得

> 固件就是绑在硬件上的控制软件。

#### 显卡（VIDEO GRAPHICS ARRAY）

显卡和主板的接口：

- PCI
- AGP
- PCIe（PCIe 1.0 中，每一条管线提供 250MBytes/s 的带宽)

显卡和屏幕的接口：

- D-Sub
- DVI
- HDMI（可以传输声音，广泛用于电视中）
- Display port

#### 硬盘与储存设备

磁盘和主板的接口

- SATA
- SAS（速度快，支持热插拔）
- USB

固态硬盘（SSD）

- 省电，快速（不需要马达传动）
- 寿命短，消耗品

### 数据表示

文字编码：unicode, UTF-8

### 软件

#### 操作系统（OS）

将所有的硬件参数封装，提供发展软件的参考接口，避免编写重复的控制码。

- 操作系统核心（kernel）：管理所有的硬件
- 系统调用（system call）：提供接口

~~~mermaid
graph LR;
a[hardware]
b[os kernel]
c[os system call]
d[app]
d --公认的系统调用参数--> c --核心可以理解的任务参数--> b --机器语言--> a
~~~

## 1. Linux 是什么以及如何学习

## 2. 主机规划和磁盘分区

## 3. 安装 CentOS7.x

## 4. 首次登陆与线上求助

### 文字指令
