# C Primer Plus

[toc]

## 基本数据类型

### int

~~~C
#include <stdio.h>

int main(void){
    int var; //为变量创建内存空间
    int var = 0; //初始化变量
    
    printf("var is %d.", var); //打印整型变量
    printf("dec = %d, oct = %#o, hex = %#x.", var, var, var); //加 # 表示输出加前缀 "0", "0x" 或 "0X".
    
    return 0;
}
~~~

使用 printf() 函数时要使得字符串内部 % 的数量和外部变量的数量一致，**这个错误编译器无法检查**。

int 类型占据 32 个 bit，考虑到正负号，能表示区间为 $[-2^{31},2^{31}-1]$ 的数据；unsigned int 类型没有正负号，能表示区间为 $[0,2^{31}-1]$ 的数据。

当整数溢出时，均从区间最左边的数开始。

~~~C
#include <stdio.h>

int main(void){
    int i = 2147483647; //2^31-1
    unsigned int j = 4294967295; //2^32-1
    
    printf("%d %d %d\n", i, i+1, i+2);
    printf("%u %u %u\n", j, j+1, j+2);
    
    return 0;
}
~~~

### char

用单引号括起来的字符表示**字符常量**，用双引号括起来的表示**字符串**。

~~~C
char c = 't'; \\correct
char c = "t"; \\wrong
~~~

### float&double

一个 float 类型占据 32 bit，其中 24 bit 用于表示不少于 6 位的有效数字，8 bit 用来表示指数（指数的范围相当于 char 类型的范围）。double 类型则可以至少表示 13 为有效数字。

~~~C
float flt = 3.4e38;
float flt_min = 0.1234e-43;
printf("%f\n", flt); //自然计数法
printf("%e\n", flt); //科学计数法
printf("%f\n", flt + 1); //overflow
printf("%f\n", flt_min / 10); //umderflow
~~~

### 其他类型

数组、指针、结构、联合

### 转义序列

| 转义字符 | 意义                           |
| -------- | ------------------------------ |
| \a       | 警报声                         |
| \r       | 回车（回到这一行的起始位置）   |
| \b       | 退格（不删除，但再输入时覆盖） |

~~~C
#include <stdio.h>

int main(void){
    float salary;
    
    printf("\aEnter your desired monthly salary: ");
    printf("$_______\b\b\b\b\b\b\b");
    scanf("%f", &salary);
    printf("\n\t$%.2f a month is $%.2f a year.", salary, salary * 12.0);
    printf("\rGee!");
    
    return 0;
}
~~~

``printf()`` 函数如何输出在屏幕上？

~~~mermaid
graph LR

a((printf))
b((suffer缓冲区))
c((screen))
standard[刷新缓冲区的 C 语言标准]
full[缓冲区满]
enter[需要换行]
input[需要输入]

a --> b
b --刷新缓冲区--> c
standard --> full
standard --> enter
standard --> input
~~~

## 字符串与格式化输入输出

### 字符串

示例：

~~~C
#include <stdio.h>
#include <string.h>
#define DENSITY 62.4

int main(void){
	float weight, volume;
	int size, letters;
	char name[40];
	
	printf("Hi! What's your first name?\n", name);
	scanf("%s", name);
	
	printf("%s, What's your weight in pounds?\n", name);
	scanf("%f", &weight);
    
	size = sizeof name;
	letters = strlen(name);
	volume = weight / DENSITY;
    
	printf("Well, %s, your volume is %2.2f cubic feets.\n", 
	name, volume);
	printf("Also, your first name has %d letters.", letters);
	printf("And we have %d bytes to store it.", size);
		
	return 0;
}
~~~

字符串本质上是一个数组，数组的每个元素的类型都是 ``char``.

数组在逻辑上连续，在物理上也连续。每个存储单元有 8 个字节，存储一个字符。

字符串数组末尾字符是 ``\0`` 空字符（null character）。因此数组的容量至少应该比字符数量多 1.

``scanf()`` **重要的问题**：字符串在调用 ``scanf()`` 函数用于输入时，函数在读到第一个**空格、制表符或换行符**时停止读取。因此如果输入一个句子，事实上只能读入一个单词。

``strlen()`` 返回从数组第一个元素到字符串末尾空字符 ``\0`` 的字符数量（不包括空字符）。

``sizeof()`` 则返回整个数组的大小（字节），包括字符串没有用到的后面的部分（垃圾值）。

### 常量

``#define`` 通过预处理器来定义常量，可以避免用变量定义时可能不小心改变的问题。这叫做**编译时替换**。

~~~C
#define NAME value
~~~

- 没有等于号
- 没有分号
- 常量名称大写

### 输入输出语句

``printf()`` 格式字符串（转换说明），待打印项 1，待打印项 2……

关于转换说明：

- 转换类型的不同字母表示以不同的对应类型显示（解码）数据
- 修饰符
  - 标记：-, +, <space>, #, 0
  - 数字：最小字段宽度
  - 小数点+数字：（小数点右边）的最大位数（浮点类型）、打印数字的最大位数（整型数字）
  - 其他字母 ``%zd`` 用来支持可移植的 ``size_t`` 类型（``sizeof()`` 的返回值的类型）

转换说明要和实际的类型一致。否则：

~~~C
#include <stdio.h>

int main(void){
	float n1 = 3.0;
    double n2 = 3.0;
    long n3 = 2000000000;
    long n4 = 1234567890;
    
    printf("%.1e %.1e %.1e %.1e\n", n1, n2, n3, n4);
    printf("%ld %ld\n", n3, n4);
    printf("%ld %ld %ld %ld\n", n1, n2, n3, n4);
	
	return 0;
}
~~~

``printf()`` 函数的参数传递过程：

~~~mermaid
graph TD

a[储存变量的内存区域]
b[栈 stack]
c[输出]

a--变量中的数据按照变量的类型依次相邻被读取到-->b--按照转换说明依次被解码显示-->c
~~~

``printf()`` 函数返回打印字符的个数（正常时），或负数（异常时）。

如果 ``printf()`` 语句太长，可以换行，但是格式字符串之间不能换行，因为换行只能用 ``\n`` 表示。

~~~C
#include <stdio.h>

int main(void){
	float n1 = 3.0;
    double n2 = 3.0;
    long n3 = 2000000000;
    long n4 = 1234567890;
    
    printf("Here's a way to print ");
    printf("a long string.\n");
    printf("Here's another way to print a \
long string.\n");
    printf("Here's the newest way to print a "
          "long string.\n");
	
	return 0;
}
~~~

``scanf()`` 读取输入的过程（以 ``%d`` 为例）

~~~mermaid
graph TD
z[开始读取]
a[跳过所有的空白字符]
b[读取正负号]
c[读取数字字符]
d[读到空白字符或者非数字字符]
e[停止读取并放回读到的空白字符作为下一项读取的起始值]
f[一开始就读到非数字字符]
g[异常并返回0]
h[读取下一项]
i[停止读取并返回读取的项数]
j[将读到的值放入指定的地址中]
if1[有无空白字符: 空格换行制表符等等]
if2[有无正负号]
z-->if1--有-->a
if1--无-->if2
a-->if2--有-->b
if2--无-->c
b-->c-->d-->e-->j--有下一项-->h-->z
c-->f-->g
j--没有下一项-->i
~~~

格式字符串中的普通字符

- 有意义的字符，如逗号。在输入时必须也在对应的位置输入相应的字符，比如逗号。当然可以在任意的位置加入空白。
- 空白，如空格、换行、制表符。这意味着跳过下一个输入项之前的所有空白。
- 注意：``%c`` 是唯一能够读入空白的转换类型，它将只读入并储存输入的第一个字符。但是如果输入以下语句（作为上面第二种情况的应用），将跳过所有的空白。

```C
scanf(" %c", chr);
```

格式字符串中的星号修饰符：通过变量来控制输入输出的格式

1. ``printf()``
   1. 整形的字段宽度 ``%*d``
   2. 浮点数的字段宽度和小数位数 ``%*.*d``
2. ``scanf()``
   - 跳过相应的输出项（就是接受输入但不把输入的值传给变量）

```C
printf("Please put in three values.\n");
scanf("%*d %*d %d", &n);
printf("The third value you put in is %d.\n",n);
```

## 运算符、表达式和语句

### 递增递减运算符

作为单目运算，其优先度高于乘法和除法。

~~~C
//POST
b = a++;
/*is equivalent to*/
b = a;
a = a + 1;

//PRE
b = ++a;
/*is equivalent to*/
a = a + 1;
b = a;
~~~

规范：

- 如果变量出现在函数参数中，不要使用递增运算符；
- 如果变量多次出现在一个表达式中，不要使用递增运算符。

逗号运算符

``a, b`` 先进行语句 ``a``，再进行语句 ``b``。和 ``;`` 一样是一个节点。语句的值是逗号以后的值 ``b``。

### 数据类型转换

级别的排序：

``long double``>``double``>``float``>``unsigned long``>``long long``>``unsigned long``>``long``>``unsigned int``>``int``>``unsigned short``>``short``>``char``

- 类型升级：涉及两种类型的计算、作为函数参数传递
- 类型降级：赋值表达式

## 控制语句

### 两种循环语句的等价结构

```C
for(a; b; c){
    d;
}

a;
while(b){
    d;
    c;
}
```

但是在存在 ``continue`` 语句的情况下会有所不同。

~~~C
int cnt = 0;
while(cnt < 10){
    ch = getchar();
    if(ch == '\n'){
        continue;
    }
    putchar(ch);
    cnt ++;
}
~~~

对于 ``while`` 循环，进入 ``continue`` 后直接回到 ``cnt<10`` 处进行判定并进行下一轮的循环。

~~~C
int cnt = 0;
for(cnt = 0; cnt < 10; cnt ++){
    ch = getchar();
    if(ch == '\n'){
        continue;
    }
    putchar(ch);
}
~~~

对于 ``for`` 循环，进入 ``continue`` 后，**先递增（进入 ``cnt ++``）**，直接回到 ``cnt<10`` 处进行判定并进行下一轮的循环。

### 逻辑运算符

逻辑运算符的运算顺序被 C 语言强制规定为从左到右，逻辑运算符是一个序列点。

~~~C
while(x++ < 10 && x + y < 20)
~~~

上述语句保证了右边的 ``x`` 已经被递增。

在进行运算时，一旦发现有使得最终结果为 0 的情况，停止运算，返回 0.

~~~C
x != 0 && (20 / x) < 5
~~~

只有当 $x \neq 0$ 时，才会对右边的式子求值。

### 条件运算符

``max = (a > b) ? a : b``

### 多重选择语句

~~~C
switch(expression){
    case label1: statement1;
    case label2: statement2;
        break;
    default: statement3;
}
~~~

根据 ``expression`` 的值跳转到指定的标签，然后执行标签之后的语句；

如果没有对应的标签，跳转到 ``default`` 语句（如果有的话），然后执行之后的语句，如果没有，直接跳出语句块；

直到遇到 ``break`` 语句（如果有的话）以后跳出语句块。

### 一些值得注意的问题

- 一定要注意数组索引的问题，不能超出索引。同时 ``for`` 循环中的迭代也应该与之对应。
- 在进行一些数学计算时，一定要考虑必要的数据类型转换。不转换可能会发生截断（例子见下）。
- ``scanf()`` 语句在将输入传入数组时，有两种不同的情况。注意 ``&`` 加还是不加。
  - ``scanf("%s", arr);``
  - ``scanf("%d",&arr[0])``

~~~C
#include <stdio.h>

long square(int root){
    return ((long) root)*root; //x
}
~~~

## 字符函数

``getchar()`` 读取一个字符并返回。

``putchar()`` 打印一个字符。

``ctype.h`` 头文件中包含了一些判定字符类型的函数，传入字符，根据类型返回 ``0`` 或非零值或者进行映射。

- ``isalnum()``
- ``isalpha()``
- ``isblank()`` 是否为标准空白字符如 ``\n`` `` `` ``\t``
- ``iscntrl()`` 是否为控制字符
- ``isdigit()`` 是否为数字
- ``isgraph()`` 是否为除了空格以外的任意可打印字符
- ``islower()``
- ``isprint()`` 是否为可打印字符
- ``ispunct()`` 是否为除了数字和字母以外的可打印字符
- ``isspace()`` 是否为空白字符
- ``isxdigit()`` 是否为十六进制数字字符
- ``tolower()`` 如果是大写，转为小写，否则不变
- ``toupper()``

未完待续

## 函数

如果函数返回值声明的类型和实际返回的类型不一致，那么实际返回的
