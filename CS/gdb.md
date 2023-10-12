# GNU Debugger

如何使用 gdb cli 工具调试 C/C++？

## compile option

```shell
gcc -g *.c
```

开始调试

```shell
gdb ./a.out
```

## breakpoint

```
(gdb) break main
(gdb) b main
```

在 break(b) 后面加函数名或行号。

临时断点：只中断一次。

```
(gdb) tb main.c:12
```

除了 breakpoint，还有 watchpoint.

监视变量，运行到变量变化为止。

```
(gdb) watch var
(gdb) c
```

删除断点

```
(gdb) delete $(index of argc)
(gdb) delete # all breakpoints
```

## run

```
(gdb) run
(gdb) r
```

先添加断点再开始运行。否则将直接正常执行。

```
(gdb) continue
(gdb) c
```

```
(gdb) next
(gdb) n
```

此时显示的代码是将要执行的代码。

```
(gdb) step
(gdb) s
```

```
(gdb) finish
(gdb) fi
```

结束在一个函数中的调试

逆向调试

```
(gdb) target record-full
```

开启记录

```
(gdb) reverse-next
(gdb) rn
```

逆向单步调试，等等。

> [!help] 经常会有问题

## check

```
(gdb) list
(gdb) l
```

默认打印当前位置的前后各 5 行。再执行一次将接着打印后面的部分。由于在 gdb 中，回车表示重复执行上一次的指令，因此可以通过回车打印很多连续的代码。例如：

```(gdb) l 1
1       #include <stdio.h>
2       #include <stdlib.h>
3       #include <string.h>
4       #include <errno.h>
5       #define F_NAME_LIM 512
6
7       int main(int argc, char **argv)
8       {
9               if (2 != argc)
10              {
(gdb)
11                      fprintf(stderr, "Usage: %s infile\n", argv[0]);
12                      exit(1);
13              }
14              FILE *ifp;
15              if (NULL == (ifp = fopen(argv[1], "rb")))
16              {
17                      fprintf(stderr, "Unable to open %s: %s\n", argv[1], strerror(errno));
18                      exit(1);
19              }
20              char out_file_name[F_NAME_LIM + 1];
(gdb)
```

```
(gdb) whatis argc
```

查看变量类型

```
(gdb) print argc
(gdb) p (char)c
```

打印变量或表达式的值。由于可以打印表达式，解析 C/C++ 语法，因此非常有用。

甚至可以用来执行函数。

```
(gdb) p myfunc(args)
```

如果程序执行到一半发现有错，但是不想退出 gdb 重新编译，可以使用这种方法临时改变变量的值。

一直查看某变量，添加变量监视窗口。

```
(gdb) display argc
(gdb) undisplay $(index of argc)
```

可以通过 ``info`` 查看监视变量编号

```
(gdb) info display
```

```
(gdb) up
(gdb) down
```

上移到调用此函数的位置，下移回来。

```
(gdb) backtrace
(gdb) bt
```

显示调用栈。

## info

查看断点 breakpoints & watchpoints

```
(gdb) i b
```

## set

中途改变数据

```
(gdb) set var x=1
(gdb) set var *p = 2
```
