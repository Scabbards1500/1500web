abstract: stack 和 stack 的实现以及常见题型
time: 2026/08/17
## Stack

Stack 就是一个 **后进先出（LIFO, Last In First Out）** 的数据结构。

你可以把它想象成一摞盘子：

```
        ↑  最先拿出来
      ┌───┐
      │ C │
      ├───┤
      │ B │
      ├───┤
      │ A │
      └───┘
        ↑
      最后放进去
```

如果依次：

```
push A
push B
push C
```

那么：

```
pop() → C
pop() → B
pop() → A
```

### 核心操作

```python
stack.append(x)   # push
stack.pop()       # pop
stack[-1]         # 看栈顶，但不删除
```

### 实现

Python 里一般直接用 `list` 实现 stack：

```python
stack = []

stack.append(1)
stack.append(2)
stack.append(3)

print(stack[-1])  # 3

x = stack.pop()
print(x)          # 3
```

所以你刷题的时候，看到：

```python
stack = []
```

基本就可以理解成：

> 我现在准备维护一个“后进先出”的东西。

## 做题

### 字符串处理相关

最经典的左右括号 lc20

需要考虑三种条件

左边，右边，左边没有只有右边

### 单调栈 monotonic stack

**单调栈是一种特殊的 Stack：我们在不断 `push / pop` 的过程中，始终让栈里的元素按照某种单调顺序排列。**

也就是说，普通 Stack 只要求：后进先出

而 Monotonic Stack 额外要求：栈内元素保持单调递增/减

应用场景

Next Greater/Smaller Element e.g lc739

Previous Greater / Smaller

如果题目出现：

“对于每个元素，找它左边/右边第一个比它大/小的元素”

脑子里直接蹦出来：Monotonic Stack
