abstract: 链表的实现以及常见题型
time: 2026/07/14
## Linked List

Linked List 是由多个 **Node（节点）** 组成的线性结构，每个 Node 存储数据 + 指向下一个 Node 的引用。

每个节点

```
Node:
    value
    next
```

Array：数据挨在一起，用 index 找
Linked List：数据散落各处，用指针串起来

```
Array:
[10][20][30][40]

Linked List:
+------+------+
| 10   | next | ---->
+------+------+
              +------+------+
              | 20   | next | ---->
              +------+------+
                            +------+------+
                            | 30   | next | ---> None
                            +------+------+
```

### Node

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
```

创建节点：

```python
node1 = Node(10)
node2 = Node(20)
node1.next = node2    # 10 -> 20 -> None
```

### Head

链表没有 index， 所以需要一个入口 Head 来表示链表第一个节点

例如：

```python
head = node1
```

```
head
 |
 v
10 -> 20 -> 30 -> None
```

只要知道 head，就能访问整个链表。

## 做题

感觉链表题有点偏好指针哇

找中点: 快慢指针

合并升序链表：双指针

### 环形链表

判断链表是否有环：快慢指针→ 相遇

推导

假设链表是：

```
head
 ↓
head
 ↓
A → B → C → D → E
                ↑               ↓
                └──── ─┘
```

- `C` 是**环入口**
- 假设第一次相遇在 `E`

```
head
  |
  | a
  ↓
入口
  ↓ b
  ● ← 第一次相遇
  |
  | c
  ↓
  └────────→ 入口
```

`a = head → 环入口`的距离

也就是：A → B → C， 从 `A` 走到 `C`， 两步，a=2

`b`：环入口 → 第一次相遇点的距离

`c`：第一次相遇点 → 环入口的距离

环的长度：

L=b+c

先跳过推导，总之， **从 head 走 `a` 步到入口；从第一次相遇点走 `c` 步也到入口。**

而由于 `a` 和 `c` 在环长意义下是对应的，所以两个指针分别从 head 和相遇点出发，以同样速度走，就会同时到达入口。
