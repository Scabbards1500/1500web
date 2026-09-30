abstract: 堆和堆的实现以及常见题型
time: 2026/08/17
## Heap

堆 = 一个能快速拿到“当前最小值/最大值”的特殊数据结构。

核心价值是**动态维护一个集合中的最大/最小元素。**

最常见的是两种：

- **Min Heap（小根堆）**：堆顶永远是最小值
- **Max Heap（大根堆）**：堆顶永远是最大值

比如：

```
       1
      / \
     3   5
    / \
   7   8
```

这是一个小根堆，因为：

1 < 3, 5

3 < 7, 8

但是注意：**堆不是完全排序的。**

比如 `[1, 3, 5, 7, 8]` 并不意味着整个数组都排好序了。

它只保证：**最小值在最上面。**

所以如果你只想反复拿“最小的那个”，堆特别方便。

### 储存

Heap 通常用一个**数组**来存。

例如：

```
        2
       / \
      3   5
     / \
    7   8
```

存成：

heap = [2, 3, 5, 7, 8]

### **操作**

#### push

把一个元素放进 Heap。向上调整（sift up）

假设：heap = [2, 3, 5, 7]

现在加入：1

最简单的做法：直接把 `1` 放到数组最后。

变成：[2, 3, 5, 7, 1]

对应：

```
       2
      / \
     3   5
    / \
   7   1
```

这显然不是 Heap，因为：

1 < 3

1 < 2

所以我们需要把 `1` 往上移动。

**Sift Up**

新元素现在 index = 4

它的 parent：**parent = (4 - 1) // 2 = 1**

所以比较：heap[4] = 1 heap[1] = 3

因为：1 < 3

交换：[2, 1, 5, 7, 3]

现在 `1` 在 index `1`。

继续找 parent：parent = (1 - 1) // 2 = 0

比较：heap[0] = 2 heap[1] = 1       1 < 2

交换：[1, 2, 5, 7, 3]

完成。

**实现**

```python
def push(self, x):
    self.heap.append(x)

    i = len(self.heap) - 1

    while i > 0:
        parent = (i - 1) // 2

        if self.heap[parent] <= self.heap[i]:
            break

        self.heap[parent], self.heap[i] = \
            self.heap[i], self.heap[parent]

        i = parent
```

把新元素放最后
↓
看看爸爸
↓
如果我比爸爸小
↓
和爸爸交换
↓
继续往上
↓
直到爸爸比我小

#### pop

把 Heap 顶部，也就是最小值拿走。向下调整（sift down）

假设：

```
        1
       / \
      3   5
     / \
    7   8
```

数组：[1, 3, 5, 7, 8]

现在：pop()

我们要删除 `1`。

但这里不能直接 del heap[0]

因为这样后面的结构全部乱掉。

正确做法是：

1. 把最后一个元素拿到顶部：

```
        8
       / \
      3   5
     /
    7
```

数组：[8, 3, 5, 7]

然后把 8 往下沉。

**Sift Down**

现在 `8` 在 index `0`。

它有两个孩子：

```
left = 2 * i + 1     left = 1 → 3
right = 2 * i + 2         right = 2 → 5
```

我们应该**找两个孩子里面更小的那个交换。**

因为我们是 Min Heap。

所以：3 < 5

跟 `3` 换：[3, 8, 5, 7]

```
        3
       / \
      8   5
     /
    7
```

`8` 现在在 index `1`。

继续：

left = 3 → 7

right = 4 → 不存在

所以和 `7` 比，7 < 8 → 交换

[3, 7, 5, 8]

完成。

实现

```python
def pop(self):
    ans = self.heap[0]

    # 最后一个元素放到顶部
    self.heap[0] = self.heap[-1]
    self.heap.pop()

    i = 0

    while True:
        left = 2 * i + 1
        right = 2 * i + 2

        smallest = i

        if left < len(self.heap) and \
           self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and \
           self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest == i:
            break

        self.heap[i], self.heap[smallest] = \
            self.heap[smallest], self.heap[i]

        i = smallest

    return ans
```

## 刷题

你刷 Heap 题的时候，可以先问自己：

> **我要维护的是 Top K 最大，还是 Top K 最小？**

然后：

| 目标 | Heap |
| --- | --- |
| Top K 最大 | **Min Heap** |
| Top K 最小 | **Max Heap** |
| 不断取最小 | Min Heap |
| 不断取最大 | Max Heap |

这个规律非常重要。

例如：

**第 K 大**不是用 Max Heap！

而是：**Min Heap，大小 K**

因为我们需要让第 K 大卡在 Heap 的最顶端。
