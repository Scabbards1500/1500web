abstract: 之前数据结构差不多学完了， 然后这里是常考算法合集，包含回溯，二分，贪心，dp
time: 2026/08/19
## 双指针 two pointers

| 类型 | 典型写法 | 思路 |
| --- | --- | --- |
| **扫描型双指针** | `for right` + `left` | 一个探索，一个维护 |
| **对撞型双指针** | `while left < right` | 左右向中间靠 |
| **快慢指针** | `slow` + `fast` | 两个指针速度不同 |

## 回溯 Backtrack

可以把 **Backtracking 理解成 DFS 的一个特殊版本**：

普通 DFS：

> 我沿着一条路走到底。

Backtracking：

> 我沿着一条路走到底，如果不满足条件，就撤销选择，换另一条路。

#### 1. Subsets（子集）

子集题因为是单方向bootstrap的，所以传索引，但是不记录数字是否使用过

比如：

- LC78 Subsets
- LC90 Subsets II

核心：

```
choose -> explore -> unchoose
```

---

#### 2. Permutations（排列）

排列题就需要记录数字是否使用过

比如：

- LC46 Permutations
- LC47 Permutations II

特点：

每次选择一个没用过的数字：

```
path = [1,2]
choose 3
path = [1,2,3]
backtrack
remove 3
```

---

#### 3. Combination（组合）

比如：

- LC77 Combinations
- LC39 Combination Sum

特点：

有选择范围：

```
start index
```

避免重复。

---

#### 4. Partition（分割）

比如：

- LC131 Palindrome Partitioning

比如：

```
"aab"

[
 ["a","a","b"],
 ["aa","b"]
]
```

本质：

枚举切割位置。

---

#### 5. Search / Constraint Satisfaction（搜索问题）

比如：

- LC79 Word Search
- LC51 N-Queens

这种更像：

> DFS + Backtracking

## 二分查找 Binary search

#### 普通二分

找到了：

```
return mid
```

因为 mid 就是答案。

---

#### 边界二分

不断压缩范围：

```
return left
```

因为最后：

```
left = right = 边界
```

## 贪心算法 Greedy Algorithm

每一步都选择当前最优的方案，希望最后得到全局最优。

比如：

你有硬币：[1, 5, 10, 20]

目标：36

如果硬币可以无限使用，我们可以：20 → 16   10 → 6  5 → 1  1 → 0

于是：20 + 10 + 5 + 1

用了 4 枚。

这就是贪心：**每次拿当前能拿的最大硬币。**

但注意：**贪心不是“看到最大就选最大”这么简单。** 因为很多问题里，局部最优会导致全局最差。

**DP vs 贪心**

DP： 我现在做这个选择，会不会影响未来？

greedy: 我现在直接选一个我认为最好的，之后不反悔。

## **动态规划 Dynamic Programming（DP）**

把一个大问题拆成很多相互关联的小问题，把小问题的答案存下来，避免重复计算。

为什么叫“动态规划”？

比如你要算：F(5)=F(4)+F(3)

而： F(4)=F(3)+F(2)

你会发现 **F(3) 被算了两次**。

如果继续往下展开，会出现大量重复计算。

DP 就是：

第一次算 F(3) → 把结果记下来

第二次需要 F(3) → 直接拿之前的结果

所以 DP 的核心其实就是**重复子问题 + 保存已经算过的结果**

以后看到 DP 题，主要问自己：

**① State：状态是什么？**

> `dp[i]` 表示什么？

比如：

> `dp[i]` = 到达第 i 阶的方法数

**② Transition：状态怎么转移？**

> 当前状态怎么从之前的状态得到？

比如：

dp[i]=dp[i−1]+dp[i−2]

**③ Initialization：初始状态是什么？**

比如：dp[1] = 1 dp[2] = 2

**④ Answer：最后答案在哪里？**

比如： return dp[n]

- **Top-down DP**：递归 + memoization（记忆化）
- **Bottom-up DP**：从小问题开始，一步步算到大问题

模板

```python
class Solution:
    def xxx(self, n: int) -> int:

        # 1. 定义 dp
        # dp[i] 表示：__________

        dp = [0] * (n + 1)

        # 2. 初始化
        dp[0] = ?
        dp[1] = ?

        # 3. 状态转移
        for i in range(2, n + 1):
            dp[i] = ?

        # 4. 返回答案
        return dp[n]
```

### 多维动态规划

> **当一个问题的“状态”不只由一个变量决定，而是需要同时记录多个维度的信息时，就把这些维度一起放进 DP 状态。**

最核心的思想其实和普通 DP 一模一样：

**定义状态 → 找状态转移 → 初始化 → 按顺序计算 → 取答案。**

遇到 DP，可以先问：**“为了决定下一步，我需要记住过去的哪些信息？”**

如果只需要：当前位置 → 一维 DP。dp[i]

如果需要：当前位置+ 一个额外限制 → 二维 DP。dp[i][j]

如果需要：当前位置+ 一个额外限制 → 三维 DP。dp[i][j][k]
