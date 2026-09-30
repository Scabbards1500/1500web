abstract: stack 和 stack 的实现以及常见题型
time: 2026/07/13
## Array 基础

Array 是一块连续的内存空间，用来存储一组相同类 型的数据。

### **创建**

```python
nums = [1, 2, 3, 4, 5]
empty = []
zeros = [0] * 5
print(zeros)
```

### 访问 Access

根据 index 获取元素

```python
nums = [10,20,30,40]
print(nums[2])
```

### 修改元素

```python
nums[2] = 100
print(nums)
```

### 获取长度 len()

```python
nums = [1,2,3,4]
print(len(nums))
```

### 添加元素 append()

添加一个元素到尾部

```python
nums = [1,2,3]
nums.append(4)
print(nums)
```

### 插入元素 insert()

index=2的位置插入。

```python
nums = [1,2,3,4]
nums.insert(2,100)
print(nums) #[1,2,100,3,4]
```

### 删除元素

**pop()**  删除最后一个

```python
nums=[1,2,3]
x=nums.pop()
print(x)
print(nums)
```

**pop(index)**删除指定位置

```python
nums=[10,20,30,40]
nums.pop(1)
print(nums) # [10,30,40]
```

**remove()**根据值删除 * 注意，只删除第一个匹配值

```python
nums=[10,20,30]
nums.remove(20)
print(nums) # [10,30]
```

### 查找 Search

**in** 查找是否存在

```python
nums=[1,2,3,4]
print(3 in nums) # True
```

**index()**找位置

```python
nums=[10,20,30]
print(nums.index(20))  #1
```

### 排序 sort()

原地排序

```python
nums=[3,1,5,2]
nums.sort()
print(nums) # [1,2,3,5]
```

降序

```python
nums.sort(reverse=True) #[5,3,2,1]
```

### 反转 reverse()

```python
nums=[1,2,3]
nums.reverse()
print(nums) #[3,2,1]
```

### 切片 slicing

左闭右开：

```
start <= index < end
```

复制数组：

```python
copy = nums[:]
```

反转：

```python
nums[::-1]
```

### 合并

```python
a=[1,2]
b=[3,4]
c=a+b
print(c) # [1,2,3,4]
```

## 常见考法

| 看到什么 | 想到什么 |
| --- | --- |
| 找是否存在 / 出现次数 | HashMap / Set |
| 有序数组找两个数 | 双指针 |
| 连续子数组最长/最短 | 滑动窗口 |
| 子数组和为 K | 前缀和 + HashMap |
| 区间重叠 | 排序 |
| 原地删除/移动 | 快慢指针 |
| 最大连续和 | Kadane |
| 下一个更大/更小 | 单调栈 |
| Top K | Heap |
| 不能用除法求乘积 | 左右前缀积 |

### HashMap 计数 / 查找型

核心套路：
**需要快速判断某个数/状态是否出现过，就用 HashMap 或 Set。**

“是否存在”
“出现次数”
“成对匹配”
“去重”
“频率最高”

- **需要保持顺序、可以重复** → 用 `list`
- **只关心"有没有"，或者需要去重** → 用 `set`

```python
seen = {}
for i, x in enumerate(nums):
    need = target - x
    if need in seen:
        return [seen[need], i]
    seen[x] = i
```

| 题型 | 代表题 | 套路 |
| --- | --- | --- |
| 两数之和 | Two Sum | 边遍历边查 `target - x` |
| 判断重复 | Contains Duplicate | `set` 去重 |
| 字母/元素频率 | Valid Anagram | `Counter` |
| 分组 | Group Anagrams | 排序后作为 key |
| 最长连续序列 | Longest Consecutive Sequence | set 判断 `x-1` 是否存在 |

### 双指针 Two Pointers

核心套路：
**数组有序，或者可以从两边夹逼，就用双指针。**

“有序数组”
“找两个数”
“删除/压缩数组”
“从两端比较”

高频题：

| 题型 | 代表题 | 套路 |
| --- | --- | --- |
| 两数和 II | Two Sum II | 左右指针夹逼 |
| 三数和 | 3Sum | 排序 + 固定一个数 + 双指针 |
| 盛最多水 | Container With Most Water | 移动短板 |
| 删除重复 | Remove Duplicates | 快慢指针 |
| 移动零 | Move Zeroes | 快慢指针 |
| 接雨水 | Trapping Rain Water | 左右最大值 / 双指针 |

```python
left, right = 0, len(nums) - 1
while left < right:
    cur = nums[left] + nums[right]
    if cur == target:
        return [left, right]
    elif cur < target:
        left += 1
    else:
        right -= 1
```

### 前缀和 Prefix Sum

核心套路：
**频繁求区间和，或者问某段 subarray 是否满足条件。**

“subarray sum”
“区间和”
“和为 k”
“连续子数组”

以后看到数组题，直接在脑子里问三件事：

1️⃣ 能不能“左边一个信息”？

prefix

2️⃣ 能不能“右边一个信息”？

suffix

3️⃣ 能不能 O(1) 合并？

answer[i] = left[i] * right[i]

不要一看到数组题就想：

我要不要开 prefix 数组？

而是先问自己：

我后面还会不会用到这个状态？

不会 → 用一个变量维护（如 left_sum、prefix、suffix）。
会，而且要按位置访问 → 开 prefix[] / suffix[]。
不会按位置访问，只关心"有没有出现过"或"出现几次" → 用 HashMap（像 LC560）。

| 题型 | 代表题 | 套路 |
| --- | --- | --- |
| 区间和 | Range Sum Query | prefix 数组 |
| 和为 K 的子数组 | Subarray Sum Equals K | prefix sum + hashmap |
| 连续数组 | Contiguous Array | 把 0 当 -1 |
| 乘积除自身 | Product of Array Except Self | 左右前缀积 |
| 最大子数组和 | Maximum Subarray | Kadane / 前缀思想 |

经典模板：和为 K 的子数组

```python
count = 0
prefix = 0
seen = {0: 1}
for x in nums:
    prefix += x
    count += seen.get(prefix - k, 0)
    seen[prefix] = seen.get(prefix, 0) + 1
return count
```

第一阶段（必须秒懂）
1480 Running Sum 🆗
303 Range Sum
724 Pivot Index
第二阶段（核心套路）
238 Product Except Self
560 Subarray Sum K
152 Max Product Subarray
第三阶段（面试级）
42 Trapping Rain Water
239 Sliding Window Maximum

### 排序 + 贪心

核心套路：
**先排序，让局部关系变简单。**

高频题：

| 题型 | 代表题 | 套路 |
| --- | --- | --- |
| 合并区间 | Merge Intervals | 按 start 排序 |
| 插入区间 | Insert Interval | 分三段处理 |
| 无重叠区间 | Non-overlapping Intervals | 按 end 排序贪心 |
| 会议室 | Meeting Rooms | 排序判断冲突 |
| 会议室 II | Meeting Rooms II | 排序 + min heap |
| 分发饼干 | Assign Cookies | 排序 + 贪心 |

合并区间模板：

```python
intervals.sort()
res = []

for start, end in intervals:
    if not res or start > res[-1][1]:
        res.append([start, end])
    else:
        res[-1][1] = max(res[-1][1], end)

return res
```

适用信号：

```
“区间”
“重叠”
“会议”
“最少数量”
“最多安排”
```

### 原地修改 / 快慢指针

核心套路：
**不能额外开数组，要求 in-place，就用快慢指针。**

高频题：

| 题型 | 代表题 | 套路 |
| --- | --- | --- |
| 删除重复元素 | Remove Duplicates from Sorted Array | slow 记录有效位置 |
| 删除指定元素 | Remove Element | slow 覆盖 |
| 移动零 | Move Zeroes | 非零元素往前放 |
| 颜色分类 | Sort Colors | 三指针 |
| 下一个排列 | Next Permutation | 从右找下降点 |

模板：

```python
slow = 0

for fast in range(len(nums)):
    if nums[fast] 满足保留条件:
        nums[slow] = nums[fast]
        slow += 1

return slow
```

适用信号：

```
“in-place”
“O(1) extra space”
“remove”
“move”
“modify nums directly”
```

### Top K / Heap

核心套路：
**找最大/最小 K 个，用 heap。**

高频题：

| 题型 | 代表题 |
| --- | --- |
| 第 K 大元素 | Kth Largest Element |
| 前 K 高频元素 | Top K Frequent Elements |
| 数据流中位数 | Find Median from Data Stream |
| 合并 K 个有序链表 | Merge K Sorted Lists |

Top K 高频模板：

```python
from collections import Counter
import heapq

count = Counter(nums)
return [x for x, freq in heapq.nlargest(k, count.items(), key=lambda p: p[1])]
```

适用信号：

```
“top k”
“kth largest”
“频率最高”
“数据流”
```

### 矩阵

矩阵一般混着别的概念考

```python
rows = len(matrix)
cols = len(matrix[0])
```
