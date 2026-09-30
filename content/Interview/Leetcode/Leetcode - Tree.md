abstract: 感觉数据结构和算法在本科阶段就是学了丢丢了学... 这就是人生罢~
time: 2026/07/19
## Tree

树其实就是**链表的升级版**：链表节点只有一个 `next`，二叉树节点有两个方向：`left` 和 `right`。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val # 当前节点的值
        self.left = left # 左子节点
        self.right = right # 右子节点
```

### 术语

```
        1
       / \
      2   3
     / \
    4   5
```

| 中文 | 英文 | 例子 |
| --- | --- | --- |
| 根节点 | **root (node)** | `1` is the **root**. |
| 节点 | **node** | Every circle is a **node**. |
| 左子节点 | **left child** | `2` is the **left child** of `1`. |
| 右子节点 | **right child** | `3` is the **right child** of `1`. |
| 子节点 | **child (node)** | `2` and `3` are the **children** of `1`. |
| 父节点 | **parent (node)** | `1` is the **parent** of `2`. |
| 兄弟节点 | **sibling** | `2` and `3` are **siblings**. |
| 叶子节点 | **leaf (node)** | `3`, `4`, and `5` are **leaf nodes**. |
| 子树 | **subtree** | The tree rooted at `2` is a **subtree**. |
| 左子树 | **left subtree** | The **left subtree** of `1`. |
| 右子树 | **right subtree** | The **right subtree** of `1`. |

深度（Depth）

节点距离**根节点**有多少条边。

高度（Height）

一个节点到**最远叶子节点**的距离。

度（Degree）

一个节点拥有多少个孩子。

### 创建

```python
root = TreeNode(1)

root.left = TreeNode(2)
root.right = TreeNode(3)

root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
```

### 访问

本质上就是沿着指针走

```python
print(root.val)              # 1
print(root.left.val)         # 2
print(root.right.val)        # 3
print(root.left.left.val)    # 4
print(root.left.right.val)   # 5
```

如果某个节点没有孩子，对应位置就是 `None`：

```python
print(root.right.left)   # None
print(root.right.right)  # None
```

### 遍历

![](https://i-blog.csdnimg.cn/direct/a5e0c4a0c0df4fdabf00a42d80fdcea2.png)

区别： 当前节点什么时候处理

#### 前序遍历

根 → 左 → 右

“我刚走到这个节点，就能决定一些事情。”

常见场景

- 复制一棵树（Clone Tree）
- 序列化二叉树
- 打印路径
- 记录从根到当前节点的信息
- 构造字符串表达式

```python
def traverse(root):
    if root is None:
        return

    print(root.val)
    traverse(root.left)
    traverse(root.right)
```

#### 中序遍历

左 → 根 → 右

BST 专属技能最强

排序、第 k 小、验证 BST、双向链表转换。

```python
def inorder(root):
    if root is None:
        return

    inorder(root.left)
    print(root.val)
    inorder(root.right)
```

#### 后序遍历

左 → 右 → 根

先知道孩子，再决定自己

“当前节点的答案依赖左右子树。”

常见场景

- 树高 / 最大深度（LC104）
- 平衡二叉树（LC110）
- 直径（LC543）
- 路径和
- 删除树 / 释放内存
- 动态规划 on tree

```python
def postorder(root):
    if root is None:
        return

    postorder(root.left)
    postorder(root.right)
    print(root.val)
```

## 做题

核心思维：递归

万能模板

```python
def preorderTraversal(root):
    result = []

    def dfs(node):  #定义函数
        if node is None:
            return

        result.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root) # 启动递归
    return result
```

105 从前序与中序遍历构造二叉树
