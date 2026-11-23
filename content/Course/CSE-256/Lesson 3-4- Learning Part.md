time: 2026/10/19


这节课主要回答以下几个问题：
- Learning Part I：应该 optimize 什么？——定义 loss。
- Learning Part II：如何 optimize？——计算 gradients，更新 parameters。


# 从 prediction 走向 learning（第 1–6 页）

前面介绍了 text classification 的基本流程：

1. 把文本转换成 feature vector，例如 Bag-of-Words（BoW）。
2. 用 linear classifier 计算各类别的 scores：

$$
s = Wx + b
$$

但这个公式只告诉我们如何 prediction，还没有告诉我们如何 learning：什么样的 $W,b$ 才是好的？

![[Pasted image 20261019002854.png]]

PPT 从 probability distribution 的角度定义目标：

$$
P_\theta(y \mid x) \approx P_{\text{data}}(y \mid x)
$$

其中：

- $P_{\text{data}}(y \mid x)$：真实世界中，给定文本 $x$ 时各 label 的 probability。
- $P_\theta(y \mid x)$：模型预测的 probability。
- $\theta = (W,b)$：模型的 parameters。

**Learning 的目标，就是调整 parameters，让 model distribution 尽量接近 data distribution。**



# 用 KL divergence 衡量 distributions 的差异（第 7–11 页）

![[Pasted image 20261019131308.png]]
先固定一封邮件，比较它对应的两组 label probabilities：
![[Pasted image 20261019003202.png]]

| Label | Data distribution p | Model distribution q |
|---|---:|---:|
| Spam | 0.9 | 0.4 |
| Not spam | 0.1 | 0.6 |

模型低估了 Spam 的 probability。我们需要一个数衡量这种差异。

PPT 使用 Kullback–Leibler divergence，简称 KL divergence，**比较的是两组「每个标签的概率」**：

$$
D_{\mathrm{KL}}(p \parallel q)
=
\sum_y p(y)\log\frac{p(y)}{q(y)}
$$
$\sum_y$ is every label y of one input

计算出来是
$$D_{\mathrm{KL}}\bigl(p(\cdot\mid x)\parallel q(\cdot\mid x)\bigr) = \underbrace{0.9\log\frac{0.9}{0.4}}_{\text{Spam}} + \underbrace{0.1\log\frac{0.1}{0.6}}_{\text{Not spam}} \approx 0.551 $$

可以理解为：

> 按照真实的 probability 加权，衡量 model distribution 与 data distribution 的差异。

它有三个重要性质：

- KL divergence 非负。
- 两个 distributions 相同时，KL divergence 为 0。
- 它不满足 symmetry，交换两个 distributions 后，结果通常不同。
	 **<span style="color: #66ccff;">KL divergence 是“以一个 distribution 为基准，衡量另一个 distribution 和它的差异”**交换两个 distribution，基准也变了</span>

因此，KL divergence 不是通常数学意义上的 distance。



**为什么要乘 p(y)？**

因为真实情况下更常出现的 label，应该在比较中占更大的权重。

例如，真实情况下 Spam 的 probability 为 0.9，那么模型对 Spam 的预测就应当占较大权重。


### 从一封邮件推广到所有邮件

模型需要处理不同的 inputs，因此还要对输入取 expectation：

把两层放到一个式子里（假设输入是离散的）：

$$\underbrace{\sum_x p_{\text{data}}(x)}_{\text{遍历邮件，按邮件出现概率加权}} \left[ \underbrace{\sum_y p_{\text{data}}(y\mid x) \log\frac{p_{\text{data}}(y\mid x)}{q_\theta(y\mid x)}}_{\text{固定这封邮件，遍历 Spam 和 Not spam}} \right]$$


比如，**内层**先算出邮件 A 的 KL 是 \(0.55\)，邮件 B 的 KL 是 \(0.10\)。如果两封邮件出现的概率分别是 \(0.8\) 和 \(0.2\)

Expectation 就是按照出现的 probability 计算 weighted average。

这里有两层平均：

1. KL 内部，对一封邮件的不同 labels 加权。
2. 外部 expectation，对不同邮件加权。

**训练目标是 minimize 这个平均 KL divergence。**




# 从 KL divergence 推到 MLE（第 12–15 页）

这是本节课最核心的推导。

⭐ 展开平均 KL divergence, 利用 log 的性质，把右边拆成两项：

![[Pasted image 20261019004720.png]]

第一项只取决于 data distribution，与 parameters 无关。调整模型不会改变这一项。

因此：

> Minimize average KL divergence，等价于 maximize expected log-likelihood。

也就是：

![[Pasted image 20261019004842.png]]

Asks Pθ to give high probability to the labels that actually occur
Problem: the expectation is still over Pdata, which we don’t have


## Maximum Likelihood Estimation（MLE）

对于真实观察到的 input–label pair，我们希望模型给正确 label 更高的 probability。**MLE 就是选择一组 parameters，让已经观察到的 training labels 尽可能有可能出现。**

由于 data distribution 未知，我们无法直接计算 expectation。

但可以用 training dataset 上的平均值近似（Monte Carlo estimate）：

![[Pasted image 20261019005239.png]]



于是得到 **Maximum Likelihood Estimation（MLE）**：

$$
\theta_{\mathrm{ML}}
=
\arg\max_\theta
\frac{1}{N}
\sum_{i=1}^{N}
\log P_\theta(y_i \mid x_i)
$$

其中，argmax 表示找到让目标最大的那组 parameters。

min KL ⇒ max expected log-likelihood ⇒ max average over D = MLE


### why likelihood？

在 samples 相互独立的通常假设下，给定这些 inputs，观察到整组 labels 的 likelihood 是：

$$
\mathcal{L}(\theta)
=
\prod_{i=1}^{N}
P_\theta(y_i \mid x_i)
$$

取 log，把乘积转换为求和：

$$
\log \mathcal{L}(\theta)
=
\sum_{i=1}^{N}
\log P_\theta(y_i \mid x_i)
$$

因为 log 是 monotonically increasing，maximize likelihood 与 maximize log-likelihood 会得到相同的最优 parameters。除以固定的 sample 数量，也不会改变最优 parameters。




# Softmax (p16-21) 

softmax把 logits 转成 probabilities

整个 prediction 流程是：

文本 → Feature vector → Linear classifier → Logits → Softmax → Probabilities

**Softmax 提供了 MLE 所需的 probabilities，但它本身还不是 loss function。**
MLE 要求模型输出 probabilities，但 linear classifier 输出的是 logits：
![[Pasted image 20261019005630.png]]


Logits 可以为负，而且总和不一定为 1，不能直接当成 probabilities。

Softmax 完成这个转换：
![[Pasted image 20261019005715.png]]

计算过程分两步：

1. 对每个 logit 取 exponential，让结果为正。
2. 进行 normalization，让所有 probabilities 的总和为 1。

PPT 中的例子：

![[Pasted image 20261019005918.png]]
模型认为 Personal 最有可能。

注意：
- Softmax 不改变 logits 的排序。最高的 logit 对应最高的 probability。
- 表格中的 probabilities 经过 rounding，总和可能不恰好等于 1。PT 中 Work 显示为 0.00，是 rounding 的结果，实际并不等于零。





## 从 MLE 得到 Softmax loss（第 18–21 页）

MLE 的目标是 maximize average log-likelihood：

$$
\max_\theta
\frac{1}{N}
\sum_{i=1}^{N}
\log P_\theta(y_i \mid x_i)
$$

训练时通常写成 minimize loss，因此加一个负号：

$$
L_{\text{data}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log P_\theta(y_i \mid x_i)
$$

对单个 sample就是 <span style="color: red;">negative log-likelihood（NLL）</span>：

$$
L_i
=
-\log P_\theta(y_i \mid x_i)
$$


代入 Softmax：

$$
L_i
=
-\log
\frac{e^{s_{y_i}}}
{\sum_j e^{s_j}}
$$

这里，分子使用的是 ground-truth class 对应的 logit，而不是 predicted class 对应的 logit。

### e.g：ground-truth 是 Spam

模型给 Spam 的 probability 只有约 0.13：

$$
L_i
=
-\log(0.13)
\approx 2.04
$$

不同 probabilities 对应的 loss：

| Ground-truth class probability | NLL |
|---:|---:|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

**给 ground-truth class 的 probability 越高，loss 越低。**

因此，这个 loss 不仅看 classification 是否正确，还考虑模型给正确答案分配了多少 probability。

## 与 cross-entropy 的关系

如果 ground-truth label 用 one-hot vector 表示，cross-entropy 为：

$$
L_i
=
-\sum_k t_k\log p_k
$$

其中：

- 正确类别对应的 one-hot entry 为 1。
- 其他类别对应的 entries 为 0。

所以求和后只剩一项：

$$
L_i = -\log p_{y_i}
$$

因此，在这个 classification setting 中：
<span style="color: #ff9933;"> Softmax loss = Cross-entropy applied to Softmax probabilities = NLL</span>

## 初始化时的 loss

第 21 页中，初始化的 weights 很小，产生的 logits 比较接近，
更一般地，如果有 K 个 classes，每个 class 的 probability 都约为：

$$
p_k \approx \frac{1}{K}
$$

那么初始 loss 大约为：

$$
L_i \approx \log K
$$

这个关系可以帮助判断刚开始训练时的 loss 是否合理。



# Regularization 与 generalization（第 22–29 页）

纯 MLE 只要求提高 training dataset 上的 likelihood。但 training dataset 是有限的，其中可能包含 noise 或偶然出现的规律。

例如，一个少见词恰好只在 training dataset 的 Spam 邮件中出现，模型可能给它特别大的 weight。但这个规律在 unseen data 上未必成立，可能造成 overfitting。

我们真正希望获得的是 generalization：模型能够在 unseen data 上做好 prediction。

### 1. Regularization 

Regularization 为 learning 加入一种偏好：在拟合 training data 的同时，限制某些形式的 model complexity。

目标变为：

![[Pasted image 20261019130822.png]]


其中：

- 第一项是 data loss，要求模型符合 training labels。
- 第二项是 regularization term，对 parameters 施加约束。
- lambda 控制 regularization 的强度。
	▶	λ = 0: No regularization
	▶	λ = ∞: Only care about regularization
	▶	λ = 0.5: Balance between fitting the data and regularization
	较小     | 对 weights 的约束较弱
	较大     | 对 weights 的约束较强 
	过大     | 可能造成 underfitting |

Underfitting 指模型受到过强限制，连 training data 中的有效规律也没有充分学到。
通常使用 validation set 选择 lambda。
PPT 中的 0.5 只是例子，不是通用的最佳值；实际效果还取决于 features 的尺度和 loss 的定义。



### 2. L2 regularization

L2 regularization 把所有 weights 的平方加起来：

$$
R(W)
=
\sum_k\sum_l W_{k,l}^{2}
$$

对于矩阵 W，更严格的 notation 是 Frobenius norm squared：

$$
R(W)=\|W\|_F^2
$$

大 weights 会受到更强的 penalty：

| Weight | 对 L2 penalty 的贡献 |
|---:|---:|
| 0.1 | 0.01 |
| 1 | 1 |
| 10 | 100 |

因此，L2 regularization 鼓励模型使用幅度较小的 weights。



### 4. 其他方式

| Method              | 基本作用                                                      |     |
| ------------------- | --------------------------------------------------------- | --- |
| L1 regularization   | 惩罚 weights 的绝对值之和，常用于鼓励 sparsity                          |     |
| Dropout             | 训练时随机屏蔽部分 units                                           |     |
| Layer normalization | 主要改善训练稳定性，也可能影响 generalization。与直接加入 L1、L2 penalty 的机制不同。 |     |

L1 regularization 的公式为：

$$
R(W)
=
\sum_k\sum_l |W_{k,l}|
$$


# 完整 training objective（第 30–32 页）

把 linear classifier、Softmax、NLL 和 L2 regularization 合在一起：

$$
L(W,b)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{
e^{(Wx_i+b)_{y_i}}
}{
\sum_j e^{(Wx_i+b)_j}
}
+
\lambda
\sum_k\sum_l W_{k,l}^{2}
$$

这个公式包含四个部分：

1. Linear classifier：计算各类别的 logits。
2. Softmax：把 logits 转换成 probabilities。
3. NLL：惩罚 ground-truth class probability 过低。
4. L2 regularization：限制 weights 的幅度。

| Concept             | 回答的问题                                       |
| ------------------- | ------------------------------------------- |
| KL divergence       | Model distribution 与 data distribution 差多少？ |
| MLE                 | 如何根据 training samples 选择 parameters？        |
| Softmax             | 如何把 logits 转换成 probabilities？               |
| NLL / Cross-entropy | 如何评价 ground-truth class 的 probability？      |
| Regularization      | 如何限制 overfitting，帮助 generalization？         |


整节课的主线是：

Loss landscape → Gradient → Analytic gradient → Gradient descent → Minibatch → Backpropagation

# 把 learning 理解为在 loss landscape 中寻找低点（第 3–8 页）

### 1. Loss 是 parameters 的函数

上一节课定义了 training loss：

$$
L(W)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log P_W(y_i \mid x_i)
$$

Training dataset 固定后，换一组 weights，就会得到一个不同的 loss。

因此，可以把 loss 看成关于 weights 的函数。

为了方便画图，假设模型只有两个 weights：
![[Pasted image 20261019133351.png]]

$$
L = L(w_1,w_2) = w_1^2 + w_2^2
$$

这时：

- 横向两个方向对应两个 weights。
- 高度对应 loss。
- 每个位置代表一组不同的 parameters。

这张曲面就叫 loss landscape。
- 高处：loss 大，当前 parameters 对 training data 的预测较差。
- 低处：loss 小，当前 parameters 对 training data 的预测较好。
- Learning：不断调整 parameters，往更低的位置移动。

它的 minimum 在：

$$
w_1=w_2=0
$$

注意，这只是帮助理解的 toy function，不是实际的 Softmax loss。

### 2. Data space 和 parameter space 不一样

第 8 页特别区分了两种空间：
![[Pasted image 20261019133606.png]]

| Space | 一个点代表什么？ |
|---|---|
| Data space | 一条 sample，例如一封邮件 |
| Parameter space | 一组 parameters，也就是一个 classifier |

在 parameter space 中移动，表示改变 classifier 的 weights。
对应到 data space，就是 classifier 的 decision boundary 在变化。

# 如何决定 parameters 的更新方向？（第 9–14 页）

### 1. Random search

不利用梯度信息，随机尝试多组 parameters，并保留 loss 最低的一组。

问题是：parameters 很多时，可能的组合极其庞大。Random search 没有充分利用当前位置附近的信息。

### 2. Iterative improvement

另一种思路是：

1. 从一组 initial parameters 开始。
2. 判断当前位置往哪个方向移动，可以降低 loss。
3. 移动一小步。
4. 在新位置重新判断。

关键问题变成：如何知道哪个方向会降低 loss？
答案是计算 gradient。




## 3. Derivative 
$$
f'(w)
=
\frac {df(x)}{dx} 
$$

一维情况下：

$$
f'(w)
=
\lim_{h \to 0}
\frac{f(w+h)-f(w)}{h}
$$

实际计算时，可以选择一个很小但不为零的 h：

$$
f'(w)
\approx
\frac{f(w+h)-f(w)}{h}
$$

这叫 finite difference。

直觉是：

> 稍微改变一个 parameter，观察 loss 改变了多少。



### 从一维 derivative 理解

![[Pasted image 20261019134113.png]]


对于 $L(w)=w^2$，derivative 为 $2w$：
$w=1.5$ 时导数为正，应减小 $w$；
$w=-1.5$ 时导数为负，应增大 $w$，即沿导数的反方向小步更新以降低 loss。

两种情况都可以用同一条 update rule：

$$
w \leftarrow w-\eta L'(w)
$$

其中 eta 是 learning rate，控制每次移动的大小。

<span style="color: #66ccff;">更新时减去 derivative，而不是直接减小 weight。Weight 可能增加，也可能减小。</span>


### Derivative、partial derivative 和 gradient

| Term               | 含义                                    |
| ------------------ | ------------------------------------- |
| Derivative         | 一维函数沿唯一输入方向的变化率                       |
| Partial derivative | 只改变一个 parameter、固定其他 parameters 时的变化率 |
| Gradient           | 把所有 partial derivatives 放在一起          |

如果模型有两个 weights：

$$
\nabla L
=
\left[
\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2}
\right]
$$

在通常的 Euclidean geometry 下，gradient 指向局部 steepest increase 的方向。

因此，negative gradient 给出了局部 steepest decrease 的方向。




# Gradients

Gradient（梯度）是由 loss 对各个参数的偏导数组成的向量，指向当前位置 loss 增长最快的方向。
**numerical gradients 用数值近似，analytic gradients 用求导公式计算。**

| 方法                      | 怎么算                      | 特点                 |
| ----------------------- | ------------------------ | ------------------ |
| **Numerical gradients** | 给参数一个小扰动，通过 loss 的变化估算梯度 | 慢、有近似误差，常用于检查梯度    |
| **Analytic gradients**  | 根据求导规则和 chain rule 计算梯度  | 快、无有限差分近似误差，用于实际训练 |

## Numerical gradients——通过小幅扰动估计 gradient（第 15–20 页）

### 3. PPT 中的 numerical example

![[Pasted image 20261019134916.png]]

当前 loss：

$$
L(W)=1.25347
$$

取：

$$
h=0.0001
$$

只增加第一个 weight 后，loss 变成：

$$
L(W+h e_1)=1.25322
$$

这里，e1 表示只改变第一个 coordinate。

于是：

$$
\frac{\partial L}{\partial w_1}
\approx
\frac{1.25322-1.25347}{0.0001}
=
-2.5
$$

这说明：

- 增加第一个 weight，会降低 loss。
- Gradient descent 会增加这个 weight。

对于第二个 weight：

$$
\frac{\partial L}{\partial w_2}
\approx
\frac{1.25353-1.25347}{0.0001}
=
0.6
$$

这说明：

- 增加第二个 weight，会提高 loss。
- Gradient descent 会减小这个 weight。

### 为什么 numerical gradients 不适合大模型？

Numerical gradients 要逐个扰动参数，30,730 个 weights 就需要计算 30,731 次 loss，计算成本高，而且结果只是近似值，会受步长 \(h\) 和浮点精度影响。

Numerical gradients 的主要问题是：

- Computational cost 高。
- 只是 approximation。
- 结果受 h 和 floating-point precision 影响。

因此，训练时通常使用 <span style="color: red;"> analytic gradients</span>

补充：numerical gradients 仍可用于 gradient checking，检查另一种 gradient 实现是否正确。

## Analytic gradients——推导 Softmax loss 的 gradient（第 21–25 页）

Analytic gradient 直接利用 calculus，根据 loss 的表达式推导。

这里先分析一条 training sample，并暂时省略 bias 和 regularization。

推不明白，先贴一下ppt吧

![[Pasted image 20261019162745.png]]

![[Pasted image 20261019162904.png]]

![[Pasted image 20261019162925.png]]

### 1. 写出 class score

第 k 个 class 的 weight vector 是 Wk：

$$
s_k = W_k \cdot x_i
$$

Softmax probability：

$$
p_k
=
\frac{e^{W_k \cdot x_i}}
{\sum_j e^{W_j \cdot x_i}}
$$

单条 sample 的 loss：

$$
L_i=-\log p_{y_i}
$$

### 2. 展开 loss

利用 log 的性质：

$$
L_i
=
-W_{y_i}\cdot x_i
+
\log
\left(
\sum_j e^{W_j\cdot x_i}
\right)
$$

现在有两项：

1. Ground-truth class 的负 score。
2. 所有 classes scores 对应的 log-sum-exp。

### 3. 对 ground-truth class 的 weights 求 gradient

第一项的 gradient：

$$
\frac{\partial}{\partial W_{y_i}}
\left(
-W_{y_i}\cdot x_i
\right)
=
-x_i
$$

第二项通过 chain rule 得到：

$$
\frac{\partial}{\partial W_{y_i}}
\log
\left(
\sum_j e^{W_j\cdot x_i}
\right)
=
p_{y_i}x_i
$$

因此：

$$
\frac{\partial L_i}{\partial W_{y_i}}
=
(p_{y_i}-1)x_i
$$

### 4. 对其他 classes 的 weights 求 gradient

当 k 不是 ground-truth class 时，第一项与 Wk 无关，所以：

$$
\frac{\partial L_i}{\partial W_k}
=
p_kx_i
$$

### 5. 统一公式

引入 one-hot target tk：

- Ground-truth class 对应 tk = 1。
- 其他 classes 对应 tk = 0。

那么：

$$
\boxed{
\frac{\partial L_i}{\partial W_k}
=
(p_k-t_k)x_i
}
$$

这个公式非常重要：

> Gradient = prediction 与 target 的差异，乘以 input features。

## 这些 gradients 如何让 classifier 学会分类？（第 26–30 页）

### 1. Ground-truth class 的 update

对 ground-truth class：

$$
W_{y_i}
\leftarrow
W_{y_i}
-
\eta(p_{y_i}-1)x_i
$$

等价于：

$$
W_{y_i}
\leftarrow
W_{y_i}
+
\eta(1-p_{y_i})x_i
$$

也就是沿着 input vector 的方向增加 class weights。

这样通常会提高该 sample 对 ground-truth class 的 score。

而且：

- 正确类别的 probability 越低，update 越强。
- 正确类别的 probability 越接近 1，update 越弱。

### 2. 其他 classes 的 update

对于其他 classes：

$$
W_k
\leftarrow
W_k-\eta p_kx_i
$$

也就是降低这些 classes 对当前 input 的 score。

模型错误分配给某个 class 的 probability 越大，对它的调整也越强。

这解释了为什么 class weight vectors 可以被理解为 templates：

- 向自己类别的 examples 靠近。
- 远离其他类别的 examples。

### 3.  text classification example

![[Pasted image 20261019163101.png]]

输入邮件：

> too many drug trials, too few patients

使用三个 binary features：

| Feature | Value |
|---|---:|
| drug | 1 |
| patients | 1 |
| baseball | 0 |
![[Pasted image 20261019164751.png]]
所以：

$$
x_i=[1,1,0]
$$

Ground-truth class 是 Health，但模型预测为：

$$
p=[0.2,\ 0.5,\ 0.3]
$$

顺序是 Health、Sports、Science。

One-hot target：

$$
t=[1,0,0]
$$

根据统一公式：

$$
\nabla_{W_{\text{Health}}}L_i
=
(0.2-1)[1,1,0]
=
[-0.8,-0.8,0]
$$

$$
\nabla_{W_{\text{Sports}}}L_i
=
0.5[1,1,0]
=
[0.5,0.5,0]
$$

$$
\nabla_{W_{\text{Science}}}L_i
=
0.3[1,1,0]
=
[0.3,0.3,0]
$$

假设 learning rate 为 0.1，实际的 weight changes 是：

| Class   | Gradient        | Weight change     |
| ------- | --------------- | ----------------- |
| Health  | [-0.8, -0.8, 0] | [+0.08, +0.08, 0] |
| Sports  | [0.5, 0.5, 0]   | [-0.05, -0.05, 0] |
| Science | [0.3, 0.3, 0]   | [-0.03, -0.03, 0] |

因此：
- drug、patients 对 Health 的 weights 增加。
- drug、patients 对 Sports、Science 的 weights 减小。
- baseball 的 weights 不受这条 sample 的 data loss 影响，因为对应 feature 为 0。

注意：如果加入 L2 regularization，即使某个 feature 为 0，对应 weight 仍可能因 regularization term 而变化。

### 4. 从单条 sample 推广到整个 dataset

完整 loss 是各 sample losses 的平均：

$$
L(W)
=
\frac{1}{N}
\sum_{i=1}^{N}L_i(W)
$$

因此：

$$
\nabla_W L(W)
=
\frac{1}{N}
\sum_{i=1}^{N}\nabla_W L_i(W)
$$

也就是：

> Gradient of an average = average of the gradients。

每条 sample 提供自己的调整意见，完整 update 使用这些意见的平均值。

## Gradient descent 与 learning rate（第 31 页）

有了 gradient，就可以执行 gradient descent：

1. Initialize parameters。
2. 计算当前 gradient。
3. 沿 negative gradient 更新 parameters。
4. 重复计算与更新。

核心公式：

$$
\boxed{
W \leftarrow W-\eta\nabla_W L(W)
}
$$

### Learning rate 的作用

| Learning rate | 可能的效果 |
|---|---|
| 太小 | 每次 update 很小，训练缓慢 |
| 合适 | 能够逐步降低 loss |
| 太大 | 可能越过低点，产生 oscillation 或 divergence |

因此，negative gradient 只是提供局部方向，不能保证任意大小的 step 都会降低 loss。

实际 update 的大小取决于两个因素：

$$
\|\Delta W\|
=
\eta\|\nabla_W L(W)\|
$$

在固定 learning rate 下，如果接近某个 minimum 时 gradient 逐渐变小，updates 也会自然变小。

PPT 用“loss 不再下降”说明停止条件。实际训练还可以根据 validation performance、gradient magnitude 或 training budget 决定何时停止。


## 三种常见 gradient descent

| Method                           | 每次 update 使用多少 samples？ |
| -------------------------------- | ----------------------: |
| Full-batch gradient descent      |     整个 training dataset |
| Minibatch gradient descent       |             一小批 samples |
| Stochastic gradient descent（SGD） |      严格来说，每次 1 条 sample |

实际使用中，SGD 这个名称也经常用于 minibatch updates。


### Minibatch gradient descent（第 32 页）

<span style="color: #66ccff;">我才知道原来我们平时选的batch size 就是minibatch....</span>

#### 1. Full-batch gradient descent 的问题

如果 training dataset 有一百万封邮件，每次 update 都遍历一百万封邮件，成本很高。

Full gradient 是：

$$
g_{\text{full}}
=
\frac{1}{N}
\sum_{i=1}^{N}\nabla_W L_i
$$

#### 2. 使用 minibatch 估计 full gradient

每次随机抽取一个 minibatch B，例如 32、64 或 256 条 samples：

$$
g_B
=
\frac{1}{|B|}
\sum_{i\in B}\nabla_W L_i
$$

然后更新：

$$
W\leftarrow W-\eta g_B
$$

不需要每次都看完所有 training samples。

#### 3. 为什么可以这样做？

上一节课已经用 training dataset 的平均值近似 data distribution 上的 expectation。

这里再用 minibatch 的平均值近似整个 training dataset 的平均值。

也就是：

Population expectation → Dataset average → Minibatch average

在合适的随机 sampling 下，minibatch gradient 是 full gradient 的 unbiased estimate。

但每次抽到的 samples 不一样，因此 updates 会带有 noise，loss 不一定每一步都下降。



## 第八章：Backpropagation 与 gradient descent 的区别（第 33 页）

这两个概念经常被混在一起，但负责不同的事情。

| Concept | 负责什么？ |
|---|---|
| Backpropagation | 高效计算 loss 对 parameters 的 gradients |
| Gradient descent | 使用 gradients 更新 parameters |
| Autograd | 自动完成 differentiation 的系统 |

Backpropagation 基于 chain rule，并且会复用 intermediate computations。
可以把训练过程写成：

Forward computation → Loss → Backpropagation → Gradients → Parameter update
因此：

> Backpropagation 解决“gradient 怎么算”，optimizer 解决“parameters 怎么更新”。


## 第九章：为什么还需要 neural networks？（第 34–36 页）

Linear classifiers 简单、快速、易解释，但对于当前 input features，只能表达 linear decision boundaries；即使 parameters 最优，也可能无法正确分类。

需要区分：

- **Optimization problem**：模型能表达，但没找到好的 parameters。
- **Model capacity limitation**：模型本身表达不了，增加训练步数也无法解决。

带 nonlinear activation 的 feedforward neural networks 能表达更复杂的 decision boundaries，训练流程仍是：

Define loss → Compute gradients → Update parameters


# 本节课最需要掌握的内容

### 1. Gradient 的含义

Gradient 表示当前位置 loss 局部上升的方向；gradient descent 沿相反方向移动。

### 2. Softmax loss 的核心 gradient

$$
\frac{\partial L_i}{\partial W_k}
=
(p_k-t_k)x_i
$$

Prediction 与 target 的差异，决定 update 的方向和强度。

### 3. Gradient descent 的 update rule

$$
W\leftarrow W-\eta\nabla_W L
$$

### 4. Full gradient 与 minibatch gradient

Full gradient 使用整个 training dataset；minibatch gradient 使用一小批 samples 进行估计。

### 5. Backpropagation 与 optimization 的分工

Backpropagation 计算 gradients；optimizer 使用 gradients 更新 parameters。