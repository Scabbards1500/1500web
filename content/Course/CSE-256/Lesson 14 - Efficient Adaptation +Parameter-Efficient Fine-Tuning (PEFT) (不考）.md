
这节课围绕一个问题展开：

> 如何以较低的训练、存储和部署成本，让预训练模型适应新的任务？

主要有三条路线：

1. **Prompting**：通过输入说明任务，不更新模型参数。
2. **PEFT**：只训练少量参数，让模型适应任务。
3. **Model Compression**：把模型压缩成更小、运行更便宜的模型。


### 只改变输入，能否完成任务？
对应 **Prompting / In-Context Learning**。
- 不更新参数。
- 适应灵活。
- 对措辞、示例和顺序敏感。

### 必须训练时，能否只训练少量参数？
对应 **PEFT**。
- LoRA：学习低秩更新。
- BitFit：只训练 Bias。
- Adapter：插入小模块。
- IA³：学习 Activation 的缩放向量。

### 部署时，能否使用更小的模型？
对应 **Model Compression / Knowledge Distillation**。
- 用 Teacher 指导 Student。
- 利用 Soft Labels 传递预测分布信息。
- 降低最终模型的存储和推理成本。

| 路线 | 是否更新参数 | 主要目标 |
|---|---|---|
| Prompting / ICL | 否 | 通过上下文适应任务 |
| Full Fine-Tuning | 更新全部或绝大部分参数 | 充分调整模型 |
| PEFT | 更新少量参数 | 降低任务适应成本 |
| Knowledge Distillation | 训练 Student | 获得更小、更便宜的模型 |




## 第二章：Prompting 与 In-Context Learning（第 6–19 页）

### 2.1 什么是 In-Context Learning？

**In-Context Learning（ICL）** 是在 Prompt 中提供任务描述或示例，让模型在当前上下文中完成任务。

特点是：

- 不更新模型参数。
- 不做 Gradient Update。
- 任务适应发生在 Inference 时。

例如：

    Input: This movie is great!
    Output: positive

    Input: This movie is terrible.
    Output: negative

    Input: I really enjoyed the story.
    Output:

模型根据上下文中的模式生成 `positive`。

这里的“学习”指模型根据上下文改变行为，并不意味着权重发生了更新。

### 2.2 为什么 Next-Token Prediction 可以完成分类？

Language Model 的基本能力是：

$$
P(x_{t+1} \mid x_1,\ldots,x_t)
$$

也就是根据前面的文本预测下一个 token。

如果把分类任务改写成文本补全，就可以使用这个能力：

    Review: The movie’s acting could’ve been better.
    Sentiment:

模型接下来生成的词，就可以作为分类结果。

> Prompting 的关键是把任务转化成模型能够继续补全的文本。

### 2.3 Zero-Shot Prompting

**Zero-Shot** 只描述任务，不提供示例。

例如：

    Classify this review as positive, neutral, or negative.

    Review: The movie’s acting could’ve been better.
    Label:

模型依靠已有能力理解任务并输出标签。

### 2.4 使用 Language Model 分类的两种方式

#### 方法一：直接生成答案

让模型生成文本，再从中提取标签。

例如：

    I would classify this review as negative.

问题是输出格式可能不稳定，需要额外解析。

#### 方法二：比较候选标签的概率

比较：

$$
P(\text{positive} \mid \text{context})
$$

$$
P(\text{neutral} \mid \text{context})
$$

$$
P(\text{negative} \mid \text{context})
$$

选择概率最高的标签：

$$
\hat y
=
\arg\max_{y \in \mathcal{Y}}
P(y \mid \text{context})
$$

这种方式需要访问模型输出概率。

如果标签由多个 token 组成，需要计算整个标签序列的概率。

### 2.5 为什么 Prompt 的措辞很重要？

语义相近的 Prompt，也可能产生明显不同的结果。

例如：

    This review is ___.

与：

    The sentiment expressed in this review is ___.

虽然任务相同，但输入形式不同，模型对标签的概率分配可能不同。

PPT 的实验说明：

> Prompt 的细微变化，可能造成较大的性能波动。

因此，不能只根据一个精心挑选的 Prompt，就认定模型在任务上很稳定。

### 2.6 Few-Shot Prompting 也会受到示例影响

**Few-Shot** 在 Prompt 中提供少量示例。

示例可以帮助模型理解：

- 输入格式。
- 输出格式。
- 可用标签。
- 输入与输出之间的关系。

但结果会受到以下因素影响：

1. 选了哪些示例。
2. 示例的排列顺序。
3. 不同标签出现的数量。

同一组示例只调整顺序，也可能改变结果。

### 2.7 Few-Shot 中的三种 Bias

#### Majority Label Bias

模型倾向于输出示例中出现次数最多的标签。

例如，示例中有很多 `positive`，模型可能更容易继续输出 `positive`。

#### Recency Bias

模型更偏向最近出现的标签。

即使正负示例数量相同，把负面示例集中放在最后，也可能让预测偏向 `negative`。

#### Common Token Bias

模型偏向在 Pretraining 中本来就更常见的标签词。

原因是分类仍通过语言生成实现，标签词本身的语言概率也会影响预测。

### 2.8 Contextual Calibration：校正标签偏好

思路是：

> 先测出模型在没有有效输入内容时，对各个标签有多偏爱，再校正实际预测。

首先使用 Content-Free Input，例如 `N/A`：

$$
p_{\text{cf}}(y)
=
P(y \mid \text{prompt}, x=\text{N/A})
$$

假设得到：

$$
p_{\text{cf}}
=
(0.70, 0.30)
$$

说明这个 Prompt 本身就让模型偏向第一个标签。

对真实输入，模型预测：

$$
P(y \mid \text{prompt},x)
=
(0.55,0.45)
$$

校正时按元素相除：

$$
\hat q(y \mid x)
\propto
\frac{P(y \mid \text{prompt},x)}
{p_{\text{cf}}(y)}
$$

得到：

$$
\left(
\frac{0.55}{0.70},
\frac{0.45}{0.30}
\right)
\approx
(0.79,1.50)
$$

归一化后：

$$
\hat q
\approx
(0.34,0.66)
$$

原本第一个标签概率更高，校正后第二个标签更高。

原因是：第一个标签原本就受到模型偏爱，实际输入反而降低了它相对于自身基准的优势。

### 2.9 为什么错误标签的示例有时也有帮助？

PPT 引用了一个反直觉的实验发现：

> 在部分实验设置中，带有错误标签的 Demonstrations，也可能优于完全没有示例。

这说明示例提供的不只有正确答案，还包括：

- 任务格式。
- 输入的类型。
- 标签集合。
- 输出应当是什么样。

但不能因此推断“标签正确与否不重要”。这个结论依赖具体任务和实验设置。

---

## 第三章：Chain-of-Thought 与 Self-Consistency（第 20–23 页）

### 3.1 Chain-of-Thought Prompting

**Chain-of-Thought（CoT）** 让模型在最终答案之前生成中间步骤。

例如：

    There are 3 cars in a parking lot.
    Another 2 cars arrive.
    How many cars are there?

中间步骤可以是：

    There were 3 cars.
    Another 2 arrived.
    3 + 2 = 5.
    The answer is 5.

对于算术、逻辑等多步骤问题，这种形式常常有助于提高准确率。

### 3.2 Zero-Shot CoT

不一定需要提供带推理过程的示例，也可以加入提示语：

    Let’s think step by step.

这叫 **Zero-Shot CoT**。

需要注意：

> 生成了中间步骤，不代表步骤一定正确，也不保证最终答案正确。

### 3.3 Self-Consistency

单次生成的推理过程可能出错，因此可以：

1. 对同一个问题采样多条推理路径。
2. 提取每条路径的最终答案。
3. 对答案进行多数投票。

例如：

| 推理路径 | 最终答案 |
|---|---|
| 路径 1 | 18 |
| 路径 2 | 26 |
| 路径 3 | 18 |

最终选择 `18`。

形式化地：

$$
\hat a
=
\arg\max_a
\sum_{i=1}^{K}
\mathbf{1}[a_i=a]
$$

其中 $a_i$ 是第 $i$ 条路径的最终答案。

它利用多次采样来提高稳定性，但会增加 Inference 成本。如果多条路径都受到相同错误影响，多数投票也可能选错。

---

## 第四章：PEFT（第 24–29 页）

只训练少量参数来适应新任务。
LoRA是 **是 PEFT 的一种方法**， 具体做法是冻结原始权重，只训练两个小矩阵 \(A\) 和 \(B\)，通过 $Delta W=BA$ 调整模型。

Why PEFT? 
**Full Fine-Tuning** 更新模型的全部参数。
当模型很大时，需要为大量参数保存：模型权重 / Gradients / Optimizer States。
如果每个任务还保存一个完整模型副本，多任务存储成本也会很高。

### 4.2 PEFT 的核心思想
**Parameter-Efficient Fine-Tuning（PEFT）** 只训练少量参数。

一个常见形式是：

- 冻结预训练参数 $\theta$。
- 添加少量可训练参数 $\phi$。
- 满足 $|\phi| \ll |\theta|$。

可以写成：

$$
f_{\theta,\phi}(x)
$$

训练时：

$$
\theta \text{ 固定，只优化 } \phi
$$

不同方法的区别在于：把这些少量参数放在哪里，以及它们如何改变计算。

严格来说，BitFit 是直接训练原模型中的 Bias，不需要额外添加一套参数。

### 4.3 PEFT 与 SFT 是什么关系？

**SFT 和 PEFT 描述的是训练的两个不同方面，可以同时使用。**

| 概念                                        | 回答的问题        | 含义                     |
| ----------------------------------------- | ------------ | ---------------------- |
| **SFT**（Supervised Fine-Tuning）           | **用什么信号训练？** | 用输入和标准答案进行监督训练，让模型模仿答案 |
| **PEFT**（Parameter-Efficient Fine-Tuning） | **训练哪些参数？**  | 只更新少量参数，降低训练和存储成本      |

因此可以同时使用： 用 Instruction–Response 数据进行 SFT，并通过 LoRA 只更新少量参数。

<span style="color: #33cc33;">PEFT 不等于模型压缩!!!</span>

PEFT 减少的是可训练参数，以及相应的训练和任务存储开销。
原来的大模型通常仍然需要加载，Forward Computation 也仍然存在。

因此：

> 只训练 0.1% 的参数，不代表训练显存或计算量只剩下 0.1%。

---

## 第五章：Low-Rank Adaptation（LoRA）（第 30–38 页）

### 5.1 把 Fine-Tuning 看作学习一个变化量

假设原始权重矩阵是：

$$
W_0 \in \mathbb{R}^{d \times d}
$$

Full Fine-Tuning 可以写成：

$$
W' = W_0 + \Delta W
$$

其中 $\Delta W$ 是任务训练带来的变化。

直接学习 $\Delta W$，需要训练 $d^2$ 个参数。

LoRA 提出的问题是：

> 任务需要的权重变化，是否可以用一个低秩矩阵表示？

### 5.2 用两个小矩阵表示更新

LoRA 将更新写成：

$$
\Delta W = BA
$$

其中：

$$
A \in \mathbb{R}^{r \times d}
$$

$$
B \in \mathbb{R}^{d \times r}
$$

并且：

$$
r \ll d
$$

于是：

$$
h = W_0x + BAx
$$

训练时冻结 $W_0$，只训练 $A$ 和 $B$。

这里采用 PPT 的简化写法，省略了实现中常见的缩放系数。

### 5.3 为什么叫 Low-Rank？

因为：

$$
\operatorname{rank}(BA) \leq r
$$

可以把这条支路理解成：

1. $A$ 将 $d$ 维输入映射到 $r$ 维。
2. $B$ 再将它映射回 $d$ 维。
3. 得到的修正量加到原模型输出上。

注意：

> 被限制为低秩的是更新量 $\Delta W$，原始权重 $W_0$ 并没有被要求是低秩矩阵。

### 5.4 参数能减少多少？

Full Fine-Tuning 一个矩阵需要：

$$
d^2
$$

LoRA 需要：

$$
dr+rd = 2dr
$$

两者比例是：

$$
\frac{2dr}{d^2}
=
\frac{2r}{d}
$$

例如：

$$
d=4096,\qquad r=8
$$

原矩阵参数量：

$$
4096^2 = 16{,}777{,}216
$$

LoRA 参数量：

$$
2 \times 4096 \times 8 = 65{,}536
$$

可训练参数约为原矩阵的：

$$
0.39\%
$$

### 5.5 为什么初始化时一个随机、一个为零？

PPT 中使用：

- $A$：随机初始化。
- $B$：初始化为零。

因此初始时：

$$
BA=0
$$

模型输出仍然是：

$$
h=W_0x
$$

也就是从预训练模型的原有行为开始训练。

补充理解：不能把两个矩阵都初始化为零，否则这条乘积支路中的两个矩阵最初都会得不到有效梯度。

### 5.6 Rank 越大越好吗？

更大的 $r$ 提供更强的更新表达能力，但不一定带来更好的效果。

PPT 展示的实验中：

- 较小的 Rank 已经可以达到不错的结果。
- 继续增加 Rank 的收益往往有限。
- 更新哪些矩阵，有时比增大 Rank 更重要。

例如，相比只更新 $W_q$，同时更新 $W_q$ 和 $W_v$ 可能更有效。

这不表示所有任务都只需要很小的 Rank，具体仍取决于任务和模型。

### 5.7 LoRA 为什么可以合并？

训练完成后计算：

$$
W_{\text{merged}} = W_0 + BA
$$

Inference 时直接使用：

$$
h=W_{\text{merged}}x
$$

不需要继续单独计算 LoRA 支路，因此合并后可以保持原线性层的计算结构。

多任务时，也可以共享一个 Base Model，只为每个任务保存对应的 $A$ 和 $B$。

---

## 第六章：Bias-Only Fine-Tuning（BitFit）（第 39 页）

### 6.1 核心思想

普通线性层是：

$$
h=Wx+b
$$

BitFit 冻结权重矩阵 $W$，只更新 Bias $b$。

PPT 展示的设置中，可训练参数占比非常小，低于 0.1%。

### 6.2 如何理解它？

对于单个线性层：

- $W$ 控制输入特征如何组合。
- $b$ 调整输出的偏移。

在深层网络中，改变 Bias 也会影响后续的激活与计算，因此仍然可以改变模型行为。

它的优点是简单、参数少，但适应能力受到限制，效果依赖任务以及模型中是否存在相应 Bias。

---

## 第七章：Adapters 与 IA³（第 40–45 页）

### 7.1 Adapter 是什么？

**Adapter** 在预训练网络中插入小型可训练模块：

- 原模型参数冻结。
- 新模块参数参与训练。

PPT 中的 Bottleneck Adapter 可以写成：

$$
\Delta h = B\sigma(Ah)
$$

其中：

- $A$：Down Projection。
- $\sigma$：Nonlinear Activation。
- $B$：Up Projection。

加上 Residual Connection 后：

$$
h' = h+B\sigma(Ah)
$$

### 7.2 Adapter 和 LoRA 的区别

两者都可以使用先降维、再升维的结构，但作用方式不同：

| 方法 | 核心形式 | 特点 |
|---|---|---|
| LoRA | $\Delta h=BAx$ | 线性修正，可以合并进权重 |
| Adapter | $\Delta h=B\sigma(Ah)$ | 通常含非线性，保留额外模块 |

由于 Adapter 通常包含非线性，不能像标准 LoRA 一样直接合并成一个固定权重更新。

![[Pasted image 20261116175025.png]]

### 7.3 Sequential Adapter

Sequential Adapter 读取原模块的输出。

先计算：

$$
h=f_\theta(x)
$$

再计算：

$$
h'=h+B\sigma(Ah)
$$

也就是：

$$
h'
=
f_\theta(x)
+
B\sigma(Af_\theta(x))
$$

### 7.4 Parallel Adapter

Parallel Adapter 与原模块并行，读取相同输入：

$$
h'
=
f_\theta(x)
+
B\sigma(Ax)
$$

两者的主要区别是：

- Sequential：根据原模块的输出生成修正。
- Parallel：直接根据原模块的输入生成修正。

### 7.5 IA³：学习缩放 Activation

![[Pasted image 20261116180056.png]]

**IA³** 不学习完整的小型网络，而是学习一个缩放向量。

简化形式：

$$
h'=\ell \odot h
$$

其中 $\odot$ 表示逐元素相乘。

例如：

- 缩放系数为 $2$：放大该维。
- 缩放系数为 $0.5$：减弱该维。
- 缩放系数为 $0$：屏蔽该维。

PPT 中主要对以下位置进行缩放：

1. Attention 的 Key。
2. Attention 的 Value。
3. FFN 的中间 Activation。

初始化时：

$$
\ell=\mathbf{1}
$$

因此初始行为与原模型相同。

直观理解：

> IA³ 学习如何调节已有特征的强弱。

---

## 第八章：PEFT 的比较与统一理解（第 46–51 页）

PPT 强调了几个经验：

1. 在相似参数预算下，LoRA 与 Adapter 的效果可能比较接近。
2. 参数放在哪里，有时比单纯增加参数量更重要。
3. 部署需求也会影响选择：LoRA 可以合并，Adapter 通常保留额外计算模块。
4. 面对较大的 Domain Shift，Full Fine-Tuning 可能有更强的适应能力；部分实验中 LoRA 对原有能力的遗忘更少。

这些是经验观察，不能作为所有场景下的保证。所以 <span style="color: #66ccff;">没有一种方法对所有任务都最好</span>

### 8.2 不同方法可以组合

可以同时加入 LoRA 和 Adapter，并学习每种修正的权重：

$$
h'
=
f_\theta(x)
+
g_1\Delta h_{\text{LoRA}}
+
g_2\Delta h_{\text{Adapter}}
$$

Gate 可以由输入决定：

$$
g_i=\operatorname{sigmoid}(w_i^\top x)
$$

因此，不同输入或不同层可以使用不同程度的修正。

PPT 提到的 UniPELT 还结合了 Prefix Tuning。

### 8.3 统一视角：如何修正 Hidden Representation？

很多 PEFT 方法都可以理解为：

$$
h' = h+\Delta h
$$

它们的区别是如何构造 $\Delta h$。

| 方法 | 如何改变原模型 |
|---|---|
| LoRA | 添加低秩线性更新 |
| BitFit | 更新原有 Bias |
| Adapter | 添加小型可训练模块 |
| IA³ | 缩放 Activation |

这有助于理解：这些方法虽然名称不同，但都在使用有限的可训练参数调整模型行为。

### 8.4 显存节省来自哪里？

冻结大部分参数，可以减少对应的：

- Gradient 存储。
- Optimizer States 存储。
- 参数更新开销。

但 Base Model 的权重和训练所需的中间计算仍然占用资源。

PPT 的显存对比表说明，LoRA 可以让一些无法进行 Full Fine-Tuning 的模型，在给定硬件条件下完成训练。

具体节省比例依赖模型、精度、Batch Size 和训练实现。

---

## 第九章：Model Compression 与 Knowledge Distillation（第 52–56 页）

PEFT 关注如何便宜地适应任务。
**Model Compression** 关注如何让最终模型本身更小、更快、更便宜。
PPT 的提纲提到了 Pruning 等方法，但本次实际重点讲的是 **Knowledge Distillation**。

![[Pasted image 20261116180945.png]]
### 9.2 Teacher 和 Student


Knowledge Distillation 使用：

- **Teacher**：较大、较强的模型。
- **Student**：较小、运行成本较低的模型。

让 Student 模仿 Teacher 的输出分布：

$$
\mathcal{L}_{\text{KD}}
=
\sum_x
D_{\mathrm{KL}}
\left(
P_T(\cdot \mid x)
\parallel
P_S(\cdot \mid x)
\right)
$$

其中：

- $P_T$：Teacher 的预测分布。
- $P_S$：Student 的预测分布。

目标是让小模型尽量保留大模型的能力。

### 9.3 为什么 Soft Labels 比 Hard Labels 信息更多？
![[Pasted image 20261116181029.png]]

假设输入是：

    The acting was ___.

Hard Label 只告诉模型正确答案是 `great`：

$$
P_{\text{hard}}(\text{great})=1
$$

Teacher 则可能给出：

| Token | 概率 |
|---|---:|
| great | 0.45 |
| good | 0.30 |
| fine | 0.15 |
| superb | 0.08 |
| awful | 0.02 |

这个分布不仅表达正确答案，还表达：

- `good` 也是合理的候选。
- `fine` 有一定可能。
- `awful` 与当前语境很不一致。

因此：

> Soft Labels 包含候选答案之间的相对关系。

Hard Label 的 Cross-Entropy 是：

$$
\mathcal{L}_{\text{hard}}
=
-\log P_S(y^* \mid x)
$$

Soft Label 的 Cross-Entropy 是：

$$
\mathcal{L}_{\text{soft}}
=
-\sum_y
P_T(y \mid x)
\log P_S(y \mid x)
$$

Teacher 固定时，最小化这个 Soft Cross-Entropy，与最小化上述 KL Divergence 对 Student 来说是等价的。

### 9.4 DistilBERT 的例子

DistilBERT 使用 BERT 作为 Teacher：

- Teacher：12 层。
- Student：6 层。
- 初始化时复制 Teacher 中隔层的参数。

训练结合了三种目标：

1. **Distillation Loss**：匹配 Teacher 的输出分布。
2. **Masked Language Modeling Loss**：学习预测被遮盖的 token。
3. **Hidden-State Alignment**：使用 Cosine Similarity 对齐内部表示。

PPT 给出的结果：

| 指标 | BERT | DistilBERT |
|---|---:|---:|
| 参数量 | 110M | 66M |
| 实验中的推理耗时 | 668 s | 410 s |
| GLUE Score | 79.5 | 77.0 |

它说明：通过 Distillation，可以用一定的性能损失换取更小的模型和更低的推理成本。

