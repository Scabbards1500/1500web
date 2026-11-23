![[Pasted image 20261116153220.png]]

这节课主要回答三个问题：

1. GPT 这类 Decoder-only 模型如何学习和生成文本？
2. 模型给出概率后，怎样选择下一个 token？
3. BART、T5 这类 Encoder–Decoder 模型与 GPT、BERT 有什么区别？

## 第一章：Decoder-only Language Models（第 5–11 页）

### 1. 从 Transformer 到 GPT
![[Pasted image 20261116140338.png]]

原始 Transformer 有两部分：

- **Encoder**：读取输入。
- **Decoder**：根据输入和已生成的内容，逐步输出文本。

原始 Decoder 内有 **Cross-Attention**，用来读取 Encoder 的输出。

GPT 的做法是：**只保留 Decoder，并去掉 Cross-Attention。** 因为它直接续写已有文本，不需要单独的 Encoder。

每个 block 的主要结构是：

- **Masked Self-Attention**
- **Feed-Forward Network**

最后通过 Linear layer 和 Softmax，得到下一个 token 的概率。

### 2. 为什么只能看左边？
![[Pasted image 20261116140421.png]]

例如训练句子：

> The cat sat on the mat.

模型学习：

| 看到的内容 | 预测目标 |
|---|---|
| The | cat |
| The cat | sat |
| The cat sat | on |

**Causal Mask** 保证每个位置只能看自己及之前的 token，不能偷看预测目标。

- **训练时**：整句话已知，可以在一次 forward pass 中并行计算多个位置的预测。
- **生成时**：未来 token 未知，需要选出一个、追加进去，再预测下一个。

因此，**训练可以并行，生成通常是 Autoregressive 的。**

### 3. GPT-2：Zero-shot

过去常见流程是：

> Pretraining → 针对某个任务进行 Supervised Fine-tuning

GPT-2 探索的是：**能不能不更新参数，仅通过上下文，让模型完成任务？**

例如摘要，在文章后追加： TL;DR:

模型接着生成摘要，因为它在预训练网页中见过这种模式。

翻译、问答也是类似的：网页里本来就有翻译对、问题与答案等内容，模型从中学会了相应的续写模式。

这就是这里讲的 **Zero-shot**：执行目标任务时，不再用该任务的数据更新参数。它不代表预训练时从未见过类似任务形式。

## 第二章：Decoding Type（第 12–17 页）

Language Model 输出的是：

$$
P(x_t \mid x_{<t})
$$

即“给定前文，下一个 token 的概率分布”。

例如：

| token | 概率 |
|---|---:|
| cat | 0.5 |
| dog | 0.3 |
| bird | 0.2 |

**Decoding 决定从这个分布中选择哪个 token。** 同一个模型，换一种 Decoding 方法，生成风格就可能明显不同。
![[Pasted image 20261116141133.png]]
### 1. Greedy Decoding

每一步都选概率最大的 token：

$$
x_t = \arg\max_x P(x \mid x_{<t})
$$

优点是简单；问题是**当前最好的选择，不一定带来整体最好的序列**。

例如，假设只生成两个 token：

- 第一步选 A 的概率是 0.6，但后续最佳选择的条件概率只有 0.4。
- 第一步选 B 的概率是 0.4，后续最佳选择的条件概率有 0.9。

两个完整路径的概率：

$$
P(A, a) = 0.6 \times 0.4 = 0.24
$$
$$
P(B, b) = 0.4 \times 0.9 = 0.36
$$

Greedy 一开始选 A，就错过了后面的更优路径。

### 2. Beam Search

**Beam Search 每一步保留 $b$ 条得分最好的候选序列**，继续扩展它们。

它比 Greedy 考虑更多路径，目标是寻找整体概率较高的文本，但不保证找到全局最优。

问题在于：

> **概率最高的文本，不一定是质量最高的文本。**

在开放式写作中，它容易生成安全、普通、重复的内容。

**哪些任务更适合 Beam Search?**

| 任务类型                      | 特点             | PPT 的主要观点                      |
| ------------------------- | -------------- | ------------------------------ |
| **Directed Generation**   | 输出受输入约束，如翻译、摘要 | 高概率输出通常更符合任务目标，Beam Search 较适用 |
| **Open-ended Generation** | 有很多合理续写，如故事创作  | 单纯最大化概率容易降低多样性、引起重复            |

这是一种经验性的区分，不代表所有翻译或摘要任务都必须用 Beam Search。


### 3. Sampling

#### Pure Sampling：按概率抽取

假设：

$$
P = (0.6, 0.3, 0.1)
$$

Sampling 的意思是：三个 token 分别有 60%、30%、10% 的机会被选中。

PPT 用区间解释：

- token 1：占据 $[0, 0.6)$。
- token 2：占据 $[0.6, 0.9)$。
- token 3：占据 $[0.9, 1]$。

随机抽取：

$$
r \sim \operatorname{Uniform}(0,1)
$$

落在哪段，就选择对应 token。

**概率最大不意味着每次都选它。**


但是！
词表中有大量低概率 token，形成 **Long Tail**。

单个尾部 token 概率很小，但它们加起来可能占据相当大的概率质量。持续生成时，就可能抽到不合适的 token，导致：

- 偏离主题。
- 语义不连贯。
- 出现奇怪的转折。

因此可以先截断不可靠的尾部，再 Sampling。

#### Top-k Sampling：固定候选数量

步骤是：

1. 保留概率最高的 $k$ 个 token。
2. 将它们的概率重新归一化。
3. 从中抽取。

例如：

| token | 原概率 |
|---|---:|
| A | 0.5 |
| B | 0.3 |
| C | 0.1 |
| D | 0.1 |

设置 $k=2$，只保留 A、B：

$$
P'(A) = \frac{0.5}{0.8} = 0.625
$$

$$
P'(B) = \frac{0.3}{0.8} = 0.375
$$

缺点是 **$k$ 固定，但每一步的分布形状不同**：

- 分布很平：合理选项很多，小 $k$ 会删掉太多。
- 分布很尖：合理选项很少，大 $k$ 会留下不可靠选项。

#### Top-p / Nucleus Sampling：固定累计概率

Top-p 不固定数量，而是：

> **按概率从高到低排列，保留累计概率至少达到 $p$ 的最小前缀。**

例如：

| token | 概率 |
|---|---:|
| A | 0.50 |
| B | 0.30 |
| C | 0.15 |
| D | 0.05 |

设置 $p=0.9$：

- A：累计 0.50。
- A、B：累计 0.80。
- A、B、C：累计 0.95，达到阈值。

因此保留 A、B、C，再归一化、抽样。

设保留的集合为 $V_p$，则：

$$
P'(x) =
\begin{cases}
\displaystyle \frac{P(x)}{\sum_{y \in V_p} P(y)}, & x \in V_p \\
0, & x \notin V_p
\end{cases}
$$

**它能随着分布变化，自动改变候选数量。**

| 方法 | 固定的是什么？ | 候选数量 |
|---|---|---|
| Top-k | token 数量 $k$ | 固定 |
| Top-p | 累计概率阈值 $p$ | 动态变化 |

## 第四章：Temperature（第 28–31 页）

Top-k、Top-p 通过删除候选控制随机性；**Temperature 通过改变概率分布的尖锐程度控制随机性。**

设模型输出的 Logits 为 $u_j$：

$$
P_j = \frac{\exp(u_j / T)}{\sum_k \exp(u_k / T)}
$$

| Temperature | 分布变化 | 生成倾向 |
|---|---|---|
| $0 < T < 1$ | 更尖锐 | 更偏向高概率 token |
| $T = 1$ | 原始分布 | 保持原始随机性 |
| $T > 1$ | 更平坦 | 更多样，也更容易出错 |

例如 Logits 为 $[2, 1, 0]$：

| $T$ | 概率分布，约 |
|---|---|
| 0.5 | $[0.867, 0.117, 0.016]$ |
| 1 | $[0.665, 0.245, 0.090]$ |
| 2 | $[0.506, 0.307, 0.186]$ |

**Temperature 不改变 Logits 的排序，只改变概率差距。** 对有限的正温度，它也不会直接删除 token。

两个极限帮助理解：

- 当 $T \to 0^+$，概率集中到最大 Logit 对应的 token；若最大值唯一，趋向 Greedy。
- 当 $T \to \infty$，对有限 Logits，概率趋向均匀分布。

注意：第 30 页 $T=10$ 的结果应约为 $[0.367, 0.332, 0.301]$，不过“趋向均匀”的结论正确。

## 第五章：让Docoding 更快，或更符合规则（第 32–36 页）

这部分两种方法解决的问题不同：
- **Speculative Decoding**：加速生成。
- **Constrained Decoding**：限制输出格式。

### 1. Speculative Decoding

小模型起草，大模型验证：**小模型接受规则与修正分布配合，保证最后的输出分布和直接用大模型 Sampling 一样。**

正常情况下，大模型生成一个 token，就需要进行一次生成步骤。

Speculative Decoding 的流程是：
![[Pasted image 20261116143226.png]]

1. 小的 **Draft Model** 连续提出多个 token。
2. 大的 **Target Model** 用一次 forward pass，并行计算这些位置的概率。
3. 按接受规则检查草稿。
4. 保留接受的前缀；遇到第一次拒绝时，进行修正，丢弃后面的草稿。

为什么拒绝后要丢弃后续内容？

因为后续草稿建立在那个被拒绝的 token 上，前文已经变化。

**精确版本的关键是：通过接受与修正机制，使最终采样分布仍然等于 Target Model 的分布。** 它不是简单地判断“小模型答案看起来是否合理”。

收益来自：一次大模型调用可以推进多个 token。但加速程度取决于草稿接受率、模型成本和硬件。

#### Self-Speculative Decoding
**同一个模型也可以承担起草和验证两个角色。**
第 35 页介绍 **LayerSkip**：

使用同一个模型较浅的层产生草稿，再用完整模型验证，从而减少对独立 Draft Model 的需求。


### 3. Constrained Decoding：不符合规则的 token 直接屏蔽

例如要求：

> 只回答 A、B、C 或 D。

即使模型认为 “Sure” 很可能出现，也把它屏蔽，只在合法选项中重新归一化、选择。

可以将不允许 token 的 Logits 设为：

$$
-\infty
$$

Softmax 后，它们的概率就是 0。

设当前允许的 token 集合为 $A_t$，则：

$$
\widetilde{P}(x) =
\begin{cases}
\displaystyle \frac{P(x)}{\sum_{y \in A_t} P(y)}, & x \in A_t \\
0, & x \notin A_t
\end{cases}
$$

它和 Top-k、Top-p 的差别是：

- Top-k、Top-p：根据模型概率决定保留谁。
- Constrained Decoding：根据外部规则决定允许谁。

**合法格式不等于答案正确**：它能保证选项属于 A–D，却不能保证选对。

## 第六章：Decoding Evaluation（第 37–41 页）


> **生成文本的 Perplexity 越低，不代表文本越好 **评价 Decoding，需要同时考虑** 连贯性、多样性和任务完成情况**。

### 1. 为什么低 Perplexity 也可能有问题？

Perplexity 衡量模型对文本的“意外程度”：

$$
\operatorname{PPL}(x_{1:N}) =
\exp\left(
-\frac{1}{N}
\sum_{t=1}^{N}
\log P(x_t \mid x_{<t})
\right)
$$

Greedy、Beam Search 本来就在偏向高概率 token，因此它们生成的文本，在同一个模型下通常具有低 Perplexity。

但这类文本可能过于普通、重复。

人类写作也会使用低概率但信息量大的词，所以**自然文本并不是每一步都选最容易预测的词**。

这里要区分：

- **固定测试集上的 Perplexity**：用于评价模型预测能力。
- **模型自己生成文本的 Perplexity**：同时受 Decoding 策略影响，不能单独代表文本质量。

### 2. Self-BLEU 看什么？

Self-BLEU 衡量生成样本彼此有多相似：

- **高 Self-BLEU**：不同样本很像，多样性较低。
- **低 Self-BLEU**：样本差异较大。

但乱码也能很多样，因此多样性也必须和连贯性一起看。



## 第七章：BEncoder-Decoder Models（第 42–50 页）

前面讲的是 Decoder-only，这部分回到 **Encoder–Decoder**：

- Encoder 阅读输入。
- Decoder 根据输入，Autoregressively 生成输出。

两者都使用“破坏文本，再恢复内容”的预训练思路。

### 1. BART：恢复完整原文
![[Pasted image 20261116144304.png]]
例如原文：

> The cat sat on the mat.

破坏后的输入可能是：

> The [MASK] mat.

训练目标是输出完整原文：

> The cat sat on the mat.

设原文为 $x$，破坏后的输入为 $\widetilde{x}$，训练目标是提高：

$$
P(x \mid \widetilde{x})
$$

PPT 介绍的破坏方式包括：

- **Token Masking**：遮住 token。
- **Token Deletion**：删除 token。
- **Text Infilling**：把一段内容替换为一个 mask。
- **Sentence Permutation**：打乱句子顺序。

用于摘要时，再进行 Fine-tuning：

- 输入：完整文章。
- 输出：摘要。

预训练让模型先学会根据输入重建语言内容，再适应摘要任务。

### 2. T5：恢复被删除的片段

T5 是 **Text-to-Text Transfer Transformer**。

与 BART 恢复整段原文不同，T5 的 **Span Corruption** 使用不同的 Sentinel Tokens 标记缺口，输出缺失片段。
![[Pasted image 20261116144359.png]]
示意：

- 原文：`The cat sat on the mat.`
- 输入：`The <X> on the <Y>`
- 输出：`<X> cat sat <Y> mat.`

每个标记帮助模型知道，输出片段对应哪个缺口。

### 3. 所有任务都写成 Text-to-Text

T5 将不同任务统一成文本输入、文本输出，用 Task Prefix 告诉模型做什么：

- 翻译：`translate English to German: ...`
- 摘要：`summarize: ...`
- 情感分类：`sentiment: ...`

例如分类也可以输出文本标签：

- 输入：`sentiment: This movie is great.`
- 输出：`positive`

因此，多种任务可以混合训练，共用一个输出形式。

### 4. BERT、BART、T5 的区别

| 模型   | 架构              | 典型预训练输入         | 预测目标         |
| ---- | --------------- | --------------- | ------------ |
| BERT | Encoder-only    | 部分 token 被遮住    | 被遮住位置的 token |
| BART | Encoder–Decoder | 被破坏的文本          | 完整原文         |
| T5   | Encoder–Decoder | 用不同标记替换若干 spans | 被移除的 spans   |

PPT 说 BERT “不能生成”，更准确地理解是：**BERT 不是为从左到右自由生成文本而设计的**。它当然能预测 masked token。

## 第八章：为什么 Decoder-only 成为主流？（第 51–54 页）

PPT 给出四个原因：

1. **目标统一**：Pretraining、Fine-tuning 和 Prompting 都可以围绕 Next-token Prediction 展开。
2. **训练信号密集**：标准 Causal LM 几乎每个位置都有预测目标；BERT 通常只在选中的 masked positions 上计算预测损失。
3. **方便使用 KV Cache**：追加新 token 时，旧位置不依赖未来，因此旧的 Key、Value 可以复用。
4. **架构简单**：一个 Transformer stack，便于扩大规模和统一使用方式。

但 PPT 也强调：**Decoder-only 并非在所有比较中都更好。** Encoder–Decoder 依然适合翻译、语音等任务。

