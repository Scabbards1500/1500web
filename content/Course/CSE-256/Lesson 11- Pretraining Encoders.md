time: 2026/11/10

![[Pasted image 20261116153247.png]]


这节课的核心是：**为什么先做 Pretraining，以及 ELMo、BERT 如何把预训练学到的知识用于不同任务。**

# 1. 为什么需要 Pretraining？（第 3–7 页）

假设要做 Sentiment Analysis（情感分析），传统方法通常需要：

> 收集带情感标签的数据，再训练一个专门的模型。

问题是：**标注数据少，而且每做一个任务，都要重新学习很多语言知识。**

Pretraining 把学习分成两个阶段：

| 阶段 | 使用的数据 | 学习目标 |
|---|---|---|
| Pretraining | 大量原始文本 | 学习可复用的语言表示 |
| Downstream adaptation | 具体任务的数据 | 将已有表示用于情感分析、问答等任务 |
PPT 展示昂贵训练成本，是为了引出一个动机：**预训练一次，再把学到的能力复用到很多任务中。**

### 1.1 为什么预测词就能学到有用的东西？

| 预测任务 | 可能需要的知识 |
|---|---|
| She ___ to the store yesterday. | Grammar：过去时 |
| The capital of Kenya is ___. | World knowledge |
| I waited an hour; the service was ___. | Sentiment |
| Ana gave Bo her keys, so ___ could drive. | 人物指代与语境 |

**为了把词预测准确，模型需要学习语法、知识和上下文关系。**

例如，Next-token Prediction 的目标是：

$$
\mathcal{L}
=
-\sum_t \log P_\theta(x_t \mid x_{<t})
$$

意思是：**根据前面的 tokens，给真实的下一个 token 更高的概率。**

训练答案直接来自原始文本，因此属于 **Self-supervised Learning**，不需要人工逐条标注。


# 2. Sentiment Neuron：预测字符也能学到情感（第 8–11 页）

研究者用大量商品评论训练一个 LSTM，任务只是：

> **预测下一个字符。**

训练过程中，没有直接要求模型判断评论是 positive 还是 negative。

但研究者发现，模型的某个 hidden unit 与情感高度相关：
- Activation 较高时，往往对应 positive sentiment。
- Activation 较低时，往往对应 negative sentiment。

例如：

> The first half is charming and funny, **but** then the plot falls apart...

模型读到前半句时，这个 unit 偏向正面；读到后面的负面描述后，它的值发生变化。

**模型为了预测评论如何继续，顺带学到了评论的情感。**

### 2.2 Probing

> 用一个简单的 classifier，检查 hidden representations 中是否包含某种信息。

例如，输入某个 neuron 的 activation，让 classifier 预测情感。

需要区分：**能从表示中读出情感信息，并不意味着所有概念都各自对应一个 neuron。**



# 3. ELMo：让同一个词根据上下文得到不同表示（第 12–16 页）

### 3.1 Static Embeddings 的问题

Word2Vec、GloVe 通常为每个词保存一个固定向量。

例如：

```text
river bank     → 河岸
bank account   → 银行账户
```

虽然意思不同，但 `bank` 查到的 embedding 相同。

我们希望得到 **Contextual Embeddings**：

> 同一个词在不同上下文中，产生不同的向量。

### 3.2 ELMo 如何利用上下文？

![[Pasted image 20261121211948.png]]

ELMo 使用两个方向的 Language Model：

| 方向 | 读取顺序 | 能利用的信息 |
|---|---|---|
| Forward LM | 从左到右 | 左侧上下文 |
| Backward LM | 从右到左 | 右侧上下文 |

把两个方向的 hidden states 拼接起来：

$$
h_{k,j}
=
\left[
\overrightarrow{h}_{k,j};
\overleftarrow{h}_{k,j}
\right]
$$

其中：

- $k$：第几个 token。
- $j$：第几层。
- $[a;b]$：Concatenation，拼接两个向量。

这样，`bank` 的表示就能包含两侧的信息。

 <span style="color: #ff9933;">**两个方向的 LSTM 分别计算，然后拼接；它们不是像 BERT 那样在每一层直接融合左右信息。**</span>

### 3.3 为什么要混合不同层？
![[Pasted image 20261121212030.png]]
不同层可能包含不同类型的信息：

- 较低层偏向 Syntax。
- 较高层偏向 Word sense 和语义信息。

ELMo 不只取最后一层，而是做加权求和：

$$
\mathrm{ELMo}_k
=
\gamma
\sum_{j=0}^{L}
s_j h_{k,j}
$$

| 符号        | 含义                    |
| --------- | --------------------- |
| $h_{k,j}$ | Token $k$ 在第 $j$ 层的表示 |
| $s_j$     | 该层的权重，经 Softmax 归一化   |
| $\gamma$  | 整体缩放系数                |
**每个下游任务学习自己的 $s_j$ 和 $\gamma$，决定更需要哪些层的信息。**


### 3.4 ELMo 怎么用于下游任务？

![[Pasted image 20261121212057.png]]
把 ELMo 表示与原来的 word embedding 拼起来：

$$
\mathrm{input}_k
=
[\mathrm{word\ embedding}_k;\mathrm{ELMo}_k]
$$

再送入自己的任务模型，例如用于 NER 的 BiLSTM + CRF。

典型训练方式：

| 部分 | 是否更新 |
|---|---|
| 预训练的 ELMo biLM | Frozen |
| 层混合权重 $s_j$、缩放系数 $\gamma$ | 更新 |
| 下游任务模型 | 更新 |
**ELMo 的主要作用是升级输入表示，下游任务仍然需要自己的模型。**



# 4. BERT（第 17–18 页）

BERT 将思路进一步推进：

> 不仅复用输入表示，还把整个预训练网络作为下游任务模型的主体。

## ELMo vs BERT 

| 方面 | ELMo | BERT |
|---|---|---|
| Architecture | LSTM | Transformer Encoder |
| Context | 两个单向 LM 的表示拼接 | 每层都可以融合左右上下文 |
| Pretraining objective | 两个方向分别预测下一个词 | Masked Language Modeling |
| 典型使用方式 | Frozen features | Fine-tune 整个网络 |

例如，做情感分析时：

```text
输入句子 → 预训练 BERT → 小型分类层 → positive / negative
```

训练时，BERT 和分类层一起更新。


Bert 的 **Bidirectional** 指：

> 一个位置可以通过 Self-Attention，同时利用左边和右边的 tokens。

它并不是“先从左读一遍，再从右读一遍”。



## BERT  Training - Masked Language Modeling（第 19–20 页）

训练中，不能直接让它预测看得见的词
假设输入是：

```text
John visited Madagascar yesterday.
```

如果要求模型预测 `Madagascar`，但输入中已经有这个词，模型就可能直接复制答案。

BERT 的办法是：

> **先隐藏部分输入，再让模型恢复原词。**

```text
输入：John visited [MASK] yesterday.
目标：Madagascar
```

这叫 **Masked Language Modeling，MLM**。
它可以同时利用- 左边：`John visited` - 右边：`yesterday` 两边的信息

### 5.2 Causal Mask vs MLM Mask 

|       | Causal Mask     | BERT 的 MLM Mask |
| ----- | --------------- | --------------- |
| 改变什么？ | Attention 的可见范围 | 输入中的部分 tokens   |
| 怎么做？  | 禁止看未来位置         | 替换被选中的词         |
| 目的    | 根据前缀预测后续 token  | 根据两侧上下文恢复原词     |

**Causal Mask 是“挡住后面”；MLM Mask 是“把部分输入词藏起来”。**

### 5.3 怎么制作mask

假设有 **1,000 个 tokens**，随机选出 **150 个位置**。模型需要预测这些位置原本是什么，**MLM loss 只在这 150 个位置上计算**。

**第二步：对选中的 150 个位置，分三种方式出题。**

|出题方式|数量|模型看到的输入|正确答案|
|---|--:|---|---|
|把原词藏起来|120 个（80%）|my dog is **[MASK]**|hairy|
|换成随机词|15 个（10%）|my dog is **apple**|hairy|
|保留原词|15 个（10%）|my dog is **hairy**|hairy|
**不管输入怎么变，答案始终是原来的词。**



### 5.4 Loss 在哪些位置计算？

你可能疑惑：**最后一种不是直接把答案给它了吗？**

对，这部分确实允许模型看到答案。但模型**不知道哪些位置被选中、哪些词被随机替换了**。这样混合训练，可以减少它对 `[MASK]` 这个特殊符号的依赖的依赖——因为真正用于分类等任务时，输入通常没有 `[MASK]`。

- **没被选中的 85%**：提供上下文，不直接计算该位置的 MLM loss。
- **被选中但保持原样的 1.5%**：也是预测目标，需要计算 loss。

Loss 只在**选中的位置**计算，包括其中保持不变的位置：

$$
\mathcal{L}_{\mathrm{MLM}}
=
-\sum_{i\in M}
\log P_\theta(x_i \mid \tilde{x})
$$

其中：

- $M$：选中位置的集合。
- $x_i$：原始 token。
- $\tilde{x}$：处理后的输入。


### 5.5 Next Sentence Prediction (NSP)

原始 BERT 还用了 **Next Sentence Prediction，NSP**：判断句子 B 是否为语料中接在句子 A 后面的句子。

PPT 随后指出，RoBERTa 的实验表明，NSP 并不是获得强效果的必要条件。



## 6. BERT 的输入与下游任务（第 21–23、39 页）

### 6.1 输入向量由三部分相加

$$
x_i
=
E_{\mathrm{token}}(t_i)
+
E_{\mathrm{segment}}(s_i)
+
E_{\mathrm{position}}(i)
$$

| 部分 | 作用 |
|---|---|
| Token Embedding | 表示 token 是什么 |
| Segment Embedding | 区分句子 A 和句子 B |
| Position Embedding | 表示 token 所在位置 |

注意： <span style="color: #ff9933;">这里是相加，不是拼接；相加后维度保持不变。</span>

原始 BERT 使用 **Learned Position Embeddings**，不是固定的 Sinusoidal Position Encoding。

### 6.2 特殊 Tokens

句子对的输入形式为：

```text
[CLS] sentence A [SEP] sentence B [SEP]
```

- `[CLS]`：通常取它的最终 hidden state 做整个输入的分类。
- `[SEP]`：分隔句子或标记结束。

`[CLS]` 并不是直接对所有 token 求平均，而是通过网络计算得到自己的表示。

### 6.3 模型大小

| 模型 | Layers | Hidden dimension | Attention heads | Parameters |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

### 6.4 不同任务如何接 Head？
![[Pasted image 20261122001045.png]]

| 任务 | 使用的表示 | 输出 |
|---|---|---|
| Sentence classification | `[CLS]` 的最终表示 | 句子类别 |
| Sentence-pair classification | 两句共同输入后的 `[CLS]` 表示 | 两句之间的关系 |
| Token classification | 每个 token 的最终表示 | 每个位置的标签 |

例如，Sentiment Analysis：

$$
p(y \mid x)
=
\operatorname{softmax}
\left(
Wh_{\mathrm{[CLS]}}+b
\right)
$$

NER 则对各位置分别分类：

```text
Ana   flew   to   Rome
PER    O      O   LOC
```

### 6.5 Fine-tuning 怎么训练？

Fine-tuning 时通常使用较小的 Learning Rate，避免一下子大幅改变预训练参数。

课件给出的设置是：

- Learning Rate：$2$–$5\times10^{-5}$。
- Epochs：约 2–4。
- 更新参数：BERT 和新增的 task head 一起更新。

第 39 页代码表达的流程是：

> 加载 pretrained weights，加一个新的 classification head，再一起训练。

第 23 页的 GLUE 结果主要说明：**同一个预训练模型可以迁移到多种语言理解任务，而不必为每个任务重新设计整套网络。**

---

###  BERT 不适合generate（第 24–31 页）

这一部分解释 **Bidirectional Attention 与 Causal Attention 在生成时的差别**。

BERT 可以勉强接着写，但不擅长

例如：

```text
The cat [MASK]        → sat
The cat sat [MASK]    → on
The cat sat on [MASK] → the
```

主要有两个问题：

1. 训练目标是恢复被选中的词，并非专门学习从左到右连续生成。
2. 新增 token 后，旧位置的表示也可能改变，需要重新计算大量内容。

因此，**不适合生成，不等于绝对不能生成。**

# KV Cache

Decoder 可以复用过去的计算

Causal Attention 规定：

> **过去的位置不能看未来。**

例如，已有：

```text
The cat
```

追加 `sat` 后：

```text
The cat sat
```

`The` 和 `cat` 仍然看不到 `sat`，所以它们在每一层的旧表示都不会因为追加而改变。

因此可以保存旧 tokens 的 **Keys 和 Values**，这就是 **KV Cache**。

## 7.3 KV Cache Calculation

对于当前新位置 $t$：

$$
o_t
=
\operatorname{softmax}
\left(
\frac{q_t K_{\leq t}^{\top}}{\sqrt{d_k}}
\right)
V_{\leq t}
$$

直观上：

- 当前 $q_t$：这一步需要寻找什么信息。
- 历史 $K$：用于计算相关性。
- 历史 $V$：提供被汇总的信息。

每一步生成时：

1. 计算新 token 在当前层的 $q_t,k_t,v_t$。
2. 将 $k_t,v_t$ 加入该层 Cache。
3. 用 $q_t$ 查询已有的 Keys，并加权汇总 Values。
4. 继续通过后面的层，最终预测下一个 token。

**每一层都有自己的 KV Cache；新 token 仍然需要经过所有层。**

### 7.4 为什么不保存旧 Queries？

因为当前输出使用的是**当前 Query**。

旧 Queries 对应旧位置的输出，而那些输出已经计算过了，当前这一步不需要重新计算。

### 7.5 为什么 BERT 的 Cache 会失效？

BERT 允许旧位置看到新 token。

追加 `sat` 后：

```text
The 和 cat 可以关注 sat
→ 第一层的输出发生变化
→ 第二层由这些输出生成的 K、V 发生变化
→ 更高层的旧 Cache 也随之失效
```

课件特别强调一个细节：

> 第一层的旧 K、V 只由原来的输入表示生成，在旧输入和位置不变时可以保持不变；问题从第二层开始。

**Decoder 能稳定缓存，是因为过去的位置在每一层都不依赖新追加的未来位置。**

### 7.6 有 Cache 每步成本也不固定

![[Pasted image 20261122012444.png]]

| 计算部分 | 是否随 Context length 增长？ |
|---|---|
| 新 token 的 projections、FFN 等 | 通常不增长 |
| 当前 Query 与历史 Keys 做 Attention | 增长 |
| 最后的词表预测 | 不随 Context length 增长 |

第 25 页的“$n^2/2$ 对比 $n$”，主要是在数：

> **需要重新处理多少个 token positions。**

不能直接理解为整个生成过程的总运算量变成线性。

## 8. KV Cache 的显存成本与 GQA（第 32–34 页）

### 8.1 KV Cache 占多少显存？

KV Cache 节省计算，但需要显存。

其容量大致为：

$$
\mathrm{Cache\ bytes}
=
B\times T\times L
\times 2
\times H_{\mathrm{KV}}
\times d_{\mathrm{head}}
\times b
$$

| 符号 | 含义 |
|---|---|
| $B$ | 同时处理的序列数 |
| $T$ | 每条序列缓存的长度 |
| $L$ | 层数 |
| $2$ | 分别保存 K 和 V |
| $H_{\mathrm{KV}}$ | KV heads 数量 |
| $d_{\mathrm{head}}$ | 每个 head 的维度 |
| $b$ | 每个数值占用的字节数 |

**Context 越长、同时服务的用户越多，Cache 就越大。**

### 8.2 如何缩小 Cache？

让多个 Query heads 共享 KV heads。

| 方法 | Query heads 与 KV heads 的关系 |
|---|---|
| Multi-Head Attention，MHA | 每个 Query head 对应自己的 KV head |
| Grouped-Query Attention，GQA | 一组 Query heads 共享一个 KV head |
| Multi-Query Attention，MQA | 所有 Query heads 共享一个 KV head |

例如，有 8 个 Query heads：

```text
MHA：8 个 KV heads
GQA：2 个 KV heads
MQA：1 个 KV head
```

相对于 MHA，后两者的 Cache 分别缩小到约 $1/4$ 和 $1/8$。

**GQA 保留了多个 Query heads，但减少了需要存储的 Keys 和 Values。**

# BERT Research

> BERT 的 研究重点逐渐从“为每个任务设计模型”，转向改进预训练目标、数据和规模。

## Seires
课件列出的后续方向包括：

- **RoBERTa**：改进训练配置，增加训练数据。
- **ELECTRA**：学习识别被替换的 tokens。
- **DeBERTa**：分别处理内容与相对位置的信息。

## Data
课件从 BERT 的 BooksCorpus 和 Wikipedia，讲到更大规模的开放语料。

来源包括：

- Web pages。
- Code。
- Books、Wikipedia。
- Papers、PDFs。
- 数学相关内容。

这里要区分：

| 概念 | 含义 |
|---|---|
| Data quantity | 数据有多少 |
| Data composition | 不同来源各占多少 |
| Data quality | 内容是否有价值、噪声是否少 |
| Data transparency | 数据来源与处理过程是否公开 |

**数据规模只是一个维度；选了什么内容、如何筛选，同样是预训练设计的一部分。**
<span style="color: #ff9933;"> Word 和 token 不能直接等同。</span>

# 三种 Transformer 的区别

| 架构 | 代表 | Attention 可见范围 | 典型用途 |
|---|---|---|---|
| Encoder-only | BERT | 输入内部双向可见 | 分类、标注、表示学习 |
| Decoder-only | GPT | 只能看当前及之前的位置 | 自回归生成 |
| Encoder-decoder | T5 | Encoder 双向；Decoder 因果，并读取 Encoder 输出 | 输入到输出的生成任务 |

### Encoder-decoder 如何工作？

1. Encoder 先理解输入。
2. Decoder 用 Causal Self-Attention 读取已生成的前缀。
3. Decoder 再通过 Cross-Attention 读取 Encoder 输出。

Cross-Attention 中：

- **Q 来自 Decoder。**
- **K、V 来自 Encoder。**
