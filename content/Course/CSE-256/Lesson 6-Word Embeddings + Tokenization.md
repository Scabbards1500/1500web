time: 2026/11/06


**Tokenization 是切分，Encoding 是编号，Embedding 是把编号映射成向量。** 在文本输入处理中，可以这样理解：

|步骤|作用|示例|
|---|---|---|
|**Tokenization**|把文本切成 tokens|`"I love cats"` → `["I", "love", "cats"]`|
|**Encoding**|将 tokens 转成 token IDs|→ `[12, 58, 306]`|
|**Embedding**|根据 ID 查找对应的 learned vector|→ 每个 token 对应一个向量|

注意 **Encoding 和 Embedding 不是一个意思**：ID 只是编号，embedding 才是模型学到的向量表示。

注意bpe 和 我们常规语境下的encoding 不一样
- **BPE encoding**：应用学到的 merge rules，把文本切成最终 tokens。
- **Token ID encoding**：把 tokens 转成对应的数字 IDs。


Words as Vectors
![[Pasted image 20261120160602.png]]

Tokenization
![[Pasted image 20261120160618.png]]

# 第一章：为什么需要 learned word representations？（第 11–14 页）

### 1. 传统方式：手工定义 meaning
传统方法用 **WordNet** 等手工构建的 lexical resource 表示词义，例如：

- **Synonymy (同义词)**：car 与 automobile。
- **Is-a relations**：mouse → rodent → animal。

但这种方法分类较粗、难以更新，也不直接提供连续的 **similarity score**，例如 good 与 great 有多接近。

因此，我们希望从 data 中学习 **word representations**，捕捉词的用法及不同程度的 similarity。不过，learned embeddings 也不会自动保持最新；它们仍然受 training corpus 和更新时间限制。

### 关键创新: 把 word 表示为 vector

每个 word 对应一个 dense vector：

$$
v_w\in\mathbb{R}^{d}
$$

例如：

$$
v_{\text{cat}}=[0.81,-0.42,0.33,\ldots]
$$

目标是让具有相似 usage 或 meaning 的 words，在 vector space 中有相关的 representations。

相似程度可以用 dot product 或 cosine similarity 衡量。

Dot product：

$$
u^\top v=\sum_j u_jv_j
$$

Cosine similarity：

$$
\operatorname{cos}(u,v)
=
\frac{u^\top v}{\|u\|\|v\|}
$$

Cosine similarity 会消除 vector magnitude 的影响，主要比较 direction。


注意！
PPT 用 good 与 bad 指向相反方向作示意。

但实际 distributional embeddings 中，antonyms 也可能很接近，因为它们常出现在类似 contexts：

- a good movie
- a bad movie

因此：<span style="color: #ff9933;">Distributional similarity 不等于 synonymy，反义关系也不保证对应 opposite directions。</span>

# 第二章：Distributional semantics（第 15–20 页）

核心假设: **出现在相似 contexts 中的 words，往往具有相近或相关的 meanings。**
### 1. 从 context 推断 meaning

PPT 使用 kombucha 举例, 即使不知道 **kombucha** 的含义，也能从 “a glass of kombucha” 和 “drink kombucha” 推测它是一种饮品。

这就是 distributional semantics 的核心直觉：**Words that occur in similar contexts tend to have related meanings.**

### 2. 为什么需要多个 dimensions？

如果只用一个 number 表示动物的 size：

- mouse 很小。
- cat、dog 较小。
- horse、tiger 较大。

那么 tiger 可能与 horse 很近，但无法表达 tiger 与 cat 的其他联系。

增加 dimensions 后，可以同时表达多种差异，例如：

- Size。
- Domestic / wild。
- Animacy。
- Grammatical properties。
- Register。


### 3. 实际 embedding dimensions 通常没有名字

真实 embeddings 不会规定：

- Dimension 1 必须表示 size。
- Dimension 2 必须表示 gender。
- Dimension 3 必须表示 royalty。

这些 values 是 training 得到的，信息通常分布在多个 dimensions 中。

因此，不能简单把每个 coordinate 当作一个明确的 semantic label。

### 4. Embedding matrix

假设 vocabulary 有 V 个 tokens，每个 embedding 有 d 个 dimensions：

$$
E\in\mathbb{R}^{|V|\times d}
$$

其中：

- 每一行对应一个 vocabulary item。
- 每一列对应一个 embedding dimension。<span style="color: #66ccff;"> Dense embeddings 的 dimension 通常远小于 vocabulary size。</span> 
- Token ID 用来选择对应的 row。

例如：

| Quantity               |      Example |
| ---------------------- | -----------: |
| Vocabulary size        |       50,000 |
| Embedding dimension    |          300 |
| Embedding matrix shape | 50,000 × 300 |



# 第四章：Word2Vec & Skip-gram  （第 21–26 页）


**Skip-gram 是 Word2Vec 的一种训练方法：给它一个中心词，让它预测周围的词（context words），从而学到 word vectors。**

Training targets 直接从 text 中构造，因此不需要人工标注的 meaning labels, 可以理解为 self-supervised learning。

可以理解为：

One-hot input → Embedding lookup → Linear output scores → Softmax

![[Pasted image 20261104013359.png]]


Hidden representation 就是 center word 的  embedding，**没有 hidden nonlinear activation**。虽然前面的两步可以合并成一个 linear transformation，但保留两组 parameters，可以用低维 embeddings 表示 word–context 关系。

**与前面“没有 nonlinearity，多层会 collapse”并不矛盾, 它的目标是学习 representations，不是依靠 deep nonlinear boundaries 完成任务。：
- Collapse 讨论的是 function 表达形式。
- Skip-gram 通过两个较小的 embedding tables 对 word-context score matrix 做 factorization。
- 这种结构共享 parameters，并要求用低维 vectors 表示 word-context relations。





### 1. Sliding window 构造 training pairs

假设文本片段为：

> problems turning into banking crises as

Window radius 为 2。

当 center word 是 into 时，context words 是：

- problems。
- turning。
- banking。
- crises。

对应 training pairs：

| Center | Context |
|---|---|
| into | problems |
| into | turning |
| into | banking |
| into | crises |

Center 移动到 banking 后，又产生：

| Center | Context |
|---|---|
| banking | turning |
| banking | into |
| banking | crises |
| banking | as |

一个 center 会产生多个 prediction targets。




### 2. 每个 word 有两套 vectors

Skip-gram 通常为每个 word 学习：
- Center vector：word 作为 center 时使用。
- Context vector：word 作为 context 时使用。

记为：

$$
v_w=\text{center vector}
$$

$$
u_w=\text{context vector}
$$


“作为 input”与“作为 prediction target”是两个不同角色，model 使用两组 parameters 表示这两个角色。

#### Context vector 

令 pw 表示 model 给 context candidate w 的 probability。

令 tw 为 one-hot target。

则：

$$
\frac{\partial L}{\partial u_w}
=
(p_w-t_w)v_c
$$

它与上一节课的 classifier gradient 完全一致：

> Error × input。

对于 observed context o：

$$
\frac{\partial L}{\partial u_o}
=
(p_o-1)v_c
$$

对于其他 context candidates：

$$
\frac{\partial L}{\partial u_w}
=
p_wv_c
$$

#### Center vector

Center vector 影响所有 output scores，因此要把路径 contributions 相加：

$$
\frac{\partial L}{\partial v_c}
=
\sum_w(p_w-t_w)u_w
$$

由于 target 是 one-hot：

$$
\boxed{
\frac{\partial L}{\partial v_c}
=
\sum_wP(w\mid c)u_w-u_o
}
$$

第一项是 model 的 expected context vector：

$$
u_{\text{expected}}
=
\sum_wP(w\mid c)u_w
$$

第二项是 observed context vector：

$$
u_{\text{observed}}=u_o
$$

因此 gradient descent 的 update 是：

$$
v_c
\leftarrow
v_c+
\eta
\left(
u_{\text{observed}}-u_{\text{expected}}
\right)
$$

直觉是：

- 向 observed context 的方向调整。
- 减去 model 当前 expected context 的影响。

注意，center gradient 为零，不一定意味着所有 context gradients 都为零。


### 3. 用 dot product 计算 score

给定 center c，context candidate o 的 score：

$$
s_o=u_o^\top v_c
$$

然后对整个 vocabulary 应用 Softmax：

$$
P(o\mid c)
=
\frac{\exp(u_o^\top v_c)}
{\sum_{w\in V}\exp(u_w^\top v_c)}
$$

这和前面的 multiclass classification 一样：

- Input：center embedding。
- Classes：vocabulary 中所有可能的 context words。
- Output：各 context words 的 probabilities。



### 4. 一对 center–context 的 loss

对于观察到的 pair：

$$
(c,o)
$$

希望提高真实 context word 的 probability：

$$
L_{c,o}=-\log P(o\mid c)
$$

例如：

$$
L=-\log P(\text{banking}\mid\text{into})
$$


### 5. 整个 corpus 的 objective

含义是：

1. 遍历每个 center position。
2. 遍历 window 内的 context positions。
3. 每个 pair 贡献一个 NLL term。

假设 corpus 长度为 T，window radius 为 m：

$$
L(\theta)
=
-\frac{1}{T}
\sum_{t=1}^{T}
\sum_{\substack{-m\leq j\leq m\\j\neq0}}
\log P(w_{t+j}\mid w_t;\theta)
$$

这里仅包含没有越过 corpus boundaries 的有效 positions。

Theta 包含两套 word vectors。


### 为什么 similar-context words 会有 related vectors？

可以用 `cat` 和 `dog` 理解：

> I **feed** my **cat**.  
> I **feed** my **dog**.

它们经常出现在类似的 contexts 中，所以训练时，模型都需要做到：

- 输入 `cat` 的 vector → 给 `feed` 较高的概率。
- 输入 `dog` 的 vector → 也给 `feed` 较高的概率。

**因为要预测的周围词相似，它们的 vectors 会受到类似的更新，往往变得相似。** 这需要大量 contexts 的共同作用，不是共享一个词就一定相似。

“变成 optimization objective”就是：**通过最小化预测周围词的 loss，让 vectors 学到这些关系。**

最后，**full-Softmax** 指每次预测都对**整个 vocabulary** 中的词计算分数，再用 Softmax 得到概率。



# 第六章： embedding Evaluation（第 28–29 页）

### 1. Intrinsic evaluation

直接检查 embedding space 的性质, 是否捕捉了词语的语义或语法关系。

#### Nearest neighbors

检查 vector 最接近的词是否在语义或用法上相关，例如 `great` 附近有 `excellent`。

#### Word analogies
检查 vector 运算能否找出对应的类比关系
经典例子：

$$
v_{\text{king}}
-
v_{\text{man}}
+
v_{\text{woman}}
\approx
v_{\text{queen}}
$$

类似关系还包括：- walking → walked。- Japan → Tokyo。

这些规律是可能观察到的 empirical patterns，不保证对所有 words 都成立。

### 2. Extrinsic evaluation

把 embeddings 放进 downstream model，观察实际 task performance。
如果 embeddings 改善了 downstream accuracy，就说明它们对该 task 有帮助。

### 3. Embedding 三种使用方式

| Strategy              | Initialization | Training 时是否更新 embeddings？ |
| --------------------- | -------------- | -------------------------- |
| From scratch          | Random         | 更新                         |
| Frozen pretrained     | Pretrained     | 不更新                        |
| Fine-tuned pretrained | Pretrained     | 更新                         |

选择取决于 task data、computational budget 和 domain match。

Pretrained embeddings 可以把大 corpus 中学到的信息带入较小的 labeled dataset，但不保证所有情况下都优于 from-scratch training。

# 第七章：GloVe（第 30 页）


| 方法                           | 怎么学习                                                                |
| ---------------------------- | ------------------------------------------------------------------- |
| **Word2Vec（以 Skip-gram 为例）** | 用中心词预测周围词，通过减小 prediction loss 学习 vectors。                          |
| **GloVe**                    | 先统计词对在 context windows 中共同出现的次数（co-occurrence），再训练 vectors 拟合这些统计量。 |
PPT 给出的简化关系：

$$
u_i^\top v_j\approx\log X_{ij}
$$

其中：

$$
X_{ij}
=
\text{words i 和 j 的 co-occurrence count}
$$

实际 GloVe objective 还包含 bias terms 和 weighting function。

两者的共同点：

- 从 raw corpus 学习。
- 利用 word-context relations。
- 输出 dense word vectors。



# 第八章：Tokenization（第 31–35 页）


**Tokenization** 就是把文本拆分成模型处理的基本单位（tokens），例如 words、subwords 或 characters。决定哪些单位拥有独立 embeddings，以及哪些 occurrences 共享 parameters。

基本流程：
Text → Tokens → Token IDs → Embedding lookup

例如：the movie was great

可能变成：

| Token | ID |
|---|---:|
| the | 11 |
| movie | 298 |
| was | 34 |
| great | 567 |

Token ID 用来选择 embedding matrix 中对应的 row。


## Tokenization 分类

| Method          | Vocabulary | Sequence length | 主要问题                                     |
| --------------- | ---------- | --------------- | ---------------------------------------- |
| Word-level      | 通常较大       | 较短              | Rare words、OOV                           |
| Character-level | 通常较小       | 较长              | 单位过细                                     |
| Subword-level   | 折中         | 折中              | Segmentation 不一定符合 linguistic boundaries |

现代 language models 广泛使用 subword 或 byte-based tokenization。

### 2. Word-level tokenization

以完整 words 为单位。

优点：

- 一个 token 通常具有较明确的 lexical meaning。
- Sequence 相对短。

问题：

- Vocabulary 可能很大。
- Unseen words 可能变成 UNK。
- Different word forms 可能分散到不同 embedding rows。
- Punctuation、abbreviations、numbers 的处理需要规则。

例如：

- open。
- opened。
- opens。
- opening。

如果各自是独立 tokens，稀有 forms 得到的 training updates 就较少。

### 3. Character-level tokenization

以 characters 为单位。

优点：

- Vocabulary 小。
- 可以组合表示很多未见过的 words。

问题：

- Sequences 更长。
- 单个 character 提供的 meaning 有限。
- Model 需要从 character sequences 中重新学习 word-like patterns。

“没有 unknown words”依赖 character coverage；如果字符本身不在 vocabulary 中，仍需要额外机制。

### 4. Subword tokenization

在完整 words 和 characters 之间折中：

- Frequent words 可以保留为完整 token。
- Rare words 可以拆成已知 pieces。
- Different word forms 可以共享部分 tokens。

例如，某个 tokenizer 可能得到：

- open。
- open + ed。
- open + ing。

具体 segmentation 取决于学到的 tokenizer，并不保证一定符合 morphology。



# 第九章：Byte Pair Encoding (BPE)（第 36、39 页）

**Byte Pair Encoding（BPE）** 是一种 subword tokenization 方法，通过反复合并最频繁出现的相邻 token pair，逐步构建 vocabulary。

PPT 使用 character-based example：

1. 从基本 characters 开始。
2. 统计相邻 symbol pairs 的 frequency。
3. Merge 最 frequent 的 pair。
4. 在更新后的 segmentation 上重新统计。
5. 重复，直到达到 vocabulary size 等 stopping criterion。

## Training

先把 words 拆成 characters，按词频统计相邻 token pairs，每次合并频率最高的一对并重新统计，直到达到目标 vocabulary size 或 merge 次数。

| Word | Frequency |
|---|---:|
| hug | 10 |
| pug | 5 |
| pun | 12 |
| bun | 4 |
| hugs | 5 |

初始 segmentation：

- h u g。
- p u g。
- p u n。
- b u n。
- h u g s。


本例依次合并：`u + g → ug`（20 次）、`u + n → un`（16 次）、`h + ug → hug`（15 次）。

最终例子中新增的 tokens 是：

- ug。
- un。
- hug。

**Frequency 要根据 corpus occurrences 加权，不能只数出现于多少种 distinct words。**

而且每次 merge 后，要根据新的 symbol sequence 重新统计。

## BPE 如何 tokenize 新文本？（第 37 页）

**Tokenize 新文本：**先拆成 characters，再反复应用当前可用且 training rank 最早的 merge，不重新统计频率。

假设学到的 merge ranks 是：

1. u + g → ug。
2. u + n → un。
3. h + ug → hug。

现在遇到 training 中没有出现过的 word：unhug

过程: `unhug` → `u n h u g` → `u n h ug` → `un h ug` → **`un hug`**。


## Training frequency vs Encoding rank

Training 时：根据 corpus frequency 学习 merge rules。

统计当前相邻 token pair 在 corpus 中出现的次数，每轮合并**频率最高**的一对，并重新统计。


Encoding 时： 使用已经固定的 training 学到的 merge ranks，决定新文本如何 segmentation。优先执行**当前可用且 rank 最早**的合并，不重新统计频率。

另外，只有 vocabulary list 还不够，merge ranks 也是 tokenizer 的组成部分。
BPE 不能简单等同于从左到右选择 longest matching token。



# 第十一章：为什么 tokenizer 会影响 model capability？（第 38–40 页）

### 1. Context budget 按 tokens 计算

同一段 text，在不同 tokenizers 下可能产生不同数量的 tokens。

如果代码中的 indentation 被拆成很多 tokens：

- 会占用更多 context budget。
- 同样长度的 context 能容纳的代码更少。


### 2. Numbers 的 segmentation 会影响 arithmetic

例如，一个 number 可能被分成多个不等长 pieces。

Model 需要在 token representations 上学习 arithmetic patterns，segmentation 会影响任务难度。

但 arithmetic performance 也取决于 architecture、training data 和 objective，并不只由 tokenizer 决定。

### 3. Subwords 不保证永远没有 UNK

能否表示任意 input，取决于 base symbols 的 coverage，以及是否有 byte fallback 等机制。

Byte-level tokenization 可以通过 bytes 覆盖任意 UTF-8 text。

### 4. 不要把 Skip-gram 当成所有 LLM 的 training objective

这节课用 Skip-gram 展示如何从 context 学 embeddings。

现代 autoregressive LLM 通常使用 next-token prediction，并在更复杂的 network 中联合学习 token embeddings 与其他 parameters。

共同点是：

- 把 text 转成 tokens。
- 为 tokens 建立 learned representations。
- 利用 text 中构造出的 prediction targets 进行 learning。

并不是所有 LLM 都先运行 Word2Vec。





# 最需要掌握的七个问题

1. Distributional semantics 的核心假设是什么？
2. Skip-gram 如何从 sliding window 构造 training pairs？
3. 为什么每个 word 有 center vector 和 context vector？
4. Center gradient 中的 observed context 和 expected context 分别是什么？
5. Word-level、character-level、subword-level tokenization 有什么 trade-offs？
6. BPE training 的 frequency 与 encoding 的 merge rank 有什么区别？