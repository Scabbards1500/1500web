time: 2026/11/06

将 RNN Language Model 用于机器翻译，利用 Attention 解决信息瓶颈，再通过 Decoding 选出完整译文。


## 1. Machine Translation：机器翻译任务（P6–9）

Machine Translation 同时需要理解与生成：

- **Language analysis**：理解源语言。
- **Language generation**：组织目标语言的表达。


翻译不是逐词替换, 面临两个基本问题：
1. **Word order changes**：不同语言的词序可能不同。
2. **Length mismatch**：一个源语言词可能对应多个目标语言词。
所以，输入与输出的长度不必相同，也不能简单地对每个输入词独立分类。

###  训练数据：Parallel Corpus（P9）

Parallel corpus 是表达相同内容的双语句子对。

| Source：法语        | Target：英语             |
| ---------------- | --------------------- |
| le chat dort     | the cat is sleeping   |
| merci beaucoup   | thank you very much   |
| où est la gare ? | where is the station? |

常见来源包括议会文件、联合国文件、电影字幕和翻译书籍。

### LM vs Machine Translation
翻译是 Conditional Language Modeling（P11）

普通语言模型建模：文本自身提供 next-token 标签。：

$$
P(Y)=\prod_{t=1}^{T}P(y_t\mid y_{<t})
$$

Machine Translation 建模：源句是输入，对应译文提供目标 token 标签：

$$
P(Y\mid X)
=
\prod_{t=1}^{T}P(y_t\mid y_{<t},X)
$$

区别在于：预测下一个目标词时，除了看目标语言前文，还要考虑源句 $X$。

**两者都可以通过 next-token prediction 训练，但翻译还需要学习源句与译文的对应关系。**


# 2. Sequence-to-Sequence：Encoder–Decoder（P10–15）

Encoder 读完整句，将信息交给 Decoder；Decoder 再逐词写出译文。

基础 Seq2Seq 使用两个 RNN：
![[Pasted image 20261106225253.png]]

| 模块      | 输入                | 作用                    |
| ------- | ----------------- | --------------------- |
| Encoder | 源句 `le chat dort` | 编码源句                  |
| Decoder | 起始符及目标语言前文        | 逐步生成 `the cat sleeps` |

Encoder 依次读取源词，得到：

$$
h_1,h_2,\ldots,h_N
$$

基础版本使用最后一个状态 $h_N$ 初始化 Decoder。

Decoder 根据自身状态预测下一个词：

$$
P(y_t\mid y_{<t},X)
=
\operatorname{softmax}(Us_t+b)
$$

## 训练：Teacher Forcing（P13）

训练时，Decoder 接收真实的上一个目标词。

| Decoder 输入 | 预测目标   |
| ---------- | ------ |
| `<s>`      | the    |
| the        | cat    |
| cat        | sleeps |
| sleeps     | `</s>` |

损失是目标序列的平均 negative log-likelihood：

$$
\mathcal L
=
-\frac1T\sum_{t=1}^{T}
\log P(y_t^*\mid y_{<t}^*,X)
$$

其中，星号表示真实译文。

每个目标位置都有损失，梯度可以从 Decoder 传回 Encoder，整个模型端到端训练。

## 推理：输入自己生成的词（P14）

测试时没有真实译文，因此流程是：

1. 输入 `<s>`，预测第一个词。
2. 将生成的词作为下一个时间步的输入。
3. 重复生成。
4. 生成 `</s>` 后停止。

训练与推理的区别：

- **Training**：读取真实的上一个词。
- **Inference**：读取自己生成的上一个词。

如果模型前面生成错了，后面就会基于错误前文继续生成。

## Information Bottleneck：固定向量的瓶颈（P15）

基础 Seq2Seq 把源句信息都压进最后一个状态 $h_N$。

例如：

- 3 个词的句子 → 一个 512 维向量。
- 50 个词的句子 → 仍然是一个 512 维向量。

句子越长，越难完整保存所有细节。
**LSTM 可以缓解长期记忆问题，但固定大小向量的信息瓶颈仍然存在。**

# 3. Attention：生成时回看源句（P16–23）
![[Pasted image 20261106225456.png]]


加入 Attention 后，模型保留：

$$
h_1,h_2,\ldots,h_N
$$

Decoder 每生成一个目标词，都可以重新读取相关源状态。

例如：

> 源句是 `le chat dort`，已经生成 `the`，准备生成 `cat`。

这一步应该重点参考 `chat` 对应的源状态。

注意：$h_i$ 是 Encoder 在位置 $i$ 的上下文表示，不只是该词的原始 embedding。

## 计算过程
> Score → Normalize → Aggregate → 与 Decoder 状态拼接 → 预测目标词。

### Step 1：Score，计算相关性（P18）

将当前 Decoder 状态 $s_t$ 与每个源状态 $h_i$ 比较：

$$
e_{t,i}=s_t^\top h_i
$$

它回答的问题是：

> 当前的生成需求，与源句中哪个位置更相关？

### Step 2：Normalize，转换为权重（P18）
算出 **“每个位置关注多少”**

对所有相关性分数使用 softmax：

$$
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}
{\sum_{j=1}^{N}\exp(e_{t,j})}
$$

因此：

$$
\sum_{i=1}^{N}\alpha_{t,i}=1
$$

PPT 中生成 `cat` 时的示例：

| 源词 | Attention weight |
|---|---:|
| le | 0.11 |
| chat | 0.82 |
| dort | 0.07 |

模型主要关注 `chat`。

### Step 3：Aggregate，加权读取信息（P18）

**按照刚才算出的权重，把源句的信息汇总成一个context vector向量$a_t$ ，交给 Decoder 使用。**

$$
a_t=\sum_{i=1}^{N}\alpha_{t,i}h_i
$$


上述例子对应：

$$
a_t
=
0.11h_{\text{le}}
+
0.82h_{\text{chat}}
+
0.07h_{\text{dort}}
$$
它主要包含 `chat` 对应的信息，同时混合其他位置的信息。

### Step4:  使用 Context Vector 预测目标词（P18）

将 context vector 与 Decoder 状态拼接：

$$
P(y_t\mid y_{<t},X)
=
\operatorname{softmax}\left(U[a_t;s_t]+b\right)
$$

其中：

- $s_t$：当前生成状态。
- $a_t$：当前从源句读取的相关信息。


生成不同目标词时，Decoder 状态不同，attention 分布也会变化。
例如：
- 生成 `the`：主要关注 `le`。
- 生成 `cat`：主要关注 `chat`。
- 生成 `sleeps`：主要关注 `dort`。

将各步的 attention 分布排起来，就得到 attention matrix：
- 横轴：源句位置。
- 纵轴：目标句位置。
- 每一行：生成一个目标词时的权重分布。

![[Pasted image 20261120193320.png]]
attention 权重表示信息读取的分配，不能直接视为严格的因果解释。

PPT 的真实翻译示例中，attention 大致沿对角线分布，说明两种语言的对应词通常顺序接近。

部分位置偏离对角线，则反映词序变化。

**训练时没有提供词对齐标签，这些对应关系通过翻译损失学得。**



## 为什么 Attention 有效？（P21–22）

#### 缓解固定向量瓶颈

Decoder 可以访问全部源状态。
源句变长时，可读取的状态数量增加，不必把全部细节都压进最后一个向量。

#### 缩短梯度路径

目标位置的损失可以通过 attention 直接影响相关源状态，不必只沿很长的 RNN 链传回去。

#### 提供可观察的对齐模式
Attention matrix 能帮助我们观察模型在不同生成步骤关注的源位置。
![[Pasted image 20261120193620.png]]

P22 的实验显示：
- 无 attention 的模型在长句上质量明显下降。
- 带 attention 的模型在该实验中更稳定。


## Query 与 Value 的含义（P23）
![[Pasted image 20261106225643.png]]

| 名称 | 对应内容 | 直觉 |
|---|---|---|
| Query | Decoder 状态 $s_t$ | 我现在需要什么信息？ |
| Values | Encoder 状态 $h_1,\ldots,h_N$ | 有哪些内容可以读取？ |

在这一版 attention 中，$h_i$ 同时承担两个角色：

1. 与 Query 比较，决定匹配程度。
2. 参与加权平均，提供读取内容。

后续课程会进一步把这两个角色拆成 Key 与 Value。

# 4. Attention 的不同形式（P24–31）

不同 attention 通常共享以下流程：

$$
e_i=\operatorname{score}(s,h_i)
$$

$$
\alpha_i=\operatorname{softmax}(e)_i
$$

$$
a=\sum_i\alpha_i h_i
$$

主要变化是：如何计算Score Function $\operatorname{score}(s,h_i)$。

| 方法             | Score function             | 参数               | 核心特点      |
| -------------- | -------------------------- | ---------------- | --------- |
| Dot-product    | $s^\top h_i$               | none             | 直接匹配，计算简单 |
| Multiplicative | $s^\top Wh_i$              | d2 d1            | 学习完整的线性匹配 |
| Reduced-rank   | $(As)^\top(Bh_i)$          | r (d1 + d2)      | 低维投影后匹配   |
| Additive       | $w^\top\tanh(W_1h_i+W_2s)$ | d3 (d1 + d2 + 1) | 使用非线性网络匹配 |

它们最终都将分数交给 softmax，再对源状态进行加权平均。

### 5.2 Dot-Product Attention（P26）

$$
e_i=s^\top h_i
$$

优点：

- Score 函数不增加可学习参数。
- 计算简单，容易通过矩阵运算并行处理。

限制：

- $s$ 与 $h_i$ 必须维度相同。
- 直接在相同坐标之间进行匹配。

这里的“无参数”仅指 score 函数。Encoder 和 Decoder 本身仍然需要训练。

### 5.3 Multiplicative / Bilinear Attention（P27）

$$
e_i=s^\top Wh_i
$$

其中：

$$
W\in\mathbb R^{d_2\times d_1}
$$

可学习矩阵 $W$ 可以决定 Decoder 的哪些特征应当与 Encoder 的哪些特征匹配。

优点：

- 支持 $s$ 和 $h_i$ 维度不同。
- 匹配方式可以通过训练学习。

代价是参数更多。例如，$512\times512$ 的矩阵约有 26 万个参数。

当维度相同且 $W=I$ 时，就退化为 dot-product attention。

### 5.4 Reduced-Rank Multiplicative Attention（P28）

将完整矩阵分解为：

$$
W=A^\top B
$$

于是：

$$
e_i
=
s^\top A^\top Bh_i
=
(As)^\top(Bh_i)
$$

其中：

$$
A\in\mathbb R^{r\times d_2},
\qquad
B\in\mathbb R^{r\times d_1}
$$

当 $r$ 较小时，参数量从：

$$
d_1d_2
$$

减少到：

$$
r(d_1+d_2)
$$

直觉是：

> 将两边映射到较小的共同空间，再计算点积。

这与 Transformer 中 Query 和 Key 先投影、再点积的形式相联系。不过完整 Transformer attention 还有缩放、Value 投影和多头等设计。

### 5.5 Additive Attention（P29）

$$
e_i
=
w^\top\tanh(W_1h_i+W_2s)
$$

它使用一个小型前馈神经网络，为源状态与 Decoder 状态的组合打分。

特点：

- 包含非线性，匹配方式更灵活。
- 相比简单点积，需要更多计算。

“Additive” 指投影后的状态相加，不是把 attention 权重直接相加。




# 5. 从 RNN 到 LSTM，再到 Attention（P31）
![[Pasted image 20261106225857.png]]

| 机制 | 信息与梯度如何传递 |
|---|---|
| RNN | 沿时间步逐步传递，长链容易出现梯度问题 |
| LSTM | 利用 cell state 和门控改善长期传递 |
| Attention | 为目标位置与源状态提供直接连接 |
LSTM 改善沿时间链传递信息的能力；Attention 进一步提供跨位置的直接读取路径。

# 6. Decoding：如何生成完整译文？（P32–36）

模型提供：

$$
P(y_t\mid y_{<t},X)
$$

Decoding 决定如何根据这些概率选择输出序列。

### Greedy Decoding（P33）

每一步选择当前概率最大的词：

$$
\hat y_t
=
\arg\max_yP(y\mid\hat y_{<t},X)
$$

优点是简单、速度快。

问题是：

> 当前最好的下一个词，不一定属于概率最高的完整句子。

例如，选了 `he hit a` 后，就不能回到 `he hit me` 这条路径。

###  目标：寻找生成概率最高的完整句子（P34）

这里的“概率最高”，指的是：**给定源句 \(X\)，模型生成整句译文 \(Y\) 的概率 P(Y| X)**。它不等于“翻译正确的概率”。

$$
\hat Y=\arg\max_Y P(Y\mid X)
$$

通常使用 log probability **累加**：

$$
\operatorname{score}(y_{1:t})
=
\sum_{i=1}^{t}
\log P(y_i\mid y_{<i},X)
$$

分数越高、越接近 0，表示概率越大。
但长度为 $T$ 的序列约有 $|V|^T$ 种候选，无法穷举。


<span style="color: #33cc33;">所以为什么不能仅用greedy decoding? </span>
比如已经生成 `he hit`，假设后续概率是：

|候选路径|下一个词的概率|随后结束的概率|后续路径概率|
|---|--:|--:|--:|
|`a → <EOS>`|0.6|0.1|\(0.6\times0.1=0.06\)|
|`me → <EOS>`|0.4|0.9|\(0.4\times0.9=0.36\)|

两条路径的前缀 `he hit` 相同，所以比较时可以省略它的概率。
**Greedy 会选 `a`，因为当前 \(0.6>0.4\)；但这两条完整路径中，`he hit me <EOS>` 的概率更高。**


### Beam Search（P35）
**每一步保留多条候选路径**，例如同时保留 `he hit a` 和 `he hit me`，再根据后续的累计分数筛选。

每一步执行：
1. 扩展当前保留的所有候选。
2. 计算扩展后序列的累计分数。
3. 在全部扩展中保留最好的 $k$ 个。

关键是：
**所有分支一起选 top-$k$，不是每个分支各保留 $k$ 个。**
它每一步大约需要考虑 $k|V|$ 个扩展分数，比穷举便宜很多。


Beam search 搜索得比 greedy 更广，但仍可能提前剪掉最终最优路径，所以不保证全局最优。

### Length Normalization（P39）

最终选择需要考虑长度归一化。

因为每增加一个 token，累计 log probability 通常都会降低，直接比较不同长度的句子可能偏向短输出。

一种简单的处理是使用平均 log probability：

$$
\operatorname{score}_{\mathrm{avg}}(Y)
=
\frac1{|Y|}
\sum_{t=1}^{|Y|}
\log P(y_t\mid y_{<t},X)
$$

也可以使用其他 length penalty，具体形式取决于实现。

# 发展与局限（P37–39）

| 时间 | 发展 |
|---|---|
| 2014 | Seq2Seq |
| 2015 | Attention |
| 2016 | Google Translate 转向神经机器翻译 |
| 2017 | Transformer |

机器翻译推动了 Seq2Seq 与 Attention 的发展，这些思想随后影响了更广泛的语言模型设计。


PPT 展示的结果中：

- 高资源语言，例如西班牙语、法语，表现较好。
- 部分低资源语言，例如 Nuer、Dinka，表现明显较差。

平行语料的数量与质量是重要因素。更强的架构本身不能消除数据不足的问题。


# 总结

| 核心连接 | 需要掌握的内容 |
|---|---|
| LM → MT | 翻译是在源句条件下预测下一个目标 token |
| RNN → Seq2Seq | Encoder 读源句，Decoder 写译文 |
| Seq2Seq → Attention | 从固定向量传递信息，发展到每一步按需读取源状态 |
| Attention variants | 主要改变如何计算相关性分数 |
| 概率模型 → Decoding | 模型给出概率，搜索算法决定实际输出 |

重点复习：

- **P11–15**：Conditional LM、Seq2Seq、Teacher Forcing 与信息瓶颈。
- **P17–18**：Attention 的 Score、Normalize、Aggregate。
- **P25–30**：不同 attention score 的区别。
- **P33–36**：Greedy 与 Beam Search 的区别。