abstract: 拿gmail 补全作的一个case study。 感觉此书知识结构真挺好的，非常系统，大脑褶皱看平了
time: 2026/07/08
## Introduction

说白了这一章就是教我们LLM补全的设计架构

Gmail's Smart Compose feature [1] assists users by suggesting the next few words as they write an email.

![](https://i-blog.csdnimg.cn/direct/9cafdebf1ee543879eefd4b8ac50820d.png)

## Clarifying Requirements

seems 我们作为被面试人需要给面试官提一串需求来去定位我们的设计

function- suggest range? personalized?

dataset - size? range? language?

output -biased?

system - user number? real-time?

## **Specifying the system's input and output**

![](https://i-blog.csdnimg.cn/direct/57ca6fc651aa4b34977b4da8f4bba341.png)

## **Choosing a suitable ML approach**

rnn vs transformer

![](https://i-blog.csdnimg.cn/direct/d66dd63590144e2793eee4ca89b841f4.png)

Various techniques are introduced to reduce the complexity of attention.

Group Attention [6] and FlashAttention [7]

## Data Preparation

data source: general data/email data

#### Text cleaning

Remove non-English text/ confidential information(脱敏）/ irrelevant characters or symbols/ duplicated data

#### Text normalization

时间，日期，电话等归一化

#### Text tokenization

![](https://i-blog.csdnimg.cn/direct/5099667cbc4d409da0e1940b95263ddb.png)

**Tokenization level**

目前最好用的是subword-level

![](https://i-blog.csdnimg.cn/direct/89315b9444574834945bc62dc703826c.png)![](https://i-blog.csdnimg.cn/direct/67801322c44a48e78e4dd67bc36b145c.png)![](https://i-blog.csdnimg.cn/direct/ed81a4cfde3f4c7f8b165dc35e812af5.png)

![](https://i-blog.csdnimg.cn/direct/1f11dcf1a2c34146910e5c901dc5c2f5.png)

**Tokenizer choose**

Byte-Pair Encoding (BPE)

SentencePiece .

Tiktoken

#### token indexing

converting textual tokens into integer numbers

![](https://i-blog.csdnimg.cn/direct/bfb550e3fa634ebcb1b3a92f2cde7d99.png)

![](https://i-blog.csdnimg.cn/direct/7d69cc0e53124596b00f1aec59b737fc.png)

## Model Development

#### Transformer Architecture

这个我们这里也不展开了，平时算法笔记也做的比较详细了

**Encoder-only** -- understanding the overallmeaning of a text.

**Decoder-only** -- processes the input sequence and generates a newsequence iteratively

**Encoder-decoder** -- encoder component processes the input sequence and a decoder uses that processed information to generate the output sequence （ e.g 翻译)

![](https://i-blog.csdnimg.cn/direct/096adbb5969a452da491c2bdc64f120b.png)

在这个续写任务中，我们当然就用decoder啦

A decoder-only Transformer consists of the following components:

#### Text embedding

把每个 token ID 映射成**高维向量**，作为神经网络真正处理的输入。

![](https://i-blog.csdnimg.cn/direct/4d69ad480c754b7a87847a2e7a51cc16.png)

![](https://i-blog.csdnimg.cn/direct/cb7bb5fa4cd0447bb7a8ca3f4859a801.png)

#### Positional encoding

positional encoding provides the Transformer with position information for each token in the input sequence 一般情况下选 fixed 准没错

![](https://i-blog.csdnimg.cn/direct/7f821f82161c4be18c0d0f42bdc875d6.png)

**Fixed positional encoding**

![](https://i-blog.csdnimg.cn/direct/01b107d4aac747f8b68cea3e01fe560a.png)![](https://i-blog.csdnimg.cn/direct/927e713d3ade478e8b90892a119da535.png)

pros: Efficiency, Support for long sequences

cons: Predefined limits(就是字数不能超), Suboptimal performance

**Learned positional encoding**

![](https://i-blog.csdnimg.cn/direct/b16b380319bf40228d123bd3e1a1ef50.png)

pros：Optimal performance

cons: Inefficiency， Lack of generalization

#### Transformer

![](https://i-blog.csdnimg.cn/direct/c93e7ef3b9c047e3a414cd725cd48afb.png)

Transformer architecture consists of a stack of blocks. Each block contains the

- **Multi-head attention:** This layer updates each embedding by using the attention mechanism. The attention mechanism captures the relationships in the sequence by allowing each embedding to attend to its preceding embeddings. Due to the nature of its mechanism, multi-head attention is commonly known as self-attention, a term we'll use throughout the rest of this book.
- **Feed forward:** This layer applies two linear transformations, with a ReLU activation in between, to each embedding in the sequence independently.

#### Prediction head

the final component in a decoder-only Transformer,  translates the Transformer's output into probabilities for every token in the vocabulary.

These probabilities are used to choose the most likely next token.

![](https://i-blog.csdnimg.cn/direct/5a788a47cf6447a5ae94547151a745b4.png)

## Training

**two-stage training**

![](https://i-blog.csdnimg.cn/direct/a8d2917406494ec08ad6535c9775fb56.png)

why two stage?

**Adaptability, Improved generalization, Fast finetuning, Handling data scarcity, Mitigating overfitting, Resource optimization**

### pretraining

一般工程中我们都拿这个阶段已经训练好的模型

The purpose of pretraining is to develop a model capable of understanding natural language, including syntax, common knowledge, and language structures.

**ML objective and loss function**

In the case of text generation, the most commonly used ML objective is "**next-token prediction**."

![](https://i-blog.csdnimg.cn/direct/4c4f1b59d2d74cda8433738978e00eee.png)

btw 这里注意区分一下masked prediction

| 训练方式 | 典型模型 | 输入能看什么 | 训练目标 | 更适合 |
| --- | --- | --- | --- | --- |
| **Next-token prediction** | GPT / LLaMA / Qwen 这类生成式 LLM | 只能看前文 | 预测下一个 token | 文本生成、对话、续写 |
| **Masked prediction / MLM** | BERT | 可以同时看左右文 | 预测被 `[MASK]` 掉的词 | 文本理解、分类、检索、匹配 |

**Cross-entropy loss**

这个也不多解释了，就是奖励应该出现的那个token

![](https://i-blog.csdnimg.cn/direct/f6838d9873a14f989deca44275b339ad.png)

In practice, the model processes all token lengths within a sequence in parallel. This allows it to compute the **loss for each token position simultaneously**. Parallelizing this step speeds up training by handling multiple tokens at once, instead of sequentially.

![](https://i-blog.csdnimg.cn/direct/f52d8c58c4be4dd8bba2ff5ec7f76e91.png)

Transformer 会同时为每个位置输出一个 next-token 概率分布， 然后每个位置都和正确答案算一次 cross-entropy loss，最后加起来或取平均

causal mask = 防止模型在预测下一个 token 时偷看后面的 token。

* BOS(beginning of sequence) 和 EOS(end of sequence) 是 序列起始符/结束符号

### finetuning

这里就用到专项(task- specified)数据集了（邮件)

**ML objective and loss function**

同 pretrain， next-token prediction +  cross-entropy loss

缺失信息处理

这个比较工程，比如你打了一个 dear， 模型怎么知道要写谁的名字呢，于是他去更深的信息去翻

![](https://i-blog.csdnimg.cn/direct/1c090d4ee4db4270b6a0d15e4fc16415.png)

### Combining various inputs

![](https://i-blog.csdnimg.cn/direct/f1122a21daf3476f85d15ae37aebb7cf.png)

总之放到工程就可以理解为把各种信息拼到一块吧，比如一个json 可以包含各个类别的信息

![](https://i-blog.csdnimg.cn/direct/bdb206aaffc84c65afa876b6b36c8267.png)

## Sampling

process of using a trained generative model to generate new data

The process continues untill the model predicts the \<EOS\> token

![](https://i-blog.csdnimg.cn/direct/aadf7f05686a4dd89aeec118e178120b.png)

一般情况下选 deterministic，**Consistency, Better handling of common phrases, Reduced risk of inappropriate suggestions**

### deterministic

generate text in a deterministic way, that is, without randomness or variability in the output.

At each step of token generation, the model **selects the token with the highest probability** from the predicted distribution.

pros：Consistency, Predictable outputs

cons: Lack of diversity, Repetitive text

![](https://i-blog.csdnimg.cn/direct/346123e21e304189b6e9bbe92c4e1388.png)

我草原来这种重复的罪魁祸首是greedy search哇

#### Greedy search

always selects the token with the highest probability as the next token

![](https://i-blog.csdnimg.cn/direct/dc86d73f4c864850a57de30a0d9704a9.png)

#### **Beam search**

保留多条高分路径

track multiple potential sequences of tokens simultaneously. At each step, the model calculates the probabilities for the next possible tokens for each sequence and selects the "top-k" most probable sequences.

![](https://i-blog.csdnimg.cn/direct/3f1ddad4c17f466082ffd04129a7c65f.png)

1. **Initialization:** Start with the user's partial email as the input to the trained model. The model predicts the probability distribution for the next token. Beam search selects the **top three tokens with the highest probabilities.**
2. **Expansion:** For each top three sequence, pass it to the model and obtain the probabilities of the next token.
3. **Pruning:** Select the **top three sequences based on their cumulative probabilities**.

Once the beam search algorithm has stopped, we select the sequence with the **highest cumulative probability** as the output.

![](https://i-blog.csdnimg.cn/direct/3ac7891b86cb450e87761f0c169ce2c3.png)

cons：Limited diversity， Struggle with long sequences

### Stochastic 随机的

Stochastic sampling methods introduce randomness into the generation process.

For example, at each step of token generation, the model samples from the predicted distribution based on the probabilities assigned to each token.

![](https://i-blog.csdnimg.cn/direct/9a53b2ff30404aa7a78e1f5e3f63f0aa.png)

**pros** Diversity  Novelty

**cons** Inconsistency  Unexpected outputs

注意: stochastic sampling 不是不会重复，而是相比 greedy / beam 这种 deterministic decoding，它更不容易被固定的最高概率路径锁死。

在邮件补全的场景下，补全的内容通常比较短，也不期望多样性，没有必要上beam search

### Deterministic vs Stochastic

| Characteristic | Deterministic methods | Stochastic methods |
| --- | --- | --- |
| **Approach** | Follow a predictable process to generate output | Generate output based on probability distribution |
| **Efficiency** | Typically less efficient due to tracking multiple paths | More efficient since randomness allows for quicker selections |
| **Quality** | Coherent and predictable | Diverse and creative |
| **Risk** | Usually lead to repetitive output for longer sequences | Might produce inappropriate output due to their creativeness |
| **Use case** | Suitable for tasks requiring consistency, such as language translation | Suitable for tasks requiring creativity, such as open-ended text generation |
| **Methods** | Greedy search, beam search | Multinomial, top-k, top-p |

## Evaluation

### Offline evaluation metrics

ensure the model's performance is acceptable before deploying it to production

Offline evaluation uses pre-collected and historical data to evaluate a model's performance

除了  Perplexity，Ecact match 还有 BLEU score and ROUGE-N， 后边讲

#### Perplexity 困惑度

This metric measures how **accurately the model predicts the exact sequence of tokens present in text data**.

越低越好

![](https://i-blog.csdnimg.cn/direct/513af977e246402f80f1d89e5bddce33.png)

![](https://i-blog.csdnimg.cn/direct/0a0994497ad544dd9cd78ac0a86b1897.png)

#### ExactMatch@N

percentage of generated phrases that are exactly N words long and that match the first N words of the ground-truth text.

![](https://i-blog.csdnimg.cn/direct/f1e67832d66147e586c9b9fc547e5ff0.png)

### Online evaluation metrics

how a model performs in real time as users interact with the system

defined based on specific requirements and needs

#### User engagement metrics

Acceptance rate

Usage rate

#### Effectiveness metrics

Average completion time

#### Latency metrics

System response time

#### Quality metrics

Feedback rate

Human evaluation

## ML System Design

这个好啊，有一说一感觉平时工作都是在前面的部分

针对Smart Compose，整个流程主要是

1. **Monitoring:** The triggering service monitors the user's activity as they type.
2. **Triggerring:** The service triggers the phrase generator once it identifies specific patterns.
3. **Beam search:** The phrase generator employs beam search to get top-k potential completions from the trained model.
4. **Filtering:** The phrase generator interacts with the filtering component to remove long suggestions and those with low confidence scores.
5. **Post-processing:** The completion with the highest score is picked and passed to the post-processing service. The service replaces gender-specific pronouns and adjusts sensitive terms.
6. **Display suggestion:** The suggestion is displayed to the user for their consideration.

![](https://i-blog.csdnimg.cn/direct/9fcdaa413885455487943394a4f82c21.png)

### Triggering service

triggering service activates the Smart Compose feature by monitoring user activity such as keystrokes.

判断需要的时候再触发

### Phrase generator

这个就是我们刚刚讨论了老半天的算法部分

It generates the most likely completion based on the partial text the user has already typed.

![](https://i-blog.csdnimg.cn/direct/cf23cc599cb240c689f919c9638e7e19.png)

Removing long suggestions

![](https://i-blog.csdnimg.cn/direct/af675dd040fb4d71bccb811d44cd1acc.png)

Removing low-confidence suggestions

![](https://i-blog.csdnimg.cn/direct/83ddf8c08f984617b569fa54df4256d0.png)

好简单粗暴....

### Post-processing service

addresses potential biases before suggestions are presented to the user

**Pronoun replacement**

**Gender-neutral word replacement**

**Lexical ( 词汇) analysis for sensitive terms**

**NSFW (Not Safe For Work) content filtering**

## Other Question to prepare

下回再说，现在主要是给书过一遍

- Supporting Smart Compose in multiple languages [28].
- Personalizing suggestions [28].
- Incorporating additional context for better predictions [28].
- Understanding how different tokenization algorithms work, such as BPE [11], SentencePiece [12], and WordPiece [29].
- Understanding different ML objectives such as masked language modeling (MLM) and its variations [18].
- The multi-token prediction objective and its pros and cons [30].
- Balancing quality and inference latency [28].
