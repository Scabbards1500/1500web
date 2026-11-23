这节课围绕三个问题展开：**去哪里找知识、什么时候查、查到后该信多少。** 前半节重点是三种 Neural Retrieval 架构，后半节讨论模型怎样学会调用检索，以及为什么**找到了正确资料，模型还是可能答错**。

## 1. 为什么模型参数里的知识不够用？（第 3–5 页）

上一讲的 **LoRA** 解决的是「怎样用更少的可训练参数来适配模型」。这节课换了一个问题：**知识经常更新时，能不能直接更新资料，让模型在回答时查阅？**

比如，课程作业截止日期从 12 月 4 日改到 12 月 6 日。如果依赖 **parametric memory**，答案来自模型权重中学到的信息。如果使用 RAG，可以更新课程公告和对应的检索索引，模型回答时就有机会读到新日期。

课件提出三个目标：

- **Knowledge editing**：更新外部文档，减少为了修改某个事实而调整模型权重的需要。
- **Attribution**：可以追查答案引用了哪篇文档，还可以进一步检查文档是否真的支持答案。
- **Efficient scaling**：让模型访问更大的知识库，每次只把少量相关内容交给生成模型。

## 2. RAG 是怎么工作的？（第 6–9 页）

给定用户问题 \(x\)，一个基本的 RAG pipeline 有两部分：

1. **Retriever** 从语料库中找出最相关的 \(k\) 个 passages。
2. **Generator** 读取问题和这些 passages，再生成答案 \(y\)。

用这份课件里的例子，用户问「PA3 什么时候交？」Retriever 找到课件第 2 页的课程通知，Generator 从中读出 **December 4th**，再组织成答案。手动把这段通知贴进 prompt 也能提供同样的证据。RAG 把**根据当前问题自动选择上下文**这一步交给检索系统。

课件还给了一种概率视角，把文档 \(d\) 当作 **latent variable**：

\[ p(y\mid x)=\sum_{d\in\mathcal Z}p(d\mid x)\,p(y\mid x,d) \]

- \(p(d\mid x)\) 是 Retriever 对「这篇文档值得参考吗？」的判断。
- \(p(y\mid x,d)\) 是 Generator 在读到该文档后生成答案 \(y\) 的概率。

可以把它理解成：**不同文档能支持不同答案，相关性高的文档获得更大权重。** 公式描述的是对文档做边缘化的建模方式。工程上也常用更直接的做法，一次取出 top-\(k\) 文档，拼到同一个 prompt 里生成。

## 3. Neural Retrieval 怎么给文档打分？（第 10–29 页）

传统 **sparse retrieval**，例如 BM25，主要依靠词项匹配。用户搜「作业什么时候交」，文档写的是「submission deadline」，就可能遇到 **vocabulary mismatch**。**Dense retrieval** 把文本映射成向量，让语义相近的表达有机会匹配上。

三种架构的区别，在于 **query 和 document 什么时候发生交互，以及交互有多细**。

|方法|Query 和 document 怎么交互|能提前算什么|主要代价|
|---|---|---|---|
|**Bi-Encoder**|各自编码成一个向量，最后算 dot product|每篇文档的单个向量|一个向量要概括整篇 passage，细节可能被压缩|
|**Cross-Encoder**|把 query 和 document 拼在一起，让 token 在编码时通过 self-attention 交互|最终相关性分数依赖当前 query|每个候选都要和 query 一起跑一次 encoder|
|**ColBERT**|各自编码并保留 token 向量，最后用 MaxSim 匹配|每篇文档的 token 向量|索引更大，匹配计算也更贵|

### 3.1 Bi-Encoder、DPR 和 ANN（第 12–20 页）

Bi-Encoder 分别编码 query 和 document，然后计算：

\[ s(q,d)=f(q)^\top g(d) \]

因为 \(g(d)\) 只依赖文档，整个语料库的 embeddings 可以**离线算好**。线上只需要编码 query，然后找与它最相近的文档向量。

**DPR（Dense Passage Retrieval）** 用 **contrastive learning** 训练这样的检索器。假设问题是「PA3 什么时候交」，写有截止日期的 passage 是 **positive**。一段同样提到 PA3、但只讲提交格式的文字，可以当 **hard negative**。训练时把正负候选的分数一起送进 softmax，让正确 passage 的得分相对更高。同一 batch 里其他问题的正例也可以拿来当 **in-batch negatives**。

如果有 2,100 万篇 passages，逐一计算 dot product 会很贵。课件用 **k-means** 讲了一种 **approximate nearest neighbor（ANN）** 的思路：先把文档向量分成若干组，查询时先找最近的组中心，再只扫描该组里的向量。在课件假设各组大致均匀、只搜最近一组的例子里，需要算约 **9,200 个分数**。对应的代价是，真正最接近的文档有可能落在相邻组，因而被漏掉。**FAISS** 是用来做这类向量搜索的库。

### 3.2 Cross-Encoder（第 21–22 页）

Cross-Encoder 把 `[query; document]` 一起送进 Transformer，输出一个相关性分数。这样，query 里的 `PA3` 和 `due` 可以在编码时直接与文档里的词交互，模型更容易判断某个日期究竟是 **PA3 的截止日期**，还是 **PA2 的截止日期**。

常见用法是 **retrieve → rerank**。先用 Bi-Encoder 从大语料库筛出一小批候选，再用 Cross-Encoder 精排。这样可以把昂贵的联合编码用在最值得细看的文档上。

### 3.3 ColBERT 和 Late Interaction（第 23–29 页）

ColBERT 也分别编码 query 和 document，但保留**每个 token 的 contextual embedding**。到了打分阶段，每个 query token 在文档里单独找最相似的 token，然后把这些最高分加起来：

\[ \operatorname{MaxSim}(q,d)=\sum_i\max_j\langle q_i,d_j\rangle \]

假设 query 是 `when PA3 due`。`when` 可能匹配日期表达，`PA3` 匹配作业编号，`due` 匹配 `deadline`。每个 query token 都能从文档的不同位置取得匹配分。**Late interaction** 指的就是这种「先独立编码、再逐 token 匹配」。

大规模检索时，ColBERT 可以先给每个 query token 找最近的文档 token，用这些 token 所属的文档作为候选，再对候选算完整的 MaxSim。因为索引里存的是**每个文档的许多向量**，压缩就很重要，课件因此提到 **ColBERTv2** 和 **quantization**。

## 4. 什么时候检索？（第 30–35 页）

|策略|怎么做|适用情况或代价|
|---|---|---|
|**Retrieve once**|回答前先查一次|适合问题明确、答案较短的场景|
|**Retrieve periodically**|每生成固定数量的 tokens 就查一次|长文可以持续补充资料，但固定频率可能产生多余调用|
|**Retrieve adaptively**|模型根据当前生成状态决定什么时候调用工具|可以在需要新证据时再查，效果取决于模型的调用判断|

课件用 **Toolformer** 解释模型怎么学会第三种策略。它先让 LM 在普通文本里提议一些可能的 API 调用，再执行这些调用，最后只保留那些**能明显降低后续 token 预测 loss** 的例子：

\[ L(\text{no call})-L(\text{call + result})\ge \tau \]

然后用这些带工具调用的文本 fine-tune LM。直观上，如果一句话接下来要写一个计算结果，而调用计算器后正确数字更容易被预测，这次调用就有理由保留。**训练信号来自工具结果是否帮助预测后续文本**。

## 5. 找到了证据，模型为什么还是答错？（第 36–49 页）

### 5.1 Underutilization 与 Overreliance

- **Underutilization**：检索到相关证据，Generator 仍按权重中记住的信息作答。例如查到了课程的新截止日期，回答却沿用旧日期。
- **Overreliance**：Generator 对检索文本信得太多。例如检索结果把另一门课的作业日期带进来，模型直接拿它回答。

这里有两处可以分别排查。先看 **Retriever 有没有找到支持答案的证据**，再看 **Generator 有没有正确理解和使用证据**。最终答错，两种原因都可能发生。

### 5.2 为什么普通训练会让模型忽略证据？

课件用 `Saint Peter` 的实验说明。模型本来就会回答 `Saint Peter`，检索段落也写着 `Saint Peter`，这时它无论走哪条信息来源，都能答对。普通的答案监督很难区分模型究竟有没有读文档。

实验把检索段落里的答案替换成 `United Nations`，观察模型是否会跟着改答。课件报告，在**原本就答对的 Natural Questions 样本子集**中，模型约一半时候跟随修改后的段落。这是对模型如何处理**记忆与证据冲突**的实验，数字对应这一特定设置。

**Counterfactual training** 把这种冲突变成训练信号：同时修改段落与 gold answer，让 gold answer 跟着段落改变。这样，继续输出记忆中的旧答案就会产生 loss。课件用 \(M=\mathrm{old}/(\mathrm{old}+\mathrm{new})\) 衡量模型回到旧答案的比例，分母只统计回答旧答案或新答案的样本。

### 5.3 RETRO、Self-RAG 和答案验证

- **RETRO** 在 pretraining 阶段就引入检索。模型为文本 chunks 检索相似片段，再通过 **cross-attention** 读取它们。课件里的 7B 对 175B 指特定语言建模评估中的比较。
- **FLARE** 根据生成中的低置信度触发检索，让模型在可能缺信息时补资料。
- **Self-RAG** 让模型判断 passage 是否相关、生成的回答是否得到证据支持。
- **Citation / verification** 要求答案指向具体证据，或者在生成后逐条核对 claim。

和你在做的 **evidence-preserving prompt injection defense** 直接相关的是最后这一点：同一段 retrieved text 可以同时含有**回答问题所需的事实**和**试图指挥 agent 的指令**。例如，一条课程通知既写了截止日期，又要求 agent 把对话转发给某人。日期可以用来回答课程问题，转发操作的权限则必须来自用户授权。这个例子把课件的 grounding 问题进一步拆成两个判断：**这段文字能支持什么事实？它有没有权要求 agent 执行操作？**