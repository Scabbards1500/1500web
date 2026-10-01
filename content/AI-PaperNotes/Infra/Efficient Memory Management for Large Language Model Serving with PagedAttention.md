论文: https://arxiv.org/abs/2309.06180
参考视频: https://www.bilibili.com/video/BV1vKzuBmEWk/?spm_id_from=333.337.search-card.all.click&vd_source=f04f16dd6fd058b8328c67a3e064abd5 
代码: https://github.com/vllm-project/vllm

1. Identify two sources of KV-cache memory waste that PagedAttention addresses. What are the tradeoffs of larger versus smaller allocation blocks?


2. For a tool-calling task on one model instance, when would you preserve, discard/recompute, or offload its KV state during a pause? How could chunked prefill change the cost of resuming, and when would it provide no benefit?

p.s 天哪真的非常与时俱进的课，昨天是20260929，然后老师就课上和我们讨论jev的事情，今天又带着我们读vllm

# Abstract

要实现大语言模型（LLM）的高吞吐量服务，需要同时将足够多的请求组成批次进行处理。然而，现有系统难以做到这一点，因为每个请求的键值缓存（KV cache）占用大量显存，而且其大小会动态增长和缩减。如果管理不当，内存碎片和冗余复制会造成严重的显存浪费，限制批次大小。

为解决这一问题，我们提出了 **PagedAttention**，一种受操作系统经典虚拟内存和分页技术启发的注意力算法。在此基础上，我们构建了大语言模型服务系统 **vLLM**，实现了：（1）KV cache 显存几乎零浪费；（2）在同一请求内部以及不同请求之间灵活共享 KV cache，进一步减少显存占用。

实验表明，与 FasterTransformer 和 Orca 等最先进的系统相比，vLLM 在保持相同延迟水平的情况下，将常用大语言模型的吞吐量提升至 **2–4 倍**。对于更长的序列、更大的模型和更复杂的解码算法，这一提升更加显著。


首先我们先快速了解一下两个infra attention 的区别

**PagedAttention 主要解决“KV cache 怎么存”；FlashAttention 主要解决“attention 怎么算，才能少搬数据”。** 两者可以配合使用。

|对比|PagedAttention|FlashAttention|
|---|---|---|
|主要瓶颈|KV cache 的预留、碎片和重复存储|attention 计算中的显存读写|
|核心方法|KV cache 分页，按需分配，用 block table 找到数据|分块计算，在 GPU 片上 SRAM 中处理，减少与显存之间的数据搬运|
|节省什么|长期保存 KV cache 所占的显存|attention 中间结果的显存占用和读写|
|主要收益|容纳更多并发请求，提高服务吞吐量|加速 attention，尤其是训练和长输入 prefill|
|是否改变 attention 数学定义|不改变|不改变，属于精确 attention|




# Contribution
它的核心想法很好理解：**把 KV cache 像操作系统的内存一样分页管理，减少显存浪费，从而让 GPU 同时服务更多请求。**


## Background

**1. 为什么 LLM serving 会遇到显存问题?**

推理时，GPU 显存主要存三类东西：

|内容|特点|
|---|---|
|模型权重|模型加载后基本固定，多请求共享|
|KV cache|每条序列都有，随着上下文增长而增长|
|中间激活|计算过程中临时使用|




**2. 以前的做法浪费在哪里？**

论文研究的已有系统通常为一条请求分配一段**连续显存**，而生成长度又无法提前准确知道，所以需要预留空间。

例如，一条请求：

- 输入有 100 个 token；
- 最多允许生成 1,000 个 token；
- 实际只生成了 80 个 token。

如果一开始就按最大长度预留，那么很多空间根本不会使用。

论文把浪费分为三种：

|浪费|含义|
|---|---|
|Reserved space|未来可能用到，现在还没有使用，但别人不能用|
|Internal fragmentation|预留了空间，最终生成结束也没用上|
|External fragmentation|空闲显存分散成碎片，难以满足一大段连续分配|


论文测得，在其比较的系统和实验条件下，只有约 **20.4%–38.2% 的 KV cache 内存真正存储了已有 token 的状态**。这说明问题不只是“显存不够”，也是“显存没有被有效使用”。


## Solution

**PagedAttention 怎么解决**

借鉴操作系统的分页机制，把一条序列的 KV cache 分成固定大小的 **block**。

例如，每个 block 存 4 个 token 的 K、V。某条序列已有 10 个 token，那么只需要 3 个 block：

|逻辑 block|保存的 token|对应 GPU 物理 block|
|---|---|---|
|0|1–4|7|
|1|5–8|1|
|2|9–10，另有两个空位|5|

这里最关键的是：

**逻辑顺序连续，物理位置可以不连续。**

模型需要的序列顺序仍然是 token 1 到 token 10，但它们的 KV cache 可以分散在 GPU 的不同位置。系统用一张 **block table** 记录映射关系。

接下来：

- 序列增长到第 11、12 个 token：填满最后一个 block。
- 增长到第 13 个 token：再分配一个新 block。
- 请求结束：释放这些 block，供其他请求使用。

所以不用一开始就预留最大长度，**需要多少，分配多少**。

对于一条不共享的序列，未使用空间只出现在最后一个 block，浪费少于一个 block 的容量。论文采用的默认 block size 是 **16 个 token**。 PagedAttention



**分散存储后，attention 怎么算？**

这正是 **PagedAttention** 本身负责的事情。

普通 attention kernel 往往依赖连续存储的 KV cache；PagedAttention kernel 根据 block table，找到各个物理 block，读取其中的 K、V，完成 attention。

数学上仍然是：

数学上仍然是：

$$
\operatorname{Attention}(q,K,V)
=
\operatorname{softmax}
\left(\frac{qK^\top}{\sqrt{d}}\right)V
$$

**改变的是数据存储和访问方式，attention 的数学定义没有改变。**

还有一个容易误解的地方：**不是每个 block 单独做 softmax，再直接把结果相加。** 权重仍然需要在全部有效历史 token 上统一归一化。

因此，它也没有通过分页缩短上下文或忽略历史 token；它主要解决的是显存管理问题。


**更进一步：不同序列还能共享 KV cache**

假设同一个 prompt 要生成三个不同答案：

> “解释什么是强化学习。”

三个答案的 prompt 一样，没必要把 prompt 的 KV cache 复制三份。

可以让三条序列的 block table 指向同一组物理 block：

|部分|存储方式|
|---|---|
|共同 prompt|共享一份 KV cache|
|各自生成的后续内容|分别存储|

但如果共享的最后一个 block 没填满，两条序列接下来要写入不同 token，怎么办？

使用 **Copy-on-write，写时复制**：

1. 只读时共享。
2. 某条序列需要修改共享 block 时，复制该 block。
3. 后续分别写入。

系统通过引用计数判断一个 block 是否被多条序列共享，引用计数归零时就能释放。

这特别适合：

- **Parallel sampling**：同一个 prompt 生成多个回答。
- **Beam search**：多个候选共享已有前缀。
- **共享前缀**：不同请求使用相同的 system prompt 或示例。

这里共享需要的是**相同的有效前缀及计算条件**；仅仅在两句话里出现同一个词，并不能共享这个词的 KV cache，因为它受前文影响。


# Method

**Swapping**
显存不足时，先用 CPU 内存暂存 KV cache，腾出显存给其他请求。(可以理解为显存不够内存凑)

**Recomputation**
暂停请求恢复计算前，需要把 KV cache 搬回 GPU 显存，不是直接用 CPU 内存里的缓存做 GPU attention。释放 KV cache，恢复时用已有 token 重新计算

所以代价是：**省下暂停期间的显存占用，但增加 GPU ↔ CPU 的传输开销。**


### 补充: chunked prefill
**Chunked prefill 就是把一次长输入的 prefill，拆成多次处理。**
比如工具返回了 **4,000 个 token**：

|方式|处理过程|
|---|---|
|普通 prefill|一次处理全部 4,000 个 token|
|Chunked prefill|每次处理 500 个 token，共 8 次|

每处理一块，就保存它的 KV cache；下一块计算时，会使用之前各块的 KV cache，**所以各块之间仍然能进行 attention，上下文没有被切断。**

这样做主要有两个作用：

- **降低单次计算的临时显存峰值**，但最终完整 KV cache 的大小没有减少。
- **给其他请求让出调度机会**：处理完一块后，可以先让其他请求生成 token，避免长 prefill 一直占用 GPU。








