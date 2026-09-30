abstract: 这里放点LLM 参数以及调整经验.jpg
time: 2026/07/02
参考

[https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)

## 流程

prefill + decode

![](https://i-blog.csdnimg.cn/direct/90c785037e294831b244eda5bdd37629.png)

### prefill

**computing the first output token**

把用户输入的整个 prompt 一次性喂给模型。

例如：`"Explain why RAG can reduce hallucination"`

假设 token 数是 10 个，那么模型会**并行处理这 10 个 token**，计算每一层的 hidden states，并生成对应的 **KV cache**。

特点：

- **高度并行**
- 主要受 **计算能力（FLOPs）** 影响
- 输入越长，Prefill 越重
- 可以认为是在做：「我先把你说的话全部理解一遍」

#### **KV cache**

推理生成时，每次生成一个 token，不可能把所有历史重新算一遍，所以需要把历史 token 的 Key 和 Value 保存下来，下次生成复用。

为啥 prefill 阶段（从输入到第一个token输出）慢： For the very first output token,  we start with an initially empty KV cache and need to calculate as many sets of *key* and *value* vectors as there are tokens in the input prompt.

### decode

**computing any later output token**

Prefill 完之后，模型开始**一个 token 一个 token 地生成答案**。

比如：Explain → why → RAG → can → reduce → hallucination → ...

**autoregressive generation：** 每生成一个 token：

1. 把新 token 输入模型
2. 利用之前保存的 **KV cache**
3. 预测下一个 token
4. 把新的 K/V 加进 cache
5. 重复

特点：

- **不能像 Prefill 那样把所有生成 token 完全并行处理（cannot be parallelized）**
- 每一步通常只处理一个新 token
- 很容易受到 **memory bandwidth / KV cache / latency** 限制
- 生成长度越长，Decode 越重

### Data Process

某种意义上也可以理解为

**natural language 上下文 → hidden state → logits → 下一个 token**

#### **hidden state**

**模型在当前这一步，对“目前已经看到的内容”的内部表示，** 是模型在网络**内部各层产生的中间表示向量**。

它不是未来 token，也不是单词本身，而是模型在每一层中产生的高维向量，可以理解为模型此刻的“压缩理解”或“中间特征”。

在 Transformer 中，假设当前序列长度为 $T$，hidden dimension 为 $d$，那么第 $l$ 层的 hidden states 可以表示为：

![](https://i-blog.csdnimg.cn/direct/31cc8020955b4dfa851c7fc9c3e64cea.png)

其中：

- $T$：当前序列长度
- $d$：每个 token 的向量维度，比如 4096
- 每个 token 在每一层都有一个长度为 $d$ 的 hidden state

例如输入：

$$
\text{The cat sat}
$$

如果分成 3 个 token，那么某一层会得到：

- “The” → 一个 4096 维向量
- “cat” → 一个 4096 维向量
- “sat” → 一个 4096 维向量

这 3 个向量合起来，就是这一层的 hidden states。

这些向量本身不是英文单词，而是模型内部的数值表示，编码了语义、上下文关系、句法角色，以及下一步生成的可能性等信息。

**生成时 hidden state 怎么产生？**

自回归模型一次只预测下一个 token。

当模型要生成第 $x$ 个 token 时，它已经看到：

$$
\text{输入} + \text{前面已生成的 } x-1 \text{ 个 token}
$$

模型会基于这些内容做 forward，并在每一层产生 hidden states。然后它只使用**最后一个位置的 hidden state** 来预测第 $x$ 个 token。

还没有生成出来的 token，并没有对应的 hidden state，因为模型还没有看到它们。

**在自回归生成里，为什么只关心“最后一个位置”的 hidden state**

GPT 这类自回归模型的目标是：

$$
P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

也就是说，模型根据当前前缀预测下一个 token。

因此，当当前上下文长度为 $t-1$ 时，最后一个位置的 hidden state 就包含了模型对整个当前上下文的总结，所以用它来预测第 $t$ 个 token。

#### LM head

LM Head 是模型最后的输出层，负责把最后一个 hidden state 转换成 logits：

$$
\text{logits} = Wh + b
$$

其中 $h$ 是最后一个位置的 hidden state。

logits 表示词表中每个 token 作为下一个 token 的分数。模型再通过 softmax 或采样策略，从中选出下一个 token。

## Metrics

![](https://i-blog.csdnimg.cn/direct/631ffcb8911b402ba73ed015d6b8a8d8.png)

因为是并行解码的，所以总token 一致的情况下prefill 阶段解码比decode 阶段更快

so input api 比 output api 要便宜

**TTFT (*Time to first token*)**

latency of the prefill phase

**TPOT (time per output token)**

latency of a single decode step

**token throughput**

tokens per second, summed up over all concurrent requests

\* only used for non-interactive situations

## Resource Utilization

prefill phase is very GPU compute-intensive

decode limited by the GPU memory bandwidth (**GPU utilization can be increased by batch processing** of multiple requests)

**并发少时**，一次读取权重只为少数请求各生成一个 token，GPU 的计算能力没用满。多加入几个请求，相当于让同一次权重读取服务更多 token，所以总吞吐量可能近似随并发数增长。这时主要受**显存读取速度**限制，即 *memory-bound*。

**并发足够多时**，每一步要计算的 token 数也多了，GPU 算力逐渐用满。再加入请求，GPU 已经没法每秒完成更多计算，所以**总 token/s 大致不再增加**。这时进入 *compute-bound* 区域。

\* 注意这里的 throughput 是**所有请求合计每秒生成的 token 数**

![](https://i-blog.csdnimg.cn/direct/553956244a64475d9fa8744c902bff3d.png)

## Concurrent Processing 并发处理

### **static batching**

(1) You start with an empty batch, (2) you fill the batch with as many items as are waiting and as fit into the batch, (3) you process the batch until all batched items are finished, and (4) you repeat the procedure with a new empty batch.

the next waiting request can only start once the longest batched request has been completed.

![](https://i-blog.csdnimg.cn/direct/2e3cd17f90004cb197d0232cbb2bfae8.png)

Static batching optimizes the *time per output token*， however  long *time to first token*

Even if some short requests finish early, the next queuing request has to wait for the longest decode in the batch to finish before its prefill can begin.

### **continuous batching**

any completed request is immediately removed from the batch, and the batch space is filled with the next request in line.

| 策略 | 新请求的 prefill | 对已有请求的影响 |
| --- | --- | --- |
| **Prefill-First** | 优先一次处理完整个 prompt | 长 prompt 会让已有请求的流式输出出现明显停顿 |
| **Chunked Prefill** | 把 prompt 分成多个 chunk，分多轮处理 | 每处理完一块，就有机会继续生成已有请求的 token；输出通常更平稳 |

**Prefill-First**

![](https://i-blog.csdnimg.cn/direct/4380d1d26ce04288a7e808a241d3cfee.png)

**Chunked Prefill**

![](https://i-blog.csdnimg.cn/direct/5ee4e5a82c654131b723699781652641.png)

## 参数

### GPU 相关

**`CUDA_VISIBLE_DEVICES`**

指定当前程序能看到哪些 GPU。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

表示使用物理第 0、1、2、3 张卡。

### 多卡/分布式训练

`WORLD_SIZE`

总进程数，通常等于使用的 GPU 数量。

例如 4 卡训练：

```
WORLD_SIZE=4
```

`RANK`

当前进程在所有进程中的全局编号。

例如 4 卡训练中，进程编号可能是：

```
RANK = 0, 1, 2, 3
```

`LOCAL_RANK`

当前进程在本机上的 GPU 编号。

例如单机 4 卡训练时：

```
LOCAL_RANK = 0, 1, 2, 3
```

在代码中常见写法：

```python
torch.cuda.set_device(local_rank)
```

### 数据加载相关

**`NUM_WORKERS`**

DataLoader 用多少个子进程加载数据。

```python
DataLoader(dataset, batch_size=32, num_workers=4)
```

含义：

```
num_workers 越大，数据加载并行度越高；
但太大可能占用过多 CPU / 内存。
```

常见设置：

```
小数据集：0 / 2
普通训练：4 / 8
大规模数据：8 / 16+
```

`num_workers` 越大越好吗？

- no

CPU 占满
内存占用变高
进程切换开销变大
甚至 DataLoader 卡住

怎么判断它是不是太小？

训练时可以观察 GPU 利用率。

如果你看到：

```
GPU 利用率一会儿 0%，一会儿 90%
显存占着，但是 GPU 经常不干活
```

很可能是数据加载跟不上。

这时候可以尝试增大

**`BATCH_SIZE`**

每次送进模型的样本数量。

```
BATCH_SIZE=32
```

batch size 越大，显存占用越高，但训练通常更稳定。

---

`GRADIENT_ACCUMULATION_STEPS`

梯度累积步数。

```
GRADIENT_ACCUMULATION_STEPS=4
```

如果显存不够，不能直接开大 batch size，就可以用梯度累积模拟更大的 batch。

有效 batch size 通常是：

```
effective batch size = batch_size × gradient_accumulation_steps × GPU数量
```

例如：

```
batch_size = 2
gradient_accumulation_steps = 8
GPU数量 = 4

effective batch size = 2 × 8 × 4 = 64
```

### 模型和缓存路径相关

`HF_HOME`

Hugging Face 的总缓存目录。

```
HF_HOME=/mnt/data/huggingface
```

模型、tokenizer、dataset 等缓存都会放到这个目录下。

---

`TRANSFORMERS_CACHE`

transformers 模型缓存目录。

```
TRANSFORMERS_CACHE=/mnt/data/hf_models
```

---

`HF_DATASETS_CACHE`

datasets 数据集缓存目录。

```
HF_DATASETS_CACHE=/mnt/data/hf_datasets
```

如果服务器默认 home 目录空间很小，就经常需要把这些缓存路径改到大磁盘上。

---

### 实验记录相关

`WANDB_PROJECT`

设置 wandb 项目名。

```
WANDB_PROJECT=my_project
```

---

`WANDB_MODE`

控制是否启用 wandb。

```
WANDB_MODE=offline
```

常见取值：

```
online：正常上传
offline：离线记录
disabled：关闭 wandb
```

### 常用训练超参数

`LEARNING_RATE`

学习率，控制参数更新步长。

```
LEARNING_RATE=2e-5
```

学习率太大容易训练不稳定，太小收敛慢。

---

`EPOCHS`

完整遍历训练集的次数。

```
EPOCHS=3
```

---

`MAX_SEQ_LEN`

最大输入长度。

```
MAX_SEQ_LEN=4096
```

LLM 训练 / 推理里非常重要。长度越大，显存和计算成本越高。

---

`SEED`

随机种子，用于复现实验。

```
SEED=42
```

---

`PRECISION`

训练精度。

常见设置：

```
fp32：最稳定，但最慢、最耗显存
fp16：更快、更省显存，但可能不稳定
bf16：更稳定的混合精度，A100/A800/H100 常用
```

例如：

```
PRECISION=bf16
```
