abstract: Generative AI System Design Interview (1) -- Introduction and Overview 学习笔记轻轻的开始准备暑期intern...我打宿傩，真的假的（
time: 2026/07/07
## GenAI overview

主要是模型和方法分类吧，这块我们熟就不管了

![](https://i-blog.csdnimg.cn/direct/29854229f12e4968b269b34c73c22d70.png)

![](https://i-blog.csdnimg.cn/direct/ac8dc53e603543d18fa29ff33b702957.png)

## Data Preparation

### **Data types**

![](https://i-blog.csdnimg.cn/direct/83cf0773e219437f97931f8b5ef17887.png)

### Data efficiency

**Efficient retrieval**

**让训练时取数据更快，减少等待。主要靠：分片并行读、索引快速找、缓存减少 I/O。**

1. **Sharding 分片**
   把大数据集切成多份，放到不同机器/设备上，并行读取。
   **核心：多人同时搬砖，比一个人快。**
2. **Indexing 建索引**
   给数据建立“目录”，比如 Lucene / Elasticsearch，方便快速定位需要的数据。
   **核心：像查字典，不用从头翻到尾。**
3. **Pre-loading / Caching 预加载/缓存**
   把常用数据提前放进内存，避免每次都从硬盘慢慢读。
   **核心：常用的东西提前放桌上，随手拿。**

## Model training

### Task-specific challenges and mitigations

不同任务有不同瓶颈。比如 **视频生成模型** 特别重，因为它需要处理大量帧、时序关系和高维视觉信息，所以训练成本很高。

常见缓解方法包括：

> **并行训练 + 混合精度 + latent diffusion 等方法，降低训练成本。**

#### **Gradient checkpointing 梯度检查点**

正常训练会保存很多中间激活值，用于反向传播，但这很占显存。

Gradient checkpointing 的做法：

不保存所有中间结果，只保存关键节点；反向传播时，缺的中间结果再重新算一遍。

所以它是用更多计算时间，换更少显存，适合显存不够但还能多算一点的情况。

#### **Mixed precision training 混合精度训练**

正常训练可能用 FP32，也就是 32-bit 精度，比较稳但慢、占显存。

混合精度训练会让大部分计算用 FP16 / BF16，关键地方保留 FP32。

所以它是：

**大部分地方用低精度加速，关键地方用高精度保稳定。**

优点：

训练更快，显存更省，效果基本不变。

**Automatic mixed precision (AMP)** 就是自动混合精度工具，PyTorch / TensorFlow 可以自动判断哪些地方用 FP16，哪些地方用 FP32。

#### **Distributed training 分布式训练**

当单机单卡训练不了大模型时，就把训练分到多张 GPU 或多台机器上。

![](https://i-blog.csdnimg.cn/direct/f7c3262ec70b4f43867b2c5d425f1d76.png)

##### **Data Parallelism 数据并行**

**每张 GPU 都放一份完整模型，但处理不同的数据。**

![](https://i-blog.csdnimg.cn/direct/df9619e230744cd688127bc11819e453.png)

比如有 4 张 GPU：

- GPU 0 训练 batch A
- GPU 1 训练 batch B
- GPU 2 训练 batch C
- GPU 3 训练 batch D

然后大家算完梯度，把结果汇总，更新模型参数。

核心一句话：**模型复制多份，数据分给不同 GPU。**

适合：**数据很大，但模型单张卡还能放下。**

**更新方式**

**Synchronous 同步更新**

所有 GPU 都算完，统一汇总梯度，再一起更新参数。

优点：结果稳定，大家用的是同一个版本的模型。
缺点：要等最慢的 GPU，所以可能慢。

一句话：大家都做完作业，再一起交。

**Asynchronous 异步更新**

哪个 GPU 先算完，就先把梯度发给参数服务器，参数服务器马上更新。

优点：更快，不用等最慢的 GPU。
缺点：不同 GPU 可能用的是稍微旧一点的模型参数，训练可能不太一致。

一句话：谁先做完谁先交，但大家手里的题目版本可能有点不同。

##### **Model Parallelism 模型并行**

**模型太大，一张 GPU 放不下，所以把模型切到多张 GPU 上。**

| 方法 | 怎么切 | 适合什么情况 |
| --- | --- | --- |
| Pipeline Parallelism | 按层切 | 模型层数很多 |
| Tensor Parallelism | 按矩阵/张量切 | 单层太大 |

**Pipeline Parallelism**

流水线并行,按层切模型。

比如一个模型有很多层：

GPU 0 负责 Layer 0–1
GPU 1 负责 Layer 2–3
GPU 2 负责 Layer 4–5

前向传播时，数据从 GPU 0 → GPU 1 → GPU 2。
反向传播时，梯度再从 GPU 2 → GPU 1 → GPU 0 传回来。

![](https://i-blog.csdnimg.cn/direct/77b45912aa4747119980b8f707043857.png)

**Tensor Parallelism**

按层内部的矩阵计算切模型

usage: Transformer 里的大线性层 / attention / MLP。

B=A×W

如果 W 太大，一张 GPU 算不动，就把 W 切成几块：

- GPU 0 算 A×W1=B1
- GPU 1 算 A×W2=B2
- 最后把 B1,B2 拼起来得到 B

![](https://i-blog.csdnimg.cn/direct/1d52c57b33ae4e5ca79b285c3fe919f5.png)

##### **Hybrid Parallelism 混合并行**

把数据并行、模型并行、流水线并行组合起来。

**超大模型训练通常都用混合并行。**

ZeRO (Zero Redundancy Optimizer)[49] from Microsoft

FSDP (Fully Sharded Data Parallel) [50] from Meta

## Model Sampling

generating new data or outputs from the trained generative model.

| 方法 | 是否随机 | 核心逻辑 | 特点 |
| --- | --- | --- | --- |
| Greedy | 否 | 每步选最高概率 | 快，但死板 |
| Beam Search | 否 | 保留多条高分路径 | 稳，但贵、可能模板化 |
| Top-k | 是 | 从前 k 个词里抽样 | 多样，但 k 要调 |
| Top-p | 是 | 从累计概率 p 的词里抽样 | 更灵活，常用于开放生成 |

Greedy 只看第一名；Beam 看前几条完整路线；Top-k 在前 k 个词里随机；Top-p 在“概率够大的候选池”里随机。

## Evaluation

**Offline evaluation**

**Offline evaluation** is the process of assessing the performance of a model or system using pre-collected data without deploying it in a real-time environment.

![](https://i-blog.csdnimg.cn/direct/e70d3bec2107473faa9d30345bb62193.png)

![](https://i-blog.csdnimg.cn/direct/6a478a222fcd422d90468c77f7d319bc.png)

**online evaluation**

The **online evaluation** assesses how the model performs in production (i.e., after deployment).

![](https://i-blog.csdnimg.cn/direct/dfc28652882f4b6a8e76fcc76e88abe5.png)
