
<span style="color: #66ccff;">
感觉对我来说收益不大啊这篇</span>

MegaScale-Infer 研究如何提高大规模 MoE 模型的推理效率，重点优化逐个生成 token 的 **decode 阶段**。

它的出发点是：**MoE 虽然减少了每个 token 的计算量，但实际推理成本不一定同比降低，因为 GPU 算力可能没有被充分利用。**

论文的核心方法是将 Attention 和 Expert FFN 分开部署，通过汇集多个 Attention 节点的 token，提高 expert 的计算效率，再利用流水线和专用通信库降低拆分带来的开销。

## 2. 为什么 MoE 容易浪费 GPU 算力？

### 2.1 Dense FFN 可以通过 batch 提高权重复用

普通 Transformer 的一层主要包含 Attention 和 FFN。

在 Dense 模型中，一个 batch 的所有 token 都经过同一个 FFN，因此可以合并成较大的矩阵乘法。同一份 FFN 权重读取后，可以供多个 token 使用。

**batch 越大，权重读取成本越容易被摊薄，GPU 越容易发挥计算能力。**

### 2.2 MoE 把 token 分散到了不同 expert

MoE 将 FFN 换成多个 expert，由 router 为每个 token 选择其中几个。

假设：

- 一个 batch 有 128 个 token。
- 模型有 8 个 expert。
- 每个 token 选择 2 个 expert。

在路由均衡的情况下，每个 expert 平均只收到：

\[
128 \times \frac{2}{8} = 32 \text{ 个 token}
\]

原本较大的矩阵乘法被分散成了多个较小的矩阵乘法。每个 expert 仍然需要读取自己的权重，却只处理少量 token。

这容易导致 **memory-bound**：执行速度主要受权重读取带宽限制，GPU 的计算能力没有充分发挥。

因此：

> MoE 减少了 FLOPs，但减少 FLOPs 不意味着执行时间和成本会同比减少。

## 3. 为什么不直接增大 batch？

增大 batch 可以让每个 expert 收到更多 token，但受到两个主要限制。

### 3.1 KV cache 占用显存

每个请求都需要保存自己的 KV cache。并发请求越多、上下文越长，KV cache 占用越大。

同时，MoE 的大量 expert 权重也需要占用显存，因此不能无限增加 batch。

### 3.2 在线服务有延迟要求

batch 过大可能增加一轮 decode 的执行时间，让用户等待下一个 token 的时间变长。

此外，Attention 和 FFN 从 batching 中获得的收益不同：

| 模块 | batch 增大时的主要变化 |
|---|---|
| FFN | 多个 token 复用相同权重，提高计算相对于权重读取的比例 |
| Decode Attention | 每个请求仍需要读取自己的 KV cache，内存访问量随请求数增加 |

这就形成了一个矛盾：

> Expert 希望汇集更多 token，但 Attention 一侧受到 KV cache 容量和响应延迟的限制。

## 4. 核心设计：Attention 与 Expert 分开部署

论文提出 **Disaggregated Expert Parallelism**，将 Attention 和 Expert FFN 部署在不同的 GPU 节点上。

| 节点类型 | 保存的内容 | 主要工作 |
|---|---|---|
| Attention 节点 | Attention 参数、请求的 KV cache | 处理分配给自己的请求 |
| Expert 节点 | 对应 expert 的权重 | 汇集 token，执行 Expert FFN |

关键在于：**多个 Attention 节点可以共同使用一组 Expert 节点。**

### 4.1 如何提高 expert 的 batch？

假设有 4 个 Attention 节点，每个节点处理 128 个 token，并将 token 按照路由结果发送到同一组 expert。

Expert 侧汇集到的总工作量来自：

\[
4 \times 128 = 512 \text{ 个 token}
\]

仍然假设有 8 个 expert、每个 token 选择 2 个，那么每个 expert 平均收到：

\[
512 \times \frac{2}{8} = 128 \text{ 个 token}
\]

因此：

- 单个 Attention 节点只承担自己请求的 KV cache。
- Expert 可以汇集多个节点的工作，执行更大的矩阵乘法。

**KV cache 留在 Attention 节点上，两侧主要传输 token 对应的中间隐藏向量。**

可以把它理解为：多个前台分别服务客户，但把加工任务集中到一组共享车间。前台独立扩容，车间汇集订单，提高设备利用率。

## 5. 分别选择并行方式和硬件

Attention 与 expert 分开后，可以独立配置资源。

### 5.1 并行方式

- **Attention 数据并行**：多个节点复制 Attention 参数，分别处理不同请求。
- **Expert 专家并行**：不同节点负责不同 expert。
- **节点内张量并行**：必要时，用多张 GPU 共同执行 Attention 模块或 expert。

### 5.2 异构硬件部署

Decode Attention 更依赖显存容量和内存带宽；Expert 汇集足够多的 token 后，更能利用 GPU 的计算能力。

因此，两侧可以使用不同型号的 GPU，根据各自需求选择更划算的硬件。

但拆分也带来了代价：**每一层都需要在 Attention 和 Expert 节点之间交换中间结果。**

## 6. Ping-pong 流水线：减少相互等待

### 6.1 直接拆分会造成空闲

对同一个 token，一层中的执行顺序仍然是：

\[
\mathrm{Attention}
\rightarrow
\mathrm{Expert}
\rightarrow
\text{下一层 Attention}
\]

如果一次只处理一个完整 batch：

- Attention 计算时，Expert 等待。
- Expert 计算时，Attention 等待。
- 网络传输期间，两侧也可能等待。

### 6.2 用多个 micro-batch 交错执行

论文将 batch 切成多个 **micro-batch**：

- Expert 处理第一批时，Attention 可以处理第二批。
- 第一批从 Expert 返回后，再进入下一层 Attention。
- 不同批次在两侧之间不断往返。

这就是 **ping-pong pipeline parallelism**。

**同一个 token 仍然遵守模型的计算依赖，但不同 micro-batch 可以并行推进，填补彼此的等待时间。**

通信也可以与其他批次的计算重叠，减少实际暴露出来的通信开销。

### 6.3 流水线有效的条件

- Attention 和 Expert 的计算时间尽量接近。
- 网络通信足够快，可以被其他计算掩盖。
- micro-batch 数量足够填满流水线。
- micro-batch 不能太小，否则 Expert 的矩阵乘法效率会下降。

这里存在一个权衡：

> 切分更多 micro-batch 有利于流水线重叠，但切得太碎又会降低 expert 的计算效率。

## 7. 专用 M2N 通信库

Attention 和 Expert 的节点数量可以不同，因此通信模式变成：

\[
M \text{ 个 Attention 节点}
\longleftrightarrow
N \text{ 个 Expert 节点}
\]

作者为此设计了专用的 **M2N 通信库**，减少：

- 不必要的 GPU 到 CPU 数据拷贝。
- 通信组初始化开销。
- GPU 同步开销。

这很关键，因为每一层都要进行两侧之间的通信。如果通信太慢，Expert 计算效率提高带来的收益可能被抵消。

论文报告，在其通信实验中，相比 NCCL，该通信库最高实现约：

- **4.2 倍通信吞吐。**
- **68.2% 的通信延迟下降。**

这些是通信组件的结果，不等于整个模型的加速倍数。

## 8. 如何选择部署配置？

系统利用性能模型，根据模型结构、请求长度、硬件性能和延迟要求搜索部署方案，包括：

- Attention 与 Expert 各自的张量并行规模。
- Attention 节点数量。
- micro-batch 数量。
- 满足延迟要求时允许的总 batch 大小。

优化目标是：

> 在满足服务延迟要求的前提下，提高单位成本的吞吐量。

因此，系统需要同时考虑计算效率、显存容量、通信开销和两侧的负载平衡。

## 9. 实验结果与适用条件

论文使用约 **132B–317B 参数规模的 MoE 模型**进行评估，报告每 GPU 的 decode 吞吐相比所评估的先进系统最高提升约 **1.9 倍**，并展示了异构 GPU 部署的成本收益。

需要区分：

- **吞吐提升**：相同 GPU 资源能够服务更多请求。
- **单请求延迟降低**：一个用户等待 token 的时间更短。

论文的 1.9 倍主要描述前者，不能直接理解为每个用户生成 token 都快了 1.9 倍。

系统收益依赖足够的并发请求、较快的网络，以及 Attention 和 Expert 两侧的负载平衡。

## 10. 核心逻辑

MoE 的稀疏路由让每个 expert 收到的 token 太少，导致权重读取成本难以摊薄。

MegaScale-Infer 将 Attention 与 Expert 分开，让多个 Attention 副本共同向 Expert 提供工作，提高 expert 的有效 batch；再通过 ping-pong 流水线、专用通信库和部署配置搜索，降低拆分带来的额外成本。