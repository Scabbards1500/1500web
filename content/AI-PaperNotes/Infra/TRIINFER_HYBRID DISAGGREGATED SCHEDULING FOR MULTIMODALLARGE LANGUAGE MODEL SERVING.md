time: 2026/11/02
### 一、问题背景

MLLM 推理分为三个阶段：

1. **Encode（编码）**：视觉塔处理图像，生成图像 token
2. **Prefill（预填充）**：将图像 token + 文本 token 送入语言模型，生成第一个输出 token 及 KV Cache
3. **Decode（解码）**：基于 KV Cache 自回归生成后续 token

现有系统（vLLM、SGLang 等）的核心问题是**将三个阶段耦合在一起**，带来三类挑战：

| 问题     | 具体表现                                     |
| ------ | ---------------------------------------- |
| 并行性不足  | 图像编码与语言处理串行，GPU 闲置                       |
| 调度粒度粗  | 无法针对各阶段特性（计算密集 vs 内存密集）单独优化，TBT SLO 频繁违规 |
| 解聚方法固定 | 单一静态解聚策略无法适应不同工作负载和 SLO 要求               |

---

### 二、核心洞察（Takeaways）

论文通过分析得出五个关键观察：

- **Takeaway-1**：三个阶段的计算/内存特性各异（Prefill 计算密集、Decode 内存密集、Encode 居中），具备并行优化空间
- **Takeaway-2**：各阶段在不同 batch size 下达到吞吐饱和点（Encode ~6，Prefill ~1，Decode ~512），超过后只增延迟
- **Takeaway-3**：现有 LLM 调度策略（prefill 优先 / stall-free）未考虑图像编码延迟，难以精确控制 TBT
- **Takeaway-4**：不同解聚方法下，实例比例对 TTFT/TBT 的影响各异，最优比例取决于 SLO 要求
- **Takeaway-5**：E+P+D / EP+D / ED+P 三种解聚方法各有适用场景，应根据工作负载动态选择

---

### 三、TriInfer 系统设计

系统由四个核心组件构成：

#### ① Request Processor（请求预处理）

将 tokenization、图像处理等 CPU/IO 密集任务提前并行处理，减少推理主路径上的 CPU 开销，提升系统可扩展性。

#### ② Batch Scheduler（批调度器）

提出**Stage-level Batching 算法**：

- 根据实例类型和 SLO 要求（TTFT/TBT）动态设置 **image budget**（图像批大小上限）和 **token budget**（文本批大小上限）
- 用二分搜索在初始化时 profile 最大 budget
- 每次迭代优先纳入 decode → chunked prefill → encode → 新请求，确保 TBT 达标的同时最大化吞吐

同时采用**双流并行（Dual-stream Parallelism）**：一个 CUDA stream 处理图像编码，另一个处理语言模型，两者并行执行，消除互相等待。

#### ③ Migrate Scheduler（迁移调度器）

负责实例间的请求迁移，防止 KV Cache / 图像 Cache 溢出。迁移流程四步：

1. 源实例发送控制信息（Cache 页表）
2. 目标实例创建页表并拉取 Cache 块
3. 源实例异步传输 Cache
4. 完成后通知源实例释放资源

支持轮询、最小负载、前缀缓存命中率优先等负载均衡策略。

#### ④ Hybrid EPD Disaggregation（混合解聚，核心创新）

系统支持三种解聚方式，并**自动选择最优方案**：

|方法|优势|
|---|---|
|**E+P+D**|Encode 实例无需存 KV Cache，图像 Cache 更大，适合严格 TTFT 场景|
|**EP+D**|只需一次 KV Cache 传输，视觉+语言模型可并行，传输开销小|
|**ED+P**|视觉与语言模型并行，Prefill 独立，有助于降低 TTFT|

**Profiler 算法（Algorithm 2）** 基于历史 trace 自动决策：

1. 统计 Encode/Prefill/Decode 的 token 总量
2. 估算三种实例的吞吐，计算各阶段所需时间
3. 按时间比例分配实例数量
4. 对三种方案回放历史 trace 测 goodput，选最优
5. 工作负载漂移时定期重新触发，整个过程约需几分钟

---

### 四、实验结果

**测试环境**：4 台服务器，各 8 块 NVIDIA H20 GPU，InfiniBand 200Gbps 互联

**主要结论**：

- 与 vLLM/SGLang 相比，在满足 P90 SLO 的前提下，goodput 最高提升 **2.4×**
- 在不同数据集上（MME/POPE/TextCaps/TextVQA/VizWiz）均有显著提升
- Ablation 实验验证：混合解聚贡献最大，stage-level 调度进一步降低 TBT 违规率
- 迁移开销极小（95% EP 迁移 < 2ms，PD 迁移 < 8ms），总延迟占比不足 1%
- 双流并行相比串行执行显著提升吞吐，效果在不同 batch size 下均稳定