这节课接着上一讲的 **Preble 和 KV-cache routing**，把 serving 的问题往三个方向展开。先讲普通 LLM 如何调度 prefill、decode 和长上下文，再讲 MoE 如何把 attention 和 experts 分开部署，最后讲一个 agent 发出多次模型调用时，如何优化**整个任务的完成时间**。

复习时最容易混的是三种“拆分”。**Prefill/decode disaggregation** 按推理阶段拆，交接请求的 **KV cache**。**Ring Attention** 按序列位置拆，同一层里循环传 **K/V blocks**。**MegaScale-Infer** 按层内模块拆，每层在 attention 和 experts 之间来回传 **token activations**。把“拆的是谁、传的是什么”分清，三种设计的收益和瓶颈就容易推出来。




这节课围绕两个场景展开。**多个请求共享同一段 prompt 时，Preble 决定把请求发到哪张 GPU，尽量复用已有的 KV cache。新请求的长 prefill 拖慢正在生成的请求时，DistServe 把两个阶段分开跑，Sarathi-Serve 则把 prefill 切小后交错执行。**

下面按主题分章，页码对应 PDF。lecture13-distributed-llm-serving-i_new.pdf

 为什么 prompt prefix 值得缓存（pp. 1–13）
![[Pasted image 20261110111843.png]]

**Prefill** 处理整段输入 prompt，生成各层的 KV cache，并预测第一个输出 token。**Decode** 随后逐步生成，每一步读取已有的 KV，并把新 token 的 KV 追加进去。

课件 p. 5 的四个例子说明了为什么前缀复用很常见。Few-shot 请求可以共享同一组示例，多轮对话可以共享聊天历史，self-consistency 的多次采样可以共享同一道题，Tree-of-Thought 的不同分支也可以共享分叉前的搜索历史。

例如两个请求都是 `系统指令 + 同一份长文档 + 各自的问题`。如果 GPU 已经缓存了系统指令和文档的 KV，第二个请求只需要为新问题计算 KV，同时仍然可以注意到整份文档。这里要求的是**从开头起 token 序列相同**。如果两段文本出现在不同前文之后，它们的 KV 一般也会不同。

![[Pasted image 20261110112951.png]]
pp. 8–13 把这个问题推广到多 GPU。**前缀树(上图）**记录哪些请求共享前文，**cache-aware routing** 决定请求去哪台 worker 才能用上这些缓存。Round-robin 即使把请求数均匀分配，也可能把共享同一份长文档的请求拆散，让几张 GPU 各算一遍。


## 2. Preble 怎么平衡缓存复用和排队（pp. 14–23）

假设 GPU 0 缓存了某份热门文档，但队列已经很长。GPU 1 几乎空闲，却需要重新处理文档。**命中缓存省下来的是 prefill 计算，用户感受到的延迟还包括排队。** 如果所有相关请求都去 GPU 0，缓存命中率可以很高，但尾延迟可能更差。

<span style="color: #33cc33;">尾延迟（tail latency）就是最慢那一小部分请求的延迟**，常用 **p95** 或 **p99** 衡量。比如 **p99 延迟是 2 秒**，意思是约 99% 的请求能在 2 秒内完成，剩下约 1% 要等更久。</span>

Preble 的 **E² scheduling** 用两个策略做这个权衡。

| 策略                 | 课件里的做法                                                       |
| ------------------ | ------------------------------------------------------------ |
| **Exploit**（p. 17） | 当已匹配的前缀比未缓存的剩余部分还长时，倾向把请求交给有缓存的 GPU，因为可省的计算较多。               |
| **Explore**（p. 18） | 为各 GPU 估算带有前缀信息的负载成本，选成本最低的 GPU。热门前缀也可以在其他 GPU 上建立副本，分担后续请求。 |

pp. 19–20 讲的是两层调度。**全局调度器决定请求去哪个实例，本地调度器决定到了以后谁先跑。** 本地纯 FCFS 会错过复用机会，纯 longest-prefix match 又可能让短匹配请求长期等候。Preble 在优先级组之间轮转，组内按先来先服务排序。p. 21 报告的实验中，平均请求延迟改善了 1.5–14.5×，p99 改善了 2–10×。

## 3. 为什么 Prefill 和 Decode 放一起会互相拖慢（pp. 25–50）

先区分两种用户能感受到的延迟。

| 指标                              | 看的是哪段时间                                        |
| ------------------------------- | ---------------------------------------------- |
| **TTFT**（Time to First Token）   | 请求发出到看到第一个 token。排队和 prefill 都会影响它。            |
| **TPOT**（Time per Output Token） | 首 token 之后，生成一个输出 token 平均要多久。它反映后续输出的速度。      |
| **Goodput**                     | 每秒有多少请求满足延迟目标（SLO）。即使吞吐量很高，超时的请求也无法贡献 goodput。 |


pp. 33–43 的时间线可以这么读。请求 A 正在逐 token 输出，这时请求 B 带着一个长 prompt 进来。长 prefill 一般有大量 token 可以并行计算，偏 **compute-bound**。Decode 每条序列每轮通常只推进一个 token，需要读模型权重和历史 KV，常偏 **memory-bound**。如果把 B 的整段 prefill 和 A 的一步 decode 放进同一轮，A 的下一 token 就得等这一长轮结束。p. 42 的一个配置里，decode step 被拉长了约 **12.6×**。

pp. 44–50 还有第二个问题。两阶段共享实例时，也共用同一套 GPU 和并行配置。即使系统主要卡在 TTFT，你也难以只给 prefill 增加资源或单独调整它的并行方式。

## 4. DistServe 怎么用分离提高 Goodput（pp. 51–71）

DistServe 把服务分成 **prefill workers** 和 **decode workers**。前者处理 prompt，后者拿到这次请求的 KV cache 后继续生成。这样两个阶段可以分别选 GPU 数量和并行策略，长 prefill 也不会占住正在 decode 的 GPU。

**p. 54 的例子最值得看。** 共用一张 GPU 时，TTFT 约束允许 3 req/s，TPOT 约束只允许 1.6 req/s，所以整体受限于 **1.6 req/s/GPU**。分离后，两张 prefill GPU 合计能处理 \(2\times 5.6=11.2\) req/s，一张 decode GPU 能处理 10 req/s。整体取瓶颈 10 req/s，再除以三张卡，就是 **约 3.3 req/s/GPU**。

代价是 **KV cache 必须从 prefill 侧交给 decode 侧**。pp. 57–62 讲了怎么让对应层的 worker 尽量在同一个节点上，以及由 decode worker 拉取 KV 的设计。p. 70 讲了分离可能吃亏的情况。短请求省下的时间可能抵不过传输成本，突发或不均衡的负载也可能让一边排队、另一边空闲。p. 71 则把两条主线接起来了：先把 prefill 路由到能复用前缀的位置，再为 decode 选择有显存和延迟余量的 worker。

## 5. Sarathi-Serve 怎么在同一张 GPU 上减轻干扰（pp. 72–80）

**Chunked prefill** 把一个长 prompt 切成多个小块，按顺序处理。每轮可以同时做一小块 prefill，并让已经在生成的请求推进一步 decode。p. 76 的图里，正在生成的 A、B 可以在新请求 C、D 做 prefill 的过程中持续输出。

pp. 77–79 用 12 个 token 举例。先算 `0–3` 并保存 KV，再算 `4–7`，最后算 `8–11`。第二块里的 token 6 仍然能看到 token `0–6`，所以因果注意力保持完整。**切块改变的是 GPU 一轮做多少工作，最终仍要保存全部 12 个输入 token 的 KV。**

p. 80 的取舍是，**块大**时单轮并行工作更多，但 decode 可能等得更久。**块小**时正在生成的请求可以更频繁地推进，但会增加 kernel launch 和历史 KV 的重复读取。新请求还得等到最后一块 prefill 完成才有首 token，所以更平滑的输出可能以更长的 TTFT 为代价。

## 6. 三个系统放在一起看（pp. 81–82）

| 系统                | 它决定什么                         | 主要权衡                      |
| ----------------- | ----------------------------- | ------------------------- |
| **Preble**        | 请求发给哪个实例                      | 前缀复用省下的计算 vs. 排队时间        |
| **DistServe**     | Prefill 和 decode 分别在哪些 GPU 上跑 | 阶段隔离与独立配置 vs. KV 传输和资源闲置  |
| **Sarathi-Serve** | 共享 GPU 的每一轮放多少 prefill 工作     | 输出流畅度 vs. 首 token 延迟和切块开销 |

最后两页把这些取舍变成了一个实验思路。固定同一条请求 trace，改变前缀共享比例、队列负载、上下文长度和链路带宽，再同时看 **TTFT、TPOT、SLO 达标率、缓存命中和 KV 传输量**。这样才能看出一个调度策略究竟省了计算，还是把等待从一个阶段转移到了另一个阶段。




## 1. Prefill 和 Decode 要不要分开？｜PDF p. 2–5

**Prefill** 读入 prompt，建立 KV cache，并在处理完最后一个 prompt token 后得到生成第一个 token 所需的 logits。**Decode** 则逐步生成后续 token，每一步都要访问已有的 KV cache。长 prefill 会占用 GPU 较久，如果和正在生成的请求共用 GPU，用户可能感到输出突然卡了一下。

把两阶段放到不同 GPU 池，可以独立扩容，也能隔离这种干扰。代价是 **prefill 做完后，需要把请求的 KV cache 交给 decode GPU**。短请求省下的时间可能覆盖不了传输开销。流量忽高忽低时，也可能出现一个池排队、另一个池空闲的情况。

这和 Preble 有直接联系。假设 GPU 0 已经缓存了某份长文档的前缀，可以把 **prefill** 路由到 GPU 0 来复用缓存，再为 **decode** 选择显存和排队情况合适的 GPU。调度器现在有两个选址决策，还要保证 KV block 在传输期间和使用期间不会被回收。

## 2. Chunked Prefill｜PDF p. 6–12

Chunked prefill 让 prefill 和 decode 继续共用 GPU 池，但把一次长 prefill 拆成多个短调度步骤。PPT 用了一个 **12 个 prompt token、每块 4 个 token** 的例子。

|迭代|这一步处理什么|前面的上下文怎么用|
|---|---|---|
|1|Token 0–3|计算并保留这 4 个位置的 K/V|
|2|Token 4–7|复用 0–3 的 K/V。例如 \(q_6\) 可以看 \(k_0,\ldots,k_6\)|
|3|Token 8–11|处理完后，最后一个 prompt 位置用于预测第一个输出 token|

**切块改变的是调度粒度。** 最后还是要保留全部 12 个位置的 KV。切完以后，系统可以在同一轮把某个请求的 prefill chunk 和其他请求的 decode tokens 混合成 batch，让正在生成的请求更规律地向前推进，也给 GPU 补充并行工作。

块太大，每轮执行时间长，其他请求的 decode 容易被拖住。块太小，kernel 启动和重复读取历史 KV 的开销会变明显。对新来的长请求，**首 token 延迟（TTFT）可能上升**，因为首 token 要等最后一个 chunk 跑完。对已经在生成的请求，**token 间延迟可能更稳定**。p. 12 讲的就是这两种用户体验之间的权衡。

## 3. 长上下文、Sequence Parallelism 和 Ring Attention｜PDF p. 13–29

### 为什么要沿序列切分？｜p. 13–14

上下文长度为 \(N\) 时，KV cache 随 \(N\) 线性增长。Dense attention 的 prefill 计算量随 \(N^2\) 增长，而每次 decode 也要读取越来越长的历史。Sequence parallelism 沿 **token 维度**分配工作，例如让 4 张 GPU 各管连续四分之一的 tokens。

问题是，**GPU 3 上的 query 依然需要访问前面 GPU 上的 K/V**。切分序列之后，系统还得安排跨卡通信。

### 两趟式 Sequence Parallelism｜p. 15–19

第一趟让 **K block** 沿环传递。每张 GPU 保留自己的 Q，用它和依次收到的 K 计算 score。收齐后，每个 query 对整行可见的 key 做 **一次全局 softmax**。第二趟再传 **V block**，每张 GPU 用对应的概率块乘 V，累加出本地 query 的输出。

p. 17 最值得看。假设一个 query 在两个块上的 scores 分别是 `[10, 9]` 和 `[1, 0]`。如果每块单独做 softmax，两块内部都会得到近似 `[0.73, 0.27]`，而整行归一化时，几乎所有概率质量都会落在第一块。**分块计算仍然需要跨块归一化。**

### Ring Attention 怎么省显存？｜p. 20–29

Ring Attention 把 **K 和 V 一起传**。每收到一块，就计算这一块的 scores，并用 **online softmax** 更新本地的运行最大值、归一化分母和加权输出。全部块处理完再得到最终输出，过程中可以丢掉临时 scores。

p. 21 可以只盯着 GPU 3 上最后一个 token 的 query。它依次看到 `KV3 → KV2 → KV1 → KV0`，每来一块就更新一次累积结果。双缓冲让 GPU 在算当前块时接收下一块，所以稳态下每轮耗时可以接近 \(\max(t_{\text{compute}}, t_{\text{transfer}})\)。

- **显存**：在 slide 对比的实现里，两趟式方案保存的 score tensor 在固定 GPU 数量下随 \(N^2\) 增长，Ring Attention 的主要缓冲区随 \(N\) 线性增长。Dense attention 的总计算量仍是二次的。
- **通信**：按 p. 28 的完整环形扫描，两种方案发送的 K/V 字节数相同。Ring Attention 的收益来自边传边算，以及省掉物化完整 score 的空间。
- **扩容**：固定上下文长度时，GPU 越多，每卡块越小。计算量随块长平方下降，传输量随块长线性下降，所以继续加 GPU 可能让通信更难被覆盖。

## 4. MoE Serving 与 MegaScale-Infer｜PDF p. 31–55

### 为什么把每层的 Attention 和 Expert FFN 拆开？｜p. 31–37

在 MoE 中，每个 token 只被路由到 \(E\) 个专家中的 \(K\) 个。假设路由均匀，一个含 \(b\) 个 tokens 的 batch 中，**每个专家平均只收到 \(bK/E\) 个 tokens**。PPT 的例子是 Mixtral 的 8 个专家、top-2 路由。全局 batch 有 156 个 tokens 时，单专家平均只有 39 个。此时，读专家权重的成本很难被足够多的 token 分摊，GPU 算力容易闲置。

增大全局 batch 可以让专家吃到更多 token，但 **attention 侧必须为更多并发序列留出 KV cache 空间**。MegaScale-Infer 把两侧拆开，让多组 attention 实例向共享的专家池供给 tokens。Attention 侧可以按 KV 容量和带宽配 GPU，专家侧可以用 expert parallelism 分布专家，必要时用 tensor parallelism 拆分单个专家。专家侧有足够大的有效 batch 后，更高的计算吞吐才容易发挥出来。

### Ping-pong pipeline 和 M2N 分别解决什么？｜p. 38–47

只有一个 microbatch 时，它要先跑 attention，再把激活发给专家，算完后传回 attention 侧进入下一层。两侧因此经常轮流空闲。**Ping-pong pipeline** 把一个 batch 拆成多个 microbatches，让 attention 算 B 时，专家同时算 A，并尽量把网络传输穿插在计算期间。讲义的实验里，从 1 个增到 2 个 microbatches 带来约 \(1.9\times\) 提升，第 3 个还能填一部分空档。继续切分会减少每个 microbatch 的 tokens，收益开始变小。

**M2N** 处理的是每层两侧之间的通信。Attention 侧可能有 \(M\) 张卡，专家侧有 \(N\) 张卡，每张卡都可能按路由向对面多张卡发激活。M2N 用预注册的缓冲区、CPU 驱动的直接 RDMA 写入，以及轻量的 CUDA stream 同步来缩短传输开销。**流水线负责安排什么时候算、什么时候传，M2N 负责缩短这一趟传输。** 当网络慢、两侧算力失衡，或者流量太低导致 microbatch 太小时，流水线的收益都会缩小。

### 热专家和实验结果｜p. 48–55

如果一批 tokens 总被路由到少数专家，**最忙的专家决定这一轮多久结束**。复制热专家并分摊 token，可以缓解长期的负载偏斜。讲义中的贪心部署策略依据上一时段的流量，因此更适合持续的热点。很短的尖峰或快速变化的专家热度，很难及时跟上。

读结果时要留意指标。p. 51 的 \(7.11\times\) 是 Scaled-MoE 在讲义实验里相对 vLLM 的**每 GPU 解码吞吐**，并受 token 间延迟约束。p. 53 的 \(4.2\times\) 则是 **256 KB 消息**下 M2N 相对 NCCL 的通信吞吐，两张图测量的层级不同。

## 5. Agentix｜PDF p. 56–79

前面系统调度的是一次推理请求。Agentix 把调度范围扩大到**一个会反复调用模型的程序**。比如一个 coding agent 先读文件，再调用模型决定改动，运行测试后又调用模型诊断错误。整个任务的耗时会累积多次模型排队、生成和工具等待。执行中还可能分支，系统事先很难知道任务会调用模型多少次。

Agentix 给程序分配 **PID**，在 **Process Table** 中记录同一程序跨模型调用的信息，例如已经获得的模型服务时间和并行调用情况。这些信息影响两层决策。

|决策|Agentix 怎么做|解决什么问题|
|---|---|---|
|请求发到哪台引擎|短请求发给负载较低的实例，长请求倾向沿用同一 PID 已经使用的实例|长请求更可能复用同一程序的上下文前缀|
|引擎上谁先执行|用多级反馈队列和抢占，根据程序已经占用的服务时间调整优先级|缓解长程序和长调用造成的队首阻塞|

和上一讲的 **Preble** 连起来看，Preble 根据前缀匹配和计算负载做路由。Agentix 还利用**哪些调用属于同一个程序**这一信息来做路由和排队。比如程序 A 已经执行了很多次模型调用，程序 B 刚开始、可能只差一次调用就能完成。即使 A 更早启动，基于累计服务时间的调度也可以让 B 先推进。p. 79 报告的是特定实验里相同延迟条件下相对 vLLM 的吞吐提升，约 \(4\)–\(15\times\)。

## 6. 生产环境里的 Agent Runtime｜PDF p. 80–83

最后四页从**模型调用的调度**转到**应用状态的持久化和恢复**。AWS AgentCore 的 session 用独立 microVM 和独立内核隔离，同一个活跃 session 内可以保留本地状态。要在临时 VM 消失后恢复任务，应用需要把必要的 checkpoint 存到 VM 外。Durable workflow 还能在等待外部输入时挂起，收到结果后继续。

跟你之前问过的 **agent 修改外部服务后崩溃** 是同一个问题。假设 agent 调用日历 API 创建事件，服务端已经写入，agent 却在保存结果前崩溃。恢复后重跑调用可能创建重复事件，直接跳过又可能漏掉一次实际上没提交的操作。要判断任务是否完成，runtime 需要能核对外部结果，或者用幂等键安全重试。因此，**模型结束生成、工具请求返回、外部状态确认提交**是三个需要分别记录的时点。



