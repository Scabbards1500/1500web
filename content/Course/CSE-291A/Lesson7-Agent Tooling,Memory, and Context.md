**LLM 如何通过 tools 执行操作，以及 agent 如何管理 memory 和 context 来持续完成任务。** 前半部分讲工具调用与执行，后半部分讲状态、记忆和上下文管理。


# 从 Inference 到 Stateful Agent（第 1–4 页）

首先要区分两个角色：

- **Model**：根据当前输入，提出下一步行动。
- **Runtime**：检查请求、实际执行操作，并把结果交回模型。字面理解就是（运行时）就是“程序真正跑起来之后的那个阶段/环境

例如，你让 agent 阅读代码，模型可能输出：

```
read_file(path="src/main.py", start_line=1, max_lines=80)
```

模型生成的是 **tool name + structured arguments**。实际读取文件的是 runtime。

整个过程是：

> 用户请求 → 模型提出 tool call → runtime 执行 → tool result 进入下一次 prompt → 模型继续推理。

这里的 **“A tool call is a contract”**，指工具接口必须明确约定：

| 部分                       | 要约定什么               |
| ------------------------ | ------------------- |
| Inputs                   | 参数类型、允许访问的路径、读取大小限制 |
| Outputs                  | 文件内容、版本信息、继续读取的标识   |
| Runtime responsibilities | 权限检查、限制执行、记录结果      |

**核心理解：agent 的能力来自 model 与 runtime 的共同配合。** 模型选对工具还不够，执行环境也必须正确处理这个请求。


## Toolformer：use of Tool

工具已经存在以后，还有一个问题：

> **模型怎么知道 when to call a tool，以及怎么填写 arguments？**

PPT 用 **Toolformer** 介绍一种训练方法：

1. 给模型少量 API 调用示例。
2. 让模型在文本中生成候选 tool calls。
3. 执行调用，把结果插入文本。
4. 检查工具结果是否降低后续文本的预测 loss。
5. 保留有帮助的调用，用这些数据 fine-tune 模型。

例如，文本中需要一个计算结果，计算器返回的数值让模型更容易正确预测后面的内容，这个调用就可能被保留。

这里的训练信号是：

> **Does the tool result improve prediction of subsequent text?**

但这个信号衡量的是预测帮助，不能直接证明调用符合权限要求或安全要求。**训练负责学习调用行为，runtime 仍负责实际执行与检查。**



# MCP：统一工具接入方式（第 7–11 页）

当 agent 要接入文件系统、数据库、搜索服务等多个系统，每个服务可能都有不同的 schema、认证方式和返回格式。

**MCP（Model Context Protocol）** 用共享协议减少这些 integration 工作。

它有三个主要角色：

|角色|作用|
|---|---|
|**Host**|agent 所在的应用，决定哪些能力提供给模型|
|**Client**|host 内连接 MCP server 的组件|
|**Server**|暴露工具、资源和交互模板|

MCP 暴露的不只是 tools：

- **Tools**：可执行函数，例如查询数据库。
- **Resources**：可读取的数据，例如文件或数据库 schema。
- **Prompts**：可复用的交互模板。

执行过程大致是：

> Discover capabilities → expose selected tool schemas → model proposes a call → validate → execute → return result.

**MCP 统一通信方式；行动是否合适，仍需要 model 和 host 判断。**

第 11 页特别强调一个分布式系统问题：**timeout 不等于操作失败。**

例如，服务已经完成写入，但回复在网络中丢失。此时直接 retry，可能重复执行。支持时应使用 **operation ID + deduplication**，并单独记录真实 execution status，不能仅依靠模型说“我成功了”或“我失败了”。



# 并行工具调用与 Critical Path（第 12–17 页）

**多个 tool calls 怎么安排，才能降低 latency？**

PPT 的例子是：

- Weather：1.0 s
- arXiv headline：1.5 s
- Final synthesis：0.5 s

顺序执行：

\[ T_{\text{sequential}}=1.0+1.5+0.5=3.0\text{ s} \]

如果前两个请求相互独立，可以并行：

\[ T_{\text{parallel}}=\max(1.0,1.5)+0.5=2.0\text{ s} \]

这个例子假设两个请求已经确定，并且 planning cost 相同。

**能否并行，取决于 dependencies。** 如果第二个调用需要第一个调用的结果，就必须等待。**LLMCompiler** 在这里展示了按依赖关系规划并行工具执行的思路。

第 16 页进一步说明：一个看起来简单的 **web search tool**，内部可能包含：

> Query rewrite、search、crawl、browser rendering、chunking、embedding、reranking、deduplication、summarization。

所以，**一个 tool 内部也可能是复杂 workflow，甚至另一个 agent。**

最终 latency 由 **critical path** 决定：依赖图中耗时最长的路径。即使很多分支同时运行，一个很慢的分支仍可能拖住最终回答。并发还需要限制，以遵守 rate limits 和本地资源约束。



# Tool Results 与 Context、Safety（第 18–20 页）

工具速度快，不代表整个 agent 快。

例如，搜索工具迅速返回几十页文本，但模型接下来要处理大量 tokens，会增加 context processing 的开销，也可能难以找到相关证据。

因此，PPT 建议：

- **Pagination**：分批返回。
- **Artifact references**：保留完整内容的引用。
- **Provenance / timestamps / errors**：保留来源、时间和错误信息。
- 关注 **useful evidence per token**：每个 token 带来多少有用证据。

安全方面，这一章提出几类问题：

- 参数合法，但执行的 action 是否符合用户意图？
- server 是否真实描述了工具能力？
- 并行调用是否会 race 或重复产生效果？
- 搜索内容是否包含指令，诱导模型改变下一步行动？

**工具输出进入 prompt 后，会影响后续决策，所以需要把它当作带来源的数据处理。**



# 四种 State 与基本 Context Policies（第 21–24 页）

这一部分非常关键，尤其能接上你前面学的 KV cache。

|State|保存什么|例子|
|---|---|---|
|**Conversation history**|消息和调用记录|用户问题、tool result|
|**Application state**|外部系统的实际状态|文件、数据库、已完成操作|
|**Agent memory**|选出的事实、摘要或经验|任务约束、历史结论|
|**KV cache**|特定 token prefix 对应的模型激活|attention 的 keys / values|

它们解决的问题不同。例如，文件已经修改属于 application state；模型记住“文件已修改”属于 agent memory。两个状态可能不一致。

**扩大 context window 也有代价：**

- 更多输入 tokens 增加 **prefill work**。
- 更长上下文增加 **KV-cache memory**。
- 更多内容不保证模型更容易使用相关证据。
- 持久状态仍需要保存在单次模型调用之外。

PPT 接着介绍四种策略：

| Policy                   | 方法                | 主要取舍             |
| ------------------------ | ----------------- | ---------------- |
| **Windowing**            | 只保留最近消息           | 简单，但可能丢失早期证据     |
| **Compaction**           | 用摘要替换旧历史          | 节省 tokens，但可能丢细节 |
| **Retrieval**            | 从外部存储检索所需记录       | 输入更集中，但需要检索      |
| **Agent-managed memory** | 模型通过 tools 主动读写记忆 | 更灵活，也增加管理成本      |

尤其要记住：**summary 是有损的，因此应保留原始记录，方便恢复遗漏信息。**



# e.g MemGPT：让 Agent 主动管理记忆（第 25–30 页）

**MemGPT** 借用了操作系统 memory hierarchy 的思路，让有限 context 配合外部存储工作。

它的层次包括：

- **Main context**：当前模型能直接看到的指令、可写 working memory 和近期消息。
- **Recall storage**：由 runtime 管理的交互历史。
- **Archival storage**：通过函数访问的外部文本和文档。

最重要的一点是：

> **只有被带入当前 prompt 的信息，模型才能直接使用。**

外部存储有一条记录，不等于模型当前已经看到它。

它的 execution loop 是：

1. 新事件触发模型推理。
2. 模型决定回答、修改 working memory，或者搜索外部存储。
3. Runtime 执行函数并加入结果。
4. 如果请求 **heartbeat**，继续运行；否则 yield。

这里的 heartbeat 可以理解为“继续执行下一轮”的控制信号。

当 context 接近满时，系统先提醒模型保存重要信息；溢出后，runtime 移除部分消息并更新 recursive summary。移出的消息仍保留在 recall storage 中。

不过，**virtual memory analogy 有边界**：

- 检索未必找回正确记录。
- Summary 未必保留原意。
- Memory access 可能需要数据库工作和额外模型调用。

PPT 展示了一个历史实验：GPT-4 在早期会话检索任务上，summary baseline 为 **32.1%**，MemGPT 为 **92.5%**。这个结果说明该任务中搜索完整历史很有帮助；两者能访问的信息不同，不能把这个数值当作通用准确率保证。


# A-MEM 与 RLM：另外两种管理信息的方式（第 31–35 页）

**A-MEM** 把经历组织成带描述和 metadata 的 notes。

新信息进入后，系统会检索相关 notes，让模型提出链接，并可能更新旧记忆。你可以把它理解成：

> 从“存一段历史”，发展到“维护相互关联、可以更新的知识笔记”。

这些 links 和 updates 也需要模型调用，因此存在额外成本。错误信息还可能通过链接和更新传播。

**Recursive Language Models（RLMs）** 则把长输入放到 execution environment 中，让模型通过程序访问它：

- 读取某些 slices。
- 把选中的部分交给 sub-models。
- 根据返回 observations 继续分析。
- 最后汇总。

所以它处理长输入的方式是 **context as program data**：根模型通过操作来探索输入，而不是一次接收全文。

这也需要预算控制，包括 recursion、execution time 和 total tokens。

第 35 页把问题归结为 **More context or more retrieval?**

- 大 prompt：预先支付更多 prefill 成本。
- Selective retrieval：减少输入，但可能增加顺序模型调用。
- 频繁改变 context：可能降低 cached prefix 的复用。

因此应在相同预算下比较 **task quality + total latency**，不能只比较某一次调用的输入长度。




# Memory Consistency

**Agent memory is not the KV cache。**

- Agent memory 是可编辑、可检索的 application data。
- KV cache 是与具体模型输入对应的 activations。
- Cache reuse 需要兼容的模型状态和匹配的 prefix。
- Agent 迁移时，可以带走 durable records，而不带走 KV cache。

第 37–38 页讨论 persistent memory 的风险：

- 不可信 tool output 可能被写入长期记忆。
- 多个任务可能同时更新一条记忆，造成覆盖。
- 原始记录修正后，summary 或 derived memory 可能仍保留旧错误。

因此要保留 **provenance**，区分 observation 与 trusted instruction，并对更新做 versioning。**错误修正也必须传播到派生记忆。**

最后几页建议做可测量的实验：

| 类别               | 记录内容                        |
| ---------------- | --------------------------- |
| Model            | tokens、prefill/decode time  |
| Tools            | durations、retries           |
| Context / memory | context size、reads / writes |
| Outcome          | task success、quality        |
| Edge resources   | energy、持续资源使用               |

Baseline 可以采用 **sequential tools + recent-message window**，然后每次只改变一个 policy，例如 bounded parallelism、summarization 或 selective retrieval。

**与你的 failure propagation and recovery 项目直接相关的是：故障不仅会传播到后续 agent output，也可能进入 summary 和 persistent memory。** 恢复时，除了重算受影响的输出，还需要识别和修正它们派生出的记忆；否则后续运行可能再次取出同一个错误。