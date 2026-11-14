time: 2026/10/28
这篇论文 《OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents》 是 UC San Diego 的 Yiying Zhang 团队发表在 MLSys 2026 的工作。

它的研究方向属于 AI Agents + Computer Systems，主要关注 Computer-Use Agent（CUA）的执行效率。

## 1. Motivation：为什么要做这个研究？

目前的 Computer-Use Agent，例如 Agent S2、UI-TARS，可以通过截图、鼠标和键盘操作电脑。

传统 Benchmark（例如 OSWorld）主要评估：

Agent 能不能成功完成任务？（Success Rate）

但作者发现了另一个问题：

即使 Agent 成功完成任务，它也可能比人类慢很多。

例如：

- 人类修改 Word 文档的行间距：不到 30 秒。
    
- AI Agent 完成同样的任务：可能需要 12 分钟。
    

原因是 Agent 通常采用循环执行：

每一步都可能需要调用 LLM，导致：

1. LLM 推理延迟很高。
    
2. Agent 执行大量不必要的操作。
    
3. 出错后反复尝试，甚至陷入死循环。
    

因此，作者提出一个问题：

> How can we systematically evaluate the efficiency of computer-use agents, beyond task success rates?

## 2. Method：论文做了什么？

这篇论文主要包含三个部分。

### 第一部分：分析 Agent 的延迟来自哪里

作者研究了两个 Computer-Use Agent：

- Agent S2
    
- GTA1
    

在 39 个 OSWorld 任务上进行详细的延迟分析。

发现：

- Planning 是最大的延迟来源，占总执行时间的 50% 以上，有时接近 75%。
    
- Reflection / Judging 是第二大开销。
    
- 随着执行步骤增加，后续步骤的耗时可能达到初始步骤的 3 倍。
    
- 错误的 Visual Grounding（例如点击错误坐标）会导致大量重复操作。
    

因此，真正的瓶颈不一定是鼠标点击本身，而是背后的 LLM 推理与反复决策。

### 第二部分：构建 OSWorld-Human Benchmark

这是论文最重要的贡献。

作者为 OSWorld 的全部 369 个任务人工标注了经过验证的人类操作路径。

他们设计了两种轨迹：

|类型|含义|示例|
|---|---|---|
|Single-Action|每个操作单独算一步|点击 → 输入 → Enter，共 3 步|
|Grouped-Action|可以连续执行的操作合并|点击 + 输入 + Enter，共 1 组|

为什么要进行 Action Grouping？

因为 Agent 不一定每点击一次鼠标就需要重新调用 LLM。

如果多个操作可以基于同一张截图完成，就可以直接批量执行。

这样能够显著减少 LLM 调用次数。

### 第三部分：提出新的效率指标 WES

作者提出 Weighted Efficiency Score（WES），同时考虑：

- Task Success Rate：任务是否完成。
    
- Step Efficiency：完成任务用了多少步。
    
- Failed Task Penalty：失败任务浪费了多少步骤。
    

核心思想是：

\[ \mathrm{Efficiency}=\frac{\text{Human Steps}}{\text{Agent Steps}} \]

例如：

人类只需要 5 步，Agent 却执行了 20 步。

那么：

\[ \mathrm{Efficiency}=\frac{5}{20}=0.25 \]

也就是说，即使 Agent 成功完成任务，其步骤效率也只有 25%。

实际 WES 还会结合任务成功率和失败任务的额外惩罚。

## 3. Experiments：实验发现了什么？

作者评估了 16 个 Computer-Use Agents。

其中表现最好的 Agent S2（Gemini 2.5）：

|Metric|Score|
|---|---|
|原始 OSWorld Success Rate|41.4%|
|Single-Action WES|15.6%|
|Grouped-Action WES|9.6%|

这说明：

Agent 的成功率虽然达到 41.4%，但考虑执行效率后，其综合表现明显下降。

论文报告，领先 Agent 的操作轨迹仍存在约 2.7–4.3 倍的效率差距。

作者认为，未来可以通过以下方式优化：

1. Action Grouping：把连续操作合并，减少模型调用。
    
2. Efficient Rollback：出错后快速回退，避免重复尝试。
    
3. Better Grounding：提高屏幕元素定位的准确性。
    
4. LLM Inference Optimization：优化推理延迟和上下文缓存。
    

## 4. 我对这篇论文的理解

我觉得这篇论文的定位非常清楚：

它主要是一篇 Benchmark + Measurement Paper，而不是提出全新 Agent 架构的 Method Paper。

它最重要的贡献是：

- 发现并量化 Computer-Use Agent 的执行效率问题。
    
- 构建人工验证的参考轨迹。
    
- 提出衡量成功率与步骤效率的新指标。
    
- 为未来 Agent Runtime Optimization 提供研究方向。
    

不过，它也存在一定局限：

- WES 主要使用步骤数量作为效率代理，而不直接测量所有 Agent 的真实端到端运行时间。
    
- 人类标注的轨迹不一定是全局最优路径。
    
- Action Grouping 在动态 GUI 中可能引入额外风险，因为界面状态随时可能变化。
    

### 一句话英文总结

> This paper introduces OSWorld-Human, a benchmark for evaluating the efficiency of computer-use agents. It identifies LLM-based planning and redundant actions as major latency bottlenecks, and proposes human-annotated trajectories and a weighted efficiency metric to assess agent performance beyond task success rates.

结合你 CSE 291A 的研究方向来看，这篇论文最值得关注的是：它把 Agent 的研究目标从单纯提高 Task Success Rate，扩展到 Success–Latency–Efficiency Trade-off。这也是 Agent Systems 领域一个很有价值的研究方向。