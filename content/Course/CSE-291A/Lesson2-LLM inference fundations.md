理解推理成本从哪里来，以及延迟、吞吐量和资源利用率之间如何权衡

- **Transformer 与自回归生成**：回顾 Attention，解释模型如何根据输入和已有输出，逐个生成 token。
- **Prefill、Decode 和 KV Cache**：Prefill 一次处理输入；Decode 逐步生成；KV Cache 保存历史 token 的 K/V，减少重复计算。
- **性能指标与瓶颈**：首 token 延迟（TTFT）、后续 token 耗时（TPOT）、吞吐量。Prefill 通常更受算力限制，Decode 在小 batch 下通常更受显存带宽限制。
- **批处理与 Orca 调度**：把多个请求一起计算；每轮移除已完成请求、加入新请求，减少空闲位置，提高 GPU 利用率。
- **调度本身也有成本**：CPU 每轮要选择请求、准备数据、处理输出；GPU 越快，这些 CPU 开销越容易成为瓶颈。
