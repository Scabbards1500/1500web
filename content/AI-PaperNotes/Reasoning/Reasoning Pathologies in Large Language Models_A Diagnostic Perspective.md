
**模型推理失败时，能否判断它具体“怎么想坏了”、从哪一步开始出问题，以及在那里干预是否有效？**

# Abstract
大语言模型的长思维链（Chain-of-Thought，CoT）推理经常以一些反复出现的方式失败：模型可能在冗余推导中循环，在不确定的中间状态间游移，过早采纳缺乏依据的答案，或在已经得出有效结论后继续生成。这些失败通常被视为彼此独立的表面现象，并通过针对具体症状的启发式方法加以处理。然而，这些方法存在两个问题：（i）即使抑制了可见症状，底层推理轨迹中的故障仍未改变；（ii）缺乏统一的分析基础，难以比较不同模型或不同推理异常的修复方法。

我们转而从 **CoT 推理的潜在状态转移**层面研究这些失败。我们提出了一个诊断框架，将思维链推理抽象为经过一系列离散潜在推理状态的轨迹，并据此按照异常在推理过程中所处的位置，将其划分为六类。每类异常均配有一个基于状态转移的检测判定条件，以及一个与之匹配的推理时控制器。我们通过受控的轨迹修改实验，在多个数据集和模型上评估该框架，涵盖算术推理、多跳问答和知识密集型推理。

实验发现，推理异常呈现出三种主要的诊断规律：
（1）异常具有按推理阶段组织的结构：终止阶段的异常最容易被可靠识别，早期分支异常也能够被检测，而推理推进阶段的异常则需要更细致、考虑图结构的分析；
（2）对比状态转移变化与表面表达变化的反事实测试表明，多个检测器追踪的是潜在轨迹的变化，而非单纯的文字改写，同时也揭示了某些释义改写会改变推理路径本身；
（ 3）当检测到的异常具有明确的局部起点时，根据检测时机实施干预最为有效，但这类干预并非始终优于通用控制方法。

这些发现共同表明，该框架可为研究 CoT 推理失败提供一个可复现的诊断基础，明确现有证据所能支持的结论边界，并为更强的闭环控制提供支撑。



# Background
模型生成很长的 Chain-of-Thought（CoT，思维链），并不代表它一直在取得进展。

比如一道题：

> 原来有 12 个苹果，卖掉 5 个，又买进 3 个，现在有多少个？

正常过程是：

> 先算剩余：12−5=7。再算买进：7+3=10。所以答案是 10。

但模型可能出现不同的失败：

- 一开始就误读成“买进 5 个”。
- 同时纠结加法、减法、比例等多种不相干策略。
- 不断重复计算同一个中间结果。
- 一直说“需要进一步考虑”，却没有解决任何新问题。
- 没处理完条件就直接给答案。
- 已经算出 10，又继续怀疑，最后改成错误答案。

这些情况最后可能都表现为“答案错误”或者“输出太长”，但**失败机制不同，需要采取的措施也不同**。

例如，限制输出长度可能缓解最后一种情况，却可能让“过早下结论”更严重。

所以作者认为，只看答案、token 数量、文字重复率是不够的。需要描述：

> 推理过程处于什么状态？它如何从一个状态走向下一个状态？异常转移发生在哪里？


# Method

## Section1： text →  discrete state trajectory
spectral embedding 负责“用向量概括当前 prefix”，clustering 负责“把相似的 prefix representations 归类”，transition analysis 再研究“这些类别之间怎么走”。

模型输出的是一系列文字形式的 reasoning steps：

$$
x_1,x_2,\ldots,x_T
$$

其中，$x_t$ 表示第 $t$ 个 reasoning step。

作者希望将它们映射为一条 discrete state trajectory：

$$
s_1,s_2,\ldots,s_T
$$

其中，$s_t$ 是一个 discrete state 的编号。例如：

$$
1\rightarrow 2\rightarrow 3\rightarrow 3\rightarrow 4\rightarrow 7
$$

通过分析这条 state trajectory，可以判断推理是否出现循环、停滞，或过早进入 terminal states。

需要注意，“state 3”并不是人工标注的“正在做减法”。这些 states 来自对模型内部 representations 的 clustering，是一种统计抽象。

### 1. 提取每一步的 hidden representations

假设第 $t$ 个 reasoning step 包含 $n_t$ 个 tokens，每个 token 的 hidden vector 维度为 $d$，则得到：

$$
H_t\in\mathbb{R}^{n_t\times d}
$$

主实验使用<span style="color: #ff9933;">最后一层 Transformer 的 hidden states</span>。

### 2. 累积整个 prefix 的信息

<span style="color: #33cc33;">注意，这里的prefix 是以reasoning step 为单位的，而非token。 </span> 但不是把这些向量直接拼起来，而是把每一步的 $(H_r^\top H_r)$ 累加成一个固定大小的矩阵$G_t\in\mathbb{R}^{d\times d}$

作者构造 cumulative Gram matrix：

$$
G_t=\sum_{r\leq t}H_r^\top H_r
$$

其中，$H_r^\top H_r$ 描述 hidden features 之间的二阶关系；累加后得到截至当前 reasoning step 的整体 representation 结构。直观来说，它不仅关注“当前这句话说了什么”，还试图概括“此前整个 reasoning prefix 形成了什么表示”。


随后，提取 $G_t$ 的 top-$k$ eigenvalues 和对应的<span style="color: #33cc33;"> eigenvectors(eigenvectors（特征向量）用来找主要方向，eigenvalues（特征值）用来衡量这些方向有多强。)</span>：

$$
(\lambda_{t,1},q_{t,1}),\ldots,(\lambda_{t,k},q_{t,k})
$$

将它们拼接为 prefix-aware step embedding, 用几个最主要的 representation directions，压缩截至当前的 reasoning prefix： 

$$
e_t=
\left[
\sqrt{\lambda_{t,1}}q_{t,1};
\ldots;
\sqrt{\lambda_{t,k}}q_{t,k}
\right]
\in\mathbb{R}^{kd}
$$

这里的小写 $k$ 表示保留的 spectral directions 数量。

<span style="color: #66ccff;">可以这么理解: hidden states 是一大堆点；eigenvectors 找出它们主要展开的方向；eigenvalues 衡量各方向的能量；作者把最强的几个方向及其强度打包，作为当前推理前缀的“结构摘要”。</span>

### 3. 聚类得到 discrete states

把连续的高维 representation，转成少量可重复识别的 discrete states，方便分析推理如何在状态之间移动。

对这些 embeddings $e_t$ 进行 K-means clustering：

$$
s_t=\kappa(e_t)\in\{1,\ldots,K\}
$$

其中，$\kappa$ 将 embedding 映射到最近的 cluster，大写 $K$ 表示 state clusters 的数量。

主实验默认使用 $K=7$，部分专项实验采用其他设置。

e.g 
原本一连串向量：$e_1,e_2,e_3,e_4,e_5,e_6 \quad\longrightarrow\quad 1,2,3,2,3,2$
这时就更容易看出：**后半段在 state 2 和 state 3 之间反复切换。**

### 4. 统计 state transitions

定义 transition matrix：
这样才能统计：

$$
P_{ij}= \frac{\text{从 state }i\text{ 到 state }j\text{ 的次数}} {\text{从 state }i\text{ 出发的总次数}}=\Pr(s_{t+1}=j\mid s_t=i) 
$$

例如：

$$
P_{3,3}=0.9
$$

表示当前处于 state 3 时，下一步仍然处于 state 3 的概率为 0.9。

不过，较高的 self-transition probability 本身并不意味着推理错误。复杂问题可能确实需要在同一类 state 中进行多步计算，因此还需要结合 reasoning phase、escape probability 等信息判断。

此外，这只是一个 first-order state-transition abstraction，并不意味着作者已经证明真实的模型推理完全满足 Markov property。



## Section2: Problem Formulation

Section 3 将“模型推理出了问题”形式化为可以检测、定位和比较的 diagnostic problem。

### 1. Diagnostic Objectives

诊断不仅要判断最终答案是否正确，还需要回答：

- 出现了哪一种 reasoning pathology？
- 最早在哪一步能够检测到它？

对每种 pathology $\ell\in\mathcal L$，定义一个 detection predicate：

$$
\Psi_\ell:[K]^*\rightarrow\{0,1\}
$$

其中：

- $[K]=\{1,\ldots,K\}$ 是所有 discrete state IDs 的集合。
- $[K]^*$ 表示由这些 IDs 构成的任意有限长度序列。
- $\Psi_\ell=1$ 表示触发该 pathology 的检测。
- $\Psi_\ell=0$ 表示尚未触发。

在第 $t$ 步，detector 根据当前已观察到的 state prefix 进行判断：

$$
\Psi_\ell(s_{1:t})
$$

例如，观察到：

$$
s_{1:6}=(1,2,3,2,3,2)
$$

loop detector 可能发现 trajectory 在 states 2 和 3 之间反复切换，从而触发检测。具体 detection conditions 在 Section 4 中定义。

#### Detection onset

作者定义首次触发检测的时间为：

$$
\tau_\ell
=
\min\{t\in[T]:\Psi_\ell(s_{1:t})=1\}
$$

如果始终没有触发，则：

$$
\tau_\ell=+\infty
$$

需要区分 detection onset 与异常真实发生的时间。例如，循环可能从第 3 步开始，但 detector 需要观察多次重复，直到第 6 步才触发。

各 pathology 的 predicates 分别定义，因此这个 formulation 并不要求不同异常互斥：同一条 trajectory 可以触发多种异常。

上述记号省略了部分 auxiliary diagnostics(除了 discrete state trajectory 之外的辅助诊断信号或指标)；具体 detector 还可能使用 <span style="color: #33cc33;">hidden representations 或 attention signals</span>。


#### Prevention objective

对应的 prevention problem 是设计 inference-time controller，在保持 answer quality、避免不必要 reasoning length 的同时，降低 pathology 的发生频率或严重程度。

### 2. Reasoning Phase Structure

相似的行为出现在不同 reasoning phases，可能具有不同含义。例如，探索其他解法在开头可能合理，但在答案已经准备好后继续探索，可能构成 terminal leakage。

作者将 state space 划分为三个互不重叠的 phase sets：

$$
[K]
=
S^{(I)}
\;\dot\cup\;
S^{(P)}
\;\dot\cup\;
S^{(T)}
$$

| Phase set | Phase | 预期功能 |
|---|---|---|
| $S^{(I)}$ | Initialisation | 理解问题、建立问题框架 |
| $S^{(P)}$ | Progression | 计算、推导、检索和验证 |
| $S^{(T)}$ | Termination | 综合并提交答案 |

#### Unsupervised phase assignment

Phase assignment 基于 reference corpus 中各 cluster 出现的平均 relative step position。

对于长度为 $T$ 的 trajectory，第 $t$ 步的 relative position 为：

$$
\frac{t}{T}
$$

作者统计每个 cluster 的平均 relative position，按其排序，再将 clusters 分配到 early、middle、late phases。

这里划分的是 state clusters，并不是把每条新 trajectory 按长度机械地切成三段。因此，一条 trajectory 可以出现：

$$
I\rightarrow P\rightarrow T\rightarrow P
$$

其中，$T\rightarrow P$ 表示进入 Termination 后又返回 Progression，是后续分析 terminal leakage 的一个线索。

这些 phase labels 是根据位置归纳得到的 proxy labels，并不保证与真实 reasoning function 完全一致。

### 3. Hitting Time

对任意 state set $A\subseteq[K]$，定义：

$$
\tau_A
=
\min\{t\in[T]:s_t\in A\}
$$

Hitting time 表示 trajectory 第一次进入集合 $A$ 的时间。如果始终没有进入，则 $\tau_A=+\infty$。

例如，假设：

$$
S^{(I)}=\{1,2\},\qquad
S^{(P)}=\{3,4,5\},\qquad
S^{(T)}=\{6,7\}
$$

对于 trajectory：

$$
1\rightarrow2\rightarrow3\rightarrow4\rightarrow6\rightarrow3\rightarrow7
$$

第一次进入 terminal set 的时间为：

$$
\tau_{S^{(T)}}=5
$$

虽然第 6 步又退出 terminal set，但首次进入的时间仍然是第 5 步。

### 4. Occupancy

Hitting time 只描述第一次进入的时间，不能区分短暂访问与长期停留。因此，作者定义 occupancy：

$$
\operatorname{Occ}_T(A)
=
\frac{1}{T}
\sum_{t=1}^{T}\mathbf{1}[s_t\in A]
$$

其中，$\mathbf{1}[\cdot]$ 是 indicator function：条件成立时为 1，否则为 0。

对于上述 trajectory，Progression states 出现在第 3、4、6 步，因此：

$$
\operatorname{Occ}_7(S^{(P)})=\frac{3}{7}
$$

Termination states 出现在第 5、7 步，因此：

$$
\operatorname{Occ}_7(S^{(T)})=\frac{2}{7}
$$

Occupancy 衡量的是 reasoning steps 的占比，不是实际耗时或 token 数量。也可以只针对某个 subinterval 计算，用于分析局部停留行为。

### 5. 这些定义如何支持后续诊断？

| Quantity                            | 回答的问题                          |
| ----------------------------------- | ------------------------------ |
| Detection predicate $\Psi_\ell$     | 当前是否触发某种 pathology 的检测？        |
| Detection onset $\tau_\ell$         | 最早何时触发检测？                      |
| Phase sets                          | 当前 state 属于哪个 reasoning phase？ |
| Hitting time $\tau_A$               | 第一次何时进入某组 states？              |
| Occupancy $\operatorname{Occ}_T(A)$ | 有多少比例的 steps 停留在这组 states？     |



## Section3:  Framework—Transition-Grounded Diagnosis and Intervention

> **每种异常具体看什么指标？指标满足什么条件时，detector 才触发？

把 Section 2 的通用定义，具体变成六类 pathology 的检测方法。按三个 reasoning phases 展开，每个 phase 对应两类异常。

三个 phase subsections 都遵循同一个叙事模板：该 phase 应完成什么 → 可能出现哪些 failures → 定义相关 quantities → 给出 detection conditions。

| Phase          | Pathology                      | 核心 detection signals                               |
| -------------- | ------------------------------ | -------------------------------------------------- |
| Initialisation | I-A: State hallucination       | Prompt–step distance、attention-graph statistics    |
| Initialisation | I-B: Divergent branching       | Early transition entropy / state diversity         |
| Progression    | II-A: Circular reasoning loops | High component occupancy + low escape mass         |
| Progression    | II-B: Entropic stagnation      | Entropy non-cooling + increasing reference drift   |
| Termination    | III-A: Premature convergence   | Early terminal hitting + low progression occupancy |
| Termination    | III-B: Terminal leakage        | Terminal exits / excess continuation               |
|                |                                |                                                    |


###  Phase I: Initialisation Failures

这一阶段检查：模型是否在推理开头建立了 grounded problem frame，以及是否形成稳定的推理路线。

#### I-A: State Hallucination

State hallucination 指初始 reasoning state 与问题不匹配。例如，题目要求计算利润，模型却按收入建立问题框架。

作者使用 prompt–step alignment 和 attention-graph diagnostics 进行检测。

**Prompt–step alignment**

定义当前 prefix-aware representation (embedding) $e_t$ 与 prompt representation $e_0$ 的距离：

$$
J_t=\|e_t-e_0\|
$$

当：

$$
J_t>\delta_{\mathrm{emb}}
$$

表示 representation distance 超过校准阈值，可能存在初始失配。
(论文中的 thresholds 会针对不同的 dataset–model setting，在 held-out validation data 上进行 calibration，并不是统一使用一个固定值。)

**Attention-graph diagnostics**

作者将当前 step 的 attention matrix 对称化，构造 undirected weighted graph：

$$
\bar W_t=\frac12(W_t+W_t^\top)
$$

定义 degree matrix 和 graph Laplacian（图拉普拉斯矩阵）：

$$
D_t=\operatorname{diag}(\bar W_t\mathbf1)
$$

$$
L_t=D_t-\bar W_t
$$

随后检查两个 spectral quantities：

| Quantity | 含义 | 异常方向 |
|---|---|---|
| Fiedler value $\lambda_2(L_t)$ | Attention graph 的 connectivity | 低于阈值 |
| High-frequency energy ratio，HFER | Attention-degree signal 在 graph 高频模式上的能量比例 | 高于阈值 |

这些 quantities 是辅助诊断信号，并不是模型理解错误的直接证明。


**Matched intervention**: 
Active context refinement：重新呈现 query 和相关 context，帮助模型重新 grounding。


#### I-B: Divergent Branching

Divergent branching 指开头出现过度分散、互不协调的策略切换，而非形成稳定的解决路线。

作者通过 early transition entropy 描述后续 state 的分散程度：

$$
\mathcal H_t
=
-\sum_jP_{s_tj}\log P_{s_tj}
$$

例如：

$$
P_{i,\cdot}=(0.95,0.05,0,\ldots)
$$

表示后续走向集中，entropy 较低；而：

$$
P_{i,\cdot}=(0.25,0.25,0.25,0.25,\ldots)
$$

表示后续走向分散，entropy 较高。

理论检测条件是早期 transition entropy 超过阈值：

$$
\mathcal H_t>\delta_H
$$

具体实现和实验还使用 early state diversity 等 proxy readouts。

这里的 entropy 是 state-transition entropy，不是 next-token entropy。较高 diversity 本身也不必然代表异常，需要结合早期位置和具体构造判断。


**Matched intervention**
Structured decomposition：注入明确的任务分解，帮助模型形成一条稳定的推理计划。

### Phase II: Progression Pathologies

这一阶段检查：模型持续生成 reasoning steps 时，是否真正取得了进展。

作者区分 compact recurrence 与 diffuse non-cooling：前者是在小范围内循环，后者是持续发散却没有收敛。

#### II-A: Circular Reasoning Loops

Circular reasoning loops 不要求文字完全重复。模型可能换着措辞表达相同内容，或在相互依赖的论点之间循环。

作者在 transition graph 中寻找 strongly connected components，简称 SCCs：component 内的 states 相互可达。

SCC 本身不构成异常，还需要同时观察：

- High occupancy：trajectory 长时间停留在该 component。
- Low escape mass：离开该 component 的概率较低。

对 component $A$，定义内部 occupancy weights：

$$
\omega_i
=
\frac{\sum_{t=1}^{T}\mathbf1[s_t=i]}
{\sum_{r\in A}\sum_{t=1}^{T}\mathbf1[s_t=r]},
\qquad i\in A
$$

对应 escape mass 为：

$$
\operatorname{Esc}(A)
=
\sum_{i\in A}\omega_i
\sum_{j\notin A}P_{ij}
$$

检测条件可以概括为：

$$
\operatorname{Occ}_T(A)>\delta_{\mathrm{scc}}
\quad\land\quad
\operatorname{Esc}(A)<\epsilon_{\mathrm{loop}}
$$

即：长期困在同一组 states 中，而且很少能离开。

**Matched intervention**

Recurrence-based loop escape：降低可能继续维持当前循环的生成选项的概率，引导 trajectory 离开 recurrent component。

#### II-B: Entropic Stagnation

Entropic stagnation 指推理持续探索，但 uncertainty 没有降低，也没有形成稳定进展。

它与循环的区别是：trajectory 不一定困在固定几个 states 中，而可能在更多 states 间游移。

作者结合 entropy non-cooling 和 reference drift 进行检测。

**Entropy non-cooling**

定义 local transition entropy：

$$
\mathcal H_t
=
-\sum_jP_{s_tj}\log P_{s_tj}
$$

以及 discrete entropy gradient：

$$
\nabla_t\mathcal H_t
=
\mathcal H_t-\mathcal H_{t-1}
$$

如果局部窗口中 entropy 持续不下降，即近似满足：

$$
\nabla_t\mathcal H_t\geq-\epsilon_H
$$

则表示 transition uncertainty 没有明显冷却。

**Reference drift**

从最近一段 state transitions 估计 rolling transition matrix $P_t$，并与高质量参考 trajectories 的 transition matrix $P^\star$ 比较：

$$
D_{\mathrm{KL}}(P_t\|P^\star)
=
\sum_i\hat\nu_t(i)
\sum_j(P_t)_{ij}
\log\frac{(P_t)_{ij}}{P^\star_{ij}}
$$

其中，$\hat\nu_t(i)$ 是 rolling window 内 state $i$ 的 empirical occupancy。

当 entropy 不冷却，同时 divergence 在窗口中上升，就可能触发 stagnation detector。

这并不要求正常推理的 entropy 始终单调下降，而是检测特定局部模式。

**Matched intervention**

Cooling scaffold / surprise-gated re-grounding：

- 总结已经确定的进展。
- 指出一个 unresolved subgoal。
- 明确一个 next step。
- 避免无关 branches；必要时重新 grounding。

### Phase III: Termination Failures

这一阶段检查：模型是否在合适的时间提交答案，以及进入 Termination 后是否保持稳定。

#### III-A: Premature Convergence

Premature convergence 指模型尚未完成必要 progression，就进入 terminal states。

作者结合两个 quantities：

- Terminal hitting time $\tau_{\mathcal T}$。
- Progression occupancy $\operatorname{Occ}_T(S^{(P)})$。

检测条件为：

$$
\tau_{\mathcal T}<\delta_\tau
\quad\land\quad
\operatorname{Occ}_T(S^{(P)})<\delta_{\mathrm{occ}}
$$

即：进入 terminal set 太早，同时 progression 占比过低。

例如：

$$
I\rightarrow I\rightarrow T
$$

可能比下面的 trajectory 更符合 premature convergence：

$$
I\rightarrow P\rightarrow P\rightarrow P\rightarrow T
$$

但短推理不必然是异常，具体判断依赖阈值校准和任务特征。

**Expected terminal hitting time**

令：

$$
\mathcal T=S^{(T)},\qquad
\mathcal N=[K]\setminus\mathcal T,\qquad
Q=P_{\mathcal N,\mathcal N}
$$

当 spectral radius $\rho(Q)<1$ 时，可以使用 absorbing-chain fundamental matrix：

$$
N=(I-Q)^{-1}=\sum_{m\geq0}Q^m
$$

计算从 non-terminal initial distribution $\nu_0$ 出发的 expected terminal hitting time：

$$
\mathbb E[\tau_{\mathcal T}]
=
\nu_0^\top N\mathbf1
$$

若 empirical estimate 不满足该条件，则使用观察到的 finite-horizon hitting time 和 occupancy。

**Matched intervention**

Phase-consistent control：鼓励补足必要 progression，避免跳过推导直接提交答案。

#### III-B: Terminal Leakage

Terminal leakage 指进入 terminal states 后，又返回 non-terminal states，或继续生成过多内容。

例如：

$$
I\rightarrow P\rightarrow T\rightarrow P\rightarrow T
$$

作者使用三类 signals：

| Signal                                     | 含义                                                |
| ------------------------------------------ | ------------------------------------------------- |
| Terminal exit mass $\epsilon_{\mathcal T}$ | 从 terminal states 转向 non-terminal states 的最大行概率质量 |
| Realised terminal-exit count $E_T$         | 实际 trajectory 中退出 terminal set 的次数                |
| Post-terminal excess $OT$                  | 首次进入 terminal set 后继续生成的 steps 数量                 |
|                                            |                                                   |

**Terminal exit mass**

$$
\epsilon_{\mathcal T}
=
\max_{i\in\mathcal T}
\sum_{j\in\mathcal N}P_{ij}
$$

数值越大，说明 terminal set 越难表现为稳定的 absorbing set。

**Post-terminal excess**

$$
OT=(T-\tau_{\mathcal T})^+
$$

其中，$(a)^+=\max(a,0)$。如果从未进入 terminal set，该项按论文约定设为 0。

Detector 在 terminal exit mass、实际 exit count 或 post-terminal excess 超过相应阈值时触发。

这一诊断依赖 terminal set 的有效性：进入某个 cluster 并不天然证明模型已经具备正确答案。

**Matched intervention**

Terminal-barrier decoding：抑制 terminal-to-non-terminal transitions，使 terminal set 更接近 absorbing set。直接 stopping 或 truncation 也是自然的 comparator。


# Experiment

Settings

|维度|设置|
|---|---|
|Datasets|GSM8K-hard、HotpotQA、GPQA|
|任务类型|算术推理、多跳问答、知识密集型推理|
|Models|按论文命名：Qwen-9B、Gemma-4、Ministral-8B|
|组合数量|\(3\times3=9\) 个 dataset–model settings|
|Natural trajectories|每个组合采样 1,000 条 CoT；GPQA 为 546 条|
|State abstraction|默认 \(K=7\)，部分专项分析调整|
|Thresholds|在各 setting 的 held-out validation slice 上校准|


他们主要通过“人为构造异常轨迹，再看 detector 能否识别”来验证方法

> **controlled surgery**：先取得一条已有 trajectory，在指定位置做针对性的修改，知道自己加入了哪类异常、位置在哪里，再检验 detector 的反应。

因此，这里的实验标签主要来自**修改操作的设计**，而不是专家对全部自然推理步骤逐一裁决。

这里我们**构造与 pathology 定义相对应的 trajectory perturbation。** 

|阶段|可以构造的修改|想检验的异常|
|---|---|---|
|Initialisation|换成不匹配的开头、混入不同路线的开头片段|Grounding failure、branching|
|Progression|重放片段、构造状态回返或 non-cooling plateau|Loops、stagnation|
|Termination|提前放入 terminal block；在 answer-ready 后加入继续探索|Premature convergence、leakage|
