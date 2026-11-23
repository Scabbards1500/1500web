这节课讲的是 **Post-Training：怎样把“会续写文本的模型”，训练成“会按要求回答的助手”**。

![[Pasted image 20261116161315.png]]

![[Pasted image 20261116164130.png]]




## 1. Instruction Finetuning：让模型学会当助手（p5-10)
用“指令＋回答”的数据，让模型学会遵循指令。

 **为什么 Pretraining 不够？**
 
Pretraining 的目标是 **Next-Token Prediction**，也就是预测下一个 token。

例如输入：

> What is the capital of France?

Base Model 可能继续写：

> What is the capital of Germany?

因为在网上，连续出现多个问题也很合理。但用户想要的是：

> Paris.

**会预测文本，不等于会遵循指令。**

### 1.1 Supervised Finetuning（SFT）
![[Pasted image 20261116154026.png]]

训练方式——用标注好的输入和输出进行监督训练。
**Supervised Finetuning（SFT）** 收集大量这样的训练对：

| Instruction | Output |
|---|---|
| 把这句话翻译成英文 | 对应的英文翻译 |
| 总结这篇文章 | 一段摘要 |
| 法国的首都是哪里？ | Paris. |

然后继续训练模型，让它在看到 Instruction 后，生成对应的 Output。

损失函数仍然是 Language Modeling Loss：

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=
-\sum_{t\in \mathrm{output}}
\log p_\theta(y_t\mid x,y_{<t})
$$

其中：

- $x$：Instruction。
- $y_t$：标准答案的第 $t$ 个 token。
- $y_{<t}$：此前的答案 tokens。

**重点：只在 Output tokens 上计算 loss。** Instruction 是条件，不是这里要求模型预测的目标。

### 1.2 为什么少量数据也可能有效？

- **Alpaca**：使用约 52K 条指令示例。
- **LIMA**：使用 1,000 条精心挑选的示例。

Pretraining 已经让模型学到大量语言知识，SFT 再通过示范，教它如何把这些能力用于回答用户。

但数据质量很重要。如果大量示范要求模型回答它原本不熟悉的事实，训练可能鼓励它在不确定时也给出自信答案。

**模仿正确答案，不自动等于学会判断自己何时不知道。**

### 1.3 SFT Limitation

主要有三个：

1. **标准答案贵**：需要有人写出完整回答。
2. **很多任务没有唯一答案**：比如写故事，存在许多好答案。
3. **Token Loss 不直接衡量整体质量**：合理的同义表达也可能因为不同于参考答案而被惩罚。

因此，下一个问题是：能否不要求模型逐字模仿，而是直接奖励“更好的回答”？


## 2. Learning from Reward：Reinforcement Learning (p16-21)
对应 PPT 第 16–21 页。


### 2.1 把语言模型看作 Policy

在 **Reinforcement Learning（RL）** 里，选择动作的规则叫 **Policy**。

对语言模型来说：

| RL 概念               | 语言模型中的含义             |
| ------------------- | -------------------- |
| State               | Prompt 和已经生成的 tokens |
| Action              | 生成下一个 token          |
| Policy $\pi_\theta$ | 模型选择 token 的概率分布     |
| Reward              | 对回答质量的评价             |

模型先生成回答：

$$
y\sim\pi_\theta(\cdot\mid x)
$$

然后得到奖励 $r(x,y)$。

最直接的目标是：

$$
\max_\theta
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
[r(x,y)]
$$

意思是：**让模型生成的回答，平均得分越来越高。**
实际训练时，还会对多个 Prompts 取平均。

### 2.2  **Reward Hacking**

Reward 往往来自一个学出来的 **Reward Model**，它对人类偏好的判断并不完美。

假设它误以为：

> 回答越长，质量越好。

模型找到了评分机制的漏洞,就可能学会不断写长答案，即使内容重复。


PPT 展示的现象是：
- Reward Model 给出的分数持续提高。
- 人类实际偏好先提高，后来反而下降。

### 2.3 用 KL Penalty 限制偏移

因此，目标变成：

![[Pasted image 20261116155432.png]]

这里 $p_{\mathrm{base}}$ 是**冻结的初始 SFT 模型**。

两个部分分别负责：

- **Reward**：鼓励更好的回答。
- **KL Penalty**：避免偏离原来的模型太远。

$\beta$ 控制两者的平衡：

| $\beta$ | 含义 |
|---|---|
| 大 | 更保守，靠近初始模型 |
| 小 | 更积极地追求奖励 |
**KL 比较的是概率分布，不是要求每次输出相同的文字。**

## 3. Policy Gradients：奖励怎样变成参数更新？

对应 PPT 第 22–43 页。

这一章主要解决三个问题：

1. 如何根据 Reward 计算梯度？
2. 如何减少梯度估计的噪声？
3. 如何复用样本并稳定训练？

### 3.1 REINFORCE：用奖励加权模型自己的回答

暂时把 Reward 看作一个固定的评分函数：

$$
J(\theta)
=
\mathbb E_{y\sim\pi_\theta}[r(y)]
$$

困难在于：生成 token 是离散采样，不能直接穿过“抽到了哪个词”进行普通反向传播。

REINFORCE 使用 **Score-Function Identity**：

$$
\nabla_\theta\pi_\theta(y)
=
\pi_\theta(y)\nabla_\theta\log\pi_\theta(y)
$$

推导过程是：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=
\sum_y r(y)\nabla_\theta\pi_\theta(y)\\
&=
\sum_y
\pi_\theta(y)r(y)
\nabla_\theta\log\pi_\theta(y)\\
&=
\mathbb E_{y\sim\pi_\theta}
\left[
r(y)\nabla_\theta\log\pi_\theta(y)
\right]
\end{aligned}
$$

于是，可以采样 $m$ 个回答，用平均值估计梯度：

$$
\widehat{\nabla_\theta J}
=
\frac{1}{m}
\sum_{i=1}^{m}
r(y^{(i)})
\nabla_\theta\log\pi_\theta(y^{(i)})
$$

**直觉：生成一个回答，看看得分，再调整这个回答出现的概率。**

- Reward 为正：提高该回答的概率。
- Reward 为负：降低该回答的概率。
- Reward 为零：该样本不提供更新信号。

Reward 可以来自人、Reward Model，甚至程序测试，**不需要对 Reward 本身求导**。

与 SFT 对比：

| 方法 | 回答从哪里来？ | 如何更新？ |
|---|---|---|
| SFT | 人写的标准答案 | 提高标准答案的概率 |
| REINFORCE | 模型自己的采样 | 按奖励提高或降低概率 |

### 3.2 把 KL 合并进 Reward

**让模型提高 Reward 的同时，不要偏离原来的模型太远。**
KL 可以写成：

$$
D_{\mathrm{KL}}(\pi_\theta\parallel p_{\mathrm{base}})
=
\mathbb E_{y\sim\pi_\theta}
\left[
\log\frac{\pi_\theta(y)}{p_{\mathrm{base}}(y)}
\right]
$$

因此，可以定义每个样本的 **Shaped Reward**：

$$
\tilde r(y)
=
r(y)
-
\beta\log
\frac{\pi_\theta(y)}{p_{\mathrm{base}}(y)}
$$

它包含两部分：

- 原始 Reward：回答好不好。
- KL 惩罚：回答在当前模型下的概率，相比初始模型偏移了多少。

在 Policy Gradient 更新中，可以使用这个包含 KL 惩罚的奖励作为权重。

### 3.3 为什么梯度估计会有 High Variance？

#### 问题一：Sparse Rewards

假设：

- 答对得 1 分。
- 答错得 0 分。
- 模型正确率只有 10%。

每批采样 10 个回答时：

| Batch | 答对数量 | 更新信号 |
|---|---:|---|
| Batch 1 | 1 | 有少量信号 |
| Batch 2 | 0 | 没有信号 |
| Batch 3 | 3 | 信号明显更强 |

因此，更新方向和幅度容易受到采样运气影响。

#### 问题二：所有奖励都有共同偏移

例如三个回答得分为：

$$
8,\quad 9,\quad 10
$$

全部使用正奖励，会同时鼓励这三个回答。但训练更需要知道它们之间的相对差异。

### 3.4 Baseline：看“比预期好多少”
**Baseline 用于减少噪声， 是一个“预期得分”，用来判断回答比预期好多少，让训练更新更稳定。**

减去一个 **Baseline**，比如 $b=9$：

$$
A(y)=r(y)-b
$$

得到：

$$
-1,\quad 0,\quad 1
$$

这个 $A(y)$ 叫 **Advantage**：

> 这个回答，比基准水平好多少？

梯度估计变成：

$$
\hat g
=
(r(y)-b)
\nabla_\theta\log\pi_\theta(y)
$$

只要 Baseline 不依赖当前采样的回答 $y$，就有：

$$
\mathbb E_{y\sim\pi_\theta}
\left[
b\nabla_\theta\log\pi_\theta(y)
\right]
=
0
$$

所以减去 Baseline 不改变期望梯度。



### 3.5 从整段回答转向每个 token

一个回答由多个 tokens 构成：

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t})
$$

因此，整段回答的梯度也可以拆成：

$$
\hat g
=
\sum_{t=1}^{T}
A(y)
\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t})
$$

这里每个 token 都使用相同的 $A(y)$。

问题是：一段回答可能前面合理、后面出错，整段共享一个 Advantage，训练信号不够细。

### 3.6 Value Model：给每个 token 一个不同的预期

**Value Model** 根据当前前缀预测：

> 从这里继续生成，预计还能得到多少奖励？

记作：

$$
V_\phi(s_t)
$$

其中 $s_t$ 是 Prompt 加上当前已经生成的内容。

每个 token 的 Advantage 可以写成：

$$
A_t=R_t-V_\phi(s_t)
$$

$R_t$ 是从这一步开始，实际获得的后续累计奖励：

$$
R_t=\sum_{k=t}^{T}r_k
$$

例如：

- 预计后面能得 8 分，实际只得 3 分：表现低于预期。
- 预计后面只能得 2 分，实际得了 6 分：表现高于预期。

Value Model 通过回归训练：

$$
\mathcal L_V(\phi)
=
\mathbb E_t
\left[
(V_\phi(s_t)-R_t)^2
\right]
$$

它为不同位置提供了更细的训练信号，但不代表能完美定位错误。

### 3.7 Importance Sampling：为什么需要概率比？

生成回答很贵，所以希望同一批回答能训练多次。

但这些回答来自旧模型 $\pi_{\mathrm{old}}$。参数更新后，当前模型已经改变，需要修正样本权重。

对整段回答，**Importance Ratio** 是：

$$
\rho(y)
=
\frac{\pi_\theta(y)}{\pi_{\mathrm{old}}(y)}
$$

对应的期望变换为：

$$
\mathbb E_{y\sim\pi_\theta}[f(y)]
=
\mathbb E_{y\sim\pi_{\mathrm{old}}}
[\rho(y)f(y)]
$$

在 PPO 的逐 token 训练中，使用：

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(y_t\mid s_t)
}{
\pi_{\mathrm{old}}(y_t\mid s_t)
}
$$

- $\rho_t>1$：当前模型更倾向于生成这个 token。
- $\rho_t<1$：当前模型更不倾向于生成它。

##  PPO

PPO 可以理解为：
**Policy Gradient + Value Model + 样本复用 + Clipping。**
### **Clipped Objective**

PPO 使用 **Clipped Objective** 防止一次更新把策略改得太猛。：

$$
J_{\mathrm{PPO}}
=
\mathbb E_t
\left[
\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

直觉是：

**已经朝有利方向改了不少，就不要继续从同一个样本上获得无限的更新激励。**

例如 $A_t>0$，说明这个 token 应该被鼓励。

若 $\epsilon=0.2$，当概率比超过 $1.2$ 后，Clipping 会截掉继续增大的额外收益。

注意：这**不是硬性保证**所有概率比都处于 $[0.8,1.2]$。


### KL Penalty vs PPO Clipping


| 对比项 | RLHF 的 **KL penalty** | PPO 的 **clipped objective** |
|---|---|---|
| 和谁比较？ | **固定的初始 SFT 模型** | **生成当前这批样本的 old policy** |
| 怎么限制？ | 偏离越远，目标里扣分越多 | 朝有利方向改到一定程度后，不再增加该样本的收益 |
| 主要管什么？ | 整个训练过程不要偏离初始模型太远 | 用同一批样本反复更新时，不要改得太猛 |
| 参照会更新吗？ | 通常一直固定 | 下一轮采样时更新 |




### 3.9 RLOO、GRPO、DPO 分别改了什么？

| 方法   | 主要思路                                         |
| ---- | -------------------------------------------- |
| RLOO | 同一个 Prompt 生成多个回答，用其他回答的平均奖励作为 Baseline      |
| GRPO | 用同组回答的奖励均值和标准差计算相对 Advantage，不需要 Value Model |
| DPO  | 直接用偏好回答对训练，不单独训练 Reward Model，也不需要在线 RL 采样循环 |

#### RLOO

对同一个 Prompt，生成 $K$ 个回答：

$$
A_i
=
r_i
-
\frac{1}{K-1}
\sum_{j\ne i}r_j
$$

意思是：

> 当前回答，比其他回答的平均表现好多少？

#### GRPO

使用同组奖励进行标准化：

$$
A_i
=
\frac{
r_i-\operatorname{mean}(r_1,\ldots,r_G)
}{
\operatorname{std}(r_1,\ldots,r_G)
}
$$

意思是：

> 当前回答，在这一组回答中表现如何？

实际实现需要处理标准差为零的情况。

#### DPO

直接使用：

- 人类偏好的回答 $y_w$。
- 人类较不喜欢的回答 $y_l$。

训练模型提高偏好回答相对于另一回答的概率，同时考虑参考模型。

它省去了单独训练 Reward Model 和在线 RL 采样的过程。

## 4. Reward Models：让人比较，让模型打分

对应 PPT 第 44–49 页。

RL 训练会产生大量回答，不可能每个都找人打分。
所以先收集一批人类判断，训练一个自动评分模型：

$$
r_\phi(x,y)
$$

以后让它给新回答提供奖励。

### 4.2 为什么收集比较，而不是绝对分数？

不同人的评分标准不同：有人觉得好答案是 8 分，有人觉得同样的答案只能给 6 分。
但他们可能都同意： A 比 B 更好。

因此收集 **Pairwise Preferences**：

$$
(x,y_w,y_l)
$$

其中：

- $y_w$：人类更喜欢的回答。
- $y_l$：人类较不喜欢的回答。

### 4.3 怎么把比较变成分数？

Reward Model 接收 Prompt 和回答，输出一个标量分数。

用两个回答的分差，预测人类偏好的概率：

$$
P(y_w\succ y_l\mid x)
=
\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
$$

其中 $\sigma$ 是 Sigmoid：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

因此：

- 两个分数相同：预测胜率为 $0.5$。
- 偏好回答的分数高很多：预测胜率接近 $1$。

### 4.4 Reward Model 的训练目标

训练损失是：

$$
\mathcal L_{\mathrm{RM}}
=
-\mathbb E_{(x,y_w,y_l)}
\left[
\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right]
$$

本质上就是：

**让人类偏好的回答，获得比另一个回答更高的分数。**

这里学到的是偏好排序，分数本身没有天然的绝对单位。

训练完成后，需要在未见过的人类比较数据上检查准确率。

## 5. RLHF in Practice：把整套流程接起来

对应 PPT 第 50–53 页。

### 5.1 RLHF 的三个阶段

**Reinforcement Learning from Human Feedback（RLHF）** 在这份 PPT 中分成三个阶段：

| 阶段              | 数据或反馈          | 学习目标        |
| --------------- | -------------- | ----------- |
| 1. SFT          | 人写的示范回答        | 学会按指令回答     |
| 2. Reward Model | 人类对回答的偏好比较     | 学会判断哪个回答更好  |
| 3. RL，例如 PPO    | 模型自己生成回答，RM 评分 | 提高奖励，同时控制偏移 |

可以记成：

**先模仿示范，再学习评分标准，最后根据评分改进。**

### 5.2 RLHF 为什么有效？

SFT 教模型模仿给定答案。

RLHF 进一步让模型根据自己生成的回答获得反馈，从而提高人类偏好的回答出现的概率。

PPT 的摘要任务实验说明：

**使用 Human Feedback 的较小模型，可以比大得多的纯监督模型更受人类偏好。**

这个结论来自特定实验，不能直接推广为所有任务中小模型都能超过大模型。

最后提醒：**模型学到的是评分者喜欢的东西，而这些偏好未必等于所有方面的质量。**
