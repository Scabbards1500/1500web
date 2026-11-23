
## Lesson 2 · Text Classification with Linear Models

  

- **Supervised text classification** 将文本转为特征 $\mathbf{x}=f(\text{text})$，用带标签的数据 $\{(x_i,y_i)\}_{i=1}^N$ 学习预测规则。常见假设是样本来自同一数据分布 $(x_i,y_i)\sim P_{\mathrm{data}}$。测试准确率是正确预测数除以测试样本数。

- **One-hot 序列**为每个 token 保留一个向量和位置。**Bag of Words（BoW）**将这些向量相加，得到 $|V|$ 维的词频向量，丢失词序。相同词频向量会得到相同预测。

- **1-NN / k-NN** 找距离最近的一个训练样本，或由最近的 $k$ 个邻居投票。小 $k$ 对局部噪声较敏感，大 $k$ 会平滑边界，也可能抹去局部模式。朴素预测单个样本的距离计算是 $O(ND)$。

- **Train / dev / test** 分别用于拟合参数、选择超参数和最终评估。$k$、特征设计和距离度量可以用 dev set 选择。

  

$$

d_1(x,x')=\sum_{j=1}^{D}|x_j-x'_j|,

\qquad

\hat y_{\mathrm{kNN}}=\arg\max_c \sum_{i\in \mathcal N_k(x)}\mathbf 1[y_i=c]

$$

  

**Linear classifier** 为每类计算一个分数。$W\in\mathbb R^{C\times D}$，每一行对应一个类别，$b_c$ 是该类的基础分数。两类分数相等的位置满足一个线性方程。

  

$$

s=Wx+b,\qquad \hat y=\arg\max_c s_c,

\qquad

(w_c-w_{c'})^\top x+(b_c-b_{c'})=0

$$

  

## Lesson 3 · Learning, KL Divergence, MLE, Softmax

  

### KL divergence → MLE → Cross-entropy

  

$$

D_{\mathrm{KL}}(p\|q)=\sum_y p(y)\log\frac{p(y)}{q(y)}

=H(p,q)-H(p)

$$

  

KL 衡量用 $q$ 近似 $p$ 的差距，通常不对称。当 $p$ 固定，$H(p)$ 对模型参数是常数。最小化对输入取期望的 conditional KL，等价于最小化真实数据下的期望 NLL。用训练集平均值近似这个期望，就得到 **Maximum Likelihood Estimation（MLE）**。 真实分布的概率权重未知，而且可能的输入太多，无法逐一枚举来精确计算期望。

  

$$

\begin{aligned}

\mathbb E_{x\sim P_{\mathrm{data}}}

 D_{\mathrm{KL}}(P_{\mathrm{data}}(\cdot\mid x)\|P_\theta(\cdot\mid x))

 &=\text{const}-\mathbb E_{(x,y)\sim P_{\mathrm{data}}}\log P_\theta(y\mid x),\\

\mathcal L_{\mathrm{likelihood}}(\theta)&=\prod_{i=1}^N P_\theta(y_i\mid x_i),\\

\hat\theta_{\mathrm{MLE}}&=\arg\min_\theta

 \left[-\frac1N\sum_{i=1}^N\log P_\theta(y_i\mid x_i)\right].

\end{aligned}

$$

  

**Softmax** 把 logits 转成总和为 1 的概率。对 one-hot target $t$，单样本 cross-entropy 就是真实类别的 negative log-likelihood。若初始化近似均匀，$C$ 分类的单样本 loss 约为 $\log C$。

  

$$

p_k=\frac{\exp(s_k)}{\sum_j\exp(s_j)},\qquad

L_i=-\sum_k t_k\log p_k=-\log p_{y_i}

=-s_{y_i}+\log\sum_j e^{s_j}

$$

  

### Regularization

  

$$

J(W,b)=\frac1N\sum_i L_i+\lambda R(W),\qquad

R_{\mathrm{L2}}(W)=\|W\|_F^2=\sum_{k,j}W_{kj}^2,\qquad

R_{\mathrm{L1}}(W)=\sum_{k,j}|W_{kj}|

$$

  

- **L2** 抑制大权重。按照这里的定义，其梯度是 $2\lambda W$。

- **L1** 倾向产生稀疏权重。$\lambda$ 用 dev set 选择。正则过强可能导致 underfitting。

- **Generalization** 看模型在未参与训练的数据上的表现。较低的 train loss 不保证较低的 dev loss。

- **训练 loss 收敛不保证训练准确率为 100%。** 若样本在当前特征空间里线性不可分，线性分类器无法全部分对。即使线性可分，正则化目标也可能为了更小的权重接受一些训练错误。

  

## Lesson 4 · Gradients and Optimization

  

**Gradient** 是 loss 对各参数的偏导数组成的向量，指向局部上升最快的方向。Random search 尝试不同参数，iterative improvement 则从当前参数出发。Finite difference 可以检查解析梯度，每个坐标都重新算 loss 的开销随参数规模增长。

  

$$

\frac{\partial J}{\partial\theta_j}

 \approx \frac{J(\theta+h e_j)-J(\theta)}{h},\qquad

\theta\leftarrow\theta-\eta\nabla_\theta J

$$

  

**Softmax + Cross-entropy 的梯度**。$t_k=\mathbf1[k=y_i]$，$w_k$ 是第 $k$ 类的权重向量。正确类的更新沿 $x_i$，其他类沿反方向，幅度由当前预测概率决定。

  

$$

\frac{\partial L_i}{\partial s_k}=p_k-t_k,\qquad

\nabla_{w_k}L_i=(p_k-t_k)x_i,\qquad

\frac{\partial L_i}{\partial b_k}=p_k-t_k

$$

  

$$

g_{\mathrm{full}}=\frac1N\sum_i\nabla_\theta L_i,\qquad

g_B=\frac1{|B|}\sum_{i\in B}\nabla_\theta L_i,\qquad

\theta\leftarrow\theta-\eta g_B

$$

  

| 更新方式 | 梯度使用的数据 |
|---|---|
| Full-batch GD | 全部训练样本 |
| Stochastic GD | 单条样本 |
| Minibatch GD | 一小批样本 |

  

均匀抽样时，minibatch gradient 是 full gradient 的无偏估计。每步更新带噪声，loss 可以暂时上升。Learning rate $\eta$ 太小会收敛慢，太大可能震荡或发散。**Backpropagation** 用 chain rule 计算梯度，**optimizer** 用梯度更新参数。线性模型的容量由特征和线性边界决定，增加更新步数无法解决表达能力不足。

  

## Lesson 5 · Feedforward Neural Networks

  

### Neuron、Activation、Forward Pass

  

$$

\sigma(z)=\frac1{1+e^{-z}},\qquad

P(y=1\mid x)=\sigma(s_1-s_0)=\sigma(w^\top x+b)

$$

  

二分类可以用一个 sigmoid，互斥多分类通常用 softmax，多个可同时成立的标签可以分别用 sigmoid。单个线性分数接 sigmoid 后，阈值 0.5 对应的决策边界仍为 $w^\top x+b=0$。

  

$$

z=W_1x+b_1,\qquad h=g(z),\qquad s=W_2h+b_2,\qquad p=\operatorname{softmax}(s)

$$

  

如果层间没有 nonlinear activation，两层可合并为 $(W_2W_1)x+W_2b_1+b_2$。隐藏单元通过最终任务的 loss 学习。

  

| Activation | 公式与重点 |
|---|---|
| Sigmoid | $\sigma'(z)=\sigma(z)(1-\sigma(z))$，两端饱和 |
| Tanh | $\tanh'(z)=1-\tanh^2(z)$，输出以 0 为中心，两端饱和 |
| ReLU | $g(z)=\max(0,z)$，正半轴导数 1，负半轴导数 0 |
| Leaky ReLU | $g(z)=\max(z,\alpha z)$，$0<\alpha<1$，负半轴仍有梯度 |
| GELU | $g(z)=z\Phi(z)$，$\Phi$ 为标准正态 CDF，平滑地按输入缩放 |

  

### Backpropagation

  

先算输出层的误差，再穿过线性层与 activation，最后用每层的输入组成 outer product。

  

$$

\begin{aligned}

\delta_2&=p-t, &

\nabla_{W_2}L&=\delta_2 h^\top, &

\nabla_{b_2}L&=\delta_2,\\

\delta_1&=(W_2^\top\delta_2)\odot g'(z), &

\nabla_{W_1}L&=\delta_1 x^\top, &

\nabla_{b_1}L&=\delta_1.

\end{aligned}

$$

  

- **Capacity & overfitting**。更大的网络可以拟合更复杂的函数。用 dev performance 检查泛化，常见控制方式有 L2、dropout 和 early stopping。Dropout 训练时随机屏蔽 activations，推理时关闭。

- **Deep Averaging Network（DAN）**。对 token embeddings 取平均后接 FFN。在无其他位置特征时，词序不同但词频相同的句子会得到相同表示。Embedding 可以固定或和分类器一起训练。

- **Hyperparameters** 包括宽度、深度、activation、learning rate、batch size、epoch 和正则强度。每个 epoch 大致处理一遍训练数据。

  

$$

a=\frac1n\sum_{i=1}^{n}E[w_i],\qquad

\frac{\partial L}{\partial c_i}=\frac1n\frac{\partial L}{\partial a},

\quad c_i=E[w_i]

$$

  

## Lesson 6 · Word Embeddings and Tokenization

  

### Distributional Semantics 与 Skip-gram

  

**Distributional hypothesis**。出现在相似上下文的词，往往语义相关。Embedding 矩阵 $E\in\mathbb R^{|V|\times d}$ 的每行对应一个 token。比较方向常用 cosine similarity。 查表时，每个 token ID 取出一行长度为 $d$ 的向量，所以 $B\times T$ 的 IDs 会得到 $B\times T\times d$ 的 embeddings。

  

$$

\cos(u,v)=\frac{u^\top v}{\|u\|\|v\|}

$$

  

**Skip-gram** 从 sliding window 构造 $(c,o)$ pairs，给定中心词 $c$，分别预测窗口内每个实际出现的 context word $o$。每个词有 center vector $v$ 和 context vector $u$ 两套参数。 训练目标直接来自无标注文本中的上下文词，因此是自监督学习。

  

$$

P(o\mid c)=\frac{\exp(u_o^\top v_c)}

{\sum_{w\in V}\exp(u_w^\top v_c)},\qquad L_{c,o}=-\log P(o\mid c)

$$

  

$$

L_{\mathrm{corpus}}=-\frac1T\sum_{t=1}^T

\sum_{\substack{1\le |j|\le m\\1\le t+j\le T}}\log P(w_{t+j}\mid w_t),\qquad

\nabla_{u_w}L=(P(w\mid c)-\mathbf1[w=o])v_c

$$

  

$$

\nabla_{v_c}L=\sum_w P(w\mid c)u_w-u_o

$$

  

中心词的梯度是「模型预期的 context vector − 实际 context vector」。沿负梯度更新，就会朝实际出现的 context vector 移动，并减去模型当前期望的部分。

  

- **Intrinsic evaluation** 看 nearest neighbors、similarity 和 analogy，例如 $v_{\mathrm{king}}-v_{\mathrm{man}}+v_{\mathrm{woman}}\approx v_{\mathrm{queen}}$。**Extrinsic evaluation** 看下游任务分数。Embeddings 可以随机初始化、预训练后固定，或用预训练向量初始化后微调。

- **GloVe** 用全局 co-occurrence counts $X_{ij}$ 学向量。核心近似可写为 $u_i^\top v_j+b_i+\tilde b_j\approx\log X_{ij}$，较常见的共现提供更有用的统计信息。

  

### Tokenization 与 BPE

  

| 粒度 | 典型取舍 |
|---|---|
| Word-level | 序列较短，词表大，罕见词可能 OOV |
| Character-level | 词表小，序列长，模型需要组合字符 |
| Subword-level | 常见词或片段可合成一个 token，罕见词可拆分 |

  

- **BPE training** 从 base symbols 开始，统计当前相邻 pair 的出现次数，按 corpus occurrences 加权。合并当前频率最高的 pair，重新统计，记录 merge rank。

- **BPE encoding** 把新文本拆成 base symbols，重复执行当前可用且训练 rank 最早的 merge。Merge ranks 与 vocabulary 共同决定分词结果。例子中规则依次是 $u+g$、$u+n$、$h+ug$，unhug 最终拆成 un 和 hug。

- **Tokenization → IDs → Embeddings**。切分决定 tokens，词表映射给出 IDs，embedding lookup 返回连续向量。切分粒度会影响 context budget 和数字等内容的表示。能否处理任意输入取决于 base symbols coverage 或 byte fallback。

  

## Lesson 7 · Language Modeling and RNNs

  

### Chain Rule 与 Perplexity

  

$$

P(w_{1:T})=\prod_{t=1}^T P(w_t\mid w_{<t}),\qquad

\bar L=-\frac1M\sum_{t=1}^M\log P(w_t\mid w_{<t}),\qquad

\mathrm{PPL}=\exp(\bar L)

$$

  

使用 $\log_2$ 时对应 $\mathrm{PPL}=2^{\bar L}$。均匀预测大小为 $|V|$ 的词表时，$\mathrm{PPL}=|V|$。比较时保持测试集、tokenization 与统计口径一致。完整句子概率可把 EOS 也作为预测目标。
**每个词都等概率时，perplexity 就等于词表大小。**
  

**Fixed-window FFN LM** 拼接最近 $k$ 个 embeddings，再预测下一词。拼接保留窗口内顺序，加和则丢失顺序。固定窗口限制可直接使用的历史长度。**RNN** 每步把当前 embedding 和上一步 hidden state 合并，历史信息通过 recurrent state 传递。 由于 $h_t$ 依赖 $h_{t-1}$，同一序列的时间步要按顺序计算。各时间步共享 $W_h,W_e,U$ 等参数，所以序列变长会增加计算量并拉长反传路径，参数量保持固定。

  

$$

h_t=\phi(W_hh_{t-1}+W_e e_t+b_h),\qquad

P(w_{t+1}\mid w_{\le t})=\operatorname{softmax}(Uh_t+b_o)

$$

  

### BPTT、Gradient Clipping、LSTM

  

$$

J_t=\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(\phi'(z_t))W_h,\qquad

\nabla_{h_k}L_t=J_{k+1}^\top\cdots J_t^\top\nabla_{h_t}L_t

$$

  

- **BPTT** 把 RNN 按时间展开，用 chain rule 对共享参数累积梯度。长 Jacobian 连乘可能产生 **vanishing gradients** 或 **exploding gradients**。

- **Gradient clipping** 在 $\|g\|>\tau$ 时用 $g\leftarrow \tau g/\|g\|$ 控制梯度范数，主要处理梯度爆炸。

- **LSTM** 维护 cell state $c_t$ 和对外输出的 hidden state $h_t$。Forget gate $f_t$ 控制保留多少，input gate $i_t$ 控制写入多少，output gate $o_t$ 控制输出多少。三类 gate 通常用 sigmoid，候选记忆 $\tilde c_t$ 通常用 tanh。

  

$$

c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\qquad

h_t=o_t\odot\tanh(c_t)

$$

  

Cell state 的直接梯度路径由 $\operatorname{diag}(f_t)$ 控制，相关 $f_t$ 接近 1 时更易跨步保存信息和梯度。RNN 可以每步输出用于 LM 或 tagging，也可以取序列表示做分类。Bidirectional RNN 拼接左右方向的状态 $[\overrightarrow h_t;\overleftarrow h_t]$，适合整段输入已知的编码任务。Deep RNN 在同一时间步堆叠多层。

  

## Lesson 8 · Machine Translation, Seq2Seq and Attention

  

### Conditional LM 与 Seq2Seq

  

Parallel corpus 提供源句 $X$ 和目标句 $Y$。Encoder 读取源句，Decoder 逐 token 生成目标句。基础 RNN Seq2Seq 用 Encoder 最后一个状态初始化 Decoder，长输入会遇到固定向量的 **information bottleneck**。

  

$$

P(Y\mid X)=\prod_{t=1}^T P(y_t\mid y_{<t},X),\qquad

L=-\frac1T\sum_t\log P(y_t^*\mid y_{<t}^*,X)

$$

  

**Teacher forcing** 训练时使用真实目标前缀。对 RNN，当前步直接输入真实的上一个 token，更早的信息经 recurrent state 传递。推理时前缀来自模型自己的输出，早期错误可能影响后续生成，即 **exposure bias**。

  

### Attention

  

Decoder 状态 $s_t$ 查询每个 Encoder 状态 $h_i$。Score 决定相关性，softmax 转成权重，聚合得到当前需要的 source context。每个生成步都能直接读取各源位置，缓解固定向量瓶颈并缩短梯度路径。权重可显示对齐模式。

  

$$

e_{t,i}=\operatorname{score}(s_t,h_i),\qquad

\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j\exp(e_{t,j})},\qquad

a_t=\sum_i\alpha_{t,i}h_i

$$

  

$$

P(\cdot\mid y_{<t},X)=\operatorname{softmax}(U[a_t;s_t]+b)

$$

  

设 $d_s,d_h$ 是 Decoder 和 Encoder 状态的维度，$r$ 是低秩维度，$d_a$ 是 additive attention 的中间宽度。

  

| Score function | 公式 | 参数量，忽略 bias |
|---|---|---|
| Dot-product | $s^\top h_i$ | 0，要求同维 |
| Multiplicative / bilinear | $s^\top Wh_i$ | $d_s d_h$ |
| Reduced-rank | $(As)^\top(Bh_i)$，$W=A^\top B$ | $r(d_s+d_h)$ |
| Additive | $w^\top\tanh(W_1h_i+W_2s)$ | $d_a(d_s+d_h+1)$ |

  

### Greedy、Beam Search、Length Normalization

  

$$

\hat y_t=\arg\max_y P(y\mid\hat y_{<t},X),\qquad

\hat Y=\arg\max_Y P(Y\mid X),\qquad

S(y_{1:t})=\sum_{i=1}^t\log P(y_i\mid y_{<i},X)

$$

  

Greedy 每步选概率最高的 token，可能错过全局最高概率的序列。Beam search 每步扩展当前 $k$ 条前缀，再从所有扩展中统一选 top-$k$。每步约比较 $k|V|$ 个扩展，beam width 为 1 时退化为 greedy。累加 log-probability 容易偏好短序列，可用平均分 $S(Y)/|Y|$ 作为一种 length normalization。EOS 标记候选结束。

  

## Lesson 9 · Self-Attention and Transformer Encoder

  

### Query、Key、Value 与矩阵形式

  

输入 $X$ 按行放置 token representations。$Q=XW_Q,\ K=XW_K,\ V=XW_V$。Self-attention 的三个投影来自同一序列，cross-attention 的 Query 和 Key/Value 来自不同序列。$A_{ij}$ 表示 Query $i$ 从位置 $j$ 读取的权重，softmax 沿每个 Query 的所有 Keys 计算。

  

$$

e_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}},\qquad

A_{ij}=\frac{e^{e_{ij}}}{\sum_m e^{e_{im}}},\qquad

z_i=\sum_j A_{ij}v_j,\qquad

Z=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V

$$

  

| 矩阵 | 单个 self-attention head 的形状 |
|---|---|
| $X$ | $n\times d$ |
| $Q,K$ | $n\times d_k$ |
| $V,Z$ | $n\times d_v$ |
| $QK^\top,A$ | $n\times n$ |

  

任意两位置可以在一层内直接交互。固定总 head 宽度时，attention 核心计算约 $O(n^2d)$，Q/K/V 和输出投影另有约 $O(nd^2)$。在分量独立、零均值且单位方差的简化假设下，$\mathrm{Var}(q^\top k)=d_k$，除以 $\sqrt{d_k}$ 把 logit 的典型尺度控制住。$\partial A_{ij}/\partial e_{ij}=A_{ij}(1-A_{ij})$，概率接近 0 或 1 时梯度变小。

  

### Multi-Head Attention、FFN、Residual

  

$$

\mathrm{head}_h=\mathrm{Attention}(XW_Q^{(h)},XW_K^{(h)},XW_V^{(h)}),\qquad

\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W_O

$$

  

- **Multi-head attention** 用不同的投影学习多套读取权重，再把各 head 的结果合并。固定模型宽度 $d$ 且每个 head 宽度为 $d/H$ 时，四套投影约有 $4d^2$ 个参数，忽略 bias。

- **Position-wise FFN** 对每个位置分别应用同一组参数。Attention 跨 token 混合信息，FFN 在单 token 内生成非线性特征，典型宽度为 $d\to4d\to d$。单个 head 的聚合输出是 Values 的加权平均。

- **Residual connection** 保留输入，加上子层的更新。每个子层保持输入输出的模型维度，以便继续堆叠。 当 $F(x)$ 很小时，前向输出接近输入。当 $\partial F/\partial x$ 也很小时，反向的 Jacobian 接近 $I$，梯度近似原样通过。

  

$$

\mathrm{FFN}(x)=W_2g(W_1x+b_1)+b_2,\qquad

x_{\ell+1}=x_\ell+F(x_\ell),\qquad

\frac{\partial x_{\ell+1}}{\partial x_\ell}=I+\frac{\partial F}{\partial x_\ell}

$$

  

## Lesson 10 · Transformer Decoder、Normalization、Position

  

### LayerNorm、RMSNorm、Pre-Norm、Post-Norm

  

归一化沿同一个 token 的 feature dimensions 计算。LayerNorm 减均值并按标准差缩放，RMSNorm 按均方根缩放。Pre/Post 描述归一化的位置，两种概念可以分别选择。

  

$$

\mu=\frac1d\sum_j h_j,\qquad

\sigma^2=\frac1d\sum_j(h_j-\mu)^2

$$

  

$$

\mathrm{LayerNorm}(h)=\gamma\odot\frac{h-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad

\mathrm{RMSNorm}(h)=\gamma\odot\frac{h}{\sqrt{\frac1d\sum_j h_j^2+\epsilon}}

$$

  

$$

\text{Post-Norm}\quad x_{\ell+1}=\mathrm{Norm}(x_\ell+F(x_\ell)),\qquad

\text{Pre-Norm}\quad x_{\ell+1}=x_\ell+F(\mathrm{Norm}(x_\ell))

$$

  

原始 Transformer 使用 post-norm 和 LayerNorm。现代模型中常见 pre-norm，LLaMA 3 是 pre-norm 与 RMSNorm 的例子。Attention scaling 控制点积随 head dimension 增长的尺度，normalization 控制输入表示的幅度。

  

### Causal Mask、Teacher Forcing、Cross-Attention

  

$$

M_{ij}=\begin{cases}0&j\le i\\-\infty&j>i\end{cases},

\qquad A=\operatorname{softmax}_{\mathrm{row}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)

$$

  

- **右移目标**。输入为 BOS、$y_1,\ldots,y_{T-1}$，监督目标为 $y_1,\ldots,y_T$。对角线可见，当前位置的输入已经给定，监督的是下一个 token。

- **训练并行**。真实前缀已知，一次 forward 可算所有位置。**推理自回归**。每个新 token 生成后才知道下一步的前缀。 Causal mask 用在 decoder 的 self-attention logits 上，防止训练时读到未来的目标 token。Cross-attention 通常可以看完整的源序列。

- **Encoder–decoder block** 依次使用 masked self-attention、cross-attention、FFN。Cross-attention 的 $Q$ 来自 Decoder，$K,V$ 来自 Encoder，score matrix 为 $n_{\mathrm{tgt}}\times n_{\mathrm{src}}$。Decoder-only 的标准 block 使用 causal self-attention 和 FFN。

  

$$

L=-\frac1T\sum_{t=1}^T\log P_\theta(y_t\mid y_{<t},X)

$$

  

### Positional Information 与 GQA

  

| 方法 | 加入位置的方式 |
|---|---|
| Learned absolute position | 查位置向量 $p_i$，输入为 $x_i+p_i$ |
| Sinusoidal encoding | 用不同频率生成 $p_i$，再与输入相加 |
| ALiBi | 对 $j\le i$ 的 logit 加上 $-m_h(i-j)$，不同 head 可有不同斜率 |
| RoPE | 按位置旋转 $Q,K$，点积中的旋转项只依赖相对位置 |

  

$$

\mathrm{PE}(i,2k)=\sin\left(\frac{i}{10000^{2k/d}}\right),\qquad

\mathrm{PE}(i,2k+1)=\cos\left(\frac{i}{10000^{2k/d}}\right)

$$

  

$$

(R_i q)^\top(R_jk)=q^\top R_i^\top R_j k=q^\top R_{j-i}k

$$

  

未加入位置信息的 full self-attention 对输入置换是等变的。Sinusoidal 公式能算出更远的位置向量，实际长 context 表现仍取决于模型。**GQA** 让一组 Query heads 共享同一个 KV head，允许保留多个 Query heads 并减少 KV cache。

  

## Lesson 11 · Pretraining Encoders

  

### Pretraining、Probing、ELMo

  

用文本自身构造预测目标，再将预训练表示迁移到有标签的任务。**Probing** 用简单模型检测固定表示中能读出什么信息。Sentiment neuron 的例子展示了字符级 LM 的内部单元可以携带情感信息。

  

$$

h_{k,j}=[\overrightarrow h_{k,j};\overleftarrow h_{k,j}],\qquad

\mathrm{ELMo}_k=\gamma\sum_{j=0}^L s_jh_{k,j},\qquad

\sum_j s_j=1

$$

  

ELMo 分别用正向与反向 LSTM LM 构建 contextual embeddings，再拼接同一位置的双向状态。下游任务学习各层的 softmax 权重 $s_j$ 与缩放 $\gamma$。同一个词在不同上下文中可得到不同表示。

  

### BERT、MLM、NSP

  

**BERT** 使用 bidirectional Transformer encoder。原始 **Masked Language Modeling（MLM）** 先选出 15% 的 token positions，其中 80% 换成 [MASK]、10% 换成随机 token、10% 保留原样。三种处理的所有选中位置都预测 original token。MLM 修改输入 token 并指定计 loss 的位置，causal mask 控制 attention 可以连接哪些位置。

  

$$

L_{\mathrm{MLM}}=-\sum_{i\in M}\log P_\theta(u_i\mid\tilde u),\qquad

x_i=E_{\mathrm{token}}(u_i)+E_{\mathrm{segment}}(s_i)+E_{\mathrm{position}}(i)

$$

  

- **Next Sentence Prediction（NSP）**。原始 BERT 另用二分类预测第二段是否为第一段真实的后续段，通常一半为真实后续，一半为随机段。NSP 和句子分类可使用 [CLS] 的表示。

- **任务 heads**。句级分类用 $\operatorname{softmax}(Wh_{\mathrm{[CLS]}}+b)$。Token tagging 对每个 token 输出类别。Extractive QA 预测答案 span 的起止位置。[SEP] 分隔段落或标记段末。Fine-tuning 通常联合更新 encoder 和 head。

- 后续方向包括 **RoBERTa** 的更强训练配置及去掉 NSP、**ELECTRA** 的 replaced token detection、**DeBERTa** 的内容与相对位置分开建模。预训练数据还要看 composition、quality 和 transparency。

  

### KV Cache

  

Causal decoder 追加 token 时，旧位置看不到后面，旧的 $K,V$ 可以复用。当前 Query 仍需查询历史 Keys，并聚合 Values。旧 Query 不参与新位置的查询。Full bidirectional encoder 在加入新 token 后，旧位置的上下文表示可能变化，所以同样的缓存逻辑不成立。 KV cache 主要用于自回归推理。训练时真实前缀已知，通常一次计算整段序列，仍用 causal mask 阻止未来信息泄漏。

  

$$

o_t=\operatorname{softmax}\left(\frac{q_tK_{\le t}^{\top}}{\sqrt{d_k}}\right)V_{\le t},

\qquad

\mathrm{KV\ bytes}=2BLTH_{\mathrm{KV}}d_{\mathrm{head}}b

$$

  

$B$ 是 batch size，$L$ 是层数，$T$ 是缓存长度，$H_{\mathrm{KV}}$ 是 KV heads 数，$d_{\mathrm{head}}$ 是 head 宽度，$b$ 是每个数的字节数。缓存后，单个新 token 的 attention 仍随 context 长度线性增长。MHA 的 KV heads 数等于 Query heads 数，GQA 每组共享一个，MQA 所有 Query heads 共用一个。

  

| 架构 | 可见范围 | 常见模型与用途 |
|---|---|---|
| Encoder-only | 输入内双向可见 | BERT，分类、标注、抽取 |
| Decoder-only | 只能看当前位置及左侧 | GPT，自回归生成 |
| Encoder–decoder | Encoder 双向，Decoder 因果并通过 cross-attention 读取 Encoder | T5、BART，翻译、摘要等条件生成 |

  

## Lesson 12 · Pretrained Decoders and Decoding Methods

  

### Causal Pretraining 与 Decoding

  

GPT 学 $P(x_t\mid x_{<t})$。Prompt 可以把下游任务组织成预测接下来的文本，GPT-2 展示了 zero-shot 使用方式。Decoding 从已训练的模型分布中选择输出，选择策略会影响质量、多样性和速度。

  

| 方法 | 逐步规则 |
|---|---|
| Greedy | 选概率最高的 token |
| Beam search | 扩展多条前缀，按累计分数全局保留 top beams |
| Pure sampling | 从原分布随机抽取 |
| Top-$k$ sampling | 保留概率最高的 $k$ 个，再归一化采样 |
| Top-$p$ / nucleus sampling | 按概率排序，保留累计概率首次达到阈值 $p$ 的最短前缀，再归一化 |
| Constrained decoding | 只保留符合约束的 tokens，例如满足 JSON grammar 的合法续写 |

  

$$

P_T(j)=\frac{\exp(u_j/T)}{\sum_k\exp(u_k/T)},\qquad

P_A(j)=\begin{cases}

 P_T(j)/\sum_{k\in A}P_T(k)&j\in A\\

 0&j\notin A

\end{cases}

$$

  

$u_j$ 是 logit，$T$ 是 temperature，$A$ 是保留集合。$0<T<1$ 使分布更尖，$T=1$ 保持原分布，$T>1$ 使分布更平。最大 logit 唯一时 $T\to0^+$ 趋近 greedy，$T\to\infty$ 趋近均匀。Top-$k$ 固定候选数，top-$p$ 的候选数随分布变化。约束可通过给非法 logits 加 $-\infty$ 来实现。

  

### Speculative Decoding 与评估

  

- **Speculative decoding**。Draft model 先连续提出多个 token，target model 在一次 forward 中验证对应位置。接受一段连续前缀，遇到第一个拒绝位置时从校正分布采样并重新起草。标准采样算法的接受概率是 $\min(1,p(x)/q(x))$，拒绝后从归一化的 $\max(p-q,0)$ 抽取，$p$ 为 target、$q$ 为 draft 的当前条件分布。正确的校正保持 target 的采样分布。

- **Self-speculative decoding** 使用同一模型的较轻路径或较浅层起草，再由完整模型验证。速度取决于接受率、起草和验证的开销。

- **Perplexity** 衡量模型对给定 token 序列的预测难度。参照 Lesson 7 的公式。生成文本的 PPL 偏低也可能来自过度偏向常见表达或重复。**Self-BLEU** 将其他生成文本当作参考，分数越高通常表示样本之间越相似、多样性越低。

  

### BART 和 T5

  

| 模型 | 预训练设计 |
|---|---|
| BART | Encoder 读被破坏的输入 $\tilde x$，Decoder 恢复完整原文 $x$，最大化 $P(x\mid\tilde x)$。破坏方式包括 masking、deletion、infilling 和句序打乱 |
| T5 | 用 sentinel tokens 替代多个缺失 spans，Decoder 输出相应的缺失片段，采用统一 text-to-text 接口和 task prefix |

  

Decoder-only 架构盛行的原因包括 next-token objective 统一、训练信号覆盖多数位置、KV cache 可复用以及单 stack 易于扩展。Encoder–decoder 适用于输入和输出分工明确的任务。

  

## Lesson 13 · Post-Training

  

### Instruction Finetuning、SFT、Reward

  

**Instruction finetuning** 用指令与示范回答训练模型。**Supervised Finetuning（SFT）** 用 teacher forcing 最大化示范回答的概率，常把 prompt 部分排除在 loss 外。SFT 的质量受示范数据影响，监督目标直接奖励复现标准答案。

  

$$

L_{\mathrm{SFT}}=-\sum_{t\in\mathrm{answer}}\log p_\theta(y_t^*\mid x,y_{<t}^*)

$$

  

生成模型是 policy $\pi_\theta$。每一步的 state $s_t=(x,y_{<t})$，action 为下一个 token $y_t$，整段回答可获得标量 reward $r(x,y)$。**Reward hacking** 指模型找到提高代理 reward 的方式，实际输出质量却下降。RLHF 常用冻结的初始 SFT policy $\pi_{\mathrm{ref}}$ 增加 KL penalty。

  

$$

\begin{aligned}

J(\theta)&=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]

-\beta D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)),\\

D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})&=

\mathbb E_{\pi_\theta}\left[\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right],\\

\tilde r(x,y)&=r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.

\end{aligned}

$$

  

### REINFORCE、Baseline、Value Model

  

使用 $\nabla_\theta\pi_\theta(y)=\pi_\theta(y)\nabla_\theta\log\pi_\theta(y)$，可以从模型自己采样的回答估计 policy gradient。序列级 reward 常在整段结束后才给出，单次采样的估计可能有较高方差。

**离散采样与黑盒奖励。** 抽到的 token 是离散选择，普通反向传播无法穿过这个选择。REINFORCE 对已采样回答的 $\log\pi_\theta(y\mid x)$ 求梯度，再用 $r(y)$ 作为数值权重。因此 reward 可以来自单元测试或人工评分，无需可微。

  

$$

\nabla_\theta\mathbb E_{\pi_\theta}[r]

=\mathbb E_{\pi_\theta}[r\nabla_\theta\log\pi_\theta(y\mid x)],\qquad

\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid s_t)

$$

  

$$

A(y)=r(y)-b(x),\qquad

\hat g=\frac1m\sum_{i=1}^m A(y^{(i)})\sum_t\nabla_\theta\log\pi_\theta(y_t^{(i)}\mid s_t^{(i)})

$$

  

**Baseline** 给 reward 提供一个预期值。高于预期的回答被鼓励，低于预期的回答被压低。只要 baseline 给定 prompt 后不依赖本次采样动作，它的梯度贡献在期望下为 0。整段答案的 advantage 可以给所有 token 共用，也可以由 **value model / critic** 给每个前缀估计不同的剩余回报。Reward model 为回答评分，value model 预测当前 policy 从这个前缀继续生成的预期回报。

**Baseline 的作用与局限。** 合适的 baseline 能去掉奖励中的共同偏移，降低梯度估计的方差。若答对得 $1$、答错得 $0$，且正确率只有 $10\%$，平均奖励约为 $0.1$。减去这个值能调整已有样本的权重，但采不到正确答案的 batch 依然缺少成功样本提供的训练信号。

  

$$

R_t=\sum_{k=t}^T r_k,\qquad

A_t=R_t-V_\phi(s_t),\qquad

L_V(\phi)=\mathbb E_t[(V_\phi(s_t)-R_t)^2]

$$

  

### Importance Sampling 与 PPO Clipping

  

旧 policy 采样得到的回答可以重用。Importance sampling 用概率比调整其贡献。整段的比率对应整段分布的期望转换，PPO 的常见 surrogate 使用每个采样 token 的比率。

  

$$

\mathbb E_{\pi_\theta}[f(y)]

=\mathbb E_{\pi_{\mathrm{old}}}

\left[\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{old}}(y\mid x)}f(y)\right],

\qquad

\rho_t(\theta)=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)}

$$

  

$$

J_{\mathrm{PPO}}=\mathbb E_t\left[

\min\left(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)

\right]

$$

  

当 $A_t>0$ 时，$\rho_t$ 超过 $1+\epsilon$ 后继续提高概率没有额外目标收益。当 $A_t<0$ 时，$\rho_t$ 低于 $1-\epsilon$ 后继续降低概率也没有额外收益。Clipping 作用于 surrogate objective，实际概率比仍可以越过区间。

  

| | RLHF 的 KL penalty | PPO 的 clipped objective |
|---|---|---|
| 参照 | 冻结的初始 SFT policy | 生成当前 rollout batch 的 old policy |
| 作用 | 对偏离 reference 的分布扣分 | 截住同批样本上朝有利方向继续改动的额外收益 |
| 更新 | reference 通常固定 | old policy 随 rollout 批次更新 |

  

### RLOO、GRPO、DPO

  

- **RLOO** 同一个 prompt 采样 $K$ 条回答，使用其他回答的平均分作 baseline。

- **GRPO** 使用同一 prompt 内的 group mean 和 std 计算相对 advantage，省去单独训练 critic 的需要。

- **DPO** 对偏好对 $(y_w,y_l)$ 直接优化 policy 相对 reference 的 log-probability 差。典型 DPO 训练使用给定偏好数据，省去单独训练 reward model 和在线 PPO rollouts。

  

$$

A_i^{\mathrm{RLOO}}=r_i-\frac1{K-1}\sum_{j\ne i}r_j,\qquad

A_i^{\mathrm{GRPO}}=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}

{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}

$$

  

$$

\begin{aligned}

m_\theta &=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}

-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)},\\

L_{\mathrm{DPO}}&=-\mathbb E\log\sigma(\beta m_\theta).

\end{aligned}

$$

  

### Reward Model 与 RLHF 流程

  

从同一个 prompt 的两个回答收集偏好。**Bradley–Terry** 用 reward 分差预测胜率，两个分数相同时预测胜率为 0.5。训练后用 held-out preference pairs 检查排序准确率。 例如，两位评分者分别给同一对回答打 $(3,5)$ 和 $(7,9)$。他们都偏好第二个回答。收集两两比较可以保留共同的排序，避开个人打分尺度的差异。

  

$$

P(y_w\succ y_l\mid x)=\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr),\qquad

L_{\mathrm{RM}}=-\mathbb E_{(x,y_w,y_l)}

\log\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)

$$

  

| 阶段 | 用什么数据 | 学到什么 |
|---|---|---|
| SFT | 指令与示范回答 | 按指令作答 |
| Reward model | 同一指令的偏好对 | 给人类更喜欢的回答更高分 |
| RL，例如 PPO | Policy 自己生成的回答、reward model 评分与 KL penalty | 提高经常获得高分的回答的生成概率，同时控制相对 reference 的偏移 |

### Best-of-N 与 PPO

**Best-of-N** 每次收到新 prompt，都从基础模型采样 $N$ 个回答，用 reward model 给它们打分，再返回得分最高的回答。基础模型本身无需为这一步更新权重，但采样和打分的成本会在每次推理时重复。**PPO** 在训练阶段生成并评分多条回答，用这些信号更新 policy，让部署后的模型更容易直接生成高分回答。PPT 第 28 页的样题考的是这笔成本发生在训练还是推理。
