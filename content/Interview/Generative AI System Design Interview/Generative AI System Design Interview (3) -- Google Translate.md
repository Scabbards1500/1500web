abstract: ai 翻译pipline
time: 2026/07/13
## Introduction

Google Translate is a widely used language translation service offered by Google. The service relies on machine learning (ML) models to understand and translate text between languages.

![](https://i-blog.csdnimg.cn/direct/d1b658bfe1c64b7cb609099a628775ab.png)

## Clarifying Requirements

initial language?

data-large dataset? -general text data?

input-specify language or automatically detect? input length?

realtime- offline? real-time?

## Specifying the system’s input and output

input：a sequence of words in the source language,  target language provided by the user

output: a sequence of words in the target language

![](https://i-blog.csdnimg.cn/direct/fa8c5a2cdb0d466b955aab88907b40f3.png)

## Choosing a suitable ML approach

### **sequence-to-sequence (seq2seq)  structure**

transform an input sequence into an output sequence, which can vary in length from the input.

e.g: text summarization, speech recognition

**Encoder-Decoder Sturcture**

- **Encoder:** Processes the input sequence and transforms it into a **sequence of context vectors**, thus encoding the information in the input sequence.
- **Decoder:** Utilizes the encoder’s context vectors to generate the output sequence one token at a time.

![](https://i-blog.csdnimg.cn/direct/1bc34463023848e390eced287858d9cf.png)

\* **encoder often use bidirection mechanisms** for understand the context from both directions.

why encoder decoder?

1. architecture separates the input understanding and output generation, which is ideal for seq2seq tasks such as language translation.

2. naturally handles variable-length sequences

3. cross-attention mechanism enables the decoder to focus dynamically on relevant parts of the input sequence during generation

## Data Preparation

**training data**

general data

translation data (sentence pairs, each containing a source-language sentence and its corresponding translation in the target language)

### Text preprocessing

1. Remove missing data/  noisy / Deduplicated data

2. **Handle named entities:** replace the tokens with the original entities, helps the model focus on the sentence context during training without being confused by uncommon terms.

For example, consider the sentence: "The California city, Burlingame, is named after diplomat Anson Burlingame." First, we detect the named entities: "California (location name),” “Burlingame (location name)," and "Anson Burlingame (person’s name)." Next, we replace these entities with placeholder tokens: "The ENTITY_1 city, ENTITY_2, is named after diplomat ENTITY_3."

### Text tokenization

subword-level (以后这个闭眼选）

word level: struggle with out-of-vocabulary (OOV) words

#### Byte-Pair Encoding (BPE)

它通过**不断合并语料中最常出现的字符对（或子词对）**，自动学习出一个既能表示常见词又能处理生僻词的词表（vocabulary）。

BPE builds a subword-level vocabulary through iterative merging.

It starts with characters and iteratively merges the most frequent combinations into new subwords. This allows the model to break down words, even rare or unseen ones, into known components, thus enabling accurate understanding and translation.

**initial setup**

a corpus with the following set of words: “cat,” “cats,” “dog,” and “dogs.”

![](https://i-blog.csdnimg.cn/direct/79e08d9cb26348c0bf23c9973996ae9b.png)

1. Add a special end token, “\</w\>,” to the end of each word to mark its boundary. This special token helps the model know when a word has ended.
2. Tokenize the corpus by breaking down each word into individual characters.
3. Initialize the vocabulary with individual characters and their frequency of occurrence.

![](https://i-blog.csdnimg.cn/direct/85d2b4f38ca44796bd1ffcd923707702.png)

**Iterative merging**

iteratively merges the most frequent character pairs into subwords. This continues until the vocabulary reaches a predefined size or meets the stopping criteria.

![](https://i-blog.csdnimg.cn/direct/70e6be84324e4e7ab3eac44d652dcd01.png)![](https://i-blog.csdnimg.cn/direct/517eabef83884da7bd794ebd1f95b241.png)

![](https://i-blog.csdnimg.cn/direct/27eeacd20c9f41c9a43371090314e936.png)

![](https://i-blog.csdnimg.cn/direct/c1970724b1174032817b986004c50c91.png)

Once the vocabulary has been created, we construct our training data by replacing each tokenized sentence with a sequence of integers. This leads to multiple tables, each for a specific language pair. Figure 11 shows the prepared translation data for the English–French and English–Korean language pairs.

![](https://i-blog.csdnimg.cn/direct/3270b9361d0a40898bf553e6e8e69739.png)

## Model Development

### Architecture

#### **Encoder**

input sequence → a sequence of embedding

![](https://i-blog.csdnimg.cn/direct/13ab304300ad400da5e531619b492429.png)

**Text embedding**

**input token → embedding vector.**

These embeddings capture the semantic information of each token.

![](https://i-blog.csdnimg.cn/direct/910cf3a839d04431ad011187eac15ce4.png)

**Positional encoding**

component injects information about the position of each token in the input sequence.

一般情况： sine–cosine encoding

**Transformer**

processes a sequence of token embeddings through a stack of Transformer blocks.

#### Decoder

generates the output sequence one token at a time using the encoder's output and the previously generated tokens.

- **Text embedding:** Converts each token in the target sequence to an embedding
- **Positional encoding:** Injects information about the position of each token
- **Transformer:** Processes the target sequence and outputs an updated sequence of embeddings
- **Prediction head:** Utilizes the updated embeddings to predict the next token.

![](https://i-blog.csdnimg.cn/direct/c5588cc960d240d7af9e13d6677b6dd7.png)

#### Encoder vs Decoder?

key differences between the encoder and decoder:

**Cross-attention layer （decoder only)**

![](https://i-blog.csdnimg.cn/direct/23d97eb1f0ae4ec295c13db5f767fa6c.png)

**Self-attention mechanism**

In the encoder, each token attends to all other tokens in the sequence. This helps the encoder understand the entire sequence comprehensively.

In the decoder, each token is restricted to attending to only those tokens that come before it by masking the future tokens in the sequence.

![](https://i-blog.csdnimg.cn/direct/8f4b29feb5af4938b3bf948730a2a711.png)

**Prediction head （decoder only)**

The decoder has a prediction head on top of the Transformer component. The prediction head usually includes a linear layer followed by a softmax layer to convert the Transformer’s output into probabilities over the vocabulary. These probabilities are used to determine the most likely next token.

## Training

### Unsupervised pretraining

Train a base model using a large corpus of general data. This creates a base model capable of understanding language, grammar, and context.

一般我们都不直接训练因为cost 很高，针对这个任务作者推荐的是Google’s T5 [12] or Meta’s BART [13]

#### **Pretraining data**

C4 [8], Wikipedia [9], and StackExchange [10].

#### **ML objective and loss function**

MLM+crossentropy loss

##### masked language modeling(MLM)

some of the input tokens are masked, and the model is trained to predict those masked tokens.

![](https://i-blog.csdnimg.cn/direct/76e86aaa0dee4123badf1225582d4311.png)

训练 **encoder** 去理解上下文（masked language understanding）

1. Randomly select a subset of tokens in the input sequence and replace them with a mask token ("[MASK]"). For example, the input sentence "Thank you for inviting me" might become "Thank [MASK] for inviting [MASK]”.
2. Feed the masked sequence to the encoder so it can understand the context despite the missing tokens. The encoder outputs a sequence of new embeddings for each token.

训练 **decoder** 去生成文本（autoregressive generation）。

1. Feed the decoder with the same input sequence, but, this time, none of the tokens are masked and the sequence has been shifted one position to the right by the insertion of a start token ("\<BOS\>").
2. The decoder predicts the next token for each position in the sequence. Each prediction uses all previous input tokens and encoded information from the encoder.

![](https://i-blog.csdnimg.cn/direct/02606829071b4c8e910062f1ee86d56c.png)

**loss 会同时更新 encoder 和 decoder**，两者联合训练（joint training）

1. Calculate the cross-entropy loss over the predicted probabilities and the ground-truth for the masked tokens only.

```
Loss反向传播
 ↓
Decoder 参数更新
 ↓
Cross Attention 参数更新
 ↓
Encoder 参数更新
```

### Supervised finetuning

adapts the base model to the specific task of language translation.

![](https://i-blog.csdnimg.cn/direct/d04458c2e5e248c998c09f1ff1e68dfa.png)

**Bilingual approach**

train models specific to each language pair.

**pro:** 1) capture the unique linguistic nuances of each language pair  2) demonstrate higher translation accuracy due to their specialized natures 3) improving performance is simpler with language-specific models because we can easily isolate and address specific issues that may arise for each language pair

**con:** costly

**Multilingual approach（simple) （trend)**

a single model is trained to translate between multiple languages.

simpler, less expensive, and easier to deploy and maintain

e.g  mT5 [14] and mBART [15]

但是我们这个 casestudy 还是bilingual的

#### Training Data

In each table, a row represents one example, containing a sequence of token IDs for the sentence in the source language and a sequence of token IDs for the translation in the target language.

![](https://i-blog.csdnimg.cn/direct/498f8dc85a1f4e2fbff06e5d6291c022.png)

#### ML objective and loss function

finetuning stage is supervised

ML objective： next-token prediction

loss function: cross-entropy

![](https://i-blog.csdnimg.cn/direct/1866105aba7e478fabcc8e33721d0c6d.png)

这个也是一口气predict 完然后再算loss的

## Sampling

trained model generates a potential translation by predicting each subsequent token based on the previously generated tokens and the context of the input sequence.

![](https://i-blog.csdnimg.cn/direct/1d97584203c44fe0a96bb283a6dd475d.png)

**beam search**

这里居然用 beam search 取多条路径吗...

和之前邮件补全的相比，翻译的上下文更长，所以需要以sequence 为单位

**Translation accuracy**：evaluates multiple possible sequences and selects the most probable one.

**Consistency：** deterministic

## Evaluation

### Offline

- **BLEU：看生成文本和参考文本有多少 n-gram 重合（偏 precision）**
- **ROUGE：看参考文本有多少内容被生成出来（偏 recall）**
- **METEOR：在 BLEU 基础上加入语义和词形匹配，更接近人工评价**

现在很多LLM评价其实已经逐渐不用这三个作为主要指标了，因为它们本质都是 **surface-level overlap**。

现在更常见：

- BERTScore（embedding similarity）
- BLEURT
- GPT-as-a-judge
- LLM evaluator
- Human evaluation

因为LLM生成任务里面：

> "The automobile is fast" 和 "The car runs quickly"

BLEU可能认为差很多，但人觉得几乎一样。

### Online

**User feedback**

**User engagement**

## Overall ML System Design

![](https://i-blog.csdnimg.cn/direct/7c0f7bf58b744bf4a6befb925facc407.png)

### Language detector

identifies the language of a given text (sequence classification task)

so we use encoder-only Transformer classifier

1. **Average pooling:** Pass the Transformer's outputs to an average pooling layer, and then a prediction head to output language class probabilities.
2. **Last token representation:** Use the last token representation from the Transformer's output and feed it to the prediction head for probability prediction.

![](https://i-blog.csdnimg.cn/direct/aba8cc757e11489d82e0a8f7a758a850.png)

### Translation service

The translation service interacts with the specific model based on the detected and desired languages.

It then applies beam search to generate a sequence of tokens in the target language and converts the tokens back into text.

The final translation is then shown to the user.

## Other Talking Points

If time permits at the end of the interview, consider discussing these additional topics:

- Supporting translation for languages with limited training data using transfer learning and multilingual models [20].
- Approaching language translation using a decoder-only Transformer [21].
- Continuously improving translation models through user feedback [22].
- Optimizing techniques for efficient inference and on-device translation [23].
- Developing a single multilingual model [24].
- Other automatic metrics such as WER and how they are calculated [25][26].
- How to build a language detection model [27].
