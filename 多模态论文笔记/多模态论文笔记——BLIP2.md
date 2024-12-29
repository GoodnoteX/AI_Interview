> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍多模态模型：BLIP2。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/129eae38dee3455d84149988b8d728a7.png#pic_center)


推荐阅读：
[BLIP2-图像文本预训练论文解读](https://blog.csdn.net/qq_41994006/article/details/129221701)
[【多模态】BLIP-2模型技术学习](https://blog.csdn.net/qq_37734256/article/details/143639691)

---

@[toc]
# 回顾BLIP
**论文：**[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/pdf/2201.12086)

**BLIP** 是旨在改进图像-文本联合学习的效率多模态模型，特别是通过**生成任务**和**对比学习结合**的方式，在**低监督甚至无监督情况下提升模型性能**。BLIP 的创新点在于它通过**多任务预训练和自引导学习（bootstrapping）机制**，能够以更少的数据达到更好的性能表现。

BLIP 的架构设计包含**图像编码器**、**文本编码器**、**视觉文本编码器**、**视觉文本解码器**。它结合了对比学习和生成式任务，以**自引导的方式**提升模型性能。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1cd89b57f6de4c489fa6355f346efc80.png)
# BLIP的问题及BLIP2的优化

在 BLIP 的基础上，**BLIP2** 进行了以下几项主要优化，这些优化显著提升了模型的性能、计算效率和适配性：
### 1. 模块化架构设计
- **BLIP 的问题**：
  - BLIP 的图像编码器、文本编码器、视觉文本编码器和解码器之间的紧密耦合关系，是造成训练成本高、灵活性不足的重要原因。
  - BLIP 的架构限制了视觉编码器和语言模型的选择，适配性不足。
- **BLIP2 的优化**：
  - 采用模块化设计，将模型分为三个模块：
    1. **视觉编码器（Image Encoder）**：用于提取图像的底层视觉特征（支持复用已有的预训练视觉模型，如 CLIP 或 ViT）。
    2. **Q-Former（Querying Transformer）**：用于从视觉特征中提取与语言相关的多模态嵌入。
    3. **预训练语言模型（LLM, Large Language Model）**：用于处理生成任务，如文本生成或问答任务。
  - 模块化设计使得 BLIP2 可以复用现有的强大视觉模型（如 CLIP、ViT）和语言模型（如 GPT、OPT），无需端到端联合训练，大大降低了开发和训练成本。

---
### 2. 引入 Q-Former 模块
- **BLIP 的问题**：
  - BLIP **直接将视觉特征与语言模型对接**，特征提取过程**可能包含冗余信息**，导致对齐效率较低。
- **BLIP2 的优化**：
  - 引入了 Q-Former，这是一个**轻量级的变换器模块**，用于从视觉特征中提取与语言模态相关的嵌入表示：
    - 用于**从视觉编码器生成**的**高维视觉特征**中提取与**语言模态相关**的**低维嵌入表示**，从而实现高效的图像-文本对齐。
  - Q-Former 的加入显著提升了图像-文本对齐的效果，同时减少了计算负担。

---
### 3. 分阶段训练策略
- **BLIP 的问题**：
  - BLIP 需要联合训练四个组件，优化难度大，训练时间长，硬件需求高。
- **BLIP2 的优化**：
  - 分阶段训练策略：
    1. **第一阶段：图像-语言对齐**：
       - 使用视觉编码器和Q-Former。但是冻结视觉编码器的权重（如 CLIP 或 ViT 的预训练模型），仅训练 Q-Former 模块，通过对比学习和图文匹配任务优化视觉-语言的对齐表示。
       - 训练 Q-Former 模块，让其能够**从视觉编码器**生成的高维特征中**提取与语言模态相关的信息**。实现视觉模态和语言模态的对齐，构建统一的多模态嵌入表示。
    2. **第二阶段：文本生成任务**：
       - 使用Q-Former和将预训练语言模型。但是冻结的预训练语言模型（如 GPT 或 OPT），仅训练 Q-Former 来适应生成任务。
       - 使用 Q-Former 提取的多模态嵌入作为语言模型的输入，适配预训练语言模型（如 GPT、OPT 等）进行文本生成任务。
  - 这种策略避免了对大型语言模型的联合训练，显著降低了训练成本。

---
### 4. 减少计算开销
- **BLIP 的问题**：
  - 计算成本高，特别是在需要训练大型语言模型时，对硬件资源需求较高。
- **BLIP2 的优化**：
  - 通过模块化设计和**冻结预训练模型参数**，**计算**需求集中在**轻量级的 Q-Former 模块上**，减少了大规模计算开销。
  - 与 BLIP 相比，BLIP2 的训练速度更快，资源需求更低，适合在资源有限的环境中使用。

---

# BLIP2
**论文：**[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)

上一节已经给出了问题及其解决方案，下面将介绍详细的实现。其改进主要体现在在**架构**和**训练过程**的优化。

## 架构
BLIP本质上是在训练一个全新的视觉-语言模型，该过程成本大。为了解决这个问题，本文提出的方法是基于现有高质量视觉模型（frozen冻结）及语言大模型（frozen冻结）进行联合训练，同时为减少计算量及防止遗忘，论文对预训练模型进行frozen。为了实现视觉和语言的对齐，作者提出Querying Transformer (Q- Former) 预训练。 
模型的架构实现为·`冻结的预训练图像编码器 + Q-Former + 冻结的预训练大型语言模型`，如下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/23d041b85ec349edbc2c7eabf67df130.png)
> 图 1. BLIP-2 框架概述：我们通过预训练一个**轻量级的查询变换器**（Querying Transformer），采用**两阶段策略**来**弥合模态间的差距**。第一阶段从冻结的图像编码器中**引导视觉-语言表征学习**【论文中图2】。第二阶段从冻结的大型语言模型（LLM）中**引导视觉到语言的生成式学习**【论文中图3】，从而实现零样本的指令化图像到文本生成（更多示例请参见图 4）。

**Q-Former的核心结构如下：**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ad2341969a26440586cc943028074395.png)

**Q-Former** 是 BLIP-2 中用于视觉-语言交互的核心部分。它用于**视觉输入（图像）和语言输入（文本）之间的相互理解和转换**。图中将其分成了两个部分：图像 Transformer（左半部分）和文本 Transformer（右半部分），它们**共享相同自注意力层self attention**，使用$BERT_{base}$的预训练权重初始化QFormer，并且随机初始化交叉注意层。Q-Former总共包含1.88亿个参数。

> Learned Queries被视为模型参数。在我们的实验中，我们使用了32个查询，其中每个查询具有768维（与Q-Former的隐藏维相同）。我们使用Z来表示输出查询表示。Z的大小（32 × 768）远小于冻结图像特征的大小（例如，ViT-L/14的大小为257 × 1024）。这种**瓶颈结构**与我们的预训练目标一起工作，迫使查询**提取与文本最相关的视觉信息**。
   
1. **图像 Transformer（左半部分）红框**：
   - 图像 Transformer 负责与**Frozen Image Encoder**交互，融合Learned Queries和Input Image中的信息，提取图像特征，


2. **文本 Transformer（右半部分）绿框**：
   - 文本 Transformer 主要用于处理输入的文本信息（Learned Queries和Input Text）。它既可以作为一个文本编码器，也可以作为文本解码器，用来生成或理解图像相关的文本内容。


在上图中，有三个输入，分别是**Learned Queries**、**Input Image** 和 **Input Text** 是三个重要的组成部分，它们在 **Q-Former** 模块中共同作用，进行图像-文本融合和交互。下面是它们的详细解释：

1. **Learned Queries** （学习到的查询）
   - **Learned Queries** 是 Q-Former 中的一种机制，指的是模型**通过训练学习得到的一组“查询向量”**。这些查询向量**用于从图像和文本中提取信息**，帮助模型聚焦于最相关的部分。它们是一个动态学习的参数，在训练过程中更新和优化，以便更好地捕捉图像和文本之间的关系。
   
   - 在 BLIP-2 中，**Learned Queries** 主要通过交互式方式提取图像和文本的交叉信息。它们在图像和文本的交互过程中充当“桥梁”，帮助模型理解图像和文本之间的关联。

   - **作用**：在 **Q-Former** 中，**Learned Queries** 的作用是引导图像和文本信息的融合，并决定哪些信息是最重要的。它们帮助 **Q-Former** 精确地匹配图像和文本，从而生成更准确的描述或进行正确的推理。

2. **Input Image** （输入图像）
   - **Input Image** 是 BLIP-2 模型中的输入之一，指的是输入给模型的原始图像数据。这些图像数据会通过 **Frozen Image Encoder**（一个预训练的图像编码器）进行编码，转换为高维的视觉特征表示。
   
   - 在 Q-Former 中，图像通过编码器转换为一个固定的特征表示，然后与 **Learned Queries** 和 **Input Text** 进行交互。这些图像特征是图像和文本匹配任务的基础，帮助模型理解图像的内容。
   
   - **作用**：图像输入提供了模型所需的视觉信息，帮助模型理解并生成与图像相关的文本描述或回答相关问题。

3. **Input Text** （输入文本）
   - **Input Text** 是 BLIP-2 模型的另一个输入，指的是输入给模型的文本数据。通常，这个文本数据是描述图像的文字信息。这些文字数据会通过 **Frozen Text Encoder**（一个预训练的文本编码器）进行编码，转换为低维的文本特征表示。
   
   - 在 Q-Former 中，文本会经过 **Text Encoder**（文本编码器）处理，转化为文本的表示。文本与图像的特征表示通过 **Learned Queries** 相互作用，共同生成最终的输出（如图像描述、问题答案等）。

   - **作用**：文本输入提供了模型所需的语言信息，帮助模型理解和生成与图像相关的语言输出。通过与图像特征的融合，文本输入使得模型能够在视觉-语言任务中进行推理和生成。


## 表征学习阶段 Representation Learning Stage
表征学习阶段【冻结的预训练图像编码器 + Q-Former】，在冻结的图像编码器中**引导视觉-语言表征学习**。使用图像-文本对进行预训练，目标是训练Q-Former，使得查询可以学习提取最能提供文本信息的视觉表示。

预训练过程如下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/104ff463b33e48f98aab9f05822bf5ca.png)
> 图 2. （左）Q-Former 和 BLIP-2 第一阶段视觉-语言表示学习目标的模型架构。我们联合优化了三个目标，这些目标通过一组可学习的嵌入（queries）来提取与文本最相关的视觉表示。（右）针对每个目标的自注意力掩码策略，用于**控制查询与文本的交互**。


1. **左图（Q-Former 和 BLIP-2 第一阶段的模型架构），上图红框**：
   - 输入图像通过 **冻结的Image Encoder（图像编码器）** 提取初始视觉特征。
   - 视觉特征与一组可学习的查询（Learned Queries，作为嵌入）通过 Q-Former 模块交互（可学习的查询通过 **自注意力（Self Attention）** 层相互作用，并且通过 **交叉注意力（Cross Attention）** 层与frozen图像特征相互作用）。
   - 和BLIP一样，BLIP2使用3个目标函数来训练模型，并且它们**共享相同的输入格式和模型参数**。每个目标函数通过**不同的注意力掩码（attention mask）策略**来控制**查询和文本**的**交互和影响**。
   - 模型目标分为三个子任务：
      - **图像文本对比学习（ITC）**——在隐空间对齐图片编码和文本编码
      - **图文匹配（ITM）**——二分类任务，让模型判断图文是否一致
      - **基于图像文本生成（ITG）**——下一词预测，让模型学会给定图片输出caption


2. **右图（注意力掩码策略）**：
   - 描绘了 Q-Former 不同任务的注意力掩码机制，用于控制**查询和文本**的交互模式：
     - **双向自注意力掩码（Bi-directional Self-Attention Mask）**：
       - 用于图像-文本匹配任务（Image-Text Matching）。
       - 允许查询和文本令牌之间的全连接交互。
     - **多模态因果自注意力掩码（Multi-modal Causal Self-Attention Mask）**：
       - 用于基于图像的文本生成任务（Image-Grounded Text Generation）。
       - 查询令牌可以访问文本令牌（包括过去和当前），但文本令牌仅关注其过去的令牌，保证生成的因果性。
     - **单模态自注意力掩码（Uni-modal Self-Attention Mask）**：
       - 用于图像-文本对比学习任务（Image-Text Contrastive Learning）。
       - 查询令牌与文本令牌的交互被掩盖，仅进行单模态内部的学习。

论文中实验了两种预训练图像编码器：
1. ViT-L/14 from CLIP
2. ViT-G/14 from EVA-CLIP
## 生成式预训练阶段 Generative Pre-training Stage
这个阶段使用【Q-Former + 冻结的预训练大型语言模型】，在冻结的预训练大型语言模型中**引导视觉到语言的生成式学习**。经过第一阶段的预训练，Q-Former有效地充当了信息瓶颈，将最有用的信息提供给LLM，同时删除不相关的视觉信息。这减少了LLM学习视觉语言对齐的负担，从而减轻了灾难性的遗忘问题。

预训练过程如下如下图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1144878ac6fe431daee5c9e3f6ffc11d.png)
>**图 3. BLIP-2 的第二阶段视觉到语言生成预训练：** 从冻结的大型语言模型（LLMs）中引导生成能力。**顶部**：从**基于解码器**的大型语言模型（例如 OPT）中引导。 **底部**：从**基于编码器-解码器**的大型语言模型（例如 FlanT5）中引导。  **全连接层**的作用是将 Q-Former 的输出维度调整为所选语言模型的输入维度。
>


论文中实验了两种LLM：
1. **无监督训练的OPT作为Decoder-based LLM**，使用语言建模损失（language modeling loss）进行预训练，冻结的 LLM 的任务是根据 Q-Former 的视觉表示生成文本，也就是说直接根据图像生成文本；
2. **基于指令训练的FlanT5作为Encoder-Decoder-based LLM**，使用前缀语言建模损失进行预训练（prefix language modeling loss）预训练，将文本分成两部分，前缀文本perfix test与视觉表示连接起来作为 LLM 编码器的输入，后缀文本用作 LLM 解码器的生成目标，也就是说根据前缀文本+图像生成后缀连续的文本。

---



> 1. **无监督训练的 OPT 作为 Decoder-based LLM**：
>    - **OPT（Open Pre-trained Transformer）** 是一种基于解码器的语言模型，通常用于自回归文本生成任务。在 **BLIP-2** 中，OPT 作为解码器使用，结合 **Q-Former**的视觉表示来生成文本。
>    - **训练方式**：OPT 使用 **语言建模损失**（language modeling loss）进行无监督训练。语言建模损失的目标是预测文本序列中的下一个词，典型的任务是让模型根据已有的文本预测下一个词或字符。在 BLIP-2中，任务是让 OPT 根据输入的视觉表示（来自 Q-Former 的输出）生成与图像相关的文本。
>    **OPT作为解码器**，它根据视觉输入生成完整的文本描述，进行 **图像到文本的生成**。适合用于 **图像到文本的直接生成** 任务。
> 
> 2. **基于指令训练的 FlanT5 作为 Encoder-Decoder-based LLM**：
>    - **FlanT5** 是一个指令调优版本的 T5（Text-to-Text Transfer Transformer），在其基础上进行了特定任务的优化，使其能够更好地处理各种指令任务。在 BLIP-2 中，FlanT5 作为**编码器-解码器模型**，其设计允许模型同时进行编码和解码。
>    - **训练方式**：FlanT5 使用 **前缀语言建模损失**（prefix language modeling loss）进行训练。这种损失函数的核心思想是将输入分为两个部分：
>      - **前缀文本**（prefix text）：这部分文本与 **视觉表示** 结合，作为 FlanT5 编码器的输入。
>      - **后缀文本**（suffix text）：这部分文本作为解码器的目标，用于生成与前缀文本相对应的文本内容。
>    - 在训练过程中，模型的任务是**根据输入的前缀文本和图像表示**来**生成后缀文本**。也就是说，模型通过 **前缀文本+视觉表示** 来生成 **后续的文本描述**。能够处理 **更复杂的多模态任务**，适合需要 **图像和文本交互理解** 的任务。



# 历史文章

## 机器学习

[机器学习笔记——损失函数、代价函数和KL散度](https://blog.csdn.net/haopinglianlian/article/details/143831958?)
[机器学习笔记——特征工程、正则化、强化学习](https://blog.csdn.net/haopinglianlian/article/details/143832118?)
[机器学习笔记——30种常见机器学习算法简要汇总](https://blog.csdn.net/haopinglianlian/article/details/143832321)
[机器学习笔记——感知机、多层感知机(MLP)、支持向量机(SVM)](https://blog.csdn.net/haopinglianlian/article/details/143832552)
[机器学习笔记——KNN（K-Nearest Neighbors，K 近邻算法）](https://blog.csdn.net/haopinglianlian/article/details/143832692)
[机器学习笔记——朴素贝叶斯算法](https://blog.csdn.net/haopinglianlian/article/details/143832781?)
[机器学习笔记——决策树](https://blog.csdn.net/haopinglianlian/article/details/143834363)
[机器学习笔记——集成学习、Bagging（随机森林）、Boosting（AdaBoost、GBDT、XGBoost、LightGBM）、Stacking](https://blog.csdn.net/haopinglianlian/article/details/143834494?)
[机器学习笔记——Boosting中常用算法（GBDT、XGBoost、LightGBM）迭代路径](https://blog.csdn.net/haopinglianlian/article/details/143834628)
[机器学习笔记——聚类算法（Kmeans、GMM-使用EM优化）](https://blog.csdn.net/haopinglianlian/article/details/143834707)
[机器学习笔记——降维](https://blog.csdn.net/haopinglianlian/article/details/143834847)

## 深度学习
[深度学习笔记——优化算法、激活函数](https://blog.csdn.net/haopinglianlian/article/details/143835137)
[深度学习——归一化、正则化](https://blog.csdn.net/haopinglianlian/article/details/143835273)
[深度学习——权重初始化、评估指标、梯度消失和梯度爆炸](https://blog.csdn.net/haopinglianlian/article/details/143835336)
[深度学习笔记——前向传播与反向传播、神经网络（前馈神经网络与反馈神经网络）、常见算法概要汇总](https://blog.csdn.net/haopinglianlian/article/details/143835406)
[深度学习笔记——卷积神经网络CNN](https://blog.csdn.net/haopinglianlian/article/details/143841327)
[深度学习笔记——循环神经网络RNN、LSTM、GRU、Bi-RNN](https://blog.csdn.net/haopinglianlian/article/details/143841402)
[深度学习笔记——Transformer](https://blog.csdn.net/haopinglianlian/article/details/143841447)
[深度学习笔记——3种常见的Transformer位置编码](https://blog.csdn.net/haopinglianlian/article/details/144021458)
[深度学习笔记——GPT、BERT、T5](https://blog.csdn.net/haopinglianlian/article/details/144092300)
[深度学习笔记——ViT、ViLT](https://blog.csdn.net/haopinglianlian/article/details/144093215)
[深度学习笔记——DiT（Diffusion Transformer）](https://blog.csdn.net/haopinglianlian/article/details/144094540)
[深度学习笔记——CLIP、BLIP](https://blog.csdn.net/haopinglianlian/article/details/144096378)
[深度学习笔记——AE、VAE](https://blog.csdn.net/haopinglianlian/article/details/144097222)
[深度学习笔记——生成对抗网络GAN](https://blog.csdn.net/haopinglianlian/article/details/144103764)
[深度学习笔记——模型训练工具（DeepSpeed、Accelerate）](https://blog.csdn.net/haopinglianlian/article/details/144107447)
[深度学习笔记——模型压缩和优化技术（蒸馏、剪枝、量化）](https://blog.csdn.net/haopinglianlian/article/details/144108373)
[深度学习笔记——模型部署](https://blog.csdn.net/haopinglianlian/article/details/144111928)

