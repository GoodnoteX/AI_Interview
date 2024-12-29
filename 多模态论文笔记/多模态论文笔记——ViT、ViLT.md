> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍Transformer架构在计算机视觉方面的成功模型，将Transformer引入图像领域：ViT、ViLT。
> 

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/65ec4118b1b24528a00f8ffa2eddf665.png#pic_center)


>@[toc]

# ViT
**ViT（Vision Transformer）** 是一种将 **Transformer 模型用于计算机视觉任务**中的创新架构。ViT 只使用了 Transformer 的**编码器** 部分进行**特征提取和表征学习**。

论文：[AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)
## 1. ViT的基本概念
ViT 的核心思想是**将传统的（CNN）的卷积操作替换为 Transformer 的注意力机制**，借鉴 Transformer 模型在自然语言处理（NLP）中的成功经验，用于**图像分类任务**。
## 2. ViT的结构与工作流程
**ViT的架构如下图所示：**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/16fc90a41cf542dca327ac503c5f9c44.png)
>图1：模型概览。我们将一幅**图像分割为固定大小的图块**，将每个**图块线性嵌入（embed）**，添加**位置嵌入（position embedding）**，并将**得到的向量序列**输入到标准的**Transformer编码器**中。为了实现**分类任务**，我们采用标准的方法，在序列中添加一个额外的**可学习的“分类标记”（classification token）**。Transformer编码器的示意图受到 Vaswani 等人 (2017) 的启发。


**ViT 模型的工作流程如下：**
### 1. 图像分块（Image Patch Tokenization）
ViT 将输入的图像划**分为固定大小的图像块（patches）**，并将这些图像块**展开为一维向量**，类似于将图像分成许多小的"单词"。然后，将每个图像块**转换为一个嵌入向量**，这些嵌入向量类似于 NLP 中的词嵌入（Word Embedding）。
>
>- 假设输入图像的尺寸是 $224 \times 224$，将其划分为尺寸为 $16 \times 16$ 的小块。这将产生 $14 \times 14 = 196$ 个图像块。
>- 每个图像块的像素值被展平成一维向量，并通过线性映射（全连接层）转换为固定维度的嵌入向量。

### 2. 位置编码（Positional Encoding）
因为 Transformer 的**注意力机制不依赖于输入的顺序**，而图像中的空间信息是重要的，因此需要给**每个图像块添加位置编码（Positional Encoding）**，以**保留图像块的位置信息**。这样，Transformer 可以理解图像块之间的相对位置关系。
 - 位置编码的方式与 NLP 中的 Transformer 类似，ViT **默认**使用 可学习的1D位置编码，将二维图像的分割图块按照固定顺序**展平成一维序列**后，为序列中的每个位置分配一个可学习的编码向量。

>本文主要解读默认的位置编码，后续提到的ViT的编码也是“可学习的1D位置编码”。
>有序其他模型有优化使用**基于频率的二维位置编码（2D Frequency Embeddings）**来编码图像块的位置。详情请参考：[深度学习笔记——常见的Transformer位置编码
](https://blog.csdn.net/haopinglianlian/article/details/144021458?)

### 3. Transformer 编码器（Transformer Encoder）
图像块和位置编码结合后，作为输入送入 **Transformer 编码器**。Transformer 编码器的每一层由**多头自注意力机制（Multi-Head Self-Attention）**和**前馈神经网络（Feed-Forward Network, FFN）**组成，并通过**残差连接和层归一化**来保持梯度稳定性。

- **多头自注意力机制**：每个图像块与其他所有**图像块之间的相似性**通过自注意力机制计算，模型通过这种机制**捕捉全局的特征表示**。
- **前馈神经网络（FFN）**：每个图像块的特征表示通过前馈网络进一步提炼。

这个过程类似于**传统的 Transformer 中对词的处理**，只不过**这里处理的是图像块**。
### 4. 分类标记（Classification Token）
ViT 模型在**输入图像块之前**，通常会**添加一个分类标记（[CLS] Token）**。这个分类标记类似于 BERT 模型中的 [CLS] 标记，用来**代表整个图像的全局特征**。最终，经过 Transformer 编码器的处理后，CLS 标记的输出**被用于进行图像分类**。

- CLS 标记的输出经过一个**全连接层**，将其映射到目标类别空间中，**得到最终的分类结果**。CLS 是 "classification" 的缩写，表示分类。它是一个附加到图像块序列之前的向量，类似于 BERT 模型中处理文本任务时添加的 [CLS] 标记。CLS 标记没有直接对应于任何特定的图像块，它只是一个特殊的**向量**，用于**捕获整个图像的全局信息**。
> [0.9, 0.05, 0.05]
> 表示 90% 的概率是“猫”，5% 的概率是“狗”，5% 的概率是其他类别。

**具体过程：**
1. **输入序列**：输入序列是由图像块嵌入和位置编码的结合体，且在序列的最前面插入了[CLS] Token。这个序列的形式如下：
$$ [CLS], patch_1, patch_2, \ldots, patch_N $$
其中[CLS]是分类标记，$patch_i$是图像的第$i$个块。
2. **Transformer编码器处理**：整个序列（包括[CLS] Token和图像块嵌入）会通过Transformer编码器进行处理。由于Transformer的**自注意力机制（Self - Attention）** 能够让每个标记关注序列中的所有标记，因此[CLS] Token会在计算过程中与所有图像块交互，“吸收”整个图像的全局信息。
3. **输出全局表示**：经过多层Transformer处理后，[CLS] Token的最终输出向量被认为是整个图像的全局特征表示。这一特征向量能够有效总结图像中的全局信息。
4. **分类任务**：最终，[CLS] Token的输出经过一个全连接层（fully connected layer），将它映射到类别标签的维度空间，用于图像的分类任务。具体来说，[CLS] Token的输出向量$z_{[CLS]}$会通过线性变换和softmax得到每个类别的概率分布，最终用于决策。



## 3. ViT的关键组件
### 1. 图像块（Patch Embedding）
ViT 将图像划**分为固定大小的图像块，并将其展平为一维向量**。这与传统 CNN 的卷积操作不同，CNN 的卷积操作是基于局部感受野，而 **ViT 直接处理全局特征**。
### 2. 多头自注意力机制（Multi-Head Self-Attention）
ViT 的核心是**使用多头自注意力机制来计算每个图像块与其他图像块之间的关系**。与 CNN 通过层级卷积提取特征不同，ViT 通过**全局**的自注意力机制捕捉图像的特征表示。
### 3. 位置编码（Positional Encoding）
ViT 通过位置编码来为每个**图像块提供位置信息**，这在视觉任务中是非常重要的，因为图像块的相对位置对分类任务有重要影响。

## 4. ViT与CNN的对比

| **对比维度**    | **CNN**                                                                 | **ViT (Vision Transformer)**                                                    |
|-----------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **局部 vs 全局** | 依赖于**卷积核的局部感受野**，逐层提取局部特征并组合成全局特征            | 通过**自注意力机制**直接捕捉图像块之间的**全局关系**                             |
| **参数规模**     | **通常参数较少**，适合**处理小数据集**，具有较好的**泛化能力**                     | 通常拥有**更多参数**，在**小数据集上容易过拟合**，但在**大规模数据集上效果出色**      |
| **数据需求**     | 在小数据集上表现稳定，**具有先验信息**（如卷积操作中的**平移不变性**）          | 缺少 CNN 中的先验信息，因此**需要大规模数据集**进行训练，在小数据集上表现不如 CNN |
## 5. ViT的优势和挑战
### 优势
- **全局信息捕捉**：ViT 通过自注意力机制能够直接捕捉图像块之间的全局关系，而不依赖于局部的卷积操作。这在处理一些全局依赖性较强的任务时表现出色。
- **充分利用数据的丰富性**：ViT 在大规模数据集上训练时，能够充分利用数据的丰富性，并展示出优越的性能。尤其在超大规模数据集（如 ImageNet21k、JFT-300M）上，ViT 的性能超过了传统 CNN。

### 挑战
- **数据需求量大**：ViT 模型的参数量较大，因此需要大规模的数据集来训练。如果数据集规模较小，ViT 容易过拟合。
- **训练复杂**：与 CNN 相比，ViT 的训练更复杂，尤其在资源有限的情况下，训练大规模的 ViT 模型会面临内存和计算资源的挑战。

## 6. ViT的应用

**主要用于图像分类任务**，但其架构可以扩展到其他计算机视觉任务，如目标检测、图像分割、视觉问答等。由于其全局特征捕捉能力，ViT 在一些需要处理全局上下文的任务中表现尤为出色。

## ViT 与 CNN 的混合模型 Hybrid ViT 

Hybrid ViT 是一种将 CNN 和 Transformer 结合的架构，它将 **CNN 用于特征提取**，**Transformer 用于全局建模**。可以补充，这种混合模型可以在**一定程度上解决 ViT 在小数据集上的表现问题**，并保留 Transformer 全局建模的优点。

--- 

# ViLT
**ViLT（Vision-and-Language Transformer）** 是一种解决视觉与语言的联合任务的**多模态模型**。**使用Transformer 的编码器**将**视觉信息**和**语言信息**整合在**同一Transformer架构**中。

它**去除了传统**视觉-语言模型中的**卷积神经网络**，ViLT 的主要创新点在于它**不依赖卷积神经网络**（CNN）来处理图像，而是通过直接将图像块和文本输入给 Transformer 模型，**实现视觉和语言的早期融合**。
> 与传统的视觉-语言模型有显著不同，传统模型通常会**先提取，再结合**（先通过卷积网络提取视觉特征，通过NLP模型提取文本特征，再与语言特征结合）。
> 早期融合和晚期融合指的是进入 **Transformer 编码器**的顺序，早期融合，在融合后再进入Transformer。但是CLIP等模型，是经过Transformer提取特征后再进行对比，属于晚期融合。

论文：[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334)
## 1. ViLT 的工作流程

**ViLT的结构如图所示：**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81de32c4668544c3b9d5728d721be4ac.png)
ViLT 直接通过**Transformer对图像和文本进行早期融合处理**，它的工作流程可以概括为以下几个步骤：
### 1. 图像和语言的输入处理
- **图像输入**：与 ViT（Vision Transformer）类似，ViLT 通过**将输入图像划分为固定大小的图像块（patch）**。例如，一个 $224 \times 224$ 的图像可以划分成多个 $16 \times 16$ 的图像块，展平后形成向量序列。
- **语言输入**：**文本输入通过词嵌入（Word Embedding）表示为向量**。文本的输入与 BERT 模型中的处理类似。

### 2. 图像和语言的融合
ViLT 的核心是通过**单一 Transformer 模型**同时处理图像和语言数据。其输入序列是**图像块和词嵌入的融合**。具体步骤如下：
- **图像块嵌入+位置编码**：每个**图像块被展平成一维向量**，并与对应的**位置编码（Positional Encoding）** 结合在一起，类似于 ViT。
- **文本嵌入+位置编码**：文本序列通过嵌入层**映射为固定维度的向量**，并且每个词也被**添加位置编码**。
- **联合输入**：**图像块嵌入和文本嵌入会串联在一起**，作为 Transformer 的输入序列。如下：
$$
[ 文本CLS, \text{文本词}_1, \text{文本词}_2, \ldots, 图像CLS, \text{图像块}_1, \text{图像块}_2, \ldots]
$$

> 文本 token 嵌入在前，图像 patch 嵌入在后。文本和图像块前各有一个【CLS】。
### 3. 自注意力机制
ViLT 使用自注意力机制来**捕捉图像块和文本词之间的相互关系**。通过**多头**自注意力机制，模型可以让每个输入块（无论是图像还是词）与其他块交互，捕捉图像和语言之间的上下文信息。

这种全局的注意力机制能够**高效地融合视觉和语言信息**，从而使得模型能够处理如图文匹配、视觉问答等跨模态任务。
### 4. 输出处理
最终，ViLT 的输出经过 Transformer 编码器处理，得到的结果可以用于多种下游任务。具体根据任务的不同，输出会有不同的处理方式：
#### 图文匹配（Image-Text Matching）
   - 使用分类头（由 Pooler 和全连接层组成），判断输入图像和文本是否匹配，输出 True 或 False。
> Pooler 是一个用于对特定位置的嵌入（通常是 [class] token 的嵌入）进行处理的模块，常用于生成分类或全局上下文的特征表示。
> 
#### 掩码语言建模（Masked Language Modeling, MLM）
   - 输入序列中某些单词被掩码 $[MASK]$，模型预测这些被掩码单词的值。
   - 通过多层感知机（MLP）输出被掩码单词的预测值（例如 "office"）。
> 是从 BERT 模型中借鉴的语言建模任务，用于训练模型的语言理解能力。
> 
#### 单词-图块对齐（Word-Patch Alignment）
   - 模型对文本中的单词和图像中的图块进行对齐，通过 OT（Optimal Transport, 最优传输）计算对应关系。
   - Transformer 编码器的输出 $(z_i^T$, $z_i^V)$ 分别表示文本和图像的嵌入特征。
> 最优传输（OT）为多模态任务中的语义对齐提供了一种强大而高效的方法。通过对分布之间的最佳匹配建模，它能够细致捕捉单词与图块的语义关系，同时具有理论和计算上的稳健性。在多模态学习中，它不仅能提升对齐质量，还能为复杂的任务（如跨模态检索和问答）提供可靠的支持。
## 2. ViLT 的主要创新点
### 1. 无卷积特征提取器
与传统的视觉-语言模型（如 LXMERT、UNITER 等）不同，ViLT 不使用卷积神经网络，而是**直接将图像切分成小块后，使用 Transformer 模型对图像和文本进行融合处理**。

- **优势**：减少了模型的计算开销，因为不需要预训练一个大型 CNN 模型来提取视觉特征。
- **挑战**：直接处理图像块可能在细粒度视觉理解任务上存在性能瓶颈，尤其是在需要精细局部信息时。

### 2. 视觉和语言的早期融合
ViLT 通过**早期融合**（early fusion）的方式，将图像块和文本词嵌入直接结合在 Transformer 的输入中。
> 早期融合和晚期融合指的是进入 **Transformer 编码器**的顺序，早期融合，在融合后再进入Transformer。但是CLIP等模型，是经过Transformer提取特征后再进行对比，属于晚期融合。

#### 什么时候用 CLIP，什么时候用 ViLT？

- **使用 CLIP**：
  - 当需要进行 **图像-文本检索**（给定图像或文本检索相关配对）时，CLIP 的对比学习在跨模态检索方面表现优异。
  - **零样本分类任务**，CLIP 在没有类别标签的条件下，通过类别描述实现分类，无需对新类别进行微调。
  - 常规的 **图像分类** 场景中，CLIP 由于有强大的跨模态对比能力，可以使用类别描述进行分类，而不需要针对每个类别进行单独的训练。
  
- **使用 ViLT**：
  - **视觉问答（VQA）** 或 **图文匹配** 任务中，ViLT 的早期融合能捕捉图像和文本间的细粒度关系，适合需要图像-文本联合推理的任务。
  - **细粒度的图文理解** 任务中，如果任务需要在图像的局部信息和文本的上下文之间进行交互，ViLT 可以更有效地捕捉图像和文本之间的深层语义关系。

**总结：**

- **CLIP** 适合在 **跨模态检索、零样本分类** 和 **简单图像分类** 场景中使用。
- **ViLT** 则更适合 **视觉问答、图文匹配** 和 **细粒度图文理解** 场景。
### 3. 简化的架构
通过使用**单一的 Transformer 模型处理图像和文本**，ViLT 提供了一种简化的架构，避免了传统视觉-语言模型中分别处理图像和文本的复杂性。这种设计大大**简化了模型的计算流程**，同时在许多视觉-语言任务上仍然保持了很高的性能。

## 3. ViLT 的优缺点
###  优点
- **计算效率高**：由于不使用 CNN 或区域提取网络（如 Faster R-CNN），ViLT 相比传统视觉-语言模型具有更少的计算开销，训练和推理速度更快。
- **模型简洁**：单一的 Transformer 模型处理视觉和语言，避免了复杂的多模块设计，架构简单易于扩展。
- **多模态融合效果好**：通过早期融合，ViLT 能够捕捉图像和语言的全局上下文信息，表现出色。

### 缺点
- **精细视觉特征提取能力较弱**：由于没有使用卷积神经网络进行图像特征提取，ViLT 在处理需要细粒度视觉理解的任务时，可能性能不如传统模型。这是因为 Transformer 对局部信息的提取能力不如 CNN。
- **对大规模数据集的依赖**：和 Vision Transformer 类似，ViLT 在较小数据集上的表现可能不如传统方法，因此需要大规模数据集进行预训练才能发挥最佳性能。

# DiT
[DiT（Diffusion Transformer）详解——AIGC时代的新宠儿](https://lichuachua.blog.csdn.net/article/details/143989135)


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

