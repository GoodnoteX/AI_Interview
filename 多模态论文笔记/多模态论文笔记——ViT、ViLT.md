> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍Transformer架构在计算机视觉方面的成功模型，将Transformer引入图像领域：ViT、ViLT。
> 

![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/多模态论文笔记/image/1.png)


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







> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)
