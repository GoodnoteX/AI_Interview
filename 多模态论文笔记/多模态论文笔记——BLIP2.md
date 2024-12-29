> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍多模态模型：BLIP2。


![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/多模态论文笔记/image/4.png)



---

@[toc]
# 回顾BLIP
**论文：**[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/pdf/2201.12086)

**BLIP** 是旨在改进图像-文本联合学习的效率多模态模型，特别是通过**生成任务**和**对比学习结合**的方式，在**低监督甚至无监督情况下提升模型性能**。BLIP 的创新点在于它通过**多任务预训练和自引导学习（bootstrapping）机制**，能够以更少的数据达到更好的性能表现。

BLIP 的架构设计包含**图像编码器**、**文本编码器**、**视觉文本编码器**、**视觉文本解码器**。它结合了对比学习和生成式任务，以**自引导的方式**提升模型性能。

# BLIP2
**论文：**[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)

上一节已经给出了问题及其解决方案，下面将介绍详细的实现。其改进主要体现在在**架构**和**训练过程**的优化。

## 架构
BLIP本质上是在训练一个全新的视觉-语言模型，该过程成本大。为了解决这个问题，本文提出的方法是基于现有高质量视觉模型（frozen冻结）及语言大模型（frozen冻结）进行联合训练，同时为减少计算量及防止遗忘，论文对预训练模型进行frozen。为了实现视觉和语言的对齐，作者提出Querying Transformer (Q- Former) 预训练。 
模型的架构实现为·`冻结的预训练图像编码器 + Q-Former + 冻结的预训练大型语言模型`，如下图：  

<img width="739" alt="image" src="https://github.com/user-attachments/assets/20726ce0-d835-4e86-b0a9-20316a7e74d1" />

> 图 1. BLIP-2 框架概述：我们通过预训练一个**轻量级的查询变换器**（Querying Transformer），采用**两阶段策略**来**弥合模态间的差距**。第一阶段从冻结的图像编码器中**引导视觉-语言表征学习**【论文中图2】。第二阶段从冻结的大型语言模型（LLM）中**引导视觉到语言的生成式学习**【论文中图3】，从而实现零样本的指令化图像到文本生成（更多示例请参见图 4）。

**Q-Former的核心结构如下：**

<img width="756" alt="image" src="https://github.com/user-attachments/assets/48aa48b0-f70f-407a-919d-538b93f71165" />

**Q-Former** 是 BLIP-2 中用于视觉-语言交互的核心部分。它用于**视觉输入（图像）和语言输入（文本）之间的相互理解和转换**。图中将其分成了两个部分：图像 Transformer（左半部分）和文本 Transformer（右半部分），它们**共享相同自注意力层self attention**，使用$BERT_{base}$的预训练权重初始化QFormer，并且随机初始化交叉注意层。Q-Former总共包含1.88亿个参数。




> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)

