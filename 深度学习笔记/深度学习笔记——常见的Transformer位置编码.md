> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍3种常见的Transformer位置编码——正弦/余弦位置编码（sin/cos）、基于频率的二维位置编码（2D Frequency Embeddings）、旋转式位置编码（RoPE）


![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/深度学习笔记/image/8.png)

---
> @[toc]

# Transformer中常见的编码方式
- 自注意力机制（Self-Attention）本身不具备任何顺序或空间位置信息。
- 为此，需**要显式地将位置信息嵌入输入特征**，以确保模型能够感知特征间的空间或时间关系。

## 正弦/余弦位置编码（Sinusoidal Positional Encoding）
在 Transformer 的原始论文（Vaswani et al., 2017）中提出的，最原始的位置编码。正弦/余弦位置编码也叫**1D Frequency Embeddings**，通过频率函数将每个位置嵌入到特征空间中。

**公式：**

$$
  PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
$$

$$
  PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)
$$
  - $pos$：表示输入序列的位置。
  - $d$：表示embedding维度。
  - 正弦和余弦的周期性特点可以让模型捕获相对位置信息。
 
**说明：**

- 正弦 $\sin$ 被应用于所有偶数维（索引为 $2i$）；
- 余弦 $\cos$ 被应用于所有奇数维（索引为 $2i+1$）。

这种设计的意义在于：
1. **区分不同维度的位置信息**：
   - 对偶数维和奇数维分别使用不同的函数，可以**让不同维度的位置信息**具有**不同的变化模式**。
   - 例如，偶数维的位置信息可能更注重某种语义，奇数维则可能补充另一种语义。
2. **模型的平移不变性**：
   - 在一些任务中，特别是**相对位置编码**时，正弦和余弦函数的周期性可以帮助模型更**容易**地捕获相对距离信息。
3. **消除对称性**：
   - 如果只用一种函数，比如全是 $\sin$，可能导致**偶数维和奇数维的输出具有对称性，降低信息的区分度**。

---

## 基于频率的二维位置编码（2D Frequency Embeddings）

主要针对Transformer处理二维数据（如图像）的情况。在 ViT（Vision Transformer）的标准实现中，将**两个**独立的 **1D Frequency Embeddings** 分别应用于图像的行（height）和列（width）方向，然后通过拼接（concat）或求和（add）来构造最终的 **2D Frequency Embeddings** 。

**实现方式：两个 1D Frequency Embeddings 构成 2D Embeddings**




> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)


