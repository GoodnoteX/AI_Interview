﻿> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。详细介绍DALL·E的核心部件之一——dVAE，在VQ-VAE的基础上使用Gumbel-Softmax实现采样，用于图像生成。


![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/多模态论文笔记/image/6.png)




> @[toc]
---
# 前情提要
## VAE

AE 和 VAE 在**结构、目的和优化方式**上存在多个重要区别：

| **特性**                  | **AE**                                                                 | **VAE**                                                                            |
|---------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **编码器输出**            | **固定的低维向量**（确定性的表示）                                         | 隐藏变量的均值 		$\mu$ 和方差 $\sigma^2$（表示**潜在空间的分布**）                 |
| **潜在空间**              | 没有明确的分布假设                                                   | 假设潜在空间遵循某种概率分布（通常为正态分布）                                   |
| **解码器**                | 从固定低维向量生成输入数据的近似                                       | 使用 **重参数化技巧** 从潜在变量的分布中采样，再通过解码器生成输入数据的近似                           |
| **损失函数**              | 仅有**重构损失**，最小化输入数据与重构数据的差异                           | **重构损失 + KL 散度**，既保证数据重构效果，又保证潜在空间的分布合理                 |
| **目的**                  | 数据降维、特征提取或数据去噪                                           | 生成新数据（如图像生成、文本生成等），同时保留对输入数据的重构能力               |
| **生成新数据的能力**      | 无法直接生成新数据                                                     | 可以通过**在潜在空间中采样生成**与训练数据相似的**全新数据**                             |



## VQ-VAE

## VAE vs. VQ-VAE

### 区别
需要明白的是，**VAE的主要作用是生成数据**；而**VQ-VAE的主要作用是压缩、重建数据**（与AE一样），如果需要生成新数据，则需要结合 PixelCNN 等生成模型。
> - VAE 的核心思想是通过**编码器学习潜在变量的`连续分布`**（通常是高斯分布，非离散），并从该分布中**采样潜在变量 z，然后由解码器生成数据**。
> - VQ-VAE模型的目标是学习如何**将输入数据编码为`离散潜在表示`，并通过解码器重建输入数据**，量化过程通过**最近邻搜索确定嵌入向量**，是一个**确定性操作**，这一过程并**不涉及离散采样**。
> - 如果需要**生成新数据**，则**需要**在离散潜在空间中随机**采样**嵌入向量。VQ-VAE 本身没有内置采样机制，通常需要结合 PixelCNN 或PixelSNAIL 等模型来完成离散采样。
> 
### 不可导问题及解决方法

- **VAE** 通过`连续潜在空间`和`重参数化技巧`避免了采样操作的不可导问题。
- **VQ-VAE** 的`潜在空间是离散的`，量化过程是不可导的，通过在`最近邻搜索中使用停止梯度传播`来解决不可导问题（dVAE中引入Gumbel-Softmax 替代停止梯度）。原本的VQ-VAE不涉及生成数据，所以不需要采样，如果需要生成数据，则需要结合 PixelCNN 等生成模型。

**VAE 和 VQ-VAE 的不可导问题及解决方法：**

| 特性                          | VAE                                  | VQ-VAE                              |
|-------------------------------|---------------------------------------|-------------------------------------|
| **潜在空间**                   | 连续空间                             | 离散空间                            |
| **不可导问题来源**             | 采样操作不可导                       | 最近邻搜索不可导                    |
| **解决方法**                   | 重参数化技巧                         | 停止梯度传播                        |
| **实现方式**                   | 分离随机性，直接优化 $\mu, \sigma$ | 解码器损失绕过量化过程优化编码器     |
| **适用场景**                   | 平滑采样和连续潜在变量建模            | 离散特征学习和高分辨率生成          |

重新参数化梯度是一种常用于训练变分自编码器（VAE）等生成模型的技术。它依赖于连续分布的可分解性，而 VQ-VAE 的离散分布（通过 one-hot 编码或 Codebook 表示）无法通过这种方式重新参数化。




> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)
