> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍VQ-VAE和VQ-VAE-2的原理和训练过程，为后面的dVAE在DALLE中的使用打下坚实的基础。


![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/多模态论文笔记/image/5.png)


> @[toc]
# AE和VAE
参考：[深度学习——AE、VAE](https://lichuachua.blog.csdn.net/article/details/143067237)
# VQ-VAE
论文：[Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937)

**VQ-VAE（Vector Quantized Variational AutoEncoder，向量量化变分自编码器）** 主要是**将连续潜在空间的点映射到最近的一组离散的向量**（即codebook中的向量）。模型引入了**离散潜在空间**的思想，克服了**传统 VAE 中连续潜在空间**表示的局限性，能够有效学习高质量的离散特征表示。
## 传统 VAE 的问题
1. **连续潜在空间的限制**：
   - VAE 的潜在变量 $z$ 是连续值，这会导致模型生成的表示较为分散、不够紧凑，**无法高效捕获复杂数据的离散结构**（如**图像中的清晰边缘、重复纹理，或离散的语音特征**）。
   
2. **后验坍塌问题**：
   - **潜在变量的表示能力未被充分利用**。指**编码器**生成的潜在表示 $z$ 对**解码器**的输出贡献非常小，**可能部分或完全被忽略**。
  >  - 当 KL 散度正则化过强时，编码器可能输出接近于先验分布（如 $\mathcal{N}(0, 1)$），导致潜在变量 $z$ 的信息丢失。

## VQ-VAE 与 VAE 的对比

| 特点               | VAE                                | VQ-VAE                              |
|--------------------|-------------------------------------|-------------------------------------|
| 潜在空间           | 连续空间                           | 离散空间                            |
| 潜在变量 $z$   | 每一维是连续的实数值，包括所有的有理数（如整数、小数和分数）以及无理数               | 每一维是离散的整数                  |
| 潜在分布建模       | 高斯分布                           | 离散分布（通过 codebook 表示）      |

## VQ-VAE 模型结构

VQ-VAE 与 VAE 的结构非常相似，只是**中间部分**不是学习概率分布，而是换成 **VQ 来学习 Codebook**。
<img width="1004" alt="image" src="https://github.com/user-attachments/assets/1b242dea-d3af-40c6-bba8-5fb915a02e3c" /><center>VAE架构图</center>
<img width="1044" alt="image" src="https://github.com/user-attachments/assets/7087d2b1-13ad-4276-bb51-73d6852ae2f4" /><center>VQ-VAE 架构图</center>


VQ-VAE 的**整体结构**如下：
<img width="1041" alt="image" src="https://github.com/user-attachments/assets/f3727100-c909-479a-9d08-c817e5071fe6" />
> **Figure 1**：
> - 左侧：描述 VQ-VAE 的图示。编码器 $z(x)$ 的输出被映射到最近的嵌入点 $e_2$。梯度 $\nabla_z L$（红色箭头）将推动编码器改变输出，从而在下一次前向传播中调整配置。
>  - 右侧：嵌入空间的可视化。编码器输出 $z_e(x)$ 被映射到最近的离散点 $e_2$。






> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)
