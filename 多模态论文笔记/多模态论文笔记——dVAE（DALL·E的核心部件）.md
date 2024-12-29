> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。详细介绍DALL·E的核心部件之一——dVAE，在VQ-VAE的基础上使用Gumbel-Softmax实现采样，用于图像生成。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/928c44cf65e64058b3f4ef6d0854e77c.png#pic_center)




> @[toc]
---
# 前情提要
## VAE
[深度学习——AE、VAE](https://lichuachua.blog.csdn.net/article/details/143067237)

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
[万字长文解读深度学习——VQ-VAE和VQ-VAE-2](https://lichuachua.blog.csdn.net/article/details/144227012)

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

---
> **VAE 的不可导问题及解决方法**
> 
> 
>**不可导问题**
> - 在训练VAE时，我们希望从一个分布中采样出一些隐变量，以生成模型的输出。然而，由于采样操作是不可导的，因此通常不能直接对采样操作求梯度。为了解决这个问题，我们可以使用重新参数化技术。
> - 在 VAE 中，潜在变量 $z$ 是通过从编码器输出的分布 $q(z|x)$ 中采样得到的：   $$   z \sim \mathcal{N}(\mu, \sigma^2)   $$
>   - $\mu$ 和 $\sigma$ 是编码器生成的分布参数。
>   - 采样操作引入随机性，而随机采样本身不可导，因此无法通过梯度反向传播来优化编码器参数。
> 
> **解决方法：重参数化技巧** 
> - 重新参数化技术的基本思想是，将采样过程拆分为两步：首先从一个固定的分布中采样一些固定的随机变量，然后通过一个确定的函数将这些随机变量转换为我们所需的随机变量。这样，我们就可以对这个确定的函数求导，从而能够计算出采样操作对于损失函数的梯度。
> 
> VAE 通过 **重参数化技巧（Reparameterization Trick）** 将采样过程分解为可导部分和不可导部分：
> 1. **分离随机性：**
>    - 采样公式改写为：
>      $$
>      z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
>      $$
>    - $\epsilon$ 是标准正态分布的随机噪声，采样只发生在 $\epsilon$ 中。
>    - $\mu$ 和 $\sigma$ 是由编码器网络直接输出的，可导。
> 2. **作用：**
>    - 随机性仅由不可导的 $\epsilon$ 控制，而 $\mu$ 和 $\sigma$ 的梯度可以正常计算，从而实现端到端训练。
> ---

> 
> **VQ-VAE 的不可导问题及解决方法**
> 
> **不可导问题**
> - 在 VQ-VAE 中，潜在变量是通过将编码器输出 $z_e(x)$ 映射到最近的嵌入向量（codebook 中的向量）得到的：   $$   z_q(x) = \arg\min_{e_k} \|z_e(x) - e_k\|_2   $$
>   - 最近邻搜索是一个不可导操作，因为argmin或argmax 是一个离散操作，涉及离散索引 $k$，因此不可导。
> 
> **解决方法：停止梯度传播（Stop Gradient）** 
> 
> VQ-VAE 使用 **停止梯度传播（Stop Gradient）** 技巧来解决不可导问题：
> 1. **停止梯度：**
>    - 在计算量化操作时，不允许梯度传播到最近邻搜索的部分。
>    - 假设 $z_q(x)$ 是量化后的嵌入向量，VQ-VAE 中的梯度计算会直接**将解码器损失作用到编码器输出 $z_e(x)$**，而不会涉及量化过程。
> 2. **公式：**
>    - $z_q(x)$ 的生成：
>      $$
>      z_q(x) = e_k, \quad k = \arg\min_{i} \|z_e(x) - e_i\|_2
>      $$
>    - 在优化过程中，损失的梯度会通过以下方式传播：
>      $$
>      z_q(x) = z_e(x) + (e_k - z_e(x)).detach()
>      $$
>      - $(e_k - z_e(x)).detach()$ 表示停止梯度传播，仅用 $z_e(x)$ 来优化编码器。
> ---

# dVAE
> dVAE第一次出现是在 Open AI 的 DALL·E 模型论文（[Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092)）中，DALL·E模型是我最近开始研究VAE系列模型的根源，论文中并没有详细给出dVAE的模型架构，更多详细的dVAE结构，强烈推荐下面三篇外文博客：
> - [Understanding VQ-VAE (DALL-E Explained Pt. 1)](https://mlberkeley.substack.com/p/vq-vae)
>
> - [How is it so good ? (DALL-E Explained Pt. 2)](https://mlberkeley.substack.com/p/dalle2)
>
> - [How OpenAI’s DALL-E works?](https://medium.com/@zaiinn440/how-openais-dall-e-works-da24ac6c12fa)


Discrete VAE（dVAE），整体来说与 VQ-VAE 类似，主要的区别是：
- 在 VQ-VAE 中使用停止梯度传播来解决最近邻搜索的离散化方法造成的不可导问题。
- 在 dVAE 中使用**Gumbel-Softmax**为**离散变量**提供了一种**连续化的近似**来解决的离散化方法造成的不可导问题。

具体来说，Gumbel-Softmax **为离散变量提供了一种连续化的近似**，使得离散潜变量的**采样过程**可以进行梯度反向传播，而不需要依赖停止梯度策略。**dVAE 本身可以独立生成图像，不一定需要与生成模型（如 PixelCNN 或 Transformer）结合使用**。

## VQ-VAE 和 dVAE 的对比


| **特性**                     | **VQ-VAE**                                  | **dVAE**                                      |
|-----------------------------|---------------------------------------------|-----------------------------------------------|
| **离散化方法**               | 最近邻搜索                                   | Gumbel-Softmax                        |
| **不可导问题解决策略**       | 停止梯度传播                                 | 连续化近似                         |
| **端到端可微性**             | 部分可微                                     | 完全可微                                       |
| **训练效率**                 | 间接优化，编码器通过解码器接收反馈            | 高效优化，编码器直接接收梯度信号               |
| **潜变量表示**               | 离散嵌入向量                                 | 平滑 one-hot 表示                              |
| **灵活性**                   | 完全离散，固定 Codebook                     | 可调节连续与离散之间的平衡                     |
| **生成能力**                 | 通常需要结合生成模型（如 PixelCNN）            | 独立生成能力强                                 |
| **适用场景**                 | 离散建模，适合高分辨率图像生成或压缩任务       | 灵活生成，适合快速原型开发或连续采样任务        |


## 背景：VQ-VAE 的停止梯度策略

在 VQ-VAE 中，量化操作（如最近邻搜索）会将连续编码器输出 $z_e(x)$ 映射到离散 Codebook 中的某个嵌入向量 $e_k$：
$$
z_q(x) = e_k, \quad k = \text{argmin}_i \| z_e(x) - e_i \|_2
$$

由于最近邻搜索的离散性，梯度无法直接通过离散化操作反向传播到编码器，因此 VQ-VAE 使用 **停止梯度策略（Stop Gradient）** 来解决不可导问题：
- 解码器的梯度绕过量化操作，直接作用于编码器输出 $z_e(x)$。
- 停止梯度的关键公式：
  $$
  z_q(x) = z_e(x) + (e_k - z_e(x)).detach()
  $$
  - $(e_k - z_e(x)).detach()$ 表示量化部分的梯度被截断，编码器无法接收到量化过程的直接信号。

### 局限性
1. 停止梯度策略只是一种**间接优化**，编码器**无法完全利用量化**的反馈信号。
2. 这种间接的训练方法**可能导致训练效率较低或模型收敛较慢**。

---

## dVAE的结构
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af8def02cdb8418681ff69c1ecf9ce50.png)
[图片来源](https://medium.com/@zaiinn440/how-openais-dall-e-works-da24ac6c12fa)
## dVAE 引入 Gumbel-Softmax 替代停止梯度策略
论文：[CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144)

Gumbel-Softmax **为离散变量提供了一种连续化的近似**，使得**离散潜变量的采样过程可以进行梯度反向传播**，而**不需要依赖停止梯度策略**，同时保持端到端可微性。


### Gumbel 分布
Gumbel 分布是一种概率分布，用于建模极值（最大值或最小值）的分布情况。它经常在极值理论（Extreme Value Theory）中使用，描述数据集中最大值或最小值的分布特性。具体来说，Gumbel(0, 1) 是一种标准化的 Gumbel 分布，其位置参数为 0，尺度参数为 1。
1. **定义域**：Gumbel 分布定义在整个实数范围 $(-\infty, +\infty)$。
2. **极值理论**：Gumbel 分布常用于建模一组数据的极值（最大值或最小值）。
3. **标准化形式（Gumbel(0, 1)）**：
    - 位置参数 $\mu = 0$：分布的中心在 0。
    - 尺度参数 $\beta = 1$：控制分布的离散程度。


#### Gumbel和高斯分布对比

| **特点**                     | **Gumbel 分布**                              | **高斯分布**                                |
|------------------------------|---------------------------------------------|--------------------------------------------|
| **数据类型**                 | 离散（类别），离散分布的采样                                 | 连续（实数值），连续分布的采样                              |
| **应用场景**                 | 离散变量采样（如 dVAE, Gumbel-Softmax）          | 连续变量采样（如 VAE, 正态分布噪声）         |
| **目标**                     | 极值建模、离散采样                           | 数据建模、噪声建模                           |
| **可微性**                   | 结合 Softmax 可实现连续化，支持梯度反传       | 本身是连续分布，天然可微                    |
| **尾部行为**                 | 长尾分布，适合极值建模                       | 轻尾分布，适合一般建模                      |
| **数学特性**                 | 极值分布理论，适合最大值或最小值问题          | 中心极限定理，适合数据聚类和分布建模         |

### Gumbel-Softmax 采样过程

1. **使用 Gumbel 分布生成一组噪声样本：**
   - 对每个类别 $i$ 从 Gumbel 分布 $\text{Gumbel}(0, 1)$ 中采样噪声 $g_i$，模拟离散采样中的随机性。
2. **通过 Softmax 函数将这些噪声样本映射到一个类别分布：**
   - 将 logits（类别概率的对数）加上 Gumbel 噪声后通过 Softmax 转化为一个概率分布：
     $$
     y_i = \frac{\exp((\log(\pi_i) + g_i)/\tau)}{\sum_{j=1}^k \exp((\log(\pi_j) + g_j)/\tau)}
     $$
   - 输出的 $y$ 是一个平滑的概率分布，近似 one-hot 编码。
3. **温度控制：**
   - 温度系数 $\tau$ 控制分布的平滑程度：
     - 当 $\tau \to 0$：Softmax 逼近 ArgMax，输出接近离散 one-hot。
     - 当 $\tau \to \infty$：分布接近均匀，类别无显著差异。


#### Gumbel-Max Trick与Gumbel-Softmax区别

| **特性**               | **Gumbel-Max Trick**                        | **Gumbel-Softmax**                        |
|-----------------------|--------------------------------------------|------------------------------------------|
| **采样目标**           | 精确离散采样                                | 连续化近似采样                            |
| **输出形式**           | 单一类别（离散值）                          | 连续概率分布                              |
| **是否可微**           | 不可微                                      | 可微                                      |
| **适用场景**           | 需要离散采样的任务                          | 深度学习中的端到端训练任务                |
| **温度参数 $\tau$**    | 不涉及                                      | 控制分布的平滑程度                        |
| **推理阶段**           | 直接使用                                   | 通常替换为 $\text{argmax}$                |

---
### 采样过程详细介绍
这个过程是**可微分**的，因此可以在**反向传播中进行梯度计算**。

如下图所示：
1. 一个图像经 Encoder 编码会生成 32x32 个 embedding；
2. embedding 和 codebook （8192 个）进行内积；
3. 内积再经 Softmax 即可得到在每个 codebook 向量的概率。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/40428ea042a948ed81bbac1c9f4fec26.png)
[dVAE获取图像，并输出每个潜在特征的码本向量集上的分类分布](https://mlberkeley.substack.com/p/dalle2)

4. 应用 Gumbel Softmax 采样即可获得新的概率分布；
5. 然后将概率分布作为权重，对相应的 codebook 向量进行累积；就可以获得 latent vector。
6. 然后 Decoder 可以基于此 latent vector 重构输出图像。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/398cf36370424862808202a1f43e5e06.png)
[从Gumbel softmax分布中采样码本向量，然后将它们传递到解码器以重建原始的编码图像](https://mlberkeley.substack.com/p/dalle2)

在上述的过程中，通过**添加 Gumbel 噪声**的方式进行离散采样，可以近似为选择 logits 中概率最大的类别，从而**提供一种可微分的方式来处理离散采样问题**。具体来说，其关键为 Gumbel-Max Trick，其中 $g_i$  是从 Gumbel(0, 1) 分布中采样得到的噪声，τ 是温度系数。需要说明的是，t 越小，此处的 Softmax 就会越逼近于 ArgMax。τ 越大，就越接近于均匀分布。这也就引入了训练的一个 Trick：训练起始的温度系数 τ 很高，在训练的过程中，逐渐降低 τ，以便其逐渐逼近 ArgMax。在推理阶段就不再需要 Gumbel Softmax，直接使用 ArgMax 即可。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0838e41f7a2348e48fe16b04eb531535.png)
[图片来源](https://medium.com/@zaiinn440/how-openais-dall-e-works-da24ac6c12fa)


通过 Gumbel-Softmax，编码器输出的 logits 可以生成一个连续近似的 one-hot 表示 $y$，公式如下：

$$
y_i = \frac{\exp((\log(\pi_i) + g_i)/\tau)}{\sum_{j=1}^k \exp((\log(\pi_j) + g_j)/\tau)}, \quad g_i \sim \text{Gumbel}(0, 1)
$$

- **$\pi_i$：** 类别 $i$ 的概率，表示离散分布中类别 $i$ 被选择的概率，满足 $\pi_i > 0$ 且 $\sum_{i=1}^k \pi_i = 1$。
- **$\log(\pi_i)$：** 类别 $i$ 的对数概率，也称为 logits（表示每个类别的概率得分）。它是为了将概率值映射到对数空间，便于数值稳定性和与 Gumbel 噪声结合。
- **$g_i$：** 从 Gumbel 分布 $\text{Gumbel}(0, 1)$ 中采样的噪声，用于引入随机性，模拟离散采样过程。
- **$\tau$：** 温度参数，控制生成分布的平滑程度：
  - 当 $\tau \to 0$，$y_i$ 趋于 one-hot 表示（接近离散分布）。
  - 当 $\tau \to \infty$，$y_i$ 趋于均匀分布（所有类别的概率接近相等）。

---

### 端到端优化的实现
- Gumbel-Softmax 的输出是一个连续变量，可以近似离散的 one-hot 表示。
- 由于其公式中仅包含可导操作，梯度可以通过 Gumbel-Softmax 直接传递到编码器的 logits，实现端到端的可微优化。
- 不需要像 VQ-VAE 一样依赖停止梯度策略来绕过不可导的离散化操作。

---
### 替代的好处
1. **完全可微：**
   - Gumbel-Softmax 的连续近似使得整个模型可以端到端训练，而不需要手动截断梯度。
2. **更直接的优化：**
   - 编码器可以接收到更完整的梯度信号，而不是依赖解码器的间接反馈。
3. **灵活的离散化：**
   - 通过调节温度参数 $\tau$，可以在连续和离散之间找到平衡，进一步增强模型的优化能力。

---
总体而言，dVAE与VQ-VAE的目标相同：它们都试图学习复杂数据分布的离散潜在表示，例如自然图像的分布。每种方法都以自己独特的方式解决问题。VQ-VAE使用矢量量化，而dVAE将离散采样问题放宽为连续近似。虽然每种技术都有自己的一套权衡，但最终它们似乎都是解决这个问题的同样有效和同样成功的方法。



参考：
[文生图模型演进：AE、VAE、VQ-VAE、VQ-GAN、DALL-E 等 8 模型](https://blog.csdn.net/2401_84033492/article/details/139077927)
[【论文精读】DALLE: Zero-Shot Text-to-Image Generation零样本文本到图像生成](https://blog.csdn.net/weixin_47748259/article/details/136333875)
[【论文精读】DALLE2: Hierarchical Text-Conditional Image Generation with CLIP Latents](https://blog.csdn.net/weixin_47748259/article/details/136413045)
[【论文精读】DALLE3：Improving Image Generation with Better Captions 通过更好的文本标注改进图像生成](https://blog.csdn.net/weixin_47748259/article/details/136900416)
[AI绘画原理解析：从CLIP、BLIP到DALLE、DALLE 2、DALLE 3、Stable Diffusion(含ControlNet详解)](https://blog.csdn.net/v_JULY_v/article/details/131205615)

参考博文（DALL·E和dVAE的很多国内文章图片都来自下面的博文）：
[Understanding VQ-VAE (DALL-E Explained Pt. 1)](https://mlberkeley.substack.com/p/vq-vae)
[How is it so good ? (DALL-E Explained Pt. 2)](https://mlberkeley.substack.com/p/dalle2)
[How OpenAI’s DALL-E works?](https://medium.com/@zaiinn440/how-openais-dall-e-works-da24ac6c12fa)



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
[深度学习笔记——VQ-VAE和VQ-VAE-2](https://blog.csdn.net/haopinglianlian/article/details/144624499)
