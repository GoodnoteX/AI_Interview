> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍Transformer架构图像生成方面的应用，将Diffusion和Transformer结合起来的模型：DiT。目前DiT已经成为了AIGC时代的新宠儿，视频和图像生成不可缺少的一部分。

![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/多模态论文笔记/image/2.png)


> @[toc]
> 
## 论文
[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
## 定义
DiT是**基于Transformer架构的扩散模型**。用于各种图像（SD3、FLUX等）和视频（Sora等）生成任务。

DiT证明了**Transformer思想与扩散模型结合的有效性**，并且还验证了**Transformer架构在扩散模型上具备较强的Scaling能力**，在稳步增大DiT模型参数量与增强数据质量时，DiT的生成性能稳步提升。

>其中最大的DiT-XL/2模型在ImageNet 256x256的类别条件生成上达到了当时的SOTA【最先进的（State Of The Art）】（FID为2.27）性能。同时在SD3和FLUX.1中也说明了较强的Scaling能力。

## 架构
DiT架构如下所示：
<img width="959" alt="image" src="https://github.com/user-attachments/assets/890976b0-b07f-4a16-b264-45608e505831" />

> 图3.扩散Transformer（DiT）架构。左：我们训练条件潜在DiT模型。输入的潜在被分解成补丁和处理的几个DiT块。右图：DiT区块的详细信息。我们用标准Transformer块的变体进行了实验，这些块通过**自适应层归一化**、**交叉注意**和**额外输入的令牌（上下文环境）** 来进行调节，其中**自适应层规范效果最好**。

- **左侧主要架构图**：训练条件潜在DiT模型(conditional latent DiT models)， **潜在输入**和条件被**分解成patch**并结合**条件信息**通过几个DiT blocks处理。本质就是噪声图片减掉预测的噪声以实现逐步复原。
  - **DiT blocks前：** 比如当输入是一张256x256x3的**图片**，得到32x32x4的**Noised Latent**，之后进行**Patch**和**位置编码**，结合当前的**Timestep t、Label y**作为输入。
  - **DiT blocks后：** 经过N个Dit Block(基于transformer)通过MLP进行输出，在 DiT 模型的最后一个 Transformer 块（DiT block）之后，需要将生成的图像 token 序列解码为以下两项输出：噪声“**Noise预测**”以及**对应的协方差矩阵**，最后经过T个step采样，得到32x32x4的**降噪后的latent**。

- **右侧DiT Block实现方式**：DiT blocks的细节，作者试验了标准transformer块的变体，这些变体通过**自适应层归一化**、**交叉注意力**和**额外输入token**来加入条件(incorporate conditioning via adaptive layer norm, cross-attention and extra input tokens，这个coditioning相当于就是带条件的去噪)，其中**自适应层归一化效果最好**。


> `下文将按照这个架构进行阐述，从左到右。`

## 与传统(U-Net)扩散模型区别
首先我们先来对比一下DiT与传统(U-Net)扩散模型区别
### 架构
- DiT将扩散模型中经典的**U-Net**架构完全**替换**成了**Transformer**架构。能够**高效地捕获数据中的依赖关系**并生成高质量的结果。
### 噪声调度策略
- DiT扩散过程的采用简单的Linear scheduler（timesteps=1000，beta_start=0.0001，beta_end=0.02）。在传统的U-Net扩散模型（SD）中，所采用的noise scheduler通常是Scaled Linear scheduler。

>TODO： 【也有说在传统的U-Net扩散模型（SD）中，所采用的noise scheduler是**带调优参数**后的**线性调度器**（Linear Scheduler）。】





> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)
