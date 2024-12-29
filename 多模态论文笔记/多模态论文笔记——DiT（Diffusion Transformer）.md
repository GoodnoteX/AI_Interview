> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍Transformer架构图像生成方面的应用，将Diffusion和Transformer结合起来的模型：DiT。目前DiT已经成为了AIGC时代的新宠儿，视频和图像生成不可缺少的一部分。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5a33e7d6791b4c94a08ffa4768e183b0.png#pic_center)


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
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/affe4797c41e4e779ba05241ca8c1064.png)
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

## 与传统扩散的相同
 - DiT的整体框架并没有采用常规的Pixel Diffusion（像素扩散）架构，而是**使用和Stable Diffusion相同的Latent Diffusion（潜变量扩散）架构**，使用了和SD一样的**VAE**模型将像素级图像压缩到低维Latent特征。这极大地**降低了扩散模型的计算复杂度**（减少Transformer的token的数量）。

## 输入图像/条件信息的Patch化（Patchify）和位置编码
在图像领域使用Transformer，首先想到的模型就是ViT（参考：[深度学习笔记——ViT、ViLT](https://blog.csdn.net/haopinglianlian/article/details/144093215)），和ViT一样，DiT也需要经过**Patch**和**位置编码**，如下图红框。


<p align="center">
    <img src="https://i-blog.csdnimg.cn/direct/9c8ea2c0090f4a068b36b51a64e1096a.png" alt="图片描述" >
</p>

### Patch化
DiT和ViT一样，首先采用一个Patch Embedding来**将输入图像Patch化**，主要作用是将VAE编码后的**二维特征转化为一维序列**，从而**得到一系列的图像tokens**，ViT具体如下图所示：
![ViT模型架构示意图](https://i-blog.csdnimg.cn/direct/2e76918932c04cfda9145413c608e929.png)
DiT在这个图像Patch化的过程中，设计了**patch size超参数**，它直接决定了**图像tokens的大小和数量**，从而影响DiT模型的整体计算量。DiT论文中共设置了三种patch size，分别是 2, 4, 8。**patch size 为 2*2 是最理想的**。（结论来自：[视频生成Sora的全面解析：从AI绘画、ViT到ViViT、TECO、DiT、VDT、NaViT等](https://blog.csdn.net/v_JULY_v/article/details/136143475)）


Latent Diffusion Transformer结构中，输入的图像在经过VAE编码器处理后，生成一个Latent特征，Patchify的目的是将Latent特征转换成一系列 T 个 token（将Latent特征进行Patch化），每个 token 的维度为 d。Patchify 创建的 token 数量 T 由补丁大小超参数 p 决定。如下图所示，将 p 减半会使 T 增加四倍，因此至少使整个 transformer Gflops 增加四倍。具体流程如下图所示：

<p align="center">
    <img src="https://i-blog.csdnimg.cn/direct/0240b3cb396c44cabe3f5e6da817c824.png" alt="图片描述">
</p>

> **图4. DiT 的输入规格**  。给定patch size是 $p \times p$，空间表示（来自 VAE 的加噪潜变量），其形状为 $I \times I \times C$，会被“划分成补丁”（patchified）为一个长度为 $T = (I/p)^2$ 的序列，隐藏维度为 $d$。较小的补丁大小 $p$ 会导致序列长度更长，因此需要更多的计算量（以 Gflops 表示）。
### 位置编码
在执行 patchify 之后，我们对所有输入 token 应用标准的 ViT 基于频率的位置嵌入（正弦-余弦版本）。图像tokens后，还要加上Positional Embeddings进行位置标记，DiT中采用经典的**非学习sin&cosine位置编码技术**。

ViT（vision transformer）采用的是2D Frequency Embeddings（两个1D Frequency Embeddings进行concat操作），详情请参考：[深度学习笔记——3种常见的Transformer位置编码](https://blog.csdn.net/haopinglianlian/article/details/144021458)

## DiT Block模块详细信息
DiT在完成输入图像的预处理后，就要将Latent特征输入到DiT Block中进行特征的提取了，与ViT不同的是，DiT作为扩散模型**还需要**在Backbone（主干）网络中**嵌入额外的条件信息**（不同模态的条件信息等），这里的条件信息就包括了Timesteps以及类别标签（文本信息）。

DiT中的Backbone网络进行了两个主要工作：

 1. 常规的**特征提取**抽象； 
 2. 对图像和特征额外的**多模态条件特征进行融合**。

额外信息都可以采用一个Embedding来进行编码，从而注入DiT中。DiT论文中为了**增强特征融合**的性能，一共设计了~~四种~~ 【三种】方案来实现两个**额外Embeddings的嵌入**(说白了，就是怎么加入conditioning)。实现方式如下图模型架构的后半部分（红框）：
>【有的文章表示设计了四种，其实是将AdaLN和AdaLN-Zero分为两种，这里按照论文中图进行解释，分为三种】
>
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e7059feb714d43868e62fa5f02c2be21.png)

Diffusion Transformer模型架构图中由右到左的顺序分别是：

- 上下文条件（In-context conditioning）

- 交叉注意力块（Cross-Attention）

- 自适应层归一化块（Adaptive Layer Normalization, AdaLN）

下面将按顺序详细介绍。
### 上下文条件化
实现机制：
- 在上下文条件化中，**条件信息 \(t\)（时间步嵌入）** 和 **\(c\)（其他条件，如类别或文本嵌入）** 被表示为**两个独立的嵌入向量**。
- 这些向量被附加到输入图像 token 序列的开头，形成一个扩展后的输入序列。
- Transformer 模块对 \(t\) 和 \(c\) 的嵌入与图像 tokens 一视同仁，这些条件化 tokens 通过多头自注意力机制与图像 token 一起参与信息交换。

这与 **ViT 中的 cls tokens** 类似，它允许我们无需修改就使用标准的 ViT 模块。在最后一个模块之后，我们从序列中移除条件化 tokens。这种方法对模型的新 Gflops 增加可以忽略不计。
### 交叉注意力模块

实现机制：
- 条件信息 \(t\)（时间步嵌入） 和 \(c\)（其他条件，如类别或文本嵌入） 被拼接（concat）为**一个长度为 2 的序列**，**与图像 token 序列分开**。
- Transformer 模块被修改为在多头自注意力模块后添加一个 **多头交叉注意力层**，专门用于让图像 token 与条件 token 进行交互，从而将条件信息显式注入到图像特征中。
    - **图像**特征作为Cross Attention机制的**查询**（Query）。
    - **条件信息**的Embeddings作为Cross Attention机制的**键**（Key）和**值**（Value）。

这种方式是**Stable Diffusion**等文生图大模型常用的方式，交叉注意力对模型的 Gflops 增加最多，大约**增加了 15% 的开销**。
### adaLN-Zero 模块
首先需要了解什么是Adaptive Layer Normalization（AdaLN），而AdaLN之前又要先知道LN，下面将一步步优化讲解：
#### Layer Normalization（LN）
首先在理解AdaLN之前，我们先简单回顾一下Layer Normalization。
> 其他归一化方法参考：[深度学习——优化算法、激活函数、归一化、正则化](https://lichuachua.blog.csdn.net/article/details/142665273)

- **层归一化（Layer Normalization, LN）** 是 Transformer 中的一个关键组件，其作用是对输入的每个特征维度归一化，从而稳定训练和加速收敛。

Layer Normalization的处理步骤主要分成下面的三步：

1. 计算输入**权重的均值和标准差**：计算模型每一层输入权重的均值和标准差。

2. 对输入**权重进行标准化**：使用计算得到的均值和标准差将输入权重标准化，使其均值为0，标准差为1。

3. 对输入权重**进行缩放和偏移**：使用可学习的缩放参数和偏移参数，对标准化后的输入权重进行线性变换，使模型能够拟合任意的分布。


**LN 的公式**：
$$
  \text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$
  其中：
  - $x$：输入特征。
  - $\mu, \sigma$：输入的均值和标准差（按特征维度计算）。
  - $\gamma, \beta$：可学习的缩放和偏移参数，用于调整归一化后的分布。


在条件**生成任务（如扩散模型）中，需要让条件信息（如时间步 \(t\) 或类别 \(c\)）对生成过程产生影响**。为此，传统的 LN 被改进为 **自适应层归一化（adaLN）**，可以**动态调整归一化的参数**以**包含条件信息**。
#### Adaptive Layer Normalization（AdaLN）
在 GANs 和具有 UNet 骨干的扩散模型中广泛使用自适应归一化层之后，探索用自适应层归一化（adaLN）替换 transformer 模块中的标准归一化层。adaLN 并不是直接学习维度规模的**缩放和偏移参数 γ 和 β**，而是从 t 和 c 的嵌入向量之和中回归得到它们。

AdaLN的`核心思想`是**根据输入的不同条件信息**，**自适应地调整Layer Normalization的缩放参数 $\gamma$ 和偏移参数 $\beta$**，**增加的 Gflops 非常少，适合大规模任务**。

**adaLN 相比LN的改进**：
  - 不再直接使用**固定**的可学习参数 $\gamma, \beta$。
  - 相反，AdaLN 仅通过**动态生成**缩放参数 $\gamma_{\text{ada}}$ 和偏移参数 $\beta_{\text{ada}}$，以条件信息 $c$ 为输入，用于调整 Layer Normalization 的行为：
$$
\text{AdaLN}(x, c) = \gamma_{\text{ada}} \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta_{\text{ada}}
$$

    - **$\gamma_{\text{ada}} = f_\gamma(c)$**：由条件信息 $c$ 通过神经网络（如 MLP）生成的缩放参数。
    - **$\beta_{\text{ada}} = f_\beta(c)$**：由条件信息 $c$ 通过神经网络生成的偏移参数。
    说明：其中 **$f(c)$ 是一个小型神经网络**（通常是一个多层感知机，MLP），以条件信息（如时间步 $t$ 和类别 $c$ 的嵌入向量）为输入，输出对应的 $\gamma, \beta$。

##### AdaLN的核心步骤

AdaLN的核心步骤包括以下三步【详细的步骤会在下一节 adaLN-Zero的核心步骤总结】：

1. **提取条件信息**
从**输入的条件**（如Text Embeddings、标签等）中**提取信息**，一般来说会专门使用一个神经网络模块（比如全连接层等）来处理输入条件，并生成与输入数据相对应的缩放参数 $\gamma$ 和偏移参数 $\beta$。

2. **生成自适应的缩放和偏移参数** 
**利用提取的条件信息**，生成自适应的缩放和偏移参数。假设输入条件为 $c$，**经过一个神经网络模块**（比如全连接层等）生成缩放参数和偏移参数如下：  
$$
\gamma_{\text{ada}} = f_{\gamma}(c), \quad \beta_{\text{ada}} = f_{\beta}(c)
$$

3. **使用自适应参数** 
**使用这些自适应参数**对输入权重进行 Layer Normalization 处理：  
$$\text{AdaLN}(x, c) = \gamma_{\text{ada}} \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta_{\text{ada}}
$$

在我们探索的三种模块设计中，**adaLN 增加的 Gflops 最少，因此是最计算高效的**。它也是唯一一个限制对所有 tokens 应用相同函数的条件化机制。
#### adaLN-Zero

AdaLN-Zero 在 AdaLN 的基础上新增了 **残差缩放参数 $\alpha$**，用于**动态控制残差路径的影响**。通过**将 $\alpha$ 初始化为零，模型的初始状态被设置为恒等函数**，从而确保输出在训练初期的稳定性。这种设计显著提升了模型的训练稳定性和收敛速度。

> 之前的 ResNets 工作发现，将每个残差块初始化为恒等函数是有益的。例如，在监督学习环境中，将每个块中最后的批量归一化缩放因子 γ 零初始化可以加速大规模训练。
> 扩散 U-Net 模型使用了类似的初始化策略，在任何残差连接之前零初始化每个块中的最终卷积层。
> 我们探索了对 adaLN DiT 模块的修改，它做了同样的事情。除了回归 γ 和 β，我们还回归了在 DiT 模块内的任何残差连接之前作用的 dimension-wise 的缩放参数 α。初始化 MLP 以输出所有 α 为零向量；这将完整的 DiT 模块初始化为恒等函数。与标准的 adaLN 模块一样，adaLNZero 对模型的 Gflops 增加可以忽略不计。
##### adaLN-Zero的核心步骤
下面将根据下图来阐述：
<p align="center">
    <img src="https://i-blog.csdnimg.cn/direct/72bcfb2b88e149e2b2f872c091bdccdd.png" alt="图片描述">
</p>

- **AdaLN 有 4 个参数：$\gamma_1, \beta_1, \gamma_2, \beta_2$**，分别用于自注意力和 MLP 模块的归一化操作，**没有残差缩放参数**。
- **AdaLN-Zero 增加了 2 个参数：$\alpha_1, \alpha_2$**，用于控制残差路径的输出，显著提升了训练稳定性和适应性，因此总共 **6 个参数**【如上图】。
- 如果任务需要更强的稳定性（如深层 Transformer 模型或大规模扩散模型训练），**AdaLN-Zero 是更优的选择**。

adaLN-Zero的核心步骤包括以下三步，和adaLN的步骤相似，只不过需要在过程中加入**维度缩放参数 $\alpha$**。
> 需要做的**额外**工作如下：
>1. 在第一步中**提取回归缩放参数 $\alpha$**  
>2. 在第二步中**生成自适应的缩放参数 $\alpha$**
>3. 在第三步中**使用$\alpha$残差路径进行控制**

1. **提取条件信息、缩放参数 $\alpha$**：从输入的条件（如Text Embeddings、标签等）中提取信息，一般来说会**专门使用一个神经网络模块（比如全连接层等）来处理输入条件**，并生成与输入数据相对应的缩放和偏移参数。
在DiT的官方实现中，使用了一个**全连接层+SiLU激活函数**来**实现这样一个输入条件的特征提取网络**：

   ```python
   # 输入条件的特征提取网络
   self.adaLN_modulation = nn.Sequential(
       nn.SiLU(),
       nn.Linear(hidden_size, 6 * hidden_size, bias=True)
   )
   # c代表输入的条件信息
   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
   ```

   同时，DiT 在每个残差模块之后还使用了一个**回归缩放参数 $\alpha$** 来对权重进行缩放调整，这个 $\alpha$ 参数也是由上述**条件特征提取网络提取**的。

   上面的代码示例中和上图（DiT Block with adaLN-Zero）我们可以看到，adaLN_modulation **计算了 6 个变量** shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp，这 6 个变量分别对应了**多头自注意力机制(MSA)**的 AdaLN 的**归一化参数与缩放参数**（下图中的 $\beta_1$, $\gamma_1$, $\alpha_1$）以及 **MLP** 模块的 AdaLN 的**归一化参数与缩放参数**（下图中的 $\beta_2$, $\gamma_2$, $\alpha_2$）。
>【在 **DiT Block** 中，**MSA（多头自注意力模块）** 和 **MLP（多层感知机模块）** 都需要分别进行一次 **adaLN-Zero** 归一化处理。每个模块的 **Layer Normalization（LN）** 都会被替换为 **adaLN-Zero**，并且两者的 **归一化参数$(\gamma, \beta$）** 和 **残差路径缩放参数$(\alpha)$** 是独立的（**需要分别提取**），具体如下：】

>Transformer 模块由 **MSA 和 MLP** 两部分组成，而它们在功能上的分工导致必须对每部分单独设计对应的条件化机制。这是因为：
>- MSA 的核心在于捕捉全局依赖关系，因此其动态参数$(\beta_1, \gamma_1, \alpha_1)$主要控制 **全局特征的动态调整**。
>- MLP 的核心在于非线性特征提取，增强局部特征表达，因此其动态参数$(\beta_2, \gamma_2, \alpha_2)$主要用于控制 **局部特征的动态变换**。
>

 2. **生成自适应的缩放和偏移参数、缩放参数 $\alpha$**：  
利用提取的条件信息，生成自适应的缩放和偏移参数。假设输入条件为 $c$，**经过一个神经网络模块**（比如全连接层等）生成缩放参数和偏移参数如下：  
$$
\gamma_{\text{ada}} = f_{\gamma}(c), \quad \beta_{\text{ada}} = f_{\beta}(c)
$$
为残差路径增加一个新的维度缩放参数 $\alpha$，由条件信息动态生成：
$$
     \alpha_{\text{ada}} = f_{\alpha}(c)
$$
**初始化为零**：在训练开始时，$\alpha_{\text{ada}} = 0$，使得模块输出仅为主路径输出，实现**恒等初始化**。

 3. **使用自适应参数、缩放参数 $\alpha$**：  

   - 在 AdaLN 的基础上，加入 $\alpha_{\text{ada}}$ **对残差路径进行缩放控制**：
$$
     \text{AdaLN-Zero}(x, c) = \alpha_{\text{ada}} \cdot \left(\gamma_{\text{ada}} \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta_{\text{ada}}\right)
$$
     - **残差路径的输出被动态调节**：通过 $\alpha_{\text{ada}}$ 的逐步增加，残差路径的影响逐渐加强。
     - 当 $\alpha_{\text{ada}} = 0$ 时，整个模块行为等效于恒等函数。

初始化所有 α 为零向量；这将完整的 DiT 模块初始化为恒等函数。adaLNZero 对模型的 Gflops 增加可以忽略不计，与标准的 adaLN 模块一样。
##### 说明
- 公式 $\text{AdaLN-Zero}(x, c)$描述的是残差路径的动态调整过程，输出为 **Residual Path Output**。
- **完整的模块输出是路径输出与残差路径输出的加权和**（AdaLN完整的模块输出换成对应公式即可）：
$$
  \text{Output} = \text{Main Path Output} + \text{AdaLN-Zero}(x, c)
$$
- 这种联系**确保了主路径与残差路径的协同作用**，结合条件化调整和归一化机制，使模型更加稳定高效地处理生成任务。

### DiT中具体的初始化
设置如下所示：
1. 对DiT Block中的AdaLN和Linear层均采用参数0初始化。
2. 对于其它网络层参数，使用正态分布初始化和xavier初始化。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c867a317e344f2ebfa06582fb98f5ce.png)
>图5.比较不同的条件反射策略。adaLN-Zero在训练的各个阶段都优于交叉注意和情境条件反射。

DiT论文中对四种方案进行了对比试验，发现采用**AdaLN-Zero效果是最好的**，所以DiT**默认**都采用这种方式来嵌入条件Embeddings。与此同时，AdaLN-Zero也成为了基于DiT架构的AI绘画大模型的必备策略。

# U-ViT（U-Net Vision Transformer）
参考：此文U-ViT部分：[视频生成Sora的全面解析：从AI绘画、ViT到ViViT、TECO、DiT、VDT、NaViT等](https://blog.csdn.net/v_JULY_v/article/details/136143475#t25)

## DiT 和 U-ViT 的对比

| **特性**               | **DiT**                                        | **U-ViT**                                     |
|-----------------------|-----------------------------------------------|----------------------------------------------|
| **模型设计灵感**        | 基于 ViT 的纯 Transformer 架构                | **结合 U-Net 和 ViT 的混合架构**                 |
| **网络结构**            | 标准 Transformer 堆叠                         | Encoder-Transformer-Decoder 框架             |
| **局部特征建模**        | 依赖 Patch Embedding 和 MLP，局部建模较弱      | 使用 U-Net 的卷积模块，局部特征建模强         |
| **全局特征建模**        | 完全由 Transformer 捕捉**全局**上下文信息          | 通过嵌入 ViT 增强全局建模能力                |
| **跳跃连接（Skip）**     | 无跳跃连接                                    | **具有跳跃连接**，保留细粒度信息                 |
| **输入表示**            | Patch Embedding 序列化输入                     | 原始图像直接输入                             |
| **适用任务**            | 高分辨率潜在空间生成任务                       | 低分辨率生成任务                             |
| **计算复杂度**          | 随序列长度增加计算复杂度显著提升                | U-Net 局部操作高效，整体复杂度较低            |
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

