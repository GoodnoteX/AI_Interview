> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍3种常见的Transformer位置编码——正弦/余弦位置编码（sin/cos）、基于频率的二维位置编码（2D Frequency Embeddings）、旋转式位置编码（RoPE）


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36785fd5eb964d00a6ef6e8beef51fc9.png#pic_center)

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

给定图像的大小为 $H \times W$，编码维度为 $D$，这种 2D 编码的计算方式如下：

1. **沿行（Height）方向生成 1D Frequency Embeddings**：
   对行索引 $x \in [0, H-1]$，生成对应的正弦和余弦位置编码：
$$
   PE_{x, 2i} = \sin\left(\frac{x}{10000^{\frac{2i}{D}}}\right), \quad PE_{x, 2i+1} = \cos\left(\frac{x}{10000^{\frac{2i}{D}}}\right)
$$
2. **沿列（Width）方向生成 1D Frequency Embeddings**：
   对列索引 $y \in [0, W-1]$，同样生成正弦和余弦位置编码：
$$
   PE_{y, 2i} = \sin\left(\frac{y}{10000^{\frac{2i}{D}}}\right), \quad PE_{y, 2i+1} = \cos\left(\frac{y}{10000^{\frac{2i}{D}}}\right)
$$
3. 最终组合：
   - **拼接：最终维度为 \(2D\)**
   
      $$
           PE_{(x, y)} = \text{concat}(PE_x, PE_y)
      $$
      
   - **求和：最终维度为 \(D\)**
   
      $$
           PE_{(x, y)} = PE_x + PE_y
      $$
      
**说明：**
1. **分解二维结构：**
   - 图像的**二维空间**本质上可以**分解**为**行和列的两个独立维度**。因此，分别对行和列编码是一种有效的做法，既利用了图像的二维特性，又保持了实现的简单性。

2. **保持 Transformer 的通用性：**
   - Transformer 本质是基于序列操作的，而将二维图像划分为行和列的独立序列后，位置编码的计算方式可以**复用 NLP 中的正/余弦编码**。

3. **减少计算复杂度：**
   - 相较于直接生成每个位置$(x, y)$的二维正弦编码，这种方法的计算复杂度更低，同时效果相近。

---
## 旋转式位置编码（Rotary Position Embeddings, RoPE）

**Rotary Position Embeddings (RoPE)** 是一种基于旋转变换的位置编码方法，同时支持**绝对位置**和**相对位置**的建模。

**传统位置编码的局限**
1. **绝对位置编码（如正弦/余弦编码）**：
   - 提供固定的绝对位置信息。
   - 不能自然建模相对位置关系。
2. **相对位置编码**：
   - 能够建模相邻元素间的相对距离。
   - 但实现复杂度较高，尤其在长序列任务中开销较大。

**RoPE 的创新点**
RoPE 提出了**旋转式变换**的思路，通过将位置信息**直接嵌入到输入特征的投影空间**，既能高效建模**绝对位置**，又能自然捕捉**相对位置**关系。




### RoPE 的数学原理

### 输入特征与位置编码表示
- 输入向量 $\mathbf{x} \in \mathbb{R}^d$（$d$为特征维度），其维度分偶数、奇数两部分，对应正弦、余弦编码，$\text{PE}_i$在偶数维度为 $\sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$，奇数维度为 $\cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)$（$\text{pos}$是位置信息）。

### 旋转变换
- RoPE核心是对特征向量用二维旋转矩阵进行旋转操作，即 $\mathbf{x}_{\text{rot}} = \mathbf{R}(\theta) \cdot \mathbf{x}$，旋转矩阵 $\mathbf{R}(\theta)$ 角度 $\theta$ 与位置有关，作用于偶数、奇数维度输入特征有变化：$\begin{bmatrix}x_{\text{even}}' \\ x_{\text{odd}}'\end{bmatrix}=\begin{bmatrix}\cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta)\end{bmatrix}\begin{bmatrix}x_{\text{even}} \\ x_{\text{odd}}\end{bmatrix}$。

### 符号意义
- $\begin{bmatrix} x_{\text{even}} \\ x_{\text{odd}} \end{bmatrix}$：原始特征向量偶数、奇数维度，$\mathbf{x}$ 分解成偶数索引部分 $x_{\text{even}}$（如 0、2、4…维）和奇数索引部分 $x_{\text{odd}}$（如 1、3、5…维）。
- $\begin{bmatrix} x_{\text{even}}' \\ x_{\text{odd}}' \end{bmatrix}$：旋转后特征向量偶数、奇数维度，是嵌入位置信息后的特征表示。
- $\begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}$：二维旋转矩阵，用于旋转变换，旋转角度 $\theta$ 与位置（时间步或空间坐标等）相关。 

**相对位置的自然建模**
- 通过旋转变换，两个特征间的相对位置关系可以直接通过旋转角度差 $(\Delta \theta)$ 捕捉：
  $$
  \text{Attention}(\mathbf{q}, \mathbf{k}) = \text{dot}(\mathbf{q}_{\text{rot}}, \mathbf{k}_{\text{rot}}).
  $$
  - $\mathbf{q}_{\text{rot}}$ 和 $\mathbf{k}_{\text{rot}}$ 是经过 RoPE 编码的查询（Query）和键（Key）向量。
  - 相对位置差的建模通过旋转后的内积自然实现。
### RoPE 的实现步骤
#### 1. 计算旋转角度
根据输入位置 $\text{pos}$ 和维度 $d$ 生成旋转角度。

**公式**
每个维度的旋转角度通过以下公式计算：
$$
\theta_{i} = \frac{\text{pos}}{10000^{2i/d}},
$$
其中：
- $\text{pos}$：输入特征的位置索引（如序列中的时间步或图像的空间位置）。
- $d$：特征向量的总维度。
- $i$：当前特征维度的索引。

**过程**
1. **分解频率因子**：
为不同的维度 \(i\) 生成对应的频率因子：
    $$
         \frac{1}{10000^{2i/d}}
    $$
     - 其中 $d$ 控制总维度范围内的频率分布：
     - 较**低维度**的频率变化较慢（低频），适合建模**全局信息**。
     - 较**高维度**的频率变化较快（高频），适合捕捉**局部细节**。

2. **结合位置计算角度**：对于每个位置 $\text{pos}$，乘以频率因子以生成旋转角度：
    $$
         \theta_{i} = \text{pos} \cdot \frac{1}{10000^{2i/d}}
    $$
     - 不同位置的旋转角度反映了其空间或时间位置信息。

**结果**
- 每个位置 $\text{pos}$ 和每个维度 $i$ 对应一个独特的旋转角度 $\theta_{i}$。
- 输出是一个长度为 $d$ 的旋转角度数组。

---

#### 2. 构造旋转矩阵

旋转矩阵用于**将偶数维和奇数维的特征进行二维旋转嵌入**。**每对偶数维和奇数维**被看作一个二维向量。

**公式**
二维旋转矩阵的形式为：
$$
\mathbf{R}(\theta) =
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$
**过程**
1. **匹配每个维度的角度**：
   - 根据上一步计算的旋转角度 $\theta_i$，生成每对偶数维和奇数维的旋转矩阵。

2. **作用对象**：
   - 偶数维 $\text{even}(i)$ 和奇数维 $\text{odd}(i+1)$ 被看作一个二维向量：
$$
     \mathbf{x}_{\text{even}}, \mathbf{x}_{\text{odd}}
$$

3. **生成旋转变换**：
   - 使用 $\cos(\theta_i)$ 和 $\sin(\theta_i)$ 填充旋转矩阵。

---

#### 3. 旋转变换

将旋转矩阵作用于特征向量的偶数维和奇数维，以嵌入位置信息。

**公式**
旋转后的特征向量表示为：
$$
	\begin{bmatrix}
	x_{\text{even}}' \\
	x_{\text{odd}}'
	\end{bmatrix}
	=
	\begin{bmatrix}
	\cos(\theta) & -\sin(\theta) \\
	\sin(\theta) & \cos(\theta)
	\end{bmatrix}
	\cdot
	\begin{bmatrix}
	x_{\text{even}} \\
	x_{\text{odd}}
	\end{bmatrix}.
$$

**过程**
1. **输入特征**：
   - 输入特征 $\mathbf{x}$ 被分解为偶数维和奇数维两部分：
$$
     \mathbf{x} = [x_{\text{even}}, x_{\text{odd}}]
$$

2. **旋转变换**：
   - 对于每对偶数维和奇数维：
$$
     x_{\text{even}}' = x_{\text{even}} \cdot \cos(\theta) - x_{\text{odd}} \cdot \sin(\theta),
$$
$$
     x_{\text{odd}}' = x_{\text{even}} \cdot \sin(\theta) + x_{\text{odd}} \cdot \cos(\theta).
$$
   - 旋转后的特征将位置信息嵌入到每个维度中。

3. **重组特征**：
   - 将旋转后的偶数维和奇数维重新合并，得到嵌入了位置信息的特征向量。

---

#### 4. 自注意力机制

使用旋转后的特征向量参与自注意力计算，在 Attention 的点积操作中显式建模 **绝对位置** 和 **相对位置信息**。

**自注意力公式**
自注意力的计算公式为：
$$
\text{Attention}(\mathbf{q}, \mathbf{k}) = \mathbf{q} \cdot \mathbf{k}
$$
- $\mathbf{q}$：查询向量（Query）。
- $\mathbf{k}$：键向量（Key）。

**RoPE 的贡献**
1. **绝对位置信息**：旋转变换后的 $\mathbf{q}$ 和 $\mathbf{k}$ 包含绝对位置信息，使模型能够感知每个特征的位置。

2. **相对位置信息**：
点积中隐含了旋转角度差 $\Delta \theta = \theta_2 - \theta_1$：
    $$
         \cos(\Delta \theta) + \sin(\Delta \theta)
    $$
     - $\Delta \theta$ 是两位置间的相对关系，直接体现在注意力值中。

---

### RoPE 的优点

1. **高效性**：
   - 不需要复杂的相对位置偏移矩阵或附加参数，直接通过旋转实现。
   - 适合长序列任务，计算复杂度低。

2. **支持绝对与相对位置**：
   - 旋转式编码不仅能捕捉绝对位置，还能通过旋转角度差捕捉相对位置关系。

3. **适配多模态任务**：
   - RoPE 能同时适用于文本、图像、视频等多模态场景的位置编码需求。
   - 在 FLUX.1 中，用于处理文本的序列关系和图像的空间关系。

4. **自然的时空特性建模**：
   - 在视频任务中，可扩展为三维旋转式编码，处理时间维和空间维的关系。


### 应用场景

1. **多模态任务**：
   - 在 FLUX.1 中，用于图像和文本模态的联合处理：
     - 文本位置被编码为序列信息。
     - 图像位置被编码为二维空间关系。

2. **视频生成**：
   - 支持视频任务的时空建模，可将时间维引入位置编码。

3. **长序列任务**：
   - 如文本生成、长文档理解中，RoPE 能显著提升相对位置的建模能力。

### 总结

旋转式位置编码（RoPE）是一种高效、灵活的位置编码方案：
- **核心机制**：通过二维旋转矩阵嵌入位置信息，既能建模绝对位置，又能自然捕捉相对位置。
- **适用场景**：从长序列任务到多模态场景，再到视频生成，RoPE 展现出强大的扩展性和适配能力。




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
[深度学习笔记——前向传播与反向传播、神经网络（前馈神经网络与反馈神经网络）、常见算法概要汇总](https://blog.csdn.net/haopinglianlian/article/details/143835406)
[深度学习笔记——卷积神经网络CNN](https://blog.csdn.net/haopinglianlian/article/details/143841327)
[深度学习笔记——循环神经网络RNN、LSTM、GRU、Bi-RNN](https://blog.csdn.net/haopinglianlian/article/details/143841402)
[深度学习笔记——Transformer](https://blog.csdn.net/haopinglianlian/article/details/143841447)




