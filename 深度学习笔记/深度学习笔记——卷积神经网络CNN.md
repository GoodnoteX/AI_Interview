﻿> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文详细介绍面试过程中可能遇到的卷积神经网络CNN知识点。


![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/深度学习笔记/image/5.png)


@[toc]

卷积神经网络 (CNN) 是一种专门用于**处理图像、视频**等数据的深度学习模型，主要用于**计算机视觉任务**，例如图像分类、目标检测和图像生成。CNN 通过卷积操作减少输入数据的尺寸，**提取出重要特征，同时保留其空间结构** ，尤其在处理高维数据时非常有效。

# 主要组件
## 输入层
   输入层用于接收原始数据，例如图像（二维或三维张量）。对于图像，通常是像素值。

## 卷积层 (Convolutional Layer)
   卷积层是 CNN 的核心组件，负责通过卷积核 (filter) **提取输入数据的局部特征** 。通过扫描输入的局部区域，卷积层可以识别特定的模式（例如边缘、角等）。卷积操作通常会**产生一组特征图**。

## 批归一化层（Batch Normalization, BN）

归一化**使训练过程更加稳定和高效**。BN 层将激活值标准化为**均值为 0、标准差为 1** 的分布，然后通过可学习的**缩放**和**平移**参数恢复数据分布。**防止梯度消失和梯度爆炸问题**。加速训练，允许使用更大的学习率。在一定程度上起到**正则化作用，减少过拟合**。
> 参考历史/后续文章【归一化部分】：[深度学习笔记——归一化、正则化]

## 激活函数 (Activation Function)
   常见的激活函数是**ReLU** 及其变体（如 Leaky ReLU、PReLU 和 ELU）。其他函数如 **Swish** 也逐渐流行（SD模型组件中GSC中的S指的就是Swish）。**Sigmoid 和 Tanh 因梯度消失问题较严重**，不适合深层 CNN 网络，因此使用较少。

## 池化层 (Pooling Layer)
   池化层用于**缩减数据的尺寸，同时保留主要特征**。最大池化（Max Pooling）是最常用的方式，它通过取每个区域内的最大值来减少数据量。这有助于减小计算量，并增强模型的平移不变性。

## 全连接层 (Fully Connected Layer)
   全连接层是CNN 的**特征整合部分**，将高维特征压缩并组合，最终**生成一个用于输出处理的向量**，**连接到输出层**。通常在 CNN 的最后几层用于**将提取到的特征映射到最终的分类或回归结果**。
 
## 批归一化层（Batch Normalization, BN）
## 输出层激活函数 (Output Layer Activation Function)
- **二分类问题**：输出层通常输出一个经过 **sigmoid 函数** 处理的值，该值表示预测属于某个类别的概率，范围在 0 到 1 之间。
  
- **多分类问题**：输出层通常使用 **softmax 函数**，将全连接层的输出向量转换为概率分布，表示样本属于不同类别的概率。概率的总和为 1。

- **回归任务**：输出层直接输出一个或多个实数，表示模型的预测值。回归任务的输出层通常不使用激活函数，或者使用**线性激活函数**，允许输出为任意实数。
## 输出层 (Output Layer)

输出层的作用是**生成最终的预测结果**。

---

# 整体流程概述

1. **输入层**：输入图像数据，可能是灰度或彩色图像。
2. **卷积层**：通过卷积核提取局部特征。
3. **批归一化**（可选）：对每一批次进行归一化，防止梯度爆炸或消失，加速训练。
4. **激活函数**：如 ReLU 引入非线性。
5. **池化层**：对特征图进行下采样，降低特征图尺寸，减少计算复杂度。
6. **重复卷积～池化层**：逐层提取更高层次的特征。
7. **全连接层**：将高层特征映射到输出空间，通常用于分类任务。
8. **批归一化**（可选）：对每一批次进行归一化，防止梯度爆炸或消失，加速训练。
9. **激活函数**：如 ReLU 引入非线性。
10. **丢弃层**（可选）：随机丢弃部分神经元，防止过拟合。
11. **输出层**：输出最终的预测结果，常用于分类或回归任务。
12. **损失函数**：衡量模型输出与真实值之间的差距。
13. **反向传播和优化**：更新模型权重，使损失函数最小化。

通过这一系列的操作，CNN 能够逐层提取图像的特征，最终输出图像的分类或回归结果。
# 激活函数
激活函数 (Activation Function) 通常应用在 **卷积层** 和 **全连接层** 后面。作用是**引入非线性，使模型能够学习和表示更复杂的特征**。也可以用在**输出层中**对输出结果进行映射，下面是激活函数在 CNN 不同层中的应用情况：
## 使用激活函数的位置
### 卷积层后
   - 在每次卷积操作之后，通常会应用激活函数（如 ReLU），以对卷积输出进行非线性变换。
   - **卷积操作本身是线性的**，即卷积核和输入的乘法与加法，所以如果没有激活函数，整个模型仍然是线性的，**无法处理复杂的非线性问题**。因此，**激活函数**是卷积神经网络中的重要组成部分，它**使网络具有表达复杂模式的能力**。

### 全连接层后
   - 全连接层通常**在 CNN 的最后几层出现**。类似卷积层，全连接层之后也需要应用激活函数，以引入非线性。
   - 激活函数可以**帮助全连接层进行分类或回归任务的特征表示**。常见的做法是在每个全连接层之后使用 ReLU 激活函数，除了最后的输出层。

### 输出层中
   - **分类任务**：对于**多分类**任务，输出层通常使用 **softmax** 作为激活函数，将输出转化为概率分布。对于**二分类**任务，输出层可以使用 **sigmoid** 激活函数，将输出限制在 0 和 1 之间。
   - **回归任务**：如果是回归问题，输出层可能**不使用激活函数**，或者使用线性激活函数。

常见的激活函数
- **ReLU (Rectified Linear Unit)**: 最常用的激活函数，公式为 $f(x) = \max(0, x)$，使负数归零，仅保留正数。
- **Sigmoid**: 常用于二分类输出层，将输出值压缩到 \( [0, 1] \) 区间。
- **Tanh**: 输出值在 \( [-1, 1] \) 之间，用于在特定任务中提供更强的梯度。
- **Softmax**: 用于多分类任务的输出层，将输出值转化为概率分布，总和为 1。

## 不使用激活函数的层

激活函数主要应用于 卷积层 和 全连接层后，**池化层和批归一化层不使用激活函数**。下面是各层是否使用激活函数的详细情况：
### 池化层 (Pooling Layer)
- 池化层的主要作用是通过下采样（如最大池化或平均池化）**减少卷积层特征图的尺寸和计算量**，同时**保留主要的特征**。池化层的操作本身是固定的，因为它的功能只是下采样特征图，保持特征的局部不变性。
### 批归一化层 (Batch Normalization Layer)
批归一化的作用是对**每一批次的输入进行归一化**，以**加速训练和提高稳定性**。它通过**标准化激活值**（使得激活值的均值为 0，标准差为 1）和加入可学习的**缩放**和**平移**参数来调整数据分布。**批归一化本身不引入非线性，但通常在激活函数之前使用**。

## 总结
激活函数主要应用在 **卷积层**和 **全连接层**之后，它们的作用是**引入非线性**，从而帮助 CNN 模型**学习更加复杂的特征和模式**。在最后的**输出层**，根据任务类型，可能会使用特定的激活函数，如 softmax 或 sigmoid。而像池化层和批归一化层这样的层次，通常不会使用激活函数。

---
# 卷积层的基本参数

卷积层的几个关键参数如下：
## 卷积核 (Filter/Kernel) 
   卷积核是一个小的矩阵，用于扫描输入数据的局部区域。卷积核的尺寸决定了**每次处理**的**局部区域大小**，常见的卷积核尺寸为 **3x3**、**5x5** 等。

## 步长 (Stride)
   步长决定了**卷积核**在输入数据上**每次移动的步长**。如果步长为 1，卷积核每次移动 1 个像素。**较大的步长会减少输出特征图的尺寸**。
## 填充 (Padding)
   填充用于**控制卷积操作后输出尺寸**。可以选择**在输入数据的边缘填充额外的像素** 。常见的填充方式有三种：

1. **"valid" 填充**：无填充，输出尺寸变小，适合逐步减少特征图尺寸。
2. **"same" 填充**：填充边缘，保持输出尺寸与输入尺寸相同，适合需要保持尺寸不变的场景。
3. **部分填充**：在这种情况下，只填充一部分（少于 "same" 填充所需的数量），因此输出尺寸会比输入尺寸小。
## 深度 (Depth)
   卷积层的深度是**指卷积核的数量**。每个卷积核提取输入数据的不同特征，因此深度越大，提取的特征越多。卷积层的深度对应于**输出特征图的数量**。
## 公式化表示卷积操作输出特征图的尺寸

卷积操作后输出特征图的尺寸可以通过以下公式计算：
$$ \text{Output Size} = \frac{(W - F + 2P)}{S}+ 1 $$
- $W$：输入图像的宽度或高度
- $F$：卷积核的尺寸（通常为方形，如3×3、5×5）
- $P$：填充（padding）的大小
- $S$：步长（stride）的大小






> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)



