﻿> 大家好，这里是好评笔记，公主号：Goodnote，专栏文章私信限时Free。本文主要阐述Boosting中常用算法（GBDT、XGBoost、LightGBM）的迭代路径。

![在这里插入图片描述](https://github.com/GoodnoteX/Ai_Interview/blob/main/机器学习笔记/image/9.png)

---
> @[toc]
---
# XGBoost 相对 GBDT 的改进
**GBDT（Gradient Boosting Decision Tree，梯度提升决策树）** 是一种集成学习算法。GBDT 使用**梯度提升（Gradient Boosting）**的思想，每一棵**决策树**都是基于前一轮预测的**残差**（即误差）来训练的，从而逐步逼近真实值。

**XGBoost** 相对传统 GBDT 在原理和实现上进行了多项改进，使得它在计算效率、模型精度、内存管理和并行性等方面有显著提升。以下是 XGBoost 相对 GBDT 的关键改进：

| 改进项                  | 具体描述                                                   | 优势                               |
|-------------------------|------------------------------------------------------------|------------------------------------|
| **正则化项**            | 引入 \($L_1$\) 和 \($L_2$\) 正则化控制叶节点权重                 | 防止过拟合，提高泛化能力           |
| **二阶导数信息**        | 使用梯度和 Hessian 信息进行优化                              | 提高收敛速度和精度                 |
| **列采样和行采样**      | 每棵树采样特征和样本                                        | 降低过拟合风险，提高泛化性         |
| **并行化处理**          | 特征分片和直方分裂，支持 GPU 加速                           | 提升训练速度                       |
| **缺失值处理**          | 自动选择缺失样本最佳分裂方向                                | 处理缺失值和稀疏数据               |
| **早停和学习率衰减**    | 监控验证集性能，学习率衰减控制每棵树贡献                     | 降低过拟合和节省计算开销           |
| **自定义目标和评估**    | 支持用户自定义目标函数和评估指标                             | 提高适应性，满足不同场景需求       |

## 引入正则化项，防止过拟合
- 在 GBDT 中，每棵树的叶子节点权重没有额外的正则化控制，容易导致模型过拟合。**XGBoost** 在每棵树的目标函数中**引入了 \( $L_1$ \) 和 \( $L_2$ \) 正则化项**，控制叶节点数量和权重大小，使模型更具泛化能力。目标函数为：
<img width="469" alt="image" src="https://github.com/user-attachments/assets/c120c49d-3711-4b12-89c4-7a898f6a9e8e" />


### 损失函数 $L(y_i, \hat{y}_i)$

损失函数 $L(y_i, \hat{y}_i)$ 测量每个样本的预测误差。例如，常用的损失函数有：
- **均方误差 (MSE)**：用于回归问题，定义为 $L(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2$。
- **对数损失 (Log Loss)**：用于二分类问题 $L(y, \hat{y}) = - \left( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)$。

### 正则化项 $\Omega(f_m)$
正则化项 $\Omega(f_m)$ 用于控制模型复杂度，包含 L1 和 L2 正则化：

$$
\Omega(f_m) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|
$$

其中：
- $T$ 是树的叶节点数目。
- $w_j$ 是第 $j$ 个叶节点的权重。
- $\gamma$ 控制叶节点的数量，较大的 $\gamma$ 倾向于减少叶节点数量，使模型更简单。
- $\lambda$ 是 L2 正则化系数，控制叶节点权重的平方和，有助于平滑叶节点的权重。
- $\alpha$ 是 L1 正则化系数，控制叶节点权重的绝对值之和，促使部分权重趋向于零，从而构建更稀疏的模型。

## 使用二阶导数信息，加速收敛
GBDT 只使用损失函数的一阶导数（负梯度）来更新模型，而 XGBoost 通过泰勒展开，将损失函数对模型输出进行二阶展开，使用**一阶导数（梯度）和二阶导数（Hessian）** 来**构建树**，优化效果更佳。**二阶导数带来更精确的梯度信息来改进模型的更新，使得模型能够更快地收敛**。目标函数的二阶近似为：
$$
\text{Obj}^{(m)} \approx \sum_{i=1}^N \left[ g_i f_m(x_i) + \frac{1}{2} h_i f_m(x_i)^2 \right] + \Omega(f_m)
$$

其中：
- $g_i = \frac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i}$ 是损失函数的一阶导数（梯度）。
- $h_i = \frac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$ 是损失函数的二阶导数（Hessian）。

### 一阶导数与二阶导数的区别

 1. 一阶导数（梯度）：**表示损失函数的斜率**，指向使损失减少的方向，即当前点处的损失下降趋势。梯度**通常用于确定优化的方向**。
 2. 二阶导数（Hessian）：**表示一阶导数的变化率**，反映了**损失曲面的曲率信息**。 Hessian 的大小可以**帮助判断步长的合适大小**，使得更新**更加精确**。
### 二阶导数的优势
1. 更准确的步长控制：根据损失曲率调整步长，避免步长过大或过小。 
2. 更快速的收敛：二阶导数的曲率信息能帮助模型更快接近最优解，减少迭代次数。
3. 更稳定的优化过程：使得模型在平坦或复杂的损失函数下依然能够有效地更新。

#### 1. 更准确的步长选择，避免过大或过小的更新
使用二阶导数信息可以调整步长，使得模型更新在合适的尺度上进行。步长过大会导致优化震荡或跳过最优解，而步长过小会导致优化缓慢。引入二阶导数可以：





> 详细全文请移步公主号：Goodnote。  
参考：[欢迎来到好评笔记（Goodnote）！](https://mp.weixin.qq.com/s/lCcceUHTrM7wOjnxkfrFsQ)
