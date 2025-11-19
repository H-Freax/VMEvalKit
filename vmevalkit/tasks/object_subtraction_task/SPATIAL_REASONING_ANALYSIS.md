# Spatial Reasoning 测试目标分析

## 概述

在VMEvalKit框架中，**Spatial Reasoning（空间推理）**是一个核心的认知能力测试维度。本文档分析spatial reasoning在object_subtraction_task中的测试目标和目的。

---

## 文献中的Spatial Reasoning测试目标

### 1. Rotation Task (3D Mental Rotation) 的测试目标

根据`rotation_task/ROTATION.md`，3D Mental Rotation Task专门测试spatial reasoning：

#### 核心测试能力：

**A. Mental Rotation Ability（心理旋转能力）**
- **测试内容**：180°水平旋转，提供最大视角变化
- **测试目标**：
  - 测试模型能否在心理上转换3D对象
  - 评估对相反视角的理解能力
  - 测试空间变换的推理能力

**B. 3D Visualization（3D可视化能力）**
- **测试内容**：从倾斜的2D投影理解复杂的3D结构
- **测试目标**：
  - 从单眼线索理解深度感知
  - 从部分视图完成形状理解
  - 理解3D空间关系

**C. Perspective Taking（视角转换）**
- **测试内容**：相机中心vs对象中心的参考框架
- **测试目标**：
  - 在移动过程中保持一致的视角
  - 理解参考框架的转换
  - 空间参考系统的理解

#### 模型能力测试：

**Geometric Understanding（几何理解）**
- 3D空间关系（voxels之间的空间关系）
- 连通性和邻接推理
- 体积形状表示

**Temporal Reasoning（时间推理）**
- 规划平滑的旋转轨迹
- 插值中间视角
- 保持时间一致性

**Visual Consistency（视觉一致性）**
- 旋转过程中的对象持久性
- 一致的照明和阴影
- 正确的遮挡顺序

---

## Object Subtraction Task Level 3 的Spatial Reasoning测试目标

### 当前实现

Level 3: **Relational Reference（关系引用）** 专门测试空间推理能力。

#### 测试的空间关系类型：

**1. 方向性空间关系（Directional Spatial Relations）**
- `leftmost` / `rightmost`: 最左侧/最右侧的对象
- `topmost` / `bottommost`: 最上方/最下方的对象
- `left_side` / `right_side`: 左侧/右侧的对象
- `top_half` / `bottom_half`: 上半部分/下半部分的对象

**测试目标**：
- ✅ 理解"左"、"右"、"上"、"下"等方向概念
- ✅ 理解相对位置（相对于画布中心或边界）
- ✅ 理解"最"的概念（极值位置）

**2. 象限关系（Quadrant Relations）**
- `top_left_quadrant` / `top_right_quadrant`
- `bottom_left_quadrant` / `bottom_right_quadrant`

**测试目标**：
- ✅ 理解2D空间的象限划分
- ✅ 理解对象在哪个象限
- ✅ 理解象限边界（需要严格判断，避免歧义）

**3. 距离关系（Distance Relations）**
- `farthest_from_center`: 距离中心最远的对象
- `nearest_to_center`: 距离中心最近的对象
- `corner_closest`: 最接近角落的对象

**测试目标**：
- ✅ 理解"距离"的概念
- ✅ 计算对象到参考点的距离
- ✅ 理解"最远"、"最近"等相对距离概念
- ✅ 理解"中心"、"角落"等空间参考点

---

## Spatial Reasoning的核心测试目标总结

### 1. **空间概念理解（Spatial Concept Understanding）**

**测试内容**：
- 方向概念：左、右、上、下
- 位置概念：象限、中心、角落
- 距离概念：远、近、最远、最近

**测试目标**：
- 模型能否理解这些空间概念？
- 模型能否将语言中的空间描述映射到视觉空间？

### 2. **相对位置推理（Relative Position Reasoning）**

**测试内容**：
- 对象之间的相对位置
- 对象相对于参考点的位置
- 对象相对于边界的位置

**测试目标**：
- 模型能否计算相对位置？
- 模型能否理解"相对于什么"的概念？
- 模型能否进行空间比较？

### 3. **空间计算能力（Spatial Computation）**

**测试内容**：
- 计算距离（欧几里得距离）
- 判断象限（基于坐标）
- 排序位置（最左、最右等）

**测试目标**：
- 模型能否进行空间计算？
- 模型能否理解坐标系统？
- 模型能否进行空间排序？

### 4. **空间抽象能力（Spatial Abstraction）**

**测试内容**：
- 从具体位置抽象出空间关系
- 理解空间模式（如"所有左侧对象"）
- 理解空间集合（如"象限内的所有对象"）

**测试目标**：
- 模型能否从具体坐标抽象出空间关系？
- 模型能否理解空间集合的概念？
- 模型能否进行空间模式识别？

---

## 与Rotation Task的区别

### Rotation Task (3D Mental Rotation)
- **维度**：3D空间
- **测试重点**：3D可视化、心理旋转、视角转换
- **复杂度**：需要理解3D结构、相机运动、视角变化
- **应用**：测试高级3D空间推理能力

### Object Subtraction Level 3 (2D Spatial Relations)
- **维度**：2D空间
- **测试重点**：2D空间关系、相对位置、距离计算
- **复杂度**：需要理解2D坐标、方向、象限
- **应用**：测试基础到中级的2D空间推理能力

**互补关系**：
- Rotation Task测试**3D空间推理**（更高级）
- Object Subtraction L3测试**2D空间推理**（更基础）
- 两者共同覆盖了spatial reasoning的不同维度

---

## 为什么Spatial Reasoning重要？

### 1. **基础认知能力**
- Spatial reasoning是许多高级认知任务的基础
- 理解空间关系是理解世界的关键能力

### 2. **视频生成的关键能力**
- 视频模型需要理解空间关系来生成合理的视频
- 空间一致性是视频质量的重要指标

### 3. **实际应用价值**
- 导航、机器人、AR/VR等应用都需要spatial reasoning
- 理解空间关系是智能系统的基本能力

### 4. **模型评估维度**
- Spatial reasoning是评估模型认知能力的重要维度
- 可以区分不同模型的推理能力水平

---

## Level 3 Spatial Reasoning的改进建议

### 当前实现的问题

1. **与L4的边界模糊**：
   - L3的"最远"、"最近"也是相对概念
   - 与L4的"最大"、"最小"在概念上相似

2. **测试覆盖度**：
   - 当前主要测试方向性和距离关系
   - 可以增加更多空间关系类型

### 建议的改进方向

**1. 明确L3的定位**
- **L3专注于空间位置关系**（2D坐标、方向、距离）
- **L4专注于对象属性关系**（大小、相似性等）
- 区分标准：L3是"在哪里"，L4是"是什么"

**2. 增加空间关系类型**
- **相对位置**："Remove objects between A and B"
- **空间模式**："Remove objects forming a line"
- **空间集合**："Remove objects in the same region"

**3. 提高测试难度**
- 增加需要多步推理的空间关系
- 增加需要空间模式识别的任务
- 增加需要空间抽象的任务

---

## 总结

### Spatial Reasoning在Object Subtraction Task中的测试目标：

1. **基础空间概念理解**：方向、位置、距离
2. **相对位置推理**：对象之间的空间关系
3. **空间计算能力**：距离计算、象限判断、位置排序
4. **空间抽象能力**：从具体位置抽象出空间模式

### 与整体框架的关系：

- **Rotation Task**：测试3D空间推理（高级）
- **Object Subtraction L3**：测试2D空间推理（基础到中级）
- **两者互补**：共同评估模型的spatial reasoning能力

### 核心价值：

Spatial reasoning是评估视频生成模型认知能力的重要维度，Object Subtraction Task的Level 3专门设计来测试这一能力，与Rotation Task形成互补，全面评估模型的空间推理能力。

