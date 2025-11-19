# Level 4: 概念抽象测试方向分析

## 当前问题

### ❌ 与L1重合的任务类型
- `keep_same_color`: "Keep only the green objects" - 这本质上和L1的"Remove all red objects"一样
- `keep_same_shape`: "Keep only the cubes" - 这本质上和L1的"Remove all spheres"一样

**问题**：这些任务都是基于**明确的视觉属性**，不是真正的"概念抽象"。

---

## 真正的"概念抽象"应该测试什么？

### 核心特征
1. **相对性**：需要比较所有对象才能判断
2. **抽象性**：不是直接属性，而是衍生概念
3. **推理性**：需要计算、比较、推理才能得出答案

---

## 建议的概念抽象测试方向

### 1. ✅ 相对大小比较（已实现）
**测试类型**：
- `remove_largest`: "Remove the largest object"
- `remove_smallest`: "Remove the smallest object"

**为什么是概念抽象**：
- 需要比较所有对象的大小
- "最大"、"最小"是相对概念，不是绝对属性
- 需要推理找出极值

**扩展方向**：
- `remove_larger_objects`: "Remove all objects larger than average"
- `remove_smaller_objects`: "Remove all objects smaller than average"
- `remove_medium_objects`: "Remove all objects that are neither the largest nor the smallest"

---

### 2. ✅ 异常值检测（已实现）
**测试类型**：
- `remove_outlier`: "Remove the object that looks different from the others"

**为什么是概念抽象**：
- 需要识别"majority pattern"
- 需要找出不符合模式的异常对象
- 需要理解"相似性"、"差异性"

**扩展方向**：
- `remove_most_different`: "Remove the object that is most different from all others"
- `remove_least_similar`: "Remove the object that is least similar to the majority"

---

### 3. 🆕 相对大小分类（建议添加）
**测试类型**：
- `remove_large_objects`: "Remove all large objects" (相对于其他对象)
- `remove_small_objects`: "Remove all small objects" (相对于其他对象)

**实现方式**：
- 计算所有对象的平均大小
- 大于平均值的 = "large"
- 小于平均值的 = "small"
- 需要确保有明显的区分度（例如：large和small之间至少有20%的差距）

**为什么是概念抽象**：
- "大"、"小"是相对概念，不是绝对尺寸
- 需要计算平均值，然后分类
- 需要理解"相对于其他对象"的含义

---

### 4. 🆕 数量关系（建议添加）
**测试类型**：
- `remove_majority`: "Remove the objects that appear most frequently" (如果某种颜色/形状占多数)
- `remove_minority`: "Remove the objects that appear least frequently" (如果某种颜色/形状占少数)

**实现方式**：
- 统计每种颜色/形状的数量
- 找出数量最多/最少的组
- 移除该组的所有对象

**为什么是概念抽象**：
- 需要理解"多数"、"少数"的概念
- 需要统计和比较数量
- 不是直接属性，而是数量关系

**注意**：这个和`keep_same_color`的区别是：
- `keep_same_color`: "Keep only the green objects" (明确说了颜色)
- `remove_majority`: "Remove the objects that appear most frequently" (没有说颜色，需要自己找)

---

### 5. 🆕 相似性/差异性（建议扩展）
**测试类型**：
- `remove_most_similar_pair`: "Remove one object from the most similar pair"
- `remove_least_similar`: "Remove the object that is least similar to any other object"

**实现方式**：
- 计算所有对象之间的相似度（基于颜色+形状+大小）
- 找出最相似的一对，移除其中一个
- 或找出最不相似的对象

**为什么是概念抽象**：
- 需要理解"相似性"的概念
- 需要计算对象之间的相似度
- 需要比较和排序

---

### 6. 🆕 模式识别（建议添加）
**测试类型**：
- `remove_pattern_breaker`: "Remove objects that break the pattern"
- `remove_non_pattern`: "Remove objects that don't follow the pattern"

**实现方式**：
- 识别场景中的模式（例如：所有对象都是同一种颜色，除了一个）
- 找出不符合模式的对象
- 移除这些对象

**为什么是概念抽象**：
- 需要识别"模式"
- 需要理解"符合模式"、"不符合模式"
- 需要抽象思维

---

### 7. 🆕 极值组合（建议添加）
**测试类型**：
- `remove_extreme_combination`: "Remove objects with the most extreme combination of properties"
- `remove_median_objects`: "Remove objects that are neither the largest nor the smallest"

**实现方式**：
- 基于多个属性（大小、位置、颜色等）的组合
- 找出极值组合
- 或找出中位数对象

**为什么是概念抽象**：
- 需要理解"极值"、"中位数"等统计概念
- 需要多维度比较
- 需要抽象推理

---

### 8. 🆕 相对位置（可选，可能更适合L3）
**测试类型**：
- `remove_central_objects`: "Remove objects closest to the center" (相对于其他对象)
- `remove_peripheral_objects`: "Remove objects farthest from the center" (相对于其他对象)

**注意**：这个可能更适合L3，因为主要是空间关系。但如果强调"相对性"和"比较"，也可以算L4。

---

## 建议的优先级

### 高优先级（应该实现）
1. ✅ `remove_largest` / `remove_smallest` - 已实现
2. ✅ `remove_outlier` - 已实现
3. 🆕 `remove_large_objects` / `remove_small_objects` - 相对大小分类
4. 🆕 `remove_majority` / `remove_minority` - 数量关系

### 中优先级（可以考虑）
5. 🆕 `remove_most_similar_pair` / `remove_least_similar` - 相似性/差异性
6. 🆕 `remove_pattern_breaker` - 模式识别

### 低优先级（可选）
7. 🆕 `remove_extreme_combination` - 极值组合
8. 🆕 `remove_median_objects` - 中位数对象

---

## 需要移除的任务类型

### ❌ 应该移除或移到L1
- `keep_same_color`: 这是明确的颜色属性，不是概念抽象
- `keep_same_shape`: 这是明确的形状属性，不是概念抽象

**替代方案**：
- 如果场景中某种颜色/形状占多数，可以改为：
  - `remove_minority`: "Remove the objects that appear least frequently"
  - 这样需要模型自己找出"少数"是什么，而不是直接说"green"

---

## 实现建议

### 第一步：移除非抽象任务
- 移除`keep_same_color`和`keep_same_shape`的生成逻辑
- 或者将它们移到L1（因为它们本质上是L1任务）

### 第二步：添加新的抽象任务
1. **相对大小分类**：
   - `remove_large_objects`: 移除所有大于平均值的对象
   - `remove_small_objects`: 移除所有小于平均值的对象
   - 确保有明显的区分度

2. **数量关系**：
   - `remove_majority`: 移除数量最多的组（如果>=60%）
   - `remove_minority`: 移除数量最少的组（如果<=40%）

### 第三步：扩展异常值检测
- 改进`remove_outlier`，使其更明确
- 添加`remove_most_different`等变体

---

## 总结

**真正的概念抽象应该**：
- ✅ 需要比较和推理
- ✅ 基于相对概念，不是绝对属性
- ✅ 需要计算或统计
- ✅ 需要理解抽象概念（如"多数"、"极值"、"相似性"）

**不应该包括**：
- ❌ 直接的颜色/形状匹配（这是L1）
- ❌ 明确的空间位置（这是L3）
- ❌ 明确列出的对象（这是L2）

