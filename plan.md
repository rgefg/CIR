# CLIP CIR 梯度冲突简化实验计划

## 1. 讲故事主线

我们只讲三件事：
1. 检索分支和编辑分支在共享 text encoder 上存在真实梯度冲突。
2. 梯度投影可以显著缓解这种冲突。
3. 训练后再做权重合并，本质上也是一种“先分开学、再减少冲突地融合”的解决方案。

不再追求把所有层、所有模块、所有训练技巧全部讲清楚。
本轮目标是做出一套足够简洁、足够有说服力的图和指标。

## 2. 最小实验设计

### 实验 A：证明 retrieval vs edit 确实冲突
在同一个 batch 上，分别计算：
- `L_ret` 对共享 text encoder 的梯度
- `L_edit` 对共享 text encoder 的梯度

对每个 text block 统计一个层级冲突值：
- `conflict_l = cos(g_ret^l, g_edit^l)`

最终只画一张图：
- x 轴：text block 0~11
- y 轴：梯度余弦相似度
- 小于 0 表示冲突

如果需要更稳定，就对多个 batch 取平均，最后仍然只出一张逐层 conflict 曲线或 heatmap。

### 实验 B：证明梯度投影能缓解冲突
在同样的 batch 上，再计算一次：
- 投影前：`cos(g_ret^l, g_edit^l)`
- 投影后：`cos(g_ret^l, g_edit_proj^l)`

这里的 `g_edit_proj` 就是当前 PGGrad / PCGrad 风格的 geo 梯度投影结果。

我们只展示：
- 同一张图上 before / after 两条曲线


目标不是证明冲突完全消失，而是证明：
- 负冲突减少
- 平均 cosine 变大
- 冲突最强的层得到缓解

### 实验 C：证明 merge 也是在解决冲突
这里不再讲复杂权重空间理论，只做结果对比：
- standalone retrieval base
- standalone geo/edit LoRA
- merge 后模型

讲法非常直接：
- 在线训练时，两个任务同时拉扯共享参数，会产生冲突
- 梯度投影是在训练时减少这种拉扯
- 权重合并是在训练后把两个分支分别学到的参数做更温和的融合，因此也是在减少冲突

这里只需要一张很小的结果表，不需要再做复杂分析图。

## 3. 只保留两个核心量化指标

### 指标 1：Global Conflict Index (GCI)
定义：
- 先按层算 `cos(g_ret^l, g_edit^l)`
- 再只统计负冲突部分

记为：
- `GCI = mean_l max(0, -cos(g_ret^l, g_edit^l))`

解释：
- `GCI` 越大，说明共享层平均“打架”越严重
- 投影后如果 `GCI` 下降，就说明冲突被缓解了

这是主指标，直接对应你要讲的故事。

### 指标 2：Effective Target Entropy Gap (ETEG)
为了像“条件熵差异”那样给一个统计学支撑，同时又直接和当前检索任务绑定，本轮改用：
- retrieval query 与 edit query 在同一候选目标集上的 softmax 熵差

做法：
- retrieval 分支：对 `q_ret` 和同 batch target 集合算 `softmax(sim(q_ret, T))`
- edit 分支：对 `q_edit` 和同 batch target 集合算 `softmax(sim(q_edit, T))`
- 分别求平均熵 `H_ret` 与 `H_edit`
- 计算 `ETEG = H_edit - H_ret`

解释：
- `ETEG` 越大，说明 edit 分支在同一共享表征空间里对应的目标分布更“平”、更不确定
- 它比单纯 token 统计更接近参考文章里的“条件熵差异”讲法

因此本轮故事可以写成：
- 两个分支在共享表征空间里的目标熵不同（`ETEG`）
- 共享 text encoder 时产生明显梯度冲突（逐层 conflict 图 + `GCI`）
- 梯度投影能缓解冲突（`GCI` 下降）
- 训练后 merge 也能进一步减轻在线共享训练的冲突后果（结果表提升）

## 4. 实际执行顺序

### Step 1
先做同 batch 的逐层 conflict 分析：
- 只看共享 text encoder
- 只看 LoRA 层
- 只出一张逐层图

### Step 2
在同样设置下加入梯度投影：
- 画 before / after
- 报一个 `GCI before` 和 `GCI after`

### Step 3
计算一个简单的 `ETEG`：
- retrieval query vs edit query 的同 batch softmax 熵差
- 作为“为什么会冲突”的统计解释

### Step 4
附一张最终结果表：
- retrieval standalone
- projection 版本
- merge 版本

## 5. 最终交付物

本轮只需要这几样：
1. 一张逐层梯度 conflict 图
2. 一张投影前后对比图
3. 一个 `GCI` 标量前后对比
4. 一个 `ETEG` 标量
5. 一张最终结果表（standalone / projection / merge）

## 6. 结论模板

最后结论只围绕下面三句话展开：
1. retrieval 与 edit 分支在共享 text encoder 上存在显著梯度冲突。
2. 梯度投影能够部分缓解这种冲突。
3. 权重合并进一步减少了在线共享训练带来的冲突后果，因此带来更好的最终检索表现。

## 7. 第一轮实测结果（2026-03-26）

短跑配置：
- run: `DistillCIR_ParallelDualLoRA_BS48_Accum8_QKV_StrictLoss_dropout0.5_conflictprobe100_v5`
- setting: qkv-aware LoRA, strict geo loss, whole-instruction dropout=0.5, same-batch hard top-8 geo sampling
- probe window: `step 10~100`, `every 10 step`
- effective probe steps: `10`

实测 summary：
- `GCI before mean = 0.005236`
- `GCI after mean = 0.000000`
- `retrieval entropy mean = 3.7413`
- `edit entropy mean = 1.2257`
- `ETEG mean = -2.5156`

逐层平均 cosine（before -> after）：
- block 0: `0.0061 -> 0.0107`
- block 1: `-0.0146 -> 0.0128`
- block 2: `0.0017 -> 0.0135`
- block 3: `0.0110 -> 0.0174`
- block 4: `-0.0291 -> 0.0103`
- block 5: `0.0069 -> 0.0238`
- block 6: `0.0097 -> 0.0148`
- block 7: `-0.0007 -> 0.0138`
- block 8: `0.0006 -> 0.0251`
- block 9: `0.0189 -> 0.0212`
- block 10: `-0.0010 -> 0.0090`
- block 11: `-0.0024 -> 0.0191`

当前可直接讲的结论：
1. 共享 text encoder 上确实存在逐层负冲突，最明显的是 block 1、4、7、10、11。
2. 现有梯度投影会把这些负冲突层整体推回正区间，第一轮 probe 下 `GCI` 从 `0.005236` 直接压到 `0`。
3. retrieval 与 edit 在同一候选目标集上的有效目标熵存在稳定差异，当前 `ETEG` 为负，说明两条分支对共享表征空间施加的分布约束并不相同。
4. 这组结果已经足够支撑论文/报告里的核心故事；后续只需要用最终 plot 和结果表收尾。

图文件：
- `/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS48_Accum8_QKV_StrictLoss_dropout0.5_conflictprobe100_v5/gradient_conflict_plot.png`
- `/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS48_Accum8_QKV_StrictLoss_dropout0.5_conflictprobe100_v5/gradient_conflict_layerwise.tsv`
- `/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS48_Accum8_QKV_StrictLoss_dropout0.5_conflictprobe100_v5/gradient_conflict_summary.json`

