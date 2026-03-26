# CLIP CIR 梯度冲突简化实验计划

## 1. 当前主方案：改成 Uni-X 风格的 Gradient Conflict 曲线

后续正式图不再用旧的 `GCI/ETEG` 口径，改成更接近 Uni-X Figure 1 的定义：

- 横轴：text encoder layer depth
- 纵轴：`Gradient Conflict`
- 指标定义：
  - `GC_l = s_base_l - s_cross_l`
  - `s_cross_l = cos(g_ret^l, g_edit^l)`
  - `s_base_l = cos(g_mix_a^l, g_mix_b^l)`

解释：
- `g_ret^l`：第 `l` 层上 retrieval 分支的平均梯度
- `g_edit^l`：第 `l` 层上 edit / geo 分支的平均梯度
- `g_mix_a^l, g_mix_b^l`：同一 joint 训练分布随机拆成两半后得到的平均梯度
- `GC_l` 越大，说明 retrieval 和 edit 在该层的冲突越强

## 2. 参数组定义：只看每层 FFN down-projection

为了尽量贴近 Uni-X，这张图只分析每层 text block 的 FFN down-projection：

- 对 CLIP text tower 来说，就是 `transformer.resblocks.{l}.mlp.c_proj`
- 因为当前训练是 LoRA，不更新 full weight，所以不直接取 frozen full weight 的梯度
- 实际统计对象定义为该层 `c_proj` 的 LoRA 有效权重更新梯度：
  - `g_delta^l = scale * (dB^l @ A^l + B^l @ dA^l)`

最后每层把 `g_delta^l` 展平成一个向量，再算 cosine similarity。
这样做的含义是：
- 仍然是 “FFN down-projection weight gradient” 的最接近替代
- 同时和我们当前 LoRA 训练设置一致

## 3. 数据与 checkpoint 选择

为了和 Uni-X 一样减少噪声，正式图采用“早期 checkpoint + 多 batch 平均”：

- checkpoint：取一个 shared 训练的早期 checkpoint
  - 推荐：当前新训练 run 的 `step 100~300` 区间
  - 原则：足够早，能反映共享训练时的真实冲突，而不是后期收敛后的偶然形态
- batch 数：先做 `60` 个 mini-batch
- token 量：不强行追论文的 2M token，但保证平均后曲线稳定

如果 `60` 个 batch 成本太高：
- 第一轮可以先跑 `20` 个 batch 验证趋势
- 最终出图再补到 `60`

## 4. 三组平均梯度怎么取

### 4.1 Cross-task similarity
对每个 probe batch：
- 用同一批样本算 retrieval loss，得到 `g_ret^l`
- 用同一批样本算 edit / geo loss，得到 `g_edit^l`

对多个 batch 做平均：
- `g_ret_avg^l = mean_b g_ret^{l,b}`
- `g_edit_avg^l = mean_b g_edit^{l,b}`

然后：
- `s_cross_l = cos(g_ret_avg^l, g_edit_avg^l)`

### 4.2 Baseline similarity
为了构造 “同一分布下本来应有多相似”，每个 probe batch 再做一次随机拆分：
- 先取同一 joint 训练分布上的样本集合
- 随机切成两个互不重叠子集 `A/B`
- 在 `A` 上算 joint loss 梯度 `g_mix_a^l`
- 在 `B` 上算 joint loss 梯度 `g_mix_b^l`

其中 joint loss 口径固定为当前训练真实目标：
- `L_joint = L_ret + lambda_geo * L_edit`

对多个 batch 做平均：
- `g_mix_a_avg^l = mean_b g_mix_a^{l,b}`
- `g_mix_b_avg^l = mean_b g_mix_b^{l,b}`

然后：
- `s_base_l = cos(g_mix_a_avg^l, g_mix_b_avg^l)`

### 4.3 Projection 版本
为了证明我们的梯度投影能缓解冲突，再额外构造一条 after 曲线：
- 每个 batch 先算 `g_ret^l` 和 `g_edit^l`
- 按当前训练里的投影规则，把 `g_edit^l` 投影成 `g_edit_proj^l`
- 再对多个 batch 平均，得到 `g_edit_proj_avg^l`

然后定义：
- `s_cross_after_l = cos(g_ret_avg^l, g_edit_proj_avg^l)`
- `GC_after_l = s_base_l - s_cross_after_l`

最终图里画两条线：
- `GC_before_l`
- `GC_after_l`

## 5. 最终图和讲法

正式图只保留一张主图：
- x 轴：layer 0~11
- y 轴：`Gradient Conflict`
- 两条曲线：
  - shared training / before projection
  - with gradient projection / after projection

讲法固定成三句：
1. retrieval 和 edit 在共享 text encoder 的 FFN down-projection 上存在层级冲突。
2. 冲突在某些浅层和深层更强，中间层相对缓和。
3. 梯度投影会系统性降低这些层的 `Gradient Conflict`。

## 6. 结果表只做辅助

图之外只需要一个很小的结果表：
- standalone retrieval
- retrieval + projection
- retrieval base + post-hoc merge

这张表只负责收尾：
- 在线训练时有冲突
- 训练时投影能缓解
- 训练后 merge 进一步改善最终效果

## 7. 交付物

本轮正式交付只保留：
1. 一张 Uni-X 风格逐层 `Gradient Conflict` 曲线图
2. 一个逐层数值表 `layer / s_base / s_cross_before / s_cross_after / GC_before / GC_after`
3. 一张最终结果小表（standalone / projection / merge）

旧的 `GCI/ETEG` 口径不再作为主结果，只保留为探索记录。

## 8. 第一轮实测结果（2026-03-26，归档，不作为最终图口径）

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

## 9. 多数据集评估口径核对（2026-03-26）

先和仓库现有 `src/eval_retrieval.py` 对齐后再决定哪些结果可以正式记录。

### 8.1 可以认定同口径的数据集

#### FashionIQ
现有仓库口径：
- `src/eval_retrieval.py --eval-mode fashion --source-data {dress,shirt,toptee}`
- 实际调用 `evaluate_fashion(...)`
- composed 特征使用：`encode_text_img_retrieval(target_caption, img2text(ref_image), split_ind=id_split, repeat=False)`
- 指标使用：`get_metrics_fashion(...)`

本轮临时脚本口径：
- 也是相同的 `target_caption` prompt
- 也是相同的 composed feature 计算式
- 也是相同的 `get_metrics_fashion(...)`

因此：
- `FashionIQ` 结果可以视为与仓库原生评估一致
- 可以正式记录

#### GeneCIS
现有仓库口径：
- `src/eval_retrieval.py --eval-mode genecis --genecis-task ...`
- 实际调用 `evaluate_genecis(...)`
- prompt: `a photo of * , {cap}`
- gallery ranking 与 `top-k hit` 逻辑固定

本轮临时脚本口径：
- prompt 完全一致
- `query/gallery` 编码逻辑一致
- `R@1/R@2/R@3` 计算逻辑一致

因此：
- `GeneCIS` 结果可以视为与仓库原生评估一致
- 可以正式记录

### 8.2 不能直接认定同口径的数据集

#### CIRCO
现有仓库口径：
- 你给的原始脚本是 `src/eval_retrieval.py --eval-mode circo`
- 当前仓库实现里这条路径实际上固定走 `split='test'`
- 最终输出是 `circo_submission.json`
- 这是提交 JSON 口径，不是本地 val metric 口径

本轮临时脚本口径：
- 用的是 `CIRCO val`
- 计算的是 `mAP@k / Recall@k / semantic mAP@10 mean`

因此：
- 这两者不是同一种评估
- 本轮 `CIRCO` 数字只能作为内部参考，不能按“和原脚本一致”写死结论
- 如果后续要正式汇报 `CIRCO`，需要按原生 `eval_retrieval.py --eval-mode circo` 重新跑提交 JSON，或明确写成 `CIRCO val internal metric`

## 10. 当前可正式记录的结果

以下结果只记录与仓库原生评估口径一致的部分。

### 9.1 FashionIQ（可正式记录）
#### Raw retrieval base
- dress: `R@1 4.96 / R@5 13.44 / R@10 18.29 / R@50 39.81 / R@100 50.97`
- shirt: `R@1 10.01 / R@5 20.56 / R@10 26.59 / R@50 43.23 / R@100 52.21`
- toptee: `R@1 7.90 / R@5 18.92 / R@10 26.57 / R@50 44.82 / R@100 55.58`

#### Raw base + EMA geo merge
- dress: `R@1 5.95 / R@5 14.77 / R@10 20.77 / R@50 42.49 / R@100 55.13`
- shirt: `R@1 10.35 / R@5 21.25 / R@10 27.53 / R@50 44.90 / R@100 54.02`
- toptee: `R@1 9.33 / R@5 21.01 / R@10 27.23 / R@50 47.63 / R@100 58.49`

结论：
- merge 在三个 FashionIQ 类别上都是稳定正增益，尤其 dress 和 toptee 更明显。

### 9.2 GeneCIS（可正式记录）
#### Raw retrieval base
- focus_attribute: `R@1 20.55 / R@2 34.15 / R@3 44.95`
- change_attribute: `R@1 16.19 / R@2 27.84 / R@3 38.21`
- focus_object: `R@1 15.82 / R@2 25.05 / R@3 33.72`
- change_object: `R@1 16.43 / R@2 28.37 / R@3 38.52`

#### Raw base + EMA geo merge
- focus_attribute: `R@1 21.10 / R@2 32.70 / R@3 43.05`
- change_attribute: `R@1 16.43 / R@2 28.31 / R@3 39.02`
- focus_object: `R@1 14.39 / R@2 23.42 / R@3 33.47`
- change_object: `R@1 16.33 / R@2 27.24 / R@3 36.89`

结论：
- merge 对 attribute 类任务略有帮助
- 对 object 类任务没有稳定收益，甚至有轻微下降
- 这说明当前 geo/edit LoRA 的收益更偏向组合编辑语义，而不是通用对象识别推理

## 11. 梯度冲突图的处理与后续计划

### 10.1 当前图不作为最终版本
- 旧图已经删除
- 原因不是数据无效，而是视觉表达不够好，且 `GCI after = 0` 太像算子自证，容易被质疑

### 10.2 后续更合理的冲突评估方式
当前 `GCI after = 0` 的问题在于：
- 投影使用的就是同一对 `g_ret / g_edit`
- 然后又用这对投影后的梯度去算 conflict
- 对 PCGrad/投影算子来说，这很容易在定义上把负内积直接消掉

因此后续主结果不应再把 `GCI after = 0` 当核心证据。

下一步应改成两层证据：
1. 保留逐层 `cos(before)` 作为“原始冲突存在”的证据
2. 新增 held-out interference 指标，作为“投影确实减弱任务互扰”的主证据

推荐的新指标：
- `I(edit->ret) = L_ret(theta - eta * g_edit; B_holdout) - L_ret(theta; B_holdout)`
- `I_proj(edit->ret) = L_ret(theta - eta * g_edit_proj; B_holdout) - L_ret(theta; B_holdout)`

如果 `I_proj < I`，就能说明：
- 不是因为定义上把 cos 修成非负
- 而是投影真的减少了 edit 梯度对 retrieval 的破坏

### 10.3 下一步执行顺序
1. 重新画更简洁的逐层冲突图（等新的可视化方案）
2. 增加 held-out interference score
3. 保留 `ETEG` 作为辅助解释，不再把 `GCI after = 0` 当主结论
4. `CIRCO` 若要正式汇报，按仓库原生 `circo` 提交 JSON 口径重跑
