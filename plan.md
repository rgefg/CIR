# CLIP CIR 梯度冲突分析计划

## 1. 目标

借鉴 Uni-X/梯度冲突分析文章的思路，系统分析当前 CLIP-based CIR 训练中：
- 检索分支（retrieval branch）
- 推理/编辑分支（当前 geo / reverse / text-only edit branch）

在共享参数上的梯度冲突模式，重点回答：
1. 冲突主要集中在哪些层？
2. 冲突是否呈现“浅层强、深层强、中间层弱”的结构性规律？
3. 当前的双分支隔离、PCGrad、EMA、qkv-aware LoRA 等设计，是否真正缓解了冲突？
4. 如果冲突确实集中在特定层段，后续是否值得做“部分共享 / 部分分叉”的结构改造？

## 2. 与参考文章的对应关系

参考文章研究的是：
- 共享 Transformer 同时处理文本序列与 VQ 视觉 token 序列时的梯度冲突。

我们当前任务的对应关系：
- 共享模块：CLIP text encoder（以及可能受影响的 retrieval-side trainable modules）
- 任务 A：CIR retrieval loss（image + instruction -> modified_caption 的检索目标）
- 任务 B：text-only 推理/编辑 loss（当前 geo / reverse branch）

需要说明：
- 文章中的“视觉 token 条件熵更高”这一解释，不能 1:1 套到 CLIP，因为我们这里视觉端不是 VQ 离散 token，而是连续视觉编码。
- 因此我们会把“信息熵差异”改写成更适合本项目的两个候选解释：
  1. retrieval instruction 与 reverse/edit instruction 的 token 统计差异；
  2. image-conditioned retrieval 目标与 text-only reverse/edit 目标的监督形式差异。

## 3. 核心假设

### H1
共享 text encoder 中，retrieval 与 edit/geo 的梯度冲突并不是均匀分布的，而是具有明显的层级结构。

### H2
冲突最可能集中在：
- text encoder 的浅层 block
- text encoder 的末端 block
- attention out_proj / qkv / mlp.c_fc / mlp.c_proj 这类关键子模块

### H3
当前的隔离设计虽然已经避免了显式串梯度，但共享 text encoder 训练目标之间仍然存在显著冲突；该冲突可能解释了：
- merge 后收益不稳定
- standalone retrieval 与 merge 最优点不一致
- qkv/no-qkv merge 行为差异明显

### H4
如果 conflict heatmap 确实呈现“浅层/深层更强，中间层更弱”，则后续有理由尝试 Uni-X 类似思路：
- 中层共享
- 浅层或深层部分解耦
- 或只对特定层段做 task-specific adapter

## 4. 要测的梯度对象

### 4.1 主分析对象
共享 text tower 的 trainable LoRA 参数：
- attn.q_proj_lora
- attn.k_proj_lora
- attn.v_proj_lora
- attn.out_proj
- mlp.c_fc
- mlp.c_proj

按 block 统计（resblock 0~11）。

### 4.2 辅助对象
- img2text 各层参数梯度
- visual tower trainable LoRA 梯度
- logit_scale 梯度

说明：
- geo/edit 分支本身不回传 visual/img2text，因此这些模块更多是“对照项”。
- 如果 visual/img2text 也出现冲突信号，说明实现上仍有隐藏耦合，需要重点排查。

## 5. 梯度冲突指标设计

### 5.1 原始层级冲突
对每个模块/层 l，定义：
- g_ret^l：retrieval loss 对该层 trainable 参数的梯度
- g_geo^l：geo/edit loss 对该层 trainable 参数的梯度
- s_l = cos(g_ret^l, g_geo^l)

解释：
- s_l < 0：发生负迁移/梯度冲突
- s_l 越负，冲突越强

### 5.2 去基线的层级冲突
参考文章思路，为避免把“同一任务内部自身梯度噪声”误判成跨任务冲突，定义：
- b_ret^l = cos(g_ret,1^l, g_ret,2^l)
- b_geo^l = cos(g_geo,1^l, g_geo,2^l)
- baseline_l = 0.5 * (b_ret^l + b_geo^l)
- conflict_l = s_l - baseline_l

其中 ret/geo 的 1,2 可通过同一 batch 划分两个子 batch 获得。

解释：
- s_l 反映跨任务梯度方向相似度
- baseline_l 反映各任务在同分布内部本身有多稳定
- conflict_l 更接近“额外冲突项”

### 5.3 额外统计量
每层再记录：
- neg_conflict_rate：多个 batch 上 s_l < 0 的比例
- grad_norm_ratio = ||g_geo^l|| / ||g_ret^l||
- projected_overlap = <g_ret^l, g_geo^l> / (||g_ret^l||·||g_geo^l||)

## 6. 实验分阶段设计

### 阶段 A：离线 checkpoint 探针（优先做）
目标：先不改训练流程，直接基于现有 checkpoint 和固定 batch 做 conflict 分析。

候选 checkpoint：
- bestever03nofreeze/epoch_0_step_1400.pt（强 retrieval 基座）
- DistillCIR_ParallelDualLoRA_BS56_Accum8_EMA1700_QKV_StrictLoss/checkpoints/epoch_1.pt
- 同目录 geo raw / geo ema LoRA

比较对象：
1. retrieval vs geo_raw
2. retrieval vs geo_ema
3. raw base vs ema base（只看 retrieval 内部稳定性）
4. qkv-aware 与 no-qkv 的 merge 最优点对应 checkpoint

产出：
- 每层 cosine heatmap
- 每层 baseline-adjusted conflict heatmap
- shallow/mid/deep 分组统计表

### 阶段 B：在线同 batch 梯度探针
目标：验证离线结论不是 checkpoint 偶然现象，而是训练过程中稳定存在。

做法：
- 在 trainer 中加入一个只在 debug/analyze 模式开启的 probe
- 每隔固定 step（例如 50/100 step）
- 对同一个 batch：
  1. 单独 backward retrieval
  2. 单独 backward geo/edit
  3. 单独对子 batch 求 ret-ret / geo-geo baseline
- 只记录梯度，不 optimizer.step

建议采样 step：
- early: 50 / 100 / 200
- middle: 600 / 1000
- late: 1400 / 1600

产出：
- 冲突随训练进度的演化曲线
- 关键层的 time-series plot

### 阶段 C：子模块级别拆解
目标：定位 attention 与 MLP 哪个冲突更主导。

对每个 block，拆成：
- q_proj_lora
- k_proj_lora
- v_proj_lora
- out_proj
- mlp.c_fc
- mlp.c_proj

产出：
- block × submodule 热图
- 哪一类参数最容易冲突的排序

### 阶段 D：信息论/统计解释分析
由于当前不是 VQ 视觉 token，因此不做“视觉 token vs 语言 token”原样复刻，而改做：

1. token 级统计：
- retrieval instruction 的 unigram / bigram / trigram 条件熵
- reverse/edit instruction 的对应条件熵
- modified_caption 的对应条件熵

2. 句长与稀有词覆盖：
- 平均长度
- unique token ratio
- rare token ratio
- numeric / color / action / relation token 占比

3. 若需要，再做 BPE 级 entropy 而不是 whitespace token 级 entropy。

目标：
- 判断 edit/reverse 指令是否比 retrieval instruction 更高熵、更稀疏或更长尾
- 为“为什么 text-only edit 分支和 retrieval 分支冲突”提供统计解释

## 7. 最小实现方案（批准后再改代码）

批准后，建议新增一个独立分析脚本，而不是先改 trainer 主逻辑：
- 新脚本候选：`data/analyze_gradient_conflict.py`

功能：
1. 加载指定 retrieval checkpoint + 可选 geo lora
2. 构造固定 batch（或从 dataloader 取前 N 个 batch）
3. 分别计算 retrieval / geo / ret-ret / geo-geo 梯度
4. 输出：
   - jsonl/tsv 数值表
   - layer heatmap png
   - summary markdown

必要参数：
- `--resume-retrieval`
- `--resume-geo-lora`
- `--batch-source train|val`
- `--num-batches`
- `--analyze-qkv`
- `--output-dir`

如果离线脚本结论明确，再决定是否加 trainer 在线 probe。

## 8. 结果判读标准

如果出现以下模式，则支持“结构性梯度冲突”假设：
- text encoder 浅层与深层的 conflict_l 明显低于中层
- qkv/out_proj/MLP 中有某一类子模块稳定为负
- 该模式在多个 batch、多个 checkpoint 上稳定存在

如果出现以下模式，则说明当前问题不主要来自共享 text encoder 冲突：
- 各层 conflict 近似随机，无稳定层级结构
- baseline-adjusted conflict 接近 0
- 只有极少数 batch 偶发负 cosine

## 9. 成功产出物

本轮分析完成后，至少应交付：
1. 一份 layer-wise conflict 表（jsonl 或 tsv）
2. 一张 text block 热图
3. 一张 submodule 热图
4. 一份 entropy / token-stat 表
5. 一段结论：
   - 冲突最强的层段
   - 是否支持 shallow/deep stronger 的现象
   - 后续结构修改建议

## 10. 后续结构改造方向（仅在分析后决定）

如果结论支持文章思路，下一步候选改法：
- 仅共享中间层 LoRA，浅层/深层拆分 retrieval 与 geo
- 保留共享主干，但对浅层或深层增加 branch-specific adapter
- 只在冲突最强的模块（例如 qkv 或 mlp.c_fc）上分叉，而不是整层分叉
- 用 conflict map 指导 PCGrad/weighting，而不是全层统一投影

## 11. 本次计划的边界

本文件只定义实验计划。
在你确认之前，不修改训练/评估主逻辑，不新增分析脚本，不改 merge 策略。


## 12. 当前实现上下文（2026-03-25）

为避免后续分析口径混乱，当前代码状态先固定记录如下：
- text tower LoRA 已切到 qkv-aware：训练与评估都支持 `q_proj_lora / k_proj_lora / v_proj_lora / attn.out_proj / mlp.c_fc / mlp.c_proj`
- geo 分支当前默认是 same-batch `hard top-8` 采样，而不是全 batch geo
- instruction dropout 当前是“整条 instruction dropout”，默认概率可在脚本中配置
- standalone CIRR val 当前训练内默认优先使用 EMA 权重记录
- cross merge eval 的历史结果表明：`raw base + ema geo` 是当前这一支里最优的 merge 组合之一

这意味着本轮梯度冲突分析不能只看“有没有冲突”，还必须区分：
1. 冲突来自 retrieval vs geo 任务目标本身
2. 冲突来自 qkv-aware LoRA 新增自由度
3. 冲突是否被 hard top-8 采样进一步放大（因为 geo 只盯 hardest samples）
4. instruction whole-dropout 是否改变了 retrieval 分支的梯度稳定性

## 13. 第一轮实际执行顺序（训练结束后直接做）

### Step 1：离线 same-batch 梯度探针
先不改 trainer，只做离线分析脚本。

输入 checkpoint：
- 当前 run 的 `epoch_1.pt`
- 当前 run 的 `epoch_1_ema.pt`
- 当前 run 的 `epoch_1_geo_lora.pt`
- 当前 run 的 `epoch_1_geo_lora_ema.pt`
- 对照 retrieval 基座：`bestever03nofreeze/epoch_0_step_1400.pt`
- 对照外部 text LoRA：`bestever03nofreeze/bestqwenlora.pt`

固定分析 batch：
- 从训练集 dataloader 取前 `N=16` 个 batch
- batch 顺序固定、seed 固定
- 每个 batch 内同时构造：
  - retrieval full-batch gradient
  - geo full-batch gradient
  - retrieval half1 vs half2 baseline
  - geo half1 vs half2 baseline

第一轮只分析 text tower，不碰 visual/img2text 的 merge。

### Step 2：层级热图
按以下三个粒度各出一套图：
- block-level：resblock 0~11
- submodule-level：q/k/v/out_proj/c_fc/c_proj
- stage-level：shallow(0~3) / middle(4~7) / deep(8~11)

第一轮重点看：
- 哪些 block 持续负 cosine
- shallow/deep 是否显著更差
- qkv 是否比 mlp 更冲突

### Step 3：加入对照项
为了避免把“训练不稳定”误判成任务冲突，加入两个对照：
- retrieval vs retrieval（不同 half-batch）
- geo vs geo（不同 half-batch）

如果 `ret-vs-geo` 的 cosine 只是和 `ret-vs-ret` 一样差，说明主要是梯度噪声；
如果 `ret-vs-geo` 系统性更差，才说明是任务冲突。

### Step 4：采样策略消融
如果第一轮确认冲突存在，再单独比较 geo 采样策略：
- `geo_sampling_mode=all`
- `geo_sampling_mode=random`
- `geo_sampling_mode=hard, topk=8`

只做短程 probe，不要求重新跑完整 1700 step。
目标是回答：
- hard top-8 是否确实提高了 geo/retrieval 的冲突强度
- 如果提高了，它是否同时带来了更可解释的 merge 增益

### Step 5：instruction dropout 对冲突的影响
如果 hard top-8 的冲突结构已经清晰，再比较：
- `instruction_dropout_prob=0.0`
- `instruction_dropout_prob=0.5`

只关注 retrieval 分支自身稳定性和 ret-vs-geo conflict 是否变化。
这一步不需要大训练，短程 probe 即可。

## 14. 交付形式（第一轮）

第一轮分析结束后，至少输出：
- `gradient_conflict_layerwise.tsv`
- `gradient_conflict_submodule.tsv`
- `gradient_conflict_summary.md`
- `ret_vs_geo_heatmap.png`
- `ret_vs_geo_adjusted_heatmap.png`
- `token_entropy_stats.tsv`

并在总结里明确回答：
1. 冲突最强的层段是不是 shallow/deep
2. 冲突主导模块是不是 qkv
3. hard top-8 是在缓解问题，还是在放大问题
4. instruction whole-dropout 对冲突是减轻还是加重
5. 下一步应该优先做结构分叉、loss weighting，还是采样策略修正

## 15. 暂定不做的事

为了让第一轮结论干净，本轮先不做：
- 不直接在训练主循环里插大量 backward probe
- 不先上部分共享/部分分叉结构改造
- 不先改 merge 逻辑去“碰运气提分”
- 不把 visual tower 也纳入第一轮核心分析对象

原则是：
先证明冲突的存在形式，再决定结构修改，而不是反过来盲改结构。
