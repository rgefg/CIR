继续排查并修改代码，目标是：在保留 geo 分支的前提下，尽可能复现没有 geo 分支时的 retrieval 表现，并最终让 inference 时 geo + retrieval 融合后的 CIRR val R@1 达到 32.3。

已知日志：
1. 当前退化实验：
/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_PCGradSoftGeo_RNGIsolatedDet_v2/cirr_val_eval.log
2. retrieval baseline：
/data2/mingyu/composed_image_retrieval/logs/DistillCIR_Local_BS56_Accum8_Fast_0.3Logit_Cleaned36_retrievareverse/out.log

其中 29.5、30.3、32.3 全都指 CIRR val R@1。
“两个分支合并后的表现”指 inference 时 geo + retrieval 融合后的最终结果。

你被允许自己查看代码、修改代码、跑实验、看日志、提前停掉异常实验。
默认使用 GPU 6、7；必要时可检查其他 GPU 是否空闲，并额外最多再加 2 张卡。

必须始终围绕这个原则修改：
geo 和 retrieval 在架构上并行，但训练本质上独立，互不影响。
重点排查所有可能的干扰来源：参数共享、梯度串扰、loss 耦合、forward 污染、logit/temperature 联动、optimizer/scheduler/grad scaler/accumulation 耦合、RNG/dropout/batch/负样本/蒸馏目标带来的隐藏耦合、训练评估不一致等。

实验判定规则：
1. 训练约 10% 时 retrieval loss 仍 > 1，视为异常，直接停掉，继续排查。
2. retrieval eval 到 600 step 仍 < 29.5，说明修改不够有效，可以继续优化，但不算达标。
3. 只有 fusion 后 CIRR val R@1 达到 32.3 才算完成。

工作要求：
- 不要只分析，要实际修改和实验。
- 每轮先提出怀疑点，再动代码，再验证。
- 明显异常要尽早停实验。
- 每次有实质性修改都要保存代码存档并写明目的。
- 优先恢复 retrieval baseline，再追求 fusion 增益。

每轮都汇报：
- 本轮假设
- 修改点
- 实验命令
- 使用 GPU
- 关键日志观察
- 是否提前停止及原因
- 当前结论
- 下一轮计划

现在开始：先对比两个日志，定位 retrieval 退化的最可疑原因，然后立刻修改并跑实验。除非遇到真正无法继续的阻塞，否则不要停下来问我。

---

2026-03-23 简要记录：bestever03nofreeze 合并排查

当前已确认的最好 retrieval 基座：
- /data2/mingyu/composed_image_retrieval/logs/bestever03nofreeze/epoch_0_step_1400.pt

当前已确认的最好外部 text LoRA：
- /data2/mingyu/composed_image_retrieval/logs/bestever03nofreeze/bestqwenlora.pt

之前“1400 + bestqwen”能跑出约 31.5 的来源：
- 使用旧版 /data2/mingyu/composed_image_retrieval/data/merge_lora_ties.py
- 使用旧版 /data2/mingyu/composed_image_retrieval/src/eval_retrieval.py
- 旧 merge 只处理主副权重交集 LoRA prefix
- 旧 eval 只给 nn.Linear 挂 LoRA

已确认的 key 情况：
- base1400 的 text tower 只有 attn.out_proj / mlp.c_fc / mlp.c_proj，没有 q_proj_lora / k_proj_lora / v_proj_lora
- base1400 还带 visual tower 的 out_proj / c_fc / c_proj LoRA
- bestqwenlora 是纯 text LoRA，包含 attn.out_proj / q_proj_lora / k_proj_lora / v_proj_lora / mlp.c_fc / mlp.c_proj

因此旧版“1400 + bestqwen”实际生效的层：
- 能 merge 且能 eval 加载：text attn.out_proj / mlp.c_fc / mlp.c_proj
- 不能真正生效：bestqwen 独有的 q_proj_lora / k_proj_lora / v_proj_lora

这解释了之前最优 merged 结果的来源：
- 它本质上是“base1400 + bestqwen 的公共 text LoRA 部分”
- 不是完整利用了 bestqwen 的 attention q/k/v LoRA

当前已做的修正方向：
- merge_lora_ties.py 新增 include-b-only，允许把副权重独有的 q/k/v LoRA 带入输出
- eval_retrieval.py 改为使用 enable_lora_on_clip，使 q/k/v attention LoRA 在 eval 时真正实例化并加载

下一步计划：
- 只保留单后台进程
- 用修正后的 merge + eval，重跑 raw1400 + bestqwen 的 CIRR test
- 核心目标是验证：把 bestqwen 的 q/k/v attention LoRA 真正挂进去后，分数是否高于旧版约 31.5 的结果

如果本轮失败，回退参考：
- 旧版最优结果应理解为“仅公共 text LoRA 生效”的基线
- 不应再把旧结果当作“完整 bestqwen attention 已合入”的证据

2026-03-23 结果补充

已确认的 merged test 结果口径：
- raw1400 + geo200_raw: 约 31
- raw1400 + geo200_ema: 30.6
- ema1400 + geo200_raw: 29.3
- ema1400 + geo200_ema: 约 28

bestqwen 合并排查结论：
- 旧版 1400 + bestqwen 约 31.5 的结果，不应再理解为“完整 qkv attention 已参与”
- 旧 merge + 旧 eval 实际只稳定覆盖了公共 text LoRA：attn.out_proj / mlp.c_fc / mlp.c_proj
- bestqwen 独有的 q_proj_lora / k_proj_lora / v_proj_lora 在旧链路里没有真正生效

当前新增确认：
- bestever03nofreeze 基座 + bestqwen，在 qkv-enabled 评估下优于 no-qkv 评估，用户反馈 R@1 提升接近 1 点
- EMA1700_CurrentLoss 的 step1400 基座 + bestqwen + qkv，用户反馈 test R@1 = 31.32

下一轮训练计划：
- 训练端也切到 qkv-aware LoRA，不再只训练 out_proj + mlp
- 继续保留双分支 EMA
- geo loss 默认切到 strict 版本，显式开启 zero-loss regularizer
- watcher 默认关闭，避免再对训练造成额外扰动
