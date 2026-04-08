# Method Section Outline (v4)

> **论文标题占位**: Training-Aware Dual-LoRA Merging for Zero-Shot Composed Image Retrieval
>
> **核心 story**：ZS-CIR 需要两种能力——**检索对齐**和**编辑几何一致性**，它们对 text encoder LoRA 的梯度存在冲突。我们提出 **同步 dual-branch 训练 + training-aware structural merging**：训练时在浅层引入 Shared-B 约束、并用梯度投影稳定联合优化；合并时根据层级冲突模式采用 **浅层 Shared-B 精确合并 / 深层 TIES conflict-aware 合并**。最终推理仍是单分支、无额外前向开销。

---

## Section 结构（4 个 subsection）

| Section | 内容 | 篇幅 |
|---------|------|------|
| **3.1 Preliminaries** | 任务定义 + 隐晦引用带过数据/检索范式 + 公式 | ~0.4 col |
| **3.2 Gradient Conflict Analysis** | 发现梯度冲突 + 实验证据(图) + 引出 dual-branch 动机 | ~0.5 col |
| **3.3 Dual-Branch Training** | 检索分支(1段) + Geo 分支(主力) + hard mining | ~1 col |
| **3.4 Training-Aware LoRA Merging** | Shared-B 理论 + Hybrid Layerwise Merge | ~1 col |

---

## Section 3.1: Preliminaries

> **写作原则**：不出现任何具体方法名（不提 DistillCIR / Pic2Word / SEARLE）。全部用 "prior work [x]" / "existing methods [x,y]" / "following [x]" 的隐晦引用方式。

### 第 1 段：Task & Data Setup（≤5 句话）

> **[任务定义]** Given a reference image $I_r$ and a relative text instruction $t$, zero-shot composed image retrieval (ZS-CIR) aims to retrieve a target image $I_t$ from a gallery, without access to task-specific annotated triplets at training time.
> **[数据，隐晦引用]** Following recent work on LLM-based training data synthesis for ZS-CIR [cite-DistillCIR], we train on automatically generated quadruplets $(I_r, t_{\text{fwd}}, c_{\text{tgt}}, t_{\text{rev}})$ derived from web-crawled image-caption pairs, where $t_{\text{fwd}}$ is a forward editing instruction, $c_{\text{tgt}}$ is the correspondingly modified caption, and $t_{\text{rev}}$ describes the inverse transformation.
> **[你的改动]** We extend the standard training triplets with reverse instructions $t_{\text{rev}}$ to enable the geometric consistency objective introduced in Sec. 3.3.

### 第 2 段：Composed Query Paradigm（标为 standard）

> **[范式，隐晦引用]** We build on the pseudo-token paradigm for composed query construction [cite-Pic2Word, cite-SEARLE]. Given a reference image $I_r$, the CLIP visual encoder produces a visual representation
> $$f_v = E_V(I_r).$$
> A textual inversion network $\phi$ maps it to a pseudo-token feature
> $$f_\phi = \phi(f_v),$$
> which replaces a placeholder token in the standard Pic2Word-style prompt template $\mathcal{T}(t)$.
>
> **[统一记号]** Let $q$ denote the composed query embedding and $f_t$ denote the target-text embedding:
> $$q = E_T(\mathcal{T}(t), f_\phi), \qquad f_t = E_T(c_{\text{tgt}}). \tag{1}$$
>
> **[检索 loss，给公式]** The retrieval objective is a symmetric cross-entropy loss widely adopted in prior composed retrieval methods [cite-DistillCIR, cite-CompoDiff]:
> $$\mathcal{L}_{\text{ret}} = \frac{1}{2}\left[\ell_{\text{CE}}(\tau \cdot Q F_t^\top, \mathbf{y}) + \ell_{\text{CE}}(\tau \cdot F_t Q^\top, \mathbf{y})\right] \tag{2}$$
> where $Q, F_t$ are L2-normalized batched query and target-text embeddings, $\tau$ is the learned temperature, and $\mathbf{y}$ denotes the in-batch positive alignment.
>
> **[转折句，告诉审稿人贡献在下面]** While this objective provides a strong baseline for composed retrieval, we identify a fundamental limitation when jointly training for compositional understanding: the text-side LoRA parameters receive conflicting gradient signals from retrieval alignment and geometric supervision, which we analyze next.

### 对应代码
- `model/model.py`: `IM2TEXT`, `Phi`, `CLIP.encode_text_img_vis()`
- `src/main.py`: img2text 初始化, LoRA 应用

---

## Section 3.2: Gradient Conflict Analysis ★ 新增

> **定位**：这个 section 是 dual-branch 设计的**实验动机**。它不是贡献本身，而是用实验证据引出为什么需要 dual-branch + merge 的设计。
>
> **对应图**：`gc_plot_iclr.pdf`（你已有的梯度冲突层级图）

### 第 1 段：问题发现 + 实验设计

> **[问题]** When adding an auxiliary geometric consistency loss (detailed in Sec. 3.3) to regularize the text encoder alongside the retrieval objective, we observe that the two losses impose conflicting gradient directions on the shared LoRA parameters of the text encoder. To quantify this, we adopt the gradient conflict metric from multi-task learning [cite-Uni-X/PCGrad]:
>
> $$\text{GC}_\ell = s_{\text{base}}^\ell - s_{\text{cross}}^\ell \tag{2}$$
>
> where $s_{\text{cross}}^\ell = \cos(\bar{g}_{\text{ret}}^\ell, \bar{g}_{\text{geo}}^\ell)$ is the cosine similarity between the average retrieval and geometric gradients at layer $\ell$, and $s_{\text{base}}^\ell = \cos(\bar{g}_A^\ell, \bar{g}_B^\ell)$ is the cosine similarity between two random-split subsets of the same joint-task gradient (serving as a same-distribution baseline). A higher $\text{GC}_\ell$ indicates stronger inter-task gradient conflict at layer $\ell$.
>
> **[度量对象]** We measure the effective LoRA gradient $g_\Delta^\ell = \alpha/r \cdot (d\mathbf{B}^\ell \mathbf{A}^\ell + \mathbf{B}^\ell d\mathbf{A}^\ell)$ on the FFN down-projection (`c_proj`) of each text transformer block, accumulated over 60 training batches from CC3M.

### 第 2 段：实验发现（引用图）

> **[核心发现，对应 Fig. X]** As shown in Fig. X, gradient conflict is **pervasive and layer-dependent**:
>
> (1) **Conflict is universal**: $s_{\text{cross}}$ is close to zero or negative across all 12 layers, while $s_{\text{base}}$ consistently exceeds 0.98, yielding $\text{GC} > 0.7$ everywhere. On average, 62% of LoRA parameters experience conflicting gradients (dot product < 0) per training batch.
>
> (2) **Conflict intensifies in deeper layers**: The deepest block (block 11) exhibits $s_{\text{cross}} = -0.39$ (anti-correlated gradients) and $\text{GC} = 1.38$, the strongest conflict. Mid-to-deep layers (blocks 6–9) also show near-zero or negative cross-task similarity.
>
> (3) **Shallow layers are relatively aligned**: Block 4 achieves the highest $s_{\text{cross}} = 0.24$, indicating partial gradient agreement in earlier layers.

### 第 3 段：Implications → 引出 dual-branch + layerwise merge 的设计

> **[设计动机]** These findings motivate two key design decisions:
>
> **(i) Synchronous dual-branch training**: Rather than forcing a single text LoRA adapter to absorb both objectives, we maintain a retrieval branch $\theta_r$ and a geometric branch $\theta_g$ with separate optimizers and gradient streams, while training them **synchronously on the same mini-batch**. Residual conflicts are further mitigated by gradient projection during optimization.
>
> **(ii) Layer-aware structural merging**: The layer-dependent conflict pattern suggests that a uniform merge strategy is suboptimal. Shallow layers, where conflicts are weaker, are suitable for structurally aligned Shared-B merging; deep layers, where conflicts are severe, require conflict-aware TIES merging in delta space. We formalize this training-merge co-design in Sec. 3.4.

### 附表：数据备查（不放正文，放 Appendix 或用于画图）

| block | s_base | s_cross | GC_before | 解读 |
|-------|--------|---------|-----------|------|
| 0 | 0.989 | 0.122 | 0.867 | 冲突 |
| 1 | 0.989 | -0.005 | 0.994 | 强冲突 |
| 4 | 0.987 | 0.242 | 0.745 | 最弱冲突 |
| 7 | 0.981 | -0.037 | 1.018 | 强冲突 |
| 11 | 0.994 | -0.385 | 1.379 | 最强冲突 |

### 对应代码 & 图
- `src/probe_offline.py`: 完整的离线梯度冲突探针
- `gc_plot_iclr.pdf`: 梯度冲突层级图 ← **作为 Fig. X 放入正文**
- `plan.md`: 实验设计文档和完整数据

---

## Section 3.3: Dual-Branch Training

### 第 1 段：Retrieval Branch（≤5 句话，定位为 "Branch 1"）

> The retrieval branch follows the composed query paradigm in Sec. 3.1. Given $I_r$, we compute $f_v = E_V(I_r)$ and $f_\phi = \phi(f_v)$, then encode the dataset-aligned prompt $\mathcal{T}(t)$ with the retrieval text encoder $E_T^{\theta_r}$ to obtain the query embedding $q$. Target captions are encoded by the same text encoder to obtain $f_t$. The symmetric contrastive loss in Eq. (2) is computed with cross-GPU negative aggregation. Trainable parameters in this branch include LoRA adapters on the visual and text encoders, the textual inversion network $\phi$, and the learned temperature $\tau$.

### 第 2 段：Geometric Consistency Branch — Motivation

> While the retrieval loss drives query-target alignment, it does not explicitly constrain the *directionality* of instruction embeddings. We therefore instantiate a second text branch $E_T^{\theta_g}$, initialized from the same frozen CLIP text weights but equipped with its own LoRA adapters. Importantly, the two branches are trained **synchronously** at every step: the retrieval branch processes the image-text batch, while the geometric branch consumes the caption/instruction tuples selected from the same batch. The visual encoder $E_V$ and textual inversion network $\phi$ are optimized only through the retrieval branch and are reused unchanged after merging.

### 第 3 段：Geo Loss Formulation

> Let $\hat{f}_{\text{src}}, \hat{f}_{\text{tgt}}, \hat{f}_{\text{fwd}}, \hat{f}_{\text{rev}}$ denote L2-normalized text embeddings produced by $E_T^{\theta_g}$ from the source caption, target caption, forward instruction, and reverse instruction, respectively. We define the normalized caption displacement
> $$\hat{\Delta} = \frac{\hat{f}_{\text{tgt}} - \hat{f}_{\text{src}}}{\|\hat{f}_{\text{tgt}} - \hat{f}_{\text{src}}\|_2}. \tag{3}$$
> We then impose three complementary constraints:
>
> **Forward alignment** — the forward instruction should point along the caption change:
> $$\mathcal{L}_{\text{fwd}} = \mathbb{E}\left[1 - \langle \hat{f}_{\text{fwd}},\, \hat{\Delta} \rangle\right] \tag{4}$$
>
> **Reverse alignment** — the reverse instruction should point in the opposite direction:
> $$\mathcal{L}_{\text{rev}} = \mathbb{E}\left[1 - \langle \hat{f}_{\text{rev}},\, -\hat{\Delta} \rangle\right] \tag{5}$$
>
> **Zero-sum regularizer** — forward and reverse embeddings should cancel:
> $$\mathcal{L}_{\text{zero}} = \mathbb{E}\left[\|\hat{f}_{\text{fwd}} + \hat{f}_{\text{rev}}\|_2\right] \tag{6}$$
>
> The geometric loss is:
> $$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{fwd}} + \mathcal{L}_{\text{rev}} + \lambda_{\text{zero}}\,\mathcal{L}_{\text{zero}} \tag{7}$$

### 第 4 段：Retrieval-Guided Hard Sample Mining

> Not all training samples contribute equally to directional learning. We therefore perform **retrieval-guided hard mining**: at each step, we compute the per-sample retrieval loss from Branch 1 (Eq. 2), filter valid tuples that contain all required captions and instructions, and select the top-$k$ hardest samples as input to the geometric branch. This focuses the geometric regularization on cases where the current retrieval model is most confused, providing more informative gradient signals. We use $k=8$ per micro-batch in practice.

### 第 5 段：Training Protocol（简要）

> The total training objective is
> $$\mathcal{L} = \mathcal{L}_{\text{ret}} + \lambda_{\text{geo}}\,\mathcal{L}_{\text{geo}}. \tag{8}$$
> The two branches are optimized **synchronously** with separate optimizers and gradient scalers. To stabilize joint training, we apply a PCGrad-style projection that removes the component of the geometric gradient that directly conflicts with the retrieval gradient on matched text-LoRA parameters. When Shared-B is enabled in shallow layers, the shared $\mathbf{B}$ is updated only by the retrieval branch, while the geometric branch updates only its task-specific $\mathbf{A}$ factors.

### 对应代码
- `src/trainer.py`: `get_loss_geo_text_branch()` (L586-693), `select_geo_subset()` (L515-583)
- `src/main.py`: `TextEncoderBranch` (L301-327)

---

## Section 3.4: Training-Aware LoRA Merging

### 第 1 段：问题引出

> After training, we retain the **retrieval branch** as the inference backbone: its visual encoder $E_V$, textual inversion network $\phi$, learned temperature $\tau$, and base text weights are kept unchanged. Only the **text-LoRA adapters** from the retrieval and geometric branches, $(\mathbf{A}_r, \mathbf{B}_r)$ and $(\mathbf{A}_g, \mathbf{B}_g)$, are merged into a single adapter $(\mathbf{A}_m, \mathbf{B}_m)$. Naïve averaging performs poorly due to parameter-level sign conflicts, especially in deep layers (Sec. 3.2). Existing post-hoc merge methods [cite-TIES, cite-DARE] treat this as a pure parameter combination problem. We argue that the merging strategy should be **co-designed with training** to structurally reduce conflicts before they arise.

### 第 2 段：Shared-B Training Design（核心 novelty）

> **Key insight.** For LoRA with $\Delta W = \mathbf{B}\mathbf{A}$, the matrix $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$ determines the output-side column subspace of the update. If two branches share the same $\mathbf{B}$, their updates are guaranteed to lie in the same column subspace, confining inter-branch differences to the task-specific $\mathbf{A}$ matrices.
>
> **Design.** For shallow blocks $\ell < N$ (default $N{=}6$), we tie the geometric-branch $\mathbf{B}$ to the retrieval-branch $\mathbf{B}$ by parameter aliasing. The geometric optimizer updates only $\mathbf{A}_g$; the shared $\mathbf{B}$ is updated solely by the retrieval branch. For deep blocks $\ell \geq N$, the two branches keep fully independent $(\mathbf{A}, \mathbf{B})$ factors. This makes the shallow merge exact while leaving the conflict-heavy deep layers to a more conservative merge strategy.
>
> **Proposition.** *If $\mathbf{B}$ is shared and the branch-specific matrices $\mathbf{A}_r,\mathbf{A}_g$ have full row rank $r$, then $\operatorname{col}(\Delta W_r) = \operatorname{col}(\Delta W_g) = \operatorname{col}(\mathbf{B})$.* (Proof in Appendix.)
>
> This establishes exact left-subspace equivalence in shallow layers and justifies why merging there can be reduced to the $\mathbf{A}$ space without loss. A longer proof and the SVD interpretation can be placed in the appendix based on `docs/shared_b_svd_proof.md`.

### 第 3 段：Hybrid Layerwise Merging

> Guided by the Shared-B guarantee and the layer-dependent conflict pattern (Sec. 3.2, Fig. X), we propose a **hybrid layerwise merge**:
>
> **Shallow layers** ($\ell < N$): $\mathbf{B}$ is shared, so merging reduces to:
> $$\mathbf{A}_m = w_r \mathbf{A}_r + w_g \mathbf{A}_g, \quad \mathbf{B}_m = \mathbf{B} \tag{7}$$
> This is exact—no information loss.
>
> **Deep layers** ($\ell \geq N$): the branches are fully independent and conflict is stronger. We apply TIES [cite] in the full delta-weight space:
> 1. Compute $\Delta_r = \mathbf{B}_r\mathbf{A}_r$, $\Delta_g = \mathbf{B}_g\mathbf{A}_g$
> 2. TIES merge (trim, elect sign, disjoint merge with density $\rho$) → $\Delta_m$
> 3. SVD re-factorization: $\Delta_m \approx \mathbf{B}_m \mathbf{A}_m$
>
> The layer split at $N$ matches the Shared-B boundary, forming a **unified train-merge pipeline**: training structure directly dictates merging strategy. This is also consistent with the gradient-conflict probe: shallow layers exhibit relatively weaker cross-objective conflict and admit exact structural merging, whereas deeper layers require a more conservative post-hoc merge.

### 第 4 段：Inference

> After merging, the model consists of a **single retrieval-branch backbone**: one CLIP visual encoder $E_V$, one textual inversion network $\phi$, and one merged text-LoRA adapter on the text encoder. The dual-branch design is purely a training strategy; inference requires **no additional parameters or forward passes** beyond the standard composed retrieval pipeline.

### 对应代码
- `src/main.py`: `tie_shared_b_between_text_branches()` (L243-267)
- `data/merge_lora_ties.py`: hybrid_layerwise merge (L259-270)

---

## Notation Cheat Sheet

| Symbol | Meaning |
|--------|---------|
| $I_r$ | Reference image |
| $t_{\text{fwd}}, t_{\text{rev}}$ | Forward / reverse instruction |
| $c_{\text{src}}, c_{\text{tgt}}$ | Source / modified caption |
| $f_v$ | Visual feature from the CLIP visual encoder |
| $f_\phi$ | Pseudo-token feature produced by $\phi(f_v)$ |
| $f_t$ | Target-text embedding from the text encoder |
| $\phi$ | Textual inversion network |
| $E_V, E_T$ | CLIP visual / text encoder |
| $\theta_r, \theta_g$ | Retrieval / geometric branch LoRA |
| $\mathbf{A}, \mathbf{B}$ | LoRA matrices ($\mathbf{A}{\in}\mathbb{R}^{r \times d_{\text{in}}}$, $\mathbf{B}{\in}\mathbb{R}^{d_{\text{out}} \times r}$) |
| $\Delta W$ | LoRA delta = $\mathbf{B}\mathbf{A}$ |
| $\text{GC}_\ell$ | Gradient conflict at layer $\ell$ |
| $N$ | Shared-B cutoff layer (default 6/12) |
| $\rho$ | TIES density |
| $w_r, w_g$ | Merge weights |

---

## Figure 规划

| Figure | 内容 | 来源 |
|--------|------|------|
| **Fig. 2 (Method Overview)** | Dual-branch 训练 → Shared-B 标注 → Merge → 单一模型推理。输入标 "LLM-generated quadruplets [cite]"，不画数据构建流程 | 需要画 |
| **Fig. 3 (Gradient Conflict)** | 12 层 GC 曲线 + 浅/深层分界线 N=6 标注 | `gc_plot_iclr.pdf` |
| **Fig. 4 (Geo Loss)** | Embedding 空间中 caption delta vs instruction 方向对齐示意 | 需要画 |

---

## 防御性写作备忘

| 审稿人可能问 | 在哪回应 |
|-------------|---------|
| "数据怎么来的" | 3.1 "Following recent work on LLM-based synthesis [cite]" + Impl. Details 补 1-2 句 |
| "检索 loss 和某方法区别" | 3.1 "widely adopted in prior composed retrieval methods [cite, cite]" — 标记为领域通用做法 |
| "梯度冲突度量为什么选这个" | 3.2 引用 Uni-X / PCGrad 的度量范式 |
| "为什么 N=6" | 3.2 Fig.3 显示浅层冲突相对较弱 + 3.4 Shared-B 理论在浅层最有效 + ablation 表 |
| "reverse instruction 怎么生成" | Impl. Details + Appendix prompt template |
| "GC 指标和实际性能的关系" | Ablation: 无 geo branch vs 有 geo branch + 各种 merge 策略的性能对比 |
| "merge 的具体范围是什么" | 3.4 第 1 段明确说明仅合并 text LoRA，视觉编码器和 $\phi$ 继承自 retrieval branch |
