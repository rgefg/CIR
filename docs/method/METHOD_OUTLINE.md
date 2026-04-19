# Method Outline (Current Design Only)

This outline is intentionally restricted to the **current** method design. Older variants are omitted.

## Core Story

We train a dual-branch CIR model:

- a **retrieval branch** that learns image-grounded composed retrieval
- a **geo branch** that regularizes edit direction in text space

The key changes relative to the older implementation are:

1. the geo branch is **not pure-text anymore**
   - its source anchor is a **blend** of text caption and image-conditioned pseudo-token anchor
2. the geo loss uses only
   - `fwd`
   - `rev`
   - `zero`
   - no reverse-consistency term
3. text LoRA uses **all-layer Shared-B**
4. merging is described with
   - **Shared-B-preserving merge**
   - **SVD denoising on merged A**
5. gradient-conflict analysis is used as motivation and stabilization evidence


## 1. Training Data

Each training example contains:

- `ref_img`
- `src_caption`
- `instruction`
- `modified_caption`
- `reverse_instruction`

The supervision comes from:
- reference image
- source caption
- forward edit instruction
- modified target caption
- reverse instruction

The data example should be shown as a JSON tuple with those fields only.


## 2. Retrieval Branch

Notation:

- `f_v = E_V(I_r)`
- `f_phi = phi(f_v)`
- `q = E_T(T(t), f_phi)`
- `f_t = E_T(c_tgt)`

where:

- `E_V` is the CLIP visual encoder
- `phi` is the SEARLE-style textual inversion network
- `E_T` is the CLIP text encoder
- `T(t)` is the placeholder prompt

The retrieval branch builds an image-grounded composed query by inserting `f_phi` into the placeholder token position of the text prompt.

Retrieval objective:

\[
\mathcal{L}_{ret}
=
\frac{1}{2}
\left[
\ell_{CE}(\tau QF_t^\top, y)
+
\ell_{CE}(\tau F_tQ^\top, y)
\right]
\]


## 3. Geo Branch

### 3.1 Geo Source Anchor

The geo branch does **not** use a pure text source anchor anymore.

We define:

- `z_src_text = E_T(c_src)`
- `z_src_image = E_T("a photo of *", f_phi)`

Then the geo source anchor is:

\[
z_{src}
=
\mathrm{norm}\big((1-w)z_{src,text} + wz_{src,image}\big)
\]

Current setting:

- `geo_src_anchor_mode = blend`
- `geo_src_image_weight = 0.25`
- `geo_src_anchor_detach = true`

This is the version that should be drawn in the method figure.

### 3.2 Geo Targets

Let:

- `z_tgt = E_T(c_tgt)`
- `z_fwd = E_T(t_fwd)`
- `z_rev = E_T(t_rev)`

Define caption displacement:

\[
\Delta = \mathrm{norm}(z_{tgt} - z_{src})
\]

### 3.3 Geo Loss

Only three terms are used:

\[
\mathcal{L}_{fwd} = 1 - \langle z_{fwd}, \Delta \rangle
\]

\[
\mathcal{L}_{rev} = 1 - \langle z_{rev}, -\Delta \rangle
\]

\[
\mathcal{L}_{zero} = \|z_{fwd} + z_{rev}\|_2
\]

Total geo loss:

\[
\mathcal{L}_{geo}
=
\mathcal{L}_{fwd}
+
\mathcal{L}_{rev}
+
\lambda_{zero}\mathcal{L}_{zero}
\]

Do **not** mention reverse-consistency in the main method description.


## 4. Joint Optimization

The two branches are trained synchronously on the same minibatch.

- retrieval branch updates:
  - visual encoder LoRA
  - text encoder LoRA
  - `phi`
  - logit scale
- geo branch updates:
  - geo text branch LoRA

The geo branch consumes the same batch supervision but focuses on edit-direction regularization.


## 5. Shared-B Design

We use **all-layer Shared-B** on the text encoder.

For LoRA:

\[
\Delta W = BA
\]

We share `B` between retrieval and geo text branches, and keep branch-specific `A`.

Current design:

- `shared_b_lora = True`
- `shared_b_num_layers = 12`
- `shared_b_retrieval_only_update = True`

Interpretation:

- the two branches share the same output-side update subspace
- branch-specific differences are carried by `A`
- the shared `B` is updated only by the retrieval branch


## 6. Merge

At inference, we keep:

- retrieval visual encoder
- retrieval `phi`
- retrieval base text weights

Only text LoRA is merged.

Current merge story to present:

1. training uses Shared-B
2. at merge time, keep the shared `B`
3. merge the task-specific `A`
4. apply **SVD denoising** on the merged `A`

Conceptually:

\[
A_m = w_r A_r + w_g A_g
\]

then

\[
A_m \leftarrow \mathrm{SVD\_topk}(A_m)
\]

while `B` is preserved from the shared branch.

This is the merge design to expose in the method section. Older merge variants do not need to appear in the writer-facing reference.


## 7. Gradient Conflict Analysis

The gradient probe is used for two purposes:

1. motivate why retrieval and geo objectives should not be naively collapsed
2. justify gradient projection as a stabilization tool

Layerwise conflict is measured by comparing:

- cross-objective cosine similarity
- same-objective random-split cosine similarity

Conflict score:

\[
GC_\ell = s_{base}^\ell - s_{cross}^\ell
\]

We also compare retrieval loss interference before and after projecting geo gradients away from conflicting retrieval gradients.


## 8. What Writer LLM Should Emphasize

The writer-facing method story should emphasize:

- a unified visual-text architecture in both branches
- geo branch uses the same image-conditioned anchor family, not a disconnected pure-text branch
- forward/reverse edit-direction supervision plus zero regularization
- all-layer Shared-B
- Shared-B-preserving SVD-denoised merge
- gradient conflict used as motivation and stabilization evidence

The writer-facing method story should avoid:

- old pure-text-only geo branch as the default design
- reverse-consistency term
- older alternative merge baselines as if they were part of the final method
- older img2text variants not used by the current design
