# Novelty Assessment

**Paper:** *Sketched Empirical-NTK Capability Fingerprints Predict Fine-Tuning Transfer and Interference in LLMs*

**Method in one line:** For each capability (task/data distribution), sketch (CountSketch → 4096-d, inner-product-preserving) the per-prompt gradient of the teacher-forced answer CE w.r.t. a small fixed parameter subset (~9M of 7.6B). The mean sketch inner product between a candidate fine-tuning set and any capability's eval set predicts the *measured* change in that capability's CE after actually fine-tuning — computed **before** training, from **base-model** gradients. Validated across 4 settings (RULER, heterogeneous capabilities, all-layer LoRA, Llama-3.1-8B), off-diagonal Spearman 0.62–0.87.

All citations below were verified against arXiv / OpenReview / PMLR / ACL Anthology / official proceedings during the review (July 2026).

---

## The 5–10 closest works

### 1. Malladi, Wettig, Yu, Chen, Arora — *A Kernel-Based View of Language Model Fine-Tuning* (ICML 2023)
[arXiv:2210.05643](https://arxiv.org/abs/2210.05643)
*(Note: the authors are Wettig & Yu, not "Wei" — correct any draft that says otherwise.)*

- **What they do:** Ask *whether* the (empirical) NTK describes fine-tuning of pretrained LMs; extend the NTK/kernel formalism to Adam via Tensor Programs; empirically show prompt-based fine-tuning is often kernel-like across ~14 NLP tasks.
- **How we differ:** They establish the *descriptive* fact (fine-tuning ≈ kernel regression on one task). We take the *constructive/predictive* step: use the kernel to forecast **cross-capability** transfer and interference before training, with **CountSketch** at **7–8B** scale, validated against **measured ΔCE**. They do not build a transfer/interference matrix, do not sketch, and do not predict forgetting.
- **Threat level: LOW–MODERATE.** This is our theoretical license, not our contribution. We must **cite it as the foundation** and must **not** claim to be first to view LLM fine-tuning as kernel-like.

### 2. Xia, Malladi, Gururangan, Arora, Chen — *LESS: Selecting Influential Data for Targeted Instruction Tuning* (ICML 2024)
[arXiv:2402.04333](https://arxiv.org/abs/2402.04333)

- **What they do:** Closest **machinery**. Build a datastore of LoRA gradients projected to low dimension (random projection), make them **optimizer-aware (Adam)** and collected **along a training trajectory**, and select training examples by gradient similarity to a **fixed target task**'s few-shot examples. ~5% selected data can beat full-data training.
- **How we differ:** (i) **Goal** — they rank data for *one* target task; we predict the *full n×n* cross-capability transfer **and interference** matrix. (ii) **Gradient source** — they use trajectory/optimizer-aware gradients (requires a warmup training run); we use **static base-model** gradients (no training, no trajectory). (iii) **Validation** — they measure downstream task accuracy of the selected subset; we validate the **first-order prediction itself against measured ΔCE** on held-out capabilities. (iv) **Robustness axis** — our LoRA setting shows the prediction survives when trained params are *disjoint* from sketched params. (v) We demonstrate **rehearsal targeting** (forgetting mitigation), not just acquisition.
- **Threat level: MODERATE–HIGH (closest single work).** Reviewers will invoke it. Position sharply: same low-dim-gradient idea, different object (interference matrix vs. target-task selection), different gradient (base-model vs. trajectory), different claim (validated ΔCE prediction).

### 3. Li, Sharma, Zhang — *Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity* (Grad-TAG, KDD 2024)
[arXiv:2409.06091](https://arxiv.org/abs/2409.06091)

- **What they do:** Train a base model once, then estimate **task affinity** for task *combinations* via a **linearization** technique using **low-dimensional gradient projections** as features in a logistic regression — explicitly to predict combined-task loss **before** running all the multitask training. Evaluated on multi-label graph tasks and instruction fine-tuning.
- **How we differ:** (i) **Prediction target** — they predict the *loss of a task combination* for **grouping decisions** (which tasks to co-train, i.e. positive transfer); we predict the **signed pairwise ΔCE** for every ordered (train, eval) pair, capturing **interference/forgetting** (positive ΔCE), which grouping methods largely ignore. (ii) **Estimator** — they fit a *learned surrogate* (logistic regression on projected gradients over sampled combinations); ours is the **parameter-free first-order NTK identity** (mean sketch inner product), no fitted model. (iii) **Scale/setting** — no 7B-scale cross-capability LLM validation against measured ΔCE. (iv) Base-model, single-pass sketches vs. their per-combination gradient features.
- **Threat level: HIGH on the phrase "gradient projections to predict task interaction before training."** Do **not** claim to be first to predict task affinity/interaction from projected gradients pre-training — Grad-TAG (and TAG below) got there. Our defensible novelty is the **signed transfer+interference matrix, parameter-free NTK inner-product estimator, LLM scale, and measured-ΔCE validation**.

### 4. Fifty, Amid, Zhao, Yu, Anil, Finn — *Efficiently Identifying Task Groupings for Multi-Task Learning* (TAG, NeurIPS 2021)
[arXiv:2109.04617](https://arxiv.org/abs/2109.04617)

- **What they do:** Measure inter-task **affinity** in a single training run by how a gradient step on one task changes another task's loss — exactly the first-order "one GD step on i changes loss on j" idea — then select groupings maximizing total affinity.
- **How we differ:** TAG measures affinity **during joint training** via lookahead loss changes (needs a live training run), for **grouping** in small-scale vision/MTL; it is not a sketched, pre-training, base-model **kernel** and does not target LLM capability forgetting. We formalize the same first-order quantity as a **static sketched eNTK** computed once, cheaply, per prompt, and validate the *signed* prediction at LLM scale.
- **Threat level: MODERATE.** Conceptual ancestor of "gradient-step effect on other task's loss." Cite prominently; differentiate on static-sketch-vs-lookahead, scale, and interference.

### 5. Doan, Bennani, Mazoure, Rabusseau, Alquier — *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix* (AISTATS 2021)
[arXiv:2010.04003](https://arxiv.org/abs/2010.04003)

- **What they do:** The **closest "forgetting from NTK alignment"** result. Prove that catastrophic forgetting between two tasks grows with their **NTK overlap**, formalize task similarity via an NTK overlap matrix, and propose PCA-OGD to mitigate.
- **How we differ:** (i) **Theoretical**, two-task **continual learning in vision/small nets**; no LLM, no sketching. (ii) Prescribes an **intervention** (orthogonal gradient descent) rather than a **validated quantitative forecast**. (iii) Overlap predicts *magnitude* of interference under aligned-task assumptions; we predict the **full signed matrix** (both damage and benefit) and check it against **measured ΔCE** across heterogeneous real capabilities and two model families.
- **Threat level: MODERATE–HIGH for the interference claim.** Do **not** claim to be first to connect NTK task alignment to forgetting — Doan owns that. Scope our claim to: **LLM-scale, measured-ΔCE-validated, sketched-fingerprint, both transfer and interference, pre-training, reusable across all pairings.**

### 6. Achille et al. — *Task2Vec: Task Embedding for Meta-Learning* (ICCV 2019)
[arXiv:1902.03545](https://arxiv.org/abs/1902.03545)

- **What they do:** Fingerprint a task as a vector using the **diagonal of the Fisher Information Matrix** (a gradient-based object) from a probe network; use embeddings for meta-learning/model selection and task similarity.
- **How we differ:** (i) **Gradient object** — they use per-parameter Fisher diagonal (a *marginal magnitude* statistic); we use **pairwise eNTK inner products** (direction × magnitude between samples) — our ablation shows Fisher-like *magnitude-only* features fail in 3/4 settings, and *direction* carries the transfer structure. (ii) **Purpose** — they *embed tasks for similarity/meta-learning*; we **predict measured fine-tuning ΔCE transfer/interference** and validate against actual training. (iii) No forgetting/interference matrix, no ΔCE validation, no LLM scale.
- **Threat level: LOW–MODERATE.** Named risk because "gradients as task fingerprints" sounds identical. The gradient object, the target quantity, and the validation are all different. Cite and differentiate explicitly (as above).

### 7. Park, Georgiev, Ilyas, Leclerc, Madry — *TRAK: Attributing Model Behavior at Scale* (ICML 2023)
[arXiv:2303.14186](https://arxiv.org/abs/2303.14186)  (and LoGra: [Choe et al., NeurIPS 2024, arXiv:2405.13954](https://arxiv.org/abs/2405.13954); DataInf; Grosse et al. 2023)

- **What they do:** Data attribution via **random (JL) projections of per-example gradients** + ensembling, for a model that has **already been trained**. LoGra/DataInf/Grosse scale influence to LLMs.
- **How we differ:** Same **random-projection primitive**, opposite **temporal direction**: TRAK/influence are **post-hoc** (attribute a *fixed trained* model's outputs to its training data); we are **forward-looking** (predict *future* fine-tuning transfer between capability *distributions* from *base-model* gradients). We also require no ensemble of trained models and no Hessian/inverse-Hessian.
- **Threat level: LOW.** Shares only the sketching machinery. Do **not** claim novelty of gradient random projection; **do** claim the forward-looking capability-transfer use.

### 8. Wang et al. — *NTK-Selector: mining general data for low-resource domain adaptation via NTKs* (arXiv:2511.07380, 2026)
[arXiv:2511.07380](https://arxiv.org/abs/2511.07380)

- **What they do:** **Concurrent (late 2025/2026).** Compute random-projected (8192-d) NTK scores at initialization, exploit stable **directional** NTK structure during instruction tuning, and select auxiliary general-domain data for a **single** low-resource target domain on Llama-3/Qwen-3.
- **How we differ:** Essentially "LESS with an NTK score" for **one target domain** — a *data-selection* method, not a **full transfer/interference matrix** and not a validated **ΔCE** predictor. Their finding that NTK *direction* is stable while magnitude drifts independently corroborates our ablation (direction > magnitude).
- **Threat level: MODERATE but CONCURRENT.** Flag as concurrent work; not prior art that pre-empts us, but a reviewer may cite it. Differentiate on scope (single-domain selection vs. cross-capability signed matrix) and validation.

### 9. Hidekel & Raviv — *Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation* (arXiv:2606.18024, 2026)
[arXiv:2606.18024](https://arxiv.org/abs/2606.18024)

- **What it is:** **The authors' own companion work** (self-citation — Ido Nitzan Hidekel matches the project author). A closed-form NTK **function-space** predictor of the forgetting vector through the cross-task kernel before a new-task step; shows forgetting concentrates in few NTK eigenmodes.
- **Relationship:** Theoretical sibling. This empirical paper *operationalizes and validates at LLM scale* a related eNTK-kernel prediction with **sketching** and **measured ΔCE**. Cite as complementary; do not present as independent external corroboration.
- **Threat level: NONE (self).** Manage as an honest self-citation.

### 10. (Runner-up thread) Gradient-conflict MTL — PCGrad, GradVaccine, Du et al.
[Yu et al. NeurIPS 2020, arXiv:2001.06782](https://arxiv.org/abs/2001.06782); [Wang et al. ICLR 2021, arXiv:2010.05874](https://arxiv.org/abs/2010.05874); [Du et al. 2018, arXiv:1812.02224](https://arxiv.org/abs/1812.02224)

- **What they do:** Use gradient **cosine similarity / conflict** (negative alignment) as a signal to modify optimization and reduce interference during multi-task training.
- **How we differ:** They *react* to conflict during training to improve optimization; we *predict* the resulting ΔCE interference matrix *before* training from sketched base-model gradients. Also GradVaccine shows gradient similarity along the trajectory correlates with transfer — supportive prior evidence, not a predictor of measured ΔCE.
- **Threat level: LOW.** Cite as the "gradient inner product = interference" intuition; we make it *predictive and pre-training*.

---

## Bottom line on the key question ("does anyone predict cross-task interference/forgetting from gradient inner products BEFORE fine-tuning?")

Yes, **partially, in adjacent forms** — and this must be stated bluntly:

- **Grad-TAG (KDD 2024)** and **TAG (NeurIPS 2021)** predict/estimate task *affinity* from gradient effects **before/without** full multitask training — but for **grouping (positive transfer)**, below LLM scale, and (Grad-TAG) via a *learned surrogate* rather than the parameter-free NTK identity.
- **Doan et al. (AISTATS 2021)** connect NTK alignment to forgetting **theoretically**, two-task, vision.
- **LESS / NTK-Selector** use projected base/trajectory gradients to select data for a **fixed single target**, not a signed cross-capability matrix.

**No verified external work** computes and **validates against measured ΔCE** a *full signed transfer-and-interference matrix over heterogeneous LLM capabilities, from a single reusable sketched base-model eNTK fingerprint, at 7–8B scale, robust to trained≠sketched parameters.* That specific combination is open. (Searched extensively; nothing closer was found.)

---

## Recommended novelty claims (SAFE to make)

1. **First to compute and empirically validate a full signed cross-capability transfer *and* interference matrix for LLM fine-tuning** from a first-order eNTK prediction, checked against **measured ΔCE** across 4 settings and 2 model families (Qwen2.5-7B, Llama-3.1-8B).
2. **First to use inner-product-preserving CountSketch fingerprints of base-model eNTK gradients** as reusable, pre-training capability descriptors at 7–8B scale — one sketch per prompt serves all capability pairings, no Hessian, no trajectory, no per-query solve.
3. **A robustness result specific to us:** the prediction holds when the *trained* parameters (all-layer LoRA) are **disjoint** from the *sketched* subset — gradient geometry is shared across the network.
4. **An ablation isolating the carrier of transfer structure:** gradient *direction* (via inner product / cosine) is necessary and sufficient; *magnitude alone* fails in 3/4 settings; *embedding/representation similarity* fails in 4/4 and can **anti-predict** — i.e., "similar data" heuristics are unreliable for transfer.
5. **A diagnostic contrast** motivating the method: activation/attention **rank** is task-intrinsic but has no incremental per-sample predictive value for correctness, whereas gradient (eNTK) geometry does.
6. **An honest null:** self-gradient *norm* is confidence-in-disguise (no gain over free logit baselines) — the *kernel structure*, not gradient magnitude, carries the signal.
7. A practical, negligible-cost workflow for **data selection, rehearsal targeting, and forgetting-risk assessment** before fine-tuning.

## Claims to AVOID (will draw a rejection or a "not novel" flag)

- ❌ "First to view / show LLM fine-tuning is NTK/kernel-like." → **Malladi et al. 2023.**
- ❌ "First to use random projections / sketching of gradients for attribution or selection." → **TRAK, LESS, LoGra.**
- ❌ "First to predict task affinity / interaction from projected gradients before training." → **Grad-TAG (KDD 2024), TAG (NeurIPS 2021).**
- ❌ "First to link NTK/gradient alignment to catastrophic forgetting." → **Doan et al. 2021** (NTK overlap matrix).
- ❌ "First to fingerprint a task with gradients." → **Task2Vec (Fisher), TaskEmb.**
- ❌ "First to select fine-tuning data by gradient similarity." → **LESS; NTK-Selector (concurrent).**
- ❌ Any framing of the self-gradient-norm signal as a novel useful method — it is reported (correctly) as a **null**; cite gradient-uncertainty prior work (Lee & AlRegib 2020) so reviewers don't think we missed it.
- ⚠️ Treat **NTK-Selector (2026)** and **Hidekel & Raviv (2026)** as **concurrent / own** work, not independent prior corroboration.

## Verdict

**Novelty is defensible if scoped precisely.** The individual ingredients (eNTK view of LLM fine-tuning; gradient random projection; gradient-based task affinity; NTK–forgetting link) all exist, and reviewers *will* cite Malladi, LESS, Grad-TAG/TAG, and Doan. The paper survives on the **specific synthesis no prior work delivers**: a *reusable sketched base-model eNTK fingerprint* that predicts the *full signed transfer-and-interference matrix* over *heterogeneous capabilities* at *7–8B scale*, *validated against measured ΔCE*, *robust to trained≠sketched parameters*, with a clean *direction-not-magnitude* ablation. Lead with that sentence; never claim the components.
