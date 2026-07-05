# Sketched Empirical-NTK Capability Fingerprints Predict Fine-Tuning Transfer and Interference in LLMs

*Working draft — Motivation and Methods sections. Numbers from `reports/methods.md`,
`reports/main/summary.md`, `reports/llama_run/summary.md` (Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct, July 2026).*

---

## What we offer / contributions

1. **A cheap, pre-training-time predictor of fine-tuning transfer and interference.**
   For each capability (task/data distribution) we compute a *capability fingerprint*:
   CountSketch-compressed per-prompt gradients of the answer cross-entropy with respect
   to a small, fixed parameter subset (~73M of 7.6B parameters). The mean sketch inner
   product between a candidate fine-tuning set and any held-out capability predicts the
   *measured* change in that capability's loss after actually fine-tuning — before any
   training is run.

2. **Validation across four settings** with the same recipe and no per-setting tuning:
   (a) homogeneous long-context tasks (RULER), (b) heterogeneous real capabilities
   (math CoT, code synthesis, QA, retrieval, aggregation), (c) a trainer whose updated
   parameters are *disjoint* from the sketched subset (LoRA on all layers), and
   (d) a second model family (Llama-3.1-8B). Off-diagonal Spearman between predicted
   kernel and measured ΔCE: **0.68, 0.87, 0.62, 0.69** (all p ≤ 0.004).

3. **An ablation isolating the ingredient that carries transfer structure.** Gradient
   *magnitude* alone fails in 3 of 4 settings (ρ = 0.25, 0.28, 0.09, n.s.); embedding
   (representation) similarity fails in 4 of 4 and is even significantly *anti*-correlated
   in the heterogeneous setting (ρ = −0.62). Gradient *direction* is necessary; the full
   inner product (direction × magnitude) is the best predictor everywhere.

4. **A motivating diagnostic study of "rank demand" in long-context tasks** showing that
   while spectral structure of attention/hidden states is task-intrinsic and stable across
   model families, it has *no incremental per-sample predictive value* for correctness once
   entropy confounds are controlled — whereas per-prompt gradient (eNTK) features are the
   strongest signal in the same data. This negative→positive contrast motivates the method:
   the useful geometry lives in gradient space, not activation space.

5. **An honest null.** The self-gradient norm (gradient of the model's own generated
   answer), proposed as a test-time correctness signal, adds nothing over free
   confidence baselines (CV ΔAUC ≈ −0.002 on both models). We report it as a caution:
   raw gradient norms are confidence-in-disguise; the kernel structure is what matters.

**Practical use case.** Before fine-tuning an LLM on new data, sketch the gradients of the
new data and of the capabilities you care about preserving (a few dozen prompts each, one
backward pass per prompt, 4096 floats stored per prompt). The kernel row tells you which
capabilities will improve, which will be damaged, and how strongly — enabling data
selection, rehearsal targeting, and forgetting risk assessment at negligible cost.

---

## 1. Motivation

### 1.1 Do long-context tasks differ in the structure they induce? Yes — but rank does not predict success

We began with a diagnostic question: which long-context tasks induce high-rank
attention/representation structure, and does rank predict correctness beyond context
length, task family, and attention entropy? On five RULER task families (single-needle
retrieval, multi-key retrieval, variable tracking, common-word aggregation, QA) at
context lengths 1k–8k, with 1,000 samples per model:

- **Rank demand is task-intrinsic and model-stable.** Hidden-state effective rank orders
  the families identically (up to one adjacent swap) in Qwen2.5-7B (QA 146 > multi-needle
  101 > aggregation 22 > tracking 2.6 > single-needle 2.1) and Llama-3.1-8B (multi-needle
  122 > QA 88 > aggregation 82 > tracking 36 > single-needle 28). The ordering is not the
  attention-entropy ordering, so it is not an entropy artifact.
- **But rank adds no per-sample predictive value.** In 5-fold cross-validated logistic
  regression predicting correctness, adding rank features on top of length, family, and
  entropy changes AUC by −0.002 (Qwen) and +0.002 (Llama). An in-sample gain of +0.010
  disappears under cross-validation — a pure overfit.

### 1.2 Where the signal actually lives: gradient space

In the same runs we computed per-prompt empirical-NTK features — gradients of the
teacher-forced answer loss, compressed by CountSketch. These showed the structure the
activation-space metrics lacked: self-kernel norms span ~600–790× across families;
within-family gradient cosine similarity is 0.21 (Qwen, all lengths; 0.44 at ≤2k) / 0.41
(Llama) versus 0.02–0.05 across families,
in both models. Prompts from the same capability occupy a tight cone in gradient space;
different capabilities are nearly orthogonal.

This suggests the correct object of study is not "what rank does the task induce" but
"what does the task's gradient geometry predict about *training*" — since first-order
learning dynamics are governed exactly by gradient inner products (the empirical NTK).
That is the method.

---

## 2. Method

### 2.1 Setup and notation

Let f_θ be a causal LM with parameters θ, and let a *capability* c be a distribution of
(prompt x, answer y) pairs. For a sample s = (x, y) define the teacher-forced answer loss
L(s; θ) = CE of y under f_θ(· | x), and the answer gradient g(s) = ∇_θ L(s; θ) restricted
to a fixed parameter subset S.

**Parameter subset.** S = the query and value projections (q_proj, v_proj) of 5 layers at
depth fractions {0, ¼, ½, ¾, 1} — approximately 73M parameters of a 7.6B model (0.97%; 105M for Llama-3.1-8B).
The subset is fixed once per model and never tuned per task.

**First-order rationale.** One SGD step on sample s_i with learning rate η changes the
loss on sample s_j by ΔL(s_j) ≈ −η ⟨g(s_i), g(s_j)⟩ + O(η²): the empirical NTK entry.
Averaging over a training set T_i and an evaluation set E_j gives a first-order prediction
of capability-level transfer (negative ΔCE) or interference (positive ΔCE):

    K[i, j] = mean over s∈T_i, s'∈E_j of ⟨g(s), g(s')⟩;   predicted ΔCE[i, j] ∝ −K[i, j].

### 2.2 Sketched fingerprints

Storing g(s) (73M floats per prompt) is wasteful; the quantity of interest is inner
products. We apply **CountSketch**: a fixed random signed-hash projection Φ: R^73M → R^4096,
drawn once per model with a fixed seed and shared across all samples, applied per parameter
tensor and summed. CountSketch is an unbiased inner-product-preserving projection
(E⟨Φg, Φg'⟩ = ⟨g, g'⟩), so the 4096-dim sketches reproduce the Gram matrix without ever
materializing gradients. The *fingerprint* of capability c is the set of sketches
{Φ g(s) : s ∈ c} — 16 KB per prompt.

Cost per prompt: one forward + one backward pass on the frozen model with gradients
enabled only on S (with gradient checkpointing, fits generation-scale memory). No
training, no labels beyond the reference answers already needed for evaluation.

### 2.3 Prediction protocol

Given capabilities c_1…c_n with disjoint train sets T_i (seed-separated from eval sets
E_j):

1. Sketch all train and eval samples once (same Φ).
2. Predicted transfer matrix: K[i, j] = mean pairwise sketch inner product T_i × E_j.
3. Ground truth: fine-tune the model on T_i alone, measure ΔCE[i, j] on every E_j,
   restore weights, repeat for each i.
4. Score: Spearman rank correlation between −K and measured ΔCE over all (i, j) pairs,
   and — the stricter test — over **off-diagonal pairs only** (pure cross-task
   transfer/interference, removing the easy "training on X helps X" signal).

### 2.4 Experimental settings (validation axes)

| setting | what it tests | trainer | capabilities |
|---|---|---|---|
| RULER 5×5 | base validity | Adam on subset S (fp32 masters, loss-scaled) | 5 RULER families @1024 |
| heterogeneous | real, dissimilar skills | same | GSM8K CoT, MBPP code, SQuAD QA, needle retrieval, word aggregation |
| LoRA | trained params ≠ sketched params | rank-16 LoRA on q/v of *all* layers | 5 RULER families |
| Llama-3.1-8B | model-family generality | Adam on subset S | 5 RULER families |

The LoRA setting is the key robustness axis: the fingerprint is computed on 5 layers'
q/v while training updates low-rank adapters on all 28 layers — the prediction survives
because gradient geometry is shared across the network, not because we sketch exactly
what we train.

n_train = 32, n_eval = 30 per capability; 3 epochs; lr 2e-5 (subset) / 2e-4 (LoRA);
outcome is gold-answer ΔCE (accuracy tracked as secondary). Fine-tuning fp16 weights uses
fp32 master copies with loss scaling and non-finite-update skipping (direct fp16 Adam
destroys the model).

### 2.5 Baselines (what carries the structure?)

Same measured ΔCE matrices, four predictors computed pre-training:

- **ntk_kernel** — mean ⟨g, g'⟩ (the method: direction × magnitude);
- **ntk_cos** — mean cosine (direction only);
- **grad_magnitude** — mean ‖g‖·‖g'‖ (magnitude only);
- **embed_cos** — mean cosine of last-hidden-state mean-pooled prompt embeddings
  (representation similarity, the standard "similar data" heuristic).

### 2.6 Results summary (off-diagonal Spearman ρ vs measured ΔCE)

| predictor | RULER | hetero | LoRA | Llama |
|---|---|---|---|---|
| **ntk_kernel (ours)** | **0.68** (p=.001) | **0.87** (p<.0001) | **0.62** (p=.004) | **0.69** (p=.0007) |
| ntk_cos | 0.67 | 0.79 | 0.55 | 0.59 |
| grad_magnitude | 0.25 (n.s.) | 0.80 | 0.28 (n.s.) | 0.09 (n.s.) |
| embed_cos | — | **−0.62** (p=.003) | −0.30 (n.s.) | 0.02 (n.s.) |

Pooled Pearson (all pairs) for ntk_kernel: 0.98 / 0.92 / 0.88 / 0.75. Direction-only
(ntk_cos) is close behind the full kernel; magnitude-only collapses except in the
heterogeneous setting (where capability-scale differences happen to align); embedding
similarity is useless or actively misleading — representationally similar data does not
predict training-time transfer.

### 2.7 The self-gradient null (cautionary)

A natural simpler idea — use the gradient norm of the model's *own generated answer* as a
test-time correctness signal — fails: AUC 0.78/0.76 (Qwen/Llama) versus 0.90 for free
entropy/logprob baselines, and CV incremental ΔAUC of −0.0016/−0.0001 on top of them.
Gradient *norms* are confidence in disguise; only the *kernel* (inner-product structure
across samples) carries information not available from logits.

---

## 3. Relation to prior work (sketch)

- Extends cross-entropy/gradient-norm forgetting predictors (per-sample forgetting in
  vision/classification) to capability-level LLM fine-tuning via the empirical NTK.
- Differs from influence functions: no Hessian inversion, no per-query solve; a single
  fixed sketch per sample supports all future pairings.
- Differs from data-selection-by-embedding (representation similarity): §2.6 shows
  embedding similarity does not predict, and can anti-predict, measured transfer.
- TRAK / gradient-sketching estimators share the random-projection machinery; here the
  target is forward-looking fine-tuning transfer between capability distributions, not
  post-hoc attribution of a fixed trained model.

## 4. Limitations (current)

- 5 capabilities per setting; short fine-tuning horizons (3 epochs × 32 samples);
  first-order theory predicts decay of accuracy for longer/larger-lr training —
  horizon-scaling experiment pending.
- ΔCE as outcome; accuracy deltas are noisier at n_eval = 30.
- 7–8B scale, two model families, fp16 Turing hardware.
- Parameter subset fixed a priori (q/v of 5 layers); subset-choice sensitivity untested.

*Pending experiments (queued): safety/multilingual/reasoning capability splits; horizon
scaling (does first-order prediction decay with training length?); rehearsal targeting
(select data to rehearse by predicted interference, compare against random rehearsal).*
