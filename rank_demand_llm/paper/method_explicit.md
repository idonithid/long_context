# The Method, Explicitly

Every symbol below maps to a concrete object in the code
(`rank_demand/ntk_features.py`, `scripts/run_interference.py`).

## 1. Setting

A frozen pretrained causal LM $f_\theta$ (Qwen2.5-7B-Instruct or
Llama-3.1-8B-Instruct), parameters $\theta \in \mathbb{R}^P$, $P \approx 7.6\cdot10^9$.

A **capability** $c_i$ is a data distribution given as two disjoint finite sets
(different generator seeds):

- train set $T_i = \{s^{(1)}, \dots, s^{(32)}\}$
- eval set $E_i = \{s^{(1)}, \dots, s^{(30)}\}$

Capabilities used: RULER task families (needle retrieval, multi-key retrieval,
variable tracking, common-word aggregation, SQuAD QA) in one setting; real skills
(GSM8K math CoT, MBPP code, SQuAD QA, retrieval, aggregation, and in the extended
run safety/hh-rlhf, XQuAD-zh multilingual, ARC-Challenge reasoning) in another.

**The question:** if we fine-tune $f_\theta$ on $T_i$, what happens to the loss on
every other capability's $E_j$ — and can we predict that *before training*?

## 2. Data points: what is $x$, what is $y$

A sample is a pair $s = (x, y)$:

- $x$ = the **prompt**: the task input (context documents + needle/question/etc.)
  rendered through the model's chat template, ending with the assistant turn opened
  and a task-specific *answer prefix* pre-filled (e.g. "The special magic number
  mentioned in the provided text is"). Token length up to 1024 in the interference
  experiments.
- $y$ = the **answer**: the gold target string (needle value, GSM8K solution with
  `#### answer`, reference Python function, SQuAD span, ...), tokenized to
  $y = (y_1, \dots, y_A)$. Multiple expected answers are joined by commas.

So concretely: `full_ids = concat(tokenize(chat_template(x)), tokenize(" " + y))`.

## 3. The loss

Teacher-forced answer cross-entropy — loss **only on answer tokens**, prompt tokens
label-masked ($-100$):

$$
\mathcal{L}(s;\theta) \;=\; -\frac{1}{A}\sum_{t=1}^{A}
\log p_\theta\!\left(y_t \,\middle|\, x,\, y_{<t}\right).
$$

This is the same quantity used three ways: (i) its gradient is the fingerprint,
(ii) it is the fine-tuning training objective on $T_i$, (iii) its change on $E_j$
is the measured outcome $\Delta\mathrm{CE}$.

## 4. The fingerprint: per-prompt gradient on a fixed subset

Fix once per model a parameter subset $S \subset \{1,\dots,P\}$: the `q_proj` and
`v_proj` weight matrices of 5 layers at depth fractions $\{0, \tfrac14, \tfrac12,
\tfrac34, 1\}$ (layers 0, 7, 14, 21, 27 of 28). Sizes: 10 tensors, **73.4M params
(0.97%)** for Qwen2.5-7B; **104.9M (1.3%)** for Llama-3.1-8B. Never tuned per task.

Per-sample gradient (one forward + one backward on the frozen model, gradients
enabled only on $S$, gradient checkpointing on, fp16 loss scaled by 1024 then
descaled in fp32):

$$
g(s) \;=\; \nabla_{\theta_S}\, \mathcal{L}(s;\theta) \;\in\; \mathbb{R}^{|S|}.
$$

## 5. NTK: why gradient inner products predict training

**One SGD step — the leading term and the curvature it drops.** Take one gradient
step on sample $s_i$ with learning rate $\eta$: $\theta' = \theta - \eta\, g(s_i)$
(restricted to $S$). Expand the loss on any other sample $s_j$ to **second order** in
the displacement $-\eta g(s_i)$:

$$
\Delta\mathcal{L}(s_j)
\;=\; \mathcal{L}(s_j;\theta') - \mathcal{L}(s_j;\theta)
\;=\; \underbrace{-\eta\, \langle g(s_i), g(s_j) \rangle}_{\text{first order (eNTK)}}
\;+\; \underbrace{\tfrac12 \eta^2\, g(s_i)^\top H_j\, g(s_i)}_{\text{curvature}}
\;+\; O\!\big(\eta^3 \|g(s_i)\|^3\big),
$$

where $H_j = \nabla^2_{\theta_S}\mathcal{L}(s_j;\theta)$ is the **Hessian** of $s_j$'s
loss at the base parameters. The dropped term is **not** a bare $O(\eta^2)$ with a
universal constant — its size is set by the curvature and the **squared gradient
norm**, bounded by

$$
\big|\tfrac12 \eta^2\, g(s_i)^\top H_j\, g(s_i)\big|
\;\le\; \tfrac12 \eta^2\, \|H_j\|_2\, \|g(s_i)\|^2 .
$$

So the first-order (eNTK) prediction dominates exactly when
$\eta\, \|H_j\|_2\, \|g(s_i)\|^2 \ll |\langle g(s_i), g(s_j)\rangle|$: small learning
rate **and** mild curvature along the update direction. The inner product
$k(s_i, s_j) = \langle g(s_i), g(s_j)\rangle$ is the **empirical neural tangent
kernel** (loss-gradient form: the parameter-space NTK $\nabla_\theta f \nabla_\theta
f^\top$ contracted with the loss Jacobians on both sides). Positive kernel → training
on $s_i$ *reduces* loss on $s_j$ (transfer); negative → *increases* it (interference).

**A whole training run.** Fine-tuning on $T_i$ is a sequence of steps with net
parameter displacement $\Delta\theta_i$. To second order,

$$
\Delta\mathcal{L}(s_j)
\;=\; \langle g(s_j), \Delta\theta_i \rangle
\;+\; \tfrac12 \Delta\theta_i^\top H_j\, \Delta\theta_i
\;+\; O\!\big(\|\Delta\theta_i\|^3\big).
$$

For a (preconditioned) descent trajectory $\Delta\theta_i = -\eta \sum_t P_t g^{(t)}$
($P_t = I$ for SGD, diagonal for Adam), two approximations collapse the leading term
to a base-model kernel: **(A1) trajectory stability** — over a short run gradients
rotate little, $g^{(t)} \approx g$ at base $\theta$; **(A2) scalar preconditioning** —
$P_t \approx cI$, discarding Adam's per-coordinate reweighting. Then
$\Delta\theta_i \approx -\eta c \sum_{s\in T_i} g(s)$ and

$$
\widehat{\Delta\mathrm{CE}}[i, j] \;\propto\; -\,K[i,j],
\qquad
K[i,j] \;=\; \frac{1}{|T_i||E_j|} \sum_{s \in T_i} \sum_{s' \in E_j}
\big\langle g(s),\, g(s') \big\rangle .
$$

$K[i,j]$ is computable **before any training**: it needs only base-model gradients.

**Error budget (why we test rank, not magnitude).** Four terms separate $K$ from
measured $\Delta\mathrm{CE}$: **(1) curvature** $\tfrac12 \Delta\theta_i^\top H_j
\Delta\theta_i$, of order $\|H_j\|_2 \|\Delta\theta_i\|^2$, growing with $\eta^2$ and
horizon$^2$; **(2) trajectory drift** (error in A1), growing with $\eta \times$ steps;
**(3) preconditioning** (error in A2, since the real optimizer is Adam); **(4) sketch
variance** $O(1/m)$. All four preserve the *sign* and coarse ordering of $K$ for small
$\eta$ and short horizons — which is why the claim is tested as *rank* correlation
(Spearman) between $-K$ and measured $\Delta\mathrm{CE}$, with the proportionality
constant (absorbing $\eta$, $c$, step count) left free, rather than as an exact
magnitude match. The horizon-scaling experiment probes when (1)–(2) grow large enough
to break the rank prediction.

## 6. CountSketch: making the fingerprint 4096 floats

Storing $g(s)$ (73M floats/prompt) is wasteful; only inner products matter.
CountSketch $\Phi: \mathbb{R}^{|S|} \to \mathbb{R}^{m}$, $m = 4096$: draw once per
model (fixed seed) a hash $h: [|S|] \to [m]$ and signs $\sigma: [|S|] \to \{\pm1\}$;

$$
(\Phi g)_r \;=\; \sum_{k:\, h(k) = r} \sigma(k)\, g_k .
$$

(Implemented per parameter tensor with a tensor-specific seed and summed into one
$m$-vector — equivalent to sketching the concatenated gradient.) Properties:

- **Unbiased:** $\mathbb{E}\,\langle \Phi g, \Phi g' \rangle = \langle g, g' \rangle$.
- **Variance** $\le \frac{1}{m}\big(\|g\|^2\|g'\|^2 + \langle g,g'\rangle^2\big)$ —
  at $m = 4096$ the relative error on our Gram entries is a few percent, far below
  the between-capability kernel spread (self-kernels span ~600–790× across
  capabilities).

The stored fingerprint of capability $c$ is $\{\Phi\, g(s) : s \in T_c \cup E_c\}$ —
**16 KB per prompt**, and $K[i,j]$ is estimated as the mean pairwise inner product
of sketches.

## 7. Ground truth: what is actually optimized

For each train capability $T_i$:

1. **Pre:** $\mathrm{CE}_j^{\text{pre}} = \frac{1}{|E_j|}\sum_{s\in E_j}\mathcal{L}(s;\theta)$ for all $j$.
2. **Fine-tune:** minimize $\frac{1}{|T_i|}\sum_{s \in T_i} \mathcal{L}(s;\theta_S)$
   with Adam, lr $2\cdot10^{-5}$, 3 epochs, gradient accumulation 8, global-norm
   clip 1.0. fp16 weights via fp32 master copies + loss scaling (direct fp16 Adam
   destroys the model). Two trainer variants: (a) **subset** — update exactly
   $\theta_S$; (b) **LoRA** — rank-16 adapters on q/v of *all 28* layers, so trained
   parameters are disjoint from the sketched subset (robustness check).
3. **Post:** re-measure; $\Delta\mathrm{CE}[i,j] = \mathrm{CE}_j^{\text{post}} - \mathrm{CE}_j^{\text{pre}}$.
4. **Restore** $\theta$ exactly; next $i$.

**Score:** Spearman$(-K[i,j],\ \Delta\mathrm{CE}[i,j])$ over all pairs, and over
off-diagonal pairs only ($i \ne j$; the strict cross-capability test that excludes
the trivial "training on X helps X" diagonal).

## 8. Results, one line each

Off-diagonal Spearman: RULER 0.68 (p=.001) · heterogeneous 0.87 (p<.0001) ·
LoRA-trainer 0.62 (p=.004) · Llama-3.1-8B 0.69 (p=.0007). Ablation: cosine-only
(direction) nearly matches; magnitude-only fails 3/4; embedding similarity fails
4/4 (anti-correlates on heterogeneous, ρ=−0.62). Gradient direction — not scale,
not representation similarity — carries the transfer structure.
