# BSQ entropy regularization

`bsq_entropy_reg`/`bernoulli_entropy` in `qcute_v5_concat.py`/`qcute_v5_stack.py`, ported from
archived `qcute/archive/qcutelm.py` (Yu et al. 2023 §3.2 MAGVIT-v2; BSQ 2024 closed-form).

Applied to raw pre-quantization projection $v \in \mathbb{R}^{dq}$ (before normalize/sign), per bit $i$:

$$p_i = \sigma(v_i)$$

Per-bit Bernoulli entropy:

$$H(p) = -\big(p\log p + (1-p)\log(1-p)\big)$$

**Per-example term** (mean over batch/sequence, summed over bits) — pushes each example toward confident bits:

$$\mathcal{H}_{\text{example}} = \mathbb{E}_{\text{batch}}\left[\sum_{i=1}^{dq} H(p_i)\right]$$

**Batch-usage term** — average probabilities across the batch first, then take entropy — pushes bit *usage* to spread out across examples:

$$\bar p_i = \mathbb{E}_{\text{batch}}[p_i], \qquad \mathcal{H}_{\text{batch}} = \sum_{i=1}^{dq} H(\bar p_i)$$

**Loss** (minimized):

$$\mathcal{L}_{\text{entropy}} = \mathcal{H}_{\text{example}} - \mathcal{H}_{\text{batch}}$$

Minimizing pulls $\mathcal{H}_{\text{example}}$ down (each example's bits become decisive/near ±1, matching the hard sign quantization) while pushing $\mathcal{H}_{\text{batch}}$ up (each bit's average usage across the batch stays near 0.5, so both corners get used) — countering collapse onto one dominant code.

Code mapping: `bernoulli_entropy(p)` = $H(p)$ elementwise; `bsq_entropy_reg(v)`: `per_example` = $\mathcal{H}_{\text{example}}$, `batch` = $\mathcal{H}_{\text{batch}}$, returns their difference. Weighted by `Config.entropy_reg_weight` (default `0.0`, off) and added to the total loss.
