"""qcute.qcutelm_pyramid config: initial test of the single-LM, flat multi-
resolution pyramid design (see qcutelm_pyramid.py's module docstring —
session: "do not separate E D codelm, just single lm, like qcute fifo").

Ks=(4,4,4) — same compression schedule as qcutelm_vlt11_k2_l3_full.py, for
a like-for-like comparison. d_model=256/n_layers=4/n_heads=4/mlp_mult=4 —
matches bytelm_xs_mtp4_ctx1024's architecture dims exactly. untie_levels
left at its default (False, shared merge/code_in_embed weights across all
3 levels — requires uniform Ks, satisfied here).

flat_len for this config = 1024 + 256 + 64 + 16 = 1360 (context_len plus
one inserted code token per Ks[i]-block at every level) — this is the
actual sequence length the single LM attends over every step.

attn_window=80 (was -1/dense in an earlier draft — found to hit a severe
MPS backward-pass slowdown specifically at T=1360, a non-power-of-2
length: forward+backward+opt.step took 9s on a single call and grew
unboundedly across repeated calls in a training-shaped loop, vs. 1.1s at
the power-of-2 T=1024 bytelm itself uses. Root cause not fully isolated
(suspected MPS SDPA-backward kernel path for odd sequence lengths,
possibly compounded across repeated calls); windowing sidesteps it
entirely and was verified both fast (~0.8-1.1s/call, stable across
repeated calls) and safe: windowed attention only RESTRICTS visibility
within the flat sequence's already-verified-correct causal order (see
module docstring's leakage test) — it can never violate it, only see
less. 80 divides flat_len evenly (1360/80=17) and is close to v11's own
default window (64).

No e_ntp_weight/code_match_weight here — this design has neither: codes
are deterministic merges of already-known content (not autoregressive
forecasts), so there's nothing to collapse and nothing to separately
predict. See module docstring for the full argument.

bit_head_mode="independent" (was "chain"): isolated component-by-component
(byte embed/merges/scatter/main-attn/gather/byte_loss-forward were all
individually fast, ~2.1s combined) — the entire slowdown traced to
BitPredictHead's chain mode's BACKWARD pass specifically (11.6s), and
only when chained through the full upstream graph (chain mode alone, on a
fresh leaf tensor, backward was fast — ~0.5s). Switching to "independent"
(plain Linear head, no per-bit chain self-attention) confirmed the fix:
forward 0.97s + backward 0.62s, matching bytelm's own known-good
performance. Trades exact bit-chain-rule factorization for speed — a
legitimate, already-designed tradeoff (see Config.bit_head_mode's
docstring), not a hack. Root cause of chain mode's backward-specific MPS
pathology not fully isolated; worth revisiting if chain mode's better
bit-calibration turns out to matter for final quality.

    uv run python -m qcute.qcutelm_pyramid --config configs/qcutelm_pyramid_v1.py
"""
from pathlib import Path

Ks = (4, 4, 4)
d_model = 256
code_dim = None            # shared space: codes are native 256-dim vectors, same as d_model
context_len = 1024
quant_type = "ifsq"
fsq_levels = 8

n_heads = 4
n_layers = 4
mlp_mult = 4
attn_window = 80            # windowed — see docstring above (dense hit a severe MPS slowdown at T=1360)
untie_levels = False        # shared merge/code_in_embed across all 3 levels
bit_head_mode = "independent"   # see docstring above — chain mode's backward pass was pathological on MPS

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
