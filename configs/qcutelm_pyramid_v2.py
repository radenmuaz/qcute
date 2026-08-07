"""qcute.qcutelm_pyramid config: v2 — retry after v1 showed early, sustained
overfitting (session: unusually early for this session's baselines — val_bpb
bottomed at ~3.61 around step 999, then rose steadily to 4.08 by step 1799
while train bpb kept falling 2.97->2.04, a clean/consistent divergence, not
noise). All architecture settings unchanged from v1 (see qcutelm_pyramid_v1.py
for the full MPS-bugfix history — windowed attn_window=80, bit_head_mode=
"independent" — both still apply here). Two changes targeting the overfitting
directly:

1. cosine_decay=True (was False): v1 held lr at peak (6e-4) for the ENTIRE
   run after warmup+constant_steps — every baseline this session used the
   same constant-after-warmup schedule and did NOT overfit this early
   (bytelm/bpelm/v11 all showed the "same overfitting shape" only in their
   final ~1000 of 8000 steps), so the schedule alone doesn't explain v1's
   much-earlier divergence — but decaying lr as training progresses is a
   standard, cheap lever to try regardless, and this file already implements
   lr_at_warmup_constant_cosine for it.
2. weight_decay=1e-4 (was 1e-5, 10x): direct L2-style regularization pressure
   against exactly this run's overfitting failure mode.

Root-cause hypothesis (not verified, noted for the record): v1's much-faster
memorization than bytelm at the same param count/data may trace to the merge
mechanism giving the model a cheap way to encode near-lossless byte identity
into inserted code tokens (no code_match_loss/cyclic-target-problem-style
pressure forces codes to be abstracted/general rather than convenient
shortcuts, unlike qcutelm_vlt11's design) — if v2 doesn't meaningfully help,
this architectural hypothesis becomes the more likely explanation and would
need a different fix (e.g. an information bottleneck on the merge itself,
not just training-loop regularization).

    uv run python -m qcute.qcutelm_pyramid --config configs/qcutelm_pyramid_v2.py
"""
from pathlib import Path

Ks = (4, 4, 4)
d_model = 256
code_dim = None
context_len = 1024
quant_type = "ifsq"
fsq_levels = 8

n_heads = 4
n_layers = 4
mlp_mult = 4
attn_window = 80
untie_levels = False
bit_head_mode = "independent"

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-4        # 10x v1's 1e-5
warmup_steps = 500
cosine_decay = True         # v1 had this False (constant lr after warmup)
constant_steps = 100
log_every = 100
eval_every = 100
eval_batches = 20
