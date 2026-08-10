"""Same base numbers as configs/qcute_refine_v4_4_bpelike_k4_1.py (Ks=(4,1), d_model=256,
n_layers=2, context_len=256, attn_window=(8,256)), but with use_gumbel_noise=True and
gumbel_tau=2.0 (softer soft-assignment before the hard STE snap) -- an attempt to fix the
codebook/index collapse found via scripts/probe_code_usage_entropy.py, NOT the self-vs-cross
conditioning question bpelike_k4_1_selfonly_only was testing (that isolation test showed
self-only collapses too, ruling out cross-level conditioning as the cause).

What the entropy probe found (see docs/status.md): code_0 (level0's own code, one per 4-byte
block) sits at 0.9-2.6 bits/8 max entropy across EVERY Ks[0]=4 config tested so far, including
the plain 1-level one -- collapse tracks Ks[0]=4 (forcing one discrete code to summarize 4 raw
bytes), not the architecture variant. code_1 (level1's own quantizer, Ks=(4,1) only) is even
worse, 0.00-0.11 bits (essentially one constant symbol). Compare to Ks[0]=1 configs (l1_k1,
l2_k1), which show 4.5-6.0 bits -- healthy, actively-used codebooks. None of the Ks[0]=4 configs
tested so far touched gumbel_tau/use_gumbel_noise (all defaults: tau=1.0, no noise) -- both are
standard VQ/discrete-bottleneck collapse mitigations (noise encourages exploration during
training, higher tau softens the assignment so gradients reach more of the codebook), worth
testing directly before concluding collapse is an inherent property of Ks[0]=4 block-grouping
rather than a fixable optimization issue.

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_bpelike_k4_1_gumbelfix.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_bpelike_k4_1_gumbelfix/run.log
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 2
context_len = 256
attn_window = (8, 256)
decode_pack_mode = "interleave"
decode_chunked = False
use_gumbel_noise = True
gumbel_tau = 2.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20

qual_gen_bytes = 64
qual_prompt_bytes = 64
