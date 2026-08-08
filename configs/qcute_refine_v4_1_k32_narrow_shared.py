"""qcute.qcute_refine_v4_1 config: architecturally matches configs/
qcute_refine_v4_k32_narrow_nonull_uncond.py (K=32/narrow-window,
Ks=(32,32), attn_window=(32,32), no null KV) — but under v4.1's own
EXTREME WEIGHT SHARING (`share_levellm=True`, the default): level 0 and
level 1 share the SAME self.blocks/ln_f/fuse_* weights (see qcute_
refine_v4_1.py's own module docstring), even though they keep their own
independent K/window (already identical here — 32/32 — but the mechanism
supports differing them, unlike v4).

Session rationale: first real training run of the new v4.1 lineage
("clone v4 to v4.1 for extreme weight sharing... only one levellm...
shared across byte and code levels"). Direct test of whether tying level
0's and level 1's trunk weights together helps or hurts vs. `qcute_
refine_v4`'s own independent-weights K=32 family (best so far: post+
no-null 2.4799, pre+no-null 2.4961 — docs/kv_contribution.md §7-10) at
a MUCH smaller param budget (sharing removes one whole level's worth of
self.blocks/ln_f/fuse_cross_pre parameters). `share_code_head` left at
its default False — the BSQ head/linear map (code_pre/embed/ntp_head)
for level 1 stays its own, only the big trunk itself is shared, per the
session's own two-separate-knobs design.

byte_repr="embed"/code_head_mode="independent" required by v4.1's
share_levellm first-impl scope (see Config's own docstring) — same
choices `k32_narrow` itself already used, no change needed there.
code_embed_mode left at "linear" (default) rather than "pq_table" —
share_levellm's first-impl scope doesn't touch code_embed_mode, kept
simple for this first run; a pq_table follow-up is a natural next step
once this baseline result is in.

    uv run python -m qcute.qcute_refine_v4_1 --config configs/qcute_refine_v4_1_k32_narrow_shared.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_1_k32_narrow_shared
"""
from pathlib import Path

Ks = (32, 32)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True
fuse_use_null_kv = False
share_levellm = True    # v4.1's whole point (default, stated explicitly for clarity)
share_code_head = False  # default — only the trunk is shared, not the BSQ head/linear map

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
