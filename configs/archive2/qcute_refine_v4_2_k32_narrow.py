"""qcute.qcute_refine_v4_2 config: architecturally matches configs/
qcute_refine_v4_1_k32_narrow_shared.py (K=32/narrow-window, no null KV,
extreme weight sharing) — but under v4.2's own UNCONDITIONAL, further
unified sharing scheme: a single shared `dq=8` BSQ code width and a
single shared embed/ntp_head/code_pre used by EVERY level, byte level
included (v4.1 still gave level 0 its own byte_embed/256-way head; v4.2
removes that special case entirely — see qcute_refine_v4_2.py's own
module docstring for the full "session ask" rationale). Fusion is
CONCAT-ONLY in this file (session: "make v4.2 use by default only concat
mode, remove any cross attn stuff") — no `fuse_position` field exists
anymore, `Config.fuse_use_null_kv` is the only fusion-shape knob left.

Session rationale: first real training run of the v4.2 lineage. `dq=8`
(the minimum/default — byte level needs exactly 8 bits, no wasted
padding-bit capacity) means level 0's own byte prediction is via an
independent-per-bit BCE head (chain_bce_loss over 8 bits), not the exact
256-way softmax `bytelm.py`/v4's `byte_repr="embed"` used — a real,
documented caveat (session: "is val bpb computation valid... since bits
are indp heads" — independent-bit BCE is a valid UPPER BOUND on true
bits-per-byte, not the exact cross-entropy, so this run's own `val_bpb`
isn't strictly apples-to-apples against baselines that use an exact
byte-level head; still a meaningful, internally-consistent number for
comparing v4.2 configs against each other).

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow
"""
from pathlib import Path

Ks = (32, 32)
dq = 8
d_model = 256
n_layers = 1
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (32, 32)

fuse_encoder_levels = True
fuse_use_null_kv = False

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
