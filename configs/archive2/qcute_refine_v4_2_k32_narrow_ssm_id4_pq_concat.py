"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_ssm_id1_pq.py, ONE change: `bit_inner_downsample=4` instead of
1 (no downsample) — same working width as `attn_id4`/`attn_id4_pq`.

Session rationale: `qcute_refine_v4_2_k32_narrow_ssm_id1_pq` (full
`d_model=256` width, no downsample) was killed for being too slow —
session compute analysis showed `BitPredictHeadSSM` at full width costs
~272x more FLOPs per bit-chain than a plain independent linear head
(dominated by the decay-weighted cumsum's `O(dq^2*d_state)` term and the
per-position head's own cost), vs. only ~20x at `downsample=4` — session:
"for this better downsample or make embeds smaller dim, like 4x."

`BitPredictHeadSSM` itself also changed substantially this session,
BEFORE this config was created (all verified via fixed/loop-consistency
+ gradient checks, `torch.allclose atol=1e-5`):
- Per-position head (`self.head`: `nn.Linear(d_inner,1)` shared by every
  bit -> `nn.Linear(2*d_inner,dq)`, one row per bit position) — session:
  "let each bit timestep use different head... similar to independent
  mode, but has state."
- CONCAT, not add (`fetched = h_scale*h + state_contrib` ->
  `torch.cat([h_scale*h, state_contrib], dim=-1)`) — session: "make
  concat mode default... h_t always concat not add with current embed."
- Trainable BOS state (`self.bos_state`, zero-init `nn.Parameter`,
  stands in for `state_proj(zeros)` at position 0) — session: "consider
  a trainable bos token init zero at dq 0."
- Per-position head computed via `einsum("njd,jd->nj", ...)` instead of
  a full `[N,dq,dq]` matrix + `torch.diagonal` — session: "use einsum" —
  removes an `n`x compute/memory waste the earlier per-position-head
  version had (computing off-diagonal entries never used).
- The recurrence itself (alpha-decayed cumulative sum over past bits)
  is UNCHANGED — session: "retain cumsum with decay but each timestep
  must concat with original h_t."

`code_embed_mode="pq_table"` carried over, same fix that resolved
`attn_id4`'s own divergence.

Everything else identical to qcute_refine_v4_2_k32_narrow_ssm_id1_pq.py:
Ks=(32,32), attn_window=(32,32), dq=8, d_model=256, n_layers=1,
context_len=1024, fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="chain", bit_head_class="ssm", steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_ssm_id4_pq_concat.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_ssm_id4_pq_concat
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
code_head_mode = "chain"
bit_head_class = "ssm"
bit_inner_downsample = 4
code_embed_mode = "pq_table"

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
