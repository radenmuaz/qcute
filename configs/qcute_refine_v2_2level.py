"""qcute.qcute_refine_v2 config: 2-level tower (level1=byte, level2=code),
cross-attention TokenizerLevel decoder (see qcute/qcute_refine_v2.py's
module docstring). TokenizerLevel_0: Q=EncoderLevel_0's own hidden states
(bytes), KV=EncoderLevel_1's own hidden states (code c_0), decodes bytes —
exactly the concrete example worked through this session. tok_head_mode=
"linear" (default): single plain nn.Linear decode head, no joint-chain
MTP (explicitly disabled this session — "disable mtp... use single linear
head, mtp as flag").

Optimizer/LR/steps: identical to bytelm_xs_mtp4_ctx1024 (same convention
as configs/qcute_refine_v1.py) — steps=8000, batch_size=16, lr_peak=6e-4,
warmup_steps=500, cosine_decay=False.

Wall-clock check (this session, MPS, fresh init): 0.625 s/step (1.60
it/s) -> ~83 min for 8000 steps. Faster than qcute_refine_v1's 1.157
s/step shape — the cross-attention decoder reuses EncoderLevel's own
already-computed hidden states (no dedicated self-attention trunk of its
own) and its default linear head avoids BitPredictHead's chain-mode cost
entirely (session diagnosis: ~200-1800x a plain Linear per call).

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_2level.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_2level
"""
from pathlib import Path

Ks = (2, 2)
dqs = (8, 8)
tier_d_models = (96, 96)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = 128

tok_d_model = 96
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20
