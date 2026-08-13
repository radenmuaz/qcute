"""qcute.qcute_refine_v2 config: 3-level tower, same recursive structure
as configs/qcute_refine_v2_2level.py extended one level further
(TokenizerLevel_0: Q=bytes, KV=code_0, decodes bytes; TokenizerLevel_1:
Q=code_0, KV=code_1, decodes code_0) — "similar recursive structure for 3
level config." Otherwise identical hyperparameters to the 2-level config
(same Ks=2 per level, same widths, same tok_head_mode="linear",
same optimizer/LR/step budget matched to bytelm_xs_mtp4_ctx1024).

Wall-clock check (this session, MPS, fresh init): 0.747 s/step (1.34
it/s) -> ~100 min for 8000 steps.

    uv run python -m qcute.qcute_refine_v2 --config configs/qcute_refine_v2_3level.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v2_3level
"""
from pathlib import Path

Ks = (2, 2, 2)
dqs = (8, 8, 8)
tier_d_models = (96, 96, 96)
tier_n_layers = (1, 1, 1)
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
