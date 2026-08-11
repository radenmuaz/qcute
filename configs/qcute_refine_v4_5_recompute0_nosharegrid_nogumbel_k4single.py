"""No-gumbel-noise variant of qcute_refine_v4_5_nosharegrid_k4single.py -- see that file's
docstring for full rationale. use_gumbel_noise=False, gumbel_tau back to default 1.0,
decode_code_ste stays False (detach, unchanged).

    uv run python -m qcute.qcute_refine_v4_5 --config configs/qcute_refine_v4_5_recompute0_nosharegrid_nogumbel_k4single.py

    # watch live:
    tail -f logs/qcute_refine_v4_5_recompute0_nosharegrid_nogumbel_k4single/run.log


decode_stage0_recompute=True variant: Stage 0 uses the SAME weights as encode_lms[i] (no independent LM, unlike decode_separate_stage0) but runs a FRESH forward pass instead of reusing encode's h tensor -- isolates whether the coupling problem is about WEIGHT sharing (predicted: behaves identically to the plain-reuse baseline) or about the graph-node reuse itself (would behave differently). See qcute_refine_v4_5.py's Config.decode_stage0_recompute docstring."""
from pathlib import Path

Ks = (4,)
d_model = 256
n_layers = 2
context_len = 256
attn_window = ((8, 256),)
cross_track_source = "decode"
decode_code_ste = False
share_level_weights = False
decode_stage0_recompute = True
use_gumbel_noise = False
gumbel_tau = 1.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64
qual_prompt_bytes = 64
