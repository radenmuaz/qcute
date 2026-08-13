"""qcute.qcute_refine_v3 config: CLONE of configs/qcute_refine_rope.py
(the v2 config, best val_bpb 2.6310 @ step 3600 — see docs/status.md),
architecture parameters UNCHANGED, running under qcute.qcute_refine_v3
instead of qcute.qcute_refine_v2 — i.e. `fuse_encoder_levels=True`
(v3's own default) is the ONLY real difference, since every other field
here is identical to qcute_refine_rope.py.

Session ask: put EncoderLevel fusion (see qcute/qcute_refine_v3.py's
module docstring for the full mechanism/rationale) on top of the SAME
architecture that produced the best-understood v2 reference point, so any
val_bpb difference is attributable to fusion itself, not a confound with
some other config change at the same time. This directly targets the
finding that motivated v3: v2's `val_bpb` (byte_loss) has zero access to
the coarser code or cross-attention at all — only `tok_loss` (a separate,
detached path) ever benefited from it. Fusion makes byte_loss itself
depend on, and learn to use, EncoderLevel[1]'s own hidden state before
computing level 0's own NTP loss.

    uv run python -m qcute.qcute_refine_v3 --config configs/qcute_refine_v3_rope.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v3_rope

    # A/B control (same file, fusion off, should reproduce v2's own
    # qcute_refine_rope.py result if the two files are truly equivalent
    # modulo fusion):
    uv run python -m qcute.qcute_refine_v3 --config configs/qcute_refine_v3_rope.py \\
        --run_name qcute_refine_v3_rope_nofuse --fuse_encoder_levels false
"""
from pathlib import Path

Ks = (4, 4)
dqs = (8, 8)
tier_d_models = (256, 256)
tier_n_layers = (1, 1)
context_len = 1024

n_heads = 4
mlp_mult = 4
attn_window = (256, 64)

byte_repr = "embed"
code_head_mode = "independent"
cross_attn_rope = True
fuse_encoder_levels = True   # v3's own default, pinned here for clarity/reproducibility

tok_d_model = 256
tok_n_heads = 4
tok_mlp_mult = 4
tok_head_mode = "linear"
tok_weight = 1.0

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
