"""qcute.bytelm config: CLONE of configs/bytelm_xs1_ctx1024.py (1-layer,
d_model=256, xs preset), ONE change: `context=8` instead of `context=1024`.

Session rationale: sanity check for `qcute_refine_v4_bpe4_imitate`'s own
result (docs/status.md) — that config crippled level 0's receptive field
to `attn_window=8` (a near-bag-of-8-bytes local view) and turned out to
be the WORST result of the whole session relative to a matched baseline,
losing even to plain `bytelm_xs1_ctx1024` (which sees the full 1024-byte
dense context). Open question that comparison never isolated: how much
of `bpe4_imitate`'s loss was really about the narrow 8-byte window itself
(any model, however simple, struggling with that little context) versus
something specific to `qcute_refine`'s own hierarchical/fusion structure?
This config answers the first half directly — `bytelm.py` has no window
concept at all (`docs/status.md`: "bytelm_xs1_ctx1024's CausalSelfAttention
has no window concept... always dense, full 1024-byte reach"), so the
only way to cripple ITS receptive field the same way is to shrink
`context` itself: with `context=8`, every training example is only 8
bytes long, so no prediction ever has more than 7 bytes of causal
history — the same effective local view `bpe4_imitate`'s level 0 had.

If this plain, architecture-free 1-layer dense model ALSO craters to
something in `bpe4_imitate`'s neighborhood (2.5073) or worse, that's
strong evidence the narrow window alone (not qcute_refine's hierarchy)
explains most of the damage. If it stays much better, that would instead
implicate qcute_refine's own structure (crippled level 0 + relying on
fusion) as the real cause.

Everything else identical to bytelm_xs1_ctx1024.py: preset="xs"
(d_model=256, n_heads=4, mtp_heads=4), n_layers=1, steps=4000.

    uv run python -m qcute.bytelm --config configs/bytelm_xs1_ctx8.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bytelm_xs1_ctx8
"""
from pathlib import Path

preset = "xs"
context = 8
n_layers = 1
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1
steps = 4000
batch_size = 16
warmup_steps = 500
cosine_decay = False
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
