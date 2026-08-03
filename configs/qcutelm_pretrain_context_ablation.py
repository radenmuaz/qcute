"""qcute.qcutelm config: --pretrain_ae-only ablation of Config.context_len —
does giving ChunkEncoder an extra 2 bytes of left context (the previous
chunk's tail, still causal) help it reconstruct better/faster than the
plain K-byte-only encoder? No LM, no --freeze_after_pretrain: this config
only exercises pretrain_autoencoder()'s plain reconstruction loop, the
same methodology as docs/status.md's tokenizer-depth sanity check (watch
train_recon_acc directly; a real ceiling shows up as a plateau below
100%, not just slower convergence).

Baseline (this file's defaults): context_len=0 — the original K-byte-only
encoder, no behavior change from before this feature existed.

    uv run python scripts/prepare_data.py    # once, if datasets/enwik8_1M.gz missing
    uv run python -m qcute.qcutelm --config configs/qcutelm_pretrain_context_ablation.py

Comparison run — same everything, +2 bytes of left context, separate
run_name so it doesn't append into the baseline's log:

    uv run python -m qcute.qcutelm --config configs/qcutelm_pretrain_context_ablation.py \\
        --context_len 2 --run_name qcutelm_pretrain_context_ablation_ctx2

    # plot either after training:
    uv run python scripts/plot_run.py logs/qcutelm_pretrain_context_ablation
    uv run python scripts/plot_run.py logs/qcutelm_pretrain_context_ablation_ctx2
"""
from pathlib import Path

bottleneck = "bsq"
K = 4
tokenizer_layers = 2
context_len = 0  # baseline; pass --context_len 2 for the comparison run (see docstring)
data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

pretrain_ae = True
pretrain_target_acc = 0.95
pretrain_steps = 40000
pretrain_lr = 6e-4  # standardized to match the rest of this session's qcutelm configs
warmup_steps = 500
cosine_decay = False  # pretrain_cosine_decay not set either — plain warmup->constant, matching the rest of the corpus
batch_size = 16
seq_chunks = 256

log_every = 100
eval_every = 200
eval_batches = 20
