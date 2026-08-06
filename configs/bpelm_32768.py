"""qcute.bpelm config: vocab bumped 8192 -> 32768 specifically to close
the byte-equivalent-context gap against the other baselines' context_len=
1024 bytes — bpelm_8192.py's context=256 tokens only spans ~845 bytes
(vocab=8192 measured ~3.3 bytes/token on the real bpelm run; the
train_bpe.py-reported sample estimate was 3.9). Tried 16384 first (3.53
bytes/token measured via train_bpe.py's own roundtrip-check sample -> 256
tokens ~= 904 bytes, still short); 32768 reached 3.80 bytes/token -> 256
tokens ~= 973 bytes, the closest power-of-2 vocab option to 1024 without
going further into the corpus-size phrase-memorization risk train_bpe.py's
own docstring flags for a corpus this small (enwik8_1M.gz, ~900,000
training bytes after val split). context left at 256 (unchanged) — vocab
was the tunable axis here, not context (no power-of-2 context lands nearer
1024 bytes than 256 already does at this vocab's bytes/token rate).

    uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz --vocab_size 32768   # already run
    uv run python -m qcute.bpelm --config configs/bpelm_32768.py

    # plot after training:
    uv run python scripts/plot_run.py logs/bpelm_32768

The config's filename stem (bpelm_32768) becomes the run_name by default.
"""
from pathlib import Path

sp_model = Path("datasets/bpe_enwik8_1M_32768.model")
data = Path("datasets/enwik8_1M.gz")
context = 256
d_model = 256
n_layers = 4
n_heads = 4
val_frac = 0.1
steps = 8000
batch_size = 16
warmup_steps = 500
lr_peak = 6e-4
log_every = 50
eval_every = 100
eval_batches = 20
