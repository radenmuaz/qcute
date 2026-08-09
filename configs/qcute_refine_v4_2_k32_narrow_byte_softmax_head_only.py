"""qcute.qcute_refine_v4_2 config: CLONE of configs/qcute_refine_v4_2_
k32_narrow_byte256.py, but a NARROWER ablation: `byte_softmax_head_only=
True` instead of `byte_head_256way=True`.

Session rationale: `byte256` (`byte_head_256way=True`) unshares THREE
things for level 0 at once — its INPUT embedding (own `nn.Embedding
(vocab, D)` instead of the shared dq-bit `CodeEmbed`), its OUTPUT head
(own `nn.Linear(D, vocab)` instead of the shared dq-bit head), and its
`code_pre` (own `nn.Linear(D, dq)` instead of the shared one feeding
level 1). It resolved `val_bpb` back to the healthy range (2.5660) but
left `val_level1_bpb_pass1` (the code level's own loss) just as unstable
as the fully-shared baseline (see docs/kv_contribution.md §11) — meaning
unsharing all three together fixed the SYMPTOM (fused bpb) without
telling us which of the three actually mattered, or whether it even
needed to be all three. Session ask ("no byte embedding, assume byte as
bits 0 to bits 255, only head is softmax"): keep level 0's INPUT
embedding and `code_pre` SHARED with the code-level pool (byte value
represented via the same dq-bit `byte_to_dqbits` path every code level
uses) — only the OUTPUT readout becomes an unshared 256-way softmax head
(`Config.byte_softmax_head_only=True`, new this session). Isolates
whether it's specifically the shared OUTPUT head (byte's 256-way target
vs. code's ~8-bit target forced through the same `nn.Linear`/
`BitPredictHead`) that mattered, independent of whether the input
embedding was also shared.

Everything else identical to qcute_refine_v4_2_k32_narrow_byte256.py:
Ks=(32,32), dq=8, d_model=256, n_layers=1, context_len=1024,
attn_window=(32,32), fuse_encoder_levels=True, fuse_use_null_kv=False,
code_head_mode="independent" (only affects the code levels' own shared
head now), steps=4000.

    uv run python -m qcute.qcute_refine_v4_2 --config configs/qcute_refine_v4_2_k32_narrow_byte_softmax_head_only.py

    # plot after training:
    uv run python scripts/plot_run.py logs/qcute_refine_v4_2_k32_narrow_byte_softmax_head_only
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
byte_softmax_head_only = True

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
