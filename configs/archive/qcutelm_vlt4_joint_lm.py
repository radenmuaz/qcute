"""qcute.qcutelm_vlt4 --joint_lm: joint training with a real latent LM
(fork of qcutelm_vlt4_ctx128.py). From random init: NTP-on-raw-bytes loss
removed entirely; only recon_loss (tokenizer, shapes the code space) +
code_lm_loss (CodeLM's next-code prediction over the code sequence — the
actual generative objective this whole project is aiming at) are trained,
simultaneously, every step. See qcute/qcutelm_vlt4.py's CodeLM/
forward_joint_lm/code_to_index for the design: codes are indices into a
fixed vocabulary (2^dq for BSQ), so CodeLM is literally qcute.bytelm's
recipe applied to code-space. code_lm_detach=True (default): only
recon_loss shapes the tokenizer/code space, code_lm_loss only trains
CodeLM itself (stability — avoids the code space being a moving target
for CodeLM and vice versa).

Hierarchical context split (per conversation): the tokenizer's own
attention span (context_len) only needs to be small (16 bytes) since it
just needs to produce a faithful per-block code — the long effective
range (256 bytes, matching bytelm_xs_mtp4's context=256 for comparability)
comes from CodeLM stacking 64 codes (256/K), not from the tokenizer's own
window. Each of the 16 independent 16-byte windows is processed in one
batched, cheap forward pass; only CodeLM's causal attention spans the
full 64-code sequence.

    uv run python -m qcute.qcutelm_vlt4 --config configs/qcutelm_vlt4_joint_lm.py
"""
from pathlib import Path

K = 4
context_len = 16   # tokenizer's own LOCAL window — small on purpose, see lm_context_bytes below
attn_window = -1    # full causal within the (already small) 16-byte window — no need to band further
dq = 10  # -> CodeLM vocab_size = 2^10 = 1024 (dq=18's 262144-way softmax was impractically slow on MPS,
         # ~13-17s/step — see session notes; a factorized per-bit CodeLM loss would decouple this from dq
         # entirely, not yet implemented, this is the pragmatic fix for now)
d_model = 128
n_heads = 4
n_layers = 4
mlp_mult = 4
code_net_layers = 0

joint_lm = True
lm_context_bytes = 256  # 256/16=16 independent tokenizer windows -> 256/4=64 codes of CodeLM context
lm_d_model = 128
lm_n_heads = 4
lm_n_layers = 4
lm_mlp_mult = 4
code_lm_weight = 1.0
code_lm_detach = True

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 100000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = True
constant_steps = 100
eval_every = 100
eval_batches = 20

recon_target_acc = 0.95
