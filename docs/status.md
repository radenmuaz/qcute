# `qcute_refine` status

Reset to a fresh log at this point in the project — the prior session-by-session narrative (v1
through the v4.4.1/v4.5.1 LM-continuation fix, the `decode_code_ste` dead-gradient investigation,
and the `v5_concat`/`v5_stack` BSQ addition) got long enough to be more archival than actionable.
Full history: [docs/archive2/status.md](archive2/status.md) (3700+ lines, newest at the bottom).
New entries go below, same convention: newest at the bottom, session-dated where useful.

## Where things stand

Active: `qcute/qcute_v5_concat.py` (forked from `qcute_refine_v4_4_1.py`'s packed self-attention
decode) and `qcute/qcute_v5_stack.py` (forked from `qcute_refine_v4_5_1.py`'s staged
cross-attention decode) — each adds `Config.quant_type: "softmax" | "bsq"`. Both are now
standalone modules (dropped the `refine_` prefix and the version-suffix/alias convention).
Configs: `configs/qcute_v5_concat_*.py`, `configs/qcute_v5_stack_*.py`.

The entire `qcute_refine` lineage (`v1.py` through `v4_5_1.py`, plus its `qcute_refine.py`
always-latest alias) and its configs/docs (`kv_contribution.md`, `bpe_like_boundaries.md`,
`qcute_refine_math.md`, `bitpredict_heads.md`, `torch_compile.md`) are now archived under
`qcute/archive2/`, `configs/archive2/`, `docs/archive2/` — see CLAUDE.md.

Standing conclusions carried forward from the archived log (still load-bearing, don't re-derive):
- **`decode_code_ste=False` (detach) stays the default.** `=True` (STE) fixes a real dead-gradient
  issue (`code_head` gets zero gradient for `n_levels=1` under detach) but empirically causes
  cond-generation to collapse into character repetition at the tiny overfit10k scale tested,
  independent of gumbel noise on/off. Revisit at larger scale.
- **Overfit10k is the standard fast-iteration testbed**: `n_bytes=10000`, `steps=1000`,
  `batch_size=16`, `lr_peak=6e-4`, `warmup_steps=100` — see `configs/*_overfit10k_*.py`. Compare
  against `qcute.bytelm`'s own parity numbers on the same slice before trusting full-scale runs.
  `qcute_refine`'s own generation has **no KV-cache path** in any version — `generate_no_cache`
  only.
- **Same-prompt generation comparison methodology**: byte offset `START=5850` into
  `datasets/enwik8_1M.gz`'s first 9000 bytes (train split), `PROMPT_LEN=64`, `GEN_LEN=64`, greedy
  argmax, used throughout for cond (`generate_no_cache`) vs. uncond (`generate_encode_only`)
  comparisons across configs/checkpoints.
- v4.4.1-style packed self-attention decode (self track) clears exact verbatim reconstruction on
  a meaningful fraction of overfit10k configs; v4.5.1-style staged cross-attention decode does
  not — the difference is architectural (staged cross-attention's own exposure bias), not the
  LM-continuation conditioning fix itself (both got it).
