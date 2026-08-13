# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                         # install/update env from pyproject.toml + uv.lock
uv run python scripts/prepare_data.py           # download/cut datasets/enwik8{,_1M}.gz
uv run python scripts/train_bpe.py --data datasets/enwik8_1M.gz   # BPE tokenizer for qcute.bpelm
uv run python -m qcute.bytelm --preset sd       # byte-level baseline LM (Phase 0), reports BPB
uv run python -m qcute.archive.qcutelm          # end-to-end tokenizer + latent LM (FSQ/BSQ) — ARCHIVED,
                                                 # superseded by qcute_refine (see Architecture below); still
                                                 # importable/runnable via its archive path for historical reference
uv run python -m qcute.bpelm --sp_model datasets/bpe_enwik8_1M_8192.model   # BPE baseline
uv run python -m qcute.bytelm --config configs/bytelm_xs_mtp4_ctx1024.py   # named, reproducible run — the
                                                 # standard byte-level baseline as of this session (context=1024,
                                                 # matching qcute_refine's own context_len); the older
                                                 # configs/bytelm_xs_mtp4.py (context=256) is superseded, kept only
                                                 # for historical reproducibility, not a comparison target anymore
uv run python scripts/plot_run.py logs/<run_name>   # train/val bpb PNG from a run's run.jsonl
uv run python -m qcute.qcute_v5_concat --config configs/qcute_v5_concat_overfit10k_k4single.py
uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_overfit10k_k4single.py
                                                 # the two active qcute prototypes (see Architecture
                                                 # below) — no version-suffix alias anymore; run
                                                 # each module directly
```

`qcute.bytelm`, `qcute.bpelm`, `qcute.qcute_v5_concat`, and
`qcute.qcute_v5_stack` all read `--help` for their full flag
list; all support `--config path.py` (see `configs/` — every config file
has its own module docstring explaining what it's testing and its exact
`uv run` invocation, copy-pasteable directly from the file), `--run_name`
(else derived from `--config`/preset — logs and checkpoints both key off
it: `logs/<run_name>/`, `checkpoints/<run_name>/`), and `--eval_only
--checkpoint_path ...`; `qcute.bytelm` additionally supports
`--qual_gen_bytes` for qualitative generation. Tiny-corpus-scale defaults
(`xs` preset) target ~4 bytes/timestep — see `qcute/bytelm.py`'s
`PRESETS` comment for why. No test suite, linter, or CI config exists yet.

**Only ever run one training job at a time.** All four modules train on
MPS; two concurrent training processes contend for the same GPU and both
slow down (observed directly: a second run caused an already-progressing
job to stall with zero throughput). Kill or wait out the current run
before launching another — never launch a second training process while
one is still active.

**When launching a training run in the background, never redirect its
stdout/stderr to `/dev/null`** — use a file instead (e.g. a scratchpad
path, or `/tmp/<pid>.log` renamed once the PID is known post-launch),
since `/dev/null` silently swallows uncaught-exception tracebacks and
anything not routed through `Logger`, making crashes invisible. Pipe
through `tr '\r' '\n'` before the redirect (`... 2>&1 | tr '\r' '\n' >
/tmp/foo.log &`) — tqdm's progress bar uses `\r` for in-place updates,
which lands as one giant unreadable line in a plain file otherwise; `tr`
turns each update into its own readable line (e.g. `loss=2.1512`). Note
`$!` after a pipe gives the last stage's PID (`tr`), not Python's — use
`pgrep -f "python3 -m qcute.<module>"` to find the actual training
process if you need its PID (e.g. to kill it). **After launching, give
the user two `tail -f` commands**: one on that raw stdout/stderr file,
and one on `logs/<run_name>/run.log` (the structured log `Logger` writes
to at `--log_every`/`--eval_every` intervals) — so they can watch it live
themselves rather than relying on being told the outcome later. Long runs
have shown unpredictable throughput (observed: a
nominal ~30-minute budget taking 2.5-3.5 hours instead) — watch actual
elapsed time/step rate early on rather than assuming a run will finish on
schedule.

## Architecture

`qcute/bytelm.py` and `qcute/bpelm.py` are self-contained baseline
modules — none import each other, deliberately not factored further yet.
Full details: [docs/architecture.md](docs/architecture.md) (also covers
`qcutelm.py`, now archived — see below).

**The entire `qcute_refine` lineage (`v1.py` through `v4_5_1.py`, plus
its `qcute_refine.py` always-latest alias) is now archived** under
`qcute/archive2/` (configs under `configs/archive2/`; its docs —
`qcute_refine_math.md`, `kv_contribution.md`, `bpe_like_boundaries.md`,
`bitpredict_heads.md`, `torch_compile.md` — under `docs/archive2/`).
**`qcute/qcute_v5_concat.py` and `qcute/qcute_v5_stack.py` (forked from
`qcute_refine_v4_4_1.py`/`qcute_refine_v4_5_1.py` respectively, dropping
the `refine_` prefix and the version-suffix/alias convention entirely)
are now the two standalone active prototypes** — each is its own
self-contained module, run directly (no promotion step, no alias file).
Configs: `configs/qcute_v5_concat_*.py`, `configs/qcute_v5_stack_*.py`.

Archived lineage summary, oldest to newest:
- `qcute_refine_v1.py`/`v2.py`/`v3.py`: pre-v4 history — BSQ code
  hand-off with a joint-chain-MTP detokenizer, then a cross-attending
  `DecoderLevel`, then EncoderLevel fusion.
- `qcute_refine_v4.py`: removes `DecoderLevel` entirely — it turned out
  to do the literal same job as fusion (predict a level's own next
  token, optionally conditioned on the coarser code) and never
  contributed to `byte_loss` even in v3. `EncoderLevel` renamed to
  `LevelLM` (the class now does both the "encode" job — PASS 1, produce
  the code — and the "decode" job — PASS 2, fused/conditioned
  prediction — so "Encoder" alone was misleading). Generation
  (`generate_no_cache`/`generate_kv_cache`) is fusion-aware — v3's own
  generation functions were copied unchanged from v2 and never touched
  cross-attention at all, a real train/inference mismatch v4 fixes.
  `qcute.bytelm`/`qcute.bpelm` also gained matching `generate_kv_cache`/
  `validate_generation` this session, same pattern.
- `qcute_refine_v4_4.py`/`qcute_refine_v4_5.py`: v4.4 adds
  packed-sequence multi-track cumulative decode (self + every coarser
  level's code, `decode_pack_mode`); v4.5 replaces that with explicit
  staged cross-attention through the same shared weights (no packed
  sequences). Both support `Config.share_level_weights` (default `True`,
  original behavior unchanged) — `False` gives every level (v4.4) or
  every level's own encode LM plus one independent LM per decode
  cross-attention track (v4.5) fully independent weights, coupled only
  through the bare integer code id crossing between them.
- `qcute_refine_v4_4_1.py`/`qcute_refine_v4_5_1.py`: fix the self
  track's decode conditioning to be genuine LM continuation (`code_b`
  conditions block `b+1`, never its own block `b` — the original
  v4.4/v4.5 mechanism was an accidental autoencoder for the self track
  specifically). Math:
  [docs/qcute_refine_v4_4_1_v4_5_1_math.md](docs/qcute_refine_v4_4_1_v4_5_1_math.md)
  (this doc itself is not archived — its "self-code LM continuation"
  mechanism carries over unchanged into `qcute_v5_concat.py`/
  `qcute_v5_stack.py`, see below).

`qcute_v5_concat.py` (from v4.4.1's packed self-attention decode) and
`qcute_v5_stack.py` (from v4.5.1's staged cross-attention decode) add
`Config.quant_type: "softmax"` (default, unchanged categorical code) or
`"bsq"` (binary spherical quantization, `Config.bsq_bits`-wide sign
code, straight-through) as an alternative to the categorical code
representation. Full detail on all of the above, including an ongoing
slow-convergence investigation (a moving-target/cascade effect for
n_levels=2 configs, compounded in the "stack"/v4.5-style decode by a
much deeper gradient path back to the code producer — decode is now
`n_layers * (1 + n_tracks)` deep sequentially, vs. the "concat"/v4.4-style
decode's flat `n_layers`), and the `decode_code_ste` findings:
[docs/status.md](docs/status.md) (reset partway through the v5 work —
full pre-reset history at
[docs/archive2/status.md](docs/archive2/status.md)).

**Standing methodology**: use a small (`n_bytes=10000`) slice of the
corpus with a short step budget as the standard fast-iteration testbed
for `qcute_v5_concat.py`/`qcute_v5_stack.py` architecture changes — see
`configs/*_overfit10k_*.py` — until a config can actually fast-overfit
that slice to a train bpb comparable to `qcute.bytelm`'s own parity
numbers on the same slice (`n_layers=1`: 0.0212 train bpb at step 1000,
~19.7 it/s; `n_layers=2`: 0.0072, ~11.4 it/s — see
`configs/bytelm_overfit10k_*.py`). Full-scale (~900k-byte) runs and
generation-quality comparisons are not trustworthy until that parity
bar is cleared — a config that hasn't even memorized a 10k-byte slice
yet tells you little about its behavior at scale.

Diagnostic: `scripts/probe_decoder_kv_contribution.py` (gradient/
ablation/attention-mass analysis of how much cross-attention KV actually
contributes vs. is ignored — run against a saved v2/v3 checkpoint, now
under `qcute.archive2.qcute_refine_v2`; v4-and-later have no
`DecoderLevel` for this script to target). Historical narrative on the
pre-v4 lineage (v2's `DecoderLevel` KV contribution, boundary-adaptivity
brainstorm, `BitPredictHead` speed comparisons, `torch.compile` on v2):
[docs/archive2/kv_contribution.md](docs/archive2/kv_contribution.md),
[docs/archive2/bpe_like_boundaries.md](docs/archive2/bpe_like_boundaries.md),
[docs/archive2/bitpredict_heads.md](docs/archive2/bitpredict_heads.md),
[docs/archive2/torch_compile.md](docs/archive2/torch_compile.md).

Several older one-off scripts (`scripts/compare_ste_vs_detach.py`,
`probe_code_usage_entropy.py`, `probe_v4_fusion_contribution.py`,
`qual_gen_v4_2.py`, `qual_gen_v4_3_vs_baseline.py`,
`test_v4_4_banded_decode.py`, `test_v4_4_chunked_decode.py`,
`qual_gen_bytelm.py`) still `import` archived `qcute_refine_vN` modules
by their pre-archive top-level path (e.g. `from qcute.qcute_refine_v4_4
import ...` instead of `from qcute.archive2.qcute_refine_v4_4 import
...`) and will `ImportError` if run as-is — not part of the two
maintained diagnostics above, left unfixed since nothing currently
depends on them; fix their import path the same way if you need to
revive one.

Every earlier qcute-lineage fork — `qcutelm.py`, `qcutelm_vlt*.py`
(`vlt` through `vlt11`), `qcutelm_pyramid.py`, `qcutelm_mergetoken_v1.py`,
`qcute_bytepool.py` — is **archived** under `qcute/archive/` (configs
under `configs/archive/`; their own design docs —
`continuous_tokenizer_handover.md`, `fifo_v2.md`, `vlt12_math.tex` — under
`docs/archive/`), superseded by the `qcute_refine` lineage. The
`qcute_refine` lineage itself (`v1.py` through `v4_5_1.py`, the pre-v5
history) is separately archived under `qcute/archive2/` — see above — a
different archive directory since it's a later, distinct lineage from
the `qcutelm.py` family. Kept for historical reference/reproducibility,
not part of active work. `qcute/bytelm.py` and `qcute/bpelm.py` are the
exception — still the active baseline comparison points, not archived.

Original design source-of-truth for the now-archived `qcutelm` lineage:
[docs/archive/continuous_tokenizer_handover.md](docs/archive/continuous_tokenizer_handover.md)
(historical — `qcute_refine`'s own design isn't specified by it).
Current progress: [docs/status.md](docs/status.md).

## Code style

Docstrings and comments must be extremely concise — assume code is self-descriptive; comments
never exceed 2 lines. Only the module-level (top-of-file) docstring is exempt from the length
limit, and should still be kept reasonably tight rather than accumulating restated history.

## Response format

Prefix every reply to the user with the current timestamp (run `date` for
the actual value — never guess it).
