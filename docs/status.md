# qcute v5 status

Reset to a fresh log at this point — the prior narrative predates the modular rewrite below and is
now more archival than actionable. Full history: [docs/archive2/status.md](archive2/status.md)
(pre-modular v5, `qcute_v5_concat.py`/`qcute_v5_stack.py` lineage) and
[docs/archive3/status.md](archive3/status.md) (the immediately-prior log, spanning the qfb fix
through the pre-modular quant-scheme/entropy-reg/per-mode-decode work). New entries go below,
same convention: newest at the bottom, session-dated where useful.

## Where things stand

The standalone `qcute/qcute_v5_concat.py` and `qcute/qcute_v5_stack.py` (dense O(L^2) references)
and the efficient-attention `qcute/qcute_v5.py`/`qcute_v5_concat.py` pair described in the archived
logs have been superseded as the *default* by a modular rewrite:

- **`qcute/qcute_v5_common.py`**: shared `Config`, `LM` (renamed from `Backbone`) class, the three
  `QuantScheme` implementations, attention primitives, training loop, checkpointing, argparser.
- **`qcute/qcute_v5_encoder.py`**: `Encoder` class (PASS 1 — produce a level's code).
- **`qcute/qcute_v5_decoder.py`**: `Decoder` base class plus `ConcatDecoder`/`StackDecoder`
  subclasses (PASS 2 — fused/conditioned prediction, generation, KV-cache, all diagnostics).
- **`qcute/qcute_v5.py`**: `QCuteLM` (renamed from `RefineLM`), the entrypoint — `--decoder_type
  concat|stack` selects which `Decoder` subclass is built.

The old standalone `qcute_v5_concat.py`/`qcute_v5_stack.py` files (pre-modular) are archived under
`qcute/archive3/` — kept for historical reference, not part of active work.

### Quant schemes

Renamed for clearer, technique-neutral geometric naming (the old names conflated the categorical
case with "softmax" and the sign-code case with "BSQ" specifically, when BSQ's own default
projects onto a hypersphere, not the corner-selecting hypercube):

| new | old | geometry |
|---|---|---|
| `SimplexQuant` (`quant_type="simplex"`) | `SoftmaxQuant` (`"softmax"`) | categorical one-hot / softmax simplex |
| `BinaryQuant` (`quant_type="binary"`) | `BSQQuant` (`"bsq"`) | sign code, hypersphere (or hypercube if `binary_lfq=True`) |
| `GridQuant` (`quant_type="grid"`) | `FSQQuant` (`"fsq"`) | axis-aligned integer grid (finite scalar quantization) |

Config fields renamed to match: `binary_bits`/`binary_lfq` (was `bsq_bits`/`bsq_lfq`), `grid_dq`/
`grid_levels`/`grid_bound` (was `fsq_dq`/`fsq_levels`/`fsq_bound`). The underlying math functions
(`gumbel_quantize`, `bsq_quantize`, `fsq_quantize`, `bsq_entropy_reg`, `CodeEmbed`, `FSQEmbed`) keep
their old names deliberately — they name the technique, not the branding that changed.

### `code_sample_mode` replaced by two independent flags (2026-08-18)

The old single `code_sample_mode: "ste"|"sample"|"soft"` enum conflated two orthogonal choices:
whether the code is hardened (discrete + straight-through) and whether stochastic noise is injected
before quantizing. Split into `Config.code_hard: bool = True` / `Config.code_sample: bool = False`
(CLI: `--code_hard true|false`, `--code_sample` store_true):

| old mode | code_hard | code_sample |
|---|---|---|
| `"ste"` (default) | `True` | `False` |
| `"sample"` | `True` | `True` |
| `"soft"` | `False` | `True` |
| *(unreachable before)* | `False` | `False` |

The fourth combination — plain relaxed code, no noise at all — was structurally impossible under
the old enum (`"soft"` always went through the noisy branch) and is the whole point of the split:
it isolates "does hardening matter" from "does noise matter" as independent ablation axes.
`gumbel_quantize`/`bsq_quantize`/`fsq_quantize` all take `(hard, sample)` positionally now instead
of a mode string. Verified via smoke test across all four combinations (`--decoder_type stack`,
Ks=(1,)): `(True,False)`, `(True,True)` via `--code_sample`, `(False,False)`, and `(False,True)`
all train/generate; `check_gen_consistency` is 0/15 for every combo except `(False,True)`
(15/15 mismatched) — expected, since injecting the same Gumbel noise at generation time that was
present during training's teacher-forced forward pass is inherently non-reproducible under greedy
argmax generation, not a bug. `GridQuant.sample_next`'s multinomial-vs-argmax branch and
`BinaryQuant`/`SimplexQuant`'s `sample_next`/`quantize` calls all updated to the new signature.

### Per-level `d_model`/`n_layers`, required quant fields, word presets (modular files only)

Scoped to the new `qcute_v5*` modular files (not the old standalone `qcute_v5_concat.py`/
`qcute_v5_stack.py`, which keep a single global `d_model`/`n_layers`):

- `Config.d_model`/`n_layers` accept either a broadcast scalar (old behavior, unchanged) or an
  explicit per-level tuple, resolved via `resolve_per_level(value, n_levels)` — same pattern
  `attn_window` already used. **No cross-level projection layers were needed**: the code
  representation passed between hierarchy levels lives in *quant-width* space (`V`=`cfg.vocab` for
  simplex, `binary_bits`-wide sign vector, `grid_dq`-wide normalized vector) which is decoupled
  from `d_model` — each level's own `code_embed`/`code_head` bridges quant-width to that level's own
  `D`, so per-level dimension "just worked" once threaded through construction.
- `Config.quant_type`, `vocab`, and the active scheme's code-count param (`binary_bits`/`grid_dq`+
  `grid_levels`) have no default — `build_argparser` raises a clear `p.error(...)` if unset via
  CLI or config file, rather than silently falling back.
- `Config.input_preset`/`output_preset: int` (required, one of `WORD_PRESET_BITS = (1, 4, 8)` — bit/
  nibble/byte) generalize level0's raw alphabet size to `2**preset`, independent of `cfg.vocab`
  (the level1+ code width). Asymmetric input/output presets are conceptually allowed but currently
  raise `NotImplementedError` if unequal — not yet implemented. `unpack_words`/`pack_words` in
  `qcute_v5_common.py` unpack/repack a byte stream into/from `bits`-wide word symbols (`bits==8` is
  an identity no-op, preserving old byte-level behavior exactly).
- Real bug caught while wiring presets: `LM.__init__` was passing the per-level embed-table
  `vocab` (e.g. 16 for level0 under nibble preset) into `quant.init_modules` instead of
  `cfg.vocab`, incorrectly shrinking level0's own code-*production* width. Fixed by using
  `cfg.vocab` unconditionally for `init_modules`, decoupling "input embedding width" (per-level)
  from "code production width" (uniform). Surfaced a second latent bug: `SimplexQuant.embed_for_
  decode`/`embed_input` had been reusing `stage_lm.embed.weight` as the code-embedding table, which
  only worked when every level shared one global vocab — fixed by giving `SimplexQuant` its own
  dedicated `code_embed = nn.Linear(V, D, bias=False)`, matching how `BinaryQuant`/`GridQuant`
  already had dedicated `CodeEmbed`/`FSQEmbed` modules for the same reason.

### Other modular-file features

- `--full_val_eval` (whole-val-set periodic eval, `eval_model_full` with `batch_size=-1`) and
  `--eval_split train|val` for `--eval_only`, mirrored from `qcute.bytelm`.
- Checkpoints now save inside `logs/<run_name>/` instead of a separate `checkpoints/<run_name>/`
  tree (`Checkpointer(args.logs_dir / run_name, ...)`) — `qcute.bytelm` was NOT updated to match,
  still uses the separate `checkpoints/` tree.
- Qualitative-generation pretty-print alignment fix: the `qual_{train,val}_level0_modefull:` log
  line was one space narrower than its sibling `level0_mode{N}:`/`prompt:`/`ground_truth:` lines;
  fixed in `qcute_v5_decoder.py`'s shared logging path (`pad = " " * 5` for the `full` tag vs. `" "
  * 6` for numeric tags), applies to both train and val labels.

### `qcute_v5_stack_noreg` runs (2026-08-18)

No-regularization (`entropy_reg_weight=0.0`) baseline grid on the new modular `qcute_v5.py`
(`--decoder_type stack`), `quant_type="simplex"`, `d_model=256`, `n_layers=1`, `context_len=256`,
8000 steps, full val eval. Logs under `logs/qcute_v5_stack_noreg/<name>/`:

| config | Ks | code_hard/code_sample | best val_bpb |
|---|---|---|---|
| `ks1.py` | `(1,)` | `True`/`False` (ste-equivalent) | 2.7246 |
| `ks21.py` | `(2,1)` | `True`/`False` | 2.8105 |
| `ks221.py` | `(2,2,1)` | `True`/`False` | 2.8414 |
| `ks1_soft.py` | `(1,)` | `False`/`False` (new noise-free-soft combo) | 2.4597 (finished) |

(A stale, incomplete `ks21` attempt from an earlier interrupted session — killed mid-run around
step 4000, never reached `best_val_bpb`-worthy convergence — is kept at
`logs/qcute_v5_stack_noreg/ks21_incomplete_prev/` for reference, not comparable to the table above.)

`bytelm` byte-level baselines on the same enwik8_1M slice, `ctx256`, full val eval, 4000 steps:

| config | best val_bpb |
|---|---|
| `bytelm_xs1_ctx256_fullval` | 2.5846 |
| `bytelm_xs2_ctx256_fullval` | 2.4502 |
| `bytelm_xs4_ctx256_fullval` | 2.4235 |

`bytelm` beats all three `code_hard=True` v5-stack configs, and `ks1` (Ks=(1,), no hierarchy) beats
`ks21`/`ks221` — consistent with the "moving-target/cascade effect" for multi-level configs noted
in the archived logs, now reproduced under the modular rewrite too. Not yet root-caused.

**`ks1_soft` (code_hard=False, code_sample=False) is tracking well below every `code_hard=True`
config, including `bytelm`'s own baselines**, at the same step count it's not yet finished. Leading
hypothesis, **not yet verified — sanity-check before reading into it**: the gain is from removing
hard discretization (STE) entirely rather than from anything about "simplex" specifically —
`code_hard=False` means the code passed downstream is the full continuous softmax distribution, so
there's no straight-through gradient bias and no information loss at the quantization bottleneck
(effectively an ungated soft-attention-style handoff between levels, not a genuine discrete code).
That's a real capability/comparability confound: a `code_hard=False` run isn't testing the same
discrete-code hypothesis the rest of the grid is (no code-usage/entropy stats are meaningful the
same way `to_ids()` bins under `code_hard=True`), so a strong `ks1_soft` val_bpb number is not
evidence that the *discrete* Ks=(1,) config is competitive — it's evidence that discretization
itself carries a cost, which is expected and not new. Needs confirming: (a) let it finish and
compare fairly at matched steps, (b) check whether `ks21_soft`/`ks221_soft` also relax to `bytelm`-
beating numbers (if the hierarchy penalty disappears too under `code_hard=False`, that further
localizes the cascade problem specifically to hard-code straight-through gradients, not to the
hierarchy/cross-attention structure itself).

`qcute_v5_stack_noreg/ks1_overfit1k.py` (n_bytes=1000, val_frac=0.5, 1000 steps) is a fast-
iteration smoke config, not a real training run — best val_bpb 5.3602, not comparable to the table
above.

### GMM/GMMDiag quantizer, bigger FSQ grids, full leaderboard (2026-08-19)

Added two new `QuantScheme`s to `qcute_v5_common.py`: **`GMMQuant`**/**`GMMDiagQuant`**
(`quant_type="gmm"`/`"gmm_diag"`, new required `Config.gmm_k`/`gmm_dq` fields) — a genuine
**shared codebook** (`GMMCodebook`/`GMMDiagCodebook`, `mu`/covariance are `nn.Parameter`s, not
per-token predictions), selected via posterior responsibility (`softmax_k[log π_k + log
N(query;μ_k,Σ_k)]`) rather than plain `argmin` — a direct generalization of VQ-VAE's nearest-
neighbor rule (reduces to it when every `Σ_k` is isotropic and `π_k` uniform). No VQ-style
commitment/codebook losses needed (the mixture NLL is already differentiable in `μ_k`/`Σ_k`
through the `logsumexp`, unlike VQ-VAE's non-differentiable `argmin`) — component collapse is
still a risk, mitigated the same way as BSQ/softmax via `entropy_reg` reusing
`softmax_entropy_reg` on the posterior logits, zero new plumbing.

`GMMQuant` (full covariance) parameterizes each component's covariance via its **precision**
Cholesky factor `A` (`Λ=AAᵀ`) instead of the covariance Cholesky — makes the NLL's Mahalanobis
term a plain matmul (`y=Aᵀ(x-μ)`, no solve/inverse) at the cost of needing one small **manual**
triangular solve (`solve_upper_triangular`, plain elementwise back-substitution, no
`torch.linalg.*` call) only on the reparameterized-sampling path — deliberate, since
`torch.linalg.solve_triangular`/`cholesky` have historically poor/no MPS support (this project
trains on `device="mps"`) and would risk a silent CPU fallback every token every step if used
directly. `GMMDiagQuant` needs no solve anywhere (elementwise both directions), `O(dq)` vs full
covariance's `O(dq²)` per component.

**Real perf bug found and fixed**: `_GMMQuantBase._select`'s `code_hard=False` branch originally
looped over all `K` components calling `codebook.sample(k)` individually — each call redundantly
recomputed the full `precision_chol()` (over all `K` components) internally, making the whole
branch `O(K²)` instead of `O(K)`. At `K=128` this was ~4x slower per step than it should've been
(confirmed via a `code_hard=False/code_sample=True` smoke test: 48s/15 steps before the fix on
CPU). Fixed by adding `GMMCodebook.sample_all`/`GMMDiagCodebook.sample_all` (one
`precision_chol()` call, one batched solve broadcasting over `K`, no Python loop) — same
`K=128` config: 21.5s/15 steps after.

**`gmm_k=8192` sizing was rejected before training**: unlike FSQ/BSQ, whose head width is
decoupled from their combinatorial code count (`dq·L`/`bits`, small regardless of total codes),
GMM's gating head has to emit one logit per component, so head width scales `∝K` directly —
`K·(1+dq+dq(dq+1)/2)` for full covariance. At `K=8192,dq=4` that's a `256→122880` linear, ~63M
params for the two heads alone, dwarfing the rest of a `d_model=256` model. Settled on **`K=256,
dq=4`** instead — same order of magnitude as `SimplexQuant`'s 256-way vocab, a fair comparison
point, head width only 3840.

**`v5_stack_gmm_ks1_256` (full covariance) was too slow to finish and was stopped**: ~4.9 min per
100 steps on MPS (799/8000 steps in ~65 min, no eval reached, so no `best_val_bpb`) — not yet
root-caused (possibly `qualitative_generate`'s autoregressive loop repeatedly rebuilding
`precision_chol()` per step; not yet profiled). `v5_stack_gmm_ks1_256_diag` (diagonal covariance,
same `K`/`dq`) is running as of this entry; expect it to be substantially faster given the `O(dq)`
vs `O(dq²)` gap already measured above.

**New diagnostic, not wired into training**: `_GMMQuantBase.bpb_bound(stage_lm, h_query,
target_repr, precision_bits)` — `ntp_loss_acc`'s `K`-way cross-entropy against `to_ids()` is only
an *exact* bpb when `code_hard=True` (the emitted code is genuinely just "which of `K`
components", fully determined by the id). Under `code_hard=False`/`code_sample=True` the actual
emitted code is a continuous vector carrying strictly more information than that id, so the
existing loss **undercounts** — not a valid bound in either direction. `bpb_bound` fixes this by
scoring the true mixture density's NLL of the exact continuous `target_repr` (not its
discretized id) and adding a stated per-dim quantization-precision correction (`Config.
gmm_bpb_precision_bits`, default 8 bits/dim — same role as the `+8` constant RealNVP/Glow-style
bits/dim reporting adds for pixel dequantization; differential entropy alone isn't bits without
one). Gives a genuine, achievable upper bound. Verified via isolated test (`K=16,dq=4,
hard=False/sample=True`): naive undercounted CE ~3.9/~3.4 bits (GMM/GMMDiag) vs `bpb_bound`
~40.8/~40.4 bits (dominated by the `dq·precision_bits=32` bit correction) — confirms the fix
charges honestly instead of silently underreporting.

### Recurring MPS full-val-eval glitch, chunked `eval_model_full` fix (2026-08-19)

Found across three separate runs (`gmm_256_diag`, `fsq_ks1_4x8`, `fsq_ks1_16x16`): decode-side
metrics (`byte_loss`, `decode_total`, `level0_ntp_loss_decode`) occasionally come back as a
literal `0.0` during a live `--full_val_eval` round on `device=mps` -- not NaN, not a display
artifact, confirmed via `run.jsonl`. Ruled out as a logic bug: reloading the exact same checkpoint
+ data and replaying the identical `eval_model_full` computation on CPU always gives correct, sane
numbers (e.g. `fsq_ks1_16x16`'s live-logged `best_val_bpb=5.9001` was actually **2.4915** on clean
CPU replay of its final checkpoint). Ruled out concurrent-job contention too -- reproduced with
zero other processes running. Most likely culprit: `eval_model_full`'s old default packed the
*entire* val set into one giant single-shot forward pass (`batch_size=-1` -> `batch_size=
n_windows`, e.g. ~390 windows at once) -- bigger than anything training itself ever does in one
call, plausibly triggering an MPS-backend numerical/synchronization issue under sustained memory
pressure.

**Fix**: `eval_model_full` now takes `sample: float | int = 1.0` (fraction or absolute window
count of the val set to evaluate, default the whole set -- unchanged coverage) and `batch_size`
is repurposed as a **chunk size** for internal batching within that selection, no longer "how much
of the val set." New CLI: `--eval_chunk_size` (default `64`, bounds each forward pass) and
`--eval_sample` (default `1.0`). `--eval_chunk_size -1` restores the old single-giant-batch
behavior for explicit opt-back-in. Also fixed `Checkpointer.is_better` to reject non-finite/
non-positive metrics outright (`qcute_v5_common.py:70-72`), so a future occurrence of this glitch
can no longer permanently freeze `best_val_bpb`/`best.pt` the way it did before this session found
it (confirmed: a `0.0` eval now correctly loses to any later real value instead of winning forever).
Verified via smoke test across default/`-1`/`--eval_sample 0.5` -- all produce sane nonzero
metrics, no crashes.

**Practical fallout**: any `best_val_bpb` logged by a run that predates this fix should be treated
as a floor, not a ground truth, if a `val_byte_loss=0.0000` line appears anywhere in its
`run.jsonl` -- the true number needs a clean CPU replay of the final checkpoint (see
`v5_stack_fsq_ks1_16x16`'s corrected entry in the leaderboard below). `v5_stack_gmm_ks1_256`
(stopped early, step 799/8000, no eval reached) and `v5_stack_gmm_ks1_256_diag` (stopped at step
7399/8000, and its logged `best_val_bpb=0.0000` is exactly this glitch, pre-fix) never produced a
trustworthy number -- need a clean rerun with the fixed `eval_model_full` before either GMM variant
gets a real leaderboard entry.

### Full leaderboard, params, FLOPs (2026-08-19)

All `Ks=(1,)`-scale, enwik8_1M, `ctx256`/`context_len=256`, sorted best to worst. `bytelm`'s
`4000`-step schedule vs. the v5-stack configs' `8000`-step schedule aren't perfectly step-matched,
kept as-is since both are each config's own converged/near-converged number. FLOPs/token is the
standard `2×params` forward-pass approximation (Kaplan et al. convention -- ignores attention's
quadratic term, negligible at `context_len=256` for these small models); FLOPs/ctx multiplies by
the full `256`-token context.

| rank | config | mechanism | best val_bpb | params | FLOPs/token | FLOPs/ctx |
|---|---|---|---|---|---|---|
| 1 | `bytelm_xs4_ctx256_fullval` | byte-level baseline (no hierarchy/quantizer at all) | 2.4235 | 3.400M | 6.80M | 1740.8M |
| 2 | `bytelm_xs2_ctx256_fullval` | byte-level baseline | 2.4502 | 1.800M | 3.60M | 921.6M |
| 3 | `qcute_v5_stack_noreg/ks1_soft` | simplex, `code_hard=False/code_sample=False` | 2.4597 | 2.103M | 4.21M | 1076.7M |
| 4 | `v5_stack_fsq_ks1_16x16` | grid/FSQ, dq=16/levels=16 | 2.4915† | 1.989M | 3.98M | 1018.4M |
| 5 | `v5_stack_fsq_ks1_16x4` | grid/FSQ, dq=16/levels=4 | 2.5229 | 1.792M | 3.58M | 917.5M |
| 6 | `v5_stack_fsq_ks1_8x8` | grid/FSQ, dq=8/levels=8 | 2.5523 | 1.784M | 3.57M | 913.4M |
| 7 | `bytelm_xs1_ctx256_fullval` | byte-level baseline | 2.5846 | 1.100M | 2.20M | 563.2M |
| 8 | `v5_stack_fsq_ks1` | grid/FSQ, dq=4/levels=8 | 2.6114 | 1.747M | 3.49M | 894.5M |
| 9 | `v5_stack_fsq_ks1_4x8` | grid/FSQ, dq=8/levels=4 | 2.6651 | 1.751M | 3.50M | 896.5M |
| 10 | `qcute_v5_stack_noreg/ks1` | simplex, `code_hard=True` | 2.7246 | 1.972M | 3.94M | 1009.7M |
| 11 | `qcute_v5_stack_noreg/ks21` | simplex, Ks=(2,1) | 2.8105 | 5.257M | 10.51M | 2691.6M |
| 12 | `qcute_v5_stack_noreg/ks221` | simplex, Ks=(2,2,1) | 2.8414 | 9.463M | 18.93M | 4845.1M |
| — | `v5_stack_gmm_ks1_256` | gmm full-cov, K=256/dq=4 | not trustworthy, see above | 1.988M | 3.98M | 1018.5M |
| — | `v5_stack_gmm_ks1_256_diag` | gmm diag, K=256/dq=4 | not trustworthy, see above | 1.982M | 3.96M | 1014.6M |

† `v5_stack_fsq_ks1_16x16`'s live-logged number was corrupted by the MPS glitch above; this is the
corrected value from a clean CPU replay of its final (step 8000) checkpoint.

Patterns holding: every non-hardened/continuous config (`ks1_soft`, `bytelm`'s own no-quantizer
baseline) clusters at the top, `bytelm_xs4` still wins outright; among genuinely discrete
`code_hard=True` schemes, FSQ/grid beats simplex at every grid size tried and the biggest grid
(`16x16`) is now the best discrete-code config overall; going deeper in the hierarchy (`ks21`,
`ks221`) consistently hurts, and costs more params/FLOPs for it. GMM still has no trustworthy
number -- both runs need a clean rerun with the fixed `eval_model_full` before ranking.
