# qcute status

Pruned 2026-08-27 — the prior narrative (GLAT-dose experiments/repetition-collapse
investigation, the `qcute_zero.py` parallel-decode-strategy pruning pass, and the full
`qcute_zero_simple`/`summ_transformer` fork story through 2026-08-25) is now archival, once this
file passed 700 lines with `summ_transformer` established as a third active lineage alongside
`qcute_lagcodec`/`qcute_zero`. Full prior history: [docs/archive6/status.md](archive6/status.md) (the
log this prune supersedes, verbatim), older still [docs/archive5/status.md](archive5/status.md),
[docs/archive4/status.md](archive4/status.md), [docs/archive3/status.md](archive3/status.md),
[docs/archive2/status.md](archive2/status.md). For current architecture (not results), see
`CLAUDE.md`'s Architecture section; for the bpb-validity/codec-vs-predictive-LM formal writeup,
see [docs/maths.md](maths.md). This file tracks results/progress going forward, same convention
as before: newest at the bottom, session-dated.

## 2026-08-25/27: `summ_transformer` full-scale 1M runs — weight sharing does not fix overfitting

Two matched full-scale (`enwik8_1M`, full 1M bytes, `context_len=256`, `attn_window=None`,
`d_model=256`/`n_layers=4`, `mtp_heads=4`, 8000 steps) `Ks=(2,1)` runs compared: `ks21_1M`
(defaults: `share_lm=False`, `share_fuse=False`, `weight_tie=False`) vs `ks21_1M_shareall`
(`share_lm=True`, `share_fuse=True`, `weight_tie=True`). Both show near-identical overfitting:
best val at step 1499 (unshared: val_loss=4.225/byte_acc=0.566/bpb~2.32; shared:
val_loss=4.260/byte_acc=0.559/bpb=2.334), degrading steadily afterward to nearly the same final
point (unshared: val_loss=7.769/byte_acc=0.507 at step 8000; shared: val_loss=7.488/byte_acc=0.513/
bpb=4.326), despite train byte_acc reaching ~0.95 in both cases. **Verdict: tying weights across
levels/fuse-stages does not meaningfully change this architecture's overfitting behavior** at this
scale/step-budget — the bottleneck is elsewhere (likely the step budget itself relative to a
900k-byte corpus, or a more fundamental capacity/regularization issue), not weight redundancy.
Both runs' `check_kv_cache_consistency` passed (`match_rate=1.0`).

`Config.bpb` metric (`loss / math.log(2)`) added to `summ_transformer.py`'s per-position metrics
dict this session — previously only raw nats loss was logged; `val_bpb` now appears directly in
`run.log` without manual conversion.

`ks221_1M` (3-level, unshared) and `ks221_1M_shareall` (3-level, all-shared) queued/in-progress as
of this writing to check whether the same pattern holds with an extra fuse stage.

## 2026-08-26: weight sharing + per-level MTP heads ported to `qcute_zero_simple`/`qcute_zero`

`summ_transformer.py` gained a `share_fuse` config flag (mirrors the existing `share_lm`: when
set, every `FuseStage` aliases to `fuse_stages[0]` instead of getting independent weights) —
`weight_tie`/`share_lm` were already present and already defaulted to untied/unshared as expected,
no change needed there.

`qcute/qcute_zero/backups/qcute_zero_simple.py` (an older single-shared-`self.blocks` prototype,
predating the active `qcute_zero.py`'s per-level `self.lms` design) was brought up to the same
convention: `self.blocks`/`self.ln_f` (always one shared stack) replaced with `self.lms`/`self.ln_fs`
(one stack per level, `share_lm`-gated), a real untied-by-default `self.head` added (`weight_tie`
flag to tie it to `embed.weight`), and `self.fuse_stages` made `share_fuse`-gated. Every readout
call site retargeted from `self.embed.weight` to `self.head.weight` (re-embedding of sampled codes
still correctly uses `self.embed.weight`). `--seed` (default 1234) + `torch.manual_seed` added to
`main()`. Verified: `ast.parse`, forward/backward for both all-shared/tied and all-separate/untied
configs, `generate_no_cache`==`generate_kv_cache` exact match, `check_kv_cache_consistency`
match_rate=1.0 — all post-refactor.

Also extended `qcute_zero_simple.py` so every level has its own MTP heads, not just the final cond
readout (previously only `extra_heads` existed, reading the post-cascade hidden state): added
`extra_heads_uncond` (`mtp_heads_uncond`/`mtp_weight_uncond`, off `h` pre-fusion) and
`extra_heads_code_per_stage` (`mtp_heads_code`/`mtp_weight_code`, per fuse-stage, `share_lm`-gated,
off each stage's own `h_code`) — exactly mirroring what the active `qcute_zero.py` already had.

A feature-disparity check between `qcute_zero.py` (active) and `qcute_zero_simple.py` (backup)
confirmed the real architectural difference is NOT weight-sharing (now at parity) but the
discrete-code bottleneck: active `qcute_zero.py` has a pluggable `Quantizer` class with
configurable `vocab`/`pq_chunks`/`quant_type` (code space can differ from and exceed one byte's
worth of bits) plus a `WordHead`/`head_word_bits` output-granularity decoupling and `global_tie`;
`qcute_zero_simple.py` hardcodes code space = byte vocab (256) via a bare `gumbel_quantize` call
reusing `head.weight`/`embed.weight` directly, one symbol per code, no PQ. `share_fuse` was then
back-ported to the active `qcute_zero.py` for parity (it previously only had `share_lm`).
`generate_speculative` was already present in both files, no port needed. Remaining asymmetry:
`generate_free_rollout` exists only in `qcute_zero_simple.py` (pruned from the active file during
the 2026-08-25 parallel-decode-strategy cleanup, see archive6).

`qcute.bytelm` gained `--seed` (default 1234) + `torch.manual_seed`, matching `qcute_zero`/
`qcute_lagcodec`/`summ_transformer` — previously the only one of the four training modules with no seed
control at all.
