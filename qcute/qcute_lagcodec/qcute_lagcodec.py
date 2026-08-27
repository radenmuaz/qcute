import torch
import torch.nn as nn
import torch.nn.functional as F

from qcute.qcute_lagcodec.qcute_lagcodec_common import Config, ROPE_PRESETS, WORD_PRESET_BITS, make_dict, resolve_per_level, run_main
from qcute.qcute_lagcodec.qcute_lagcodec_decoder import make_decoder
from qcute.qcute_lagcodec.qcute_lagcodec_encoder import Encoder


class QCuteLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.rope_preset is not None:
            cfg.rope_base = ROPE_PRESETS[cfg.rope_preset]
        self.cfg = cfg
        self.n_levels = len(cfg.Ks)
        self.d_models = resolve_per_level(cfg.d_model, self.n_levels)
        self.n_layers_list = resolve_per_level(cfg.n_layers, self.n_levels)
        for i, D in enumerate(self.d_models):
            assert D % cfg.n_heads == 0, f"level{i} d_model ({D}) must be divisible by n_heads ({cfg.n_heads})"

        assert cfg.input_preset in WORD_PRESET_BITS, f"input_preset must be one of {WORD_PRESET_BITS}, got {cfg.input_preset}"
        assert cfg.output_preset in WORD_PRESET_BITS, f"output_preset must be one of {WORD_PRESET_BITS}, got {cfg.output_preset}"
        if cfg.input_preset != cfg.output_preset:
            raise NotImplementedError(
                f"asymmetric input_preset ({cfg.input_preset}) != output_preset ({cfg.output_preset}) "
                f"is not yet implemented -- both must currently match")
        level0_vocab = 2 ** cfg.input_preset
        self.vocabs = [level0_vocab] + [cfg.vocab] * (self.n_levels - 1)
        if cfg.code_head_tied and level0_vocab != cfg.vocab:
            raise NotImplementedError(
                f"code_head_tied=True ties level0's own code production to its (word-alphabet-sized, "
                f"{level0_vocab}) embed table -- but level0's code must be cfg.vocab-sized ({cfg.vocab}) "
                f"to feed level1. Not supported when input_preset's alphabet differs from cfg.vocab.")

        seq_lens = [cfg.context_len]
        for k in cfg.Ks[:-1]:
            assert seq_lens[-1] % k == 0
            seq_lens += [seq_lens[-1] // k]
        assert seq_lens[-1] % cfg.Ks[-1] == 0
        self.seq_lens = seq_lens

        def _norm_w(x):
            return None if x == -1 else x

        def _norm_track0(x0):
            # track0's window is a SHARED knob by default (same value gates both its own byte/
            # code self-attention AND its cross-attention into level (i+1)'s code, see
            # StackDecoder's docstring) -- but a (self_window, cross_window) 2-tuple here lets a
            # config decouple them (2026-08-23, chat: "impl so that give nested tuple, each level
            # decoder can fine grained how much window from level 0 to last"). Threaded through as
            # a 2-tuple end to end (decode_windows[i][0], _track0/encode_like_self_attn_decode's
            # self_window/own_code_window params, _stack_generate_blockwise's matching call) so
            # training and generation windowing never diverge -- see chat 2026-08-23's earlier
            # StackDecoderLocal generation-mismatch bug for why that divergence matters.
            if isinstance(x0, (tuple, list)):
                assert len(x0) == 2, f"track0 window must be a scalar or (self_window, cross_window), got {x0!r}"
                return (_norm_w(x0[0]), _norm_w(x0[1]))
            return _norm_w(x0)

        raw_windows = cfg.attn_window if isinstance(cfg.attn_window, (tuple, list)) else (cfg.attn_window,) * self.n_levels
        assert len(raw_windows) == self.n_levels
        windows: list = []
        decode_windows: list = []
        for i, w in enumerate(raw_windows):
            n_sources = self.n_levels - i
            if isinstance(w, (tuple, list)):
                ew, dw = w
            else:
                ew = dw = w
            windows += [_norm_w(ew)]
            if isinstance(dw, (tuple, list)):
                assert len(dw) == n_sources
                decode_windows += [[_norm_track0(dw[0])] + [_norm_w(x) for x in dw[1:]]]
            else:
                decode_windows += [[_norm_track0(dw)] + [_norm_w(dw)] * (n_sources - 1)]
        self.windows = windows
        self.decode_windows = decode_windows

        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window

        self.encoders = nn.ModuleList([Encoder(cfg, self.d_models[i], self.n_layers_list[i], self.vocabs[i])
                                        for i in range(self.n_levels)])
        self.decoder = make_decoder(cfg, self.n_levels, self.encoders, self.d_models, self.n_layers_list, self.vocabs)

        # Printed AFTER decoder construction (2026-08-23 fix): decode_windows' own n_sources
        # (n_levels-i) is a nominal upper bound, computed before StackDecoder-family decoders'
        # own hard-exclusion of the topmost level's code from conditioning -- printing straight
        # from decode_windows silently overstated real track counts (e.g. ks21 showed 2 tracks,
        # own+level1, when level0 actually has zero upper tracks; ks221 showed 3, own+level1+
        # level2, when level0 actually has 2). Truncate to len(self.decoder.stage_lms[i]) --
        # own-code stage + real upper tracks -- when the decoder exposes that per-level structure
        # (StackDecoder family); other decoder types (e.g. concat) keep the nominal count as-is.
        for i, dwlist in enumerate(decode_windows):
            stage_lms = getattr(self.decoder, "stage_lms", None)
            if stage_lms is not None and i < len(stage_lms) and hasattr(stage_lms[i], "__len__"):
                n_real = len(stage_lms[i])  # own-code stage + real upper tracks
                dwlist = dwlist[:n_real]
            cum_K, per_track, invisible_srcs = 1, [], []
            for src_offset, dwindow in enumerate(dwlist):
                cum_K *= cfg.Ks[i + src_offset]
                self_w = None
                is_track0 = src_offset == 0
                if is_track0 and isinstance(dwindow, tuple):
                    self_w, dwindow = dwindow  # track0: (self_window, cross_window)
                if dwindow is None:
                    tag = "full"
                else:
                    # track0's cross_window is already a code-count (compared against block_lag in
                    # encode_like_self_attn_decode) -- unlike every other track, which is measured
                    # in raw byte-position units (window // cum_K), see split_track0_window.
                    n_codes = dwindow if is_track0 else dwindow // cum_K
                    tag = f"{n_codes}codes"
                    if dwindow != 0 and n_codes == 0:
                        invisible_srcs += [cum_K]
                if self_w is not None:
                    tag += f" (self_window={self_w})"
                per_track += [f"K={cum_K}:{tag}"]
            print(f"decode effective codes level{i}: " + ", ".join(per_track))
            if invisible_srcs:
                print(f"WARNING: level{i} decode_window too small for cumulative K in {invisible_srcs}")

        # Uncertainty weighting (Kendall/Gal/Cipolla 2018): one learnable log-variance per NTP task,
        # replacing byte_ntp_weight/code_ntp_weight/decode_ntp_weight's fixed scalars when enabled.
        # Layout: [0:n_levels)=each level's own encode loss, [n_levels:2*n_levels)=each level's own
        # decode loss, [2*n_levels]=the bundled decode_stage_extra loss (count varies with cond_depth,
        # so it's weighted as one aggregate task rather than per-source).
        if cfg.uncertainty_weighting:
            self.uncertainty_log_vars = nn.Parameter(torch.zeros(2 * self.n_levels + 1))

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True,
             max_srcs: int | None | tuple = None, want_next_query: bool = False) -> dict:
        """max_srcs: scalar (broadcast to every level, legacy behavior) or a per-level tuple/list
        (max_srcs[i] used for level i's decode_level call) -- lets a curriculum drop level i's
        conditioning on a coarser level asymmetrically, e.g. (2, 1, None) on a Ks=(2,2,1) model
        makes level0 keep its level1 track (drop level2) while level1 drops ITS only upper track
        (level2) too, unlike a scalar max_srcs=2 which can't do both at once (see chat 2026-08-21:
        level1 always has exactly one upper track in a 3-level model, so a global cap of 2 never
        removes it)."""
        cfg = self.cfg
        seq_repr = byte_ids
        encode_losses, encode_accs, h_list, c_list, x_list, encode_entropy_regs = [], [], [], [], [], []

        for i in range(self.n_levels):
            want_ntp = compute_ntp and (i == 0 or cfg.code_ntp_weight > 0)
            out = self.encoders[i](seq_repr, level=i, window=self.windows[i], compute_ntp=want_ntp)
            encode_losses += [out["ntp_loss"]]
            encode_accs += [out["ntp_acc"]]
            h_list += [out["hidden"]]
            c_list += [out["code"]]
            x_list += [seq_repr]
            encode_entropy_regs += [out["entropy_reg"]]
            seq_repr = out["code"]

        # encoder_ste_p (STE training of the level-above's own NTP head via decode's reconstruction
        # loss as feedback -- not "consistency" in a reconstruction-comparison sense, see
        # docs/status.md's 2026-08-23 rename note): with probability cfg.encoder_ste_p (one coin
        # flip per forward pass), every non-top level's code is resampled from the level-above's OWN
        # NTP prediction (sample_next(), STE unless detach_ss_sample) instead of the ground-truth
        # code. cfg.encoder_ste_skip_real (default False) controls what happens to the real-code
        # pass when this fires:
        #   False (default, additive): the real-code decode pass below ALWAYS runs unconditionally;
        #     when encoder_ste_p also fires, a SECOND, separate decode pass with the self-sampled
        #     code runs too, its loss added UNWEIGHTED on top (encoder_ste_total) -- decode's own
        #     gradient path is unaffected either way, only the code-producer sees this signal as
        #     something new. Empirically more stable (2026-08-23 comparison) since the "official"
        #     decode_losses/byte_acc never see anything but the real code.
        #   True (skip, formerly cfg.scheduled_sampling_p): the self-sampled code REPLACES the
        #     real-code pass entirely for this step -- one pass, either real or sampled, mutually
        #     exclusive, same substitution behavior the old scheduled_sampling_p had (unified into
        #     this one mechanism 2026-08-23 rather than keeping two separate flags). More faithful
        #     to the true (100% self-sampled) generation-time conditioning distribution, but
        #     empirically less stable -- the dominant training signal itself becomes noisy whenever
        #     the upper encoder's own forecast is still bad.
        run_ste = (torch.is_grad_enabled() and cfg.encoder_ste_p > 0 and self.n_levels > 1
                   and torch.rand(()).item() < cfg.encoder_ste_p)
        c_list_ste = None
        if run_ste:
            c_list_ste = list(c_list)
            for i in range(self.n_levels - 1):
                h_upper = h_list[i + 1]
                if h_upper.shape[1] < 2:
                    continue
                enc_upper = self.encoders[i + 1]
                predicted = enc_upper.quant.sample_next(enc_upper.lm, h_upper[:, :-1, :], cfg.vocab)
                c_list_ste[i] = torch.cat([c_list[i][:, :1, :], predicted], dim=1)
        skip_real = run_ste and cfg.encoder_ste_skip_real
        main_c_list = c_list_ste if skip_real else c_list

        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        decode_derived_c: dict = {}
        h_out = list(h_list)
        next_query: list = [None] * self.n_levels
        embed_weights: list = [None] * self.n_levels
        decode_stage_extra_losses: list = []

        decode_levels = range(1) if cfg.decode_scope == "level0_only" else range(self.n_levels)
        for i in reversed(decode_levels):
            max_srcs_i = max_srcs[i] if isinstance(max_srcs, (list, tuple)) else max_srcs
            result = self.decoder.decode_level(self, i, x_list, main_c_list, decode_derived_c,
                                                compute_ntp, max_srcs_i, want_next_query)
            if result is None:
                continue
            decode_losses[i] = result["loss"]
            decode_accs[i] = result["acc"]
            h_out[i] = result["hidden"]
            embed_weights[i] = result["embed_weight"]
            decode_stage_extra_losses += result["extra_losses"]
            if i == 0:
                next_query[i] = result["query_last"]
            if max_srcs_i is None and result["code"] is not None:
                decode_derived_c[i] = result["code"]

        encoder_ste_losses: list = []
        if run_ste and not skip_real:
            decode_derived_c_ste: dict = {}
            for i in reversed(decode_levels):
                max_srcs_i = max_srcs[i] if isinstance(max_srcs, (list, tuple)) else max_srcs
                result = self.decoder.decode_level(self, i, x_list, c_list_ste,
                                                    decode_derived_c_ste, compute_ntp,
                                                    max_srcs_i, False)
                if result is not None:
                    encoder_ste_losses.append(result["loss"])
                    if max_srcs_i is None and result["code"] is not None:
                        decode_derived_c_ste[i] = result["code"]

        return make_dict(encode_losses=encode_losses, encode_accs=encode_accs, decode_losses=decode_losses,
                          decode_accs=decode_accs, h_list=h_out, c_list=c_list, next_query=next_query,
                          decode_derived_c=decode_derived_c, h0_encode=h_list[0],
                          decode_stage_extra_losses=decode_stage_extra_losses,
                          encoder_ste_losses=encoder_ste_losses,
                          encode_entropy_regs=encode_entropy_regs, embed_weights=embed_weights)

    def forward(self, byte_ids: torch.Tensor, max_srcs: int | None | tuple = None,
                _skip_byte_consistency: bool = False) -> tuple:
        cfg = self.cfg
        result = self._run(byte_ids, max_srcs=max_srcs)
        encode_losses, encode_accs = result["encode_losses"], result["encode_accs"]
        decode_losses, decode_accs = result["decode_losses"], result["decode_accs"]
        h0_encode = result["h0_encode"]
        encode_entropy_regs = result["encode_entropy_regs"]
        decode_stage_extra_losses = result["decode_stage_extra_losses"]
        encoder_ste_losses = result["encoder_ste_losses"]

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        # qcute_lagcodec: the K0-1-partial encoder/decoder blend below only applies when level0 IS the
        # top level (n_levels==1, still genuine K-1-shifted NTP) -- for n_levels>=2, level0 is a
        # non-top autoencoder (see StackDecoderV1) whose decode loss already covers every position
        # 0..n_blocks*K0-1 directly, no encoder-fallback prefix needed.
        K0 = cfg.Ks[0]
        if decode_losses[0] is not None and K0 > 1 and self.n_levels == 1:
            D0 = h0_encode.shape[-1]
            h0_partial = h0_encode[:, :K0 - 1, :].reshape(-1, D0)
            tgt0_partial = byte_ids[:, 1:K0].reshape(-1)
            logits0 = F.linear(h0_partial, self.encoders[0].byte_output_weight)
            enc_partial_loss = F.cross_entropy(logits0, tgt0_partial)
            n_enc, n_dec = K0 - 1, byte_ids.shape[1] - K0
            byte_loss_full = (enc_partial_loss * n_enc + decode_losses[0] * n_dec) / (n_enc + n_dec)
        else:
            byte_loss_full = byte_loss

        uncertainty_sigmas = {}
        if cfg.uncertainty_weighting:
            lv = self.uncertainty_log_vars

            def uw_term(loss_val, idx):
                return torch.exp(-lv[idx]) * loss_val + lv[idx]

            encode_total = torch.stack([uw_term(l, i) for i, l in enumerate(encode_losses)]).sum()
            decode_terms = [uw_term(l, self.n_levels + i) for i, l in enumerate(decode_losses) if l is not None]
            decode_total = torch.stack(decode_terms).sum() if decode_terms else byte_loss.new_zeros(())
            decode_stage_extra_total = (uw_term(torch.stack(decode_stage_extra_losses).sum(), 2 * self.n_levels)
                                         if decode_stage_extra_losses else byte_loss.new_zeros(()))
            with torch.no_grad():
                sigma = torch.exp(0.5 * lv)
            uncertainty_sigmas = {
                **{f"uncertainty_sigma_encode{i}": sigma[i] for i in range(self.n_levels)},
                **{f"uncertainty_sigma_decode{i}": sigma[self.n_levels + i] for i in range(self.n_levels)
                   if decode_losses[i] is not None},
                "uncertainty_sigma_stage_extra": sigma[2 * self.n_levels],
            }
        else:
            encode_code_total = (torch.stack(encode_losses[1:]).sum() if self.n_levels > 1
                                  else byte_loss.new_zeros(()))
            encode_total = cfg.byte_ntp_weight * encode_losses[0] + cfg.code_ntp_weight * encode_code_total

            decode_ntp_weight = (cfg.decode_ntp_weight if isinstance(cfg.decode_ntp_weight, (tuple, list))
                                  else (cfg.decode_ntp_weight,) * self.n_levels)
            decode_terms = [decode_ntp_weight[i] * l for i, l in enumerate(decode_losses) if l is not None]
            decode_total = torch.stack(decode_terms).sum() if decode_terms else byte_loss.new_zeros(())

            decode_stage_extra_weight = sum(decode_ntp_weight) / len(decode_ntp_weight)
            decode_stage_extra_total = (decode_stage_extra_weight * torch.stack(decode_stage_extra_losses).sum()
                                         if decode_stage_extra_losses else byte_loss.new_zeros(()))

        entropy_reg_terms = [r for r in encode_entropy_regs if r is not None]
        entropy_reg_total = (torch.stack(entropy_reg_terms).sum() if entropy_reg_terms
                              else byte_loss.new_zeros(()))

        encoder_ste_total = (torch.stack(encoder_ste_losses).sum() if encoder_ste_losses
                              else byte_loss.new_zeros(()))

        # byte_consistency_p (2026-08-23, "true" consistency training in byte space, distinct from
        # encoder_ste_p's code-level swap): with probability cfg.byte_consistency_p (its own coin
        # flip, independent of encoder_ste_p), argmax level0's own byte-level reconstruction logits,
        # detach (no gradient into whatever produced them), and run the WHOLE model again on this
        # self-predicted byte sequence -- self-supervised as always (the second pass reconstructs
        # ITS OWN input, exactly like the first pass does), so this tests whole-model idempotence/
        # stability under one round of self-feeding, not just decode's code-level robustness.
        # _skip_byte_consistency=True guards the recursive call so this can only ever fire once per
        # top-level forward() call, never cascading.
        byte_consistency_total = byte_loss.new_zeros(())
        if (not _skip_byte_consistency and torch.is_grad_enabled() and cfg.byte_consistency_p > 0
                and torch.rand(()).item() < cfg.byte_consistency_p):
            embed0 = (result["embed_weights"][0] if result["embed_weights"][0] is not None
                      else self.encoders[0].byte_output_weight)
            logits0 = F.linear(result["h_list"][0], embed0)
            predicted_bytes = logits0.argmax(-1).detach()
            byte_consistency_total, _ = self.forward(predicted_bytes, max_srcs=max_srcs,
                                                       _skip_byte_consistency=True)

        loss = (encode_total + decode_total + decode_stage_extra_total
                + cfg.entropy_reg_weight * entropy_reg_total + encoder_ste_total
                + byte_consistency_total)
        ntp_total = torch.stack(encode_losses + [l for l in decode_losses if l is not None]
                                 + decode_stage_extra_losses).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_loss_full": byte_loss_full, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total,
            "decode_stage_extra_total": decode_stage_extra_total, "ntp_loss_total": ntp_total,
            "entropy_reg_total": entropy_reg_total, "encoder_ste_total": encoder_ste_total,
            "byte_consistency_total": byte_consistency_total,
            **uncertainty_sigmas,
            **{f"level{i}_ntp_loss_encode": l for i, l in enumerate(encode_losses)},
            **{f"level{i}_ntp_acc_encode": a for i, a in enumerate(encode_accs)},
            **{f"level{i}_ntp_loss_decode": l for i, l in enumerate(decode_losses) if l is not None},
            **{f"level{i}_ntp_acc_decode": a for i, a in enumerate(decode_accs) if a is not None},
            **{f"level{i}_entropy_reg": r for i, r in enumerate(encode_entropy_regs) if r is not None},
        }
        return loss, metrics


def main():
    run_main(QCuteLM)


if __name__ == "__main__":
    main()
