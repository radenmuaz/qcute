import torch
import torch.nn as nn
import torch.nn.functional as F

from qcute.v5_old.qcute_v5_common import Config, WORD_PRESET_BITS, make_dict, resolve_per_level, run_main
from qcute.v5_old.qcute_v5_decoder import make_decoder
from qcute.v5_old.qcute_v5_encoder import Encoder


class QCuteLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
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
            windows += [None if ew == -1 else ew]
            if isinstance(dw, (tuple, list)):
                assert len(dw) == n_sources
                decode_windows += [[None if x == -1 else x for x in dw]]
            else:
                decode_windows += [[None if dw == -1 else dw] * n_sources]
        self.windows = windows
        self.decode_windows = decode_windows

        for i, dwlist in enumerate(decode_windows):
            cum_K, per_track, invisible_srcs = 1, [], []
            for src_offset, dwindow in enumerate(dwlist):
                cum_K *= cfg.Ks[i + src_offset]
                if dwindow is None:
                    per_track += [f"K={cum_K}:full"]
                else:
                    n_codes = dwindow // cum_K
                    per_track += [f"K={cum_K}:{n_codes}codes"]
                    if dwindow != 0 and n_codes == 0:
                        invisible_srcs += [cum_K]
            print(f"decode effective codes level{i}: " + ", ".join(per_track))
            if invisible_srcs:
                print(f"WARNING: level{i} decode_window too small for cumulative K in {invisible_srcs}")
        for i, (L, window) in enumerate(zip(seq_lens, windows)):
            if window is not None:
                assert L % window == 0 or L <= window

        self.encoders = nn.ModuleList([Encoder(cfg, self.d_models[i], self.n_layers_list[i], self.vocabs[i])
                                        for i in range(self.n_levels)])
        self.decoder = make_decoder(cfg, self.n_levels, self.encoders, self.d_models, self.n_layers_list, self.vocabs)

    def _run(self, byte_ids: torch.Tensor, compute_ntp: bool = True, max_decode_sources: int | None = None,
             want_next_query: bool = False) -> dict:
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

        decode_losses: list = [None] * self.n_levels
        decode_accs: list = [None] * self.n_levels
        decode_derived_c: dict = {}
        h_out = list(h_list)
        next_query: list = [None] * self.n_levels
        embed_weights: list = [None] * self.n_levels
        decode_stage_extra_losses: list = []

        for i in reversed(range(self.n_levels)):
            result = self.decoder.decode_level(self, i, x_list, c_list, decode_derived_c,
                                                compute_ntp, max_decode_sources, want_next_query)
            if result is None:
                continue
            decode_losses[i] = result["loss"]
            decode_accs[i] = result["acc"]
            h_out[i] = result["hidden"]
            embed_weights[i] = result["embed_weight"]
            decode_stage_extra_losses += result["extra_losses"]
            if i == 0:
                next_query[i] = result["query_last"]
            if max_decode_sources is None:
                decode_derived_c[i] = result["code"]

        return make_dict(encode_losses=encode_losses, encode_accs=encode_accs, decode_losses=decode_losses,
                          decode_accs=decode_accs, h_list=h_out, c_list=c_list, next_query=next_query,
                          decode_derived_c=decode_derived_c, h0_encode=h_list[0],
                          decode_stage_extra_losses=decode_stage_extra_losses,
                          encode_entropy_regs=encode_entropy_regs, embed_weights=embed_weights)

    def forward(self, byte_ids: torch.Tensor) -> tuple:
        cfg = self.cfg
        result = self._run(byte_ids)
        encode_losses, encode_accs = result["encode_losses"], result["encode_accs"]
        decode_losses, decode_accs = result["decode_losses"], result["decode_accs"]
        h0_encode = result["h0_encode"]
        encode_entropy_regs = result["encode_entropy_regs"]
        decode_stage_extra_losses = result["decode_stage_extra_losses"]

        byte_loss = decode_losses[0] if decode_losses[0] is not None else encode_losses[0]
        byte_acc = decode_accs[0] if decode_accs[0] is not None else encode_accs[0]

        K0 = cfg.Ks[0]
        if decode_losses[0] is not None and K0 > 1:
            D0 = h0_encode.shape[-1]
            h0_partial = h0_encode[:, :K0 - 1, :].reshape(-1, D0)
            tgt0_partial = byte_ids[:, 1:K0].reshape(-1)
            logits0 = F.linear(h0_partial, self.encoders[0].embed.weight)
            enc_partial_loss = F.cross_entropy(logits0, tgt0_partial)
            n_enc, n_dec = K0 - 1, byte_ids.shape[1] - K0
            byte_loss_full = (enc_partial_loss * n_enc + decode_losses[0] * n_dec) / (n_enc + n_dec)
        else:
            byte_loss_full = byte_loss

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

        loss = encode_total + decode_total + decode_stage_extra_total + cfg.entropy_reg_weight * entropy_reg_total
        ntp_total = torch.stack(encode_losses + [l for l in decode_losses if l is not None]
                                 + decode_stage_extra_losses).sum()
        metrics = {
            "loss": loss, "byte_loss": byte_loss, "byte_loss_full": byte_loss_full, "byte_acc": byte_acc,
            "encode_total": encode_total, "decode_total": decode_total,
            "decode_stage_extra_total": decode_stage_extra_total, "ntp_loss_total": ntp_total,
            "entropy_reg_total": entropy_reg_total,
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
