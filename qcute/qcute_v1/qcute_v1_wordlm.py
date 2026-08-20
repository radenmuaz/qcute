"""qcute.qcute_v1.qcute_v1_wordlm — plain unconditional next-word LM baseline (the bytelm.py job),
ported to run through qcute_v1's own shared training infra instead of a standalone module, so
its logs/checkpoints/run.jsonl are directly comparable to every other qcute_v1 config with no
format adapter. Reuses Encoder/LM as-is (RoPE, Block, embed, ntp_loss_acc) via
Encoder.forward(..., compute_code=False), which skips the quantize/extract_code pass entirely
(not just discards its output) -- no decode stage, no bottleneck, no wasted quantize FLOPs, i.e.
"encode pass, unconditional". quant_type is irrelevant to correctness here (never invoked);
configs should set quant_type="simplex" since it needs no extra required fields.

uv run python -m qcute.qcute_v1.qcute_v1_wordlm --config configs/v1_word/xs.py
"""
from types import SimpleNamespace

import torch.nn as nn

from qcute.qcute_v1.qcute_v1_common import Config, resolve_per_level, run_main
from qcute.qcute_v1.qcute_v1_encoder import Encoder


class WordLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.input_preset != cfg.output_preset:
            raise NotImplementedError(
                f"WordLM requires input_preset == output_preset (got {cfg.input_preset} != "
                f"{cfg.output_preset}) -- asymmetric bitwidths not yet supported")
        self.cfg = cfg
        self.n_levels = 1
        self.seq_lens = [cfg.context_len]
        d_model = resolve_per_level(cfg.d_model, 1)[0]
        n_layers = resolve_per_level(cfg.n_layers, 1)[0]
        vocab = 2 ** cfg.input_preset
        self.encoder = Encoder(cfg, d_model, n_layers, vocab)
        # plain (non-nn.ModuleList) attributes -- avoids double-registering self.encoder's params
        # under two state_dict keys. Satisfies train()'s unconditional set_quant_dropout_p call
        # (encoders: at least the level0 quant; decoder.stage_lms: no decode stage here, empty).
        self.encoders = [self.encoder]
        self.decoder = SimpleNamespace(stage_lms=[])

    def forward(self, word_ids):
        out = self.encoder(word_ids, level=0, window=None, compute_ntp=True, compute_code=False)
        loss = out["ntp_loss"]
        metrics = {"loss": loss, "byte_loss": loss, "byte_loss_full": loss, "byte_acc": out["ntp_acc"]}
        return loss, metrics


def main():
    run_main(WordLM)


if __name__ == "__main__":
    main()
