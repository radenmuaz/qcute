import torch
import torch.nn as nn

from qcute.qcute_v5_common import LM, Config, apply_rope, make_dict, rope_cos_sin


class Encoder(nn.Module):
    def __init__(self, cfg: Config, d_model: int, n_layers: int, vocab: int):
        super().__init__()
        self.cfg = cfg
        self.lm = LM(cfg, d_model, n_layers, vocab)

    @property
    def embed(self):
        return self.lm.embed

    @property
    def quant(self):
        return self.lm.quant

    def forward(self, seq_repr: torch.Tensor, level: int, window: int | None, compute_ntp: bool = True) -> dict:
        cfg = self.cfg
        bb = self.lm
        K = cfg.Ks[level]
        D = bb.d_model
        is_byte_level = level == 0
        x = bb.embed_input(seq_repr, is_byte_level)
        L = seq_repr.shape[1] if is_byte_level else seq_repr.shape[1]
        x0 = x
        head_dim = D // cfg.n_heads
        cos, sin = rope_cos_sin(L, head_dim, cfg.rope_base, x.device)
        for block in bb.blocks:
            x = block(x, cos, sin, window)
        h = bb.ln_f(x)

        if compute_ntp:
            h_flat = h[:, :-1, :].reshape(-1, D)
            ntp_loss, ntp_acc = bb.ntp_loss_acc(h_flat, seq_repr[:, 1:], is_byte_level)
        else:
            ntp_loss = h.new_zeros(())
            ntp_acc = h.new_zeros(())

        extracted = bb.extract_code(h, x0, K, window)
        return make_dict(code=extracted["code"], ntp_loss=ntp_loss, ntp_acc=ntp_acc, hidden=h,
                          entropy_reg=extracted["entropy_reg"])
