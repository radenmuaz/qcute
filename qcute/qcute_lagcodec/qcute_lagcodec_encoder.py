import torch
import torch.nn as nn

from qcute.qcute_lagcodec.qcute_lagcodec_common import LM, Config, apply_rope, make_dict, rope_cos_sin


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

    @property
    def byte_output_weight(self):
        return self.lm.byte_output_weight

    def forward(self, seq_repr: torch.Tensor, level: int, window: int | None, compute_ntp: bool = True,
                compute_code: bool = True) -> dict:
        """NOTE for generation (chat 2026-08-20, to stop this being re-derived/re-confused every
        session): THIS is where "the next token/code at any level" genuinely comes from. `hidden`
        below is a real, uncircular NTP hidden state over `seq_repr` -- level0's own bytes, or level
        j's own INPUT (= level (j-1)'s code stream) for j>0 -- `ntp_loss` trains `h[:,p,:]` to
        predict `seq_repr[:,p+1]` directly, no autoencoder/own-code circularity anywhere in this
        path (unlike decode_level's non-top branches, which reconstruct a block from its OWN code
        and therefore CANNOT answer "what's next" for something that doesn't exist yet -- see
        StackDecoder.decode_level's query_last=None comment). To generate: sample from an
        ENCODER's own `hidden[:, -1, :]` to get the next value in THAT level's input alphabet (e.g.
        level1 sampling gives the next level0 code) -- `generate_level_codes` in qcute_lagcodec_decoder.py
        already implements exactly this, one call per new code. Feed that new code down into the
        level-below's decode_level as an ordinary (now-real) code value; recurse down to level0 for
        actual new bytes. This is docs/qcute_lagcodec_plan.md's "path (b)" (upper-level LM predicts the
        next code directly) -- already the documented design, just re-stated here at the point
        where it's actually implemented, since decode_level's own local view make it easy to assume
        (wrongly) that decode needs to solve this itself."""
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

        if compute_code:
            extracted = bb.extract_code(h, x0, K, window)
            code, entropy_reg = extracted["code"], extracted["entropy_reg"]
        else:
            code = entropy_reg = None
        return make_dict(code=code, ntp_loss=ntp_loss, ntp_acc=ntp_acc, hidden=h, entropy_reg=entropy_reg)
