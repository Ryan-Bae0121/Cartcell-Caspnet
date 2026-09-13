"""Phase 2 model: CNN + Core Register module (briefing section 2, Phase 2).

peptide (B,23,25) --stem conv--> (B,128,25) --CoreRegister--> core_vec (B,128), pos_prob (B,17)
pseudo  (B,10,34) --pseudo_encoder--> mhc_vec (B,64)
concat(core_vec, mhc_vec) --MLP--> cls_logit (B,1), reg_yhat (B,1)

forward returns (cls_logit, reg_yhat, pos_prob).
"""

import argparse

import torch
import torch.nn as nn

try:
    from .common import conv_stack, CoreRegister, PseudoSeqEncoder
except ImportError:  # allow `python models/phase2.py`
    from common import conv_stack, CoreRegister, PseudoSeqEncoder


class Phase2Model(nn.Module):
    def __init__(
        self,
        pep_in=23,
        pep_channels=(64, 128),
        core_kernel=9,
        pseudo_in=10,
        pseudo_channels=(32, 64),
        mlp_hidden=128,
        dropout=0.5,
    ):
        super().__init__()
        self.pep_stem, pep_out = conv_stack(pep_in, list(pep_channels), kernel_size=3)
        self.core = CoreRegister(pep_out, core_dim=pep_out, core_kernel=core_kernel)
        self.pseudo_encoder = PseudoSeqEncoder(pseudo_in, tuple(pseudo_channels))

        fused = pep_out + self.pseudo_encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(fused, mlp_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.cls_out = nn.Linear(mlp_hidden, 1)
        self.reg_out = nn.Linear(mlp_hidden, 1)

    def forward(self, peptide, pseudo):
        pep_feat = self.pep_stem(peptide)                 # (B, C, L)
        core_vec, pos_prob = self.core(pep_feat)          # (B, C), (B, T)
        mhc_vec = self.pseudo_encoder(pseudo)             # (B, C2)
        fused = torch.cat([core_vec, mhc_vec], dim=1)
        h = self.head(fused)
        return self.cls_out(h), self.reg_out(h), pos_prob


def build_from_config(cfg):
    return Phase2Model(
        pep_channels=tuple(cfg.get("pep_channels", (64, 128))),
        core_kernel=cfg.get("core_kernel", 9),
        pseudo_channels=tuple(cfg.get("pseudo_channels", (32, 64))),
        mlp_hidden=cfg.get("mlp_hidden", 128),
        dropout=cfg.get("dropout", 0.5),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    B = 32
    model = Phase2Model()
    pep = torch.randn(B, 23, 25)
    pseudo = torch.randn(B, 10, 34)
    cls_logit, reg_yhat, pos_prob = model(pep, pseudo)
    assert cls_logit.shape == (B, 1), cls_logit.shape
    assert reg_yhat.shape == (B, 1), reg_yhat.shape
    assert pos_prob.shape == (B, 17), pos_prob.shape
    row_sums = pos_prob.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), row_sums[:3]

    loss = cls_logit.pow(2).mean() + reg_yhat.pow(2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"selftest OK | params={n_params} | pos_prob rows sum to 1 | backward OK")
