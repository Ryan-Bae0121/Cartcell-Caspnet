"""Phase 3 model: Multi-branch CapsNet + Dynamic Routing (briefing section 2, Phase 3).

Three branches build 56 primary capsules total:

  full peptide branch : pep_stem -> global avg pool -> Linear -> 32 capsules
  core branch         : pep_stem -> CoreRegister    -> Linear -> 16 capsules
  MHC branch          : pseudo_encoder              -> Linear ->  8 capsules

56 primary capsules -> DynamicRouting (3 iters) -> 3 output capsules:
  [0] negative-class capsule, [1] positive/strong-binder-class capsule, [2] regression capsule.

  cls_logit = ||v_pos|| - ||v_neg||           (briefing: "두 class capsule의 norm 차이 -> logit")
  reg_yhat  = Linear(v_reg)                   (briefing: "regression capsule -> Linear -> yhat")
  pos_prob  = CoreRegister's softmax position distribution (interpretability, same as Phase 2)
  coupling_cls / coupling_reg = per-primary-capsule coupling mass routed to the
                                 class capsules / regression capsule (interpretability)

forward returns (cls_logit, reg_yhat, pos_prob, coupling_cls, coupling_reg).

``pep_stem`` / ``core`` / ``pseudo_encoder`` reuse the exact module names Phase2Model
uses, so a Phase-2 checkpoint's weights can warm-start these branches directly --
see ``warm_start_from_phase2`` below. This is the briefing's Phase-3 bug note made
moot from the start: unify ``mhc_encoder``/``pseudo_encoder`` naming so warm-start
never silently drops the MHC branch.
"""

import argparse

import torch
import torch.nn as nn

try:
    from .common import conv_stack, CoreRegister, PseudoSeqEncoder, DynamicRouting, squash
    from .phase2 import Phase2Model
except ImportError:  # allow `python models/phase3.py`
    from common import conv_stack, CoreRegister, PseudoSeqEncoder, DynamicRouting, squash
    from phase2 import Phase2Model


class Phase3Model(nn.Module):
    def __init__(
        self,
        pep_in=23,
        pep_channels=(64, 128),
        core_kernel=9,
        pseudo_in=10,
        pseudo_channels=(32, 64),
        cap_dim=8,
        out_cap_dim=16,
        routing_iters=3,
        n_full_caps=32,
        n_core_caps=16,
        n_mhc_caps=8,
        dropout=0.3,
    ):
        super().__init__()
        # Same names/shapes as Phase2Model -> warm-startable.
        self.pep_stem, pep_out = conv_stack(pep_in, list(pep_channels), kernel_size=3)
        self.core = CoreRegister(pep_out, core_dim=pep_out, core_kernel=core_kernel)
        self.pseudo_encoder = PseudoSeqEncoder(pseudo_in, tuple(pseudo_channels))

        self.full_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.cap_dim = cap_dim
        self.n_full_caps = n_full_caps
        self.n_core_caps = n_core_caps
        self.n_mhc_caps = n_mhc_caps

        self.full_proj = nn.Linear(pep_out, n_full_caps * cap_dim)
        self.core_proj = nn.Linear(pep_out, n_core_caps * cap_dim)
        self.mhc_proj = nn.Linear(self.pseudo_encoder.out_dim, n_mhc_caps * cap_dim)

        n_primary = n_full_caps + n_core_caps + n_mhc_caps  # 56
        self.n_primary = n_primary
        self.routing = DynamicRouting(n_primary, n_out=3, in_dim=cap_dim,
                                       out_dim=out_cap_dim, n_iters=routing_iters)
        self.reg_out = nn.Linear(out_cap_dim, 1)

    def _to_caps(self, proj, vec, n_caps):
        b = vec.size(0)
        s = proj(self.dropout(vec)).view(b, n_caps, self.cap_dim)
        return squash(s, dim=-1)

    def forward(self, peptide, pseudo):
        pep_feat = self.pep_stem(peptide)                # (B, C, L)
        core_vec, pos_prob = self.core(pep_feat)         # (B, C), (B, T)
        full_vec = self.full_pool(pep_feat).squeeze(2)   # (B, C)
        mhc_vec = self.pseudo_encoder(pseudo)             # (B, C2)

        full_caps = self._to_caps(self.full_proj, full_vec, self.n_full_caps)  # (B, 32, cap_dim)
        core_caps = self._to_caps(self.core_proj, core_vec, self.n_core_caps)  # (B, 16, cap_dim)
        mhc_caps = self._to_caps(self.mhc_proj, mhc_vec, self.n_mhc_caps)      # (B, 8, cap_dim)

        primary = torch.cat([full_caps, core_caps, mhc_caps], dim=1)  # (B, 56, cap_dim)
        v, c = self.routing(primary)                                   # (B, 3, out_dim), (B, 56, 3)

        v_neg, v_pos, v_reg = v[:, 0], v[:, 1], v[:, 2]
        cls_logit = (v_pos.norm(dim=1) - v_neg.norm(dim=1)).unsqueeze(1)  # (B, 1)
        reg_yhat = self.reg_out(v_reg)                                    # (B, 1)

        coupling_cls = c[:, :, 0] + c[:, :, 1]   # (B, 56) mass routed to either class capsule
        coupling_reg = c[:, :, 2]                # (B, 56) mass routed to the regression capsule

        return cls_logit, reg_yhat, pos_prob, coupling_cls, coupling_reg


def build_from_config(cfg):
    return Phase3Model(
        pep_channels=tuple(cfg.get("pep_channels", (64, 128))),
        core_kernel=cfg.get("core_kernel", 9),
        pseudo_channels=tuple(cfg.get("pseudo_channels", (32, 64))),
        cap_dim=cfg.get("cap_dim", 8),
        out_cap_dim=cfg.get("out_cap_dim", 16),
        routing_iters=cfg.get("routing_iters", 3),
        n_full_caps=cfg.get("n_full_caps", 32),
        n_core_caps=cfg.get("n_core_caps", 16),
        n_mhc_caps=cfg.get("n_mhc_caps", 8),
        dropout=cfg.get("dropout", 0.3),
    )


# Phase2Model prefixes that Phase3Model's pep_stem/core/pseudo_encoder can absorb
# directly (identical submodule names + shapes when pep_channels/pseudo_channels/
# core_kernel match between the two configs).
_SHARED_PREFIXES = ("pep_stem.", "core.")
_MHC_PREFIX = "pseudo_encoder."


def warm_start_from_phase2(model, phase2_state_dict, core_only=False):
    """Copy pep_stem / core (+ pseudo_encoder unless ``core_only``) weights from a
    Phase2Model state_dict into ``model`` in place.

    Mirrors the server's ``--warm_start_core_only`` flag (briefing section 2:
    "full warm-start (core + MHC) 이 core-only 보다 회귀 안정성 우수" -> full is the
    default). Keys with no matching name/shape in ``model`` are skipped rather
    than raising, so this stays robust if the two configs drift apart. Returns
    the list of parameter/buffer names actually copied.
    """
    prefixes = _SHARED_PREFIXES if core_only else _SHARED_PREFIXES + (_MHC_PREFIX,)
    own = model.state_dict()
    loaded = []
    for k, v in phase2_state_dict.items():
        if not k.startswith(prefixes):
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
    model.load_state_dict(own)
    return loaded


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    B = 16
    model = Phase3Model()
    pep = torch.randn(B, 23, 25)
    pseudo = torch.randn(B, 10, 34)
    cls_logit, reg_yhat, pos_prob, coupling_cls, coupling_reg = model(pep, pseudo)

    n_primary = 32 + 16 + 8
    assert cls_logit.shape == (B, 1), cls_logit.shape
    assert reg_yhat.shape == (B, 1), reg_yhat.shape
    assert pos_prob.shape == (B, 17), pos_prob.shape
    assert coupling_cls.shape == (B, n_primary), coupling_cls.shape
    assert coupling_reg.shape == (B, n_primary), coupling_reg.shape

    row_sums = pos_prob.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), row_sums[:3]
    csum = coupling_cls + coupling_reg
    assert torch.allclose(csum, torch.ones(B, n_primary), atol=1e-4), csum[0, :5]

    loss = cls_logit.pow(2).mean() + reg_yhat.pow(2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"selftest OK | params={n_params} | primary_caps={n_primary} | "
          f"pos_prob rows sum to 1 | coupling rows sum to 1 | backward OK")

    # warm-start smoke test: Phase2Model's default config matches Phase3Model's
    # pep_stem/core/pseudo_encoder shapes, so every tensor in those 3 submodules
    # should copy across.
    p2 = Phase2Model()
    p2_state = p2.state_dict()
    n_shared = sum(1 for k in p2_state if k.startswith(("pep_stem.", "core.")))
    n_mhc = sum(1 for k in p2_state if k.startswith("pseudo_encoder."))

    fresh = Phase3Model()
    loaded_full = warm_start_from_phase2(fresh, p2_state, core_only=False)
    loaded_core = warm_start_from_phase2(Phase3Model(), p2_state, core_only=True)
    print(f"warm_start smoke test | phase2 tensors={len(p2_state)} "
          f"(shared={n_shared}, mhc={n_mhc}) | "
          f"loaded full={len(loaded_full)} core_only={len(loaded_core)}")
    assert len(loaded_full) == n_shared + n_mhc, "expected every shared+MHC tensor to transfer"
    assert len(loaded_core) == n_shared, "core_only must load only pep_stem/core, skip pseudo_encoder.*"
