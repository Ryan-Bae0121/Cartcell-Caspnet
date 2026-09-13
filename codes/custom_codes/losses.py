"""Multi-task loss for Phase 2 (briefing section 3).

    L = L_cls + lambda_reg * L_reg + entropy_beta * L_entropy

  * L_cls      : BCEWithLogitsLoss(pos_weight)  on the strong-binder label
  * L_reg      : MSE  on the normalised affinity
  * L_entropy  : mean row-wise entropy of pos_prob, added as a PENALTY so the
                 binding-core position distribution stays concentrated
  * lambda_reg : 0.1 for Phase 2
"""

import torch
import torch.nn as nn

try:
    from .common import entropy
except ImportError:
    from models.common import entropy


class Phase2Loss(nn.Module):
    def __init__(self, pos_weight, lambda_reg=0.1, entropy_beta=0.01):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.lambda_reg = lambda_reg
        self.entropy_beta = entropy_beta
        self.mse = nn.MSELoss()

    def forward(self, cls_logit, reg_yhat, pos_prob, label, affinity):
        label = label.view(-1, 1).float()
        affinity = affinity.view(-1, 1).float()
        l_cls = nn.functional.binary_cross_entropy_with_logits(
            cls_logit, label, pos_weight=self.pos_weight
        )
        l_reg = self.mse(reg_yhat, affinity)
        l_ent = entropy(pos_prob).mean()
        total = l_cls + self.lambda_reg * l_reg + self.entropy_beta * l_ent
        return total, {
            "loss": total.item(),
            "cls": l_cls.item(),
            "reg": l_reg.item(),
            "entropy": l_ent.item(),
        }


class Phase3Loss(Phase2Loss):
    """Same multi-task objective as Phase 2 -- Phase3Model's forward returns
    (cls_logit, reg_yhat, pos_prob, coupling_cls, coupling_reg); only the first
    three feed the loss, so this reuses Phase2Loss verbatim.

    Default ``lambda_reg=0.5`` (briefing section 3 + server state's
    ``warmfix_full_rerun``: Phase3 uses 0.5 in both the 5-fold run and LOMO,
    vs. 0.1 for Phase 2)."""

    def __init__(self, pos_weight, lambda_reg=0.5, entropy_beta=0.01):
        super().__init__(pos_weight, lambda_reg, entropy_beta)
