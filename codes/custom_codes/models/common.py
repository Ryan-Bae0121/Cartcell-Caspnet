"""Shared building blocks for the cartcell rebuild.

``CoreRegister`` and ``PseudoSeqEncoder`` are the two Phase-2 branches described
in the briefing (section 2, Phase 2). They are kept here so Phase 3 can import
the same modules for warm-start with matching parameter names -- the allele
branch is called ``pseudo_encoder`` everywhere (briefing's Phase-3 bug note:
unify ``mhc_encoder`` / ``pseudo_encoder`` from the start).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_stack(in_ch, channels, kernel_size=3):
    """Length-preserving Conv1d stack: Conv -> BN -> LeakyReLU per layer."""
    layers = []
    prev = in_ch
    for ch in channels:
        layers += [
            nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(),
        ]
        prev = ch
    return nn.Sequential(*layers), prev


class CoreRegister(nn.Module):
    """Peptide feature map -> soft 9-mer binding-core vector.

    A ``kernel=core_kernel`` valid conv turns the length-L feature map into
    ``T = L - core_kernel + 1`` candidate-position feature columns; a parallel
    1-channel conv scores each position; softmax over T gives ``pos_prob``;
    the core vector is the pos_prob-weighted sum of the position columns.

    Allele conditioning (``mhc_dim`` set)
    -------------------------------------
    With ``mhc_dim=None`` the position score comes from the peptide alone, so
    ``pos_prob`` cannot depend on the allele -- measured, not assumed: corrupting
    the MHC input (zero / permuted / swapped / fresh fingerprint) leaves the
    predicted core unmoved for 100% of peptides (``ablate_pseudo.py``). Any
    allele-dependent register behaviour is therefore architecturally impossible,
    with real pseudo-sequences as much as with placeholders.

    Passing ``mhc_dim`` adds a cosine similarity between each candidate core's
    features and a projection of ``mhc_vec``, i.e. "which of the T candidate
    9-mers fits this pocket best", behind a single learnable gate:

        score = score_conv(x) + g * cos(feat_t, W mhc_vec)

    Three deliberate choices:

    * The term is *content-based*. A per-position bias (``score += W mhc_vec``
      with ``W`` mapping to T) would instead memorise "allele A likes index 3",
      which says nothing about the residues at that index. A position-constant
      offset would also cancel in the softmax entirely.
    * Both vectors are L2-normalised, so the term is a bounded [-1, 1] cosine
      and its scale does not depend on the feature magnitudes. An unnormalised
      dot product needed ``W`` near unit scale to reach parity with
      ``score_conv`` here (measured: ``feat`` std 0.087, ``score_conv`` std
      0.068), which makes the pathway's influence an accident of initialisation.
    * ``cond_gate`` is zero-initialised, so at step 0 the conditioned model is
      *mathematically identical* to the unconditioned one: peptide-side weights
      stay warm-startable from existing checkpoints and training only has to
      learn the added pathway. ``mhc_proj`` keeps its default init on purpose --
      zero-initialising *both* would make each one's gradient proportional to
      the other's zero and the pathway could never start learning. The learned
      ``cond_gate`` value is also directly reportable: it is how strongly the
      trained model chose to let the pocket move the register.

    ``mhc_dim=None`` also keeps the ``state_dict`` byte-identical to the
    pre-conditioning model, so old checkpoints load with no key mismatch -- that
    is the unconditioned arm of the ablation, not a legacy path.
    """

    def __init__(self, in_ch, core_dim=128, core_kernel=9, mhc_dim=None):
        super().__init__()
        self.core_kernel = core_kernel
        self.core_dim = core_dim
        self.feat_conv = nn.Conv1d(in_ch, core_dim, kernel_size=core_kernel)
        self.score_conv = nn.Conv1d(in_ch, 1, kernel_size=core_kernel)
        if mhc_dim:
            self.mhc_proj = nn.Linear(mhc_dim, core_dim)
            self.cond_gate = nn.Parameter(torch.zeros(1))
        else:
            self.mhc_proj = None
            self.cond_gate = None

    @property
    def conditioned(self):
        return self.mhc_proj is not None

    def forward(self, x, mhc_vec=None):
        # x: (B, in_ch, L)
        feat = self.feat_conv(x)                     # (B, core_dim, T)
        score = self.score_conv(x).squeeze(1)        # (B, T)
        if self.mhc_proj is not None:
            if mhc_vec is None:
                raise ValueError(
                    "CoreRegister was built with mhc_dim set but forward() got no "
                    "mhc_vec -- the caller must encode the allele before selecting "
                    "the core, not after."
                )
            m = F.normalize(self.mhc_proj(mhc_vec), dim=1)      # (B, core_dim)
            cos = (F.normalize(feat, dim=1) * m.unsqueeze(2)).sum(dim=1)  # (B, T) in [-1, 1]
            score = score + self.cond_gate * cos
        pos_prob = F.softmax(score, dim=1)           # (B, T)
        core_vec = torch.bmm(feat, pos_prob.unsqueeze(2)).squeeze(2)  # (B, core_dim)
        return core_vec, pos_prob


class PseudoSeqEncoder(nn.Module):
    """Allele pseudo-sequence (10, 34) -> fixed-size ``mhc_vec``."""

    def __init__(self, in_ch=10, channels=(32, 64), kernel_size=3):
        super().__init__()
        self.conv, out_ch = conv_stack(in_ch, list(channels), kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_ch

    def forward(self, x):
        y = self.conv(x)                 # (B, C, 34)
        return self.pool(y).squeeze(2)   # (B, C)


def entropy(prob, eps=1e-8):
    """Row-wise Shannon entropy of a probability tensor ``(B, T)`` -> ``(B,)``."""
    return -(prob * (prob + eps).log()).sum(dim=1)


def squash(s, dim=-1, eps=1e-8):
    """CapsNet squash non-linearity (briefing section 2, Phase 3):

        v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)

    Keeps direction, maps length into [0, 1). ``eps`` avoids a 0/0 grad when a
    capsule's pre-squash vector is exactly zero.
    """
    sq_norm = (s * s).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * s / torch.sqrt(sq_norm + eps)


class DynamicRouting(nn.Module):
    """Routing-by-agreement from ``n_in`` primary capsules to ``n_out`` output
    capsules (briefing section 2, Phase 3).

        u_hat_(j|i) = W_ij . u_i
        c_ij        = softmax_j(b_ij)          (softmax over output capsules)
        s_j         = sum_i c_ij . u_hat_(j|i)
        v_j         = squash(s_j)
        b_ij       <- b_ij + u_hat_(j|i) . v_j     (skipped on the last iteration)

    ``b_ij`` starts at zero every forward call (routing state is not learned;
    only ``W`` is). Returns ``(v, c)``:
      * ``v`` -- ``(B, n_out, out_dim)`` output capsule vectors
      * ``c`` -- ``(B, n_in, n_out)`` final coupling coefficients, useful for
                 interpretability (how much each primary capsule contributed
                 to each output capsule).
    """

    def __init__(self, n_in, n_out, in_dim, out_dim, n_iters=3):
        super().__init__()
        self.n_in, self.n_out, self.n_iters = n_in, n_out, n_iters
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, out_dim, in_dim))

    def forward(self, u):
        # u: (B, n_in, in_dim)
        B = u.size(0)
        u_hat = torch.einsum("ijoc,bic->bijo", self.W, u)  # (B, n_in, n_out, out_dim)
        b = torch.zeros(B, self.n_in, self.n_out, device=u.device, dtype=u.dtype)
        v, c = None, None
        for it in range(self.n_iters):
            c = torch.softmax(b, dim=2)                        # (B, n_in, n_out)
            s = torch.einsum("bij,bijo->bjo", c, u_hat)         # (B, n_out, out_dim)
            v = squash(s, dim=-1)
            if it < self.n_iters - 1:
                b = b + torch.einsum("bijo,bjo->bij", u_hat, v)
        return v, c
