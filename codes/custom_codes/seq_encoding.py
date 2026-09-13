"""Sequence encoding for the cartcell rebuild (MHC-II, Phase 2).

Two encoders:
  * ``blosum_encode``          -> peptide, 23-dim BLOSUM62, zero-padded to ``max_len``
  * ``simple_blosum10_encode`` -> allele pseudo-sequence, 10-dim reduced BLOSUM62,
                                  zero-padded to 34 (briefing section 1.4:
                                  "34 residue, 10차원 단순화 BLOSUM62")

The BLOSUM62 matrix is copied verbatim from the base repo
(``codes/Anthem_codes/seq_encoding.py``) so peptide encoding matches the
upstream CapsNet-MHC convention. The 10-dim reduced encoding takes the first
10 columns of each BLOSUM62 row - deterministic and documented, nothing learned.
"""

import torch

# ---------------------------------------------------------------------------
# BLOSUM62 (23-dim rows) -- verbatim from codes/Anthem_codes/seq_encoding.py
# ---------------------------------------------------------------------------
BLOSUM62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1],
    'B': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1],
    'Z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1],
}

# Standard 20 amino acids (used to reject non-standard peptides upstream).
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

PEP_DIM = 23           # peptide BLOSUM62 channel count
PEP_MAX_LEN = 25       # briefing 1.1: peptide length 13-25 aa
PSEUDO_DIM = 10        # allele reduced-BLOSUM channel count
PSEUDO_LEN = 34        # briefing 1.4: 34-residue pseudo-sequence


def blosum_encode(seq, max_len=PEP_MAX_LEN):
    """Encode ``seq`` as a ``(23, max_len)`` float tensor, zero-padded.

    Returns ``(tensor, mask)`` where ``mask`` is ``(max_len,)`` with 1.0 at
    valid residue positions. Unknown residues -> all-zero column, mask 0.
    """
    tensor = torch.zeros((PEP_DIM, max_len), dtype=torch.float32)
    mask = torch.zeros((max_len,), dtype=torch.float32)
    for i, aa in enumerate(seq[:max_len]):
        row = BLOSUM62.get(aa.upper())
        if row is not None:
            tensor[:, i] = torch.tensor(row, dtype=torch.float32)
            mask[i] = 1.0
    return tensor, mask


def simple_blosum10_encode(seq, max_len=PSEUDO_LEN):
    """Reduced 10-dim BLOSUM62 encoding for the allele pseudo-sequence.

    Deterministic dimensionality reduction: take the first 10 columns of each
    BLOSUM62 row. Returns ``(tensor, mask)`` with ``tensor`` shape
    ``(10, max_len)``.
    """
    tensor = torch.zeros((PSEUDO_DIM, max_len), dtype=torch.float32)
    mask = torch.zeros((max_len,), dtype=torch.float32)
    for i, aa in enumerate(seq[:max_len]):
        row = BLOSUM62.get(aa.upper())
        if row is not None:
            tensor[:, i] = torch.tensor(row[:PSEUDO_DIM], dtype=torch.float32)
            mask[i] = 1.0
    return tensor, mask


if __name__ == "__main__":
    pt, pm = blosum_encode("KTSLYNLRRGTALAIP")
    at, am = simple_blosum10_encode("YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY")
    print("peptide", tuple(pt.shape), "mask sum", pm.sum().item())
    print("pseudo ", tuple(at.shape), "mask sum", am.sum().item())
