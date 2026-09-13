"""Load train/test TSV, encode, and iterate batches (CPU tensors).

TSV columns: HLA \t peptide \t affinity \t label \t ic50_numeric
  * HLA is already the normalised allele key (see data_pipeline / mhc_pseudo).

``DataProvider`` holds the full row set plus per-residue encodings (cached per
unique peptide / allele). ``iter_batches(indices, ...)`` yields dict batches so
callers drive their own k-fold / OOF logic.
"""

import numpy as np
import pandas as pd
import torch

from seq_encoding import blosum_encode, simple_blosum10_encode, PEP_MAX_LEN, PSEUDO_LEN
from mhc_pseudo import PseudoSeqTable


class DataProvider:
    def __init__(self, tsv_path, pseudo_csv, pep_max_len=PEP_MAX_LEN, pseudo_len=PSEUDO_LEN):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.df["peptide"] = self.df["peptide"].astype(str)
        self.df["HLA"] = self.df["HLA"].astype(str)
        self.pep_max_len = pep_max_len
        self.pseudo_len = pseudo_len
        self.pseudo = PseudoSeqTable.load(pseudo_csv)

        missing = sorted(set(self.df["HLA"]) - self.pseudo.keys)
        if missing:
            raise KeyError(f"{len(missing)} alleles have no pseudo-sequence: {missing[:5]}...")

        self._pep_cache = {}
        self._mhc_cache = {}
        self.labels = self.df["label"].to_numpy(dtype=np.float32)
        self.affinities = self.df["affinity"].to_numpy(dtype=np.float32)
        self.alleles = self.df["HLA"].to_numpy()

    def __len__(self):
        return len(self.df)

    def pos_weight(self, indices=None):
        y = self.labels if indices is None else self.labels[indices]
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        return n_neg / max(n_pos, 1.0)

    def _encode_pep(self, pep):
        t = self._pep_cache.get(pep)
        if t is None:
            t = blosum_encode(pep, self.pep_max_len)[0]
            self._pep_cache[pep] = t
        return t

    def _encode_mhc(self, allele_key):
        t = self._mhc_cache.get(allele_key)
        if t is None:
            seq = self.pseudo.get(allele_key)
            t = simple_blosum10_encode(seq, self.pseudo_len)[0]
            self._mhc_cache[allele_key] = t
        return t

    def iter_batches(self, indices, batch_size=32, shuffle=False, drop_last=False, seed=None):
        idx = np.asarray(indices)
        if shuffle:
            rng = np.random.default_rng(seed)
            idx = idx.copy()
            rng.shuffle(idx)
        n = len(idx)
        for start in range(0, n, batch_size):
            chunk = idx[start:start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            peps = torch.stack([self._encode_pep(self.df.iat[i, 1]) for i in chunk], dim=0)
            mhcs = torch.stack([self._encode_mhc(self.df.iat[i, 0]) for i in chunk], dim=0)
            yield {
                "peptide": peps,                                  # (B, 23, pep_max_len)
                "pseudo": mhcs,                                   # (B, 10, 34)
                "label": torch.from_numpy(self.labels[chunk]),    # (B,)
                "affinity": torch.from_numpy(self.affinities[chunk]),  # (B,)
                "index": chunk,
            }


if __name__ == "__main__":
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(HERE, "..", "..", "custom_dataset", "Anthem_dataset")
    dp = DataProvider(os.path.join(base, "train_data.txt"),
                      os.path.join(base, "mhc_ii_pseudo.csv"))
    print("rows:", len(dp), "| pos_weight:", round(dp.pos_weight(), 2))
    b = next(dp.iter_batches(range(len(dp)), batch_size=32))
    print({k: tuple(v.shape) for k, v in b.items() if hasattr(v, "shape")})
