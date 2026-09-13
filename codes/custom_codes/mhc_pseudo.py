"""MHC-II allele name normalisation and pseudo-sequence lookup.

Briefing section 1.4 flags this as an UNRESOLVED item: the real 34-residue
pseudo-sequences (IMGT/HLA-derived, NetMHCIIpan-style) were never wired in and
the server copy used placeholder text. This module keeps that state explicit:

  * ``normalize_allele``  -> canonical key shared by equivalent allele spellings
  * ``PseudoSeqTable``    -> loads ``mhc_ii_pseudo.csv``; every row is tagged
                             ``real`` or ``placeholder`` and a loud warning is
                             printed when placeholders are in use.
  * ``build_pseudo_csv``  -> (re)generates the CSV, keeping any ``real`` rows and
                             filling the rest with deterministic placeholders.

Swap placeholders for real sequences by editing ``mhc_ii_pseudo.csv`` in place
(set the ``source`` column to ``real``) and retraining.
"""

import csv
import hashlib
import os
import warnings

from seq_encoding import STANDARD_AA, PSEUDO_LEN

_AA = "ACDEFGHIKLMNPQRSTVWY"  # deterministic placeholder alphabet

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "custom_dataset", "Anthem_dataset", "mhc_ii_pseudo.csv",
)


# ---------------------------------------------------------------------------
# Allele name normalisation
# ---------------------------------------------------------------------------
def normalize_allele(raw, collapse_dra=True):
    """Map an IEDB allele string to a canonical key.

    Examples
    --------
    ``HLA-DRA*01:01/DRB1*04:01`` -> ``DRB1*04:01``   (DRA is monomorphic)
    ``HLA-DRB1*04:01``           -> ``DRB1*04:01``
    ``HLA-DPA1*01:03/DPB1*04:01``-> ``DPA1*01:03-DPB1*04:01`` (DPA polymorphic)
    ``HLA-DPB1*04:01``           -> ``DPB1*04:01``
    ``HLA-DRB3``                 -> ``DRB3``
    """
    s = raw.strip()
    if s.upper().startswith("HLA-"):
        s = s[4:]
    parts = [p for p in s.split("/") if p]

    if collapse_dra:
        parts = [p for p in parts if not p.upper().startswith("DRA")]

    if not parts:                       # was DRA-only, shouldn't happen
        return s
    if len(parts) == 1:
        return parts[0]
    return "-".join(parts)


# ---------------------------------------------------------------------------
# Deterministic placeholder pseudo-sequence
# ---------------------------------------------------------------------------
def placeholder_pseudo(allele_key, length=PSEUDO_LEN):
    """Repeatable pseudo-random 34-mer derived from the allele key.

    Deterministic across runs/machines (sha256 of the key). NOT biologically
    meaningful - a stand-in so the pipeline and model plumbing can run.
    """
    digest = hashlib.sha256(allele_key.encode("utf-8")).digest()
    # stretch the 32-byte digest to `length` bytes
    while len(digest) < length:
        digest += hashlib.sha256(digest).digest()
    return "".join(_AA[b % len(_AA)] for b in digest[:length])


# ---------------------------------------------------------------------------
# CSV table
# ---------------------------------------------------------------------------
class PseudoSeqTable:
    """allele_key -> (pseudo_sequence, source)."""

    def __init__(self, rows):
        self._rows = dict(rows)  # key -> (seq, source)
        self._n_placeholder = sum(1 for _, src in self._rows.values() if src != "real")
        if self._n_placeholder:
            warnings.warn(
                "MHC-II pseudo-sequences: {}/{} alleles use PLACEHOLDER text "
                "(not real IMGT/HLA sequences). See briefing 1.4. Results are "
                "plumbing-only until real sequences are wired in.".format(
                    self._n_placeholder, len(self._rows)
                ),
                stacklevel=2,
            )

    @classmethod
    def load(cls, path=DEFAULT_CSV):
        rows = []
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append((r["allele_key"], (r["pseudo_sequence"], r.get("source", "placeholder"))))
        return cls(rows)

    def __contains__(self, key):
        return key in self._rows

    def __len__(self):
        return len(self._rows)

    def get(self, key):
        entry = self._rows.get(key)
        return entry[0] if entry else None

    @property
    def n_placeholder(self):
        return self._n_placeholder

    @property
    def keys(self):
        return set(self._rows)


def build_pseudo_csv(allele_keys, path=DEFAULT_CSV):
    """Write/refresh the pseudo-sequence CSV for ``allele_keys``.

    Existing ``real`` rows are preserved; every other required key gets a
    deterministic placeholder. Returns the ``PseudoSeqTable``.
    """
    existing = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                existing[r["allele_key"]] = (r["pseudo_sequence"], r.get("source", "placeholder"))

    rows = []
    for key in sorted(set(allele_keys)):
        if key in existing and existing[key][1] == "real":
            rows.append((key, existing[key][0], "real"))
        else:
            rows.append((key, placeholder_pseudo(key), "placeholder"))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["allele_key", "pseudo_sequence", "source"])
        w.writerows(rows)

    return PseudoSeqTable([(k, (s, src)) for k, s, src in rows])


if __name__ == "__main__":
    for a in ["HLA-DRA*01:01/DRB1*04:01", "HLA-DRB1*04:01",
              "HLA-DPA1*01:03/DPB1*04:01", "HLA-DPB1*04:01", "HLA-DRB3"]:
        k = normalize_allele(a)
        print(f"{a:32s} -> {k:22s} {placeholder_pseudo(k)}")
