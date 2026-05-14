"""Tools to score and analyze PPIs."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


def _get_top_n_idx(a: ArrayLike, top_n: int = 5, descending: bool = False):
    a = a.ravel()
    if descending:
        a = -a
    if a.size <= top_n:
        idx = range(a.size)
    else:
        cut = np.sort(a)[top_n]
        idx = np.flatnonzero(a <= cut)
    return ";".join(map(str, idx))


def score_contact_map(M: ArrayLike, top_n: int = 5) -> dict:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if M.size > 0:
        weightsA = U[:, 0]
        weightsB = Vt[0, :]
        sigma1 = s[0]
        sigma2 = s[1] if len(s) > 1 else -1.
    else:
        weightsA = U[:]
        weightsB = Vt[:]
        sigma1 = -1.
        sigma2 = -1.

    if M.shape[0] == M.shape[1] and np.all(M.ravel()[0] == np.diag(M)):
        mask = (np.ones_like(M) - np.eye(M.shape[0])).astype(bool)
    else:
        mask = np.ones_like(M).astype(bool)
    mask_size = mask.sum()
    at_least_one = mask_size > 0
    masked_M = M[mask]
        
    return {
        "mean": np.mean(masked_M) if at_least_one else None,
        "median": np.median(masked_M) if at_least_one else None,
        "var": np.var(masked_M) if at_least_one else None,
        "minimum": np.min(masked_M) if at_least_one else None,
        "maximum": np.max(masked_M) if at_least_one else None,
        "sigma1": sigma1,
        "focality": sigma1 / sigma2 if len(s) > 0 and sigma2 > 0. else -1.,
        "top_A": _get_top_n_idx(weightsA, top_n=top_n, descending=True),
        "top_B": _get_top_n_idx(weightsB, top_n=top_n, descending=True),
        # "weights_A": weightsA,   # per-residue weights, protein A
        # "weights_B": weightsB,  # per-residue weights, protein B
    }


