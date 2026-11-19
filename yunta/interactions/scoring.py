"""Tools to score and analyze PPIs."""

import numpy as np
from numpy.typing import ArrayLike


def _top_n_idx(weights: ArrayLike, top_n: int = 5) -> str:
    """Indices of the ``top_n`` largest weights, highest first, as a ``;``-joined string.

    Examples
    ========
    >>> _top_n_idx([.1, .9, .3, .5], top_n=2)
    '1;3'
    >>> _top_n_idx([.1, .9], top_n=5)
    '1;0'
    >>> _top_n_idx([], top_n=5)
    ''

    """
    weights = np.asarray(weights).ravel()
    idx = np.argsort(-weights, kind="stable")[:top_n]
    return ";".join(map(str, idx.tolist()))


def score_contact_map(M: ArrayLike, top_n: int = 5) -> dict:
    """Summarise an inter-chain contact/coupling matrix.

    ``M`` is the ``(len_A, len_B)`` inter-chain block, so every entry is a legitimate
    residue pair and nothing is masked.
    The leading singular vectors are sign-fixed so the top residues are the ones
    with the *largest* weights, for matrices that are mostly positive.

    Examples
    ========
    >>> import numpy as np
    >>> M = np.zeros((4, 3)); M[1, 2] = 1.; M[2, 2] = .5
    >>> scores = score_contact_map(M, top_n=2)
    >>> scores["top_A"], scores["top_B"]
    ('1;2', '2;0')
    >>> round(float(scores["maximum"]), 2)
    1.0
    >>> score_contact_map(np.zeros((0, 3)))["sigma1"]
    -1.0

    """
    M = np.asarray(M, dtype=np.float64)
    empty = {
        "mean": None, "median": None, "var": None, "minimum": None, "maximum": None,
        "sigma1": -1., "focality": -1., "top_A": "", "top_B": "",
    }
    if M.size == 0 or not np.all(np.isfinite(M)):
        return empty
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return empty | {
            "mean": np.mean(M), "median": np.median(M), "var": np.var(M),
            "minimum": np.min(M), "maximum": np.max(M),
        }
    weights_a, weights_b = U[:, 0], Vt[0, :]
    if weights_a.sum() < 0:  # SVD sign is arbitrary; make the dominant direction positive
        weights_a, weights_b = -weights_a, -weights_b
    sigma1 = float(s[0])
    sigma2 = float(s[1]) if len(s) > 1 else -1.
    return {
        "mean": np.mean(M),
        "median": np.median(M),
        "var": np.var(M),
        "minimum": np.min(M),
        "maximum": np.max(M),
        "sigma1": sigma1,
        "focality": sigma1 / sigma2 if sigma2 > 0. else -1.,
        "top_A": _top_n_idx(weights_a, top_n=top_n),
        "top_B": _top_n_idx(weights_b, top_n=top_n),
    }
