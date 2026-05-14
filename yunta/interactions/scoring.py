"""Tools to score and analyze PPIs."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


def score_contact_map(M: np.ndarray, top_n: int = 5) -> dict:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    weightsA = U[:, 0]
    weightsB = Vt[0, :]
    return {
        "mean": np.mean(M),
        "median": np.median(M),
        "var": np.var(M),
        "minimum": np.min(M),
        "maximum": np.max(M),
        "sigma1":   s[0],
        "focality": s[0] / s[1] if s[1] > 0 else np.inf,
        "top_A": np.flatnonzero(-weightsA < np.sort(-weightsA)[top_n]),
        "top_B": np.flatnonzero(-weightsB < np.sort(-weightsB)[top_n]),
        # "weights_A": weightsA,   # per-residue weights, protein A
        # "weights_B": weightsB,  # per-residue weights, protein B
    }


