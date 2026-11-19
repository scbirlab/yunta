"""DCA on synthetic data with a planted inter-chain coupling."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yunta.interactions.dca.dca_torch import calculate_dca, _calculate_dca


def _planted_msa(n_rows=600, len_a=12, len_b=10, coupled=((3, 4),), seed=0):
    """Random MSA where chain-A column ``i`` determines chain-B column ``j``."""
    rng = np.random.default_rng(seed)
    msa = rng.integers(0, 20, size=(n_rows, len_a + len_b))
    for i, j in coupled:
        msa[:, len_a + j] = (msa[:, i] * 7 + 3) % 20  # deterministic, bijective partner
    return msa.astype(np.uint8)


def test_planted_coupling_is_top_interchain_pair():
    len_a = 12
    msa = _planted_msa(len_a=len_a)
    scores = calculate_dca(msa, apc=False, gpu=False)
    inter = scores[:len_a, len_a:]
    assert np.unravel_index(np.argmax(inter), inter.shape) == (3, 4)
    assert inter[3, 4] > 3 * np.median(inter)


def test_output_is_symmetric_with_zero_diagonal():
    scores = calculate_dca(_planted_msa(), apc=True, gpu=False)
    assert np.allclose(scores, scores.T, atol=1e-4)
    assert np.all(np.diag(scores) == 0)


def test_float64_is_honoured():
    """Regression: the dtype was resolved twice and float64 silently fell back to float32."""
    msa = torch.tensor(_planted_msa(n_rows=200), dtype=torch.int64)
    assert _calculate_dca(msa, gpu=False, dtype=torch.float64).dtype == torch.float64
    assert _calculate_dca(msa, gpu=False, dtype="float64").dtype == torch.float64
    assert _calculate_dca(msa, gpu=False, dtype="float32").dtype == torch.float32
    assert calculate_dca(msa.numpy().astype(np.uint8), gpu=False, dtype="float64").dtype == np.float64
