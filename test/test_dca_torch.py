"""Check the optimised torch DCA against the straightforward reference port."""
from pathlib import Path

import numpy as np
import pytest
import torch

from yunta.interactions.dca.dca_torch import (
    _calculate_dca,
    _calculate_dca_reference,
    calculate_dca,
)
from yunta.structs.msa import MSA, _A3M_ALPHABET_SIZE

INPUTS = Path(__file__).parent / "inputs"


def _random_msa(n_row=60, n_col=14, seed=0):
    """Small MSA with a conserved column, an all-gap column and unobserved states."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 6, size=(1, n_col))  # restricted alphabet -> unobserved states
    msa = np.where(rng.random((n_row, n_col)) < .3, rng.integers(0, 6, size=(n_row, n_col)), base)
    msa[:, 3] = base[0, 3]  # conserved
    msa[:, 7] = _A3M_ALPHABET_SIZE - 1  # all gap
    msa[::5, 9:12] = _A3M_ALPHABET_SIZE - 1
    msa[1] = msa[0]  # duplicate sequences -> weights != 1
    return msa


def _real_msa(name="WWM1_YEAST.a3m", max_rows=None):
    msa = np.asarray(MSA.from_file(str(INPUTS / name)).sequence_token_ids)
    return msa if max_rows is None else msa[:max_rows]


@pytest.mark.parametrize("apc", [False, True])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("msa_fn", [_random_msa, _real_msa], ids=["synthetic", "WWM1"])
def test_matches_reference_float64(msa_fn, prune, apc):
    msa = torch.as_tensor(msa_fn(), dtype=torch.long)
    kwargs = dict(apc=apc, gpu=False, dtype="float64")
    expected = _calculate_dca_reference(msa, **kwargs)
    observed = _calculate_dca(msa, prune=prune, **kwargs)
    assert observed.dtype == torch.float64
    torch.testing.assert_close(observed, expected, rtol=1e-8, atol=1e-9)
    assert torch.all(torch.diagonal(observed) == 0)


def test_matches_reference_float32():
    msa = torch.as_tensor(_real_msa("P9WGF1_Mtb_Mmr.a3m", max_rows=800), dtype=torch.long)
    expected = _calculate_dca_reference(msa, apc=True, gpu=False)
    observed = _calculate_dca(msa, apc=True, gpu=False)
    scale = expected.abs().max()
    assert (observed - expected).abs().max() / scale < 1e-4


def test_block_sizes_do_not_change_results(monkeypatch):
    from yunta.interactions.dca import dca_torch
    msa = torch.as_tensor(_random_msa(n_row=70), dtype=torch.long)
    expected = _calculate_dca(msa, apc=True, gpu=False, dtype="float64")
    monkeypatch.setattr(dca_torch, "_WEIGHT_BLOCK", 7)
    monkeypatch.setattr(dca_torch, "_GRAM_BLOCK", 5)
    monkeypatch.setattr(dca_torch, "_SCORE_BLOCK", 3)
    observed = _calculate_dca(msa, apc=True, gpu=False, dtype="float64")
    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-12)


def test_dtype_is_respected():
    msa = _random_msa()
    assert calculate_dca(msa, dtype="float64", gpu=False).dtype == np.float64
    assert calculate_dca(msa, dtype="float32", gpu=False).dtype == np.float32


def test_block_size_default_is_read_at_call_time(monkeypatch):
    """Regression: defaults must not be frozen at function definition."""
    from yunta.interactions.dca import dca_torch
    import inspect
    assert inspect.signature(dca_torch._symmetric_gram).parameters["block_size"].default is None
    assert inspect.signature(dca_torch._sequence_weights).parameters["block_size"].default is None
    monkeypatch.setattr(dca_torch, "_GRAM_BLOCK", 6)
    x = torch.rand(10, 20, dtype=torch.float64)
    orig = torch.Tensor.__setitem__
    count = {"n": 0}
    def counting(self, idx, val):
        count["n"] += 1
        return orig(self, idx, val)
    monkeypatch.setattr(torch.Tensor, "__setitem__", counting)
    dca_torch._symmetric_gram(x)
    # upper tiles: ceil(20/6)=4, mirror: 4 -> 8 assignments (a 512 block would give 2)
    assert count["n"] == 8


def test_lu_fallback_when_cholesky_fails(monkeypatch):
    """If Cholesky reports failure, the LU path must give the same answer."""
    msa = torch.as_tensor(_random_msa(), dtype=torch.long)
    expected = _calculate_dca(msa, gpu=False, dtype="float64")
    real = torch.linalg.cholesky_ex
    monkeypatch.setattr(
        torch.linalg, "cholesky_ex",
        lambda a, out=None: (real(a)[0], torch.tensor(1, dtype=torch.int32)),  # ignores `out`: matrix left unfactored
    )
    observed = _calculate_dca(msa, gpu=False, dtype="float64")
    torch.testing.assert_close(observed, expected, rtol=1e-8, atol=1e-9)


def test_invalid_dtype_rejected():
    with pytest.raises(ValueError):
        calculate_dca(_random_msa(), dtype="float16", gpu=False)


@pytest.mark.parametrize("kind", [
    "conserved",
    "correlated",
    pytest.param("shallow_long", marks=pytest.mark.slow),  # inverts a ~2500-feature matrix
])
def test_float32_stress_ranking(kind):
    """float32 error on small scores and ranking of the strongest couplings vs float64 reference."""
    rng = np.random.default_rng(1)
    if kind == "conserved":   # most columns almost invariant
        msa = np.where(rng.random((300, 40)) < .02, rng.integers(0, 20, (300, 40)), 3)
    elif kind == "correlated":  # blocks of columns copied from each other
        base = rng.integers(0, 20, (300, 10))
        msa = np.concatenate([base, base, np.where(rng.random((300, 10)) < .1, rng.integers(0, 20, (300, 10)), base)], 1)
    else:  # fewer sequences than features
        msa = rng.integers(0, 20, (25, 120))
    msa_t = torch.as_tensor(msa, dtype=torch.long)
    ref = _calculate_dca_reference(msa_t, gpu=False, dtype="float64", apc=False)
    new = _calculate_dca(msa_t, gpu=False, dtype="float32", apc=False).double()
    off = ~torch.eye(ref.shape[0], dtype=torch.bool)
    # error relative to each entry's own scale (floor at 1% of the max to avoid dividing by ~0)
    floor = ref.max() * 1e-2
    rel = ((new - ref).abs() / ref.abs().clamp(min=floor))[off]
    assert rel.max() < 1e-3
    k = 20
    i, j = torch.triu_indices(ref.shape[0], ref.shape[1], offset=1)  # unique residue pairs
    top_ref = set(torch.topk(ref[i, j], k).indices.tolist())
    top_new = set(torch.topk(new[i, j], k).indices.tolist())
    assert len(top_ref & top_new) >= k - 1


def test_dtype_objects_validated():
    with pytest.raises(ValueError):
        calculate_dca(_random_msa(), dtype=torch.float16, gpu=False)
    assert calculate_dca(_random_msa(), dtype=torch.float64, gpu=False).dtype == np.float64


@pytest.mark.parametrize("name", ["WWM1_YEAST.a3m", "P9WGF1_Mtb_Mmr.a3m", "CDC42_HUMAN.a3m"])
def test_cholesky_succeeds_on_real_msas(name, monkeypatch):
    """The Cholesky path (not the LU fallback) is taken on realistic alignments in float32."""
    path = INPUTS / name if (INPUTS / name).exists() else INPUTS / "human" / name
    msa = np.asarray(MSA.from_file(str(path)).sequence_token_ids)[:600]
    calls = []
    real = torch.linalg.cholesky_ex

    def tracked(a, out=None):
        chol, info = real(a) if out is None else real(a, out=out)
        calls.append(int(info))
        return chol, info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", tracked)
    _calculate_dca(torch.as_tensor(msa, dtype=torch.long), gpu=False)
    assert calls == [0]


def test_single_sequence_rejected():
    with pytest.raises(ValueError, match="at least two"):
        calculate_dca(np.zeros((1, 5), dtype=int), gpu=False)
