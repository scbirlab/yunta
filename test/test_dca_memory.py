"""DCA memory: peak memory against the reference, streamed one-hot matrix, in-place factorisation."""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


_PEAK = textwrap.dedent("""
    import resource, sys, numpy as np, torch
    from yunta.interactions.dca import dca_torch as d
    rng = np.random.default_rng(0); base = rng.integers(0, 21, 300)
    x = np.where(rng.random((5000, 300)) < .5, base, rng.integers(0, 21, (5000, 300)))
    # ru_maxrss is KiB on Linux, bytes on macOS
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    before = rss()
    fn = d._calculate_dca_reference if sys.argv[1] == "reference" else d._calculate_dca
    with torch.no_grad():
        fn(torch.as_tensor(x, dtype=torch.long), gpu=False)
    print(rss() - before)
""")


@pytest.mark.slow
def test_peak_memory_is_lower_than_the_reference():
    """Relative (same machine, fresh processes), so independent of platform and absolute RAM."""
    def peak(which):
        out = subprocess.run([sys.executable, "-c", _PEAK, which], capture_output=True, text=True, check=True)
        return float(out.stdout.strip().splitlines()[-1])
    old_inc, new_inc = peak("reference"), min(peak("optimised") for _ in range(2))
    assert new_inc <= .9 * old_inc, (old_inc, new_inc)


# --- AF2 configuration (no JAX run needed) --------------------------------------------
# --- DCA: the (M, N) one-hot matrix is streamed, never built whole ---------------------------------------
def _deep_msa(m=60, length=12, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 6, length)
    x = np.where(rng.random((m, length)) < .5, base, rng.integers(0, 6, (m, length)))
    x[1] = x[0]
    return torch.as_tensor(x, dtype=torch.long)


def test_one_hot_is_streamed_in_row_slabs_and_results_do_not_depend_on_slab_size(monkeypatch):
    from yunta.interactions.dca import dca_torch as d
    x = _deep_msa()
    expected = d._calculate_dca(x, apc=True, gpu=False, dtype="float64")
    widest = []
    real = d._compact_one_hot
    monkeypatch.setattr(d, "_compact_one_hot", lambda xs, *a, **k: (widest.append(xs.shape[0]), real(xs, *a, **k))[1])
    monkeypatch.setattr(d, "_ROW_BLOCK", 7)
    monkeypatch.setattr(d, "_WEIGHT_BLOCK", 9)
    observed = d._calculate_dca(x, apc=True, gpu=False, dtype="float64")
    assert max(widest) <= 9 < x.shape[0]          # never the whole alignment at once
    torch.testing.assert_close(observed, expected, rtol=1e-9, atol=1e-11)


def test_precomputed_weights_path_matches_and_streams(monkeypatch):
    from yunta.interactions.dca import dca_torch as d
    x = _deep_msa()
    w = d.sequence_weights(x.numpy(), gpu=False, dtype="float64")
    expected = d._calculate_dca(x, gpu=False, dtype="float64")
    monkeypatch.setattr(d, "_ROW_BLOCK", 5)
    observed = d._calculate_dca(x, gpu=False, dtype="float64", weights=torch.as_tensor(w))
    torch.testing.assert_close(observed, expected, rtol=1e-9, atol=1e-11)


def test_lu_fallback_rebuilds_the_matrix_destroyed_by_the_in_place_factorisation(monkeypatch):
    """A failed in-place Cholesky has overwritten the covariance; the fallback must rebuild it."""
    from yunta.interactions.dca import dca_torch as d
    x = _deep_msa()
    expected = d._calculate_dca(x, gpu=False, dtype="float64")
    real = torch.linalg.cholesky_ex

    def failing(a, out=None):
        real(a, out=out)                      # really factorise in place (destroys `a`) ...
        return out[0], torch.tensor(1, dtype=torch.int32)  # ... and report failure

    monkeypatch.setattr(torch.linalg, "cholesky_ex", failing)
    torch.testing.assert_close(d._calculate_dca(x, gpu=False, dtype="float64"), expected, rtol=1e-8, atol=1e-10)
