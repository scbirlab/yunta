"""Gated smoke test: real OpenFold3 on a GPU. Skipped unless openfold3, CUDA and weights are present.

Run with ``YUNTA_OF3_SMOKE=1 pytest test/test_of3_gpu_smoke.py``. Two queries go through ONE model
object so per-query state and artefact reading are exercised across calls.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("YUNTA_OF3_SMOKE"), reason="set YUNTA_OF3_SMOKE=1 on a GPU host")


def test_two_sequential_queries_with_structure():
    pytest.importorskip("openfold3")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import numpy as np
    from yunta.interactions.runner import OF3Runner
    from yunta.structs.msa import PairedMSA
    from test_chunking import _synthetic_paired_msas
    runner = OF3Runner()
    model = runner.make_model(cpu=False)
    for i in range(2):
        paired = PairedMSA.from_msa(*_synthetic_paired_msas(n_rows=16, len_a=30 + i, len_b=25, coupled=()))
        matrix, interaction, metrics = runner.run_paired(paired, cpu=False, model=model, structure=True)
        assert interaction.shape == (30 + i, 25) and np.isfinite(metrics.ptm)
        assert metrics.structure.startswith(("ATOM", "HEADER", "REMARK", "CRYST1", "MODEL"))
