"""OpenFold3 PDB output is only produced for whole-complex calls when requested."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_of3_structure_is_a_pdb_only_when_requested(tmp_path):
    pytest.importorskip("biotite")
    from test_of3 import _fake_outputs
    from yunta.interactions.of3.run import read_prediction
    _fake_outputs(tmp_path, "q", 3, n_res=6, len_a=4)
    assert "structure" not in read_prediction(tmp_path, "q", 3)
    pdb = read_prediction(tmp_path, "q", 3, structure=True)["structure"]
    lines = [l for l in pdb.splitlines() if l.startswith("ATOM")]
    assert len(lines) == 6 * 4 and {l[21] for l in lines} == {"A", "B"}


def test_of3_runner_returns_pdb_for_whole_complex_only(tmp_path):
    pytest.importorskip("biotite")
    from test_chunking import _synthetic_paired_msas
    from test_of3 import FakeModel
    from yunta.interactions.runner import OF3Runner
    from yunta.structs.msa import PairedMSA
    paired = PairedMSA.from_msa(*_synthetic_paired_msas(n_rows=12, len_a=14, len_b=10, coupled=()))
    whole = OF3Runner().run_paired(paired, cpu=True, model=FakeModel(tmp_path / "w"), structure=True)[2]
    assert whole.structure.count("ATOM") == 24 * 4 and {l[21] for l in whole.structure.splitlines() if l.startswith("ATOM")} == {"A", "B"}
    none = OF3Runner().run_paired(paired, cpu=True, model=FakeModel(tmp_path / "n"))[2]
    assert none.structure is None
    chunked = OF3Runner().run_paired(paired, cpu=True, model=FakeModel(tmp_path / "c"), structure=True, chunksize=14)[2]
    assert chunked.structure is None
