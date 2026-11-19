"""OpenFold3 backend tests, with a stand-in for the model (OpenFold3 itself needs GPU + weights)."""
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

biotite = pytest.importorskip("biotite")
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

from test_chunking import _synthetic_paired_msas
from yunta.interactions.of3.run import post_of3, read_prediction, run_of3, write_query
from yunta.interactions.runner import OF3Runner
from yunta.structs.msa import PairedMSA


def _fake_outputs(out_dir: Path, query_id: str, seed: int, n_res: int, len_a: int, full_fmt="json"):
    """Write what OpenFold3's output writer writes for one sample."""
    atoms = []
    for i in range(n_res):
        for name, off in (("N", 0.), ("CA", 1.), ("CB", 2.), ("C", 3.)):
            a = struc.Atom([3.8 * i + off * .3, 0., 0.], chain_id="A" if i < len_a else "B", res_id=i + 1 - (0 if i < len_a else len_a),
                           res_name="ALA", atom_name=name, element=name[0], hetero=False)
            atoms.append(a)
    arr = struc.array(atoms)
    arr.set_annotation("b_factor", np.full(arr.array_length(), 80.))
    sample = out_dir / query_id / f"seed_{seed}"
    sample.mkdir(parents=True)
    prefix = sample / f"{query_id}_seed_{seed}_sample_1"
    f = pdbx.CIFFile(); pdbx.set_structure(f, arr, data_block="m")
    f.write(f"{prefix}_model.cif")
    Path(f"{prefix}_confidences_aggregated.json").write_text(json.dumps({"ptm": .61, "iptm": .42, "avg_plddt": 80.}))
    pae = np.full((n_res, n_res), 4.); pae[:len_a, len_a:] = pae[len_a:, :len_a] = 3.
    if full_fmt == "json":
        Path(f"{prefix}_confidences.json").write_text(json.dumps({"plddt": [80.] * n_res, "pae": pae.tolist(), "pde": pae.tolist()}))
    else:
        np.savez_compressed(f"{prefix}_confidences.npz", plddt=np.full(n_res, 80.), pae=pae, pde=pae)


def _fake_capture(n_res: int, len_a: int):
    """What OpenFold3 computes but does not write: PAE bin probabilities and contact probabilities."""
    from yunta.interactions.confidence import PAE_N_BINS
    probs = np.zeros((n_res, n_res, PAE_N_BINS), dtype=np.float32)
    probs[..., 8] = 1.                      # every pair: PAE in bin 8 (4.25 A)
    probs[:len_a, len_a:, 8] = 0.; probs[len_a:, :len_a, 8] = 0.
    probs[:len_a, len_a:, 6] = 1.; probs[len_a:, :len_a, 6] = 1.   # inter-chain: bin 6 (3.25 A)
    contact = np.zeros((n_res, n_res), dtype=np.float32); contact[:len_a, len_a:] = contact[len_a:, :len_a] = .5
    return {"pae_probs": probs, "contact_probs": contact}


class FakeModel:
    def __init__(self, out_dir, full_fmt="json", capture=True):
        self.out_dir, self.full_fmt, self.queries, self.capture = Path(out_dir), full_fmt, [], capture

    def predict(self, query_json, query_id, seed, structure=False):
        spec = json.loads(Path(query_json).read_text())["queries"][query_id]["chains"]
        self.queries.append(spec)
        len_a = len(spec[0]["sequence"]); n = len_a + len(spec[1]["sequence"])
        _fake_outputs(self.out_dir, query_id, seed, n, len_a, self.full_fmt)
        captured = _fake_capture(n, len_a) if self.capture else None
        return read_prediction(self.out_dir, query_id, seed, structure=structure, captured=captured)


def test_write_query_splits_paired_rows_per_chain(tmp_path):
    rows = ["ACDE" "FG-H", "A-DE" "FGKH", "ACDQ" "FG-H"]
    path = write_query(tmp_path, "q1", rows, ["query", "h1", "h2"], 4)
    chains = json.loads(path.read_text())["queries"]["q1"]["chains"]
    assert [c["sequence"] for c in chains] == ["ACDE", "FGH"]  # gaps removed from the query only
    a3m_a = (tmp_path / "q1" / "A" / "colabfold_paired.a3m").read_text().split()
    a3m_b = (tmp_path / "q1" / "B" / "colabfold_paired.a3m").read_text().split()
    assert a3m_a[1::2] == ["ACDE", "A-DE", "ACDQ"] and a3m_b[1::2] == ["FG-H", "FGKH", "FG-H"]  # row-aligned
    assert (tmp_path / "q1" / "A" / "colabfold_main.a3m").read_text().count(">") == 1  # query only


@pytest.mark.parametrize("fmt", ["json", "npz"])
def test_read_prediction(tmp_path, fmt):
    _fake_outputs(tmp_path, "q", 7, n_res=6, len_a=4, full_fmt=fmt)
    pred = read_prediction(tmp_path, "q", 7)
    assert pred["contact_dists"].shape == (6, 6) and pred["pae"].shape == (6, 6)
    assert pred["ptm_model"] == .61 and pred["iptm_model"] == .42  # OpenFold3's own, kept for cross-checking
    assert "tm" not in pred  # no bin probabilities were captured, so no TM-based scores
    assert pred["plddt"].tolist() == [80.] * 6


def test_runner_end_to_end_with_fake_model_whole_and_chunked(tmp_path):
    a, b = _synthetic_paired_msas(n_rows=12, len_a=14, len_b=10, coupled=())
    paired = PairedMSA.from_msa(a, b)
    whole = OF3Runner().run_paired(paired, cpu=True, model=FakeModel(tmp_path / "w"), chunksize=1000, seed=1)
    matrix, interaction, metrics = whole
    assert matrix.shape == (24, 24) and interaction.shape == (14, 10)
    assert metrics.ptm_model == .61 and metrics.iptm_model == .42 and metrics.seed == 1
    assert 0. < metrics.ptm < 1. and 0. < metrics.iptm < 1. and 0. < metrics.actifptm < 1.  # yunta's shared code
    assert metrics.n_contacts > 0 and not np.isnan(metrics.ipsae) and not np.isnan(metrics.lis)
    chunked = OF3Runner().run_paired(paired, cpu=True, model=FakeModel(tmp_path / "c"), chunksize=14, seed=1)[2]
    assert whole[2].score_mode == "whole" and chunked.score_mode == "blockwise"
    assert np.isnan(chunked.iptm) and np.isnan(chunked.actifptm)   # need one model's output
    assert not np.isnan(chunked.ipsae) and not np.isnan(chunked.lis)  # blockwise approximations
    assert chunked.mean_interface_plddt == pytest.approx(80.)


def test_ensure_checkpoint_uses_given_path_and_complains_if_missing(tmp_path):
    from yunta.interactions.of3.run import ensure_checkpoint
    ckpt = tmp_path / "w.pt"; ckpt.write_bytes(b"x")
    assert ensure_checkpoint(checkpoint=ckpt) == ckpt
    with pytest.raises(FileNotFoundError):
        ensure_checkpoint(checkpoint=tmp_path / "missing.pt")


def test_without_captured_probabilities_tm_scores_are_nan_not_the_models_own_numbers(tmp_path):
    a, b = _synthetic_paired_msas(n_rows=12, len_a=14, len_b=10, coupled=())
    metrics = OF3Runner().run_paired(PairedMSA.from_msa(a, b), cpu=True,
                                     model=FakeModel(tmp_path / "n", capture=False), chunksize=1000, seed=1)[2]
    assert np.isnan(metrics.ptm) and np.isnan(metrics.iptm) and np.isnan(metrics.actifptm)
    assert metrics.ptm_model == .61 and not np.isnan(metrics.ipsae)  # PAE-only scores still work


def test_capture_hook_keeps_pae_and_contact_probabilities_and_still_calls_the_writer():
    torch = pytest.importorskip("torch")
    from yunta.interactions.of3.run import OF3Model
    n = 6
    seen = []

    class Writer:
        def write_all_outputs(self, batch, outputs, confidence_scores):
            seen.append(sorted(confidence_scores))

    model = OF3Model(runner=SimpleNamespace(callbacks=[object(), Writer()]), output_dir=Path("."))
    assert model.install_capture() and model.capturing
    probs = torch.softmax(torch.randn(1, 1, n, n, 64), dim=-1)   # [batch, sample, tokens, tokens, bins]
    contact = torch.rand(1, 1, n, n)
    model.runner.callbacks[1].write_all_outputs({"query_id": ["qa"]}, {}, {"pae_probs": probs, "contact_probs": contact, "plddt": 1})
    got = model.captured["qa"]
    assert got["pae_probs"].shape == (n, n, 64) and got["contact_probs"].shape == (n, n)
    assert np.allclose(got["pae_probs"], probs[0, 0].numpy()) and seen == [["contact_probs", "pae_probs", "plddt"]]


def test_capture_hook_reports_a_missing_writer_instead_of_failing(capsys):
    from yunta.interactions.of3.run import OF3Model
    model = OF3Model(runner=SimpleNamespace(callbacks=[object()]), output_dir=Path("."))
    assert model.install_capture() is False and not model.capturing
    assert "will be NaN" in capsys.readouterr().err
