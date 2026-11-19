import sys
import numpy as np
import pytest

from test_chunking import _synthetic_paired_msas
from yunta.coevo import CoevoPredictor

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def msas():
    return _synthetic_paired_msas(n_rows=200, len_a=20, len_b=16)


def test_pairs_once_and_streams_predictions(msas):
    a, b = msas
    predictor = CoevoPredictor([(a, b), (a, None)], cpu=True)
    assert len(predictor) == 2 and not predictor.skipped
    preds = list(predictor.predict("dca", apc=True))
    assert [p.matrix.shape for p in preds] == [(36, 36), (40, 40)]
    assert np.unravel_index(np.argmax(preds[0].interaction), preds[0].interaction.shape) == (7, 5)


def test_rewired_control_loses_the_planted_signal(msas):
    a, b = msas
    real = next(CoevoPredictor([(a, b)], cpu=True).predict("dca", apc=True))
    ctrl = next(CoevoPredictor([(a, b)], cpu=True, pairing="random-rows", seed=1).predict("dca", apc=True))
    assert real.interaction.max() > 3 * ctrl.interaction.max()


def test_unknown_method_rejected(msas):
    with pytest.raises(ValueError):
        next(CoevoPredictor([msas], cpu=True).predict("nope"))


def test_cli_pair_and_coevo(tmp_path, msas, monkeypatch):
    from yunta.cli import main
    for name, m in zip("ab", msas):
        with open(tmp_path / f"{name}.a3m", "w") as f:
            m.write(f)
    argv = ["yunta", "coevo", str(tmp_path / "a.a3m"), "--msa2", str(tmp_path / "b.a3m"),
            "-m", "dca", "--cpu", "--pairing", "rewired", "--seed", "1", "-o", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert (tmp_path / "out" / "dca.tsv").exists()
    monkeypatch.setattr(sys, "argv", ["yunta", "pair", str(tmp_path / "a.a3m"), "--msa2", str(tmp_path / "b.a3m"),
                                      "--pairing", "random-species", "--seed", "2", "-o", str(tmp_path / "p")])
    main()
    assert list((tmp_path / "p").glob("*.a3m"))
