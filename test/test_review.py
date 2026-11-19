"""Regression tests for the review of the refactor: paths, errors, control invariants, reuse, lifecycle."""
import json
import sys
from collections import Counter

import numpy as np
import pytest

from test_chunking import _synthetic_paired_msas
from yunta.coevo import CoevoPredictor
from yunta.interactions.runner import _pair_msas
from yunta.structs.msa import MSA, MSALine, PairedMSA, _symmetrise
from yunta.structs.pairing import InvalidControlError, UnpairableError, multiplicity

torch = pytest.importorskip("torch")


def _msa(prefix, species, length=10, seed=0):
    rng = np.random.default_rng(seed)
    letters = np.array(list("ARNDCQEGHILKMFPSTWYV"))
    return MSA([
        MSALine(name=f">sp|{prefix}{r}|{prefix}{r}_X",
                description=f"Gene OS=Species {sp} OX={sp} GN=x PE=1 SV=1",
                sequence="".join(letters[rng.integers(0, 20, length)]))
        for r, sp in enumerate(species)
    ])


def _species(paired):
    return [tuple(d.species_id for d in line.description) for line in paired.lines[1:]]


@pytest.fixture
def host_pathogen():
    """Query species 1/1001; species-level map with many-to-many edges, so rows must be reused."""
    hosts = list(range(1, 31))
    pathogens = [1000 + i for i in range(1, 31)]
    rows1 = [hosts[0]] + hosts * 2   # duplicates of species: several rows per species
    rows2 = [pathogens[0]] + pathogens * 2
    amap = {f"NCBI:{h}": [f"Species {p}" for p in pathogens if (h + p) % 3 == 0 or p == 1000 + h] for h in hosts}
    return _msa("A", rows1, seed=1), _msa("B", rows2, seed=2), amap


# --- P1: output paths --------------------------------------------------------------
def test_cli_writes_everything_inside_output_dir(tmp_path, monkeypatch):
    from yunta.cli import main
    a, b = _synthetic_paired_msas(n_rows=60, len_a=12, len_b=10)
    for name, m in zip("ab", (a, b)):
        with open(tmp_path / f"{name}.a3m", "w") as f:
            m.write(f)
    out = tmp_path / "results"
    for argv in (
        ["pair", str(tmp_path / "a.a3m"), "--msa2", str(tmp_path / "b.a3m"), "-o", str(out / "paired")],
        ["coevo", str(tmp_path / "a.a3m"), "--msa2", str(tmp_path / "b.a3m"), "-m", "dca", "--cpu", "-o", str(out / "pred")],
    ):
        monkeypatch.setattr(sys, "argv", ["yunta"] + argv)
        main()
    created = {p.relative_to(tmp_path).parts[0] for p in tmp_path.iterdir()}
    assert created == {"a.a3m", "b.a3m", "results"}, f"stray files outside -o: {created}"
    assert {p.suffix for p in (out / "paired").iterdir()} == {".a3m", ".json"}
    assert (out / "pred" / "dca.tsv").exists() and list((out / "pred").glob("dca.*.npz"))


# --- P1: error handling ------------------------------------------------------------
def test_invalid_pairing_mode_raises_instead_of_skipping(host_pathogen):
    a, b, _ = host_pathogen
    with pytest.raises(ValueError, match="pairing"):
        _pair_msas(a, b, pairing="shuffled")
    with pytest.raises(ValueError):
        CoevoPredictor([(a, b)], pairing="shuffled")


def test_bad_interaction_map_type_raises(host_pathogen):
    a, b, _ = host_pathogen
    with pytest.raises(ValueError, match="interaction_map"):
        _pair_msas(a, b, interaction_map=42)


def test_unpairable_inputs_are_skipped_with_none():
    a = _msa("A", [1, 2, 3])
    b = _msa("B", [10, 11, 12])  # no shared species, and strict same-species pairing
    assert _pair_msas(a, b, enforce_ref_match=True) is None
    assert issubclass(UnpairableError, AttributeError) and issubclass(UnpairableError, ValueError)


# --- P1/P2: controls ---------------------------------------------------------------
@pytest.mark.parametrize("pairing", ["rewired", "random-species", "random-rows"])
def test_controls_meet_their_invariants_and_report_them(host_pathogen, pairing):
    a, b, amap = host_pathogen
    real = PairedMSA.from_msa(a, b, interaction_map=amap)
    ctrl = PairedMSA.from_msa(a, b, interaction_map=amap, pairing=pairing, seed=4)
    r = ctrl.pairing_info
    assert r.method == pairing and r.seed == 4 and r.valid, r.notes
    assert r.n_pairs == r.target_pairs == real.pairing_info.n_pairs  # depth preserved exactly
    assert r.retained_known == 0
    assert len(ctrl) == len(real)
    known = set()
    for h, ps in amap.items():
        for p in ps:
            known.add((h, p)); known.add((p, h))
    got = _species(ctrl)
    rows_known = [(x, y) for x, y in got if (x, y) in known]
    assert rows_known == [], f"control kept {len(rows_known)} known species pairs"


def test_rewired_preserves_per_species_multiplicity(host_pathogen):
    a, b, amap = host_pathogen
    real = PairedMSA.from_msa(a, b, interaction_map=amap)
    ctrl = PairedMSA.from_msa(a, b, interaction_map=amap, pairing="rewired", seed=1)
    assert ctrl.pairing_info.multiplicity_ok
    assert multiplicity(_species(ctrl)) == multiplicity(_species(real))


def test_control_seed_round_trip_reproduces_pairing(host_pathogen):
    a, b, amap = host_pathogen
    first = PairedMSA.from_msa(a, b, interaction_map=amap, pairing="random-rows")  # no seed given
    seed = first.pairing_info.seed
    assert isinstance(seed, int)
    again = PairedMSA.from_msa(a, b, interaction_map=amap, pairing="random-rows", seed=seed)
    assert again.sequences() == first.sequences()


def test_infeasible_control_is_flagged_and_strict_rejects():
    """Complete bipartite map: no pair lies outside the map, so no valid control exists."""
    sp = list(range(1, 7))
    a, b = _msa("A", sp, seed=1), _msa("B", sp, seed=2)
    amap = {f"NCBI:{i}": [f"NCBI:{j}" for j in sp] for i in sp}
    ctrl = PairedMSA.from_msa(a, b, interaction_map=amap, pairing="random-species", seed=1, strict_controls=False)
    assert not ctrl.pairing_info.valid and ctrl.pairing_info.notes
    with pytest.raises(InvalidControlError):  # strict is the default
        PairedMSA.from_msa(a, b, interaction_map=amap, pairing="random-species", seed=1)


def test_metrics_record_pairing_provenance_and_score_mode(host_pathogen):
    a, b, amap = host_pathogen
    pred = next(CoevoPredictor([(a, b)], interaction_map=amap, pairing="rewired", seed=9, cpu=True).predict("dca"))
    m = pred.metrics
    assert (m.pairing_method, m.pairing_seed, m.pairing_valid, m.score_mode) == ("rewired", 9, True, "whole")
    assert m.pairing_depth == m.pairing_target_depth and m.pairing_retained_known == 0


# --- P1: two-stage workflow --------------------------------------------------------
def test_coevo_from_saved_paired_msa_matches_in_process_pairing(tmp_path, monkeypatch, host_pathogen):
    from yunta.cli import main
    a, b, amap = host_pathogen
    a, b = _synthetic_paired_msas(n_rows=80, len_a=14, len_b=12)
    for name, m in zip("ab", (a, b)):
        with open(tmp_path / f"{name}.a3m", "w") as f:
            m.write(f)
    monkeypatch.setattr(sys, "argv", ["yunta", "pair", str(tmp_path / "a.a3m"), "--msa2", str(tmp_path / "b.a3m"),
                                      "--pairing", "random-rows", "--seed", "3", "-o", str(tmp_path / "p")])
    main()
    a3m = next((tmp_path / "p").glob("*.a3m"))
    meta = json.loads(a3m.with_suffix(".json").read_text())
    assert meta["chain_a_length"] == 14 and meta["pairing"]["seed"] == 3
    monkeypatch.setattr(sys, "argv", ["yunta", "coevo", str(a3m), "--paired", "-m", "dca", "--cpu", "-o", str(tmp_path / "o")])
    main()
    row = (tmp_path / "o" / "dca.tsv").read_text().splitlines()
    header, values = row[0].split("\t"), row[1].split("\t")
    got = dict(zip(header, values))
    assert got["pairing_method"] == "random-rows" and got["pairing_seed"] == "3" and got["chain_a_len"] == "14"
    direct = next(CoevoPredictor([(MSA.from_file(str(tmp_path / "a.a3m")), MSA.from_file(str(tmp_path / "b.a3m")))],
                                 pairing="random-rows", seed=3, cpu=True).predict("dca"))
    assert direct.paired_msa.sequences() == PairedMSA.read(str(a3m), 14).sequences()


def test_paired_input_without_sidecar_is_an_error(tmp_path, monkeypatch):
    from yunta.cli import main
    a, b = _synthetic_paired_msas(n_rows=20, len_a=8, len_b=6)
    p = PairedMSA.from_msa(a, b)
    with open(tmp_path / "x.a3m", "w") as f:
        p.write(f)
    monkeypatch.setattr(sys, "argv", ["yunta", "coevo", str(tmp_path / "x.a3m"), "--paired", "-m", "dca", "--cpu", "-o", str(tmp_path / "o")])
    with pytest.raises(FileNotFoundError, match="sidecar"):
        main()


# --- P2: lifecycle -----------------------------------------------------------------
class _Closable:
    closed = 0
    def close(self):
        type(self).closed += 1


def test_models_are_closed_by_close_model_context_and_error(host_pathogen):
    a, b, _ = host_pathogen
    _Closable.closed = 0
    with CoevoPredictor([(a, b)], cpu=True) as p:
        p._models["x"] = _Closable()
        p._models["y"] = _Closable()
        p.close_model("x")
        assert _Closable.closed == 1
    assert _Closable.closed == 2 and not p._models

    class Boom(CoevoPredictor):
        def predict(self, method, **kw):
            self._models[method] = _Closable()
            raise RuntimeError("boom")
            yield
    _Closable.closed = 0
    pred = Boom([(a, b)], cpu=True)
    with pytest.raises(RuntimeError):
        list(pred.predict_all(["dca"]))
    assert _Closable.closed == 1


# --- P2: chunking / caches / scores ------------------------------------------------
def test_symmetrise_cache_sees_in_place_mutation():
    m = {"a": ["b"]}
    assert "z" not in _symmetrise(m)
    m["a"].append("z")  # mutate a value list in place
    assert "z" in _symmetrise(m)["a"]
