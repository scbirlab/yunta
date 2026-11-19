"""Second review: final-alignment validation, strict-by-default controls, composition-preserving
control, deterministic feasibility, and chunking opt-in."""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from test_chunking import _synthetic_paired_msas
from test_review import _msa, host_pathogen  # noqa: F401  (fixture)
from yunta.coevo import CoevoPredictor
from yunta.interactions.chunking import plan_blocks
from yunta.interactions.runner import DEFAULT_CHUNKSIZE, Runner, _pair_msas
from yunta.structs.msa import MSA, MSALine, PairedMSA
from yunta.structs.pairing import (
    InvalidControlError, PairingReport, allowed_row_pair_capacity, permute_partners, sample_random_rows,
)

torch = pytest.importorskip("torch")
LETTERS = np.array(list("ARNDCQEGHILKMFPSTWYV"))


def _gapped_msa(prefix, species, gapped: dict, length=10, seed=0):
    """One row per species (query row first); ``gapped[sp]`` residues of that species' row are gaps."""
    rng = np.random.default_rng(seed)
    lines = []
    for r, sp in enumerate(species):
        seq = list(LETTERS[rng.integers(0, 20, length)])
        if r > 0:
            for k in range(gapped.get(sp, 0)):
                seq[k] = "-"
        lines.append(MSALine(name=f">sp|{prefix}{r}|{prefix}{r}_X",
                             description=f"Gene OS=Species {sp} OX={sp} GN=x PE=1 SV=1", sequence="".join(seq)))
    return MSA(lines)


SPECIES = [1] + list(range(1, 21))


# --- P1: validate the FINAL (post gap-filter) alignment ----------------------------------------
def test_control_that_loses_rows_to_the_gap_filter_is_rejected_by_default():
    """Real pairs have joint gap .4 (pass a .5 filter); mixed control pairs reach .8 and are filtered out."""
    a = _gapped_msa("A", SPECIES, {i: 8 for i in range(1, 11)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 8 for i in range(11, 21)}, seed=2)
    real = _pair_msas(a, b, max_gap_fraction=.5, pairing="map")
    assert real.pairing_info.final_pairs == real.pairing_info.final_target_pairs == 19 or real.pairing_info.final_pairs >= 19
    with pytest.raises(InvalidControlError, match="final depth"):
        _pair_msas(a, b, max_gap_fraction=.5, pairing="random-species", seed=3)


def test_allow_invalid_flags_it_and_records_pre_and_post_filter_depths():
    a = _gapped_msa("A", SPECIES, {i: 8 for i in range(1, 11)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 8 for i in range(11, 21)}, seed=2)
    ctrl = _pair_msas(a, b, max_gap_fraction=.5, pairing="random-species", seed=3, strict_controls=False)
    r = ctrl.pairing_info
    assert not r.valid and any("final depth" in n for n in r.notes)
    assert r.n_pairs == r.target_pairs          # valid *as generated* (pre-filter)...
    assert r.final_pairs < r.final_target_pairs  # ...but not as scored
    assert r.n_eff is not None and r.reference_n_eff is not None and r.mean_gap is not None


def test_control_with_surplus_rows_is_subsampled_to_the_real_final_depth():
    """Real pairs are too gappy (joint .9 > .5) so most vanish; the control keeps more and is trimmed to match."""
    a = _gapped_msa("A", SPECIES, {i: 9 for i in range(1, 11)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 9 for i in range(1, 11)}, seed=2)
    real = _pair_msas(a, b, max_gap_fraction=.5, pairing="map")
    ctrl = _pair_msas(a, b, max_gap_fraction=.5, pairing="random-species", seed=4)
    r = ctrl.pairing_info
    assert r.valid and r.subsampled_to_match
    assert r.final_pairs == r.final_target_pairs == real.pairing_info.final_pairs
    assert len(ctrl) == len(real)
    again = _pair_msas(a, b, max_gap_fraction=.5, pairing="random-species", seed=4)
    assert again.sequences() == ctrl.sequences()  # the trimming is seeded


def test_final_stats_reach_the_metrics(host_pathogen):  # noqa: F811
    a, b, amap = host_pathogen
    m = next(CoevoPredictor([(a, b)], interaction_map=amap, pairing="rewired", seed=2, cpu=True).predict("dca")).metrics
    assert m.pairing_final_depth == m.pairing_final_target_depth
    assert m.pairing_mean_gap is not None and m.pairing_reference_n_eff is not None
    assert m.n_eff == m.n_eff  # reused from the pairing step, not recomputed


# --- P2: strict by default ---------------------------------------------------------------------
def test_strict_is_default_and_batches_skip_and_list_rejected_controls():
    sp = list(range(1, 7))
    a, b = _msa("A", sp, seed=1), _msa("B", sp, seed=2)
    amap = {f"NCBI:{i}": [f"NCBI:{j}" for j in sp] for i in sp}  # complete: no valid control exists
    with pytest.raises(InvalidControlError):
        _pair_msas(a, b, interaction_map=amap, pairing="random-species", seed=1)
    p = CoevoPredictor([(a, b)], interaction_map=amap, pairing="random-species", seed=1, cpu=True)
    assert len(p) == 0 and p.skipped == [0] and "depth" in p.rejected[0]
    loose = CoevoPredictor([(a, b)], interaction_map=amap, pairing="random-species", seed=1,
                           strict_controls=False, cpu=True)
    # relaxed: the control is invalid AND has no paired rows at all, so there is nothing to score
    assert len(loose) == 0 and loose.skipped == [0] and loose.rejected == {}


def test_cli_flag_inverts_strictness():
    from yunta.cli import _pairing_kwargs
    base = dict(interspecies=False, strict_match=False, pairing="map", seed=None, max_gap_fraction=.9)
    assert _pairing_kwargs(Namespace(allow_invalid_controls=False, **base))["strict_controls"] is True
    assert _pairing_kwargs(Namespace(allow_invalid_controls=True, **base))["strict_controls"] is False


# --- P2: control that preserves composition ---------------------------------------------------
def test_permute_partners_keeps_every_row_and_breaks_every_known_pair(host_pathogen):  # noqa: F811
    a, b, amap = host_pathogen
    real = _pair_msas(a, b, interaction_map=amap, pairing="map")
    ctrl = _pair_msas(a, b, interaction_map=amap, pairing="permute-partners", seed=5)
    la = real.chain_a_length
    split = lambda m: ([s[:la] for s in m.sequences()[1:]], [s[la:] for s in m.sequences()[1:]])
    (ra, rb), (ca, cb) = split(real), split(ctrl)
    assert sorted(ra) == sorted(ca) and sorted(rb) == sorted(cb)   # identical composition and gaps per chain
    assert (ra, rb) != (ca, cb) and ra == ca                       # A rows in place, partners moved
    r = ctrl.pairing_info
    assert r.valid and r.retained_known == 0 and r.multiplicity_ok and r.n_pairs == r.target_pairs
    allowed = {(k, "NCBI:" + v.split()[1]) for k, vs in amap.items() for v in vs}
    assert not any(sp in allowed for sp in _species_pairs(ctrl)[1:])  # [0] is the real query-species row
    assert _pair_msas(a, b, interaction_map=amap, pairing="permute-partners", seed=5).sequences() == ctrl.sequences()
    assert _pair_msas(a, b, interaction_map=amap, pairing="permute-partners", seed=6).sequences() != ctrl.sequences()


def _species_pairs(paired):
    return [tuple(d.species_id for d in line.description) for line in paired.lines[1:]]


def test_permute_partners_unit_and_infeasible_case():
    keys = [f"s{i}" for i in range(12)]
    perm, kept = permute_partners(keys, keys, {(k, k) for k in keys}, np.random.default_rng(0))
    assert sorted(perm) == list(range(12)) and kept == 0
    everything = {(a, b) for a in keys for b in keys}
    _, kept = permute_partners(keys, keys, everything, np.random.default_rng(0))
    assert kept == 12  # nothing allowed: reported, not hidden


# --- P2: deterministic feasibility ----------------------------------------------------------------
def test_capacity_counts_allowed_row_pairs_exactly():
    k1, k2 = ["a", "a", "b", "c"], ["x", "y", "y"]
    known = {("a", "y"), ("c", "x")}
    brute = sum((u, v) not in known for u in k1 for v in k2)
    assert allowed_row_pair_capacity(k1, k2, known) == brute == 12 - 4 - 1


def test_exhausted_sampling_budget_is_recovered_exactly_but_impossible_is_not():
    n = 40
    forbidden = lambda i, j: i != j                      # exactly one allowed partner per row
    allowed = lambda: np.eye(n, dtype=bool)
    rng = np.random.default_rng(0)
    weak = sample_random_rows(n, n, n, rng, forbidden, max_tries_factor=1)
    assert len(weak) < n                                 # the budget really does run out
    full = sample_random_rows(n, n, n, np.random.default_rng(0), forbidden, max_tries_factor=1, allowed_matrix=allowed)
    assert full == [(i, i) for i in range(n)]
    over = sample_random_rows(n, n, n + 5, np.random.default_rng(0), forbidden, max_tries_factor=1, allowed_matrix=allowed)
    assert len(over) == n                                # only n allowed pairs exist


def test_random_rows_reports_infeasible_distinctly():
    sp = list(range(1, 7))
    a, b = _msa("A", sp, seed=1), _msa("B", sp, seed=2)
    amap = {f"NCBI:{i}": [f"NCBI:{j}" for j in sp] for i in sp}
    ctrl = PairedMSA.from_msa(a, b, interaction_map=amap, pairing="random-rows", seed=1, strict_controls=False)
    assert any(n.startswith("infeasible: only 0 allowed row pairs") for n in ctrl.pairing_info.notes)


# --- chunking is opt-in -------------------------------------------------------------------------------
class _CountingRunner(Runner):
    calls = None

    @staticmethod
    def _run_chunk(paired_msa, model=None, **kw):
        _CountingRunner.calls.append(len(paired_msa[0]))
        n = len(paired_msa[0])
        return np.zeros((n, n)), {}


@pytest.fixture
def wide():
    a, b = _synthetic_paired_msas(n_rows=10, len_a=500, len_b=420, coupled=())
    return PairedMSA.from_msa(a, b)


def test_defaults_never_chunk_even_beyond_the_old_750_column_default(wide):
    assert DEFAULT_CHUNKSIZE is None and CoevoPredictor(pairs=[]).chunksize is None
    assert [len(b) for b in plan_blocks(500, 420)] == [920]
    _CountingRunner.calls = []
    _CountingRunner().run_paired(wide, cpu=True)
    assert _CountingRunner.calls == [920]
    _CountingRunner.calls = []
    _CountingRunner().run_paired(wide, cpu=True, chunksize=500, blocks="all")
    assert len(_CountingRunner.calls) > 1 and all(n <= 500 for n in _CountingRunner.calls)


def test_whole_complex_memory_warning_suggests_chunking(wide, capsys, monkeypatch):
    from yunta.interactions import runner as R
    monkeypatch.setattr(R, "available_memory_gib", lambda cpu=False: 1.0)
    from yunta.interactions import DCARunner
    DCARunner()._warn_if_too_big(wide, cpu=True)  # 920 columns need ~ 10 GiB
    err = capsys.readouterr().err
    assert "needs roughly" in err and "--chunksize" in err
    monkeypatch.setattr(R, "available_memory_gib", lambda cpu=False: 1e6)
    DCARunner()._warn_if_too_big(wide, cpu=True)
    assert capsys.readouterr().err == ""


# --- third review: permute-partners composition after filtering; final multiplicity -------------------
def _asym_msas():
    """A is gappy for species 1-10, B for species 11-20: every real pair has joint gap .4 but a mixed pair .8."""
    a = _gapped_msa("A", SPECIES, {i: 8 for i in range(1, 11)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 8 for i in range(11, 21)}, seed=2)
    return a, b


def _chain_multisets(paired):
    la = paired.chain_a_length
    rows = [r for r in paired.sequences()[1:]]
    return sorted(r[:la] for r in rows), sorted(r[la:] for r in rows)


def test_permute_partners_keeps_exact_final_rows_under_asymmetric_gap_filtering():
    a, b = _asym_msas()
    real = _pair_msas(a, b, max_gap_fraction=.5, pairing="map")
    ctrl = _pair_msas(a, b, max_gap_fraction=.5, pairing="permute-partners", seed=7)
    assert len(real) == len(ctrl) and _chain_multisets(real) == _chain_multisets(ctrl)
    assert [r[:real.chain_a_length] for r in ctrl.sequences()] == [r[:real.chain_a_length] for r in real.sequences()]
    r = ctrl.pairing_info
    assert r.valid and not r.subsampled_to_match
    assert r.final_composition_ok and r.final_multiplicity_ok and r.final_retained_real == 0
    assert ctrl.sequences() != real.sequences()  # and the partners really moved


def test_permute_partners_filters_real_rows_first_when_some_real_pairs_are_too_gappy():
    a = _gapped_msa("A", SPECIES, {i: 9 for i in range(1, 8)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 9 for i in range(1, 8)}, seed=2)   # real species 1-7 pairs: .9 joint -> dropped
    real = _pair_msas(a, b, max_gap_fraction=.5, pairing="map")
    ctrl = _pair_msas(a, b, max_gap_fraction=.5, pairing="permute-partners", seed=3)
    assert len(ctrl) == len(real) and _chain_multisets(ctrl) == _chain_multisets(real)
    assert ctrl.pairing_info.valid and ctrl.pairing_info.final_composition_ok


def test_rewired_rejected_when_final_multiplicity_no_longer_matches():
    """Equal final depth is not equal multiplicity: trimming a rewired control to the real final depth
    changes which species it uses."""
    a = _gapped_msa("A", SPECIES, {i: 9 for i in range(1, 11)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 9 for i in range(1, 11)}, seed=2)
    with pytest.raises(InvalidControlError, match="multiplicity"):
        _pair_msas(a, b, max_gap_fraction=.5, pairing="rewired", seed=4)
    loose = _pair_msas(a, b, max_gap_fraction=.5, pairing="rewired", seed=4, strict_controls=False)
    r = loose.pairing_info
    assert not r.valid and r.final_multiplicity_ok is False and r.final_pairs == r.final_target_pairs
    assert r.multiplicity_ok is True  # the as-generated check is kept separately


def test_check_final_reports_each_failed_invariant():
    r = PairingReport("permute-partners", 1, target_pairs=3, n_pairs=3, retained_known=0, multiplicity_ok=True).finalise()
    r.check_final(3, 3, multiplicity_ok=True, composition_ok=False, retained_real=0)
    assert r.notes == ["final per-chain sequence composition differs from the real pairing"] and not r.valid
    r.check_final(3, 3, multiplicity_ok=True, composition_ok=True, retained_real=0)
    assert r.valid and r.final_composition_ok


def test_new_final_invariants_reach_the_metrics(host_pathogen):  # noqa: F811
    a, b, amap = host_pathogen
    m = next(CoevoPredictor([(a, b)], interaction_map=amap, pairing="permute-partners", seed=2, cpu=True).predict("dca")).metrics
    assert m.pairing_final_composition_ok and m.pairing_final_multiplicity_ok and m.pairing_final_retained_real == 0


# --- rejection rate is logged ---------------------------------------------------------------------------
def test_rejection_rate_is_logged_with_reasons(capsys):
    a = _gapped_msa("A", SPECIES, {i: 9 for i in range(1, 11)}, seed=1)
    b = _gapped_msa("B", SPECIES, {i: 9 for i in range(1, 11)}, seed=2)
    p = CoevoPredictor([(a, b), (a, b)], pairing="rewired", seed=4, max_gap_fraction=.5, cpu=True)
    err = capsys.readouterr().err
    assert p.rejection_rate == 1.0
    assert "Negative control 'rewired': 2/2 pairs rejected (100%)" in err and "multiplicity" in err
    assert "[WARN] Most 'rewired' controls were rejected" in err and "--allow-invalid-controls" in err


def test_rejection_log_is_silent_for_real_pairing_and_clean_controls(capsys):
    from yunta.structs.pairing import log_rejections
    log_rejections("map", 5, {})
    a = _gapped_msa("A", SPECIES, {}, seed=1); b = _gapped_msa("B", SPECIES, {}, seed=2)
    CoevoPredictor([(a, b)], pairing="map", cpu=True)
    assert "rejected" not in capsys.readouterr().err
    log_rejections("rewired", 4, {})
    assert "0/4 pairs rejected (0%)" in capsys.readouterr().err


def test_pair_with_no_paired_homologs_is_skipped_not_scored(capsys):
    """DCA is undefined for a single sequence; a screen must skip such a pair, not abort."""
    a = _gapped_msa("A", [1, 2, 3], {}, seed=1)
    b = _gapped_msa("B", [1, 7, 8], {}, seed=2)   # only the query species is shared
    assert _pair_msas(a, b) is None
    assert "no paired homologs" in capsys.readouterr().err
    assert len(CoevoPredictor([(a, b)], cpu=True)) == 0
