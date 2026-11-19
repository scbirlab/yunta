"""Regression tests for correctness and performance fixes."""

from functools import partial
from time import perf_counter

import numpy as np
import pytest

from yunta.interactions.scoring import score_contact_map
from yunta.structs.metrics import InteractionMetrics
from yunta.structs.msa import MSA, MSALine, PairedMSA, tokenize


def _line(uid, ox, os_, seq):
    return MSALine(
        name=f">sp|{uid}|{uid}_X",
        description=f"Gene OS={os_} OX={ox} GN=x PE=1 SV=1",
        sequence=seq,
    )


# --------------------------------------------------------------------------- Neff
def test_neff_clusters_at_identity_threshold():
    """Regression: the threshold was inverted, clustering at 38% instead of 62% identity."""
    base = "ACDEFGHIKLMNPQRSTVWY" * 2
    near = base[:-5] + "AAAAA"                       # 87.5% identical to base
    distant = "".join("W" if c != "W" else "Y" for c in base)  # 0% identical
    mid = base[:20] + "".join("W" if c != "W" else "Y" for c in base[20:])  # 50% identical
    msa = MSA([_line(f"P{i}", 9606, "Homo sapiens", s) for i, s in enumerate([base, near, distant, mid])])
    assert msa.neff(.62, verbose=False) == 3   # {base, near}, {distant}, {mid}
    assert msa.neff(.40, verbose=False) == 2   # mid (50%) now joins base's cluster
    assert msa.neff(1.0, verbose=False) == 4


def _best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return min(times)


def _naive_neff(seqs, threshold=.62):
    """Reference: the original per-pair Python loop (greedy clustering by identity)."""
    L = len(seqs[0])
    max_mismatch = L * (1 - threshold)
    reps = []
    for s in seqs:
        if not any(sum(a != b for a, b in zip(s, r)) <= max_mismatch for r in reps):
            reps.append(s)
    return len(reps)


def test_neff_is_vectorised_and_agrees_with_naive_reference():
    """Hardware-independent: compare with a naive loop on the same input instead of a wall-clock limit."""
    rng = np.random.default_rng(0)
    letters = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    seqs = ["".join(rng.choice(letters, 120)) for _ in range(400)]
    msa = MSA([_line(f"P{i}", 9606, "Homo sapiens", s) for i, s in enumerate(seqs)])
    assert msa.neff(verbose=False) == _naive_neff(seqs) == 400
    fast = _best_of(lambda: msa.neff(verbose=False))
    slow = _best_of(lambda: _naive_neff(seqs), repeats=1)
    assert fast * 5 < slow, f"vectorised Neff ({fast:.3f}s) is not clearly faster than the naive loop ({slow:.3f}s)"


# --------------------------------------------------------------------------- tokenisation
def test_tokenize_matches_alphabet_and_maps_unknown_to_gap():
    assert tokenize(["ARNDCQEGHILKMFPSTWYV-"])[0].tolist() == list(range(21))
    assert set(tokenize(["XBZUO.*0123456789"])[0].tolist()) == {20}


def test_msa_token_ids_are_uint8_array():
    msa = MSA([_line("P1", 9606, "Homo sapiens", "ACDE"), _line("P2", 9606, "Homo sapiens", "AC-E")])
    assert msa.sequence_token_ids.dtype == np.uint8
    assert msa.sequence_token_ids.shape == (2, 4)


# --------------------------------------------------------------------------- query row
def test_filters_never_drop_query_row():
    """Regression: an unknown-species query was silently dropped, making row 1 the 'reference'."""
    query = MSALine(name=">sp|Q1|Q_X", description="Gene GN=x", sequence="ACDE")  # no species
    other = _line("P2", 9606, "Homo sapiens", "ACDE")
    msa = MSA([query, other])
    assert msa.filter_by_known_species().lines[0] is query
    gappy_query = MSA([MSALine(name=">sp|Q1|Q_X", description="Gene GN=x", sequence="A---"), other])
    assert gappy_query.filter_by_gap_fraction(.5).lines[0].unique_id == "Q1"


def test_unknown_species_query_is_skipped_not_mispaired():
    from yunta.interactions.runner import _pair_msas
    query = MSALine(name=">sp|Q1|Q_X", description="Gene GN=x", sequence="ACDE")
    other = _line("P2", 9606, "Homo sapiens", "ACDE")
    partner = _line("P3", 9606, "Homo sapiens", "WYWY")
    assert _pair_msas(MSA([query, other]), MSA([partner])) is None


def test_filter_does_not_mutate_original():
    msa = MSA([_line("P1", 9606, "Homo sapiens", "ACDE"), _line("P2", 9606, "Homo sapiens", "A---")])
    filtered = msa.filter_by_gap_fraction(.5)
    assert len(msa) == 2 and len(filtered) == 1
    assert msa.sequence_token_ids.shape[0] == 2 and filtered.sequence_token_ids.shape[0] == 1


# --------------------------------------------------------------------------- pairing perf
def _make_species_msas(n_rows, n_species, seed=1):
    rng = np.random.default_rng(seed)
    species = rng.integers(0, n_species, n_rows)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 20)) for _ in range(n_rows)]
    def make(prefix):
        return MSA([_line(f"{prefix}{i}", int(sp) + 1, f"Genus species{sp}", s) for i, (sp, s) in enumerate(zip(species, seqs))])
    return make("A"), make("B")


def test_pairing_scales_linearly_not_quadratically_in_rows():
    """Regression: grouping recomputed the whole species set for every row (quadratic).

    Compares 4x the rows against 1x (same species density): linear work is ~4x, quadratic ~16x.
    Relative, so it does not depend on CI hardware.
    """
    small_a, small_b = _make_species_msas(1500, 400)
    big_a, big_b = _make_species_msas(6000, 1600)
    t_small = _best_of(lambda: PairedMSA.join_msa(small_a, small_b))
    t_big = _best_of(lambda: PairedMSA.join_msa(big_a, big_b))
    lines, _ = PairedMSA.join_msa(big_a, big_b)
    assert len(lines) > 1600 * 0.9
    assert t_big < 9 * t_small, f"4x rows took {t_big / t_small:.1f}x as long (expected ~4x, quadratic would be ~16x)"


def test_pairing_allows_species_name_fallback_only_when_map_says_so():
    host = _line("P0", 9606, "Homo sapiens", "ACDE")
    host2 = _line("P1", 10090, "Mus musculus", "ACDF")
    virus = _line("V0", 10376, "Epstein-Barr virus", "WYWY")
    virus2 = _line("V1", 10376, "Epstein-Barr virus", "WYWA")
    imap = {"NCBI:9606": ["NCBI:10376"]}  # mouse is not a listed host
    lines, _ = PairedMSA.join_msa(MSA([host, host2]), MSA([virus, virus2]), interaction_map=imap)
    assert [l.unique_id for l in lines] == ["P0-V0"]


# --------------------------------------------------------------------------- scoring
def test_top_indices_are_exactly_top_n_and_sign_fixed():
    M = np.zeros((6, 5)); M[4, 1] = 1.; M[2, 1] = .4; M[4, 3] = .3
    scores = score_contact_map(-M * -1, top_n=2)
    assert scores["top_A"].split(";")[0] == "4" and len(scores["top_A"].split(";")) == 2
    assert scores["top_B"].split(";")[0] == "1"
    # unchanged under negation of the SVD sign convention, i.e. no 'least important' residues
    assert score_contact_map(M, top_n=2)["top_A"] == scores["top_A"]


def test_square_interaction_block_diagonal_is_not_masked():
    """Regression: a constant diagonal in the A x B block was treated as 'self-pairs' and removed."""
    M = np.full((3, 3), 5.); M[0, 1] = 9.
    assert score_contact_map(M)["mean"] == pytest.approx(M.mean())


def test_empty_block_scores_do_not_raise():
    assert score_contact_map(np.zeros((0, 4)))["sigma1"] == -1.


# --------------------------------------------------------------------------- metrics
def test_metric_ids_with_hyphens_are_split_sensibly():
    kw = dict(seq_len=1, chain_a_len=1, chain_b_len=1, msa1_depth=1, msa2_depth=1, msa_depth=1,
              n_eff=1, mean=0., median=0., maximum=0., minimum=0., var=0., sigma1=0., focality=0.,
              top_A="", top_B="")
    m = InteractionMetrics(ID="P12345-2-Q99999-1", **kw)
    assert (m.uniprot_id_1, m.uniprot_id_2) == ("P12345-2", "Q99999-1")


def test_paired_ids_property_handles_isoforms():
    msa1 = MSA([_line("P12345-2", 9606, "Homo sapiens", "ACDE")])
    msa2 = MSA([_line("Q99999-1", 9606, "Homo sapiens", "WYWY")])
    paired = PairedMSA.from_msa(msa1, msa2)
    assert paired.ids == ("P12345-2", "Q99999-1")


# --------------------------------------------------------------------------- screening loop
class _FakeRunner:
    made = 0
    runs = []

    def make_model(self, **kwargs):
        type(self).made += 1
        return object()

    def run(self, msa1, msa2=None, **kwargs):
        type(self).runs.append((msa1.name, None if msa2 is None else msa2.name, kwargs))
        return np.zeros((1, 1)), np.zeros((1, 1)), object()


def _fake_msas(n):
    return [MSA([_line(f"P{i}", 9606, "Homo sapiens", "ACDE")]) for i in range(n)]


def test_model_is_built_once_and_pairs_are_unordered():
    from yunta.screening import _screen_many_vs_many, _screen_one_vs_many
    _FakeRunner.made, _FakeRunner.runs = 0, []
    msas = _fake_msas(4)
    _screen_many_vs_many(partial(_screen_one_vs_many, _FakeRunner), msas, None, runner=_FakeRunner)
    assert _FakeRunner.made == 1
    pairs = {(a, b) for a, b, _ in _FakeRunner.runs}
    assert len(_FakeRunner.runs) == 4 * 5 // 2            # i <= j, not 4 * 4
    assert all((b, a) not in pairs for a, b in pairs if a != b)
    assert sum(b is None for _, b, _ in _FakeRunner.runs) == 4  # homodimers use msa2=None


def test_strict_match_is_forwarded_to_the_runner():
    """Regression: `enforce_ref_match` was accepted but never passed on, so --strict-match did nothing."""
    from yunta.screening import _screen_one_vs_many
    _FakeRunner.made, _FakeRunner.runs = 0, []
    a, b = _fake_msas(2)
    _screen_one_vs_many(_FakeRunner, a, [b], enforce_ref_match=True)
    assert _FakeRunner.runs[0][2]["enforce_ref_match"] is True
