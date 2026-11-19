import numpy as np
import pytest

from yunta.structs.msa import MSA, MSALine, PairedMSA
from yunta.structs.pairing import rewire_bipartite


def _msa(prefix, species_ids, length=12, seed=0):
    rng = np.random.default_rng(seed)
    letters = np.array(list("ARNDCQEGHILKMFPSTWYV"))
    lines = []
    for r, sp in enumerate(species_ids):
        seq = "".join(letters[rng.integers(0, 20, length)])
        lines.append(MSALine(
            name=f">sp|{prefix}{r}|{prefix}{r}_X",
            description=f"Gene OS=Species {sp} OX={sp} GN=x PE=1 SV=1",
            sequence=seq,
        ))
    return MSA(lines)


def _species_of(paired):
    return [tuple(d.species_id for d in line.description) for line in paired.lines[1:]]


@pytest.fixture
def msas():
    sp = [100 + i for i in range(40)]  # query species first
    return _msa("A", sp, seed=1), _msa("B", sp, seed=2)


def test_map_pairing_is_same_species(msas):
    paired = PairedMSA.from_msa(*msas)
    assert all(a == b for a, b in _species_of(paired))


@pytest.mark.parametrize("pairing", ["rewired", "random-species", "random-rows"])
def test_controls_have_same_depth_and_break_known_pairs(msas, pairing):
    real = PairedMSA.from_msa(*msas)
    ctrl = PairedMSA.from_msa(*msas, pairing=pairing, seed=3)
    assert ctrl.seq_length == real.seq_length and ctrl.chain_a_length == real.chain_a_length
    assert ctrl.lines[0].sequence == real.lines[0].sequence  # query row preserved
    assert len(ctrl) == len(real)
    same = sum(a == b for a, b in _species_of(ctrl))
    assert same < len(ctrl) - 1  # most rows no longer same-species


def test_controls_are_reproducible_and_seed_dependent(msas):
    a = PairedMSA.from_msa(*msas, pairing="rewired", seed=5)
    b = PairedMSA.from_msa(*msas, pairing="rewired", seed=5)
    c = PairedMSA.from_msa(*msas, pairing="rewired", seed=6)
    assert a.sequences() == b.sequences()
    assert a.sequences() != c.sequences()


def test_unknown_pairing_rejected(msas):
    with pytest.raises(ValueError):
        PairedMSA.from_msa(*msas, pairing="nope")


def test_rewire_preserves_degrees_and_avoids_known_edges():
    rng = np.random.default_rng(0)
    edges = {(f"a{i}", f"b{j}") for i in range(8) for j in range(8) if (i + j) % 3 == 0}
    new, kept = rewire_bipartite(edges, rng)
    assert len(new) == len(edges)
    for k in (0, 1):
        assert sorted(e[k] for e in new) == sorted(e[k] for e in edges)
    assert kept == len(new & edges)
