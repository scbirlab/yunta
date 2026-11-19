"""Chunked / blocked inference: planning, chain breaks, stitching and DCA."""

from itertools import product

import numpy as np
import pytest

from yunta.interactions.chunking import (
    Block, Segment, assemble_matrix, assemble_vector, block_input, plan_blocks, run_blocks, split_evenly,
)

LAYOUTS = [(5, 4, 6), (40, 30, 30), (100, 7, 24), (7, 100, 24), (300, 300, 100), (751, 10, 750), (10, 751, 750), (3, 3, 4)]


@pytest.mark.parametrize("len_a,len_b,chunksize", LAYOUTS)
@pytest.mark.parametrize("blocks", ["all", "interchain"])
def test_plan_covers_every_required_pair_within_budget(len_a, len_b, chunksize, blocks):
    plan = plan_blocks(len_a, len_b, chunksize=chunksize, blocks=blocks)
    n = len_a + len_b
    assert max(len(b) for b in plan) <= max(chunksize, 2) or plan[0].is_whole
    covered = np.zeros((n, n), dtype=int)
    for block in plan:
        cols = block.columns()
        if block.is_pair:
            first, second = block.segments
            covered[first.slice, second.slice] += 1
            covered[second.slice, first.slice] += 1
        else:
            covered[np.ix_(cols, cols)] += 1
    wanted = np.zeros((n, n), dtype=bool)
    wanted[:len_a, len_a:] = wanted[len_a:, :len_a] = True
    if blocks == "all":
        wanted[:] = True
    assert np.all(covered[wanted] >= 1)
    if blocks == "all":
        assert np.all(covered == 1), "every residue pair should come from exactly one block"


@pytest.mark.parametrize("len_a,len_b,chunksize", LAYOUTS)
def test_no_block_straddles_the_chain_boundary_and_break_is_correct(len_a, len_b, chunksize):
    """Regression: the break used to be ``chain_a_length - i0``, which put the chain break
    beyond the block (no break at all) for cross-chunk blocks when chain A was longer than a chunk."""
    for block in plan_blocks(len_a, len_b, chunksize=chunksize):
        cols = block.columns()
        if block.is_whole:
            assert block.break_at == len_a
            continue
        before, after = cols[:block.break_at], cols[block.break_at:]
        for part in (before, after):
            if len(part):
                assert (part < len_a).all() or (part >= len_a).all()
        if block.is_pair:
            assert len(before) == len(block.segments[0]) and len(after) == len(block.segments[1])
        else:
            assert block.break_at == len(block)


@pytest.mark.parametrize("length,max_size", [(751, 750), (1501, 750), (10, 3), (1, 5), (0, 5)])
def test_split_is_balanced_with_no_tiny_remainder(length, max_size):
    """Regression: np.split every ``chunksize`` left e.g. a 1-column chunk for 751 columns."""
    segments = split_evenly(length, max_size)
    assert sum(len(s) for s in segments) == length
    if len(segments) > 1:
        sizes = [len(s) for s in segments]
        assert max(sizes) <= max_size and max(sizes) - min(sizes) <= 1


def _identity_alignment(n_cols, n_rows=3):
    return np.tile(np.arange(n_cols, dtype=np.int32), (n_rows, 1))


@pytest.mark.parametrize("len_a,len_b,chunksize", LAYOUTS)
def test_stitched_matrix_equals_unchunked_for_a_pairwise_function(len_a, len_b, chunksize):
    n = len_a + len_b
    seen_breaks = []

    def fn(sub, chain_a_length):
        seen_breaks.append(chain_a_length)
        cols = sub[0].astype(np.int64)
        return (1000 * cols[:, None] + cols[None, :]).astype(np.float32), {"n": len(cols)}

    full, extras = run_blocks(_identity_alignment(n), len_a, fn, chunksize=chunksize, progress=False)
    truth = (1000 * np.arange(n)[:, None] + np.arange(n)[None, :]).astype(np.float32)
    assert np.array_equal(full, truth)
    assert len(extras) == len(plan_blocks(len_a, len_b, chunksize))

    inter, _ = run_blocks(_identity_alignment(n), len_a, fn, chunksize=chunksize, blocks="interchain", progress=False)
    assert np.array_equal(inter[:len_a, len_a:], truth[:len_a, len_a:])
    assert np.array_equal(inter[len_a:, :len_a], truth[len_a:, :len_a])


def test_interchain_mode_runs_fewer_blocks():
    assert len(plan_blocks(300, 300, 100, "interchain")) < len(plan_blocks(300, 300, 100, "all"))
    assert not any(b.intra_chain() for b in plan_blocks(300, 300, 100, "interchain"))


def test_vector_assembly_is_order_independent():
    plan = plan_blocks(6, 6, chunksize=6)
    vectors = [(b, np.random.default_rng(i).random(len(b))) for i, b in enumerate(plan)]
    forward = assemble_vector(vectors, 12)
    backward = assemble_vector(vectors[::-1], 12)
    assert np.allclose(forward, backward)
    assert not np.isnan(forward).any()


# ----------------------------------------------------------------------------------------- DCA
torch = pytest.importorskip("torch")


def _synthetic_paired_msas(n_rows=400, len_a=40, len_b=30, coupled=((7, 5),), seed=0):
    from yunta.structs.msa import MSA, MSALine
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 20, (n_rows, len_a))
    b = rng.integers(0, 20, (n_rows, len_b))
    for i, j in coupled:
        b[:, j] = (a[:, i] * 7 + 3) % 20
    letters = np.array(list("ARNDCQEGHILKMFPSTWYV"))

    def build(tokens, prefix):
        return MSA([
            MSALine(
                name=f">sp|{prefix}{r}|{prefix}{r}_X",
                description=f"Gene OS=Species {r} OX={1000 + r} GN=x PE=1 SV=1",
                sequence="".join(letters[tokens[r]]),
            )
            for r in range(len(tokens))
        ])
    return build(a, "A"), build(b, "B")


def test_chunked_dca_recovers_planted_coupling_like_unchunked():
    from yunta.interactions.runner import DCARunner
    msa1, msa2 = _synthetic_paired_msas()
    _, unchunked, m_full = DCARunner().run(msa1, msa2, chunksize=1000, apc=True, cpu=True)
    _, chunked, m_chunk = DCARunner().run(msa1, msa2, chunksize=30, apc=True, cpu=True)
    assert np.unravel_index(np.argmax(unchunked), unchunked.shape) == (7, 5)
    assert np.unravel_index(np.argmax(chunked), chunked.shape) == (7, 5)
    assert m_chunk.top_A.split(";")[0] == "7" and m_chunk.top_B.split(";")[0] == "5"
    assert m_full.n_eff == m_chunk.n_eff


def test_apc_is_global_not_per_block():
    """With APC applied per block, entries of the assembled matrix would not satisfy the
    global APC identity. Check the runner's output equals APC(raw assembled matrix)."""
    from yunta.interactions.dca.dca_torch import apc_correction
    from yunta.interactions.runner import DCARunner
    msa1, msa2 = _synthetic_paired_msas(n_rows=200, len_a=20, len_b=16)
    raw, *_ = DCARunner().run(msa1, msa2, chunksize=18, apc=False, cpu=True)
    corrected, *_ = DCARunner().run(msa1, msa2, chunksize=18, apc=True, cpu=True)
    assert np.allclose(corrected, apc_correction(raw), atol=1e-5)
    assert np.all(np.diag(corrected) == 0)


# ----------------------------------------------------------------------------------------- AF2 stitching
@pytest.mark.parametrize("blocks", ["all", "interchain"])
def test_af2_post_scoring_matches_unchunked_geometry(blocks):
    from scipy.spatial.distance import cdist
    from yunta.interactions.af2.run import post_af2
    from yunta.structs.msa import PairedMSA

    len_a, len_b = 35, 25
    msa1, msa2 = _synthetic_paired_msas(n_rows=20, len_a=len_a, len_b=len_b, coupled=())
    paired = PairedMSA.from_msa(msa1, msa2)
    rng = np.random.default_rng(3)
    coords = np.cumsum(rng.normal(scale=3., size=(len_a + len_b, 3)), axis=0)
    dist = cdist(coords, coords)
    plddt = rng.uniform(50, 95, len_a + len_b)

    def block_result(block):
        cols = block.columns() if not block.is_whole else np.arange(len_a + len_b)
        return (block, (dist[np.ix_(cols, cols)], {"plddt": plddt[cols], "ptm": .5}))

    expected_contacts = int((dist[:len_a, len_a:] <= 8.).sum())
    assert expected_contacts > 0
    for chunksize in (1000, 30, 20):
        results = [block_result(b) for b in plan_blocks(len_a, len_b, chunksize, blocks)]
        info = post_af2(paired, results)
        assert info["n_contacts"] == expected_contacts
        assert 0. < info["pdockq"] < 1.
        assert (info["ptm"] == .5) == (chunksize == 1000)  # whole-complex-only score
