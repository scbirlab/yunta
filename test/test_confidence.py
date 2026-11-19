import numpy as np
import pytest

from yunta.interactions import confidence as c


def _complex(len_a=30, len_b=20, good=True, seed=0):
    rng = np.random.default_rng(seed)
    n = len_a + len_b
    pae = np.full((n, n), 4.)
    inter = 3. if good else 28.
    pae[:len_a, len_a:] = pae[len_a:, :len_a] = inter
    pae += rng.normal(0, .1, (n, n)).clip(-.5, .5)
    return pae.clip(.2, 31.75)


def test_confident_interface_scores_higher_on_every_pae_metric():
    good, bad = _complex(good=True), _complex(good=False)
    assert c.ipsae(good, 30) > 0.1 > c.ipsae(bad, 30)
    assert c.lis(good, 30) > c.lis(bad, 30)
    assert c.mean_interchain_pae(good, 30) < c.mean_interchain_pae(bad, 30)
    dists = np.full((50, 50), 30.); dists[:30, 30:] = dists[30:, :30] = 5.
    plddt = np.full(50, 90.)
    assert c.pdockq2(good, plddt, dists, 30) > c.pdockq2(bad, plddt, dists, 30)


def test_missing_pae_gives_nan_not_zero():
    for fn in (c.ipsae, c.lis, c.mean_interchain_pae):
        assert np.isnan(fn(None, 5))
    assert np.isnan(c.ptm(None)) and np.isnan(c.iptm(None, 5))


def test_ipsae_matches_hand_calculation():
    pae = np.full((4, 4), 30.)
    pae[0, 2] = pae[0, 3] = 2.  # residue 0 of A sees both of B at PAE 2
    expected_d0 = 1.0  # n0=2 is below the reference code's floor, so d0 is clamped to 1
    expected = 1. / (1. + (2. / expected_d0) ** 2)
    assert c.ipsae(pae, 2, pae_cutoff=10) == pytest.approx(expected)


def test_pdockq_reference_values():
    assert c.pdockq(.1, 3) == pytest.approx(0.018259536451737016)
    assert c.pdockq(.9, 100) == pytest.approx(0.018284286329741297)
    assert c.pdockq(90, 0) == 0.


def test_tm_scores_from_probabilities():
    n, bins = 40, 64
    centers = np.linspace(0, 31.5, bins)
    probs = np.zeros((n, n, bins)); probs[..., 0] = 1.  # certain, ~zero error
    tm = c.tm_matrix(probs, centers, n)
    assert c.ptm(tm) == pytest.approx(1., abs=1e-3) and c.iptm(tm, 25) == pytest.approx(1., abs=1e-3)
    probs[..., 0] = 0; probs[..., -1] = 1.  # certain, max error
    low = c.tm_matrix(probs, centers, n)
    assert c.iptm(low, 25) < .1


def test_actifptm_weights_by_contact_probability():
    tm = np.full((6, 6), .1); tm[:3, 3:] = tm[3:, :3] = .9; tm[0, 3] = .1
    focus = np.zeros((3, 3)); focus[1, 1] = 1.
    # weighting on a confident contact pair gives a high score even if other pairs are poor
    assert c.actifptm(tm, focus, 3) == pytest.approx(.9)


def test_score_complex_without_pae():
    dists = np.full((10, 10), 30.); dists[:4, 4:] = dists[4:, :4] = 6.
    s = c.score_complex(np.full(10, 80.), dists, 4)
    assert s["n_contacts"] == 24 and s["mean_interface_plddt"] == pytest.approx(80.)
    assert np.isnan(s["ipsae"]) and 0 < s["pdockq"] < 1


def test_assign_chains_and_structure_roundtrip():
    from yunta.io import assign_chains
    rec = "ATOM  {i:5d}  CA  GLY A{r:4d}      11.104   6.134  -6.504  1.00 90.00           C"
    pdb = "\n".join(rec.format(i=r, r=r) for r in range(1, 6))
    out = assign_chains(pdb, 3).split("\n")
    assert [line[21] for line in out] == list("AAABB")


def test_af2_compact_reduces_probabilities_to_matrices():
    from yunta.interactions.af2.run import _compact
    n, bins = 12, 64
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(bins), size=(n, n)).astype(np.float32)
    logits = np.log(rng.dirichlet(np.ones(bins), size=(n, n)))
    result = {
        "plddt": np.full(n, 80.), "ptm": .4,
        "predicted_aligned_error": np.full((n, n), 6.),
        "aligned_confidence_probs": probs,
        "distogram": {"logits": logits, "bin_edges": np.linspace(2.3125, 21.6875, bins - 1)},
    }
    out = _compact(result)
    assert out["pae"].shape == out["tm"].shape == out["contact_prob_full"].shape == (n, n)
    assert ((out["tm"] > 0) & (out["tm"] < 1)).all() and (out["contact_prob_full"] <= 1).all()
    assert 0 <= c.iptm(out["tm"], 5) <= 1


# ------------------------------------------------------------------ reference implementations
# Independent, deliberately naive transcriptions of the published/reference code, compared
# with the vectorised implementations on random inputs.
def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _random_prediction(n=40, len_a=25, seed=0, bins=64):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n, bins)) * 2
    centers = np.linspace(0, 31, bins - 1)
    step = centers[1] - centers[0]
    bin_centers = np.concatenate([centers + step / 2, [centers[-1] + step * 1.5]])
    probs = _softmax(logits)
    pae = (probs * bin_centers).sum(-1)
    return logits, centers, bin_centers, probs, pae, len_a


def test_ptm_matches_vendored_alphafold_reference():
    from yunta.interactions.af2.src_speedppi.alphafold.confidence import predicted_tm_score
    logits, breaks, centers, probs, _, _ = _random_prediction()
    ref = float(predicted_tm_score(logits, breaks))
    assert c.ptm(c.tm_matrix(probs, centers, logits.shape[0])) == pytest.approx(ref, rel=1e-6)


def test_iptm_matches_alphafold_multimer_reference():
    """AF-Multimer ``predicted_tm_score(..., asym_id, interface=True)`` transcribed."""
    logits, breaks, centers, probs, _, len_a = _random_prediction(seed=1)
    n = logits.shape[0]
    asym = np.array([0] * len_a + [1] * (n - len_a))
    weights = np.ones(n)
    d0 = 1.24 * (max(weights.sum(), 19) - 15) ** (1 / 3) - 1.8
    term = (probs * (1 / (1 + centers ** 2 / d0 ** 2))).sum(-1)
    pair_mask = asym[:, None] != asym[None, :]
    term = term * pair_mask
    pair_w = pair_mask * (weights[None, :] * weights[:, None])
    normed = pair_w / (1e-8 + pair_w.sum(-1, keepdims=True))
    per_alignment = (term * normed).sum(-1)
    ref = float(per_alignment[(per_alignment * weights).argmax()])
    assert c.iptm(c.tm_matrix(probs, centers, n), len_a) == pytest.approx(ref, rel=1e-6)


def _ref_chain_scores(pae, plddt, dists, len_a, pae_cutoff=10., dist_cutoff=8.):
    """Loop transcription of ipsae.py's pDockQ2, LIS and ipSAE (d0res), per direction."""
    n = pae.shape[0]
    chain = np.array([0] * len_a + [1] * (n - len_a))
    out = {}
    for c1, c2 in ((0, 1), (1, 0)):
        idx1, idx2 = np.where(chain == c1)[0], np.where(chain == c2)[0]
        # pDockQ2
        s, npairs, uniq = 0., 0, set()
        for i in idx1:
            near = [j for j in idx2 if dists[i, j] <= dist_cutoff]
            if near:
                uniq.add(i); uniq.update(near)
                s += sum(1 / (1 + (pae[i, j] / 10.) ** 2) for j in near)
                npairs += len(near)
        if npairs:
            x = np.mean([plddt[i] for i in uniq]) * (s / npairs)
            pdq2 = 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005
        else:
            pdq2 = 0.
        # LIS
        vals = [pae[i, j] for i in idx1 for j in idx2 if pae[i, j] < 12]
        lis = float(np.mean([(12 - v) / 12 for v in vals])) if vals else 0.
        # ipSAE d0res
        best = 0.
        for i in idx1:
            valid = [j for j in idx2 if pae[i, j] < pae_cutoff]
            if valid:
                d0 = max(1., 1.24 * (max(26, len(valid)) - 15) ** (1 / 3) - 1.8)
                best = max(best, float(np.mean([1 / (1 + (pae[i, j] / d0) ** 2) for j in valid])))
        out[(c1, c2)] = (pdq2, lis, best)
    return out


@pytest.mark.parametrize("seed", range(4))
def test_pdockq2_lis_ipsae_match_loop_reference(seed):
    rng = np.random.default_rng(seed)
    n, len_a = 36, 20
    pae = rng.uniform(0.5, 31, (n, n)); plddt = rng.uniform(40, 98, n)
    coords = rng.normal(scale=7, size=(n, 3))
    dists = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    ref = _ref_chain_scores(pae, plddt, dists, len_a)
    assert c.pdockq2(pae, plddt, dists, len_a) == pytest.approx(np.mean([ref[(0, 1)][0], ref[(1, 0)][0]]))
    assert c.lis(pae, len_a) == pytest.approx(np.mean([ref[(0, 1)][1], ref[(1, 0)][1]]))
    assert c.ipsae(pae, len_a) == pytest.approx(max(ref[(0, 1)][2], ref[(1, 0)][2]))


# ----------------------------------------------------------------------------------------- one path for every backend
from types import SimpleNamespace
from yunta.interactions import confidence


def _record(n=12, len_a=7, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.random((n, n, confidence.PAE_N_BINS)).astype(np.float32) ** 4
    probs /= probs.sum(-1, keepdims=True)
    contact = rng.random((n, n)).astype(np.float32)
    return confidence.compact_prediction(rng.uniform(40, 95, n), pae_probs=probs, contact_prob=contact, ptm=.5), len_a


def _whole_results(record, n, len_a):
    from scipy.spatial.distance import cdist
    from yunta.interactions.chunking import plan_blocks
    coords = np.cumsum(np.random.default_rng(1).normal(scale=3., size=(n, 3)), axis=0)
    (block,) = plan_blocks(len_a, n - len_a, chunksize=None)
    return [(block, (cdist(coords, coords), record))], SimpleNamespace(seq_length=n, chain_a_length=len_a)


def test_af2_and_openfold3_scores_come_from_the_same_code():
    """Identical canonical records give identical scores whichever backend's ``post_*`` is used."""
    pytest.importorskip("biotite")
    from yunta.interactions.af2.run import post_af2
    from yunta.interactions.of3.run import post_of3
    record, len_a = _record()
    results, paired = _whole_results(record, 12, len_a)
    af2, of3 = post_af2(paired, results), post_of3(paired, [(b, (d, dict(record, iptm_model=.3, ptm_model=.4))) for b, (d, _) in results])
    shared = ["n_contacts", "mean_plddt_a", "mean_plddt_b", "mean_interface_plddt", "pdockq", "pdockq2",
              "ptm", "iptm", "actifptm", "ipsae", "lis", "mean_interchain_pae"]
    for key in shared:
        assert af2[key] == pytest.approx(of3[key], nan_ok=True), key
    assert not any(np.isnan(af2[k]) for k in shared)
    assert af2["ptm_af2"] == .5 and of3["ptm_model"] == .4 and of3["iptm_model"] == .3  # model-specific values stay apart
    direct, _ = confidence.score_blocks(paired, results)
    assert direct["iptm"] == af2["iptm"]


def test_chunked_blocks_give_nan_for_tm_scores_in_both_backends():
    from yunta.interactions.chunking import plan_blocks
    from yunta.interactions.of3.run import post_of3
    pytest.importorskip("biotite")
    record, len_a = _record()
    n = 12
    blocks = plan_blocks(len_a, n - len_a, chunksize=8, blocks="interchain")
    assert len(blocks) > 1
    results = []
    for b in blocks:
        cols = b.columns()
        sub = confidence.compact_prediction(
            np.full(len(cols), 80.), pae_probs=record["tm"][np.ix_(cols, cols)][..., None] * 0 + 1. / 64 * np.ones(64))
        results.append((b, (np.full((len(cols), len(cols)), 5.), sub)))
    out = post_of3(SimpleNamespace(seq_length=n, chain_a_length=len_a), results)
    assert np.isnan(out["ptm"]) and np.isnan(out["iptm"]) and np.isnan(out["actifptm"])
    assert not np.isnan(out["ipsae"])  # blockwise approximation


def test_compact_prediction_rejects_unit_mistakes():
    n = 4
    good = np.full((n, n, confidence.PAE_N_BINS), 1. / confidence.PAE_N_BINS)
    confidence.compact_prediction(np.full(n, 80.), pae_probs=good)
    with pytest.raises(ValueError, match="0-100"):
        confidence.compact_prediction(np.full(n, .8))                       # pLDDT as a fraction
    with pytest.raises(ValueError, match="sum to 1"):
        confidence.compact_prediction(np.full(n, 80.), pae_probs=np.log(good))   # logits, not probabilities
    with pytest.raises(ValueError, match="expected"):
        confidence.compact_prediction(np.full(n, 80.), pae_probs=good[..., :32])  # wrong bin count
    with pytest.raises(ValueError, match="Angstrom"):
        confidence.compact_prediction(np.full(n, 80.), pae=np.full((n, n), 300.))  # not Angstrom
    with pytest.raises(ValueError, match="contact"):
        confidence.compact_prediction(np.full(n, 80.), contact_prob=np.full((n, n), 5.))


def test_pae_bin_centres_are_the_same_for_af2_and_openfold3():
    """AF2: 63 edges over 0-31 A; OpenFold3: 64 bins over 0-32 A. Both give centres 0.25 ... 31.75."""
    assert confidence.PAE_BIN_CENTERS[0] == .25 and confidence.PAE_BIN_CENTERS[-1] == 31.75
    pytest.importorskip("jax")
    from yunta.interactions.af2.src_speedppi.alphafold.confidence import _calculate_bin_centers
    assert np.allclose(_calculate_bin_centers(np.linspace(0., 31., 63)), confidence.PAE_BIN_CENTERS)


def test_expected_pae_matches_the_probability_weighted_mean():
    probs = np.zeros((2, 2, confidence.PAE_N_BINS)); probs[..., 0] = .5; probs[..., 63] = .5
    assert confidence.expected_pae(probs)[0, 0] == pytest.approx(16.)
