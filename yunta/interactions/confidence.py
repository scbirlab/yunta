"""Backend-agnostic confidence scores for a predicted two-chain complex.

Every function takes plain arrays (PAE, pLDDT, Cβ distances), so the same scores serve
AlphaFold2 and any other backend that provides them (e.g. OpenFold3). Chain A occupies
rows/columns ``[:len_a]`` and chain B ``[len_a:]``.

Definitions follow the reference implementations:

* **pDockQ** — Bryant, Pozzati & Elofsson, *Nat Commun* 2022:
  ``0.724 / (1 + exp(-0.052 (x - 152.611))) + 0.018``, ``x = mean interface pLDDT * log10(n contacts)``.
* **pDockQ2** — Zhu, Shenoy, Kundrotas & Elofsson, *Bioinformatics* 2023 (constants and
  ``d0 = 10`` Å from the paper; 8 Å Cβ interface cutoff as in Dunbrack's ``ipsae.py``):
  ``1.31 / (1 + exp(-0.075 (x - 84.733))) + 0.005`` with
  ``x = mean interface pLDDT * mean(1 / (1 + (PAE/d0)^2))`` over contacting pairs.
* **ipTM / pTM** — Chen et al. / AF-Multimer: the predicted-TM term ``E[1/(1+(e/d0)^2)]`` from the PAE
  bin probabilities with ``d0 = 1.24 (L-15)^(1/3) - 1.8`` (L>=19 for pTM, total residues);
  ipTM restricts the expectation to the *other* chain and takes the best alignment residue.
* **ipSAE** — Dunbrack, bioRxiv 2025 (doi 10.1101/2025.02.10.637595): for every residue of one chain,
  the mean of ``1/(1+(PAE/d0)^2)`` over residues of the other chain with ``PAE < cutoff``, with ``d0``
  set by that residue's partner count (``L`` floored at 26, ``d0`` floored at 1, as in the
  reference array code); the maximum over residues, and the larger of the two directions.
* **LIS** — Kim et al., 2024: mean of ``(12 - PAE)/12`` over inter-chain pairs with ``PAE < 12``
  (strict, as in ``ipsae.py``).

PAE-derived scores need the PAE of the *whole* complex. When a complex is split into chunks
(see :mod:`yunta.interactions.chunking`) that matrix is never computed, so callers pass
``None`` and these scores are ``NaN`` rather than a misleading approximation.

Examples
========
>>> import numpy as np
>>> pae = np.full((6, 6), 20.); pae[:3, 3:] = pae[3:, :3] = 2.
>>> round(lis(pae, 3), 3)
0.833
>>> round(ipsae(pae, 3, pae_cutoff=10), 3)  # few partners => small d0 => modest score
0.2
>>> bool(np.isnan(ipsae(None, 3)))
True
"""

import numpy as np

NAN = float("nan")


def d0_tm(n: float, floor: float = 19., minimum: float | None = None) -> float:
    """TM-score normalisation length ``1.24 (n-15)^(1/3) - 1.8`` with ``n`` floored.

    >>> round(d0_tm(100.), 3)
    3.652
    >>> d0_tm(5., minimum=1.)
    1.0
    """
    d0 = 1.24 * (max(float(n), floor) - 15.) ** (1. / 3.) - 1.8
    return max(d0, minimum) if minimum is not None else d0


def _tm_term(pae: np.ndarray, d0) -> np.ndarray:
    return 1. / (1. + np.square(pae / d0))


def _split(m: np.ndarray, len_a: int):
    """Inter-chain blocks A->B (rows A, cols B) and B->A."""
    return m[:len_a, len_a:], m[len_a:, :len_a]


def pdockq(mean_interface_plddt: float, n_contacts: int) -> float:
    """pDockQ (Bryant 2022)."""
    if n_contacts < 1:
        return 0.
    x = mean_interface_plddt * np.log10(n_contacts)
    return float(.724 / (1 + np.exp(-.052 * (x - 152.611))) + .018)


def interface_residues(contact_dists: np.ndarray, len_a: int, cutoff: float = 8.):
    """Boolean masks of chain-A and chain-B residues with an inter-chain Cβ within ``cutoff``."""
    inter = contact_dists[:len_a, len_a:] <= cutoff
    return inter.any(axis=1), inter.any(axis=0), inter


def pdockq2(
    pae: np.ndarray | None,
    plddt: np.ndarray,
    contact_dists: np.ndarray,
    len_a: int,
    cutoff: float = 8.,
    d0: float = 10.,
) -> float:
    """pDockQ2 (Zhu 2023), symmetrised as the mean of the A->B and B->A scores."""
    if pae is None:
        return NAN
    scores = []
    for direction in ("ab", "ba"):
        dist = contact_dists[:len_a, len_a:] if direction == "ab" else contact_dists[len_a:, :len_a]
        pae_d = pae[:len_a, len_a:] if direction == "ab" else pae[len_a:, :len_a]
        near = dist <= cutoff
        if not near.any():
            scores.append(0.)
            continue
        rows, cols = near.any(axis=1), near.any(axis=0)
        pl_a, pl_b = (plddt[:len_a], plddt[len_a:]) if direction == "ab" else (plddt[len_a:], plddt[:len_a])
        mean_plddt = np.concatenate([pl_a[rows], pl_b[cols]]).mean()
        mean_tm = _tm_term(pae_d[near], d0).mean()
        x = mean_plddt * mean_tm
        scores.append(1.31 / (1. + np.exp(-.075 * (x - 84.733))) + .005)
    return float(np.mean(scores))


def lis(pae: np.ndarray | None, len_a: int, cutoff: float = 12.) -> float:
    """Local Interaction Score (Kim 2024), mean of the two directions."""
    if pae is None:
        return NAN
    out = []
    for block in _split(pae, len_a):
        valid = block[block < cutoff]
        out.append(float(((cutoff - valid) / cutoff).mean()) if valid.size else 0.)
    return float(np.mean(out))


def ipsae(pae: np.ndarray | None, len_a: int, pae_cutoff: float = 10.) -> float:
    """ipSAE (Dunbrack 2025), the larger of the A->B and B->A scores (``d0res`` variant)."""
    if pae is None:
        return NAN
    best = []
    for block in _split(pae, len_a):
        if block.size == 0:
            best.append(0.)
            continue
        valid = block < pae_cutoff
        n0 = valid.sum(axis=1)
        d0 = np.maximum(1., 1.24 * (np.maximum(26., n0) - 15.) ** (1. / 3.) - 1.8)
        tm = _tm_term(block, d0[:, None])
        per_res = np.where(n0 > 0, (tm * valid).sum(axis=1) / np.maximum(n0, 1), 0.)
        best.append(float(per_res.max()))
    return float(max(best))


def mean_interchain_pae(pae: np.ndarray | None, len_a: int) -> float:
    """Mean PAE over inter-chain pairs (both directions); lower is better."""
    if pae is None:
        return NAN
    return float(np.mean([b.mean() for b in _split(pae, len_a) if b.size] or [NAN]))


def tm_matrix(probs: np.ndarray, bin_centers: np.ndarray, n_res: int) -> np.ndarray:
    """``E[1/(1+(e/d0)^2)]`` per residue pair from PAE bin probabilities (``d0`` from ``n_res``).

    The result feeds :func:`ptm` and :func:`iptm`; keeping this ``(L, L)`` matrix instead of
    the ``(L, L, 64)`` probabilities is what makes per-block storage cheap.
    """
    d0 = d0_tm(n_res)
    per_bin = 1. / (1. + np.square(bin_centers) / d0 ** 2)
    return (probs * per_bin).sum(axis=-1)


def ptm(tm: np.ndarray | None) -> float:
    """pTM from :func:`tm_matrix`: best alignment residue, average over all residues."""
    if tm is None:
        return NAN
    return float(tm.mean(axis=-1).max())


def iptm(tm: np.ndarray | None, len_a: int) -> float:
    """ipTM (AF-Multimer): per alignment residue, average over the *other* chain; best residue.

    For two chains this is the larger of the A->B and B->A values.
    """
    if tm is None:
        return NAN
    ab, ba = _split(tm, len_a)
    vals = [b.mean(axis=1).max() for b in (ab, ba) if b.size]
    return float(max(vals)) if vals else NAN


def actifptm_contacts(distogram_probs: np.ndarray | None, bin_edges: np.ndarray | None,
                      len_a: int, contact_cutoff: float = 8.) -> np.ndarray | None:
    """Inter-chain contact probability ``P(d < cutoff)`` per residue pair from the distogram.

    ``bin_edges`` are the upper edges of all bins but the last (AF2: 63 edges for 64 bins), so
    bins ``0..n-1`` with ``edge <= cutoff`` lie wholly inside the cutoff. Returns the
    ``(len_a, len_b)`` matrix, or ``None`` without a distogram. These are the weights of
    actifpTM (Varga et al. 2025).

    >>> p = np.zeros((2, 2, 4)); p[..., 0] = .5; p[..., 3] = .5
    >>> actifptm_contacts(p, np.array([4., 8., 12.]), 1).tolist()
    [[0.5]]
    """
    if distogram_probs is None or bin_edges is None:
        return None
    n = int((np.asarray(bin_edges) <= contact_cutoff).sum())
    return distogram_probs[:len_a, len_a:, :n].sum(axis=-1)


def actifptm(tm: np.ndarray | None, contact_prob: np.ndarray | None, len_a: int) -> float:
    """actifpTM: ipTM with each partner residue weighted by its contact probability."""
    if tm is None or contact_prob is None:
        return NAN
    vals = []
    for block, w in ((tm[:len_a, len_a:], contact_prob), (tm[len_a:, :len_a], contact_prob.T)):
        if block.size and w.sum() > 0:
            vals.append(((block * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-8)).max())
    return float(max(vals)) if vals else 0.


def chain_plddt(plddt: np.ndarray, len_a: int) -> tuple[float, float]:
    """Mean pLDDT of chain A and chain B."""
    a, b = plddt[:len_a], plddt[len_a:]
    return float(a.mean()) if a.size else NAN, float(b.mean()) if b.size else NAN


def score_complex(
    plddt: np.ndarray,
    contact_dists: np.ndarray,
    len_a: int,
    pae: np.ndarray | None = None,
    tm: np.ndarray | None = None,
    contact_prob: np.ndarray | None = None,
    contact_radius: float = 8.,
    pae_cutoff: float = 10.,
) -> dict:
    """All scores for one complex. PAE/TM-derived ones are NaN when those inputs are ``None``."""
    rows, cols, inter = interface_residues(contact_dists, len_a, contact_radius)
    n_contacts = int(inter.sum())
    if n_contacts:
        interface_plddt = float(np.concatenate([plddt[:len_a][rows], plddt[len_a:][cols]]).mean())
    else:
        interface_plddt = 0.
    plddt_a, plddt_b = chain_plddt(plddt, len_a)
    return {
        "n_contacts": n_contacts,
        "mean_plddt_a": plddt_a,
        "mean_plddt_b": plddt_b,
        "mean_interface_plddt": interface_plddt,
        "pdockq": pdockq(interface_plddt, n_contacts),
        "pdockq2": pdockq2(pae, plddt, contact_dists, len_a, contact_radius),
        "ptm": ptm(tm),
        "iptm": iptm(tm, len_a),
        "actifptm": actifptm(tm, contact_prob, len_a),
        "ipsae": ipsae(pae, len_a, pae_cutoff),
        "lis": lis(pae, len_a),
        "mean_interchain_pae": mean_interchain_pae(pae, len_a),
    }


# ----------------------------------------------------------------------------------------------
# One path for every backend
#
# A backend adapter's only job is to hand over a *canonical record* (see ``compact_prediction``):
# pLDDT on a 0-100 scale, PAE bin probabilities, and the probability of a contact. Every scalar
# score is then computed here, from that record, by the same functions, so a score named
# ``iptm`` means the same thing whichever model produced it. The model's own reported pTM/ipTM
# (where it has them) are kept as separate ``*_model`` / ``ptm_af2`` columns for cross-checking.

PAE_N_BINS = 64
PAE_MAX = 32.
#: PAE bin centres shared by AlphaFold2 (63 edges over 0-31 A) and OpenFold3 (64 bins over 0-32 A).
PAE_BIN_CENTERS = (np.arange(PAE_N_BINS) + .5) * (PAE_MAX / PAE_N_BINS)


def expected_pae(probs: np.ndarray, bin_centers: np.ndarray = PAE_BIN_CENTERS) -> np.ndarray:
    """Expected PAE (A) per residue pair from bin probabilities."""
    return (probs * bin_centers).sum(axis=-1)


def compact_prediction(
    plddt,
    pae_probs=None,
    contact_prob=None,
    pae_bin_centers: np.ndarray = PAE_BIN_CENTERS,
    pae=None,
    **model_values,
) -> dict:
    """Validate a backend's raw outputs and reduce them to the canonical per-block record.

    ``plddt`` must be on a 0-100 scale; ``pae_probs`` is ``(L, L, n_bins)`` and sums to 1 over bins;
    ``contact_prob`` is ``(L, L)``, ``P(d < 8 A)``. The record holds ``plddt``, ``pae`` (expected PAE),
    ``tm`` (the ``(L, L)`` pTM/ipTM building block) and ``contact_prob_full``; any ``**model_values``
    (e.g. ``ptm``, ``iptm`` as the model itself reported them) are passed through untouched.
    Unit mistakes raise rather than silently skew a score.

    >>> rec = compact_prediction(np.full(3, 80.), pae_probs=np.full((3, 3, 64), 1 / 64.))
    >>> rec["pae"].shape, round(float(rec["pae"][0, 0]), 2)
    ((3, 3), 16.0)
    >>> compact_prediction(np.full(3, .8))
    Traceback (most recent call last):
    ...
    ValueError: pLDDT must be on a 0-100 scale (max 0.8 looks like 0-1)
    """
    plddt = np.asarray(plddt, dtype=np.float32)
    if plddt.size:
        if not np.isfinite(plddt).all() or plddt.min() < 0 or plddt.max() > 100.01:
            raise ValueError("pLDDT must be finite and within 0-100")
        if plddt.max() <= 1.5:
            raise ValueError(f"pLDDT must be on a 0-100 scale (max {float(plddt.max()):.3g} looks like 0-1)")
    n = plddt.shape[0]
    record = dict(model_values)
    record["plddt"] = plddt
    if pae_probs is not None:
        probs = np.asarray(pae_probs, dtype=np.float32)
        if probs.shape != (n, n, len(pae_bin_centers)):
            raise ValueError(f"PAE probabilities have shape {probs.shape}, expected {(n, n, len(pae_bin_centers))}")
        if not np.allclose(probs.sum(axis=-1), 1., atol=1e-2):
            raise ValueError("PAE bin probabilities must sum to 1 over the last axis (are these logits?)")
        record["pae"] = expected_pae(probs, pae_bin_centers).astype(np.float32)
        record["tm"] = tm_matrix(probs, np.asarray(pae_bin_centers), n).astype(np.float32)
    elif pae is not None:  # expected PAE only (no bin probabilities): TM-based scores are unavailable
        pae = np.asarray(pae, dtype=np.float32)
        if pae.shape != (n, n) or not np.isfinite(pae).all() or pae.min() < 0 or pae.max() > PAE_MAX + 1:
            raise ValueError(f"PAE must be (L, L), finite and in Angstrom (0-{PAE_MAX:g}); got shape {pae.shape}, "
                             f"range {float(np.nanmin(pae)):.3g}-{float(np.nanmax(pae)):.3g}")
        record["pae"] = pae
    if contact_prob is not None:
        contact_prob = np.asarray(contact_prob, dtype=np.float32)
        if contact_prob.shape != (n, n) or contact_prob.min() < -1e-4 or contact_prob.max() > 1 + 1e-4:
            raise ValueError("contact probabilities must be (L, L) and within [0, 1]")
        record["contact_prob_full"] = contact_prob
    return record


def score_blocks(paired_msa, results, contact_radius: float = 8., pae_cutoff: float = 10.):
    """Scores for a complex from per-block ``(block, (contact_dists, record))`` results.

    The single implementation behind every backend's ``post_*``. pLDDT is averaged over the
    blocks that contain a residue; unseen residue pairs are infinitely far apart; PAE-based scores
    are blockwise approximations when chunked; pTM, ipTM and actifpTM need one whole-complex call
    (NaN otherwise). Returns ``(scores, whole_record)``; ``whole_record`` is ``{}`` when chunked.
    """
    from .chunking import assemble_matrix, assemble_pae, assemble_vector
    n_cols, len_a = paired_msa.seq_length, paired_msa.chain_a_length
    results = list(results)
    contact_dist = assemble_matrix(((b, e[0]) for b, e in results), n_cols, fill=np.inf)
    plddt = assemble_vector(((b, e[1]["plddt"]) for b, e in results), n_cols)
    whole = [e[1] for b, e in results if b.is_whole]
    one = whole[0] if len(whole) == 1 else {}
    contact_prob = one.get("contact_prob_full")
    scores = score_complex(
        plddt, contact_dist, len_a,
        pae=assemble_pae(results, n_cols),
        tm=one.get("tm"),
        contact_prob=None if contact_prob is None else contact_prob[:len_a, len_a:],
        contact_radius=contact_radius,
        pae_cutoff=pae_cutoff,
    )
    return scores, one
