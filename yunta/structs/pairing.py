"""Pairing strategies, including negative controls that ignore known interactions.

Co-evolution between two proteins is read from a *paired* alignment: each row joins
a sequence of protein A with a sequence of protein B. Which rows are joined decides
what signal DCA/RF2t/AF2 can see, so every screen needs a control that keeps everything
about the pairing *except* the biology: the same depth, the same species multiplicities
and the same row-selection rule, but partners that are not known to interact.

Strategies (``pairing=``)
-------------------------
``"map"`` (default)
    The real pairing: species linked by the interaction map (same species,
    the built-in host-pathogen map, or a custom map). Unchanged behaviour.
``"rewired"``
    Degree-preserving *negative control*. The bipartite graph of species pairs that the
    map allows is randomly rewired by double edge swaps (Maslov & Sneppen, *Science* 2002,
    the standard null model for interaction networks) with the extra constraint that no
    swap may create a known (map-allowed) pair. Every species keeps exactly the number of
    partners it had, so the number of paired rows and each species' multiplicity are
    identical to the real pairing, but no known interacting species pair is retained
    (when the graph allows it; the count of retained known pairs is reported).
``"random-species"``
    Depth-matched uniform control: the same *number* of species pairs, drawn uniformly
    from the (species in A) x (species in B) pairs that the map does **not** allow.
    Weaker than ``"rewired"`` (per-species multiplicities are not preserved).
``"random-rows"``
    Taxonomy-blind control: the same number of paired rows, with rows of A and B matched
    uniformly at random (rows may be reused, as they are in the real pairing when a species
    has several partners), rejecting row pairs whose species the map allows to pair. The
    classical "shuffled pairing" ablation (cf. the random initial pairing of the iterative
    pairing algorithm of Bitbol et al., *PNAS* 2016); it also removes shared-phylogeny signal.

``"permute-partners"``
    Composition-preserving control: take the *exact rows* the real pairing selected and
    permute the B-chain rows among them, rejecting any permutation that restores a known
    species pair. Every A row and every B row is used exactly as often as in the real
    pairing, so each chain's sequence composition, gap distribution and redundancy are
    identical to the real alignment; only which A row sits next to which B row changes.
    Unlike ``"random-rows"`` (which draws new rows, so depth-matched alignments can still
    differ in gaps and diversity), a difference from this control is attributable to the
    pairing alone.

Invariants, verified at run time
--------------------------------
Every control is checked and summarised in a :class:`PairingReport` (also written to the
metrics and to the ``yunta pair`` sidecar). A control is ``valid`` only if

* **depth**: the number of paired rows equals the real pairing's (all controls);
* **no known pairs**: zero retained map-allowed species pairs (all controls);
* **multiplicity**: every species has the same number of partners as in the real pairing
  (``rewired`` only; the other controls do not claim it).

Controls are validated twice: on the pairing as generated, and again on the **final**
alignment after the gap-fraction filter (``max_gap_fraction``) that is what the models see.
After filtering a control must have the same number of paired rows as the *real* pairing
put through the same filter (a control with more rows is subsampled, with its seed, to match;
one with fewer is invalid). The final Neff and mean gap fraction of the control and of the
real reference are recorded so residual differences in diversity are visible.

Validation is strict by default: an invalid control raises :class:`InvalidControlError` (batch
runners skip and list the pair instead of scoring it). ``strict_controls=False`` (CLI
``--allow-invalid-controls``) scores it anyway with ``pairing_valid = False``.

How to read the control
-----------------------
Score the real pairing and ``K`` replicates of a control (different ``seed``) for the same
protein pair. A signal that is not clearly above the control distribution is explained by
alignment depth, gap pattern and diversity, not by the known interaction. This is the
pair-level analogue of the negative sets used in proteome-scale screens, where
non-interacting protein pairs are drawn from different complexes (Cong et al., *Science*
2019); here the *pairing* is randomised while the two proteins stay the same.

Caveat: shared phylogeny alone can produce inter-protein signal when species are paired
correctly (Gandarilla-Perez et al.; Bitbol et al.), so a control that also breaks
phylogenetic correspondence (all of the above) is expected to read *lower* than a
phylogeny-matched control. Treat a real score above control as necessary, not sufficient.

Examples
========
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> edges = {("h1", "p1"), ("h1", "p2"), ("h2", "p1"), ("h2", "p3"), ("h3", "p2"), ("h3", "p3")}
>>> new, kept = rewire_bipartite(edges, rng)
>>> len(new) == len(edges), kept == len(new & edges)
(True, True)
>>> from collections import Counter
>>> Counter(a for a, _ in new) == Counter(a for a, _ in edges)
True
>>> Counter(b for _, b in new) == Counter(b for _, b in edges)
True

"""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal
import secrets
import zlib

import numpy as np

PAIRING_METHODS = ("map", "rewired", "random-species", "random-rows", "permute-partners")
PairingMethod = Literal["map", "rewired", "random-species", "random-rows", "permute-partners"]
Edge = tuple[str, str]


class UnpairableError(AttributeError, ValueError):
    """The two MSAs cannot be paired (no allowed pairs, unknown or mismatched query species).

    Distinct from configuration errors (bad ``pairing``, bad ``interaction_map``), which stay
    plain ``ValueError``s and must not be swallowed as "skip this pair". It also derives from
    ``AttributeError`` and ``ValueError`` for compatibility with callers that caught those.
    """


class InvalidControlError(RuntimeError):
    """A negative control failed its invariants (strict validation is the default)."""


@dataclass
class PairingReport:
    """What a pairing did, and whether a control met its invariants.

    ``target_pairs`` is the number of (non-query) species pairs the real map pairing gives;
    ``n_pairs`` the number this pairing gave. ``retained_known`` counts pairs that the map
    allows (always 0 for a valid control).

    >>> r = PairingReport("rewired", 3, target_pairs=4, n_pairs=4, retained_known=0, multiplicity_ok=True)
    >>> r.finalise().valid
    True
    >>> PairingReport("random-rows", 3, target_pairs=4, n_pairs=2, retained_known=0).finalise().notes
    ['depth 2 != target 4']
    """

    method: str
    seed: int | None = None
    target_pairs: int | None = None
    n_pairs: int | None = None
    retained_known: int | None = None
    multiplicity_ok: bool | None = None  # only claimed by "rewired" and "permute-partners"
    valid: bool = True
    notes: list = field(default_factory=list)
    # Final (post gap-filter) alignment, filled by ``check_final``; depths exclude the query row.
    final_pairs: int | None = None
    final_target_pairs: int | None = None
    n_eff: float | None = None
    mean_gap: float | None = None
    reference_n_eff: float | None = None
    reference_mean_gap: float | None = None
    subsampled_to_match: bool = False
    # Invariants re-measured on the final alignments (control vs the real pairing put through the same
    # filter). ``multiplicity_ok`` above is the *as generated* (pre-filter) check; these are separate.
    final_multiplicity_ok: bool | None = None  # per-species multiplicity; claimed by rewired, permute-partners
    final_composition_ok: bool | None = None  # per-chain row multisets; claimed by permute-partners
    final_retained_real: int | None = None  # control rows whose species pair is a real (reference) pair
    _pre_notes: list = field(default_factory=list, repr=False)

    def finalise(self) -> "PairingReport":
        self.notes = []
        if self.method != "map":
            if self.n_pairs != self.target_pairs:
                self.notes.append(f"depth {self.n_pairs} != target {self.target_pairs}")
            if self.retained_known:
                self.notes.append(f"{self.retained_known} known pairs retained")
            if self.method in ("rewired", "permute-partners") and self.multiplicity_ok is False:
                self.notes.append("species multiplicity not preserved")
        self._pre_notes = list(self.notes)
        self.valid = not self.notes
        return self

    def check_final(
        self,
        final_pairs: int,
        final_target_pairs: int | None,
        n_eff: float | None = None,
        mean_gap: float | None = None,
        reference_n_eff: float | None = None,
        reference_mean_gap: float | None = None,
        multiplicity_ok: bool | None = None,
        composition_ok: bool | None = None,
        retained_real: int | None = None,
    ) -> "PairingReport":
        """Validate the alignment that is actually scored (after the gap filter).

        Depth is always checked. ``multiplicity_ok`` (rewired, permute-partners), ``composition_ok``
        (permute-partners) and ``retained_real`` are measured on the final control and final reference
        alignments by the caller; ``None`` means "not claimed by this method".

        >>> r = PairingReport("random-rows", 1, target_pairs=5, n_pairs=5, retained_known=0).finalise()
        >>> r.check_final(final_pairs=4, final_target_pairs=4).valid
        True
        >>> r.check_final(final_pairs=3, final_target_pairs=4).notes
        ['final depth 3 != real final depth 4 (after gap filtering)']
        >>> r = PairingReport("rewired", 1, target_pairs=5, n_pairs=5, retained_known=0, multiplicity_ok=True).finalise()
        >>> r.check_final(4, 4, multiplicity_ok=False, retained_real=2).notes
        ['final per-species multiplicity differs from the real pairing', 'final alignment retains 2 real species pairs']
        """
        self.final_pairs, self.final_target_pairs = final_pairs, final_target_pairs
        self.n_eff, self.mean_gap = n_eff, mean_gap
        self.reference_n_eff, self.reference_mean_gap = reference_n_eff, reference_mean_gap
        self.final_multiplicity_ok, self.final_composition_ok = multiplicity_ok, composition_ok
        self.final_retained_real = retained_real
        self.notes = list(self._pre_notes)
        if self.method != "map" and final_target_pairs is not None and final_pairs != final_target_pairs:
            self.notes.append(
                f"final depth {final_pairs} != real final depth {final_target_pairs} (after gap filtering)"
            )
        if self.method != "map":
            if multiplicity_ok is False:
                self.notes.append("final per-species multiplicity differs from the real pairing")
            if composition_ok is False:
                self.notes.append("final per-chain sequence composition differs from the real pairing")
            if retained_real:
                self.notes.append(f"final alignment retains {retained_real} real species pairs")
        self.valid = not self.notes
        return self

    def as_dict(self) -> dict:
        out = asdict(self)
        out.pop("_pre_notes", None)
        return out


def make_rng(seed: int | None, *keys: str) -> tuple[np.random.Generator, int]:
    """A generator that is reproducible per (seed, protein pair), plus the seed used.

    Mixing in the query IDs gives every pair in a screen its own stream while one
    ``--seed`` still reproduces the whole screen. A fresh random seed is drawn (and returned,
    to be recorded) if none is given.

    >>> a, s = make_rng(1, "P1", "P2"); b, _ = make_rng(1, "P1", "P2"); c, _ = make_rng(1, "P1", "P3")
    >>> bool(a.integers(1 << 30) == b.integers(1 << 30)), s
    (True, 1)
    """
    if seed is None:
        seed = secrets.randbelow(2**31)
    mix = zlib.crc32("|".join(keys).encode())
    return np.random.default_rng([seed, mix]), seed


def rewire_bipartite(
    edges: Iterable[Edge],
    rng: np.random.Generator,
    forbid_known: bool = True,
    max_tries_per_edge: int = 200,
    extra_forbidden: Iterable[Edge] = (),
) -> tuple[set[Edge], int]:
    """Degree-preserving rewiring of a bipartite edge set by double edge swaps.

    ``(a, b), (c, d) -> (a, d), (c, b)`` keeps every node's degree (Maslov & Sneppen, 2002).
    With ``forbid_known`` a swap is only accepted if both new edges are absent from the
    *original* graph, and only original edges are chosen as the first edge of a swap, so the
    procedure runs until no original edge remains (or no feasible swap is found).

    Returns the new edge set and the number of original edges that could not be replaced.
    """
    edges = sorted(set(edges))
    n = len(edges)
    original = frozenset(edges)
    forbidden = original | frozenset(extra_forbidden)
    if n < 2:
        return set(edges), n
    current = set(edges)
    pending = {i for i, e in enumerate(edges)}  # indices still holding an original edge
    tries, max_tries = 0, max_tries_per_edge * n
    while pending and tries < max_tries:
        tries += 1
        i = int(rng.choice(sorted(pending))) if len(pending) < 64 else int(rng.choice(list(pending)))
        j = int(rng.integers(n))
        if i == j:
            continue
        (a, b), (c, d) = edges[i], edges[j]
        if a == c or b == d:
            continue
        first, second = (a, d), (c, b)
        if first in current or second in current:
            continue
        if forbid_known and (first in forbidden or second in forbidden):
            continue
        current -= {edges[i], edges[j]}
        current |= {first, second}
        edges[i], edges[j] = first, second
        pending.discard(i)
        pending.discard(j)
        for k in (i, j):
            if edges[k] in original:
                pending.add(k)
    return current, sum(e in original for e in current)


def sample_random_pairs(
    species1: Sequence[str],
    species2: Sequence[str],
    n_pairs: int,
    rng: np.random.Generator,
    forbidden: Iterable[Edge] = (),
) -> set[Edge]:
    """``n_pairs`` distinct (species in A, species in B) pairs drawn uniformly, never from ``forbidden``.

    Fewer are returned only if fewer allowed pairs exist.

    >>> import numpy as np
    >>> bad = {("a", "x"), ("b", "y")}
    >>> pairs = sample_random_pairs(["a", "b", "c"], ["x", "y"], 4, np.random.default_rng(0), bad)
    >>> len(pairs), pairs.isdisjoint(bad), pairs <= {(a, b) for a in "abc" for b in "xy"}
    (4, True, True)
    >>> len(sample_random_pairs(["a"], ["x"], 3, np.random.default_rng(0), {("a", "x")}))
    0
    """
    species1, species2 = sorted(species1), sorted(species2)
    n2 = len(species2)
    total = len(species1) * n2
    index1 = {s: i for i, s in enumerate(species1)}
    index2 = {s: i for i, s in enumerate(species2)}
    banned = {index1[a] * n2 + index2[b] for a, b in forbidden if a in index1 and b in index2}
    if total - len(banned) <= n_pairs or total <= 2_000_000:
        allowed = np.setdiff1d(np.arange(total), np.fromiter(banned, dtype=np.int64, count=len(banned)))
        chosen = rng.choice(allowed, size=min(n_pairs, len(allowed)), replace=False) if len(allowed) else []
    else:  # large sparse case: rejection sampling
        chosen_set: set[int] = set()
        while len(chosen_set) < n_pairs:
            for k in rng.integers(0, total, size=2 * (n_pairs - len(chosen_set)) + 8).tolist():
                if k not in banned:
                    chosen_set.add(k)
                if len(chosen_set) == n_pairs:
                    break
        chosen = list(chosen_set)
    return {(species1[int(k) // n2], species2[int(k) % n2]) for k in chosen}


def allowed_row_pair_capacity(
    keys1: Sequence, keys2: Sequence, known: Iterable[tuple],
) -> int:
    """Exactly how many (row in A, row in B) pairs are allowed, i.e. not a known species pair.

    ``keys1[i]``/``keys2[j]`` are the species of each row; ``known`` the forbidden species pairs.
    ``n1 * n2 - sum over known (a, b) of count1[a] * count2[b]``. This is deterministic, so it
    tells a genuinely infeasible control from one that merely exhausted a sampling budget.

    >>> allowed_row_pair_capacity(["a", "a", "b"], ["x", "y"], {("a", "x")})
    4
    """
    c1, c2 = Counter(keys1), Counter(keys2)
    return len(keys1) * len(keys2) - sum(c1.get(a, 0) * c2.get(b, 0) for a, b in set(known))


def sample_random_rows(
    n_rows1: int,
    n_rows2: int,
    n_pairs: int,
    rng: np.random.Generator,
    forbidden: Callable[[int, int], bool] | None = None,
    max_tries_factor: int = 50,
    allowed_matrix: Callable[[], np.ndarray] | None = None,
) -> list[tuple[int, int]]:
    """``n_pairs`` distinct (row in A, row in B) pairs, uniform over allowed pairs.

    Rows may be reused across pairs (as in the real pairing, where one row serves every
    partner species). Pairs for which ``forbidden(i, j)`` is true are rejected. Rejection
    sampling has a bounded budget; if it runs out and ``allowed_matrix`` (a callable returning
    the boolean ``(n_rows1, n_rows2)`` matrix of allowed pairs) is given, the remainder is
    drawn exactly from the enumerated allowed pairs, so fewer than ``n_pairs`` come back only
    when fewer than ``n_pairs`` allowed pairs exist.

    >>> import numpy as np
    >>> rows = sample_random_rows(5, 7, 10, np.random.default_rng(0), lambda i, j: i == j)
    >>> len(rows), len(set(rows)), any(i == j for i, j in rows)
    (10, 10, False)
    >>> sample_random_rows(2, 2, 10, np.random.default_rng(0))  # only 4 distinct pairs exist
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> dense = lambda: np.eye(30, 31, dtype=bool)  # a single allowed pair per row
    >>> len(sample_random_rows(30, 31, 30, np.random.default_rng(0), lambda i, j: i != j,
    ...                        max_tries_factor=1, allowed_matrix=dense))
    30
    """
    n_pairs_max = n_rows1 * n_rows2
    if n_pairs >= n_pairs_max and forbidden is None:
        return [(i, j) for i in range(n_rows1) for j in range(n_rows2)]
    chosen: dict[tuple[int, int], None] = {}
    tries = 0
    while len(chosen) < min(n_pairs, n_pairs_max) and tries < max_tries_factor * max(n_pairs, 1):
        tries += 1
        pair = (int(rng.integers(n_rows1)), int(rng.integers(n_rows2)))
        if pair in chosen or (forbidden is not None and forbidden(*pair)):
            continue
        chosen[pair] = None
    if len(chosen) < min(n_pairs, n_pairs_max) and allowed_matrix is not None:
        flat = np.flatnonzero(np.asarray(allowed_matrix()).ravel())  # exact: every allowed pair
        taken = {i * n_rows2 + j for i, j in chosen}
        pool = np.array([k for k in flat if int(k) not in taken], dtype=np.int64)
        need = min(n_pairs, n_pairs_max) - len(chosen)
        if len(pool):
            for k in rng.choice(pool, size=min(need, len(pool)), replace=False):
                chosen[(int(k) // n_rows2, int(k) % n_rows2)] = None
    return sorted(chosen)


def permute_partners(
    keys_a: Sequence,
    keys_b: Sequence,
    known: set[tuple],
    rng: np.random.Generator,
    max_rounds: int = 200,
) -> tuple[list[int], int]:
    """Permute the B rows among paired rows so that no known species pair is restored.

    ``keys_a[i]``, ``keys_b[i]`` are the species of the A and B row of real pair ``i``. Returns
    ``(perm, n_retained)``: pair ``i`` now joins A row ``i`` with B row ``perm[i]``. ``perm`` is a
    permutation, so every row keeps its use count. Pairs that land on a known species pair
    (including keeping their real partner) are repaired by swapping with a random position
    where both resulting pairs are allowed; ``n_retained`` counts those that could not be repaired.

    >>> import numpy as np
    >>> a = [f"s{i}" for i in range(20)]; known = {(x, x) for x in a}
    >>> perm, kept = permute_partners(a, a, known, np.random.default_rng(1))
    >>> sorted(perm) == list(range(20)), kept, any(i == p for i, p in enumerate(perm))
    (True, 0, False)
    """
    n = len(keys_a)
    perm = [int(i) for i in rng.permutation(n)]
    bad = lambda i, j: (keys_a[i], keys_b[j]) in known
    for _ in range(max_rounds):
        offenders = [i for i in range(n) if bad(i, perm[i])]
        if not offenders:
            break
        progressed = False
        for i in offenders:
            if not bad(i, perm[i]):
                continue
            for k in rng.permutation(n)[: min(n, 64)]:
                k = int(k)
                if k != i and not bad(i, perm[k]) and not bad(k, perm[i]):
                    perm[i], perm[k] = perm[k], perm[i]
                    progressed = True
                    break
        if not progressed:
            break
    return perm, sum(bad(i, perm[i]) for i in range(n))


def multiplicity(edges: Iterable[Edge]) -> tuple[Counter, Counter]:
    """Per-species partner counts on each side."""
    edges = list(edges)
    return Counter(a for a, _ in edges), Counter(b for _, b in edges)


def validate_pairing(pairing: str) -> str:
    if pairing not in PAIRING_METHODS:
        raise ValueError(f"`pairing` must be one of {PAIRING_METHODS}, got {pairing!r}.")
    return pairing


def _rejection_reason(message: str) -> str:
    """Short category of a control-rejection message."""
    for key, label in (
        ("infeasible", "infeasible (too few allowed pairs)"),
        ("multiplicity", "final multiplicity differs"),
        ("composition", "final composition differs"),
        ("retains", "retains real species pairs"),
        ("final depth", "final depth below the real pairing"),
        ("depth", "depth differs"),
    ):
        if key in message:
            return label
    return "other"


def log_rejections(method: str, n_pairs: int, rejected: dict) -> None:
    """Log how many negative controls were rejected and why (one INFO line; a WARN if most were).

    ``rejected`` maps a pair (index or name) to its rejection message. Silent for the real pairing.
    """
    from carabiner import print_err
    if method == "map" or not n_pairs:
        return
    n_rej = len(rejected)
    why = "; ".join(f"{k} x{v}" for k, v in Counter(_rejection_reason(m) for m in rejected.values()).most_common())
    print_err(
        f"[INFO] Negative control '{method}': {n_rej}/{n_pairs} pairs rejected ({100 * n_rej / n_pairs:.0f}%)"
        + (f" ({why}); use --allow-invalid-controls to score them, flagged pairing_valid=False" if n_rej else "") + "."
    )
    if n_rej / n_pairs > .5:
        print_err(
            f"[WARN] Most '{method}' controls were rejected, so the scored controls are a biased subset; "
            "compare real and control on the same pairs only."
        )
