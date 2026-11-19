"""Running interaction calculations."""
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable, Mapping
from abc import ABC, abstractmethod
import random
from collections import Counter
from functools import partial

from carabiner import print_err
if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.typing import ArrayLike
else:
    ndarray, ArrayLike = Any, Any

import numpy as np

from .chunking import Block, run_blocks
from .scoring import score_contact_map
from ..structs.metrics import AF2Metrics, DCAMetrics, RF2TMetrics, InteractionMetrics
from ..structs.msa import MSA, PairedMSA, __BLOCK_GAPS__ as _BLOCK_MARK
from ..structs.pairing import InvalidControlError, UnpairableError, make_rng, multiplicity, validate_pairing

DEFAULT_CHUNKSIZE: int | None = None  # chunking is opt-in: None runs the whole complex

def _paired_rows(paired_msa: PairedMSA) -> list[int]:
    """Indices of genuinely paired rows (query first); ``blocked`` rows of unpaired sequences are excluded."""
    return [i for i, line in enumerate(paired_msa.lines) if _BLOCK_MARK not in str(line.name)]


def _final_edges(paired_msa: PairedMSA) -> list[tuple]:
    """(species of A row, species of B row) for every paired row of the final alignment.

    The query pair (row 0) and the real query-species row that every control keeps are excluded, matching the
    convention of the pairing report (depths and multiplicities exclude the query species pair).
    """
    key = lambda d: d.species_id if d.species_id != -1 else d.generic_species_name
    lines = paired_msa.lines
    query = (key(lines[0].description[0]), key(lines[0].description[1]))
    edges = [(key(lines[i].description[0]), key(lines[i].description[1])) for i in _paired_rows(paired_msa)[1:]]
    return [e for e in edges if e != query]


def _final_composition(paired_msa: PairedMSA) -> tuple[Counter, Counter]:
    """Multisets of the A and B sequences over the non-query paired rows of the final alignment."""
    la = paired_msa.chain_a_length
    rows = [paired_msa.lines[i].sequence for i in _paired_rows(paired_msa)[1:]]
    return Counter(r[:la] for r in rows), Counter(r[la:] for r in rows)


def _mean_gap(paired_msa: PairedMSA) -> float:
    return float(np.mean([line.gap_fraction for line in paired_msa.lines])) if len(paired_msa) else 0.


def _pair_msas(
    msa1: MSA, 
    msa2: MSA | None = None,
    max_gap_fraction: float = 1.,
    blocked: bool = False,
    interaction_map: str | Mapping[str, Iterable[str]] | None = None,
    enforce_ref_match: bool = False,
    pairing: str = "map",
    seed: int | None = None,
    strict_controls: bool = True,
) -> PairedMSA | None:
    """Pair two MSAs, or return ``None`` if they are *unpairable* (and warn).

    Only :class:`~yunta.structs.pairing.UnpairableError` means "skip this pair". Invalid
    arguments (unknown ``pairing``, bad ``interaction_map``) and failed controls (strict by
    default) propagate, so a mistake cannot masquerade as an empty result.

    A negative control is validated twice: as generated, and again here on the **final**
    alignment after the gap filter, against the real pairing put through the same filter
    (same ``max_gap_fraction``). A control with more final rows than the real reference is
    subsampled (seeded) to match; one with fewer is invalid. Final Neff and mean gap fraction
    of control and reference are recorded on ``pairing_info``.
    """
    validate_pairing(pairing)
    kwargs = dict(
        blocked=blocked, 
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
    )
    try:
        paired_msa = PairedMSA.from_msa(
            msa1, msa2, pairing=pairing, seed=seed, strict_controls=strict_controls,
            max_gap_fraction=max_gap_fraction, **kwargs,  # (only permute-partners uses it inside the join)
        )
    except UnpairableError as e:  # no pairs, or unknown/mismatched query species
        print_err(f"[WARN] Skipping pair,", e)
        return None
    if len(_paired_rows(paired_msa)) < 2:  # only the query pair: nothing to learn from (and DCA is undefined)
        print_err(f"[WARN] Skipping pair, {paired_msa.name} has no paired homologs besides the query.")
        return None
    if pairing == "permute-partners":
        # The real rows were gap-filtered *before* permuting (inside the join), so the control holds exactly
        # the real final A and B rows. Filtering the permuted rows again would change which rows survive and
        # break the composition guarantee, so only the unpaired (blocked) rows are filtered here.
        paired_idx = set(_paired_rows(paired_msa))
        final = paired_msa._filter_by_index(
            i for i, line in enumerate(paired_msa.lines)
            if i in paired_idx or line.gap_fraction <= max_gap_fraction
        ) if max_gap_fraction < 1. else paired_msa
    else:
        final = paired_msa.filter_by_gap_fraction(max_gap_fraction)  # keeps pairing_info (shallow copy)
    report = final.pairing_info
    if report is None:
        return final
    if pairing == "map":
        report.n_eff, report.mean_gap = final.neff(verbose=False), _mean_gap(final)
        report.final_pairs = report.final_target_pairs = len(_paired_rows(final)) - 1
        return final
    reference = PairedMSA.from_msa(
        msa1, msa2, pairing="map", strict_controls=False, **kwargs,
    ).filter_by_gap_fraction(max_gap_fraction)
    target = len(_paired_rows(reference)) - 1
    rows = _paired_rows(final)
    if len(rows) - 1 > target and pairing != "permute-partners":  # same filtering policy, same depth: drop surplus control rows
        rng, _ = make_rng(report.seed, "final-depth", *map(str, paired_msa.ids))
        drop = set(rng.choice(rows[1:], size=len(rows) - 1 - target, replace=False).tolist())
        final = final._filter_by_index(i for i in range(len(final)) if i not in drop)
        report.subsampled_to_match = True
    edges_c, edges_r = _final_edges(final), _final_edges(reference)
    mult_ok = comp_ok = None
    if pairing in ("rewired", "permute-partners"):  # methods that claim multiplicity preservation
        mult_ok = multiplicity(edges_c) == multiplicity(edges_r)
    if pairing == "permute-partners":
        comp_ok = _final_composition(final) == _final_composition(reference)
    report.check_final(
        final_pairs=len(_paired_rows(final)) - 1,
        final_target_pairs=target,
        n_eff=final.neff(verbose=False),
        mean_gap=_mean_gap(final),
        reference_n_eff=reference.neff(verbose=False),
        reference_mean_gap=_mean_gap(reference),
        multiplicity_ok=mult_ok,
        composition_ok=comp_ok,
        retained_real=len(set(edges_c) & set(edges_r)),
    )
    if not report.valid:
        message = f"negative control '{pairing}' violates its invariants: {'; '.join(report.notes)}"
        if strict_controls:
            raise InvalidControlError(message)
        print_err(f"[WARN] {message}. Flagged pairing_valid=False.")
    return final


def _letter_alignment(paired_msa: PairedMSA):
    """Residue letters as an ``(n_rows, n_cols)`` array, plus a function joining slices to strings.

    AF2's feature pipeline wants the original letters, not tokens.
    """
    import numpy as np
    n_rows = len(paired_msa)
    letters = np.frombuffer(
        "".join(paired_msa.sequences()).encode("ascii", "replace"), 
        dtype="S1",
    ).reshape(n_rows, paired_msa.seq_length)
    return letters, lambda sub: [b"".join(row).decode("ascii") for row in sub]


def available_memory_gib(cpu: bool = False) -> float | None:
    """Free GPU memory (or system RAM when ``cpu``/no GPU), in GiB; ``None`` if unknown."""
    try:
        if not cpu:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.mem_get_info()[0] / 2**30
        import os
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ImportError, ValueError, OSError, AttributeError):
        return None


class Runner(ABC):

    metric_container = InteractionMetrics
    use_sequences = False
    use_id = False
    stochastic = False  # if True, a seed is drawn up front so it can be recorded

    @staticmethod
    def make_model(cpu=False, **kwargs):
        return None

    @staticmethod
    @abstractmethod
    def _run_chunk(
        paired_msa: PairedMSA,
        model: Callable | None = None,
        **kwargs
    ) -> Iterable[ArrayLike]:
        ...

    default_blocks = "all"  # which block pairs to run if the alignment must be chunked

    @staticmethod
    def estimate_memory_gib(n_cols: int, n_rows: int) -> float | None:
        """Rough peak memory (GiB) of one whole-complex call, or ``None`` if unknown."""
        return None

    def _warn_if_too_big(self, paired_msa: PairedMSA, cpu: bool) -> None:
        """Warn, before an out-of-memory crash, when a whole-complex call looks too big."""
        need = self.estimate_memory_gib(paired_msa.seq_length, len(paired_msa))
        have = available_memory_gib(cpu)
        if need is not None and have is not None and need > 0.9 * have:
            print_err(
                f"[WARN] {paired_msa.name}: a whole-complex {type(self).__name__} call over "
                f"{paired_msa.seq_length} columns needs roughly {need:.0f} GiB but about {have:.0f} GiB "
                "is free. If it fails, opt in to chunking with chunksize / --chunksize."
            )

    @staticmethod
    def prepare(paired_msa: PairedMSA, **kwargs) -> dict:
        """Compute anything that must be *global* to the alignment, once, before chunking.

        Returned items are passed to every ``_run_chunk`` call.
        """
        return {}

    @staticmethod
    def finalize_matrix(paired_msa: PairedMSA, matrix: "ndarray", **kwargs) -> "ndarray":
        """Post-process the assembled full matrix (e.g. global APC)."""
        return matrix

    @staticmethod
    def post_run(
        paired_msa: PairedMSA,
        results: list[tuple[Block, tuple]],
        **kwargs
    ) -> dict:
        return {}

    def run(
        self,
        msa1: MSA,
        msa2: MSA | None = None,
        max_gap_fraction: float = .9,
        interaction_map: str | Mapping[str, Iterable[str]] | None = None,
        cpu: bool = False,
        model: Callable | None = None,
        chunksize: int | None = DEFAULT_CHUNKSIZE,
        enforce_ref_match: bool = False,
        model_kwargs: dict | None = None,
        blocks: str | None = None,
        pairing: str = "map",
        pairing_seed: int | None = None,
        strict_controls: bool = True,
        **kwargs
    ):
        paired_msa = _pair_msas(
            msa1, 
            msa2, 
            max_gap_fraction=max_gap_fraction, 
            interaction_map=interaction_map,
            enforce_ref_match=enforce_ref_match,
            pairing=pairing,
            seed=pairing_seed,
            strict_controls=strict_controls,
        )
        if paired_msa is None:  # failure
            return None, None, None
        return self.run_paired(
            paired_msa,
            msa1_depth=len(msa1),
            msa2_depth=len(msa2) if msa2 is not None else len(msa1),
            cpu=cpu,
            model=model,
            chunksize=chunksize,
            model_kwargs=model_kwargs,
            blocks=blocks,
            **kwargs,
        )

    def run_paired(
        self,
        paired_msa: PairedMSA,
        msa1_depth: int | None = None,
        msa2_depth: int | None = None,
        cpu: bool = False,
        model: Callable | None = None,
        chunksize: int | None = DEFAULT_CHUNKSIZE,
        model_kwargs: dict | None = None,
        blocks: str | None = None,
        **kwargs
    ):
        """Run on an already-paired MSA (so one pairing can feed several methods).

        ``chunksize=None`` (the default) models the whole complex in one call; pass an integer to
        opt in to blockwise inference of at most that many columns per call (an approximation:
        see ``score_mode``).
        """
        if self.stochastic and kwargs.get("seed") is None:
            kwargs["seed"] = random.randrange(2**31 - 1)  # recorded in the metrics
        print_err("[INFO] Generated", paired_msa)
        info_n = paired_msa.pairing_info
        neff = info_n.n_eff if info_n is not None and info_n.n_eff is not None else paired_msa.neff()  # computed once in _pair_msas

        if model is None:
            model = self.make_model(cpu=cpu, **(model_kwargs or {}))

        if self.use_sequences:
            alignment, to_input = _letter_alignment(paired_msa)
        else:
            alignment, to_input = np.asarray(paired_msa.sequence_token_ids), None
        chunk_kwargs = self.prepare(paired_msa, cpu=cpu, **kwargs) | kwargs
        if chunksize is None:
            self._warn_if_too_big(paired_msa, cpu)
        if self.use_id:
            chunk_kwargs["_id"] = paired_msa.name
        results, others = run_blocks(
            alignment,
            paired_msa.chain_a_length,
            interaction_fn=partial(self._run_chunk, model=model, cpu=cpu),
            chunksize=chunksize,
            blocks=blocks or self.default_blocks,
            to_input=to_input,
            **chunk_kwargs,
        )
        results = self.finalize_matrix(paired_msa, results, **kwargs)
        result_interaction = results[:paired_msa.chain_a_length, paired_msa.chain_a_length:]
        scores = score_contact_map(result_interaction)
        info = self.post_run(paired_msa, others, **kwargs)
        metrics = self.metric_container(
            ID=paired_msa.name, 
            seq_len=paired_msa.seq_length,
            chain_a_len=paired_msa.chain_a_length,
            chain_b_len=paired_msa.chain_b_length,
            msa1_depth=msa1_depth if msa1_depth is not None else len(paired_msa),
            msa2_depth=msa2_depth if msa2_depth is not None else len(paired_msa),
            msa_depth=len(paired_msa),
            n_eff=neff,
            **scores,
            **info,
        )
        metrics.uniprot_id_1, metrics.uniprot_id_2 = paired_msa.ids  # IDs may contain hyphens
        metrics.score_mode = "whole" if len(others) == 1 and others[0][0].is_whole else "blockwise"
        if metrics.score_mode == "blockwise":
            print_err(
                f"[WARN] {paired_msa.name}: {paired_msa.seq_length} columns exceed the requested chunksize={chunksize}; "
                "scores are blockwise approximations (score_mode=blockwise) and are not comparable with "
                "whole-complex scores. See benchmarks/chunked_vs_full_dca.py."
            )
        info_p = paired_msa.pairing_info
        if info_p is not None:
            metrics.pairing_method, metrics.pairing_seed = info_p.method, info_p.seed
            metrics.pairing_target_depth, metrics.pairing_depth = info_p.target_pairs, info_p.n_pairs
            metrics.pairing_retained_known, metrics.pairing_valid = info_p.retained_known, info_p.valid
            metrics.pairing_final_depth, metrics.pairing_final_target_depth = info_p.final_pairs, info_p.final_target_pairs
            metrics.pairing_mean_gap = info_p.mean_gap
            metrics.pairing_reference_n_eff, metrics.pairing_reference_mean_gap = info_p.reference_n_eff, info_p.reference_mean_gap
            metrics.pairing_final_multiplicity_ok = info_p.final_multiplicity_ok
            metrics.pairing_final_composition_ok = info_p.final_composition_ok
            metrics.pairing_final_retained_real = info_p.final_retained_real
        return results, result_interaction, metrics
    

class AF2Runner(Runner):

    # Only inter-chain blocks: an intra-chain block of two segments would be modelled as two
    # chains, putting an artificial chain break inside a real chain.
    default_blocks = "interchain"
    metric_container = AF2Metrics
    use_sequences = True
    use_id = True
    stochastic = True

    @staticmethod
    def make_model(
        cpu=False, 
        max_recycles: int = 10,
        param_dir: str | None = None,
        **kwargs
    ):
        from .af2.modelling import make_model_runner
        return make_model_runner(
            max_recycles=max_recycles, 
            param_dir=param_dir, 
            **kwargs,
        )

    @staticmethod
    def _run_chunk(
        paired_msa: ArrayLike,
        chain_a_length: int,
        _id: str,
        model: Callable | None = None,
        seed: int | None = None,
        **kwargs
    ):
        from .af2 import run_af2
        return run_af2(
            paired_msa=paired_msa,
            chain_a_length=chain_a_length,
            _id=_id,
            model=model,
            seed=seed,
        )

    @staticmethod
    def post_run(
        paired_msa: PairedMSA,
        results: list[tuple[Block, tuple]],
        seed: int | None = None,
        **kwargs
    ):
        from .af2 import post_af2
        info = post_af2(paired_msa, results)
        return info | {"seed": seed}


class DCARunner(Runner):
    """DCA is run on raw couplings per block; reweighting and APC use the full alignment.

    Sequence weights are computed once on the whole paired MSA (``prepare``) and shared by
    every block. APC is applied once to the assembled matrix (``finalize_matrix``): applying
    it per block (as before) removes a different background from every block, so blocks
    were not comparable with one another or with the unchunked result.
    """

    metric_container = DCAMetrics

    @staticmethod
    def estimate_memory_gib(n_cols: int, n_rows: int) -> float:
        # shrunk covariance and its Cholesky factor/inverse, (21 L)^2 float32 each, plus the
        # one-hot alignment and its centred copy, (rows x 21 L)
        d = 21 * n_cols
        return (2 * d * d + 2 * n_rows * d) * 4 / 2**30

    @staticmethod
    def prepare(paired_msa: PairedMSA, cpu: bool = False, **kwargs) -> dict:
        from .dca.dca_torch import sequence_weights
        import numpy as np
        return {"weights": sequence_weights(np.asarray(paired_msa.sequence_token_ids), gpu=not cpu)}

    @staticmethod
    def _run_chunk(
        paired_msa: PairedMSA,
        model: Callable | None = None,
        cpu: bool = False,
        weights: ArrayLike | None = None,
        **kwargs
    ):
        from .dca.dca_torch import calculate_dca
        return calculate_dca(
            msa=paired_msa, 
            apc=False,
            gpu=not cpu,
            weights=weights,
        ), {}

    @staticmethod
    def finalize_matrix(paired_msa: PairedMSA, matrix, apc: bool = True, **kwargs):
        from .dca.dca_torch import apc_correction
        return apc_correction(matrix) if apc else matrix

    @staticmethod
    def post_run(
        paired_msa: PairedMSA,
        results: list[tuple[Block, tuple]],
        apc: bool = True,
        **kwargs
    ):
        return {"apc": apc}


class RF2TRunner(Runner):
    
    metric_container = RF2TMetrics

    @staticmethod
    def make_model(cpu=False, **kwargs):
        from rf2t_micro.predict_msa import Predictor
        import torch
        if not cpu and torch.cuda.is_available():
            print_err("[INFO] Using GPU for RF2t")
            torch.cuda.empty_cache()
        else:
             print_err("[INFO] Using CPU for RF2t")
        model = Predictor(use_cpu=cpu)
        return model

    @staticmethod
    def _run_chunk(
        paired_msa: PairedMSA,
        chain_a_length: int,
        model: Callable | None = None,
        **kwargs
    ):
        result, cα_coords = model.predict(
            paired_msa, 
            chain_a_length=chain_a_length,
        )
        return result, {}
    