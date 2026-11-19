"""Batch co-evolution prediction: pair once, load each model once, run many methods.

Design note: why a class?
-------------------------
A function per method (``dca_many_vs_many``, ``model_many_vs_many``...) re-did three things
for every method in a screen: parse the MSAs, pair them, and build the model. With
``yunta coevo --method af2 dca rf2t`` that triples the pairing work, and for AF2 the
model build (JAX compile) dominates small jobs. :class:`CoevoPredictor` owns exactly the
state that is expensive and shared:

* the paired MSAs, built **once** in ``__init__`` (the pairing, including negative
  controls, is independent of the method), and
* the models, built **lazily, once per method**, then reused across the whole batch.

Use it as a context manager (``with CoevoPredictor(...) as p:``) so models are always
released. It deliberately does *not* own scoring or I/O: each method is a small method returning
plain arrays and metric dataclasses, so the class stays a thin batch wrapper over
:class:`~yunta.interactions.runner.Runner` rather than a second framework.

Examples
========
>>> CoevoPredictor.METHODS
('dca', 'rf2t', 'af2')
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from carabiner import print_err
from tqdm.auto import tqdm

from .interactions import AF2Runner, DCARunner, RF2TRunner
from .interactions.runner import DEFAULT_CHUNKSIZE, _pair_msas
from .structs.msa import MSA, PairedMSA
from .structs.pairing import InvalidControlError, log_rejections

_RUNNERS = {"dca": DCARunner, "rf2t": RF2TRunner, "af2": AF2Runner}


@dataclass
class Prediction:
    """Result of one method on one pair."""
    method: str
    matrix: object        # (L, L) full matrix
    interaction: object   # (len_a, len_b) inter-chain block
    metrics: object       # metric dataclass
    paired_msa: PairedMSA


@dataclass
class CoevoPredictor:
    """Prepare a batch of MSA pairs once; predict interactions with any method.

    Parameters
    ----------
    pairs : iterable of (MSA, MSA | None)
        ``None`` as the second MSA means a homodimer.
    interaction_map, enforce_ref_match, max_gap_fraction, pairing, seed
        Passed to the pairing step (see :mod:`yunta.structs.pairing`).
    """

    pairs: Iterable[tuple[MSA, MSA | None]]
    interaction_map: str | Mapping | None = None
    enforce_ref_match: bool = False
    max_gap_fraction: float = .9
    pairing: str = "map"
    seed: int | None = None
    strict_controls: bool = True
    cpu: bool = False
    chunksize: int | None = DEFAULT_CHUNKSIZE  # None: whole complex; an integer opts in to chunking

    METHODS = ("dca", "rf2t", "af2")

    paired: list = field(init=False, default_factory=list)
    depths: list = field(init=False, default_factory=list)
    skipped: list = field(init=False, default_factory=list)
    rejected: dict = field(init=False, default_factory=dict)  # pair index -> why a control was rejected
    _models: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        for i, (msa1, msa2) in enumerate(tqdm(list(self.pairs), desc="Pairing")):
            try:
                paired = _pair_msas(
                    msa1, msa2,
                    max_gap_fraction=self.max_gap_fraction,
                    interaction_map=self.interaction_map,
                    enforce_ref_match=self.enforce_ref_match,
                    pairing=self.pairing,
                    seed=self.seed,
                    strict_controls=self.strict_controls,
                )
            except InvalidControlError as e:  # an invalid control is not scored; the rest of the screen runs
                print_err(f"[WARN] Rejected pair {i}: {e}")
                self.rejected[i] = str(e)
                self.skipped.append(i)
                continue
            if paired is None:
                self.skipped.append(i)
            else:
                self.paired.append(paired)
                self.depths.append((len(msa1), len(msa2) if msa2 is not None else len(msa1)))
        self._log_rejections(n_pairs=len(self.paired) + len(self.skipped))

    @property
    def rejection_rate(self) -> float:
        """Fraction of the submitted pairs whose negative control was rejected (0 for real pairing)."""
        n = len(self.paired) + len(self.skipped)
        return len(self.rejected) / n if n else 0.

    def _log_rejections(self, n_pairs: int) -> None:
        log_rejections(self.pairing, n_pairs, self.rejected)

    @classmethod
    def from_paired(
        cls,
        paired: Iterable[PairedMSA],
        depths: Iterable[tuple[int, int]] | None = None,
        **kwargs,
    ) -> "CoevoPredictor":
        """Start from already-paired MSAs (e.g. read back with :meth:`PairedMSA.read`): no re-pairing."""
        predictor = cls(pairs=[], **kwargs)
        predictor.paired = list(paired)
        predictor.depths = list(depths) if depths is not None else [(len(p), len(p)) for p in predictor.paired]
        return predictor

    @property
    def reports(self) -> list:
        """The :class:`~yunta.structs.pairing.PairingReport` of each prepared pair (seed included)."""
        return [p.pairing_info for p in self.paired]

    def __len__(self) -> int:
        return len(self.paired)

    def close_model(self, method: str) -> None:
        """Release one method's model, calling its ``close()`` (temp files, GPU memory) if it has one."""
        model = self._models.pop(method, None)
        if model is not None and hasattr(model, "close"):
            model.close()

    def close(self) -> None:
        """Release every model."""
        for method in list(self._models):
            self.close_model(method)

    def __enter__(self) -> "CoevoPredictor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def model(self, method: str, **model_kwargs) -> Callable | None:
        """The model for ``method``, built on first use and cached."""
        if method not in self._models:
            self._models[method] = _RUNNERS[method].make_model(cpu=self.cpu, **model_kwargs)
        return self._models[method]

    def predict(self, method: str, model_kwargs: dict | None = None, **kwargs) -> Iterable[Prediction]:
        """Yield a :class:`Prediction` per pair for one method (a generator: results stream)."""
        if method not in _RUNNERS:
            raise ValueError(f"`method` must be one of {self.METHODS}, got {method!r}.")
        runner = _RUNNERS[method]()
        model = self.model(method, **(model_kwargs or {}))  # released by close_model / close / `with`
        for paired, (d1, d2) in zip(self.paired, self.depths):
            matrix, interaction, metrics = runner.run_paired(
                paired, msa1_depth=d1, msa2_depth=d2,
                cpu=self.cpu, model=model, chunksize=self.chunksize, **kwargs,
            )
            yield Prediction(method, matrix, interaction, metrics, paired)

    def predict_all(self, methods: Iterable[str], **kwargs) -> Iterable[Prediction]:
        """Methods one after another, so only one model is resident at a time."""
        for method in methods:
            try:
                yield from self.predict(method, **kwargs.get(method, {}))
            finally:
                self.close_model(method)  # free GPU memory/temp files before the next backend, even on error
