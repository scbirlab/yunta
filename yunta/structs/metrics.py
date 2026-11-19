"""Data structures."""
from typing import ClassVar, Iterable, Union
from csv import DictWriter
from dataclasses import asdict, dataclass, field, fields
from io import TextIOWrapper
import os

from carabiner.cast import cast


@dataclass
class InteractionMetrics:
    """Storage for PPI metrics.
    """
    _name: ClassVar[str] = "__no_method_specified__"
    ID: str
    uniprot_id_1: str = field(init=False)
    uniprot_id_2: str = field(init=False)
    seq_len: int
    chain_a_len: int
    chain_b_len: int
    msa1_depth: int
    msa2_depth: int
    msa_depth: int
    n_eff: int
    mean: float = field(metadata={"named": True})
    median: float = field(metadata={"named": True})
    maximum: float = field(metadata={"named": True})
    minimum: float = field(metadata={"named": True})
    var: float = field(metadata={"named": True})
    sigma1: float = field(metadata={"named": True})
    focality: float = field(metadata={"named": True})
    top_A: Iterable[int] = field(metadata={"named": True})
    top_B: Iterable[int] = field(metadata={"named": True})
    # weights_A: Iterable[float] = field(metadata={"named": True})
    # weights_B: Iterable[float] = field(metadata={"named": True})
    # Provenance, filled in by the runner. ``score_mode`` is "whole" when the complex went through
    # the method in one call, and "blockwise" when it was chunked: chunked scores come from
    # independent sub-models and are approximations, not values for any one physical complex.
    score_mode: str = field(init=False, default="whole")
    pairing_method: str = field(init=False, default="map")
    pairing_seed: int = field(init=False, default=None)
    pairing_target_depth: int = field(init=False, default=None)
    pairing_depth: int = field(init=False, default=None)
    pairing_retained_known: int = field(init=False, default=None)
    pairing_valid: bool = field(init=False, default=True)
    # The alignment actually scored (after the gap filter), for the real pairing and, for a
    # control, the real reference put through the same filter.
    pairing_final_depth: int = field(init=False, default=None)
    pairing_final_target_depth: int = field(init=False, default=None)
    pairing_mean_gap: float = field(init=False, default=None)
    pairing_reference_n_eff: float = field(init=False, default=None)
    pairing_reference_mean_gap: float = field(init=False, default=None)
    pairing_final_multiplicity_ok: bool = field(init=False, default=None)
    pairing_final_composition_ok: bool = field(init=False, default=None)
    pairing_final_retained_real: int = field(init=False, default=None)

    def __post_init__(self):
        # Best guess from the ID; callers that know the true IDs (which may contain
        # hyphens, e.g. isoforms) overwrite these.
        parts = self.ID.split('-')
        mid = len(parts) // 2
        self.uniprot_id_1, self.uniprot_id_2 = '-'.join(parts[:mid]), '-'.join(parts[mid:])

    def _get_named_params(self):
        return {f.name for f in fields(self) if f.metadata.get("named")}

    def write(
        self,
        filename: os.PathLike,
        mode: str = "w",
        skip_header: bool = False
    ) -> None:
        opened_here = not isinstance(filename, TextIOWrapper)
        f = cast(filename, to=TextIOWrapper, mode=mode)
        if hasattr(f, 'name') and not f.name.startswith('<'):
            if dirname := os.path.dirname(f.name):
                os.makedirs(dirname, exist_ok=True)
        named = self._get_named_params()
        fieldnames = [
            f"{self._name}:{name}" if name in named
            else name 
            for name in asdict(self)
        ]
        w = DictWriter(
            f, 
            fieldnames=fieldnames, 
            delimiter="\t",
        )
        if not skip_header:
            w.writeheader()
        row = {
            (f"{self._name}:{k}" if k in named else k): v
            for k, v in asdict(self).items()
        }
        w.writerow(row)
        if opened_here:  # was never closed, so output relied on garbage collection to flush
            f.close()
        return None


@dataclass
class DCAMetrics(InteractionMetrics):
    """Storage for DCA metrics.
    """
    _name: ClassVar[str] = "DCA"
    apc: bool = field(metadata={"named": True})


@dataclass
class RF2TMetrics(InteractionMetrics):
    """Storage for RosettaFold 2-track metrics.
    """
    _name: ClassVar[str] = "RF2t"


@dataclass
class OF3Metrics(InteractionMetrics):
    """Storage for OpenFold3 model metrics (pTM/ipTM/actifpTM computed by yunta; ``*_model`` are OpenFold3's own)."""
    _name: ClassVar[str] = "OF3"
    n_contacts: int = field(metadata={"named": True})
    mean_plddt_a: float = field(metadata={"named": True})
    mean_plddt_b: float = field(metadata={"named": True})
    mean_interface_plddt: float = field(metadata={"named": True})
    pdockq: float = field(metadata={"named": True})
    pdockq2: float = field(metadata={"named": True})
    ptm: float = field(metadata={"named": True})
    iptm: float = field(metadata={"named": True})
    actifptm: float = field(metadata={"named": True})
    ipsae: float = field(metadata={"named": True})
    lis: float = field(metadata={"named": True})
    mean_interchain_pae: float = field(metadata={"named": True})
    ptm_model: float = field(metadata={"named": True})
    iptm_model: float = field(metadata={"named": True})
    seed: int = field(metadata={"named": True})


@dataclass
class AF2Metrics(InteractionMetrics):
    """Storage for AlphaFold2 model metrics.
    """
    _name: ClassVar[str] = "AF2"
    n_contacts: int = field(metadata={"named": True})
    mean_plddt_a: float = field(metadata={"named": True})
    mean_plddt_b: float = field(metadata={"named": True})
    mean_interface_plddt: float = field(metadata={"named": True})
    pdockq: float = field(metadata={"named": True})
    pdockq2: float = field(metadata={"named": True})
    ptm: float = field(metadata={"named": True})
    iptm: float = field(metadata={"named": True})
    actifptm: float = field(metadata={"named": True})
    ipsae: float = field(metadata={"named": True})
    lis: float = field(metadata={"named": True})
    mean_interchain_pae: float = field(metadata={"named": True})
    ptm_af2: float = field(metadata={"named": True})
    seed: int = field(metadata={"named": True})
    # max_recycles: int = field(metadata={"named": True})
