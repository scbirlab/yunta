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

    def __post_init__(self):
        self.uniprot_id_1, self.uniprot_id_2 = self.ID.split('-')

    def _get_named_params(self):
        return {f.name for f in fields(self) if f.metadata.get("named")}

    def write(
        self,
        filename: os.PathLike,
        mode: str = "w",
        skip_header: bool = False
    ) -> None:
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
class AF2Metrics(InteractionMetrics):
    """Storage for AlphaFold2 model metrics.
    """
    _name: ClassVar[str] = "AF2"
    n_contacts: int = field(metadata={"named": True})
    mean_interface_plddt: float = field(metadata={"named": True})
    pdockq: float = field(metadata={"named": True})
    seed: int = field(metadata={"named": True})
    max_recycles: int = field(metadata={"named": True})
