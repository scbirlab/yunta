"""Data structures."""

from typing import Any, Dict, Iterable

from dataclasses import dataclass, field


@dataclass
class DCAMetrics:
     
    """Storage for DCA metrics.
    """
    
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
    apc: bool
    mean: float
    median: float
    maximum: float
    minimum: float

    def __post_init__(self):
        self.uniprot_id_1, self.uniprot_id_2 = self.ID.split('-')


@dataclass
class RF2TMetrics:

    """Storage for RosettaFold 2-track metrics.
    """
    
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
    mean: float
    median: float
    maximum: float
    minimum: float

    def __post_init__(self):
        self.uniprot_id_1, self.uniprot_id_2 = self.ID.split('-')


@dataclass
class ModelMetrics:
     
    """Storage for AlphaFold2 model metrics.
    """
    
    ID: str
    uniprot_id_1: str = field(init=False)
    uniprot_id_2: str = field(init=False)
    n_contacts: int
    mean_interface_plddt: float
    pdockq: float

    def __post_init__(self):
        self.uniprot_id_1, self.uniprot_id_2 = self.ID.split('-')