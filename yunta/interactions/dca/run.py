from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional
if TYPE_CHECKING:
    from numpy import ndarray
else:
    ndarray = Any

from .dca_torch import calculate_dca
from ...structs.msa import PairedMSA


def make_dca_model(cpu=True, , **kwargs): 
    return None


def paired_dca(
    paired_msa: PairedMSA,
    model: Optional[Callable] = None,
    apc: bool = False
) -> ndarray:
    return calculate_dca(
        msa=paired_msa, 
        apc=apc,
    ), {}
