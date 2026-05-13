from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional
if TYPE_CHECKING:
    from numpy import ndarray
else:
    ndarray = Any

from ...structs.msa import PairedMSA


def make_rf2t_model(cpu: bool = True, **kwargs) -> Callable:
    from rf2t_micro.predict_msa import Predictor
    import torch
    if not cpu:
        torch.cuda.empty_cache()
    model = Predictor(use_cpu=cpu)
    return model


def rf2track(
    paired_msa: PairedMSA,
    model: Optional[Callable] = None
) -> ndarray:
    
    result, cα_coords = model.predict(
        m, 
        chain_a_length=paired_msa.chain_a_length,
    )
    return result, {}
