"""Running interaction calculations."""
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from itertools import combinations

from carabiner import print_err
if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.typing import ArrayLike
else:
    ndarray, ArrayLike = Any, Any

from .scoring import score_contact_map
from ..structs.metrics import AF2Metrics, DCAMetrics, RF2TMetrics, InteractionMetrics
from ..structs.msa import MSA, PairedMSA

DEFAULT_CHUNKSIZE: int = 750

def _pair_msas(
    msa1: MSA, 
    msa2: Optional[MSA] = None,
    max_gap_fraction: float = 1.,
    blocked: bool = False,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False
) -> PairedMSA:
    return (
        PairedMSA.from_msa(
            msa1, 
            msa2, 
            blocked=blocked, 
            interaction_map=interaction_map,
            enforce_ref_match=enforce_ref_match,
        )
        .filter_by_gap_fraction(max_gap_fraction)
    )


def _calculate_interaction_blocks(
    paired_msa: PairedMSA,
    interaction_fn: Callable,
    chunksize: int = DEFAULT_CHUNKSIZE,
    use_sequences: bool = False,
    use_id: bool = False,
    **kwargs
) -> Tuple[ndarray, dict]:

    import numpy as np
    from tqdm.auto import tqdm

    if not use_sequences:
        token_ids = np.asarray(paired_msa.sequence_token_ids)
    else:
        token_ids = np.asarray([list(line) for line in paired_msa.sequences()])
    if use_id:
        kwargs |= {"_id": paired_msa.name}
    n_msa_columns = token_ids.shape[-1]

    if n_msa_columns > chunksize:
        print_err(f"[INFO] Splitting MSA with {n_msa_columns} columns into pairs of {chunksize}-column chunks.")
        chunks = np.split(token_ids, list(range(chunksize, n_msa_columns, chunksize)), axis=-1)
        n_chunks = len(chunks)
        n_blocks = n_chunks * n_chunks
        print_err(f"[INFO] Split MSA with {n_msa_columns} columns into {n_chunks} x {chunksize}-column chunks ({n_blocks} blocks).")

        result = np.zeros(
            (n_msa_columns, n_msa_columns), 
            dtype=np.float32,
        )
        _others = defaultdict(list)
        
        # Pass 1: diagonal blocks — each chunk vs itself
        for i, chunk_i in enumerate(tqdm(
            chunks, 
            total=n_chunks, 
            desc="Running diagonal blocks",
        )):
            i0 = i * chunksize
            n = chunk_i.shape[-1]
            si = slice(i0, i0 + n)
            block, *others = interaction_fn(
                chunk_i,
                chain_a_length=max(0, min(n, paired_msa.chain_a_length - i0)),
                **kwargs,
            )
            result[si, si] = block
            for o in others:
                _others[(i0, i0, n, n)].append(o)
        # Pass 2: off-diagonal blocks
        for (i, chunk_i), (j, chunk_j) in tqdm(
            combinations(enumerate(chunks), 2), 
            total=n_chunks * (n_chunks - 1) // 2, 
            desc="Running off-diagonal blocks",
        ):
            i0, j0 = i * chunksize, j * chunksize
            n, m = chunk_i.shape[-1], chunk_j.shape[-1]
            si, sj = slice(i0, i0 + n), slice(j0, j0 + m)
            block, *others = interaction_fn(
                np.concatenate([chunk_i, chunk_j], axis=-1),
                chain_a_length=max(0, paired_msa.chain_a_length - i0),
                **kwargs,
            )
            result[si, sj] = block[:n, n:]
            result[sj, si] = block[n:, :n]
            for o in others:
                _others[(i0,j0,n,m)].append(o)
    else:
        result, *others = interaction_fn(
            token_ids, 
            chain_a_length=paired_msa.chain_a_length,
            **kwargs,
        )
        _others = defaultdict(list)
        for o in others:
            _others[(0,0,n_msa_columns,n_msa_columns)].append(o)
    return result, _others


class Runner(ABC):

    metric_container = InteractionMetrics
    use_sequences = False
    use_id = False

    @staticmethod
    def make_model(cpu=True, **kwargs):
        return None

    @staticmethod
    @abstractmethod
    def _run_chunk(
        paired_msa: PairedMSA,
        model: Optional[Callable] = None,
        **kwargs
    ) -> Iterable[ArrayLike]:
        ...

    @staticmethod
    def post_run(
        paired_msa: PairedMSA,
        results: Dict[Tuple, Any],
        **kwargs
    ) -> dict:
        return {}

    def run(
        self,
        msa1: MSA,
        msa2: Optional[MSA] = None,
        max_gap_fraction: float = .9,
        interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
        cpu: bool = True,
        model: Optional[Callable] = None,
        chunksize: int = DEFAULT_CHUNKSIZE,
        enforce_ref_match: bool = False,
        model_kwargs: Optional[dict] = None,
        **kwargs
    ):
        paired_msa = _pair_msas(
            msa1, 
            msa2, 
            max_gap_fraction=max_gap_fraction, 
            interaction_map=interaction_map,
            enforce_ref_match=enforce_ref_match,
        )
        print_err("[INFO] Generated", paired_msa)
        neff = paired_msa.neff()

        if model is None:
            model = self.make_model(cpu=cpu, **(model_kwargs or {}))

        results, others = _calculate_interaction_blocks(
            paired_msa,
            interaction_fn=partial(self._run_chunk, model=model),
            chunksize=chunksize,
            use_sequences=self.use_sequences,
            use_id=self.use_id,
            **kwargs
        )
        result_interaction = results[:paired_msa.chain_a_length, paired_msa.chain_a_length:]
        info = self.post_run(paired_msa, others, **kwargs)
        scores = score_contact_map(result_interaction)
        metrics = self.metric_container(
            ID=paired_msa.name, 
            seq_len=paired_msa.seq_length,
            chain_a_len=paired_msa.chain_a_length,
            chain_b_len=paired_msa.chain_b_length,
            msa1_depth=len(msa1),
            msa2_depth=len(msa2) if msa2 is not None else len(msa1),
            msa_depth=len(paired_msa),
            n_eff=neff,
            **scores,
            **info,
        )
        return results, result_interaction, metrics
    

class AF2Runner(Runner):

    metric_container = AF2Metrics
    use_sequences = True
    use_id = True

    @staticmethod
    def make_model(
        cpu=True, 
        max_recycles: int = 10,
        param_dir: Optional[str] = None,
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
        model: Optional[Callable] = None,
        seed: Optional[int] = None,
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
        results: Dict[Tuple, Any],
        seed: Optional[int] = None,
        **kwargs
    ):
        from .af2 import post_af2
        info = post_af2(paired_msa, results, **kwargs)
        return info | {"seed": seed}


class DCARunner(Runner):

    metric_container = DCAMetrics

    @staticmethod
    def _run_chunk(
        paired_msa: PairedMSA,
        model: Optional[Callable] = None,
        apc: bool = True,
        **kwargs
    ):
        from .dca.dca_torch import calculate_dca
        return calculate_dca(
            msa=paired_msa, 
            apc=apc,
        ), {}

    @staticmethod
    def post_run(
        paired_msa: PairedMSA,
        results: Dict[Tuple, Any],
        apc: bool = True,
        **kwargs
    ):
        return {"apc": apc}



class RF2TRunner(Runner):
    
    metric_container = RF2TMetrics

    @staticmethod
    def make_model(cpu=True, **kwargs):
        from rf2t_micro.predict_msa import Predictor
        import torch
        if not cpu:
            torch.cuda.empty_cache()
        model = Predictor(use_cpu=cpu)
        return model

    @staticmethod
    def _run_chunk(
        paired_msa: PairedMSA,
        chain_a_length: int,
        model: Optional[Callable] = None,
        **kwargs
    ):
        result, cα_coords = model.predict(
            paired_msa, 
            chain_a_length=chain_a_length,
        )
        return result, {}
    