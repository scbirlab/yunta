from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Tuple, Union
if TYPE_CHECKING:
    from numpy import ndarray
else:
    ndarray = Any

from .scoring import score_contact_map
from ..structs.metrics import RF2TMetrics
from ..structs.msa import MSA, PairedMSA

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
    chunksize: int = 750,
    **kwargs
) -> np.ndarray:

    token_ids = np.asarray(paired_msa.sequence_token_ids)
    n_msa_columns = token_ids.shape[-1]

    if n_msa_columns > chunksize:
        print_err(f"[INFO] Splitting MSA with {n_msa_columns} columns into pairs of {chunksize}-column chunks.")
        chunks = np.split(token_ids, list(range(chunksize, n_msa_columns, chunksize)), axis=-1)
        n_chunks = len(chunks)
        print_err(f"[INFO] Split MSA with {n_msa_columns} columns into {n_chunks} x {chunksize}-column chunks.")

        result = np.zeros((n_msa_columns, n_msa_columns), dtype=np.float32)

        for (i, chunk_i), (j, chunk_j) in combinations(enumerate(chunks), 2):
            i0, j0 = i * chunksize, j * chunksize
            n = chunk_i.shape[-1]
            si, sj = slice(i0, i0 + n), slice(j0, j0 + chunk_j.shape[-1])
            block = interaction_fn(
                np.concatenate([chunk_i, chunk_j], axis=-1),
                chain_a_length=max(0, paired_msa.chain_a_length - i0),
                **kwargs,
            )
            result[si, si], result[si, sj] = block[:n, :n], block[:n, n:]
            result[sj, si], result[sj, sj] = block[n:, :n], block[n:, n:]
    else:
        result, info = interaction_fn(
            token_ids, 
            chain_a_length=paired_msa.chain_a_length,
            **kwargs,
        )
    return result, info


def calculate_interaction(
    msa1: MSA,
    interaction_fn: Callable,
    model_factory: Callable,
    metric_container: Metrics,
    msa2: Optional[MSA] = None,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    cpu: bool = True,
    model: Optional[Callable] = None,
    chunksize: int = 1500,
    enforce_ref_match: bool = False,
    **kwargs
) -> Tuple[ndarray, ndarray, RF2TMetrics]:
    
    paired_msa = _pair_msas(
        msa1, 
        msa2, 
        max_gap_fraction=max_gap_fraction, 
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
    )
    print_err("[INFO] Generated", paired_msa)
    chain_a_length = paired_msa.chain_a_length
    chain_b_length = paired_msa.chain_b_length
    neff = paired_msa.neff()

    if model is None:
        model = model_factory(cpu=cpu)

    result = _calculate_interaction_blocks(
        paired_msa,
        interaction_fn=interaction_fn,
        chunksize=chunksize,
        **kwargs
    )
    
    result_interaction = result[:chain_a_length, chain_a_length:]
    scores = score_contact_map(result_interaction)
    metrics = metric_container(
        ID=paired_msa.name, 
        seq_len=paired_msa.seq_length,
        chain_a_len=paired_msa.chain_a_length,
        chain_b_len=paired_msa.chain_b_length,
        msa1_depth=len(msa1),
        msa2_depth=len(msa2) if msa2 is not None else len(msa1),
        msa_depth=len(paired_msa),
        n_eff=neff,
        **scores,
    )
    return result, result_interaction, metrics
