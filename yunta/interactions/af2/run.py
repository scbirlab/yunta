"""."""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Iterable, Mapping, Optional, Union
import os
import random
import sys
from time import time

from carabiner import print_err

if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.typing import ArrayLike
else:
    ndarray, ArrayLike = Any, Any

from .modelling import make_model_runner
from ..chunking import Block
from .scoring import get_contact_matrix
from ...io import save_design
from ...structs.msa import PairedMSA


def _get_af2_features(msa_seqs, chain_a_length, _id) -> Dict[str, Union[str, int]]:

    from .src_speedppi.alphafold.data import foldonly

    # The msas must be str representations of the blocked+paired MSAs here
    # Define the data pipeline
    ids = _id.split("-")
    feature_dict = foldonly.FoldDataPipeline().process(
        input_sequence=msa_seqs[0],  # reference
        input_description="_".join(sorted(ids)),
        input_msas=[msa_seqs],
    )
    # Introduce chain breaks for oligomers
    feature_dict['residue_index'][chain_a_length:] += 200
    feature_dict['ID'] = _id
    return feature_dict


def model_protein_interaction(
    msa_seqs, 
    chain_a_length: int,
    _id: str,
    model: Optional[Callable] = None,
    seed: Optional[int] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None
) -> Tuple[Mapping[str, Union[float, int]], Any, Any]:
    
    """Model a single PPI using a pair of MSA files.
    
    """

    if seed is None:
        seed = random.randrange(2**31 - 1)
    if model is None:
        from .modelling import make_model_runner
        model = make_model_runner(**(model_kwargs or {}))

    feature_dict = _get_af2_features(msa_seqs, chain_a_length, _id)
    if getattr(model, "adaptive_msa", False):  # pad to this alignment's depth, not to 512/5120 rows
        from .modelling import adapt_msa_padding
        adapt_msa_padding(model.config, feature_dict["msa"].shape[0], limits=getattr(model, "msa_limits", None))
    print_err(f"[INFO] Running AF2 on pair {feature_dict['ID']}...")
    # Run the model - on GPU
    t0 = time()
    #TODO: Swap the AlphaFold2 protein modelling for OpenFold (faster? PyTorch, open source)
    print_err(f"[INFO] Processing features...")
    processed_feature_dict = model.process_features(
        feature_dict, 
        random_seed=seed,
    )
    print_err(f"[INFO] Doing forward pass...")
    result = model.predict(processed_feature_dict)
    print_err(f"[INFO] It took AF2 {(time() - t0):.2f} s to predict the interaction.")
    return feature_dict, processed_feature_dict, result


def get_unrelaxed_protein(
    processed_feature_dict: Mapping[str, Any],
    prediction_result: Mapping[str, Any]
):
    """Evalulate a model and save PDB.
    
    """
    import numpy as np
    from .src_speedppi.alphafold import protein, residue_constants
    plddt_b_factors = np.repeat(
        prediction_result['plddt'][:, np.newaxis], 
        residue_constants.atom_type_num, 
        axis=-1,
    )
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    print_err(f"[INFO] Predicting PDB structure...")
    return protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
    )

def _compact(prediction_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Reduce the raw AF2 result to the canonical record (see ``confidence.compact_prediction``).

    The raw result holds ``(L, L, 64)`` PAE and distogram probabilities per block; scoring
    needs only ``(L, L)`` matrices, so those are reduced here.
    """
    import numpy as np
    from .. import confidence
    probs = prediction_result.get("aligned_confidence_probs")
    distogram = prediction_result.get("distogram")
    contact_prob = None
    if distogram is not None:
        from scipy.special import softmax
        contact_prob = _full_contact_prob(
            softmax(np.asarray(distogram["logits"]), axis=-1), np.asarray(distogram["bin_edges"]),
        )
    return confidence.compact_prediction(
        prediction_result["plddt"],
        pae_probs=None if probs is None else np.asarray(probs),
        contact_prob=contact_prob,
        ptm=float(prediction_result["ptm"]) if "ptm" in prediction_result else None,
    )


def _full_contact_prob(probs, bin_edges, cutoff: float = 8.):
    """``P(d < cutoff)`` for every residue pair, ``(L, L)``."""
    import numpy as np
    n = int((bin_edges <= cutoff).sum())
    return probs[..., :n].sum(axis=-1).astype(np.float32)


def af2(
    paired_msa: ArrayLike,
    chain_a_length: int,
    _id: str,
    model: Optional[Callable] = None,
    seed: Optional[int] = None,
    return_structure: bool = False,
) -> ndarray:
    
    feature_dict, processed_feature_dict, prediction_result = model_protein_interaction(
        msa_seqs=paired_msa,
        chain_a_length=chain_a_length,
        _id=_id,
        model=model,
        seed=seed,
    )
    unrelaxed_protein = get_unrelaxed_protein(
        processed_feature_dict, 
        prediction_result,
    )
    inv_contact_dists, contact_dists = get_contact_matrix(
        unrelaxed_protein,
    )
    compact = _compact(prediction_result)
    if return_structure:  # PDB text, pLDDT in the B-factor column, chains split at the break
        from .src_speedppi.alphafold import protein
        from ...io import assign_chains
        pdb, _ = protein.to_pdb(unrelaxed_protein)
        compact["structure"] = assign_chains(pdb, chain_a_length)
    return inv_contact_dists, contact_dists, compact


def post_af2(
    paired_msa: PairedMSA,
    results: List[Tuple[Block, tuple]],
    contact_radius: float = 8.,
    pae_cutoff: float = 10.,
    **kwargs
) -> Dict[str, Any]:
    """Score the interface from the blocks' Cβ distances, pLDDT and (if whole) PAE.

    ``results`` is ``[(block, (contact_dists, compact_prediction)), ...]``. Each block's
    pLDDT is averaged over every block that contains the residue (not last-write-wins),
    and unseen residue pairs are ``inf`` apart, so they can never count as contacts.
    pTM, ipTM and actifpTM need one model's PAE/distogram probabilities, so they are only defined
    for a whole-complex call (NaN otherwise). The PAE-based scores (ipSAE, pDockQ2, LIS, mean
    inter-chain PAE) are computed from the assembled PAE when chunked, as **blockwise
    approximations**; the metrics' ``score_mode`` column says which case applies. Contact and
    pLDDT scores are likewise blockwise when chunked.
    """
    from .. import confidence
    scores, one = confidence.score_blocks(
        paired_msa, results, contact_radius=contact_radius, pae_cutoff=pae_cutoff,
    )
    scores["ptm_af2"] = one.get("ptm")  # AF2's own pTM; "ptm" is recomputed by the shared code
    if "structure" in one:
        scores["structure"] = one["structure"]
    return scores
