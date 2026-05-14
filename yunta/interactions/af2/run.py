"""."""
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple, Iterable, Mapping, Optional, Union
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
from .scoring import _post_score_ppi, get_contact_matrix
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
        seed = random.randrange(sys.maxsize)
    if model is None:
        from .modelling import make_model_runner
        model = make_model_runner(**(model_kwargs or {}))

    feature_dict = _get_af2_features(msa_seqs, chain_a_length, _id)
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
    print_err(f"[INFO] It took AF2 {time() - t0} s to predict the interaction.")
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

def af2(
    paired_msa: PairedMSA,
    chain_a_length: int,
    _id: str,
    model: Optional[Callable] = None,
    seed: Optional[int] = None
) -> ndarray:
    
    feature_dict, processed_feature_dict, prediction_result = model_protein_interaction(
        msa_seqs=paired_msa,
        chain_a_length=chain_a_length,
        _id=_id,
        model=model,
        seed=seed
    )
    unrelaxed_protein = get_unrelaxed_protein(
        processed_feature_dict, 
        prediction_result,
    )
    inv_contact_dists, contact_dists = get_contact_matrix(
        unrelaxed_protein,
    )
    return inv_contact_dists, contact_dists, prediction_result


def post_af2(
    paired_msa: PairedMSA,
    results: Dict[Tuple, Any]
) -> ndarray:
    import numpy as np
    if len(results) == 1:
        (i0, j0, n, m), (contact_block, plddt_block) = next(iter(results.items()))
        contact_dist = contact_block
        plddt = plddt_block["plddt"]
    else:
        n_msa_columns = paired_msa.seq_length
        contact_dist = np.zeros(
            (n_msa_columns, n_msa_columns), 
            dtype=np.float32,
        )
        plddt = np.zeros(
            (n_msa_columns,), 
            dtype=np.float32,
        )
        for (i0, j0, n, m), (contact_block, plddt_block) in results.items():
            si, sj = slice(i0, i0 + n), slice(j0, j0 + m)
            plddt_block = plddt_block["plddt"]
            if i0 == j0:
                contact_dist[si, si] = contact_block
                plddt[si] = plddt_block[:n]
            else:
                contact_dist[si, sj] = contact_block[:n, n:]
                contact_dist[sj, si] = contact_block[n:, :n]
                plddt[sj] = plddt_block[n:]
    contact_dist_interaction = contact_dist[:paired_msa.chain_a_length, paired_msa.chain_a_length:]
    return _post_score_ppi(
        contact_dist_interaction,
        plddt,
        chain_a_length=paired_msa.chain_a_length,
        contact_radius=8.,
    )


# TODO: Wire this back in one day
# Needs function parameters to be updated.
def save_pdb(
    paired_msa: PairedMSA,
    model: Optional[Callable] = None,
    seed: Optional[int] = None,
    output_dir: os.PathLike = "."
):
    from .src_speedppi.alphafold import protein
    feature_dict, processed_feature_dict, prediction_result = model_protein_interaction(
        msa_seqs=paired_msa,
        chain_a_length=paired_msa.chain_a_length,
        _id=paired_msa.name,
        model=model,
        seed=seed,
    )
    unrelaxed_protein = get_unrelaxed_protein(
        processed_feature_dict, 
        prediction_result,
    )
    #Save if pDockQ > t
    filename = os.path.join(output_dir, f"{feature_dict['ID']}.pdb")
    print_err(f"[INFO] Saving {feature_dict['ID']} as {filename}.")
    pdb, _ = protein.to_pdb(unrelaxed_protein)
    return save_design(pdb, filename, paired_msa.chain_a_length)


