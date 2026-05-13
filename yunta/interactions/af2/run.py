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


def _get_af2_features(paired_msa: PairedMSA) -> Dict[str, Union[str, int]]:

    from ...src_speedppi.alphafold.data import foldonly

    msa_seqs = paired_msa.sequences()
    # The msas must be str representations of the blocked+paired MSAs here
    # Define the data pipeline
    ids = list(getattr(name, "unique_id") for name in paired_msa.lines[0].name)
    feature_dict = foldonly.FoldDataPipeline().process(
        input_sequence=msa_seqs[0],  # reference
        input_description="_".join(sorted(ids)),
        input_msas=[msa_seqs],
    )
    # Introduce chain breaks for oligomers
    feature_dict['residue_index'][paired_msa.chain_a_length:] += 200
    feature_dict['ID'] = "-".join(sorted(ids))
    feature_dict['uniprot_id_1'], feature_dict['uniprot_id_2'] = ids
    return feature_dict


def model_protein_interaction(
    paired_msa: PairedMSA,
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

    feature_dict = _get_af2_features(paired_msa)
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
    from ...src_speedppi.alphafold import protein, residue_constants
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
    model: Optional[Callable] = None,
    seed: Optional[int] = None
) -> ndarray:
    
    feature_dict, processed_feature_dict, prediction_result = model_protein_interaction(
        paired_msa=paired_msa,
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
            contact_dist[si, si] = contact_block[:n, :n]
            contact_dist[si, sj] = contact_block[:n, n:]
            contact_dist[sj, si] = contact_block[n:, :n]
            contact_dist[sj, sj] = contact_block[n:, n:]
            plddt_block = plddt_block["plddt"]
            plddt[si], plddt[sj] = plddt_block[:n], plddt_block[n:]
    return _post_score_ppi(
        contact_dist,
        plddt,
        chain_a_length=paired_msa.chain_a_length,
        contact_radius=8.,
    )

def save_pdb(
    paired_msa: PairedMSA,
    model: Optional[Callable] = None,
    seed: Optional[int] = None,
    output_dir: os.PathLike = "."
):
    from ...src_speedppi.alphafold import protein
    feature_dict, processed_feature_dict, prediction_result = model_protein_interaction(
        paired_msa=paired_msa,
        model=model,
        seed=seed
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


