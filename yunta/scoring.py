"""Tools to score and analyze PPIs."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


def score_contact_map(M: np.ndarray, top_n: int = 10) -> dict:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    weightsA = U[:, 0]
    weightsB = Vt[0, :]
    return {
        "mean": np.mean(M),
        "median": np.median(M),
        "var": np.var(M),
        "minimum": np.min(M),
        "maximum": np.max(M),
        "sigma1":   s[0],
        "focality": s[0] / s[1] if s[1] > 0 else np.inf,
        "top_A": np.flatnonzero(-weightsA < np.sort(-weightsA)[top_n]),
        "top_B": np.flatnonzero(-weightsB < np.sort(-weightsB)[top_n]),
        "weights_A": weightsA,   # per-residue weights, protein A
        "weights_B": weightsB,  # per-residue weights, protein B
    }


def _euclidean_dist(
    x: ArrayLike, 
    y: Optional[ArrayLike] = None
) -> float:
    if y is None:
        y = x
    x, y = x[...,np.newaxis], y[...,np.newaxis,:]
    return np.sqrt(np.sum(np.square(x - y),
                   axis=-1))


def _pdockq(avg_interface_plddt: float, n_interface_contacts: int):

    """Calculate the pDockQ.

    Examples
    --------
    >>> import numpy as np; np.set_printoptions(legacy="1.25")
    >>> _pdockq(.1, 3)
    0.018259536451737016
    >>> _pdockq(.9, 100)
    0.018284286329741297
    """

    x = avg_interface_plddt * np.log10(n_interface_contacts)
    return .724 / (1 + np.exp(-.052 * (x - 152.611))) + .018


def _score_ppi(
    cb_coords: ArrayLike, 
    plddt: ArrayLike,  
    chain_a_length: int,
    contact_radius: float = 8.
) -> Tuple[float, float, int]:
    
    #Cβs within 8 Å from each other from different chains are used to define the interface.
    cβ_dists = _euclidean_dist(cb_coords)

    #Get contacts
    contact_dists = cβ_dists[:chain_a_length,chain_a_length:] #upper triangular --> first dim = chain 1
    contacts = np.argwhere(contact_dists <= contact_radius)

    base_plddt_stats = {
        "mean": 0.,
        "median": 0.,
        "var": 0.,
        "minimum": 0.,
        "maximum": 0.,
    }
    if contacts.shape[0] < 1:  # no contacts
        pdockq, avg_interface_plddt, n_interface_contacts = 0., base_plddt_stats, 0
    else:
        #Get plddt per chain
        plddt1, plddt2 = plddt[:chain_a_length], plddt[chain_a_length:]
        #Get the average interface plDDT
        _plddt = np.concatenate([
            p[np.unique(contacts[:,i])] 
            for i, p in enumerate([plddt1, plddt2])
        ])
        inv_contact_dists = 1. / contact_dists
        avg_interface_plddt = score_contact_map(inv_contact_dists) | {
            "mean_plddt": np.mean(_plddt),
        } 
        #Get the number of interface contacts
        n_interface_contacts = contacts.shape[0]
        pdockq = _pdockq(avg_interface_plddt["mean_plddt"], n_interface_contacts)

    return pdockq, avg_interface_plddt, n_interface_contacts


def score_ppi(unrelaxed_protein, plddt, chain_a_length: int) -> Tuple[float, float, int]:

    """Score the PPI.

    """
    from .src_speedppi.alphafold import protein

    #Get the pdb and Cβ coords
    _, cβ_coords = protein.to_pdb(unrelaxed_protein)
    #Score - calculate the pDockQ (number of interface residues and average interface plDDT)
    return _score_ppi(cβ_coords, plddt, chain_a_length)

    