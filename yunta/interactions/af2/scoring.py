
from typing import Optional, Tuple
from numpy import ndarray
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist


# def _euclidean_dist(x, y=None):
#     if y is None:
#         y = x
#     x, y = x[:, np.newaxis, :], y[np.newaxis, :, :]    # (N,1,D) and (1,M,D)
#     return np.sqrt(np.sum(np.square(x - y), axis=-1))  # (N, M)


def _pdockq(
    avg_interface_plddt: float, 
    n_interface_contacts: int
):
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


def get_contact_matrix(unrelaxed_protein) -> Tuple[ndarray, ...]:
    from .src_speedppi.alphafold import protein
    #Get the pdb and Cβ coords
    _, cβ_coords = protein.to_pdb(unrelaxed_protein)
    contact_dists = cdist(cβ_coords, cβ_coords)
    inv_contact_dists = 1. / contact_dists
    return inv_contact_dists, contact_dists


def _post_score_ppi(
    contact_dists: ArrayLike,
    plddt: ArrayLike,
    chain_a_length: int,
    contact_radius: float = 8.
):
    contacts = np.argwhere(contact_dists <= contact_radius)
    n_contacts = contacts.shape[0]
    info = {
        "n_contacts": n_contacts,
        "mean_interface_plddt": 0.,
        "pdockq": 0.,
    }
    if n_contacts >= 1:  # no contacts
        #Get plddt per chain
        plddt1, plddt2 = plddt[:chain_a_length], plddt[chain_a_length:]
        #Get the average interface plDDT
        _plddt = np.concatenate([
            p[np.unique(contacts[:,i])] 
            for i, p in enumerate([plddt1, plddt2])
        ])
        mean_interface_plddt = np.mean(_plddt)
        info |= {
            "mean_interface_plddt": mean_interface_plddt, 
            "pdockq": _pdockq(mean_interface_plddt, info["n_contacts"]),
        }
    return info
