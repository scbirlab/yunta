"""DCA for GPU implemented in pytorch."""

from typing import Optional, Union

from io import TextIOWrapper
import sys

from carabiner import print_err
import numpy as np
import torch
from torch import FloatTensor, Tensor
import torch.nn.functional as F

from .structs.msa import MSA, _A3M_ALPHABET

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _torch_cov(
    x: Tensor, 
    w: Optional[Tensor] = None
) -> FloatTensor:
    if w is None:
        return torch.cov(x)
    else:
        num_points = torch.sum(w) - torch.sqrt(torch.mean(w))
        x_mean = torch.sum(
            x * w.unsqueeze(-1), 
            dim=0, keepdim=True
        ) / num_points
        x = (x - x_mean) * torch.sqrt(w.unsqueeze(-1))
        return torch.matmul(x.transpose(-2, -1), x) / num_points


def _get_wip(
    x: Tensor, 
    apc: bool = False, 
    gpu: bool = True, 
    dtype=torch.float16
) -> FloatTensor:
    device = torch.device("cuda" if( torch.cuda.is_available() and gpu) else "cpu")
    # x = x.to(device)
    alphabet_size = torch.tensor(len(_A3M_ALPHABET), device=device)
    n_row, n_col = x.shape
    msa_one_hot = (
        F.one_hot(
            x.to(torch.int64), 
            num_classes=alphabet_size,
        )
        .to(dtype)
    )  # nrow, ncol, alphabet_size
    identity_counts = torch.tensordot(
        msa_one_hot,
        msa_one_hot, 
        dims=[[1,2], [1,2]],
    )  # nrow, nrow
    # # Should be ~alphabet_size-fold fewer calculations and lower memory
    # # but actually appears to be slower?
    # identity_counts = (torch.eq(x, x)
    #                    .to(dtype)
    #                    .sum(dim=-1))  # nrow, nrow
    identity_cutoff = torch.tensor(n_col * .8, device=device)
    x_cut = (identity_counts > identity_cutoff).to(dtype)

    weights = 1. / torch.sum(x_cut, dim=-1)  # nrow
    msa_concat_one_hot = msa_one_hot.view(n_row, n_col * alphabet_size)

    shrinkage_coeff = torch.eye(msa_concat_one_hot.shape[-1], 
                                device=device) * (4.5 / torch.sqrt(torch.sum(weights)))
    covariance_matrix = _torch_cov(msa_concat_one_hot, weights)  # nrow, nrow
    shrunk_cov_matrix = covariance_matrix + shrinkage_coeff
    shrunk_cov_matrix_inv = torch.linalg.inv(shrunk_cov_matrix).view(n_col, alphabet_size, n_col, alphabet_size)

    shrunk_cov_matrix_inv_no_gaps = shrunk_cov_matrix_inv[:,:-1,:,:-1]
    I_ncol = torch.eye(n_col, device=device)
    interchain_scores = torch.sqrt(torch.sum(torch.square(shrunk_cov_matrix_inv_no_gaps), 
                                             dim=(1, 3))) * (1. - I_ncol)

    if apc:
        apc_factor = (
            torch.sum(
                interchain_scores, 
                dim=0, keepdim=True
            )
            * torch.sum(
                interchain_scores, 
                dim=1, keepdim=True,
            ) 
            / torch.sum(interchain_scores))
        interchain_scores = (interchain_scores - apc_factor) * (1. - I_ncol)

    return interchain_scores


def calculate_dca(
    msa: MSA, 
    apc: bool = False,
    gpu: bool = True,
) -> np.ndarray:

    """
    
    """
    msa_token_ids = torch.tensor(
        msa.sequence_token_ids,
        dtype=torch.int64,
        device=DEVICE,
    )
    
    try:
        wip = _get_wip(msa_token_ids, apc=apc, gpu=torch.cuda.is_available())
    except torch.cuda.OutOfMemoryError as e:
        print_err("GPU memory exhausted; falling back to CPU.")
        wip = _get_wip(msa_token_ids.to('cpu'), apc=apc, gpu=False)

    return wip.detach().cpu().numpy()