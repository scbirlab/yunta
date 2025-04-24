"""DCA for GPU implemented in pytorch."""

from typing import Optional, Union

from io import TextIOWrapper
import sys

from carabiner import print_err
import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import FloatTensor, Tensor
import torch.nn.functional as F

from .structs.msa import _A3M_ALPHABET, _A3M_ALPHABET_SIZE

NON_GAP_IDX = [i for i, char in enumerate(_A3M_ALPHABET) if char != "-"]
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


def two_site_frequency_count(
    x: Tensor,
    min_identical_fraction: float = .8,
):
    n_row, n_col, alphabet_size = x.shape  # (M, L, 21)
    dot_product = torch.tensordot(
        x, x, 
        dims=([1,2], [1,2]),
    )  # (M, M)
    identity_cutoff = n_col * min_identical_fraction
    pairs_above_id_cutoff = (dot_product > identity_cutoff)#.to(dtype)  # nrow, nrow
    bias_correction = 1. / torch.sum(pairs_above_id_cutoff, dim=-1)  # ma (nrow)
    effective_sequence_number = torch.sum(bias_correction)  # Meff 
    return dot_product, bias_correction, effective_sequence_number


def _cov_shrinkage(
    x: Tensor,
    bias_correction: Optional[Tensor] = None,
    shrinkage_factor: Optional[float] = None,
    effective_sequence_number: Optional[int] = None,
    device = DEVICE
) -> FloatTensor:  
    covariance_matrix = _torch_cov(
        x,  # (M, L * 21)
        w=bias_correction,
    )  # nrow, nrow
    if shrinkage_factor is not None:
        shrinkage_coeff = torch.eye(
            x.shape[-1], 
            device=device,
        ) * (
            shrinkage_factor / torch.sqrt(effective_sequence_number)
        )
        covariance_matrix += shrinkage_coeff
    return covariance_matrix


def _calculate_dca(
    x: Tensor, 
    apc: bool = False, 
    gpu: bool = True, 
    min_identical_fraction: float = .8,
    shrinkage_factor: float = 4.5,
    dtype=torch.float32
) -> FloatTensor:
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
    x = x.to(device)
    n_row, n_col = x.shape
    msa_one_hot = F.one_hot(
        x.to(torch.long), 
        num_classes=_A3M_ALPHABET_SIZE,
    ).to(torch.float32)   # (M, L, 21)
    (
        dot_product, 
        bias_correction, 
        effective_sequence_number,
    ) = two_site_frequency_count(
        msa_one_hot,
        min_identical_fraction=min_identical_fraction,
    ) # (M, M), 1, 1
    msa_flat_one_hot = msa_one_hot.flatten(start_dim=-2)  # (M, L * 21)
    cov_matrix = _cov_shrinkage(
        msa_flat_one_hot,
        bias_correction=bias_correction,
        shrinkage_factor=shrinkage_factor,
        effective_sequence_number=effective_sequence_number,
        device=device,
    )
    cov_matrix = (
        torch.linalg.inv(cov_matrix)
        .view(*((n_col, _A3M_ALPHABET_SIZE) * 2))
    )  # (L, 21, L, 21)
    non_gap_idx = torch.as_tensor(
        NON_GAP_IDX,
        dtype=torch.long,
        device=device
    )
    for dim in (1, 3):
        cov_matrix = torch.index_select(
            cov_matrix,
            index=non_gap_idx,
            dim=dim,
        )  # drop the gap char dimensions   # (L, 20, L, 20)
    I_ncol = torch.eye(n_col, device=device)
    interchain_scores = (
        cov_matrix
        .square()
        .sum(dim=(1, 3))
        .sqrt()
    ) * (1. - I_ncol)

    if apc:
        apc_factor = (
            interchain_scores
            .sum(dim=0, keepdim=True)
            * 
            interchain_scores
            .sum(dim=1, keepdim=True)
            / 
            interchain_scores.sum()
        )
        interchain_scores -= apc_factor 

    return interchain_scores * (1. - I_ncol)


def calculate_dca(
    msa: ArrayLike, 
    apc: bool = False,
    gpu: bool = True,
    min_identical_fraction: float = .8,
    shrinkage_factor: float = 4.5,
    dtype=torch.float32
) -> np.ndarray:

    """
    
    """
    with torch.set_grad_enabled(False):
        msa_token_ids = torch.tensor(
            msa,
            dtype=torch.int64,
            device=DEVICE,
        )
        kwargs = {
            "apc": apc,
            "min_identical_fraction": min_identical_fraction,
            "shrinkage_factor": shrinkage_factor,
            "dtype": dtype,
        }
        try:
            wip = _calculate_dca(
                msa_token_ids, 
                gpu=torch.cuda.is_available(),
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            print_err("GPU memory exhausted; falling back to CPU.")
            wip = _calculate_dca(
                msa_token_ids.to('cpu'), 
                gpu=False,
                **kwargs,
            )

    return wip.detach().cpu().numpy()