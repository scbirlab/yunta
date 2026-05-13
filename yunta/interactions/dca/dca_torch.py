"""Code from Humpreys, Science, 2021 (DOI: 10.1126/science.abm4805), reimplemented for pytorch.

According to the paper's supplementary material:
"We reimplemented DCA so that it can be computed using GPUs to speed up the calculation.
The code is in the box below. In addition, we applied average product correction (APC).[...]"

```python
import sys
import numpy as np
import string
import tensorflow as tf

def parse_a3m(a3mlines):
    seqs = []
    labels = []
    for line in a3mlines:
        if line[0] == '>':
            labels.append(line.rstrip())
        else:
            seqs.append(line[:-1])
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    seq_num = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq_num[seq_num == alphabet[i]] = i
    seq_num[seq_num > 20] = 20
    return {'seqs' : seq_num, 'labels' : labels }

def tf_cov(x,w=None):
    if w is None:
        num_points = tf.cast(tf.shape(x)[0],tf.float32) - 1
        x_mean = tf.reduce_mean(x, axis=0, keep_dims=True)
        x = (x - x_mean)
    else:
        num_points = tf.reduce_sum(w) - tf.sqrt(tf.reduce_mean(w))
        x_mean = tf.reduce_sum(x * w[:,None], axis=0, keepdims=True) / num_points
        x = (x - x_mean) * tf.sqrt(w[:,None])
    return tf.matmul(tf.transpose(x),x)/num_points

fp = open(sys.argv[1] + ".alignments", "r")
pair2a3m = {}
pair = ""
pairs = []
a3m = []
for line in fp:
    if line[:2] == ">>":
        if pair and a3m:
            pair2a3m[pair] = a3m
            pairs.append(pair)
        pair = line[2:-1]
        a3m = []
    else:
        a3m.append(line)
fp.close()
if pair and a3m:
    pair2a3m[pair] = a3m
    pairs.append(pair)

config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9))

with tf.Graph().as_default():
    x = tf.placeholder(tf.uint8,shape=(None,None),name="x")
    x_shape = tf.shape(x)
    x_nr = x_shape[0]
    x_nc = x_shape[1]
    x_ns = 21
    x_msa = tf.one_hot(x,x_ns)
    x_cutoff = tf.cast(x_nc,tf.float32) * 0.8
    x_pw = tf.tensordot(x_msa, x_msa, [[1,2], [1,2]])
    x_cut = x_pw > x_cutoff
    x_weights = 1.0/tf.reduce_sum(tf.cast(x_cut, dtype=tf.float32),-1)
    x_feat = tf.reshape(x_msa,(x_nr,x_nc*x_ns))
    x_c = tf_cov(x_feat,x_weights) + tf.eye(x_nc*x_ns) *
    4.5/tf.sqrt(tf.reduce_sum(x_weights))
    x_c_inv = tf.linalg.inv(x_c)
    x_w = tf.reshape(x_c_inv,(x_nc,x_ns,x_nc,x_ns))
    x_wi = tf.sqrt(tf.reduce_sum(tf.square(x_w[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(x_nc))
    # APC
    #x_ap = tf.reduce_sum(x_wi,0,keepdims=True) * tf.reduce_sum(x_wi,1,keepdims=True) /
    tf.reduce_sum(x_wi)
    #x_wip = (x_wi - x_ap) * (1-tf.eye(x_nc))

    with tf.Session(config=config) as sess:
        results = []
        get_pairs = []
        for pair in pairs:
            print (pair)
            msa = parse_a3m(pair2a3m[pair])
            try:
                wip = sess.run(x_wi,{x:msa['seqs']})
                results.append(wip.astype(np.float16))
                get_pairs.append(pair)
            except tf.errors.ResourceExhaustedError as e:
                pass
        np.savez_compressed(sys.argv[1], *results, names=get_pairs)
        rp = open(sys.argv[1] + ".log","w")
        for pair in get_pairs:
            rp.write(pair + "\n")
        rp.close()
```
"""
from typing import TYPE_CHECKING, Any, Optional, Union

from carabiner import print_err

from numpy.typing import ArrayLike
import torch
if TYPE_CHECKING:
    from numpy import ndarray
    from torch import FloatTensor, Tensor
else:
    ndarray = Any
    FloatTensor, Tensor = Any, Any

from ...structs.msa import _A3M_ALPHABET, _A3M_ALPHABET_SIZE

NON_GAP_IDX = [i for i, char in enumerate(_A3M_ALPHABET) if char != "-"]
DEVICE = "cpu"

def _torch_cov(
    x: Tensor, 
    w: Optional[Tensor] = None
) -> FloatTensor:
    import torch
    if w is None:
        return torch.cov(x)
    else:
        num_points = torch.sum(w) - torch.sqrt(torch.mean(w))
        x_mean = torch.sum(
            x * w.unsqueeze(-1), 
            dim=0, 
            keepdim=True,
        ) / num_points
        x = (x - x_mean) * torch.sqrt(w.unsqueeze(-1))
        return torch.matmul(x.transpose(-2, -1), x) / num_points


def two_site_frequency_count(
    x: Tensor,
    min_identical_fraction: float = .8,
):
    import torch
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
    device = None
) -> FloatTensor:
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dtype: str = "float32"
) -> FloatTensor:
    import torch
    import torch.nn.functional as F

    dtype = torch.getattr(dtype, torch.float32)
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
    x = x.to(device)
    n_row, n_col = x.shape
    msa_one_hot = F.one_hot(
        x.to(torch.long), 
        num_classes=_A3M_ALPHABET_SIZE,
    ).to(dtype)   # (M, L, 21)
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
    dtype: str = "float32"
) -> ndarray:

    """
    
    """
    import torch
    with torch.set_grad_enabled(False):
        msa_token_ids = torch.tensor(
            msa,
            dtype=torch.int64,
            device=torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu"),
        )
        kwargs = {
            "apc": apc,
            "min_identical_fraction": min_identical_fraction,
            "shrinkage_factor": shrinkage_factor,
            "dtype": torch.getattr(dtype, torch.float32),
        }
        try:
            wip = _calculate_dca(
                msa_token_ids, 
                gpu=torch.cuda.is_available(),
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            print_err("[WARN] GPU memory exhausted; falling back to CPU.")
            wip = _calculate_dca(
                msa_token_ids.to('cpu'), 
                gpu=False,
                **kwargs,
            )

    return wip.detach().cpu().numpy()
