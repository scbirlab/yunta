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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

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
GAP_IDX: int = _A3M_ALPHABET.index("-")
DEVICE = "cpu"

# Tile sizes for the blocked CPU/GPU kernels below. They only bound peak memory
# and GEMM granularity; results do not depend on them.
_WEIGHT_BLOCK: int = 2048
_GRAM_BLOCK: int = 512
_SCORE_BLOCK: int = 1024
_ROW_BLOCK: int = 1024  # sequences per slab when the (M, N) one-hot matrix is streamed


def _as_dtype(dtype: Union[str, "torch.dtype"]) -> "torch.dtype":
    """Accept ``"float32"``/``"float64"`` or a ``torch.dtype``."""
    if isinstance(dtype, torch.dtype):
        if dtype in (torch.float32, torch.float64):
            return dtype
        raise ValueError(f"Unsupported dtype {dtype}; use torch.float32 or torch.float64.")
    if dtype == "float64":
        return torch.float64
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype!r}; use 'float32' or 'float64'.")


_resolve_dtype = _as_dtype  # name used elsewhere in the package


###########################################################################
# Optimised implementation
#
# Mathematically identical to ``_calculate_dca_reference`` (below), which is the
# straight port of Humphreys et al. 2021. Differences are purely computational:
#
#  1. States never observed at a site have an exactly-zero covariance row/column, so
#     (C + lambda*I) is block diagonal for them and they cannot affect any
#     inter-site coupling. They are dropped before the O(N^3) steps (exact, and a
#     large win for shallow MSAs, where most of the 21*L states are unobserved).
#  2. Pairwise identities (sequence weights) are computed in symmetric tiles, so
#     only the upper triangle is multiplied, and the (M, M) matrix is never
#     materialised (previously 4*M^2 bytes: ~10 GB for M=50k).
#  3. The weighted covariance is built uncentred with one symmetric tiled GEMM plus
#     a rank-1 correction (same weighted mean / normalisation as the reference),
#     in place, without the centred (M, N) temporaries.
#  4. C + lambda*I is symmetric positive definite, so a Cholesky factorisation and
#     ``cholesky_inverse`` (~N^3 flops) replace the LU-based ``inv`` (~2 N^3).
#  5. Site-pair Frobenius norms are accumulated tile by tile instead of two full
#     ``index_select`` copies of the (N, N) inverse.
###########################################################################

def _observed_states(x: Tensor, n_states: int) -> Tensor:
    """Boolean mask over the flattened (L * n_states) one-hot features: True if seen."""
    n_col = x.shape[1]
    flat_index = (torch.arange(n_col, device=x.device) * n_states + x).reshape(-1)
    observed = torch.zeros(n_col * n_states, dtype=torch.bool, device=x.device)
    observed[flat_index] = True
    return observed


def _compact_one_hot(
    x: Tensor,
    feature_idx: Tensor,
    n_states: int,
    dtype: "torch.dtype"
) -> Tensor:
    """One-hot encode ``x`` (M, L) keeping only the flattened features in ``feature_idx``."""
    n_row, n_col = x.shape
    column_map = torch.full((n_col * n_states,), -1, dtype=torch.long, device=x.device)
    column_map[feature_idx] = torch.arange(feature_idx.numel(), device=x.device)
    columns = column_map[torch.arange(n_col, device=x.device) * n_states + x]  # (M, L)
    one_hot = torch.zeros(n_row, feature_idx.numel(), dtype=dtype, device=x.device)
    one_hot.scatter_(1, columns, 1.)
    return one_hot


def _sequence_weights(
    one_hot: Union[Tensor, Callable[[int, int], Tensor]],
    n_col: int,
    min_identical_fraction: float = .8,
    block_size: Optional[int] = None,
    n_row: Optional[int] = None,
) -> Tensor:
    """Per-sequence weights 1 / (number of sequences with identity > cutoff).

    Identity between two sequences is the dot product of their one-hot encodings
    (number of identical positions, gaps included), as in the reference.

    ``one_hot`` is either the dense ``(M, N)`` matrix or a function ``rows(i0, i1)`` returning rows
    ``i0:i1`` (with ``n_row`` giving M), so the whole matrix never needs to exist.
    """
    block_size = _WEIGHT_BLOCK if block_size is None else block_size
    rows = one_hot if callable(one_hot) else (lambda a, b: one_hot[a:b])
    n_row = one_hot.shape[0] if n_row is None else n_row
    device = rows(0, 1).device
    identity_cutoff = n_col * min_identical_fraction
    n_neighbours = torch.zeros(n_row, dtype=torch.long, device=device)
    for i0 in range(0, n_row, block_size):
        rows_i = rows(i0, i0 + block_size)
        for j0 in range(i0, n_row, block_size):
            rows_j = rows_i if j0 == i0 else rows(j0, j0 + block_size)
            above = (rows_i @ rows_j.T) > identity_cutoff
            n_neighbours[i0:i0 + rows_i.shape[0]] += above.sum(dim=1)
            if j0 > i0:  # symmetric counterpart of the tile
                n_neighbours[j0:j0 + rows_j.shape[0]] += above.sum(dim=0)
    return 1. / n_neighbours.to(rows(0, 1).dtype)


def _accumulate_gram(gram: Tensor, x: Tensor, block_size: Optional[int] = None) -> None:
    """``gram[upper] += x.T @ x`` for ``x`` of shape (m, N), upper-triangular tiles only.

    Reads column slices of ``x`` directly (BLAS takes a leading dimension), so no transposed copy.
    """
    block_size = _GRAM_BLOCK if block_size is None else block_size
    n = x.shape[1]
    for b0 in range(0, n, block_size):
        b1 = min(b0 + block_size, n)
        gram[b0:b1, b0:] += x[:, b0:b1].T @ x[:, b0:]


def _mirror_upper(gram: Tensor, block_size: Optional[int] = None) -> Tensor:
    """Copy the upper triangle of ``gram`` onto the lower one, tile by tile."""
    block_size = _GRAM_BLOCK if block_size is None else block_size
    n = gram.shape[0]
    for b0 in range(0, n, block_size):
        b1 = min(b0 + block_size, n)
        gram[b1:, b0:b1] = gram[b0:b1, b1:].T
    return gram


def _symmetric_gram(x: Tensor, block_size: Optional[int] = None) -> Tensor:
    """``x.T @ x`` for ``x`` of shape (M, N), multiplying only upper-triangular tiles."""
    gram = torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    _accumulate_gram(gram, x, block_size)
    return _mirror_upper(gram, block_size)


def _shrunk_covariance(
    x: Tensor,
    feature_idx: Tensor,
    n_states: int,
    dtype: "torch.dtype",
    weights: Optional[Tensor],
    min_identical_fraction: float,
    shrinkage_factor: float,
) -> Tensor:
    """Weighted covariance plus ridge of the (pruned) one-hot encoding of ``x``, ``(N', N')``.

    The one-hot matrix is streamed in row slabs, so memory is independent of the alignment depth M apart
    from the tokens themselves. Called a second time (from the tokens) in the rare event that the in-place
    Cholesky factorisation destroyed the matrix and the LU fallback is needed.
    """
    n_row, n_col = x.shape
    n_feat = feature_idx.numel()
    rows = lambda a, b: _compact_one_hot(x[a:b], feature_idx, n_states, dtype)  # (b - a, N'), built on demand
    if weights is None:
        weights = _sequence_weights(rows, n_col, min_identical_fraction=min_identical_fraction, n_row=n_row)  # (M,)
    else:  # sequence weights computed once on the full alignment (see `sequence_weights`)
        weights = torch.as_tensor(weights, dtype=dtype, device=x.device)
    effective_sequence_number = weights.sum()  # Meff
    # Same (slightly non-standard) weighted mean/normalisation as the reference
    num_points = effective_sequence_number - torch.sqrt(weights.mean())
    if not float(num_points) > 0.:  # e.g. a single sequence: the estimator is undefined
        raise ValueError("DCA needs at least two sequences (weighted denominator is not positive).")
    mean = torch.zeros(n_feat, dtype=dtype, device=x.device)
    cov = torch.zeros(n_feat, n_feat, dtype=dtype, device=x.device)
    for r0 in range(0, n_row, _ROW_BLOCK):  # only one (block, N') slab of the one-hot matrix exists at a time
        block = rows(r0, r0 + _ROW_BLOCK)
        w = weights[r0:r0 + block.shape[0]]
        mean += w @ block
        block.mul_(torch.sqrt(w).unsqueeze(-1))  # in place: sqrt(w) * x
        _accumulate_gram(cov, block)  # sum_m w_m x_m x_m^T
        del block
    mean /= num_points
    _mirror_upper(cov)
    # sum_m w_m (x_m - mean)(x_m - mean)^T = G - (2 * num_points - Meff) * mean mean^T
    cov.addr_(mean, mean, alpha=-float(2. * num_points - effective_sequence_number))
    cov.div_(num_points)
    cov.diagonal().add_(float(shrinkage_factor / torch.sqrt(effective_sequence_number)))
    return cov


def _calculate_dca(
    x: Tensor,
    apc: bool = False,
    gpu: bool = True,
    min_identical_fraction: float = .8,
    shrinkage_factor: float = 4.5,
    dtype: Union[str, "torch.dtype"] = "float32",
    prune: bool = True,
    weights: Optional[Tensor] = None,
) -> FloatTensor:
    dtype = _as_dtype(dtype)
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
    x = x.to(device=device, dtype=torch.long)
    n_row, n_col = x.shape
    n_states = _A3M_ALPHABET_SIZE

    if prune:
        feature_idx = _observed_states(x, n_states).nonzero().squeeze(-1)
    else:
        feature_idx = torch.arange(n_col * n_states, device=device)
    build = lambda: _shrunk_covariance(
        x, feature_idx, n_states, dtype, weights, min_identical_fraction, shrinkage_factor,
    )

    # In exact arithmetic cov = Xc^T Xc / num_points + ridge*I with Xc = sqrt(w) * (x - mean), a Gram
    # matrix for ANY mean, so it is positive definite when num_points > 0 and the ridge > 0.
    # (Rounding in the uncentred form can break this when the ridge is tiny, hence the LU fallback.)
    # Cholesky is ~2x cheaper than LU. Both the factorisation and the inverse run in place, so
    # only ONE (N', N') matrix is live at a time (two with out-of-place calls); if the factorisation
    # fails the matrix is rebuilt from the tokens for the LU fallback.
    precision = build()
    chol, info = torch.linalg.cholesky_ex(precision, out=(precision, torch.zeros((), dtype=torch.int32, device=device)))
    if int(info) == 0:
        precision = torch.cholesky_inverse(chol, out=chol)  # in place
    else:  # numerically not PD (e.g. shrinkage_factor=0): the factorisation overwrote the matrix; rebuild, use LU
        del chol, precision
        precision = torch.linalg.inv(build())
    chol = None  # (aliases `precision` on the Cholesky path)

    # Frobenius norm over non-gap states of each (site i, site j) block, accumulated in tiles
    site = feature_idx // n_states
    non_gap = (feature_idx % n_states != GAP_IDX).nonzero().squeeze(-1)
    non_gap_site = site[non_gap]
    by_site = torch.zeros(n_col, non_gap.numel(), dtype=dtype, device=device)
    for r0 in range(0, non_gap.numel(), _SCORE_BLOCK):
        rows = non_gap[r0:r0 + _SCORE_BLOCK]
        by_site.index_add_(
            0,
            non_gap_site[r0:r0 + _SCORE_BLOCK],
            precision[rows][:, non_gap].square_(),
        )
    del precision
    I_ncol = torch.eye(n_col, dtype=dtype, device=device)
    interchain_scores = (
        torch.zeros(n_col, n_col, dtype=dtype, device=device)
        .index_add_(1, non_gap_site, by_site)
        .sqrt_()
    ) * (1. - I_ncol)

    if apc:
        apc_factor = (
            interchain_scores.sum(dim=0, keepdim=True)
            * interchain_scores.sum(dim=1, keepdim=True)
            / interchain_scores.sum()
        )
        interchain_scores -= apc_factor

    return interchain_scores * (1. - I_ncol)


###########################################################################
# Reference implementation (straight port of the Humphreys et al. 2021 code).
# Kept for testing the optimised implementation and for reproducing old results.
###########################################################################

def _torch_cov(
    x: Tensor, 
    w: Optional[Tensor] = None
) -> FloatTensor:
    if w is None:
        return torch.cov(x.T)
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
    min_identical_fraction: float = .8
):
    n_row, n_col, alphabet_size = x.shape  # (M, L, 21)
    dot_product = torch.tensordot(
        x, x, 
        dims=([1,2], [1,2]),
    )  # (M, M)
    identity_cutoff = n_col * min_identical_fraction
    pairs_above_id_cutoff = (dot_product > identity_cutoff)#.to(dtype)  # nrow, nrow
    bias_correction = 1. / torch.sum(pairs_above_id_cutoff, dim=-1).to(x.dtype)  # ma (nrow); keep float64 if requested
    effective_sequence_number = torch.sum(bias_correction)  # Meff 
    return bias_correction, effective_sequence_number


def _cov_shrinkage(
    x: Tensor,
    bias_correction: Optional[Tensor] = None,
    shrinkage_factor: Optional[float] = None,
    effective_sequence_number: Optional[int] = None,
    device = None
) -> FloatTensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covariance_matrix = _torch_cov(
        x,  # (M, L * 21)
        w=bias_correction,
    )  # nrow, nrow
    if shrinkage_factor is not None:
        shrinkage_coeff = torch.eye(
            x.shape[-1], 
            dtype=x.dtype,
            device=device,
        ) * (
            shrinkage_factor / torch.sqrt(effective_sequence_number)
        )
        covariance_matrix += shrinkage_coeff
    return covariance_matrix


def _calculate_dca_reference(
    x: Tensor, 
    apc: bool = False, 
    gpu: bool = True, 
    min_identical_fraction: float = .8,
    shrinkage_factor: float = 4.5,
    dtype: Union[str, "torch.dtype"] = "float32"
) -> FloatTensor:
    import torch.nn.functional as F

    dtype = _as_dtype(dtype)
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
    x = x.to(device)
    n_row, n_col = x.shape
    msa_one_hot = F.one_hot(
        x.to(torch.long), 
        num_classes=_A3M_ALPHABET_SIZE,
    ).to(dtype)   # (M, L, 21)
    (
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
    I_ncol = torch.eye(n_col, dtype=dtype, device=device)
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


def apc_correction(scores: "ndarray") -> "ndarray":
    """Average product correction of a (full, square) coupling matrix.

    When a long alignment is split into blocks this must be applied once to the
    *assembled* matrix: the row/column background it removes involves every position.

    Examples
    ========
    >>> import numpy as np
    >>> m = np.outer([1., 2., 3.], [1., 2., 3.]); np.fill_diagonal(m, 0.)
    >>> corrected = apc_correction(m)
    >>> corrected.shape, bool(np.all(np.diag(corrected) == 0))
    ((3, 3), True)

    """
    import numpy as np
    scores = np.asarray(scores, dtype=np.float64)
    total = scores.sum()
    if total == 0:
        return scores.astype(np.float32)
    corrected = scores - scores.sum(axis=0, keepdims=True) * scores.sum(axis=1, keepdims=True) / total
    np.fill_diagonal(corrected, 0.)
    return corrected.astype(np.float32)


def sequence_weights(
    msa: ArrayLike,
    min_identical_fraction: float = .8,
    gpu: bool = True,
    dtype: str = "float32",
) -> "ndarray":
    """Per-sequence redundancy weights ``1 / (number of sequences >= 80% identical)``.

    Compute these on the *full* paired alignment and pass them to ``calculate_dca`` for
    every block, so each block uses the same effective sample size (identity over a block's
    few columns would otherwise give every block different weights).
    """
    dtype = _as_dtype(dtype)
    with torch.set_grad_enabled(False):
        device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        x = torch.as_tensor(msa, dtype=torch.long, device=device)
        feature_idx = _observed_states(x, _A3M_ALPHABET_SIZE).nonzero().squeeze(-1)
        one_hot = _compact_one_hot(x, feature_idx, _A3M_ALPHABET_SIZE, dtype)  # identities need only observed states
        weights = _sequence_weights(one_hot, x.shape[1], min_identical_fraction=min_identical_fraction)
    return weights.detach().cpu().numpy()


def calculate_dca(
    msa: ArrayLike, 
    apc: bool = False,
    gpu: bool = True,
    min_identical_fraction: float = .8,
    shrinkage_factor: float = 4.5,
    dtype: str = "float32",
    prune: bool = True,
    weights: Optional[ArrayLike] = None,
) -> ndarray:

    """Direct-coupling-like scores (Humphreys et al. 2021) for an MSA of token ids.

    Parameters
    ==========
    msa : array of shape (M, L) of integer tokens (see ``_A3M_ALPHABET``).
    apc : apply average product correction.
    gpu : use CUDA if available. Falls back to CPU on CUDA out-of-memory.
    min_identical_fraction : sequences with identity above this share weight.
    shrinkage_factor : ridge regulariser, ``shrinkage_factor / sqrt(Meff)``.
    dtype : "float32" or "float64".
    prune : drop states never observed at a site before inverting (exact; faster).
    weights : optional precomputed sequence weights (see `sequence_weights`).

    Returns
    =======
    (L, L) array of scores with zero diagonal.

    """
    use_gpu = gpu and torch.cuda.is_available()
    with torch.set_grad_enabled(False):
        msa_token_ids = torch.as_tensor(
            msa,
            dtype=torch.int64,
        )
        kwargs = {
            "apc": apc,
            "min_identical_fraction": min_identical_fraction,
            "shrinkage_factor": shrinkage_factor,
            "dtype": _as_dtype(dtype),
            "prune": prune,
            "weights": weights,
        }
        try:
            # _calculate_dca moves the tokens to the device itself, so a CUDA OOM
            # during the transfer is also caught here
            wip = _calculate_dca(
                msa_token_ids, 
                gpu=use_gpu,
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            print_err("[WARN] GPU memory exhausted; falling back to CPU.")
            torch.cuda.empty_cache()
            wip = _calculate_dca(
                msa_token_ids.to('cpu'), 
                gpu=False,
                **kwargs,
            )

    return wip.detach().cpu().numpy()
