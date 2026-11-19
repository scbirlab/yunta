"""Split long paired alignments into blocks that fit in memory, and stitch the results back.

A paired MSA has ``L = len_A + len_B`` columns. Models whose cost grows quadratically
with ``L`` cannot always run on the whole complex, so we run them on *blocks*:

* **Segments** are contiguous column ranges that lie entirely inside one chain, so no
  segment ever straddles the A|B boundary.
* A **block** is one segment (giving intra-segment couplings) or an ordered pair of
  segments (giving the couplings *between* them). Because the two segments of a block
  are always distinct chains from the model's point of view, a *single* chain break at
  ``block.break_at`` describes the block exactly. (Previously the break was
  ``chain_a_length - i0``, which is only right for a few block layouts, so cross-chunk
  contacts were modelled as one continuous chain.)

``chunksize`` is the maximum number of columns in any single model call.

Examples
========
>>> [(len(b), b.break_at, b.is_whole) for b in plan_blocks(5, 4, chunksize=100)]
[(9, 5, True)]
>>> blocks = plan_blocks(5, 4, chunksize=6)
>>> sorted({len(b) for b in blocks})
[2, 3, 4, 5]
>>> max(len(b) for b in blocks) <= 6
True
>>> [(s.start, s.stop) for s in split_evenly(751, 750)]
[(0, 376), (376, 751)]

"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class Segment:
    """Columns ``[start, stop)`` of the paired MSA, within chain ``chain`` (0 = A, 1 = B)."""
    start: int
    stop: int
    chain: int = 0

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class Block:
    """One model call: a segment, a pair of segments, or the whole alignment.

    ``break_at`` is the column (within the block) where the second chain starts;
    equal to ``len(block)`` when there is no break.
    """
    segments: tuple[Segment, ...]
    break_at: int

    def __len__(self) -> int:
        return sum(len(s) for s in self.segments)

    @property
    def is_pair(self) -> bool:
        return len(self.segments) == 2

    @property
    def is_whole(self) -> bool:
        """True for the single, unchunked model call over the full alignment."""
        return len(self.segments) == 1 and self.segments[0].chain == -1

    def columns(self) -> np.ndarray:
        """Original column indices of this block, in block order."""
        return np.concatenate([np.arange(s.start, s.stop) for s in self.segments])

    def intra_chain(self) -> bool:
        return len({s.chain for s in self.segments}) == 1 and self.segments[0].chain != -1


def split_evenly(length: int, max_size: int, offset: int = 0, chain: int = 0) -> list[Segment]:
    """Split ``length`` columns into the fewest, near-equal segments of at most ``max_size``.

    The previous scheme (``np.split`` every ``chunksize``) left a tiny trailing chunk,
    e.g. 751 columns -> 750 + 1, which is useless as a model input.
    """
    if length <= 0:
        return []
    max_size = max(1, max_size)
    n_segments = ceil(length / max_size)
    base, extra = divmod(length, n_segments)  # first `extra` segments get one more column
    segments, start = [], 0
    for i in range(n_segments):
        size = base + (1 if i < extra else 0)
        segments.append(Segment(offset + start, offset + start + size, chain))
        start += size
    return segments


def plan_blocks(
    chain_a_length: int,
    chain_b_length: int,
    chunksize: int | None = None,
    blocks: Literal["all", "interchain"] = "all",
) -> list[Block]:
    """Plan the model calls needed to cover a paired alignment.

    Parameters
    ----------
    chunksize : int, optional
        Maximum columns in any one call. ``None`` (the default) never chunks: the whole
        complex is one call. If the whole alignment fits, there is a single call anyway.
    blocks : {"all", "interchain"}
        ``"all"`` covers every residue pair (needed for a full contact map or for DCA's
        global APC). ``"interchain"`` covers only A x B, which is all that interaction
        scores use and avoids the intra-chain calls (the expensive ones for AF2/OpenFold).

    """
    n_cols = chain_a_length + chain_b_length
    if chunksize is None or n_cols <= chunksize:
        return [Block((Segment(0, n_cols, chain=-1),), break_at=chain_a_length)]
    chunksize = max(2, chunksize)
    half = chunksize // 2
    if blocks == "interchain":
        # Give the longer chain the budget the shorter one doesn't use.
        if chain_a_length <= half:
            max_a, max_b = chain_a_length, chunksize - chain_a_length
        elif chain_b_length <= half:
            max_a, max_b = chunksize - chain_b_length, chain_b_length
        else:
            max_a = max_b = half
    else:
        max_a = max_b = half
    segments_a = split_evenly(chain_a_length, max_a, offset=0, chain=0)
    segments_b = split_evenly(chain_b_length, max_b, offset=chain_a_length, chain=1)
    planned = []
    if blocks == "all":
        for segment in segments_a + segments_b:
            planned.append(Block((segment,), break_at=len(segment)))
        for group in (segments_a, segments_b):
            for i, first in enumerate(group):
                for second in group[i + 1:]:
                    planned.append(Block((first, second), break_at=len(first)))
    for first in segments_a:
        for second in segments_b:
            planned.append(Block((first, second), break_at=len(first)))
    return planned


def block_input(columns: np.ndarray, block: Block) -> np.ndarray:
    """Slice an alignment array ``(n_rows, n_cols)`` down to a block's columns."""
    if block.is_whole:
        return columns
    return columns[:, block.columns()]


def assemble_matrix(
    results: Iterable[tuple[Block, np.ndarray]],
    n_cols: int,
    fill: float = 0.,
    dtype=np.float32,
) -> np.ndarray:
    """Place per-block ``(len(block), len(block))`` matrices into the full ``(L, L)`` matrix.

    Intra-segment entries come from single-segment blocks; for segment pairs only the
    cross-segment quadrants are used (the intra-segment quadrants are redundant).

    Examples
    ========
    >>> blocks = plan_blocks(3, 2, chunksize=4)
    >>> truth = np.arange(25, dtype=float).reshape(5, 5)
    >>> out = assemble_matrix(((b, truth[np.ix_(b.columns(), b.columns())]) for b in blocks), 5)
    >>> np.array_equal(out, truth)
    True

    """
    full = np.full((n_cols, n_cols), fill, dtype=dtype)
    for block, matrix in results:
        matrix = np.asarray(matrix)
        if block.is_whole:
            full[:, :] = matrix
        elif not block.is_pair:
            sl = block.segments[0].slice
            full[sl, sl] = matrix
        else:
            first, second = block.segments
            n = len(first)
            full[first.slice, second.slice] = matrix[:n, n:]
            full[second.slice, first.slice] = matrix[n:, :n]
    return full


def assemble_vector(
    results: Iterable[tuple[Block, np.ndarray]],
    n_cols: int,
) -> np.ndarray:
    """Average per-residue vectors (e.g. pLDDT) over every block that contains the residue.

    The previous code let the last block overwrite earlier ones, so a residue's value
    depended on iteration order. The mean is order independent.

    Examples
    ========
    >>> blocks = plan_blocks(2, 2, chunksize=2)
    >>> assemble_vector(((b, b.columns().astype(float)) for b in blocks), 4).tolist()
    [0.0, 1.0, 2.0, 3.0]

    """
    total = np.zeros(n_cols, dtype=np.float64)
    count = np.zeros(n_cols, dtype=np.int64)
    for block, vector in results:
        columns = np.arange(n_cols) if block.is_whole else block.columns()
        total[columns] += np.asarray(vector, dtype=np.float64)
        count[columns] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    return np.where(count > 0, out, np.nan).astype(np.float32)


def run_blocks(
    alignment: np.ndarray,
    chain_a_length: int,
    interaction_fn: Callable[..., Sequence[Any]],
    chunksize: int | None = None,
    blocks: Literal["all", "interchain"] = "all",
    to_input: Callable[[np.ndarray], Any] | None = None,
    progress: bool = True,
    **kwargs,
) -> tuple[np.ndarray, list[tuple[Block, tuple]]]:
    """Run ``interaction_fn`` over every planned block and stitch the main matrix.

    ``interaction_fn(block_alignment, chain_a_length=break_at, **kwargs)`` must return
    ``(matrix, *extras)`` with ``matrix`` of shape ``(len(block), len(block))``.

    Returns the full ``(L, L)`` matrix and ``[(block, extras), ...]``.
    Columns of entries no block covers (only with ``blocks="interchain"``) are ``0``.
    """
    from carabiner import print_err
    n_cols = alignment.shape[-1]
    plan = plan_blocks(chain_a_length, n_cols - chain_a_length, chunksize=chunksize, blocks=blocks)
    if len(plan) > 1:
        print_err(
            f"[INFO] Splitting MSA with {n_cols} columns into {len(plan)} blocks "
            f"of at most {chunksize} columns ({blocks})."
        )
    iterator = plan
    if progress and len(plan) > 1:
        from tqdm.auto import tqdm
        iterator = tqdm(plan, desc="Running blocks")
    matrices, extras = [], []
    for block in iterator:
        sub = block_input(alignment, block)
        if to_input is not None:
            sub = to_input(sub)
        matrix, *others = interaction_fn(sub, chain_a_length=block.break_at, **kwargs)
        matrices.append((block, matrix))
        extras.append((block, tuple(others)))
    return assemble_matrix(matrices, n_cols), extras
