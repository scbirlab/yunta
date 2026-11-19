"""Benchmark: chunked vs unchunked DCA on real MSAs (the chunked result is an approximation).

For a set of MSA pairs, compute the inter-chain DCA matrix with the whole alignment in one call
(``chunksize`` >= L) and with chunking, then report

* Spearman rank correlation of all inter-chain scores,
* top-K overlap (fraction of the K highest-scoring residue pairs shared; K = 50 and 200),
* scale: ratio of the standard deviations of the two matrices (calibration),

so chunked scores are never silently mixed with unchunked ones.

    python benchmarks/chunked_vs_full_dca.py test/inputs/DYR_YEAST.a3m test/inputs/CAPZA_YEAST.a3m ...
"""
import sys
import numpy as np
from scipy.stats import spearmanr

from yunta.interactions.runner import DCARunner
from yunta.structs.msa import MSA


def topk_overlap(a, b, k):
    ia = set(np.argsort(a.ravel())[-k:].tolist()); ib = set(np.argsort(b.ravel())[-k:].tolist())
    return len(ia & ib) / k


def compare(path1, path2, chunk_fraction, apc=True):
    m1, m2 = MSA.from_file(path1), MSA.from_file(path2)
    full = DCARunner().run(m1, m2, chunksize=10_000, apc=apc, cpu=True)[1]
    la, lb = full.shape
    chunksize = max(16, int((la + lb) * chunk_fraction))
    chunked = DCARunner().run(m1, m2, chunksize=chunksize, apc=apc, cpu=True)[1]
    return {
        "pair": f"{path1.split('/')[-1]} x {path2.split('/')[-1]}", "len": f"{la}+{lb}", "chunksize": chunksize,
        "spearman": spearmanr(full.ravel(), chunked.ravel())[0],
        "top50": topk_overlap(full, chunked, 50), "top200": topk_overlap(full, chunked, 200),
        "std_ratio": float(chunked.std() / full.std()),
    }


if __name__ == "__main__":
    files = sys.argv[1:]
    print("pair\tlen\tchunksize\tspearman\ttop50\ttop200\tstd_ratio")
    for i in range(0, len(files) - 1, 2):
        for frac in (.5, .25):
            r = compare(files[i], files[i + 1], frac)
            print("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in r))
