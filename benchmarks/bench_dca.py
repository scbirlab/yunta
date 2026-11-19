"""Compare the reference and optimised torch DCA on A3M files.

Usage: python benchmarks/bench_dca.py test/inputs/DYR_YEAST.a3m [more.a3m ...] [--threads N] [--apc]
Each (implementation, file) runs in a fresh process so peak RSS is meaningful.
"""
import argparse, os, subprocess, sys, tempfile

CHILD = r"""
import sys, time, resource, numpy as np, torch
from yunta.structs.msa import MSA
from yunta.interactions.dca import dca_torch as d
impl, path, threads, apc, outpath = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
torch.set_num_threads(threads)
x = torch.as_tensor(np.asarray(MSA.from_file(path).sequence_token_ids), dtype=torch.long)
fn = d._calculate_dca_reference if impl == "reference" else d._calculate_dca
with torch.no_grad():
    t = time.perf_counter(); out = fn(x, apc=apc, gpu=False); dt = time.perf_counter() - t
np.save(outpath, out.numpy())
print(f"{impl:9s} M={x.shape[0]:6d} L={x.shape[1]:4d} threads={threads} time={dt:7.2f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)/1e9:.2f}GB")
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--apc", action="store_true")
    a = ap.parse_args()
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        for f in a.files:
            print(f)
            paths = {}
            for impl in ("reference", "optimised"):
                paths[impl] = os.path.join(tmp, f"{impl}.npy")
                subprocess.run([sys.executable, "-c", CHILD, impl, f, str(a.threads), "1" if a.apc else "0", paths[impl]], check=True)
            r, o = np.load(paths["reference"]), np.load(paths["optimised"])
            print(f"  max|diff|={abs(r-o).max():.2e}  relF={np.linalg.norm(r-o)/np.linalg.norm(r):.2e}")
