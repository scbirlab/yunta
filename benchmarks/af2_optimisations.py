"""Benchmark the AF2 optimisations (patch 9) and the DCA memory work.

Usage (needs jax+haiku+tensorflow-cpu for the forward-pass part; run alone on a quiet machine)::

    git worktree add /tmp/yunta_base 7118285        # last commit before the AF2 optimisations
    python benchmarks/af2_optimisations.py --baseline /tmp/yunta_base --out benchmarks/af2_optimisations.md

* features / Cβ: wall time, baseline checkout vs this checkout (each in its own process).
* forward: reduced AF2 config, random weights, CPU. "stock" = templates on + fixed padding (what the
  baseline did); "optimised" = template-free + padding to real depth. Compares times AND outputs.
  Absolute numbers are not GPU numbers; only the ratios are meaningful, and only roughly.
* DCA: peak-RSS increment, the straight-port reference vs the optimised implementation.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PART_FEATURES_CB = r"""
import json, sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
rng = np.random.default_rng(0)
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY-"))
def best(fn, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)
res = {}
from yunta.interactions.af2.run import _get_af2_features
for n, L in ((500, 300), (3000, 400)):
    q = "".join(rng.choice(AA[:-1], L))
    msa = [q] + ["".join(np.where(rng.random(L) < .25, "-", rng.choice(AA[:-1], L))) for _ in range(n - 1)]
    res[f"features_{n}x{L}"] = best(lambda: _get_af2_features(msa, L // 2, "a-b"), reps=3)
from yunta.interactions.af2.src_speedppi.alphafold import protein
from yunta.interactions.af2 import scoring
N = 600
prot = protein.Protein(aatype=rng.integers(0, 20, N), atom_positions=rng.normal(size=(N, 37, 3)).astype(np.float32),
                       atom_mask=(rng.random((N, 37)) < .9).astype(np.float32), residue_index=np.arange(1, N + 1),
                       b_factors=np.zeros((N, 37), np.float32))
res["contact_matrix_600res"] = best(lambda: scoring.get_contact_matrix(prot))
print(json.dumps(res))
"""

_PART_FORWARD = r"""
import json, sys, time, numpy as np
sys.path.insert(0, sys.argv[1]); n_rows = int(sys.argv[2])
from yunta.interactions.af2 import run
from yunta.interactions.af2.modelling import inference_config
from yunta.interactions.af2.src_speedppi.alphafold.model import model, config
rng = np.random.default_rng(0)
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
L = 28; q = "".join(rng.choice(AA, L))
msa = [q] + ["".join(np.where(rng.random(L) < .2, "-", rng.choice(AA, L))) for _ in range(n_rows - 1)]

def reduce(c):
    c.data.eval.max_msa_clusters = 64 + 4; c.data.common.max_extra_msa = 128
    c.model.embeddings_and_evoformer.evoformer_num_block = 2
    c.model.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    c.model.heads.structure_module.num_layer = 2
    return c

def stock():
    c = config.model_config("model_1_ptm"); c.model.num_recycle = 2; c.data.common.num_recycle = 2
    c.model.recycle_early_stop_tolerance = -1.
    return reduce(c)

def optimised():
    return reduce(inference_config("model_1_ptm", max_recycles=2))

out, params = {}, None
for name, mk, adaptive in (("stock", stock, False), ("optimised", optimised, True)):
    c = mk(); m = model.RunModel(c, params); m.adaptive_msa = adaptive
    m.msa_limits = (c.data.eval.max_msa_clusters, c.data.common.max_extra_msa)
    params = params or m.params
    t = time.perf_counter(); _, pf, r = run.model_protein_interaction(msa, 14, "a-b", model=m, seed=5)
    first = time.perf_counter() - t                     # includes XLA compile
    t = time.perf_counter(); run.model_protein_interaction(msa, 14, "a-b", model=m, seed=6)
    second = time.perf_counter() - t                    # compiled
    out[name] = dict(first=first, second=second, extra_rows=int(pf["extra_msa"].shape[0]),
                     plddt=np.asarray(r["plddt"]).tolist(), pae=np.asarray(r["predicted_aligned_error"]).tolist())
print(json.dumps(out))
"""

_PART_FORWARD_ADAPT_ONLY = r"""
import json, sys, time, numpy as np
sys.path.insert(0, sys.argv[1]); n_rows = int(sys.argv[2])
from yunta.interactions.af2 import run
from yunta.interactions.af2.modelling import inference_config
from yunta.interactions.af2.src_speedppi.alphafold.model import model
rng = np.random.default_rng(0)
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
L = 28; q = "".join(rng.choice(AA, L))
msa = [q] + ["".join(np.where(rng.random(L) < .2, "-", rng.choice(AA, L))) for _ in range(n_rows - 1)]
out, params = {}, None
for adaptive in (False, True):
    c = inference_config("model_1_ptm", max_recycles=2)
    c.data.eval.max_msa_clusters = 64; c.data.common.max_extra_msa = 128
    c.model.embeddings_and_evoformer.evoformer_num_block = 2
    c.model.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    c.model.heads.structure_module.num_layer = 2
    m = model.RunModel(c, params); m.adaptive_msa = adaptive
    m.msa_limits = (c.data.eval.max_msa_clusters, c.data.common.max_extra_msa); params = params or m.params
    t = time.perf_counter(); _, pf, r = run.model_protein_interaction(msa, 14, "a-b", model=m, seed=5)
    first = time.perf_counter() - t
    t = time.perf_counter(); run.model_protein_interaction(msa, 14, "a-b", model=m, seed=6)
    out["adaptive" if adaptive else "fixed"] = dict(
        first=first, second=time.perf_counter() - t,
        plddt=np.asarray(r["plddt"]).tolist(), pae=np.asarray(r["predicted_aligned_error"]).tolist(),
        coords=np.asarray(r["structure_module"]["final_atom_positions"]).tolist())
print(json.dumps(out))
"""

_PART_AF2_MEM = r"""
import json, resource, sys, time, numpy as np
sys.path.insert(0, sys.argv[4])
variant, n_rows, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
from yunta.interactions.af2 import run
from yunta.interactions.af2.modelling import inference_config
from yunta.interactions.af2.src_speedppi.alphafold.model import model
rng = np.random.default_rng(0); AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
q = "".join(rng.choice(AA, L)); msa = [q] + ["".join(np.where(rng.random(L) < .2, "-", rng.choice(AA, L))) for _ in range(n_rows - 1)]
c = inference_config("model_1_ptm", max_recycles=1)       # stock 512 clusters / 5120 extra rows (not reduced)
c.model.embeddings_and_evoformer.evoformer_num_block = 2; c.model.embeddings_and_evoformer.extra_msa_stack_num_block = 1
c.model.heads.structure_module.num_layer = 2
if variant.startswith("sub"):
    c.model.global_config.subbatch_size = int(variant[3:])
m = model.RunModel(c, None); m.adaptive_msa = variant != "fixed"
m.msa_limits = (c.data.eval.max_msa_clusters, c.data.common.max_extra_msa)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter(); _, pf, r = run.model_protein_interaction(msa, L // 2, "a-b", model=m, seed=1)
print(json.dumps(dict(variant=variant, rows=n_rows, L=L, peak_mib=round(rss()), seconds=round(time.perf_counter() - t),
      plddt=float(np.mean(r["plddt"])))), flush=True)
"""

_PART_DCA = r"""
import resource, sys, time, numpy as np
sys.path.insert(0, sys.argv[2])
import torch
from yunta.interactions.dca import dca_torch as mod
fn = mod._calculate_dca_reference if sys.argv[1] == "reference" else mod.calculate_dca
M, L = int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(0); base = rng.integers(0, 21, L)
x = np.where(rng.random((M, L)) < .5, base, rng.integers(0, 21, (M, L)))
# ru_maxrss is KiB on Linux but bytes on macOS
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
b = rss(); t = time.perf_counter(); fn(torch.as_tensor(x), gpu=False)
print(b, rss(), rss() - b, time.perf_counter() - t)  # baseline, absolute peak, increment (MiB), seconds
"""


def run_py(code, *args, python=sys.executable):
    p = subprocess.run([python, "-c", code, *map(str, args)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return p.stdout.strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="checkout of the commit before the AF2 optimisations")
    ap.add_argument("--python", default=sys.executable, help="interpreter with jax/haiku/tensorflow")
    ap.add_argument("--dca-python", default=sys.executable, help="interpreter with torch for the DCA part")
    ap.add_argument("--out", default="benchmarks/af2_optimisations.md")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--af2-memory", action="store_true", help="also measure AF2 peak RSS at stock padding (slow)")
    ap.add_argument("--dca-only", action="store_true", help="skip every AF2 part (no jax needed)")
    ap.add_argument("--dca-size", nargs=2, type=int, default=[6000, 400])
    a = ap.parse_args()
    lines = ["# AF2 optimisation benchmark", "",
             f"Machine: {os.cpu_count()} CPU cores, no GPU, random (untrained) weights. Run {time.strftime('%Y-%m-%d')}.",
             "Timings are indicative only; they are CPU numbers, not GPU numbers.", ""]

    feats = {}
    for name, root in (() if a.dca_only else (("baseline", a.baseline), ("optimised", str(ROOT)))):
        feats[name] = json.loads(run_py(_PART_FEATURES_CB, root, python=a.python))
    lines += ["## Feature building and Cβ extraction (best of N, seconds)", "",
              "| step | baseline | optimised | speed-up |", "|---|---|---|---|"]
    for k in feats.get("baseline", {}):
        b, o = feats["baseline"][k], feats["optimised"][k]
        lines.append(f"| {k} | {b:.3f} | {o:.3f} | {b / o:.1f}x |")

    if not (a.skip_forward or a.dca_only):
        import numpy as np
        lines += ["", "## Forward pass (reduced model: 2 evoformer blocks, 2 recycles; seconds)", "",
                  "`stock` = templates on, fixed padding (64 clusters + 4 template rows, 128 extra rows); "
                  "`optimised` = template-free, padded to the real depth. `first` includes XLA compilation.", "",
                  "| MSA rows | variant | first call | compiled call | max abs Δ pLDDT vs stock |",
                  "|---|---|---|---|---|"]
        for n in (12, 40):
            r = json.loads(run_py(_PART_FORWARD, str(ROOT), n, python=a.python))
            d = float(np.abs(np.array(r["stock"]["plddt"]) - np.array(r["optimised"]["plddt"])).max())
            lines.append(f"| {n} | stock | {r['stock']['first']:.1f} | {r['stock']['second']:.1f} | - |")
            lines.append(f"| {n} | optimised | {r['optimised']['first']:.1f} | {r['optimised']['second']:.1f} | {d:.3g} |")
        lines += ["", "(`stock` and `optimised` also differ because template inputs are removed, so the pLDDT "
                  "difference above mixes both effects.)", "",
                  "### Adaptive padding alone (same template-free model, padding on/off; 64 clusters, 128 extra rows)", "",
                  "| MSA rows | padding | first call | compiled call | max abs diff pLDDT / PAE / coords vs fixed |",
                  "|---|---|---|---|---|"]
        for n in (12, 40):
            r = json.loads(run_py(_PART_FORWARD_ADAPT_ONLY, str(ROOT), n, python=a.python))
            diffs = [float(np.abs(np.array(r["adaptive"][k]) - np.array(r["fixed"][k])).max()) for k in ("plddt", "pae", "coords")]
            lines.append(f"| {n} | fixed | {r['fixed']['first']:.1f} | {r['fixed']['second']:.1f} | - |")
            lines.append(f"| {n} | adaptive | {r['adaptive']['first']:.1f} | {r['adaptive']['second']:.1f} | "
                         f"{diffs[0]:.2g} / {diffs[1]:.2g} / {diffs[2]:.2g} |")

    if a.af2_memory and not a.dca_only:
        lines += ["", "## AF2 peak memory at stock MSA limits (512 clusters + 5120 extra rows)", "",
                  "Reduced model (2 evoformer blocks, 1 recycle), 64 MSA rows, 160 residues, random weights, CPU. Peak RSS of the whole "
                  "process (includes JAX/TensorFlow runtime) and wall time including XLA compilation; each variant in a fresh process.", "",
                  "| variant | peak RSS (MiB) | seconds | mean pLDDT |", "|---|---|---|---|"]
        for v, label in (("fixed", "stock padding"), ("adaptive", "adaptive padding"), ("sub1", "adaptive + subbatch_size=1")):
            r = json.loads(run_py(_PART_AF2_MEM, v, 64, 160, str(ROOT), python=a.python))
            lines.append(f"| {label} | {r['peak_mib']} | {r['seconds']} | {r['plddt']:.4f} |")

    M, L = a.dca_size
    lines += ["", f"## DCA peak memory ({M} rows x {L} columns, CPU, float32)", "",
              "Peak RSS is the process maximum, in MiB (ru_maxrss, converted for Linux/macOS units). The "
              "*increment* is peak minus the RSS before the call and isolates DCA; the *absolute* peak is "
              "what the machine must actually have (it includes the interpreter, torch and the input).", "",
              "| implementation | RSS before | absolute peak | peak increment | seconds |", "|---|---|---|---|---|"]
    for label, mod in (("reference (straight port)", "reference"), ("optimised", "optimised")):
        before, peak, inc, sec = run_py(_PART_DCA, mod, str(ROOT), M, L, python=a.dca_python).split()
        lines.append(f"| {label} | {float(before):.0f} | {float(peak):.0f} | {float(inc):.0f} | {float(sec):.1f} |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
