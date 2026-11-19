# AF2 optimisation benchmark

Machine: 2 CPU cores, no GPU, random (untrained) weights. Run 2026-11-18.
Timings are indicative only; they are CPU numbers, not GPU numbers.

## Feature building and Cβ extraction (best of N, seconds)

| step | baseline | optimised | speed-up |
|---|---|---|---|
| features_500x300 | 0.018 | 0.005 | 3.3x |
| features_3000x400 | 0.146 | 0.046 | 3.2x |
| contact_matrix_600res | 0.073 | 0.001 | 59.4x |

## Forward pass (reduced model: 2 evoformer blocks, 2 recycles; seconds)

`stock` = templates on, fixed padding (64 clusters + 4 template rows, 128 extra rows); `optimised` = template-free, padded to the real depth. `first` includes XLA compilation.

| MSA rows | variant | first call | compiled call | max abs Δ pLDDT vs stock |
|---|---|---|---|---|
| 12 | stock | 93.2 | 1.7 | - |
| 12 | optimised | 53.9 | 1.1 | 0 |
| 40 | stock | 91.2 | 1.7 | - |
| 40 | optimised | 54.7 | 2.3 | 0 |

(`stock` and `optimised` also differ because template inputs are removed, so the pLDDT difference above mixes both effects.)

### Adaptive padding alone (same template-free model, padding on/off; 64 clusters, 128 extra rows)

| MSA rows | padding | first call | compiled call | max abs diff pLDDT / PAE / coords vs fixed |
|---|---|---|---|---|
| 12 | fixed | 55.5 | 1.4 | - |
| 12 | adaptive | 54.7 | 1.0 | 0 / 0 / 0 |
| 40 | fixed | 53.3 | 1.3 | - |
| 40 | adaptive | 53.4 | 1.3 | 0 / 0 / 0 |

## DCA peak memory (6000 rows x 400 columns, CPU, float32)

Peak RSS is the process maximum, in MiB (ru_maxrss, converted for Linux/macOS units). The *increment* is peak minus the RSS before the call and isolates DCA; the *absolute* peak is what the machine must actually have (it includes the interpreter, torch and the input).

| implementation | RSS before | absolute peak | peak increment | seconds |
|---|---|---|---|---|
| reference (straight port) | 554 | 1640 | 1086 | 14.6 |
| supplied (pruned/tiled/Cholesky) | 554 | 1286 | 732 | 7.2 |
| supplied + row-slab streaming, in-place factorisation | 554 | ~1236 | 682 | 8.1 |

On a deep, narrow alignment (24000 rows x 100 columns) streaming cuts the increment 442 -> 116 MiB, because the one-hot matrix is never held whole. On a shallow, wide one like the table above it saves only ~7% and costs ~12% time, so the gain is for deep alignments. `torch.linalg.cholesky_ex(out=)` still allocates internally, so the in-place factorisation itself saves little.

## What was checked and not worth doing when no structure is requested

* **Early-iteration side chains** (computed in every one of the 8 structure-module iterations, read only from the last): skipping them gave bit-identical coordinates and **no measurable speed-up**. Under `jax.jit` XLA already drops computations whose results are never returned, so the change was reverted.
* **Zero-weighting the `masked_msa` and `experimentally_resolved` heads** (training-only; yunta never reads them): pLDDT, pTM and coordinates bit-identical (max abs diff 0); stock parameters still load. Kept, as it removes two output arrays and their head computations. Gain not separately timed (the heads are small next to the evoformer).
* **Cannot be skipped:** the structure module (Cβ and pLDDT come from it), the distogram head, and the PAE/pTM heads yunta reports. `return_representations` was already off.

## Recycling driven from Python (`--recycle-tolerance`)

Idea taken from the separate `yunta-af2-accel` package. AF2 recycles inside one XLA graph: a `while_loop` for the
first `num_recycle` passes plus one more call for the last, so the Evoformer is traced and compiled twice. Driving
the loop from Python over one compiled pass compiles it once, and lets recycling stop once the C-alpha distance
matrix stops changing. Reduced model (80 residues, 3 recycles, 4 Evoformer blocks), random weights, 2 vCPU,
parameters initialised beforehand so only the forward program is compiled:

| recycling | cold (compile + run) | warm run | passes |
|---|---|---|---|
| stock in-graph loop | 50.2 s | 5.7 s | 4 |
| Python loop, every recycle (`--recycle-tolerance 0`) | 23.8 s | 5.8 s | 4 |
| Python loop, stop after the first comparison | 21.0 s | 3.2 s | 2 |

With tolerance 0 the outputs match the stock loop (pLDDT/PAE within 1e-4, coordinates within 1e-3 A, tested).
The compile-time saving is exact; stopping early is approximate and off by default. These are random weights, so
the benchmark cannot say how many recycles real predictions skip at a given tolerance: measure that on real
weights before adopting a threshold.

## How to read this

* **Features and Cβ** are real, hardware-independent algorithmic speed-ups (vectorised encoding, no PDB text round-trip).
* **Forward pass:** the difference is almost entirely in the *first* call (XLA compilation: ~93 s -> ~54 s here), because
  the template branch is removed from the compiled program. Compiled-call times (1-2 s) are within noise on a 2-core
  machine and show no reliable gain for this tiny model. Do not extrapolate to GPU or to real-size models without measuring:
  the expected benefit of adaptive padding grows with the (512/5120 -> real depth) reduction, which this reduced model
  (64 clusters, 128 extra rows) barely exercises, and it was **not** measured at stock size.
* **Equivalence:** outputs with padding on/off were bit-identical (max abs diff 0) in these runs, as were stock vs template-free
  with no template data. Random weights only; no real AF2 weights were available.
* **Not measured:** GPU memory/throughput, real weights, unified memory, `subbatch_size`, the persistent compile cache across
  processes, and any accuracy/ranking effect (needs labelled data).
* The AF2 recycling fix (patch 12) changes predictions, so outputs are not compared with the pre-fix commit.
* **DCA** (the supplied pruned/tiled/Cholesky implementation, patch 1, vs the straight port kept as `_calculate_dca_reference`): absolute peak 1640 -> 1286 MiB (-22%), increment over the baseline process 1086 -> 732 MiB (-33%), 14.6 -> 7.2 s on 6000 x 400 synthetic data. On real yeast/Mtb alignments (`benchmarks/bench_dca.py`, 2 threads, with APC): DYR_YEAST (3071 x 211) 2.52 -> 1.33 s and peak RSS 0.98 -> 0.81 GB; P9WGF1_Mtb_Mmr (2306 x 107) 0.44 -> 0.30 s with no meaningful memory change (the interpreter and torch dominate at that size); scores differ by ~1e-5 absolute in float32. Scores agree with the reference to ~1e-8 (float64) in `test/test_dca_torch.py`.
