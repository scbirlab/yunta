r"""Run OpenFold3 on a paired MSA and read back coordinates and confidences.

How a paired MSA is given to OpenFold3
--------------------------------------
OpenFold3 reads a query JSON with one entry per chain. A chain's alignments are picked by
*file stem*: ``colabfold_paired`` (rows aligned across chains, in the same order in both
chains' files) is the default paired alignment and ``colabfold_main`` the default unpaired
one. We therefore write, for each block of the complex,

* ``<work>/<query>/A/colabfold_paired.a3m`` and ``.../B/colabfold_paired.a3m``: the two
  halves of the paired MSA (row ``i`` of A is paired with row ``i`` of B), and
* ``colabfold_main.a3m`` per chain holding just the query row, so the *only* co-evolutionary
  information the model sees is the pairing (the same as the AF2 backend). This is what
  makes the negative-control pairings meaningful for OpenFold3 too.

MSA-server and template use are switched off.

Weights
-------
``ensure_checkpoint`` downloads the default checkpoint on first use with OpenFold3's own
``download_model_parameters`` (anonymous S3 access to ``openfold3-data``) into
``$OPENFOLD_CACHE`` or ``~/.openfold3``, and reuses it afterwards. Pass ``checkpoint=`` (a
path) to use your own. Downloads are single-flight (see :mod:`yunta.weights`): parallel
processes sharing the cache wait for one download, which appears by atomic rename.

Amortising the model load
-------------------------
:class:`OF3Model` builds one ``InferenceExperimentRunner`` and loads the checkpoint once.
OpenFold3 caches its data module for one query set, so that cache is cleared before each
query; the (expensive) Lightning module and trainer are kept.

Examples
========
>>> from yunta.interactions.of3.run import _write_a3m
>>> import io
>>> buf = io.StringIO(); _write_a3m(buf, ["ACD", "A-D"], ["q", "hit1"])
>>> buf.getvalue()
'>q\nACD\n>hit1\nA-D\n'
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import shutil
import tempfile
import uuid
from typing import Any, Callable

from carabiner import print_err
import numpy as np
from scipy.spatial.distance import cdist

from .. import confidence
from ...weights import fetch_once
from ..chunking import Block

DEFAULT_CHECKPOINT = "openbind-2025-06-30-174k"


def _write_a3m(handle, sequences, names) -> None:
    for name, seq in zip(names, sequences):
        handle.write(f">{name}\n{seq}\n")


def write_query(
    workdir: str | os.PathLike,
    query_id: str,
    rows: list[str],
    names: list[str],
    chain_a_length: int,
) -> Path:
    """Write per-chain A3M files and the OpenFold3 query JSON for one block.

    ``rows`` are the aligned paired rows (query first), as full-width strings; each is
    split at ``chain_a_length`` into the two chains' halves.
    """
    workdir = Path(workdir) / query_id
    chains = {}
    for chain_id, (lo, hi) in {"A": (0, chain_a_length), "B": (chain_a_length, None)}.items():
        halves = [r[lo:hi] for r in rows]
        directory = workdir / chain_id
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "colabfold_paired.a3m", "w") as f:
            _write_a3m(f, halves, names)
        with open(directory / "colabfold_main.a3m", "w") as f:  # query only
            _write_a3m(f, halves[:1], names[:1])
        chains[chain_id] = {
            "molecule_type": "protein",
            "chain_ids": [chain_id],
            "sequence": halves[0].replace("-", ""),
            "main_msa_file_paths": [str(directory / "colabfold_main.a3m")],
            "paired_msa_file_paths": [str(directory / "colabfold_paired.a3m")],
        }
    query = {"queries": {query_id: {"chains": list(chains.values())}}}
    path = workdir / "query.json"
    path.write_text(json.dumps(query, indent=2))
    return path


def ensure_checkpoint(name: str = DEFAULT_CHECKPOINT, checkpoint: str | os.PathLike | None = None) -> Path:
    """Path to the OpenFold3 weights, downloading them on first use."""
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"OpenFold3 checkpoint not found: {checkpoint}")
        return checkpoint
    try:
        from openfold3.entry_points.parameters import (
            DEFAULT_CACHE_PATH, OPENFOLD_MODEL_CHECKPOINT_REGISTRY,
            download_model_parameters, get_default_checkpoint_dir,
        )
    except ImportError as e:
        raise ImportError('OpenFold3 is not installed. Try: pip install "yunta[of3]"') from e
    cache = Path(os.environ.get("OPENFOLD_CACHE") or DEFAULT_CACHE_PATH)
    cache.mkdir(parents=True, exist_ok=True)
    param_dir = get_default_checkpoint_dir(cache_path=cache)
    file_name = OPENFOLD_MODEL_CHECKPOINT_REGISTRY[name].file_name
    path = param_dir / file_name

    def produce(tmp: Path) -> None:
        # OpenFold3 writes straight to its final name, so a concurrent reader could see a
        # partial file. Download into a private directory and let fetch_once rename it.
        print_err(f"[INFO] Downloading OpenFold3 weights '{name}' to {path} (first use only)...")
        staging = tmp.with_name(tmp.name + ".dir")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            download_model_parameters(staging, name, skip_confirmation=True)
            downloaded = staging / file_name
            if not downloaded.is_file():
                raise FileNotFoundError(f"OpenFold3 did not write {file_name} to {staging}")
            os.replace(downloaded, tmp)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return fetch_once(path, produce, validate=lambda p: p.stat().st_size > 0)


def _protein_atoms(cif_path: Path):
    """Protein atoms (with per-atom pLDDT in ``b_factor``) of the first model of an OpenFold3 CIF."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1, extra_fields=["b_factor"])
    return atoms[struc.filter_amino_acids(atoms)]


def _arrays_from_atoms(atoms):
    import biotite.structure as struc

    starts = struc.get_residue_starts(atoms)
    coords, plddt = [], []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else atoms.array_length()
        res = atoms[start:end]
        wanted = res[res.atom_name == "CB"]
        if len(wanted) == 0:
            wanted = res[res.atom_name == "CA"]
        coords.append(wanted.coord[0])
        plddt.append(float(res.b_factor.mean()))
    return np.asarray(coords, dtype=np.float64), np.asarray(plddt, dtype=np.float32)


def _residue_arrays(cif_path: Path):
    """Per-residue Cβ (Cα for glycine) coordinates and per-residue mean B-factor (pLDDT)."""
    return _arrays_from_atoms(_protein_atoms(cif_path))


def _pdb_text(atoms) -> str:
    """PDB text (chains as OpenFold3 named them, pLDDT in the B-factor column) for protein atoms."""
    import io
    import biotite.structure.io.pdb as pdb

    handle = io.StringIO()
    out = pdb.PDBFile()
    out.set_structure(atoms)
    out.write(handle)
    return handle.getvalue()


def read_prediction(
    out_dir: str | os.PathLike, query_id: str, seed: int, sample: int = 1, structure: bool = False,
    captured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one OpenFold3 sample into the canonical record (Cβ distances plus ``compact_prediction``).

    ``captured`` holds the arrays OpenFold3 computes but does not write to disk (``pae_probs``,
    ``contact_probs``; see :class:`OF3Model`). With them, pTM, ipTM and actifpTM are computed by
    yunta's shared code from the PAE bin probabilities, exactly as for AF2. Without them (an
    OpenFold3 that did not expose them) only the expected PAE on disk is available, so those three
    scores are NaN; OpenFold3's own pTM/ipTM are kept in ``ptm_model``/``iptm_model`` either way.
    With ``structure=True`` the model is also returned as PDB text under ``"structure"``.
    """
    sample_dir = Path(out_dir) / query_id / f"seed_{seed}"
    prefix = sample_dir / f"{query_id}_seed_{seed}_sample_{sample}"
    atoms = _protein_atoms(Path(f"{prefix}_model.cif"))
    coords, plddt = _arrays_from_atoms(atoms)
    aggregated = json.loads(Path(f"{prefix}_confidences_aggregated.json").read_text())
    captured = dict(captured or {})
    pae = None
    if captured.get("pae_probs") is None:
        for suffix in ("npz", "json"):
            full = Path(f"{prefix}_confidences.{suffix}")
            if full.exists():
                if suffix == "npz":
                    with np.load(full) as z:
                        pae = np.asarray(z["pae"], dtype=np.float32)
                else:
                    pae = np.asarray(json.loads(full.read_text())["pae"], dtype=np.float32)
                break
    record = confidence.compact_prediction(
        plddt,
        pae_probs=captured.get("pae_probs"),
        contact_prob=captured.get("contact_probs"),
        pae=pae,
        ptm_model=aggregated.get("ptm"),
        iptm_model=aggregated.get("iptm"),
    )
    out = {"contact_dists": cdist(coords, coords), **record}
    if structure:
        out["structure"] = _pdb_text(atoms)
    return out


#: Settings that make OpenFold3 keep the PAE bin probabilities and the distogram contact
#: probabilities in its confidence outputs. (It computes both either way; this only stops it
#: discarding them.)
CONFIDENCE_RETURNS = {
    "confidence": {"pae": {"return_probs": True}, "distogram": {"return_contact_probs": True}},
}


def _first(x) -> np.ndarray:
    """``[batch, sample, ...]`` tensor -> numpy for batch 0, sample 0."""
    x = x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
    return x[0, 0] if x.ndim >= 3 and x.shape[0] == 1 else x


@dataclass
class OF3Model:
    """A loaded OpenFold3 runner, reused across queries.

    OpenFold3 computes the PAE bin probabilities and the distogram contact probabilities but its
    output writer only saves the expected PAE. :meth:`install_capture` wraps the writer so the two
    arrays are kept per query (and reduced by ``compact_prediction`` to ``(L, L)`` matrices).
    """

    runner: Any
    output_dir: Path
    _owns_dir: bool = field(default=False, repr=False)
    captured: dict = field(default_factory=dict, repr=False)
    capturing: bool = field(default=False, repr=False)

    # Properties OpenFold3 caches for one query set; cleared before each query.
    _PER_QUERY_CACHE = ("lightning_data_module", "data_module_config")

    def install_capture(self) -> bool:
        """Wrap OpenFold3's output writer to keep ``pae_probs`` and ``contact_probs``. Returns success."""
        try:
            writer = next(c for c in self.runner.callbacks if hasattr(c, "write_all_outputs"))
            original = writer.write_all_outputs
        except (StopIteration, AttributeError):
            print_err("[WARN] OpenFold3's output writer was not found: pTM, ipTM and actifpTM will be NaN.")
            return False

        def write_all_outputs(batch, outputs, confidence_scores):
            for b, query_id in enumerate(batch["query_id"]):
                try:
                    self.captured[query_id] = {
                        key: _first(confidence_scores[key][b:b + 1])
                        for key in ("pae_probs", "contact_probs") if key in confidence_scores
                    }
                except Exception as e:  # never let capture break the prediction
                    print_err(f"[WARN] Could not keep OpenFold3 confidence probabilities for {query_id}: {e}")
            return original(batch, outputs, confidence_scores)

        writer.write_all_outputs = write_all_outputs
        self.capturing = True
        return True

    def predict(self, query_json: Path, query_id: str, seed: int, structure: bool = False) -> dict[str, Any]:
        from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet

        for name in self._PER_QUERY_CACHE:
            self.runner.__dict__.pop(name, None)
        self.runner.seeds = [seed]
        self.runner.experiment_config.experiment_settings.seeds = [seed]
        self.runner.run(InferenceQuerySet.from_json(query_json))
        captured = self.captured.pop(query_id, None)
        if self.capturing and not captured:
            print_err(f"[WARN] No OpenFold3 confidence probabilities captured for {query_id}; "
                      "pTM, ipTM and actifpTM will be NaN.")
        return read_prediction(self.output_dir, query_id, seed, structure=structure, captured=captured)

    def close(self) -> None:
        if self._owns_dir:
            shutil.rmtree(self.output_dir, ignore_errors=True)


def make_of3_model(
    cpu: bool = False,
    checkpoint: str | os.PathLike | None = None,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    num_diffusion_samples: int = 1,
    output_dir: str | os.PathLike | None = None,
    **kwargs,
) -> OF3Model:
    """Build an ``InferenceExperimentRunner`` and load the weights once."""
    ckpt = ensure_checkpoint(checkpoint_name, checkpoint)
    try:
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.entry_points.validator import InferenceExperimentConfig
    except ImportError as e:
        raise ImportError('OpenFold3 is not installed. Try: pip install "yunta[of3]"') from e
    owns = output_dir is None
    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="yunta_of3_"))
    out.mkdir(parents=True, exist_ok=True)
    config = InferenceExperimentConfig(
        inference_ckpt_path=ckpt,
        experiment_settings={
            "output_dir": out, "seeds": [42], "use_msa_server": False,
            "use_templates": False, "skip_existing": False,
        },
        pl_trainer_args={"accelerator": "cpu", "devices": 1} if cpu else {"devices": 1},
        **({"model_update": {"presets": ["predict"], "custom": CONFIDENCE_RETURNS}} | kwargs),
    )
    runner = InferenceExperimentRunner(
        config, num_diffusion_samples=num_diffusion_samples,
        use_msa_server=False, use_templates=False, output_dir=out,
    )
    runner.setup()
    model = OF3Model(runner=runner, output_dir=out, _owns_dir=owns)
    model.install_capture()
    return model


def run_of3(
    paired_msa,                 # (n_rows, n_cols) array of residue letters (bytes), query first
    chain_a_length: int,
    _id: str,
    model: OF3Model,
    seed: int | None = None,
    names: list[str] | None = None,
    to_input: Callable | None = None,
    workdir: str | os.PathLike | None = None,
    return_structure: bool = False,
):
    """Model one block; returns ``(inverse distances, distances, compact prediction)``.

    ``return_structure`` adds the model as PDB text (``"structure"``); request it only for a
    whole-complex call.
    """
    seed = 42 if seed is None else seed
    rows = to_input(paired_msa) if to_input is not None else list(paired_msa)
    names = names or [_id] + [f"hit{i}" for i in range(1, len(rows))]
    # unique per call: every block of a chunked complex shares `_id`, and a reused id would
    # collide in the output directory (and could be read back stale)
    query_id = f"q{_safe(_id)}_{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="yunta_of3_q_") as tmp:
        query = write_query(workdir or tmp, query_id, rows, names, chain_a_length)
        pred = model.predict(query, query_id, seed, structure=return_structure)
    dists = pred["contact_dists"]
    with np.errstate(divide="ignore"):
        inv = 1. / dists
    inv[np.isinf(inv)] = 0.
    return inv, dists, {k: v for k, v in pred.items() if k != "contact_dists"}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:80]


def post_of3(paired_msa, results: list[tuple[Block, tuple]], contact_radius: float = 8., pae_cutoff: float = 10., **kwargs) -> dict:
    """Interface scores from OpenFold3 blocks, by the same code as the AF2 backend (``confidence.score_blocks``).

    pTM, ipTM and actifpTM are recomputed from the PAE bin probabilities (``ptm_model``/``iptm_model``
    hold OpenFold3's own values for cross-checking) and need a whole-complex call (else NaN).
    PAE-based scores are blockwise approximations when chunked (see ``score_mode``).
    """
    scores, one = confidence.score_blocks(paired_msa, results, contact_radius=contact_radius, pae_cutoff=pae_cutoff)
    nan = float("nan")
    for key in ("ptm_model", "iptm_model"):
        scores[key] = one.get(key) if one.get(key) is not None else nan
    if "structure" in one:
        scores["structure"] = one["structure"]
    return scores
