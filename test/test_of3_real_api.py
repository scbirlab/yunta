"""Checks against the *real* OpenFold3 package (skipped if it is not importable).

These do not run the model (no GPU/weights needed); they validate what yunta hands to OpenFold3
against OpenFold3's own schemas and class layout, which the stand-in model in test_of3.py cannot.
"""
import json
from functools import cached_property

import pytest

pytest.importorskip("openfold3.entry_points.experiment_runner")
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner  # noqa: E402
from openfold3.entry_points.validator import InferenceExperimentConfig  # noqa: E402
from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet  # noqa: E402

from yunta.interactions.of3.run import DEFAULT_CHECKPOINT, OF3Model, write_query  # noqa: E402


def test_generated_query_validates_against_openfold3_schema(tmp_path):
    rows = ["ACDEFG" "HIKL", "ACDEFG" "HIKL", "A-DEFG" "HI-L"]
    path = write_query(tmp_path, "q1", rows, ["query", "h1", "h2"], 6)
    qs = InferenceQuerySet.from_json(path)  # raises on any schema/path error
    chains = qs.queries["q1"].chains
    assert [c.chain_ids for c in chains] == [["A"], ["B"]]
    assert [c.sequence for c in chains] == ["ACDEFG", "HIKL"]
    assert all(c.paired_msa_file_paths and c.main_msa_file_paths for c in chains)


def test_default_checkpoint_name_is_in_openfold3_registry():
    from openfold3.entry_points.parameters import OPENFOLD_MODEL_CHECKPOINT_REGISTRY
    assert DEFAULT_CHECKPOINT in OPENFOLD_MODEL_CHECKPOINT_REGISTRY


@pytest.mark.parametrize("cpu", [True, False])
def test_experiment_config_and_runner_accept_our_arguments(tmp_path, cpu):
    ckpt = tmp_path / "w.pt"; ckpt.write_bytes(b"x")  # only existence is validated at construction
    out = tmp_path / "out"
    config = InferenceExperimentConfig(
        inference_ckpt_path=ckpt,
        experiment_settings={"output_dir": out, "seeds": [42], "use_msa_server": False,
                             "use_templates": False, "skip_existing": False},
        pl_trainer_args={"accelerator": "cpu", "devices": 1} if cpu else {"devices": 1},
    )
    runner = InferenceExperimentRunner(config, num_diffusion_samples=1, use_msa_server=False,
                                       use_templates=False, output_dir=out)
    assert runner.use_msa_server is False and runner.use_templates is False
    assert runner.num_diffusion_samples == 1


def test_per_query_cache_names_are_real_cached_properties():
    """OF3Model clears these between queries; if OpenFold3 renamed them the reset would silently no-op."""
    for name in OF3Model._PER_QUERY_CACHE:
        assert isinstance(getattr(InferenceExperimentRunner, name), cached_property), name
    for attr in ("setup", "run"):
        assert callable(getattr(InferenceExperimentRunner, attr))


def test_confidence_return_settings_are_accepted_and_reach_the_model_config(tmp_path):
    """The settings that keep pae_probs / contact_probs must be valid and be applied by OpenFold3."""
    from yunta.interactions.of3.run import CONFIDENCE_RETURNS
    ckpt = tmp_path / "w.pt"; ckpt.write_bytes(b"x")
    config = InferenceExperimentConfig(
        inference_ckpt_path=ckpt,
        experiment_settings={"output_dir": tmp_path / "o", "seeds": [42], "use_msa_server": False,
                             "use_templates": False, "skip_existing": False},
        model_update={"presets": ["predict"], "custom": CONFIDENCE_RETURNS},
    )
    runner = InferenceExperimentRunner(config, num_diffusion_samples=1, use_msa_server=False,
                                       use_templates=False, output_dir=tmp_path / "o")
    cfg = runner.model_config
    assert cfg.confidence.pae.return_probs and cfg.confidence.distogram.return_contact_probs
    assert (cfg.confidence.pae.bin_min, cfg.confidence.pae.bin_max, cfg.confidence.pae.no_bins) == (0, 32, 64)  # PAE_BIN_CENTERS assumes this


def test_output_writer_still_exposes_write_all_outputs():
    """The capture hook wraps this method; if OpenFold3 renamed it the hook would silently find nothing."""
    from openfold3.core.runners.writer import OF3OutputWriter
    import inspect
    assert list(inspect.signature(OF3OutputWriter.write_all_outputs).parameters) == [
        "self", "batch", "outputs", "confidence_scores"]
