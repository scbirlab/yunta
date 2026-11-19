"""AF2 memory options: adaptive MSA padding, unified memory, training-only heads."""
import sys
from types import SimpleNamespace

import pytest


def _cfg(clusters=512, extra=5120, templates=4, reduce=True):
    return SimpleNamespace(data=SimpleNamespace(
        common=SimpleNamespace(max_extra_msa=extra, reduce_msa_clusters_by_max_templates=reduce),
        eval=SimpleNamespace(max_msa_clusters=clusters, max_templates=templates)))


def test_adaptive_padding_buckets_and_caps():
    from yunta.interactions.af2.modelling import adapt_msa_padding
    cfg = _cfg(templates=0, reduce=False)
    limits = adapt_msa_padding(cfg, 20)
    assert limits == (512, 5120)
    assert cfg.data.eval.max_msa_clusters < 512 and cfg.data.common.max_extra_msa < 5120
    small = (cfg.data.eval.max_msa_clusters, cfg.data.common.max_extra_msa)
    adapt_msa_padding(cfg, 100000, limits)                       # deep MSA returns to the stock sizes
    assert (cfg.data.eval.max_msa_clusters, cfg.data.common.max_extra_msa) == (512, 5120)
    adapt_msa_padding(cfg, 21, limits)
    assert (cfg.data.eval.max_msa_clusters, cfg.data.common.max_extra_msa) == small  # bucketed, stable


def test_adaptive_padding_keeps_the_template_reserve():
    from yunta.interactions.af2.modelling import adapt_msa_padding
    cfg = _cfg(clusters=516, templates=4, reduce=True)
    adapt_msa_padding(cfg, 10)
    assert cfg.data.eval.max_msa_clusters >= 4 + 1


def test_unified_memory_sets_env_only_before_jax_starts(monkeypatch):
    from yunta.interactions.af2 import modelling
    monkeypatch.delenv("TF_FORCE_UNIFIED_MEMORY", raising=False)
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    assert modelling.configure_unified_memory(3.0) is True
    import os
    assert os.environ["TF_FORCE_UNIFIED_MEMORY"] == "1" and os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "3.0"


# --- OF3 / PDB -----------------------------------------------------------------------
def test_training_only_heads_are_off():
    pytest.importorskip("ml_collections")
    from yunta.interactions.af2.modelling import inference_config

    heads = inference_config("model_1_ptm").model.heads
    assert heads.masked_msa.weight == 0.0
    assert heads.experimentally_resolved.weight == 0.0
    # heads yunta reads must stay on
    assert heads.distogram.weight > 0 and heads.predicted_lddt.weight > 0
    assert heads.structure_module.weight > 0
