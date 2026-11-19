"""Tools for setting up and using models."""

from typing import Any, Optional
import os

from carabiner import print_err

from ...weights import get_model_weights

_COMPILE_CACHE_ENV = "YUNTA_JAX_CACHE"


def inference_config(
    model_name: str = "model_1_ptm",
    num_ensemble: int = 1,
    max_recycles: int = 10,
    use_templates: bool = False,
) -> Any:
    """AlphaFold2 config for MSA-only inference.

    yunta never supplies templates. Models 1 and 2 are nonetheless configured with template
    embedding (four padded, fully masked templates), which costs a template pair stack per
    recycle and four extra MSA rows. The embedding is already multiplied by
    ``sum(template_mask) > 0`` (zero here) and the extra rows are masked, so switching it off
    gives the same outputs (checked bit-for-bit in the tests) at a fraction of the compute.
    Pass ``use_templates=True`` to keep the stock configuration.
    """
    from .src_speedppi.alphafold.model import config

    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = max_recycles
    model_config.model.num_recycle = max_recycles
    # Training-only heads: yunta never reads their outputs, and zero-weight heads are not built
    # (stock parameters still load, since extra entries are ignored).
    for head in ("masked_msa", "experimentally_resolved"):
        model_config.model.heads[head].weight = 0.0
    if not use_templates:
        evoformer = model_config.model.embeddings_and_evoformer
        evoformer.template.enabled = False
        evoformer.template.embed_torsion_angles = False
        model_config.data.common.use_templates = False
        # reduce_msa_clusters_by_max_templates is left as configured: it sets how many MSA
        # clusters are sampled (512 - 4 for models 1/2) and so must not change.
    return model_config


MSA_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)


def _bucket(n: int, cap: int) -> int:
    """Smallest bucket >= n (so only a handful of shapes are ever compiled), at most ``cap``."""
    return min(cap, next((b for b in MSA_BUCKETS if b >= n), cap))


def adapt_msa_padding(model_config: Any, n_rows: int, limits: Optional[tuple] = None) -> tuple:
    """Pad the sampled/extra MSA to the alignment's real depth instead of the full 512/5120 rows.

    AF2 pads the cluster MSA to ``max_msa_clusters`` and the extra MSA to ``max_extra_msa``
    with fully masked rows. Paired MSAs are usually far shallower than that, so most of the
    evoformer and extra-MSA-stack compute and memory went to rows that cannot contribute.
    Masked rows are excluded from every attention, outer product and average, so this
    changes the result only by floating-point summation order. Sizes snap to a few buckets
    (:data:`MSA_BUCKETS`) so a screen compiles a handful of programs, not one per depth.

    ``limits`` is ``(max_msa_clusters, max_extra_msa)`` of the stock configuration (read from
    ``model_config`` the first time and then pass them back); returns them for reuse.
    Sets ``data.eval.max_msa_clusters`` and ``data.common.max_extra_msa`` in place.
    """
    common, ev = model_config.data.common, model_config.data.eval
    if limits is None:
        limits = (ev.max_msa_clusters, common.max_extra_msa)
    max_clusters, max_extra = limits
    reserved = ev.max_templates if common.reduce_msa_clusters_by_max_templates else 0  # rows kept for templates
    stock_pad = max_clusters - reserved
    clusters = _bucket(min(n_rows, stock_pad), stock_pad)
    extra = _bucket(max(n_rows - clusters, 1), max_extra)
    ev.max_msa_clusters = clusters + reserved
    common.max_extra_msa = extra
    return limits


def configure_unified_memory(mem_fraction: float = 4.0) -> bool:
    """Allow GPU memory over-subscription (spills to host RAM). Returns False if too late.

    Both settings are read when the JAX/XLA backend initialises, so this only works before
    JAX has created its backend; afterwards it warns and changes nothing.
    """
    import sys
    if "jax" in sys.modules:
        try:
            from jax._src import xla_bridge
            if xla_bridge.backends_are_initialized():
                print_err("[WARN] JAX is already initialised; unified memory must be enabled before first use.")
                return False
        except Exception:  # private API moved: be conservative and still set the variables
            pass
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    return True


def enable_compilation_cache(path: Optional[str] = None) -> Optional[str]:
    """Persist XLA compilations so every process (and later run) skips recompiling.

    AF2 recompiles for each new residue count (~minutes on GPU), and parallel workers all
    repeat it. The cache directory is ``path``, else ``$YUNTA_JAX_CACHE``; with neither it is
    left to JAX's own settings. Returns the directory in use, if any.
    """
    path = path or os.environ.get(_COMPILE_CACHE_ENV)
    if path:
        import jax
        os.makedirs(path, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", path)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    return path


def make_model_runner(
    num_ensemble: int = 1,
    max_recycles: int = 10,
    param_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    use_templates: bool = False,
    adaptive_msa: bool = True,
    subbatch_size: Optional[int] = None,
    unified_memory: bool = False,
    recycle_tolerance: Optional[float] = None,
):
    
    """Generate an AlphaFold2 model runner.

    Memory options (all leave the model unchanged; they trade compute or placement for peak memory):

    * ``adaptive_msa`` (default on): pad the MSA to the alignment's real depth (see
      :func:`adapt_msa_padding`) rather than 512 clusters + 5120 extra rows.
    * ``subbatch_size``: rows/columns processed at once in AF2's attention and outer-product
      sub-batching (stock: 4). Smaller lowers peak memory at some cost in speed; this is AF2's own
      mechanism for long sequences.
    * ``recycle_tolerance`` (default off; this one *is* approximate): stop recycling early once the RMS
      change of the C-alpha distance matrix between two passes falls below this many Angstrom. ``0``
      runs every recycle and reproduces the stock result, only driven from Python. See
      ``RunModel.predict``.
    * ``unified_memory``: let the GPU spill to host RAM (``TF_FORCE_UNIFIED_MEMORY``,
      ``XLA_PYTHON_CLIENT_MEM_FRACTION=4``) so complexes larger than device memory run, slowly,
      instead of failing. Must be requested before JAX is first used.
    """
    if unified_memory:
        configure_unified_memory()
    
    from .src_speedppi.alphafold.model import data, model
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("Tensorflow is not installed. Try pip install yunta[af].") from e
    tf.config.experimental.set_visible_devices([], "GPU")  # TF is only used for CPU feature processing
    from jax.lib import xla_bridge

    enable_compilation_cache()
    print_err(f"[INFO] Setting up AlphaFold2 model. XLA platform available: {xla_bridge.get_backend().platform}")

    if model_name is None:
        model_name = 'model_1_ptm'
    if param_dir is None:
        param_dir = get_model_weights(model_name)

    model_config = inference_config(
        model_name,
        num_ensemble=num_ensemble,
        max_recycles=max_recycles,
        use_templates=use_templates,
    )
    model_params = data.get_model_haiku_params(
        model_name=model_name, 
        data_dir=param_dir,
    )

    if subbatch_size is not None:
        if subbatch_size < 1:
            raise ValueError(f"subbatch_size must be >= 1, got {subbatch_size}")
        model_config.model.global_config.subbatch_size = subbatch_size
    runner = model.RunModel(model_config, model_params)
    if recycle_tolerance is not None and recycle_tolerance < 0:
        raise ValueError(f"recycle_tolerance must be >= 0, got {recycle_tolerance}")
    runner.recycle_tolerance = recycle_tolerance
    runner.adaptive_msa = adaptive_msa
    runner.msa_limits = (model_config.data.eval.max_msa_clusters, model_config.data.common.max_extra_msa)
    return runner
