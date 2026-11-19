"""Regression tests for the AF2 inference optimisations (patch 7).

Static checks need nothing but the standard library; the numerical-equivalence checks
need NumPy/absl; the model-level checks need JAX, haiku and TensorFlow and are skipped
without them.
"""

import ast
import os
import string
import time

import numpy as np
import pytest

VENDORED = os.path.join(
    os.path.dirname(__file__), "..", "yunta", "interactions", "af2", "src_speedppi", "alphafold",
)
PKG = "alphafold"
AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


# ---------------------------------------------------------------- static: dead code stays gone

def _modules():
    mods = {}
    for d, _, fs in os.walk(VENDORED):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(d, f)
                rel = os.path.relpath(p, VENDORED)[:-3].replace(os.sep, ".")
                mods[rel[:-9] if rel.endswith(".__init__") else ("" if rel == "__init__" else rel)] = p
    return mods


def _imports(mod, path):
    """Vendored modules imported by ``mod`` (relative and absolute spellings)."""
    out = set()
    is_pkg = path.endswith("__init__.py")
    base = mod.split(".") if mod else []
    base = base if is_pkg else base[:-1]
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                b = base[: len(base) - (n.level - 1)] if n.level > 1 else base
                tgt = ".".join(b + ([n.module] if n.module else []))
            else:
                tgt = (n.module or "")
                tgt = tgt[len(PKG) + 1:] if tgt.startswith(PKG + ".") else ("" if tgt == PKG else None)
                if tgt is None:
                    continue
            out.add(tgt)
            out |= {(tgt + "." + a.name).strip(".") for a in n.names}
    return out


def test_every_vendored_module_is_reachable_from_the_entry_points():
    mods = _modules()
    stack = ["model.model", "model.config", "model.data", "data.foldonly", "protein", "confidence"]
    seen = set()
    while stack:
        m = stack.pop()
        if m in seen or m not in mods:
            continue
        seen.add(m)
        stack += list(_imports(m, mods[m]))
        parts = m.split(".")
        stack += [".".join(parts[:i]) for i in range(len(parts))]
    assert sorted(set(mods) - seen) == [], "unreachable vendored modules (dead code)"


def test_no_upstream_test_files_or_template_pipeline_vendored():
    names = {os.path.basename(p) for p in _modules().values()}
    assert not [n for n in names if n.endswith("_test.py")]
    for gone in ("templates.py", "mmcif_parsing.py", "pipeline.py", "msaonly.py", "pair_msas.py", "lddt.py"):
        assert gone not in names
    assert not os.path.exists(os.path.join(VENDORED, "data", "tools"))


def test_training_only_functions_are_gone():
    banned = {
        "loss", "backbone_loss", "sidechain_loss", "supervised_chi_loss", "structural_violation_loss",
        "find_structural_violations", "compute_violation_metrics", "frame_aligned_point_error",
        "between_residue_bond_loss", "between_residue_clash_loss", "within_residue_violations",
        "find_optimal_renaming", "atom37_to_frames", "from_pdb_string", "tf_example_to_features",
        "eval_shape", "make_atom14_dists_bounds", "load_stereo_chemical_props",
    }
    found = []
    for mod, path in _modules().items():
        for n in ast.walk(ast.parse(open(path).read())):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in banned:
                found.append((mod, n.name))
    assert found == []


def test_vendored_code_has_no_unused_heavy_dependencies_and_no_compute_loss_plumbing():
    for mod, path in _modules().items():
        src = open(path).read()
        assert "compute_loss" not in src, mod
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "Bio", mod
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "Bio" for a in n.names), mod


def test_no_undefined_names_in_pruned_modules():
    """Cheap pyflakes substitute: every module still compiles and has no dangling ``loss`` call."""
    for mod, path in _modules().items():
        src = open(path).read()
        compile(src, path, "exec")
    modules_src = open(_modules()["model.modules"]).read()
    assert "loss(" not in modules_src.replace("compute_loss", "")


# ---------------------------------------------------------------- features: identical, faster

def _import_foldonly():
    from yunta.interactions.af2.src_speedppi.alphafold.data import foldonly, parsers
    return foldonly, parsers


def _reference_parse_a3m(sequences):
    """The original per-character implementation."""
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec, deletion_count = [], 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)
    table = str.maketrans("", "", string.ascii_lowercase)
    return [s.translate(table) for s in sequences], deletion_matrix


def _random_msa(n, L, seed=0, gaps=0.2, inserts=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = list(rng.choice(AA, L))
        for i in np.flatnonzero(rng.random(L) < gaps):
            row[i] = "-"
        for i in np.flatnonzero(rng.random(L) < inserts):
            row[i] = row[i] + "".join(rng.choice(list(string.ascii_lowercase), rng.integers(1, 4)))
        rows.append("".join(row))
    return rows


@pytest.mark.parametrize("inserts", [0.0, 0.1])
def test_parse_a3m_matches_reference(inserts):
    _, parsers = _import_foldonly()
    rows = _random_msa(40, 30, seed=1, inserts=inserts)
    assert parsers.parse_a3m(rows) == _reference_parse_a3m(rows)


def test_msa_features_match_dict_encoding_and_dedupe_across_msas():
    foldonly, parsers = _import_foldonly()
    from yunta.interactions.af2.src_speedppi.alphafold import residue_constants as rc
    a, b = _random_msa(30, 25, seed=2), _random_msa(20, 25, seed=3)
    b[4] = a[7]  # duplicate across MSAs: first occurrence wins
    parsed = [parsers.parse_a3m(m) for m in (a, b)]
    feats = foldonly.make_msa_features([p[0] for p in parsed], [p[1] for p in parsed])
    seen, expected = set(), []
    for m in (a, b):
        for s in m:
            if s not in seen:
                seen.add(s)
                expected.append([rc.HHBLITS_AA_TO_ID[c] for c in s])
    assert np.array_equal(feats["msa"], np.array(expected, dtype=np.int32))
    assert feats["msa"].dtype == np.int32
    assert feats["deletion_matrix_int"].shape == feats["msa"].shape
    assert (feats["num_alignments"] == len(expected)).all() and feats["num_alignments"].shape == (25,)


def test_invalid_residue_still_raises_keyerror():
    foldonly, parsers = _import_foldonly()
    rows = ["ACDE", "AC*E"]
    msa, dm = parsers.parse_a3m(rows)
    with pytest.raises(KeyError):
        foldonly.make_msa_features([msa], [dm])


def test_pipeline_features_have_template_placeholders_but_no_template_search():
    foldonly, _ = _import_foldonly()
    rows = _random_msa(6, 12, seed=4, gaps=0.0)
    f = foldonly.FoldDataPipeline().process(rows[0], "a_b", [rows])
    assert {"aatype", "msa", "deletion_matrix_int", "residue_index", "seq_length", "num_alignments"} <= set(f)
    assert f["template_aatype"].size == 0 and f["template_all_atom_positions"].dtype == np.float32
    import inspect
    assert "template_search" not in inspect.signature(foldonly.FoldDataPipeline.process).parameters


def test_vectorised_encoding_is_faster_than_the_per_character_loop():
    foldonly, parsers = _import_foldonly()
    from yunta.interactions.af2.src_speedppi.alphafold import residue_constants as rc
    rows = _random_msa(1500, 300, seed=5)

    def old():
        return np.array([[rc.HHBLITS_AA_TO_ID[c] for c in s] for s in rows], dtype=np.int32)

    def best(fn, reps=3):
        out = []
        for _ in range(reps):
            t = time.perf_counter(); r = fn(); out.append(time.perf_counter() - t)
        return min(out), r

    t_old, ref = best(old)
    t_new, new = best(lambda: foldonly._encode(rows, 300))
    assert np.array_equal(ref, new)
    assert t_new * 3 < t_old  # relative, not absolute: robust to slow or busy machines


# ---------------------------------------------------------------- Cβ extraction


def _reference_cb(prot, rc):
    """The original: collect while walking every atom of every residue."""
    out = []
    restypes = rc.restypes + ["X"]
    for i in range(prot.aatype.shape[0]):
        res3 = rc.restype_1to3.get(restypes[prot.aatype[i]], "UNK")
        for name, pos, mask in zip(rc.atom_types, prot.atom_positions[i], prot.atom_mask[i]):
            if mask < 0.5:
                continue
            if name == "CB" or (name == "CA" and res3 == "GLY"):
                out.append(pos)
    return np.array(out)


def _random_protein(n=40, seed=0):
    from yunta.interactions.af2.src_speedppi.alphafold import protein, residue_constants as rc
    rng = np.random.default_rng(seed)
    aatype = rng.integers(0, 21, n)
    mask = (rng.random((n, 37)) < 0.9).astype(np.float32)
    return protein, rc, protein.Protein(
        aatype=aatype, atom_positions=rng.normal(size=(n, 37, 3)).astype(np.float32),
        atom_mask=mask, residue_index=np.arange(1, n + 1), b_factors=np.zeros((n, 37), np.float32),
    )


@pytest.mark.parametrize("seed", range(5))
def test_cb_coordinates_match_the_pdb_writer_loop(seed):
    protein, rc, prot = _random_protein(seed=seed)
    ref = _reference_cb(prot, rc)
    assert np.array_equal(protein.cb_coordinates(prot), ref)
    pdb, cb = protein.to_pdb(prot)
    assert np.array_equal(cb, ref) and pdb.startswith("MODEL")


def test_contact_matrix_does_not_build_pdb_text(monkeypatch):
    pytest.importorskip("scipy")
    protein, rc, prot = _random_protein(seed=9)
    from yunta.interactions.af2 import scoring

    def boom(*a, **k):
        raise AssertionError("to_pdb must not be called just to get Cβ coordinates")

    monkeypatch.setattr(protein, "to_pdb", boom)
    inv, d = scoring.get_contact_matrix(prot)
    assert inv.shape == d.shape and (np.diag(inv) == 0).all()


# ---------------------------------------------------------------- inference config

def test_inference_config_disables_templates_but_keeps_msa_clusters():
    pytest.importorskip("ml_collections")
    from yunta.interactions.af2.modelling import inference_config
    for name in ("model_1_ptm", "model_2_ptm", "model_3_ptm"):
        stock = inference_config(name, use_templates=True)
        lean = inference_config(name, use_templates=False)
        assert lean.model.embeddings_and_evoformer.template.enabled is False
        assert lean.model.embeddings_and_evoformer.template.embed_torsion_angles is False
        assert lean.data.common.use_templates is False
        # MSA subsampling must not change: same number of sampled clusters as the stock config
        assert (lean.data.common.reduce_msa_clusters_by_max_templates
                == stock.data.common.reduce_msa_clusters_by_max_templates)
        assert lean.data.eval.max_msa_clusters == stock.data.eval.max_msa_clusters
        assert lean.data.common.max_extra_msa == stock.data.common.max_extra_msa
    assert inference_config("model_1_ptm", use_templates=True).model.embeddings_and_evoformer.template.enabled
    assert inference_config("model_1_ptm", num_ensemble=2, max_recycles=3).model.num_recycle == 3


def test_compilation_cache_is_opt_in(monkeypatch, tmp_path):
    from yunta.interactions.af2 import modelling
    monkeypatch.delenv("YUNTA_JAX_CACHE", raising=False)
    assert modelling.enable_compilation_cache() is None
    jax = pytest.importorskip("jax")
    target = tmp_path / "xla"
    assert modelling.enable_compilation_cache(str(target)) == str(target) and target.is_dir()
    assert jax.config.jax_compilation_cache_dir == str(target)


# ---------------------------------------------------------------- model level (JAX + haiku + TF)

@pytest.mark.slow
def test_template_free_model_is_bit_identical_to_stock_and_runs_end_to_end():
    pytest.importorskip("jax"); pytest.importorskip("haiku"); pytest.importorskip("tensorflow")
    pytest.importorskip("ml_collections"); pytest.importorskip("tree")
    from yunta.interactions.af2 import run
    from yunta.interactions.af2.modelling import inference_config
    from yunta.interactions.af2.src_speedppi.alphafold.model import model

    msa = _random_msa(1, 24, seed=6, gaps=0.0) + _random_msa(9, 24, seed=7, gaps=0.15)  # gap-free query

    def small(use_templates):
        c = inference_config("model_1_ptm", num_ensemble=1, max_recycles=1, use_templates=use_templates)
        c.data.eval.max_msa_clusters = 8
        c.data.common.max_extra_msa = 16
        c.model.embeddings_and_evoformer.evoformer_num_block = 1
        c.model.embeddings_and_evoformer.extra_msa_stack_num_block = 1
        c.model.heads.structure_module.num_layer = 1
        return c

    out, params = {}, None
    for flag in (True, False):
        m = model.RunModel(small(flag), params)
        _, _, out[flag] = run.model_protein_interaction(msa, 12, "a-b", model=m, seed=3)
        params = params or m.params  # same (random) weights for both
    stock, lean = out[True], out[False]
    for key in ("plddt", "predicted_aligned_error", "ptm"):
        assert np.array_equal(np.asarray(stock[key]), np.asarray(lean[key])), key
    assert np.array_equal(stock["structure_module"]["final_atom_positions"],
                          lean["structure_module"]["final_atom_positions"])
    assert set(lean["structure_module"]) == {"final_atom_positions", "final_atom_mask"}


@pytest.mark.slow
def test_recycle_early_stop_matches_the_stock_loop_and_stops_early():
    pytest.importorskip("jax"); pytest.importorskip("haiku"); pytest.importorskip("tensorflow")
    pytest.importorskip("ml_collections"); pytest.importorskip("tree")
    from yunta.interactions.af2 import run
    from yunta.interactions.af2.modelling import inference_config
    from yunta.interactions.af2.src_speedppi.alphafold.model import model

    msa = _random_msa(1, 24, seed=6, gaps=0.0) + _random_msa(9, 24, seed=7, gaps=0.15)
    c = inference_config("model_1_ptm", num_ensemble=1, max_recycles=3)
    c.data.eval.max_msa_clusters = 8
    c.data.common.max_extra_msa = 16
    c.model.embeddings_and_evoformer.evoformer_num_block = 1
    c.model.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    c.model.heads.structure_module.num_layer = 1

    out, passes, params = {}, {}, None
    for tol in (None, 0.0, 1e9):  # stock loop / python loop running every recycle / stop at once
        m = model.RunModel(c, params)
        m.recycle_tolerance = tol
        _, _, out[tol] = run.model_protein_interaction(msa, 12, "a-b", model=m, seed=3)
        passes[tol] = m.last_num_passes
        params = params or m.params
    assert passes == {None: 4, 0.0: 4, 1e9: 2}  # a huge tolerance stops after the first comparison
    for key in ("plddt", "predicted_aligned_error"):
        np.testing.assert_allclose(np.asarray(out[0.0][key]), np.asarray(out[None][key]), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out[0.0]["structure_module"]["final_atom_positions"],
                               out[None]["structure_module"]["final_atom_positions"], atol=1e-3)
    assert "representations" not in out[0.0]
