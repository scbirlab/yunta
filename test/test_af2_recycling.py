import ast
from pathlib import Path


def test_vendored_af2_never_discards_an_immutable_jax_update():
    """``x.at[i].set(v)`` as a bare statement is a no-op in JAX (the recycled first row was lost)."""
    src = (Path(__file__).parent.parent / "yunta/interactions/af2/src_speedppi/alphafold/model/modules.py").read_text()
    bad = [n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
           and isinstance(n.value.func, ast.Attribute) and n.value.func.attr in {"set", "add", "multiply", "min", "max"}
           and isinstance(n.value.func.value, ast.Subscript)
           and isinstance(n.value.func.value.value, ast.Attribute) and n.value.func.value.value.attr == "at"]
    assert not bad, f"discarded .at[...] update(s) at line(s) {bad}"
