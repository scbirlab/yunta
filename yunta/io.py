"""Tools for input and output."""

from typing import Any, Union
from csv import DictWriter
from dataclasses import is_dataclass, asdict
from io import TextIOWrapper
import os

from carabiner import print_err
from carabiner.cast import cast


def assign_chains(pdb_info: str, chainA_length: int) -> str:
    """Label residues ``A`` up to ``chainA_length`` and ``B`` after it in PDB text.

    >>> line = "ATOM      1  N   MET A   1      11.104   6.134  -6.504  1.00 90.00           N"
    >>> out = assign_chains(line.replace("MET A   1", "MET A   3"), 2)
    >>> out[21]
    'B'
    """
    from .structs.pdb_structs import ATMRecord

    chain_name, out = 'A', []
    for line in pdb_info.split('\n'):
        try:
            record = ATMRecord(line)
            if record.res_no > chainA_length:
                chain_name = 'B'
            out.append(f"{line[:21]}{chain_name}{line[22:]}")
        except (ValueError, IndexError):
            out.append(line)
    return '\n'.join(out)


def save_design(
    pdb_info,
    output_name: str, 
    chainA_length: int
) -> None:
    """Save the resulting protein-protein model to a pdb file."""
    with open(output_name, 'w') as f:
        print(assign_chains(pdb_info, chainA_length), file=f)
    return None


def write_metrics(
    metrics: Any,
    filename: os.PathLike, 
    mode: str = 'w'
) -> None:
    
    """
    
    """

    if is_dataclass(metrics):
        metrics = [metrics]
    metrics = list(metrics)
    if not all(is_dataclass(m) for m in metrics):
        raise TypeError("All metrics must be dataclass objects.")
    print_err(f"[INFO] Writing metrics to {filename}")
    for i, metric in enumerate(metrics):
        metric.write(
            filename=filename,
            mode=mode if i == 0 else "a",
            skip_header=mode != 'w' or i > 0,
        )
    return None
