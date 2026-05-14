"""Tools for input and output."""

from typing import Any, Union
from csv import DictWriter
from dataclasses import is_dataclass, asdict
from io import TextIOWrapper
import os

from carabiner import print_err
from carabiner.cast import cast


def save_design(
    pdb_info,
    output_name: str, 
    chainA_length: int
) -> None:

    """Save the resulting protein-peptide design to a pdb file.

    """

    from .structs.pdb_structs import ATMRecord

    chain_name = 'A'
    with open(output_name, 'w') as f:
        pdb_contents = pdb_info.split('\n')
        for line in pdb_contents:
            try:
                record = ATMRecord(line)
                if record.res_no > chainA_length:
                    chain_name = 'B'
                outline = f"{line[:21]}{chain_name}{line[22:]}"
                print(outline, file=f)
            except (ValueError, IndexError):
                print(line, file=f)
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
