"""Tools for input and output."""

from typing import Any, Dict, Iterable, Optional, Tuple, Union

from csv import DictWriter
from dataclasses import is_dataclass, asdict
from io import TextIOWrapper
import os
import shutil
import tempfile

from bioino import FastaCollection
from carabiner import print_err
from carabiner.cast import cast
import numpy as np
from pandas import DataFrame

from .structs.pdb_structs import ATMRecord
from .structs.metrics import ModelMetrics

def save_design(pdb_info,
                output_name: str, 
                chainA_length: int) -> None:

    """Save the resulting protein-peptide design to a pdb file.

    """

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
            except Exception:
                print(line, file=f)
    return None


def write_metrics(
    metrics: Union[Iterable, Any],
    filename: Union[str, TextIOWrapper], 
    mode: str = 'w',
):
    
    """
    
    """

    if is_dataclass(metrics):
        metrics = [metrics]
    if not all(is_dataclass(m) for m in metrics):
        raise TypeError("All metrics must be dataclass objects.")
    f = cast(filename, to=TextIOWrapper, mode=mode)
    if f.name != '<stdout>':
        dirname = os.path.dirname(f.name)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)

    w = DictWriter(f, fieldnames=list(asdict(metrics[0])), delimiter='\t')
    if mode == 'w':
        w.writeheader()
    for metric in metrics:
        w.writerow(asdict(metric))
    return None
