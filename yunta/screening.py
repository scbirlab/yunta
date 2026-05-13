"""Tools to screen for PPIs."""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from io import TextIOWrapper
from itertools import combinations
import os
import random
import sys
from time import time

from carabiner import cast, print_err
import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from .structs.metrics import DCAMetrics, ModelMetrics, RF2TMetrics
from .structs.msa import MSA, PairedMSA
from .interactions import 


def _normalize_msa_file2(
    msa_file2: Optional[Union[str, TextIOWrapper, Iterable]],
) -> list:
    if msa_file2 is None:
        return [None]
    if isinstance(msa_file2, (str, TextIOWrapper)):
        return [msa_file2]
    return list(msa_file2)


def _screen_many_vs_many(
    one_vs_many_fn: Callable,
    msa_files1: Iterable,
    msa_files2: Optional[Iterable] = None,
    **kwargs,
) -> list:
    msa_files1 = list(msa_files1)
    if msa_files2 is None:
        print_err("[WARN] No second set of MSAs provided; screening all pairwise interactions.")
        msa_files2 = msa_files1[:]
    print_err(f"[INFO] Screening {len(msa_files1)} MSAs against {len(msa_files2)} MSAs...")
    results = []
    for msa_file1 in tqdm(msa_files1):
        results += one_vs_many_fn(
            msa_file1=msa_file1, 
            msa_file2=msa_files2, 
            **kwargs,
        )
    return results


def rf2track_one_vs_many(
    msa_file1: Union[str, TextIOWrapper],
    msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    cpu: bool = True
) -> List[Tuple[np.ndarray, np.ndarray, RF2TMetrics]]:

    from rf2t_micro.predict_msa import Predictor
    import torch

    if msa_file2 is None:
        msa_file2 = [None]
    if isinstance(msa_file2, str) or isinstance(msa_file2, TextIOWrapper):
        msa_file2 = [msa_file2]
    msa1 = MSA.from_file(msa_file1)
    results = []
    print_err(f"[INFO] Calculating contact matrix for {msa_file1} against {len(msa_file2)} MSAs...")

    torch.cuda.empty_cache()
    model = Predictor(use_cpu=cpu)
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        results.append(
            rf2track(
                msa1=msa1,
                msa2=msa2,
                model=model,
                max_gap_fraction=max_gap_fraction, 
                interaction_map=interaction_map,
            )
        )

    return results


def dca_one_vs_many(
    msa_file1: Union[str, TextIOWrapper], 
    msa_file2: Optional[Union[str, TextIOWrapper]] = None,
    apc: bool = False,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False
) -> List[DCAMetrics]:

    msa1 = MSA.from_file(msa_file1)
    msa_file2 = _normalize_msa_file2(msa_file2)
    results = []
    print_err(f"[INFO] Calculating DCA for {msa_file1} against {len(msa_file2)} MSAs...")
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        results.append(
            paired_dca(
                msa1=msa1,
                msa2=msa2,
                apc=apc,
                max_gap_fraction=max_gap_fraction,
                interaction_map=interaction_map,
                enforce_ref_match=enforce_ref_match,
            )
        )

    return results


def dca_many_vs_many(
    msa_files1: Iterable[Union[str, TextIOWrapper]],
    msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    apc: bool = False,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, DCAMetrics]]:
    results = []
    # msa_files1 = cast(msa_files1, to=lost)
    if msa_files2 is None:
        print_err("[WARN] No second set of MSAs provided,"
                  " so screening all pairwise interactions from the first set.")
        msa_files2 = [f for f in msa_files1]
    print_err(f"[INFO] Screening {len(msa_files1)} MSAs against {len(msa_files2)} MSAs...")
    for msa_file1 in tqdm(msa_files1):
        results += dca_one_vs_many(
                msa_file1=msa_file1,
                msa_file2=msa_files2,
                apc=apc,
                max_gap_fraction=max_gap_fraction,
                interaction_map=interaction_map,
                enforce_ref_match=enforce_ref_match,
            )
    return results


def dca_many_vs_many(
    msa_files1: Iterable[Union[str, TextIOWrapper]],
    msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    apc: bool = False,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, DCAMetrics]]:
    return _screen_many_vs_many(
        dca_one_vs_many, 
        msa_files1=msa_files1, 
        msa_files2=msa_files1, 
        apc=apc,
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
    )


def model_one_vs_many(
    msa_file1: Union[str, TextIOWrapper],
    output_dir: str,
    msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    pdockq_t: float = .5,
    force_save: bool = False,
    seed: Optional[int] = None,
    model_runner: Optional = None,
    *args, **kwargs
) -> List[ModelMetrics]:

    if model_runner is None:
        from .modelling import make_model_runner
        model_runner = make_model_runner(*args, **kwargs)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    msa1 = MSA.from_file(msa_file1)
    msa_file2 = _normalize_msa_file2(msa_file2)
    metrics = []
    print_err(f"[INFO] Running {msa_file1} against {len(msa_file2)} MSAs...")
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        metrics.append(
            build_evaluate_and_save_model(
                msa1=msa1,
                msa2=msa2,
                output_dir=output_dir,
                pdockq_t=pdockq_t,
                force_save=force_save,
                seed=seed,
                model_runner=model_runner,
                max_gap_fraction=max_gap_fraction,
                interaction_map=interaction_map,
            )
        )

    return metrics
    

def model_many_vs_many(
    msa_files1: Iterable[Union[str, TextIOWrapper]],
    output_dir: str,
    msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = .9,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    pdockq_t: float = .5,
    force_save: bool = True,
    seed: Optional[int] = None,
    model_runner: Optional = None,
    *args, **kwargs
) -> List[ModelMetrics]:

    if model_runner is None:
        from .modelling import make_model_runner
        model_runner = make_model_runner(*args, **kwargs)
    return _screen_many_vs_many(
        model_one_vs_many, 
        msa_files1=msa_files1, 
        msa_files2=msa_files2, 
        output_dir=output_dir, 
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        pdockq_t=pdockq_t,
        force_save=force_save,
        seed=seed,
        model_runner=model_runner,
    )
