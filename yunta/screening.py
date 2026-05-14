"""Tools to screen for PPIs."""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from io import TextIOWrapper
from itertools import combinations
import os
import random
import sys
from time import time

from carabiner import print_err
import numpy as np
from tqdm.auto import tqdm

from .structs.metrics import AF2Metrics, DCAMetrics, RF2TMetrics
from .structs.msa import MSA
from .interactions import AF2Runner, DCARunner, RF2TRunner

DEFAULT_MAX_GAP_FRACTION: float = .9


def _normalize_msa_file2(
    msa_file2: Optional[Union[str, TextIOWrapper, Iterable]],
) -> list:
    if msa_file2 is None:
        return [None]
    if isinstance(msa_file2, (str, TextIOWrapper)):
        return [msa_file2]
    return list(msa_file2)


def _screen_one_vs_many(
    runner: Callable,
    msa_file1: Union[str, TextIOWrapper],
    msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    **kwargs
) -> list:
    msa1 = MSA.from_file(msa_file1)
    msa_file2 = _normalize_msa_file2(msa_file2)
    results = []
    print_err(f"[INFO] Calculating contact matrix for {msa_file1} against {len(msa_file2)} MSAs using {runner}")
    for msa2 in tqdm(msa_file2):
        if msa2 is not None:
            msa2 = MSA.from_file(msa2)
        results.append(
            runner().run(
                msa1=msa1,
                msa2=msa2,
                max_gap_fraction=max_gap_fraction, 
                interaction_map=interaction_map,
                **kwargs,
            )
        )
    return results


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
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    cpu: bool = True,
    **kwargs
) -> List[Tuple[np.ndarray, np.ndarray, RF2TMetrics]]:
    return _screen_one_vs_many(
        runner=RF2TRunner,
        msa_file1=msa_file1,
        msa_file2=msa_file2,
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
        cpu=cpu,
        **kwargs
    )


def dca_one_vs_many(
    msa_file1: Union[str, TextIOWrapper], 
    msa_file2: Optional[Union[str, TextIOWrapper]] = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    apc: bool = False,
    **kwargs
) -> List[DCAMetrics]:
    return _screen_one_vs_many(
        runner=DCARunner,
        msa_file1=msa_file1,
        msa_file2=msa_file2,
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
        apc=apc,
        **kwargs
    )


def dca_many_vs_many(
    msa_files1: Iterable[Union[str, TextIOWrapper]],
    msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    apc: bool = False,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, DCAMetrics]]:
    return _screen_many_vs_many(
        dca_one_vs_many, 
        msa_files1=msa_files1, 
        msa_files2=msa_files2, 
        apc=apc,
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
    )


def model_one_vs_many(
    msa_file1: Union[str, TextIOWrapper],
    msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    seed: Optional[int] = None,
    max_recycles: int = 10,
    param_dir: Optional[str] = None,
    **kwargs
) -> List[AF2Metrics]:
    return _screen_one_vs_many(
        runner=AF2Runner,
        msa_file1=msa_file1,
        msa_file2=msa_file2,
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
        seed=seed,
        model_kwargs={"max_recycles": max_recycles, "param_dir": param_dir},
        **kwargs
    )
    

def model_many_vs_many(
    msa_files1: Iterable[Union[str, TextIOWrapper]],
    msa_files2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    max_recycles: int = 10,
    param_dir: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> List[AF2Metrics]:
    return _screen_many_vs_many(
        model_one_vs_many, 
        msa_files1=msa_files1,
        msa_files2=msa_files2,
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
        seed=seed,
        model_kwargs={"max_recycles": max_recycles, "param_dir": param_dir}
        **kwargs
    )
