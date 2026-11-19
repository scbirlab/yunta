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


def _as_msa(msa: Union[MSA, str, TextIOWrapper, None], cache: Optional[dict] = None) -> Optional[MSA]:
    """Load an MSA from a file, reusing ``cache`` (keyed by file name) if given."""
    if msa is None or isinstance(msa, MSA):
        return msa
    key = getattr(msa, "name", msa)
    if cache is not None and key in cache:
        return cache[key]
    loaded = MSA.from_file(msa)
    if cache is not None:
        cache[key] = loaded
    return loaded


def _screen_one_vs_many(
    runner: Callable,
    msa_file1: Union[MSA, str, TextIOWrapper],
    msa_file2: Optional[Iterable[Union[MSA, str, TextIOWrapper]]] = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    model: Optional[Callable] = None,
    msa_cache: Optional[dict] = None,
    **kwargs
) -> list:
    """Run ``runner`` on one MSA against each of a set of others, loading the model once."""
    runner_instance = runner()
    msa1 = _as_msa(msa_file1, msa_cache)
    msa_file2 = _normalize_msa_file2(msa_file2)
    if model is None:  # previously rebuilt (and AF2 re-JIT-compiled) for every pair
        model = runner_instance.make_model(
            cpu=kwargs.get("cpu", False), 
            **(kwargs.get("model_kwargs") or {}),
        )
    results = []
    print_err(f"[INFO] Calculating contact matrix for {msa1.name} against {len(msa_file2)} MSAs using {runner}")
    for msa2 in tqdm(msa_file2):
        msa2 = _as_msa(msa2, msa_cache)
        result = runner_instance.run(
            msa1=msa1,
            msa2=msa2,
            max_gap_fraction=max_gap_fraction, 
            interaction_map=interaction_map,
            enforce_ref_match=enforce_ref_match,
            model=model,
            **kwargs,
        )
        if result[0] is not None:
            results.append(result)
        else:
            print_err(f"[WARN] Failed calculation for {msa1}, {msa2}. Continuing.")
            continue
    return results


def _unordered_pairs(msa_files: list) -> list:
    """All unordered pairs ``(i, j)``, ``i <= j``; the diagonal is a homodimer (``None``).

    Examples
    ========
    >>> _unordered_pairs(["a", "b", "c"])
    [('a', [None, 'b', 'c']), ('b', [None, 'c']), ('c', [None])]

    """
    return [
        (msa_file, [None if j == i else other for j, other in enumerate(msa_files) if j >= i])
        for i, msa_file in enumerate(msa_files)
    ]


def _screen_many_vs_many(
    one_vs_many_fn: Callable,
    msa_files1: Iterable,
    msa_files2: Optional[Iterable] = None,
    runner: Optional[Callable] = None,
    model_kwargs: Optional[dict] = None,
    **kwargs,
) -> list:
    msa_files1 = list(msa_files1)
    msa_cache = {}  # parse each MSA once, not once per partner
    model = None
    if runner is not None:  # build the model once for the whole screen
        model = runner().make_model(cpu=kwargs.get("cpu", False), **(model_kwargs or {}))
    if msa_files2 is None:
        print_err("[WARN] No second set of MSAs provided; screening all unordered pairs (including homodimers).")
        jobs = _unordered_pairs(msa_files1)
        print_err(f"[INFO] Screening {len(msa_files1)} MSAs against each other ({sum(len(j[1]) for j in jobs)} pairs)...")
    else:
        msa_files2 = list(msa_files2)
        jobs = [(msa_file1, msa_files2) for msa_file1 in msa_files1]
        print_err(f"[INFO] Screening {len(msa_files1)} MSAs against {len(msa_files2)} MSAs...")
    results = []
    for msa_file1, partners in tqdm(jobs):
        results += one_vs_many_fn(
            msa_file1=msa_file1, 
            msa_file2=partners, 
            model=model,
            msa_cache=msa_cache,
            **kwargs,
        )
    return results


def rf2track_one_vs_many(
    msa_file1: Union[str, TextIOWrapper],
    msa_file2: Optional[Iterable[Union[str, TextIOWrapper]]] = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
    enforce_ref_match: bool = False,
    cpu: bool = False,
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
        runner=DCARunner,
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
        runner=AF2Runner,
        model_kwargs={"max_recycles": max_recycles, "param_dir": param_dir},
        max_gap_fraction=max_gap_fraction,
        interaction_map=interaction_map,
        enforce_ref_match=enforce_ref_match,
        seed=seed,
        **kwargs
    )
