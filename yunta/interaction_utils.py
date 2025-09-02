"""Utilities for interactions across organisms."""

from typing import Dict, Iterable, List

from collections import defaultdict
from copy import deepcopy
from itertools import product
import json
import gzip
import os

from carabiner import cast, print_err
import pandas as pd
from tqdm.auto import tqdm

_INTERACTION_FILE_LOADED: bool = False
ORGANISM_INTERACTIONS: dict = {}
TEST_MODE: str = str(os.environ.get("YUNTA_TEST", "0"))
CACHE_PATH: str = str(os.environ.get(
    "YUNTA_CACHE", 
    os.path.join(
        os.path.realpath(os.path.expanduser("~")),
        ".cache",
        "yunta",
    ),
))
USE_CACHE: str = str(os.environ.get("YUNTA_USE_CACHE", "False"))

_data_root = os.path.join(
    os.path.dirname(__file__), 
    "data",
)
_data_csv_path = os.path.join(
    _data_root,
    "20250409_hpi.csv",
)


def _name_normalizer(x: Iterable[str]):
    x = [str(name).split("subsp.")[0].split("sp.")[0].split("(")[0].strip("'").strip().casefold() for name in x]
    x = [" ".join(name.split(" ")[:2]) if (not "virus" in name and not "phage" in name) else name for name in x]
    return [f"{name[0].upper()}{name[1:]}" for name in x]

def _create_data_json(
    csv_path: str,
    json_path: str,
    name2ncbi_path: str,
    ncbi_columns: Iterable[str] = ("pathogen_taxon_id", "host_taxon_id"),
    name_cols: Iterable[str] = ("pathogen_name", "host_name"),
    prefix: str = "NCBI:",
    test_mode: bool = False
) -> None:
    col1, col2 = ncbi_columns
    name_col1, name_col2 = name_cols

    df = pd.read_csv(csv_path)[[col1, col2, name_col1, name_col2]].drop_duplicates().dropna()
    df = df.assign(**{
        col: _name_normalizer(df[col]) for col in name_cols
    }).drop_duplicates()

    interaction_map = {
        "ncbi": defaultdict(set),
        "name": defaultdict(set),
    }
    columns = {
        "ncbi": ncbi_columns,
        "name": name_cols,
    }

    for key, grouping_cols in columns.items():
        for grouping_col in grouping_cols:
            if key == "ncbi":
                this_prefix, typer = prefix, int
            else:
                this_prefix, typer = "", str
            for organism, organism_df in tqdm(df.groupby(grouping_col)):
                org_key = f"{this_prefix}{typer(organism)}"
                for value_col in grouping_cols:
                    if value_col != grouping_col:
                        values = set(f"{this_prefix}{typer(v)}" for v in organism_df[value_col].unique())
                        interaction_map[key][org_key] |= values
    
    name_to_ncbi = defaultdict(set)
    for row in df.itertuples(index=False):
        for name_col, ncbi_col in zip(name_cols, ncbi_columns):
            name_to_ncbi[getattr(row, name_col)].add(f"{prefix}{int(getattr(row, ncbi_col))}")
    # make sure NCBI Taxon IDs aren't overly specific
    # interaction_map_expanded = deepcopy(interaction_map)
    if not test_mode:
        with open(name2ncbi_path, "w") as f:
            json.dump( 
                {key: sorted(val) for key, val in name_to_ncbi.items()}, 
                f, 
                sort_keys=True, 
                indent=4,
            )
    additional = defaultdict(set)
    for key, value in tqdm(interaction_map["name"].items()):
        vals_to_add = set()
        for v in value:
            try:
                ncbi_keys = name_to_ncbi[v]
            except KeyError:
                pass
            else:
                vals_to_add |= ncbi_keys
        interaction_map["name"][key] |= vals_to_add
        try:
            ncbi_keys = name_to_ncbi[key]
        except KeyError:
            pass
        else:
            for ncbi_key in ncbi_keys:
                additional[ncbi_key] = interaction_map["name"][key]
    interaction_map["name"].update(additional)
    if not test_mode:
        interaction_map = {key: sorted(val) for key, val in interaction_map["name"].items()}
    if test_mode:
        print_err("Loading interaction map directly into memory")
        return interaction_map["name"]
    with gzip.open(json_path, mode='wt', encoding='UTF-8') as f:
        json.dump(interaction_map, f, sort_keys=True, indent=4)
    return None


def organism_interactions(
    cache: str = CACHE_PATH, 
    use_cache: bool = False
) -> Dict[str, List[str]]:

    use_cache = use_cache or (USE_CACHE == "True")
    test_mode = (TEST_MODE == "1") or not use_cache
    _data_json_path, _name2ncbi_path = (
        os.path.join(cache, filename) 
        for filename in ("interactions.json.gz", "name-to-ncbi.json")
    )

    if test_mode and len(ORGANISM_INTERACTIONS) == 0:
        print_err("Building organism interaction lookup table and loading into memory...", flush=True)
        ORGANISM_INTERACTIONS.update(
            _create_data_json(
                _data_csv_path, 
                _data_json_path, 
                _name2ncbi_path, 
                test_mode=test_mode,
            )
        )
    else:
        if not os.path.exists(CACHE_PATH):
            try:
                os.makedirs(CACHE_PATH)
            except OSError:
                if len(ORGANISM_INTERACTIONS) == 0:
                    print_err("File system not writable; building organism interaction lookup table and loading into memory...", flush=True)
                    ORGANISM_INTERACTIONS.update(
                        _create_data_json(
                            _data_csv_path, 
                            _data_json_path, 
                            _name2ncbi_path, 
                            test_mode=test_mode,
                        )
                    )
                return ORGANISM_INTERACTIONS
        if not os.path.exists(_data_json_path):
            print_err("Organism interaction lookup table not yet built; building...", flush=True)
            _create_data_json(_data_csv_path, _data_json_path, _name2ncbi_path)
            print_err("Done!")
        global _INTERACTION_FILE_LOADED
        if not _INTERACTION_FILE_LOADED:
            with gzip.open(_data_json_path, "rt", encoding='UTF-8') as f:
                ORGANISM_INTERACTIONS.update(json.load(f))
            _INTERACTION_FILE_LOADED = True

    return ORGANISM_INTERACTIONS