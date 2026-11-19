"""Data structures for multiple sequence alignments."""

from collections.abc import Iterable, Mapping
from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field, fields, replace
from io import TextIOWrapper
from itertools import dropwhile
import os
import sys

from carabiner import cast, print_err
import numpy as np
from tqdm.auto import tqdm

from ..interaction_utils import organism_interactions, _name_normalizer
from .pairing import (
    InvalidControlError, PairingReport, UnpairableError, allowed_row_pair_capacity, make_rng,
    multiplicity, permute_partners, rewire_bipartite, sample_random_pairs, sample_random_rows,
    validate_pairing,
)

_A3M_ALPHABET = tuple("ARNDCQEGHILKMFPSTWYV-")
_A3M_ALPHABET_SIZE: int = len(_A3M_ALPHABET)
_A3M_ALPHABET_DICT = dict(zip(_A3M_ALPHABET, range(_A3M_ALPHABET_SIZE)))
_PAIRED_SPACER = ':::'
# Byte -> token lookup. Anything outside the alphabet (X, B, Z, U, O, ...) maps to gap.
_TOKEN_LUT = np.full(256, _A3M_ALPHABET_SIZE - 1, dtype=np.uint8)
for _letter, _idx in _A3M_ALPHABET_DICT.items():
    _TOKEN_LUT[ord(_letter)] = _idx


def tokenize(sequences: Iterable[str]) -> np.ndarray:
    """Convert equal-length sequences to a ``(n_seq, n_col)`` ``uint8`` token array.

    Examples
    ========
    >>> tokenize(["AR-", "XYZ"]).tolist()
    [[0, 1, 20], [20, 18, 20]]
    >>> tokenize([]).shape
    (0, 0)

    """
    sequences = list(sequences)
    if len(sequences) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    n_col = len(sequences[0])
    buffer = np.frombuffer("".join(sequences).encode("ascii", "replace"), dtype=np.uint8)
    return _TOKEN_LUT[buffer].reshape(len(sequences), n_col)

__BLOCK_GAPS__ = "__BLOCK_GAPS__"

@dataclass
class MSAName:
    """Parsed MSA sequence name (header line).

    Splits UniProt-style pipe-delimited names into database, unique ID,
    and entry name. Non-pipe-delimited headers get sentinel values.

    Examples
    ========
    >>> MSAName('>sp|P07807|DYR_YEAST').database
    'sp'
    >>> MSAName('>sp|P07807|DYR_YEAST').unique_id
    'P07807'
    >>> MSAName('>sp|P07807|DYR_YEAST').entry_name
    'DYR_YEAST'
    >>> MSAName('>UniRef90_A0A1B2').unique_id
    'A0A1B2'
    >>> MSAName('>UniRef90_A0A1B2').database
    'UniRef90'
    >>> MSAName('>MGYP000745883360').unique_id
    'MGYP000745883360'

    """
    name: str
    database: str = field(init=False)
    unique_id: str = field(init=False)
    entry_name: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.name, str):
            try:
                self._input_name = "".join(self.name)
            except TypeError:
                raise TypeError(f"MSA name `{self.name}` is type {type(self.name)}.")
        else:
            self._input_name = self.name
        self.name = self._input_name.removeprefix(">").rstrip()  # Strip out leading ">"
        db, _id, _name = None, None, None
        if "|" in self.name:
            parts = self.name.split("|", maxsplit=2)
            if len(parts) == 3:
                db, _id, _name = parts
        elif self.name.startswith("UniRef"):
            parts = self.name.split("_", maxsplit=1)
            db = parts[0]
            _id = parts[1] if len(parts) == 2 else None
        
        self.database = db or "__NO_DB_NAME__"
        self.unique_id = _id or self.name or "__NO_ENTRY_ID__"
        self.entry_name = _name or _id or self.name or "__NO_ENTRY_NAME__"

    def __str__(self) -> str:
        return self.name


@dataclass
class MSADescription:
    """Parsed MSA sequence description line.

    Extracts species identity from UniProt-style description fields.
    Prefers the NCBI taxon ID (``OX=``), falls back to ``TaxID=``,
    then to the species name (``OS=``), then to ``-1`` if none present.

    >>> MSADescription('Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1').species_id
    'NCBI:562'
    >>> MSADescription('Gene OS=Mycobacterium tuberculosis TaxID=1773 GN=y').species_id
    'NCBI:1773'
    >>> MSADescription('Gene OS=Borrelia burgdorferi GN=z').species_id
    'Name:Borrelia burgdorferi'
    >>> MSADescription('Gene GN=w PE=4 SV=1').species_id
    -1
    >>> MSADescription('Gene GN=w PE=4 SV=1').generic_species_name is None
    True
    >>> MSADescription('Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1').generic_species_name
    'Escherichia coli'

    """
    description: str
    species_id: str = field(init=False)
    prefix: str = field(init=False)
    info: Mapping[str, int | str] = field(init=False)
    _verbose: bool = False

    def __post_init__(self):
        desc_esc_eq = ":eq:".join(self.description.split(" = "))
        split_on_eq = [item.strip() for item in desc_esc_eq.split("=")]
        prefix_split_on_space = split_on_eq[0].split()
        if len(prefix_split_on_space) > 1:
            self.prefix = " ".join(prefix_split_on_space[:-1])
            split_on_eq[0] = prefix_split_on_space[-1]
        else:
            self.prefix = None
        info = {}
        for i in range(1, len(split_on_eq)):
            key = split_on_eq[i - 1].split()[-1].strip()
            val = ' '.join(split_on_eq[i].split()[:-1]).strip()
            if val.isdigit():
                val = int(val)
            info[key] = val
        self.info = info
        taxon_id = self.info.get("OX", self.info.get("TaxID")) or -1
        if taxon_id == "":
            taxon_id = -1
        if isinstance(taxon_id, str):
            if taxon_id.isdigit():
                taxon_id = int(taxon_id)
            else:
                taxon_id = -1
        if taxon_id > -1:  # NCBI identifier. Doesn't exist for everything
            species_id = f"NCBI:{taxon_id}"
        elif "OS" in self.info:  # UniProt species name fallback
            species_id = f"Name:{self.info['OS']}"
        else:
            species_id = -1
            if self.description != '__BLOCK_GAPS__' and self._verbose:
                print_err(f"[WARN] MSA has no species info. Description string: {self.description.rstrip()}")
        self.taxon_id = taxon_id
        self.species_id = species_id
        
        if species_id == -1:
            self.generic_species_name = None
        else:
            os_val = self.info.get('OS', '')
            if os_val:
                normed_name = _name_normalizer([os_val])
                try:
                    self.generic_species_name = normed_name[0]
                except IndexError:
                    self.generic_species_name = os_val
            else:
                self.generic_species_name = None
            
    def __str__(self) -> str:
        return self.description


@dataclass
class MSALine:
    """A single aligned sequence with its header metadata.

    Lowercase letters (insertions in A3M format) are stripped from the
    sequence. Gap fraction is computed over the uppercase-only sequence.

    Examples
    ========
    >>> line = MSALine(
    ...     name='>sp|P12345|GENE_ECOLI',
    ...     description='Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1',
    ...     sequence='MARNDagctagQE--W',
    ... )
    >>> line.sequence
    'MARNDQE--W'
    >>> line.gap_fraction
    0.2
    >>> len(line)
    10

    """
    name: str
    description: str
    sequence: str
    entry_name: str = field(init=False)
    gap_fraction: float = field(init=False)

    def __post_init__(self):
        self._input_sequence = self.sequence
        self.sequence = ''.join(letter for letter in self._input_sequence if not letter.islower())  # remove insertions(?)
        self.name = MSAName(self.name)
        self.unique_id = self.name.unique_id
        self.entry_name = self.name.entry_name
        self.description = MSADescription(self.description)
        self.gap_fraction = self.sequence.count('-') / float(len(self))

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        return f"MSALine(name='{self.name}', length={len(self)})"

    def __str__(self) -> str:
        return f">{str(self.name)} {str(self.description)}\n{self._input_sequence}"


class PairedMSALine(MSALine):

    def __post_init__(self):
        self._input_sequence = self.sequence
        if not _PAIRED_SPACER in self.name:
            raise ValueError(f"Paired MSA must contain '{_PAIRED_SPACER}' separator in name: {self.name}")
        self.name = tuple(MSAName(name) for name in self.name.split(_PAIRED_SPACER))
        self.unique_id = '-'.join(name.unique_id for name in self.name)
        self.entry_name = '-'.join(name.entry_name for name in self.name)
        self.description = tuple(MSADescription(desc) for desc in self.description.split(_PAIRED_SPACER))
        self.gap_fraction = self.sequence.count('-') / float(len(self))

    def __repr__(self) -> str:
        return "Paired " + super().__repr__()

    def __str__(self) -> str:
        return f">{_PAIRED_SPACER.join(map(str, self.name))} {_PAIRED_SPACER.join(map(str, self.description))}\n{self._input_sequence}"


@dataclass
class MSA:

    """MSA object which can be used for downstream analyses.
    """

    lines: Iterable[MSALine]
    name: str = field(init=False)
    sequence_labels: Iterable[str] = field(init=False)
    seq_length: int = field(init=False)
    sequence_token_ids: Iterable[int] = field(init=False)

    def __post_init__(self):
        self.lines = tuple(self.lines)
        self.name = self.lines[0].unique_id
        self.sequence_labels = tuple(line.unique_id for line in self.lines)
        seq_lengths = [len(line) for line in self.lines]
        seq_length = set(seq_lengths)
        if len(seq_length) > 1:
            raise AttributeError(f"Sequence lengths are not uniform! Found lengths: {', '.join(map(str, seq_length))}")
        if len(seq_length) == 0:
            raise AttributeError(f"No sequences! The lines are: {', '.join(map(str, self.lines))}")
        self.seq_length = seq_length.pop()
        self.sequence_token_ids = tokenize(line.sequence for line in self.lines)

    def sequences(self) -> list[str]:
        return [line.sequence for line in self.lines]

    def gap_fraction(self) -> list[float]:
        return [line.gap_fraction for line in self.lines]

    def truncate(self, n: int) -> 'MSA':
        return replace(self, lines=[
            MSALine(
                sequence=line.sequence[:n],
                description=str(line.description),
                name=str(line.name),
            ) 
            for line in self.lines
        ])

    @classmethod
    def from_file(cls, file: str | TextIOWrapper) -> 'MSA':
        from bioino import FastaCollection
        collection = list(FastaCollection.from_file(file).sequences)
        # print(collection[0])
        return cls(MSALine(**asdict(seq)) for seq in tqdm(collection))

    def __len__(self):
        return len(self.lines)

    def neff(self, identity_threshold: float = .62, verbose: bool = True) -> int:
        """Number of effective sequences: greedy clusters at a sequence-identity threshold.

        Two sequences are in the same cluster if they are identical at
        ``>= identity_threshold`` of alignment columns (gaps count as characters,
        as in the DCA reweighting scheme).

        Examples
        ========
        >>> m = MSA([MSALine(name=f">sp|P{i}|X", description="", sequence=s)
        ...          for i, s in enumerate(["AAAAAAAAAA", "AAAAAAAAAC", "CDEFGHIKLM"])])
        >>> m.neff(verbose=False)
        2
        >>> m.neff(identity_threshold=.95, verbose=False)
        3

        """
        arr = np.asarray(self.sequence_token_ids)
        n_rows = arr.shape[0]
        # Rows with identity >= threshold have <= L * (1 - threshold) mismatches.
        max_mismatches = self.seq_length * (1. - identity_threshold)
        remaining = np.arange(n_rows)
        n_effective = 0
        while remaining.size > 0:
            representative = arr[remaining[0]]
            mismatches = np.count_nonzero(arr[remaining] != representative, axis=1)
            remaining = remaining[mismatches > max_mismatches]
            n_effective += 1
        if verbose:
            print_err(
                f"[INFO] Neff = {n_effective} at identity threshold {identity_threshold} "
                f"(<= {max_mismatches:.1f} mismatches over {self.seq_length} positions)."
            )
        return n_effective

    def _filter_by_index(self, indices: Iterable[int], keep_query: bool = True) -> 'MSA':
        """Return a shallow copy containing only the rows in ``indices`` (order preserved).

        The first row is the query, which defines ``name`` and the pairing reference,
        so it is always retained unless ``keep_query=False``.

        """
        start_len = len(self)
        keep = sorted(set(indices) | ({0} if (keep_query and start_len > 0) else set()))
        new_copy = copy(self)
        new_copy.lines = tuple(self.lines[i] for i in keep)
        new_copy.sequence_labels = tuple(self.sequence_labels[i] for i in keep)
        new_copy.sequence_token_ids = np.asarray(self.sequence_token_ids)[keep]
        print_err(f"[INFO] Filtered out {start_len - len(new_copy)}/{start_len} lines from MSA.")
        return new_copy

    def filter_by_known_species(self) -> 'MSA':
        """Drop rows without species info (the query row is always kept)."""
        print_err("[INFO] Filtering MSA by known species.")
        indices_to_keep = (
            i for i, line in enumerate(self.lines) 
            if line.description.species_id != -1
        )
        return self._filter_by_index(indices_to_keep)

    def filter_by_gap_fraction(self, max_gap_fraction: float = 1.) -> 'MSA':
        if max_gap_fraction < 1.:
            print_err(f"[INFO] Filtering MSA by gap fraction < {max_gap_fraction}.")
            indices_to_keep = (
                i for i, line in enumerate(self.lines)
                if line.gap_fraction <= max_gap_fraction
            )
            return self._filter_by_index(indices_to_keep)
        else:
            return self

    def __str__(self) -> str:
        return f"MSA(name={self.name}) of sequence length {self.seq_length}, with {len(self)} sequences."

    def write(self, file=sys.stdout) -> None:
        for line in self.lines:
            print(line, file=file)
        return None


_SYMMETRIC_CACHE: list = []  # single slot: (content fingerprint, symmetrised map)


def _fingerprint(interaction_map: Mapping[str, Iterable[str]]) -> int:
    """Hash of the map's *contents* (not its identity), so in-place edits invalidate the cache."""
    return hash(tuple((k, tuple(v)) for k, v in interaction_map.items()))


def _symmetrise(interaction_map: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    """Make an interaction map undirected: ``b in out[a]`` iff ``b in map[a]`` or ``a in map[b]``.

    The last result is cached by a hash of the map's contents (the built-in map is large and
    shared across calls); mutating a map in place therefore never returns stale relationships.

    Examples
    ========
    >>> {k: sorted(v) for k, v in sorted(_symmetrise({"a": ["b"], "c": ["a"]}).items())}
    {'a': ['b', 'c'], 'b': ['a'], 'c': ['a']}
    >>> m = {"a": ["b"]}; _ = _symmetrise(m); m["a"] = ["z"]
    >>> sorted(_symmetrise(m)["z"])
    ['a']

    """
    key = _fingerprint(interaction_map)
    if _SYMMETRIC_CACHE and _SYMMETRIC_CACHE[0][0] == key:
        return _SYMMETRIC_CACHE[0][1]
    out = defaultdict(set)
    for k, values in interaction_map.items():
        for value in values:
            out[k].add(value)
            out[value].add(k)
    out = dict(out)
    _SYMMETRIC_CACHE[:] = [(key, out)]
    return out


def _allowed_species_pairs(
    species_msa1: Mapping[str, list],
    species_msa2: Mapping[str, list],
    interaction_map: Mapping[str, Iterable[str]],
) -> set[tuple[str, str]]:
    """Find species pairs (one from each MSA) that the interaction map allows to pair.

    A pair ``(s1, s2)`` is allowed if either species (or the generic species name of the
    group's first row) is listed as a partner of the other, in either direction.
    Cost is proportional to the number of map edges touched, not ``len(s1) * len(s2)``.

    Examples
    ========
    >>> class _D:  # minimal stand-in for MSADescription
    ...     def __init__(self, g): self.generic_species_name = g
    >>> class _L:
    ...     def __init__(self, g): self.description = _D(g)
    >>> g1 = {"NCBI:9606": [_L("Homo sapiens")]}
    >>> g2 = {"NCBI:82830": [_L("Epstein-Barr virus")], "NCBI:5": [_L("Other")]}
    >>> sorted(_allowed_species_pairs(g1, g2, {"NCBI:82830": ["NCBI:9606"]}))
    [('NCBI:9606', 'NCBI:82830')]

    """
    symmetric = _symmetrise(interaction_map)
    by_key2 = defaultdict(set)
    for sp2, lines2 in species_msa2.items():
        by_key2[sp2].add(sp2)
        name2 = lines2[0].description.generic_species_name if lines2 else None
        if name2 is not None:
            by_key2[name2].add(sp2)
    pairs = set()
    for sp1, lines1 in species_msa1.items():
        name1 = lines1[0].description.generic_species_name if lines1 else None
        partners = set(symmetric.get(sp1, ()))
        if name1 is not None:
            partners |= symmetric.get(name1, set())
        for partner in partners:
            for sp2 in by_key2.get(partner, ()):
                pairs.add((sp1, sp2))
    return pairs


@dataclass
class PairedMSA(MSA):
    """Paired MSA object which can be used for co-evolutionary analyses.
    """
    chain_a_length: int
    chain_b_length: int = field(init=False)
    pairing_info: "PairingReport | None" = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.chain_b_length = self.seq_length - self.chain_a_length

    @property
    def ids(self) -> tuple[str, str]:
        """Unique IDs of the two query sequences (unlike splitting ``name`` on ``-``,
        this is safe for IDs that contain hyphens, e.g. UniProt isoforms ``P12345-2``).

        >>> line = PairedMSALine(name=">sp|P1-2|A:::>sp|Q9|B", description=":::", sequence="AC")
        >>> PairedMSA(lines=[line], chain_a_length=1).ids
        ('P1-2', 'Q9')

        """
        first = self.lines[0].name
        return first[0].unique_id, first[1].unique_id

    def split(
        self
    ):
        msa1 = MSA([
            MSALine(
                sequence=line.sequence[:self.chain_a_length],
                description=str(line.description[0]),
                name=str(line.name[0]),
            ) 
            for line in self.lines
        ])
        msa2 = MSA([
            MSALine(
                sequence=line.sequence[self.chain_a_length:],
                description=str(line.description[1]),
                name=str(line.name[1]),
            )
            for line in self.lines
        ])
        return msa1, msa2

    @staticmethod
    def _check_ref_match(
        msa1: MSA,
        msa2: MSA,
        interaction_map: Mapping[str, Iterable[str]] | None = None,
        name_attr: str = "species_id"
    ) -> None:
        """Validate that the reference sequences (first lines) of two MSAs
        are from organisms that can interact.

        Passes silently on a valid pair.
        
        Raises
        ======
        ``UnpairableError`` (an ``AttributeError`` and a ``ValueError``)
            If neither organism is in the other's allowed set, or a species ID is unknown (``-1``).

        ``interaction_map`` is checked in both directions: the pair is valid
        if ``id2`` is in ``map[id1]`` OR ``id1`` is in ``map[id2]``.

        """
        id1, id2 = (
            getattr(msa.lines[0].description, name_attr) for msa in (msa1, msa2)
        )
        if id1 == -1 or id2 == -1:
            raise UnpairableError(f"At least one MSA reference species is unknown: {id1}, {id2}")
        if interaction_map is not None:
            allowed_id2 = interaction_map.get(id1, [])
            allowed_id1 = interaction_map.get(id2, [])
        
            if not id2 in allowed_id2 and not id1 in allowed_id1:
                raise UnpairableError(
                    f"""
                    MSA reference species do not match: 
                    
                    ID1: {id1}; allowed: {allowed_id1}
                    ID2: {id2}; allowed: {allowed_id2}

                    """
                )
            if id1 == -1 or all(_id2 == -1 for _id2 in allowed_id2):
                raise UnpairableError(f"At least one MSA reference species is unknown: {id1}, {id2}")
        else:
            if id1 != id2:
                raise UnpairableError(
                    f"""
                    MSA reference species do not match, and no interaction map provided: 
                    
                    ID1: {id1}; ID2: {id2}

                    """
                )
        return None

    @staticmethod
    def join_msa(*args, **kwargs) -> tuple[list[PairedMSALine], int]:
        """Pair two MSAs; returns ``(lines, chain_a_length)``. See :meth:`_join_msa` for the arguments."""
        lines, seq_length, _ = PairedMSA._join_msa(*args, **kwargs)
        return lines, seq_length

    @staticmethod
    def _join_msa(
        msa1: MSA, 
        msa2: MSA | None = None, 
        blocked: bool = False,
        interaction_map: str | Mapping[str, Iterable[str]] | None = None,
        strict_species_match: bool = False,
        enforce_ref_match: bool = False,
        name_attr: str = "species_id",
        pairing: str = "map",
        seed: int | None = None,
        strict_controls: bool = True,
        max_gap_fraction: float = 1.,
    ) -> tuple[list[PairedMSALine], int, PairingReport]:
        """...

        ``max_gap_fraction`` is used **only** by ``permute-partners``: the real pairs are gap-filtered
        first and the survivors are then permuted, so the control has exactly the real final A and B
        rows (the permuted rows must not be filtered or subsampled again).
        """
        validate_pairing(pairing)
        if strict_species_match or interaction_map is None:
            fallback_name_attr = name_attr
        else:
            fallback_name_attr = "generic_species_name"
        if msa2 is None:
            msa2 = msa1
        msa1_known, msa2_known = (msa.filter_by_known_species() for msa in (msa1, msa2))

        if interaction_map is None:
            strict_intraspecies = True
            all_species = [
                getattr(line.description, name_attr)
                for msa in (msa1_known, msa2_known) 
                for line in msa.lines
            ]
            interaction_map = {
                _species: set([_species]) for _species in all_species
            }
        elif interaction_map == "builtin":
            strict_intraspecies = False
            interaction_map = organism_interactions()
        elif isinstance(interaction_map, Mapping):
            strict_intraspecies = False
            interaction_map = {
                key: set(cast(val, to=list) + [key]) 
                for key, val in interaction_map.items()
            }
        else: 
            raise ValueError(
                f"""
                If provided, interaction_map must be a `dict` or `Mapping`,
                but was `{type(interaction_map)}`: {interaction_map}.
                """
            )

        if enforce_ref_match or strict_intraspecies:
            try:
                PairedMSA._check_ref_match(
                    msa1=msa1_known, 
                    msa2=msa2_known, 
                    interaction_map=interaction_map, 
                    name_attr=name_attr,
                )
            except AttributeError as e:
                if fallback_name_attr != name_attr:
                    PairedMSA._check_ref_match(
                        msa1=msa1_known, 
                        msa2=msa2_known, 
                        interaction_map=interaction_map, 
                        name_attr=fallback_name_attr,
                    )
                    query_name_attr = fallback_name_attr
                else:
                    raise e
            else:
                query_name_attr = name_attr
        else:
            print_err(
                "[WARN] HPI map constraint relaxed for query sequences. "
                "Aligned sequence pairing still enforced."
            )
            query_name_attr = name_attr
        
        #Get the matches
        def _get_query(m, name_attr, fallback_name_attr):
            name = getattr(m.lines[0].description, name_attr)
            if name in interaction_map:
                return name
            else:
                return getattr(m.lines[0].description, fallback_name_attr)

        query_pair = (
            _get_query(msa1_known, name_attr, fallback_name_attr),
            _get_query(msa2_known, name_attr, fallback_name_attr),
        )

        def _species_key(line) -> str | None:
            """Group key for a row: its species ID if the map knows it, else the generic name."""
            name = getattr(line.description, name_attr)
            if name in interaction_map:
                return name
            return getattr(line.description, fallback_name_attr)

        def _group_by_species(m) -> dict:
            d = defaultdict(list)
            for line in m.lines[1:]:  # row 0 is the query, handled separately
                key = _species_key(line)
                if key is not None:
                    d[key].append(line)
            return d

        species_msa1, species_msa2 = _group_by_species(msa1_known), _group_by_species(msa2_known)
        species_pairs = _allowed_species_pairs(species_msa1, species_msa2, interaction_map)
        query_pair_allowed = query_pair in species_pairs
        species_pairs.discard(query_pair)
        species_pairs = (
            ([query_pair] if query_pair_allowed else []) 
            + sorted(species_pairs, key=lambda p: tuple(s or "" for s in p))
        )
        # Always add reference pair first — map check already done (or deliberately relaxed)
        real_pairs = {p for p in species_pairs if p != query_pair}
        row_pairs = None  # explicit (row in A, row in B) pairs for the taxonomy-blind control
        report = PairingReport("map", None, target_pairs=len(real_pairs), n_pairs=len(real_pairs), retained_known=None)
        if pairing != "map":
            species_pairs, row_pairs, report = PairedMSA._control_pairs(
                pairing, species_pairs, query_pair, query_pair_allowed,
                species_msa1, species_msa2,
                [_species_key(l) for l in msa1_known.lines], [_species_key(l) for l in msa2_known.lines],
                seed,
            )
            if not report.valid:
                message = f"negative control '{pairing}' violates its invariants: {'; '.join(report.notes)}"
                if strict_controls:
                    raise InvalidControlError(message)
                print_err(f"[WARN] {message}. Flagged pairing_valid=False.")
        _ref1, _ref2 = msa1_known.lines[0], msa2_known.lines[0]
        msa_lines = [
            PairedMSALine(
                name=_PAIRED_SPACER.join([str(_ref1.name), str(_ref2.name)]),
                description=_PAIRED_SPACER.join([str(_ref1.description), str(_ref2.description)]),
                sequence="".join([_ref1.sequence, _ref2.sequence]),
            )
        ]
        matched_species = {query_pair}

        selected = []  # (species A, species B, row A, row B) in the real pairing's row order
        for _sp1, _sp2 in ([] if row_pairs is not None else species_pairs):
            _lines1, _lines2 = species_msa1.get(_sp1), species_msa2.get(_sp2)
            if _lines1 and _lines2:  # the query pair is first, and is not required to have homologs
                line1 = min(_lines1, key=lambda l: l.gap_fraction)
                line2 = min(_lines2, key=lambda l: l.gap_fraction)
                selected.append((_sp1, _sp2, line1, line2))
                matched_species |= set([_sp1, _sp2])
        if pairing == "permute-partners":
            # Keep the real rows, permute the B partners among the non-query pairs.
            if max_gap_fraction < 1.:  # filter the REAL rows, then permute exactly the survivors
                selected = [
                    t for t in selected
                    if (t[2].sequence.count("-") + t[3].sequence.count("-")) / (len(t[2].sequence) + len(t[3].sequence))
                    <= max_gap_fraction
                ]
            body = [t for t in selected if (t[0], t[1]) != query_pair or not query_pair_allowed]
            head_rows = selected[: len(selected) - len(body)]
            perm_rng, _ = make_rng(report.seed, str(query_pair[0]), str(query_pair[1]))
            perm, retained = permute_partners(
                [t[0] for t in body], [t[1] for t in body], set(species_pairs) | {query_pair}, perm_rng,
            )
            body = [(t[0], body[perm[i]][1], t[2], body[perm[i]][3]) for i, t in enumerate(body)]
            selected = head_rows + body
            report.n_pairs, report.target_pairs = len(body), len(body)
            report.retained_known, report.multiplicity_ok = retained, True
            report.finalise()
            if not report.valid:
                message = f"negative control '{pairing}' violates its invariants: {'; '.join(report.notes)}"
                if strict_controls:
                    raise InvalidControlError(message)
                print_err(f"[WARN] {message}. Flagged pairing_valid=False.")
        for _sp1, _sp2, line1, line2 in selected:
            msa_lines.append(
                PairedMSALine(
                    name=_PAIRED_SPACER.join([str(line1.name), str(line2.name)]),
                    description=_PAIRED_SPACER.join([str(line1.description), str(line2.description)]),
                    sequence="".join([line1.sequence, line2.sequence]),
                )
            )

        for i, j in (row_pairs or []):
            line1, line2 = msa1_known.lines[i], msa2_known.lines[j]
            msa_lines.append(
                PairedMSALine(
                    name=_PAIRED_SPACER.join([str(line1.name), str(line2.name)]),
                    description=_PAIRED_SPACER.join([str(line1.description), str(line2.description)]),
                    sequence="".join([line1.sequence, line2.sequence]),
                )
            )

        if blocked:  # make blocked MSA for the individual proteins not belonging to a species pair
            idx1, idx2 = (
                [
                    i for i, line in enumerate(lines) 
                    if not any(
                        getattr(line.description, _attr) in matched_species 
                        for _attr in (name_attr, fallback_name_attr)
                    )
                ] for lines in (msa1.lines, msa2.lines)
            )
            msa1, msa2 = (  # the query rows are already in the paired block
                msa._filter_by_index(idx, keep_query=False)
                for idx, msa in zip((idx1, idx2), (msa1, msa2))
            )
            msa_lines += PairedMSA.__make_blocked(msa1, msa2)
        
        return msa_lines, msa1.seq_length, report

    @staticmethod
    def _control_pairs(
        pairing, species_pairs, query_pair, query_pair_allowed,
        species_msa1, species_msa2, row_keys1, row_keys2, seed,
    ):
        """Replace the map's species pairs by a negative-control pairing (see ``yunta.structs.pairing``).

        Returns ``(species_pairs, row_pairs, report)``; ``row_pairs`` is None except for
        ``"random-rows"``. The query pair stays first (row 0 is the real query pair). Invariants
        (depth, no known pairs, and for ``"rewired"`` multiplicity) are measured, not assumed.
        """
        rng, used_seed = make_rng(seed, str(query_pair[0]), str(query_pair[1]))
        real = {p for p in species_pairs if p != query_pair}
        known = real | ({query_pair} if query_pair_allowed else set())
        head = [query_pair] if query_pair_allowed else []
        order = lambda ps: sorted(ps, key=lambda p: tuple(s or "" for s in p))
        report = PairingReport(pairing, used_seed, target_pairs=len(real))
        infeasible = None
        if pairing == "rewired":
            new, _ = rewire_bipartite(real, rng, extra_forbidden={query_pair})
            report.multiplicity_ok = multiplicity(new) == multiplicity(real)
            species_out, row_out, new_pairs = head + order(new), None, new
        elif pairing == "random-species":
            new = sample_random_pairs(
                list(species_msa1), list(species_msa2), len(real), rng, forbidden=known | {query_pair},
            )
            species_out, row_out, new_pairs = head + order(new), None, new
        elif pairing == "permute-partners":
            # Species pairs and row selection are the real ones; the permutation of B rows
            # among them is applied by the caller once the rows are chosen.
            species_out, row_out, new_pairs = head + order(real), None, set()
            report.n_pairs = len(real)
        else:  # random-rows
            kk = known | {query_pair}
            allowed = lambda i, j: (row_keys1[i + 1], row_keys2[j + 1]) in kk
            k1, k2 = list(row_keys1[1:]), list(row_keys2[1:])

            def allowed_matrix():
                u1 = {k: n for n, k in enumerate(dict.fromkeys(k1))}
                u2 = {k: n for n, k in enumerate(dict.fromkeys(k2))}
                forb = np.zeros((len(u1), len(u2)), dtype=bool)
                for a, b in kk:
                    if a in u1 and b in u2:
                        forb[u1[a], u2[b]] = True
                return ~forb[np.array([u1[k] for k in k1], dtype=int)][:, np.array([u2[k] for k in k2], dtype=int)]

            # Species-level controls keep the real query-species row (``head``); a row-level
            # control has no head, so it draws that row too to reach the same depth.
            target_rows = len(real) + len(head)
            rows = sample_random_rows(
                len(k1), len(k2), target_rows, rng, forbidden=allowed, allowed_matrix=allowed_matrix,
            )
            species_out, row_out = [], [(i + 1, j + 1) for i, j in rows]  # +1: row 0 is the query
            new_pairs = {(row_keys1[i], row_keys2[j]) for i, j in row_out}
            report.n_pairs = len(row_out) - len(head)  # same convention as the map: query-species row excluded
            capacity = allowed_row_pair_capacity(k1, k2, kk)
            if capacity < target_rows:
                infeasible = f"infeasible: only {capacity} allowed row pairs exist for target depth {target_rows}"
        if pairing not in ("random-rows", "permute-partners"):
            report.n_pairs = len(new_pairs)
        if pairing == "permute-partners":
            report.retained_known = 0  # measured after the permutation, by the caller
        elif pairing == "random-rows":
            report.retained_known = sum((row_keys1[i], row_keys2[j]) in known for i, j in row_out)
        else:
            report.retained_known = len(set(new_pairs) & known)
        report.finalise()
        if infeasible:  # distinguishes a truly impossible control from an exhausted sampling budget
            report.notes.append(infeasible)
            report._pre_notes = list(report.notes)
            report.valid = False
        return species_out, row_out, report

    @staticmethod
    def __make_blocked(
        msa1: MSA, 
        msa2: MSA, 
        gap_char: str = '-'
    ) -> list[PairedMSALine]:
        gaps1, gaps2 = (gap_char * msa.seq_length for msa in (msa1, msa2))
        # The msas must be str representations of the blocked+paired MSAs here
        block1 = [
            PairedMSALine(
                name=f"{line.name}{_PAIRED_SPACER}xx|{__BLOCK_GAPS__}|{__BLOCK_GAPS__}", 
                description=f"{line.description}{_PAIRED_SPACER}{__BLOCK_GAPS__}", 
                sequence="".join([line.sequence, gaps2])
            ) for line in msa1.lines
        ]
        block2 = [
            PairedMSALine(
                name=f"xx|{__BLOCK_GAPS__}|{__BLOCK_GAPS__}{_PAIRED_SPACER}{line.name}", 
                description=f"{__BLOCK_GAPS__}{_PAIRED_SPACER}{line.description}", 
                sequence="".join([gaps1, line.sequence])
            ) for line in msa2.lines
        ]
        return block1 + block2

    @classmethod
    def from_msa(
        cls, 
        msa1: MSA, 
        msa2: MSA | None = None,
        blocked: bool = False,
        interaction_map: str | Mapping[str, Iterable[str]] | None = None,
        strict_species_match: bool = False,
        enforce_ref_match: bool = False,
        **kwargs
    ) -> 'PairedMSA':
        msa_lines, chain_a_length, report = cls._join_msa(
            msa1, 
            msa2, 
            blocked=blocked, 
            interaction_map=interaction_map, 
            strict_species_match=strict_species_match,
            enforce_ref_match=enforce_ref_match,
            **kwargs
        )
        if len(msa_lines) > 0:
            return cls(lines=msa_lines, chain_a_length=chain_a_length, pairing_info=report)
        else:
            raise UnpairableError(f"Pairing {msa1} and {msa2} resulted in no pairs.")

    @classmethod
    def read(cls, file: str | TextIOWrapper, chain_a_length: int) -> 'PairedMSA':
        """Read a paired A3M written by :meth:`MSA.write` (e.g. by ``yunta pair``), without re-pairing.

        ``chain_a_length`` is the width of the first chain (stored in the ``yunta pair``
        sidecar JSON); the file itself has no chain boundary.

        >>> import io
        >>> line = PairedMSALine(name=">sp|P1|P1_X:::>sp|Q2|Q2_X", description="a:::b", sequence="ACDE")
        >>> buf = io.StringIO(); PairedMSA(lines=[line], chain_a_length=3).write(buf)
        >>> back = PairedMSA.read(io.StringIO(buf.getvalue()), 3)
        >>> back.ids, back.chain_a_length, back.chain_b_length, back.sequences()
        (('P1', 'Q2'), 3, 1, ['ACDE'])
        """
        opened = isinstance(file, (str, os.PathLike))
        handle = open(file) if opened else file
        try:
            text = handle.read()
        finally:
            if opened:
                handle.close()
        lines, header, seq = [], None, []
        for raw in text.splitlines() + [">"]:
            if raw.startswith(">"):
                if header is not None:
                    name, _, description = header.partition(" ")
                    lines.append(PairedMSALine(name=name, description=description, sequence="".join(seq)))
                header, seq = raw[1:], []
            elif raw.strip():
                seq.append(raw.strip())
        if not lines:
            raise UnpairableError(f"No sequences in paired MSA {file}.")
        return cls(lines=lines, chain_a_length=chain_a_length)

    @classmethod
    def from_file(
        cls, 
        file1: str | TextIOWrapper,
        file2: str | TextIOWrapper | None = None,
        blocked: bool = False,
        **kwargs
    ) -> 'PairedMSA':
        """Read A3M file(s).

        """
        msa1 = MSA.from_file(file1)
        if file2 is None:
            msa2 = msa1
        else:
            msa2 = MSA.from_file(file2)
        return cls.from_msa(msa1, msa2, blocked=blocked, **kwargs)

    def __str__(self) -> str:
        return "Paired " + super().__str__()