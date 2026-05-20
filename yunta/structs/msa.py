"""Data structures for multiple sequence alignments."""

from typing import Iterable, List, Mapping, Tuple, Optional, Union
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from io import TextIOWrapper
from itertools import dropwhile, product
import sys

from carabiner import cast, print_err
from tqdm.auto import tqdm

from ..interaction_utils import organism_interactions, _name_normalizer

_A3M_ALPHABET = tuple("ARNDCQEGHILKMFPSTWYV-")
_A3M_ALPHABET_SIZE: int = len(_A3M_ALPHABET)
_A3M_ALPHABET_DICT = dict(zip(_A3M_ALPHABET, range(_A3M_ALPHABET_SIZE)))
_PAIRED_SPACER = ':::'
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
        if "|" in self.name:
            parts = self.name.split("|", maxsplit=2)
            if len(parts) == 3:
                self.database, self.unique_id, self.entry_name = parts
            else:
                # Preserve useful identifier rather than sentinel
                self.database = "__NO_NAME__"
                self.unique_id = self.name if self.name else "__NO_ENTRY_ID__"
                self.entry_name = "__NO_ENTRY_NAME__"
        elif self.name.startswith("UniRef"):
            parts = self.name.split("_", maxsplit=1)
            if len(parts) == 2:
                self.database, self.unique_id = parts
            else:
                # Preserve useful identifier rather than sentinel
                self.database = "__NO_NAME__"
                self.unique_id = self.name if self.name else "__NO_ENTRY_ID__"
                self.entry_name = "__NO_ENTRY_NAME__"
        else:
            self.database = "__NO_NAME__"
            self.unique_id = self.name if self.name else "__NO_ENTRY_ID__"
            self.entry_name = "__NO_ENTRY_NAME__"

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
    info: Mapping[str, Union[int, str]] = field(init=False)
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
        taxon_id = self.info.get("OX", self.info.get("TaxID"))
        if taxon_id is not None and taxon_id.isdigit():  # NCBI identifier. Doesn't exist for everything
            self.taxon_id = int(taxon_id)
            species_id = f"NCBI:{self.taxon_id}"
        else:
            self.taxon_id = -1
        elif "OS" in self.info:  # UniProt species name fallback
            species_id = f"Name:{self.info['OS']}"
        else:
            species_id = -1
            if self.description != '__BLOCK_GAPS__' and self._verbose:
                print_err(f"[WARN] MSA has no species info. Description string: {self.description.rstrip()}")
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
        self.sequence_token_ids = [
            [_A3M_ALPHABET_DICT.get(letter, len(_A3M_ALPHABET_DICT) - 1) 
            for letter in line.sequence]
            for line in self.lines
        ]

    def sequences(self) -> List[str]:
        return [line.sequence for line in self.lines]

    def gap_fraction(self) -> List[float]:
        return [line.gap_fraction for line in self.lines]

    @classmethod
    def from_file(cls, file: Union[str, TextIOWrapper]) -> 'MSA':
        from bioino import FastaCollection
        collection = list(FastaCollection.from_file(file).sequences)
        # print(collection[0])
        return cls(MSALine(**asdict(seq)) for seq in tqdm(collection))

    def __len__(self):
        return len(self.lines)

    def neff(self, identity_threshold: float = .62) -> int:
        """Calculate the number of effective sequences.
        """
        import numpy as np
        remaining_msa = deepcopy(self.sequence_token_ids)
        threshold = float(self.seq_length * identity_threshold)
        n_effective = 0
        print_err(
            f"[INFO] Clustering at identity threshold: {identity_threshold} ",
            f"({threshold}/{self.seq_length} positions): "
        )
        while len(remaining_msa) > 0:
            print_err(
                f"\r:: Neff = {n_effective} | remaining to cluster: {len(remaining_msa)}", 
                end='',
            )
            arr = np.array(remaining_msa, dtype=np.int8)
            diffs = np.sum(arr != arr[0], axis=1)
            remaining_msa = arr[diffs > threshold].tolist()
            n_effective += 1
        print_err(f"\n[INFO] Neff = {n_effective}")
        return n_effective

    def _filter_by_index(self, indices=Iterable[int]) -> 'MSA':
        start_len = len(self)
        indices = set(indices)
        new_copy = deepcopy(self)
        for _field in fields(self):
            if _field.name not in ('name', 'seq_length'):
                original_items = getattr(self, _field.name) 
                if not isinstance(original_items, int):  # seq length
                    setattr(
                        new_copy, 
                        _field.name, 
                        [
                            item for i, item in enumerate(original_items) 
                            if i in indices
                        ],
                    )
        final_len = len(new_copy)
        print_err(f"[INFO] Filtered out {start_len - final_len}/{start_len} lines from MSA.")
        return new_copy

    def filter_by_known_species(self) -> 'MSA':
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


class PairedMSA(MSA):

    """Paired MSA object which can be used for co-evolutionary analyses.
    """

    def __init__(self, 
                 chain_a_length: int, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_a_length = chain_a_length
        self.chain_b_length = self.seq_length - self.chain_a_length

    def split(
        self
    ):
        msa1 = MSA([
            replace(
                line, 
                sequence=line.sequence[self.chain_a_length:],
                description=line.description.split(_PAIRED_SPACER)[0],
                name=line.name.split(_PAIRED_SPACER)[0],
            ) 
            for line in self.lines
        ])
        msa2 = MSA([
            replace(
                line, 
                sequence=line.sequence[:self.chain_a_length],
                description=line.description.split(_PAIRED_SPACER)[1],
                name=line.name.split(_PAIRED_SPACER)[1],
            )
            for line in self.lines
        ])
        return msa1, msa2

    @staticmethod
    def _check_ref_match(
        msa1: MSA,
        msa2: MSA,
        interaction_map: Optional[Mapping[str, Iterable[str]]] = None,
        name_attr: str = "species_id"
    ) -> None:
        """Validate that the reference sequences (first lines) of two MSAs
        are from organisms that can interact.

        Passes silently on a valid pair.
        
        Raises
        ======
        ``AttributeError``
            If neither organism is in the other's allowed set
        ``ValueError`` if a species ID is unknown (``-1``).

        ``interaction_map`` is checked in both directions: the pair is valid
        if ``id2`` is in ``map[id1]`` OR ``id1`` is in ``map[id2]``.

        """
        id1, id2 = (
            getattr(msa.lines[0].description, name_attr) for msa in (msa1, msa2)
        )
        if id1 == -1 or id2 == -1:
            raise ValueError(f"At least one MSA reference species is unknown: {id1}, {id2}")
        if interaction_map is not None:
            allowed_id2 = interaction_map.get(id1, [])
            allowed_id1 = interaction_map.get(id2, [])
        
            if not id2 in allowed_id2 and not id1 in allowed_id1:
                raise AttributeError(
                    f"""
                    MSA reference species do not match: 
                    
                    ID1: {id1}; allowed: {allowed_id1}
                    ID2: {id2}; allowed: {allowed_id2}

                    """
                )
            if id1 == -1 or all(_id2 == -1 for _id2 in allowed_id2):
                raise ValueError(f"At least one MSA reference species is unknown: {id1}, {id2}")
        else:
            if id1 != id2:
                raise AttributeError(
                    f"""
                    MSA reference species do not match, and no interaction map provided: 
                    
                    ID1: {id1}; ID2: {id2}

                    """
                )
        return None

    @staticmethod
    def join_msa(
        msa1: MSA, 
        msa2: Optional[MSA] = None, 
        blocked: bool = False,
        interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
        strict_species_match: bool = False,
        enforce_ref_match: bool = False,
        name_attr: str = "species_id"
    ) -> Tuple[List[PairedMSALine], int]:
        if strict_species_match or interaction_map is None:
            fallback_name_attr = name_attr
        else:
            fallback_name_attr = "generic_species_name"
        if msa2 is None:
            msa2 = deepcopy(msa1)
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
        query_pair = tuple(
            getattr(msa.lines[0].description, name_attr)
            if getattr(msa.lines[0].description, name_attr) in interaction_map
            else getattr(msa.lines[0].description, fallback_name_attr)
            for msa in (msa1_known, msa2_known)
        )

        species_msa1, species_msa2  = (
            {
                _species: [
                    line for line in msa.lines 
                    if _species in set(
                        getattr(line.description, _attr) 
                        for _attr in (name_attr, fallback_name_attr)
                    )
                ] for _species in set(
                    getattr(line.description, name_attr)
                    if getattr(line.description, name_attr) in interaction_map
                    else getattr(line.description, fallback_name_attr)
                    for line in msa.lines
                )
            } for msa in (msa1_known, msa2_known)
        )
        species_pairs = set(product(species_msa1, species_msa2))
        try:
            species_pairs.remove(query_pair)
        except KeyError:
            sep = '\n\t- '
            print_err(
                f"""
                [ERROR] Query species {query_pair} is not among the shared species in the MSAs:" 
                    - {sep.join(map(str, sorted(species_pairs)))}
                """
            )
            raise KeyError(f"Query species {':'.join(query_pair)} is not among the shared species in the MSAs")
        species_pairs = [query_pair] + sorted(species_pairs)
        msa_lines, matched_species = [], set()
        for _sp1, _sp2 in species_pairs:
            _lines1, _lines2 = species_msa1[_sp1], species_msa2[_sp2]
            # Name-level fallback keys for cross-strain matching
            # (e.g. NCBI:10710 → "Enterobacteria phage lambda" matches "Escherichia coli")
            if _lines1:
                _sp1_name = _lines1[0].description.generic_species_name
            else:
                _sp1_name = _sp1
            if _lines2:
                _sp2_name = _lines2[0].description.generic_species_name
            else:
                _sp2_name = _sp2
            if any(
                (
                    sA in interaction_map.get(sB, []) 
                    or sA in interaction_map.get(sB_name, [])
                ) for (sA, sB, sB_name) in [
                    (_sp1, _sp2, _sp2_name), 
                    (_sp2, _sp1, _sp1_name),
                ]
            ):
                line1, line2 = _lines1[0], _lines2[0]
                msa_lines.append(
                    PairedMSALine(
                        name=_PAIRED_SPACER.join([str(line1.name), str(line2.name)]),
                        description=_PAIRED_SPACER.join([str(line1.description), str(line2.description)]),
                        sequence="".join([line1.sequence, line2.sequence]),
                    )
                )
                matched_species |= set([_sp1, _sp2])
            
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
            msa1, msa2 = (
                msa._filter_by_index(idx) 
                for idx, msa in zip((idx1, idx2), (msa1, msa2))
            )
            msa_lines += PairedMSA.__make_blocked(msa1, msa2)
        
        return msa_lines, msa1.seq_length

    @staticmethod
    def __make_blocked(
        msa1: MSA, 
        msa2: MSA, 
        gap_char: str = '-'
    ) -> List[PairedMSALine]:
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
        msa2: Optional[MSA] = None,
        blocked: bool = False,
        interaction_map: Optional[Union[str, Mapping[str, Iterable[str]]]] = None,
        strict_species_match: bool = False,
        enforce_ref_match: bool = False,
        **kwargs
    ) -> 'PairedMSA':
        msa_lines, chain_a_length = cls.join_msa(
            msa1, 
            msa2, 
            blocked=blocked, 
            interaction_map=interaction_map, 
            strict_species_match=strict_species_match,
            enforce_ref_match=enforce_ref_match,
            **kwargs
        )
        if len(msa_lines) > 0:
            return cls(lines=msa_lines, chain_a_length=chain_a_length)
        else:
            raise AttributeError(f"Pairing {msa1} and {msa2} resulted in no pairs.")

    @classmethod
    def from_file(
        cls, 
        file1: Union[str, TextIOWrapper],
        file2: Optional[Union[str, TextIOWrapper]] = None,
        blocked: bool = False,
        **kwargs
    ) -> 'PairedMSA':
        """Read A3M file(s).

        """
        msa1 = MSA.from_file(file1)
        if file2 is None:
            msa2 = deepcopy(msa1)
        else:
            msa2 = MSA.from_file(file2)
        return cls.from_msa(msa1, msa2, blocked=blocked, **kwargs)

    def __str__(self) -> str:
        return "Paired " + super().__str__()