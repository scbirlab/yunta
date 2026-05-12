"""Tests for PairedMSA._check_ref_match."""

import pytest
from yunta.structs.msa import PairedMSA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Desc:
    def __init__(self, species_id, generic_species_name=None):
        self.species_id = species_id
        self.generic_species_name = generic_species_name


class _Line:
    def __init__(self, species_id, generic_species_name=None):
        self.description = _Desc(species_id, generic_species_name)


class _MSA:
    def __init__(self, species_id, generic_species_name=None):
        self.lines = [_Line(species_id, generic_species_name)]


@pytest.fixture
def host_pathogen_map():
    """Minimal bidirectional host-pathogen interaction map.

    NCBI:562  = Escherichia coli (species)
    NCBI:10710 = Enterobacteria phage lambda
    """
    return {
        "NCBI:562":   ["NCBI:10710"],
        "NCBI:10710": ["NCBI:562"],
    }


@pytest.fixture
def name_map():
    """Name-keyed interaction map for testing the name_attr fallback."""
    return {
        "Escherichia coli": ["Enterobacteria phage lambda"],
        "Enterobacteria phage lambda": ["Escherichia coli"],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_explicit_forward(host_pathogen_map):
    """Host → pathogen direction resolves."""
    PairedMSA._check_ref_match(
        _MSA("NCBI:562"), _MSA("NCBI:10710"), host_pathogen_map
    )


def test_explicit_reverse(host_pathogen_map):
    """Pathogen → host direction also resolves (OR logic)."""
    PairedMSA._check_ref_match(
        _MSA("NCBI:10710"), _MSA("NCBI:562"), host_pathogen_map
    )


def test_name_attr_fallback(name_map):
    """Matching works when name_attr points to non-default description field."""
    PairedMSA._check_ref_match(
        _MSA("Escherichia coli"),
        _MSA("Enterobacteria phage lambda"),
        name_map,
        name_attr="species_id",
    )


def test_same_species_no_map():
    """Without an interaction map, identical species IDs are accepted."""
    PairedMSA._check_ref_match(_MSA("NCBI:562"), _MSA("NCBI:562"), None)


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

def test_not_in_map_raises(host_pathogen_map):
    """Species not present in either map direction raises AttributeError."""
    with pytest.raises(AttributeError):
        PairedMSA._check_ref_match(
            _MSA("NCBI:562"), _MSA("NCBI:83333"), host_pathogen_map
        )


def test_empty_map_raises():
    """Empty interaction map rejects any pair."""
    with pytest.raises(AttributeError):
        PairedMSA._check_ref_match(_MSA("NCBI:562"), _MSA("NCBI:10710"), {})


# ---------------------------------------------------------------------------
# Unknown species (-1) behaviour
# ---------------------------------------------------------------------------

def test_both_unknown_no_map_raises():
    """Both species unknown (id == -1) with no map raises ValueError."""
    with pytest.raises(ValueError):
        PairedMSA._check_ref_match(_MSA(-1), _MSA(-1), None)


def test_unknown_id1_with_map_raises(host_pathogen_map):
    """Unknown id1 (-1) raises; AttributeError fires before ValueError
    because -1 is not a key in the interaction map.
    """
    with pytest.raises((AttributeError, ValueError)):
        PairedMSA._check_ref_match(_MSA(-1), _MSA("NCBI:10710"), host_pathogen_map)
