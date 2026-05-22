"""Integration tests for MSAName, MSADescription, and PairedMSA.join_msa."""

import pytest
from yunta.structs.msa import (
    MSA,
    MSADescription,
    MSALine,
    MSAName,
    PairedMSA,
    __BLOCK_GAPS__,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line(uid, entry, ox, os, seq):
    """Build a real MSALine with UniProt-style header."""
    return MSALine(
        name=f">sp|{uid}|{entry}",
        description=f"Gene OS={os} OX={ox} GN=x PE=1 SV=1",
        sequence=seq,
    )


def _uniref_line(acc, taxid, seq):
    """Build a UniRef-style MSALine (no OX/OS, uses TaxID)."""
    return MSALine(
        name=f">UniRef100_{acc}",
        description=f"Hypothetical protein n=1 Tax=Unknown TaxID={taxid} RepID={acc}",
        sequence=seq,
    )


def _no_species_line(acc, seq):
    """Build an MSALine with no parseable species info."""
    return MSALine(
        name=f">UniRef100_{acc}",
        description="Hypothetical protein",
        sequence=seq,
    )


# ---------------------------------------------------------------------------
# Interaction map fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def hpi_map():
    """Bidirectional host-pathogen interaction map.

    Host:     NCBI:9606  (Homo sapiens)
              NCBI:10090 (Mus musculus)
    Pathogen: NCBI:82830 (EBV AG876 strain)
              NCBI:10376 (EBV species)
    """
    return {
        "NCBI:9606":  ["NCBI:82830", "NCBI:10376"],
        "NCBI:10090": ["NCBI:82830", "NCBI:10376"],
        "NCBI:82830": ["NCBI:9606",  "NCBI:10090"],
        "NCBI:10376": ["NCBI:9606",  "NCBI:10090"],
    }


# ===========================================================================
# MSAName
# ===========================================================================

class TestMSAName:

    @pytest.mark.parametrize("name,exp_db,exp_id,exp_entry", [
        (">sp|P07807|DYR_YEAST",          "sp",            "P07807",          "DYR_YEAST"),
        (">tr|A0A1P7TZK3|A0A1P7TZK3_EBV", "tr",           "A0A1P7TZK3",      "A0A1P7TZK3_EBV"),
        (">UniRef100_A0A0R5ZDS2",          "UniRef100",    "A0A0R5ZDS2",      "A0A0R5ZDS2"),
        (">UniRef90_A0A1B2",               "UniRef90",     "A0A1B2",          "A0A1B2"),
        (">MGYP000745883360",              "__NO_DB_NAME__","MGYP000745883360","MGYP000745883360"),
        ("sp|P07807|DYR_YEAST",           "sp",            "P07807",          "DYR_YEAST"),  # no >
    ])
    def test_parsing(self, name, exp_db, exp_id, exp_entry):
        n = MSAName(name)
        assert n.database  == exp_db
        assert n.unique_id == exp_id
        assert n.entry_name == exp_entry

    def test_str_roundtrip(self):
        """__str__ returns name without leading >."""
        n = MSAName(">sp|P07807|DYR_YEAST")
        assert str(n) == "sp|P07807|DYR_YEAST"

    def test_no_sentinel_unique_id_for_unknown(self):
        """Non-pipe, non-UniRef names use full name as unique_id rather than sentinel."""
        n = MSAName(">MGYP000745883360")
        assert n.unique_id == "MGYP000745883360"
        assert "__NO_ENTRY_ID__" not in n.unique_id


# ===========================================================================
# MSADescription
# ===========================================================================

class TestMSADescription:

    @pytest.mark.parametrize("desc,exp_sid,exp_tid", [
        # Standard UniProt with OX
        ("Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1",     "NCBI:562",  562),
        # OX takes priority over OS
        ("Gene OS=Homo sapiens OX=9606 GN=x PE=1 SV=1",         "NCBI:9606", 9606),
        # TaxID fallback (no OX)
        ("Gene OS=Mycobacterium tuberculosis TaxID=1773 GN=y",   "NCBI:1773", 1773),
        # UniRef-style: TaxID but no OS
        ("Viral DNA n=1 Tax=Harp seal herpesvirus TaxID=1419109 RepID=X", "NCBI:1419109", 1419109),
        # OS-only fallback (no numeric taxon)
        ("Gene OS=Borrelia burgdorferi GN=z",  "Name:Borrelia burgdorferi", -1),
        # No species info at all
        ("Gene GN=w PE=4 SV=1",               -1, -1),
        # Empty OX value falls back to OS
        ("Gene OS=Some organism OX= GN=x",    "Name:Some organism", -1),
    ])
    def test_species_id(self, desc, exp_sid, exp_tid):
        d = MSADescription(desc)
        assert d.species_id == exp_sid
        assert d.taxon_id   == exp_tid

    def test_generic_species_name_present(self):
        d = MSADescription("Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1")
        assert d.generic_species_name == "Escherichia coli"

    def test_generic_species_name_none_without_os(self):
        d = MSADescription("Viral DNA TaxID=1419109")
        assert d.generic_species_name is None

    def test_generic_species_name_none_without_species(self):
        d = MSADescription("Gene GN=w PE=4 SV=1")
        assert d.generic_species_name is None


# ===========================================================================
# MSALine
# ===========================================================================

class TestMSALine:

    def test_insertion_stripping(self):
        line = MSALine(
            name=">sp|P12345|GENE_ECOLI",
            description="Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1",
            sequence="MARNDagctagQE--W",
        )
        assert line.sequence == "MARNDQE--W"

    def test_gap_fraction(self):
        line = MSALine(
            name=">sp|P12345|GENE_ECOLI",
            description="Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1",
            sequence="MARND-----",
        )
        assert line.gap_fraction == pytest.approx(0.5)

    def test_str_uses_input_sequence(self):
        """__str__ should output the original A3M sequence, not the stripped version."""
        raw = "MARNDagctagQE--W"
        line = MSALine(
            name=">sp|P12345|GENE_ECOLI",
            description="Gene OS=Escherichia coli OX=562 GN=x PE=1 SV=1",
            sequence=raw,
        )
        assert raw in str(line)


# ===========================================================================
# PairedMSA / join_msa
# ===========================================================================

class TestJoinMSA:

    # -----------------------------------------------------------------------
    # Intraspecies (no interaction_map)
    # -----------------------------------------------------------------------

    def test_intraspecies_pairs_matching_species(self):
        """Sequences from the same species are paired; different species are paired to same."""
        l1a = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens", "ACDEFGHIKL")
        l1b = _line("P00002", "BBB_MOUSE", 10090, "Mus musculus", "ACDEFGHIKL")
        l2a = _line("P00003", "CCC_HUMAN", 9606,  "Homo sapiens", "MNQRSTVWYK")
        l2b = _line("P00004", "DDD_MOUSE", 10090, "Mus musculus", "MNQRSTVWYK")

        lines, chain_a = PairedMSA.join_msa(MSA([l1a, l1b]), MSA([l2a, l2b]))

        assert chain_a == 10
        assert len(lines) == 2
        ids = {l.unique_id for l in lines}
        assert "P00001-P00003" in ids
        assert "P00002-P00004" in ids

    def test_intraspecies_sequence_concatenated(self):
        l1 = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")
        l2 = _line("P00002", "BBB_HUMAN", 9606, "Homo sapiens", "MNQRSTVWYK")

        lines, _ = PairedMSA.join_msa(MSA([l1]), MSA([l2]))

        assert len(lines) == 1
        assert lines[0].sequence == "ACDEFGHIKLMNQRSTVWYK"

    def test_no_shared_species_raises(self):
        l1 = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens", "ACDEFGHIKL")
        l2 = _line("P00002", "BBB_ECOLI", 562,   "Escherichia coli", "MNQRSTVWYK")

        with pytest.raises((AttributeError, KeyError)):
            PairedMSA.join_msa(MSA([l1]), MSA([l2]))

    # -----------------------------------------------------------------------
    # Interspecies (explicit interaction_map)
    # -----------------------------------------------------------------------

    def test_interspecies_pairs_host_pathogen(self, hpi_map):
        host  = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens",             "ACDEFGHIKL")
        virus = _line("P00002", "BBB_EBV",   82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")

        lines, chain_a = PairedMSA.join_msa(
            MSA([host]), MSA([virus]), interaction_map=hpi_map
        )

        assert len(lines) == 1
        assert chain_a == 10
        assert lines[0].sequence == "ACDEFGHIKLMNQRSTVWYK"

    def test_interspecies_multiple_species_all_paired(self, hpi_map):
        host1  = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens",             "ACDEFGHIKL")
        host2  = _line("P00002", "BBB_MOUSE", 10090, "Mus musculus",             "ACDEFGHIKL")
        virus1 = _line("P00003", "CCC_EBV1",  82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")
        virus2 = _line("P00004", "DDD_EBV2",  10376, "Epstein-Barr virus",       "MNQRSTVWYK")

        lines, _ = PairedMSA.join_msa(
            MSA([host1, host2]), MSA([virus1, virus2]), interaction_map=hpi_map
        )

        assert len(lines) == 2
        ids = {l.unique_id for l in lines}
        assert "P00001-P00003" in ids
        assert "P00002-P00004" in ids

    # -----------------------------------------------------------------------
    # None-species filtering (regression: TypeError: '<' not supported)
    # -----------------------------------------------------------------------

    def test_none_species_does_not_crash(self, hpi_map):
        """Lines with no parseable species are silently dropped; no TypeError on sort."""
        host      = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens",             "ACDEFGHIKL")
        no_sp     = _no_species_line("A0A000000", "ACDEFGHIKL")
        virus     = _line("P00002", "BBB_EBV",  82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")

        # This previously crashed with TypeError: '<' not supported between str and NoneType
        lines, _ = PairedMSA.join_msa(
            MSA([host, no_sp]), MSA([virus]), interaction_map=hpi_map
        )

        assert len(lines) == 1
        assert lines[0].unique_id == "P00001-P00002"

    def test_uniref_taxid_only_not_paired_as_none(self, hpi_map):
        """UniRef lines with TaxID but no OS get a valid species_id, not None."""
        host   = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")
        uniref = _uniref_line("A0A0R5ZDS2", 82830, "MNQRSTVWYK")

        assert uniref.description.species_id == "NCBI:82830"

        lines, _ = PairedMSA.join_msa(
            MSA([host]), MSA([uniref]), interaction_map=hpi_map
        )

        assert len(lines) == 1

    # -----------------------------------------------------------------------
    # Gap-fraction-based selection
    # -----------------------------------------------------------------------

    def test_min_gap_fraction_selected(self, hpi_map):
        """When multiple sequences exist for a species, the least-gappy is chosen."""
        host_ref = _line("P0000x", "BBB_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")  # 0.0
        host_gappy = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens", "ACDE------")  # 0.6
        host_clean = _line("P00002", "BBB_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")  # 0.0
        virus = _line("P00003", "CCC_EBV",  82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")
        virus2 = _line("P00003b", "CCC_EBVb",  10376, "EBV", "MNQRS--WYK")

        lines, _ = PairedMSA.join_msa(
            MSA([host_ref, host_gappy, host_clean]), 
            MSA([virus, virus2]), 
            interaction_map=hpi_map,
        )

        assert len(lines) == 2
        assert lines[0].unique_id.split("-")[0] == "P0000x"
        assert lines[1].unique_id.split("-")[0] == "P00002"

    def test_min_gap_fraction_both_sides(self, hpi_map):
        """Gap-fraction selection applies independently to both MSAs."""
        host_ref = _line("P0000x", "BBB_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")  # 0.0
        host_gappy = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens", "ACDE------")  # 0.6
        host_clean = _line("P00002", "BBB_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")  # 0.0
        virus_gappy = _line("P00002", "BBB_EBV1",  82830, "Epstein-Barr virus AG876", "MN--------")  # 0.8
        virus_clean = _line("P00003", "CCC_EBV2",  82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")  # 0.0
        virus2 = _line("P00003b", "CCC_EBVb",  10376, "EBV", "MNQRS--WYK")

        lines, _ = PairedMSA.join_msa(
            MSA([host_ref, host_gappy, host_clean]), 
            MSA([virus2, virus_gappy, virus_clean]), 
            interaction_map=hpi_map,
        )

        assert len(lines) == 2
        assert lines[0].unique_id.split("-")[1] == "P00003b"
        assert lines[1].unique_id.split("-")[1] == "P00003"

    # -----------------------------------------------------------------------
    # Blocked MSA
    # -----------------------------------------------------------------------

    def test_blocked_unmatched_go_to_block(self, hpi_map):
        """Sequences with no cross-species counterpart appear as block entries."""
        host_paired = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens", "ACDEFGHIKL")
        host_unpaired = _line("P00002", "BBB_ECOLI", 562,  "Escherichia coli", "ACDEFGHIKL")
        virus = _line("P00003", "CCC_EBV",  82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")

        lines, chain_a = PairedMSA.join_msa(
            MSA([host_paired, host_unpaired]),
            MSA([virus]),
            interaction_map=hpi_map,
            blocked=True,
        )

        # 1 paired + 1 block (host_unpaired|gaps)
        assert len(lines) == 3

        paired_ids = {l.unique_id for l in lines if __BLOCK_GAPS__ not in l.unique_id}
        block_ids = {l.unique_id for l in lines if __BLOCK_GAPS__ in l.unique_id}

        assert len(paired_ids) == 1
        assert len(block_ids)  == 1

    def test_blocked_gap_fill_correct_length(self, hpi_map):
        """Block entries are padded with gaps to the correct combined length."""
        host_paired   = _line("P00001", "AAA_HUMAN", 9606, "Homo sapiens",             "ACDEFGHIKL")
        host_unpaired = _line("P00002", "BBB_ECOLI", 562,  "Escherichia coli",         "ACDEFGHIKL")
        virus         = _line("P00003", "CCC_EBV",  82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")

        lines, chain_a = PairedMSA.join_msa(
            MSA([host_paired, host_unpaired]),
            MSA([virus]),
            interaction_map=hpi_map,
            blocked=True,
        )

        expected_total = chain_a + 10  # both sequences are length 10
        for l in lines:
            assert len(l.sequence) == expected_total

    # -----------------------------------------------------------------------
    # chain_a_length
    # -----------------------------------------------------------------------

    def test_chain_a_length_equals_msa1_seq_length(self, hpi_map):
        host  = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens",             "ACDEFGHIKL")   # L=10
        virus = _line("P00002", "BBB_EBV",   82830, "Epstein-Barr virus AG876", "MNQRSTVWYKMN") # L=12

        _, chain_a = PairedMSA.join_msa(
            MSA([host]), MSA([virus]), interaction_map=hpi_map
        )

        assert chain_a == 10


# ===========================================================================
# PairedMSA.split()
# ===========================================================================

class TestPairedMSASplit:

    def _make_paired(self, hpi_map):
        host  = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens",             "ACDEFGHIKL")
        virus = _line("P00002", "BBB_EBV",   82830, "Epstein-Barr virus AG876", "MNQRSTVWYK")
        return PairedMSA.from_msa(MSA([host]), MSA([virus]), interaction_map=hpi_map)

    def test_split_returns_two_msas(self, hpi_map):
        paired = self._make_paired(hpi_map)
        m1, m2 = paired.split()
        assert isinstance(m1, MSA)
        assert isinstance(m2, MSA)

    def test_split_sequence_slices_correct(self, hpi_map):
        paired = self._make_paired(hpi_map)
        m1, m2 = paired.split()
        assert m1.lines[0].sequence == "ACDEFGHIKL"
        assert m2.lines[0].sequence == "MNQRSTVWYK"

    def test_split_names_correct(self, hpi_map):
        paired = self._make_paired(hpi_map)
        m1, m2 = paired.split()
        assert m1.lines[0].unique_id == "P00001"
        assert m2.lines[0].unique_id == "P00002"


# ===========================================================================
# PairedMSA dataclass fields
# ===========================================================================

class TestPairedMSADataclass:

    def test_chain_a_length_is_dataclass_field(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(PairedMSA)}
        assert "chain_a_length" in field_names
        assert "chain_b_length" in field_names

    def test_chain_b_length_computed(self, hpi_map):
        host  = _line("P00001", "AAA_HUMAN", 9606,  "Homo sapiens",             "ACDEFGHIKL")   # 10
        virus = _line("P00002", "BBB_EBV",   82830, "Epstein-Barr virus AG876", "MNQRSTVWYKMN") # 12
        paired = PairedMSA.from_msa(MSA([host]), MSA([virus]), interaction_map=hpi_map)
        assert paired.chain_a_length == 10
        assert paired.chain_b_length == 12
        assert paired.seq_length == 22
