"""Tests for host-pathogen interaction database construction.

Tests verify:
  1. Name normalisation handles real organism name variants correctly.
  2. NCBI Taxon ID keys in the built database are strain-specific (specificity).
  3. Normalised species-name keys aggregate across strains (sensitivity).
  4. Organisms known only by name can still be looked up after the fix.
  5. Virus / phage names are never truncated and their interactions round-trip.
  6. All interactions are indexed bidirectionally.

Nomenclature in the tests follows the raw CSV values
(lowercase organism names exactly as stored in ``20250409_hpi.csv``).

Note on expected failures
--------------------------
``TestSpecificity`` tests expose the current cross-contamination bug:
NCBI keys are assigned the merged interaction set of their normalised name
(``interaction_utils._create_data_json`` lines 108-115) rather than the
strain-specific interactions from the NCBI index (lines 61-81, which is
discarded on line 117).  These tests will **fail** on the current code and
must **pass** after the bug fix is applied.
"""

from pathlib import Path
from textwrap import dedent

import pytest

from yunta.interaction_utils import _create_data_json, _name_normalizer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_interaction_map(csv_text: str, tmp_path: Path) -> dict:
    """Write *csv_text* to a temp file and build an interaction map in memory.

    Uses ``test_mode=True`` so no files are written to the filesystem and no
    cache or network access is required.  Returns the raw dict of
    ``{key: set_of_values}`` produced by ``_create_data_json``.
    """
    csv_path = tmp_path / "hpi.csv"
    csv_path.write_text(csv_text)
    return _create_data_json(
        str(csv_path),
        str(tmp_path / "interactions.json.gz"),  # not written in test_mode
        str(tmp_path / "name2ncbi.json"),         # not written in test_mode
        test_mode=True,
    )


# ---------------------------------------------------------------------------
# 1. Name normalisation
# ---------------------------------------------------------------------------

class TestNameNormalizer:
    """_name_normalizer must collapse strain / subspecies detail to genus+species
    while keeping virus and phage names in full."""

    @pytest.mark.parametrize("raw,expected", [
        # Simple two-word names: capitalise first letter, otherwise unchanged.
        # Values taken verbatim from 20250409_hpi.csv.
        ("homo sapiens",           "Homo sapiens"),
        ("mus musculus",           "Mus musculus"),
        ("oryza sativa",           "Oryza sativa"),
        # Three-or-more-word bacterial / fungal names: truncate to first two words.
        ("escherichia coli k-12",  "Escherichia coli"),
        ("mycobacterium tuberculosis h37rv", "Mycobacterium tuberculosis"),
        # Parenthetical strain designators removed.
        ("saccharomyces cerevisiae (strain atcc 204508 / s288c)",
         "Saccharomyces cerevisiae"),
        # Subspecies designators stripped.
        ("homo sapiens subsp. sapiens",         "Homo sapiens"),
        ("salmonella enterica subsp. enterica",  "Salmonella enterica"),
        # Virus names must NOT be truncated regardless of word count.
        ("human immunodeficiency virus 1", "Human immunodeficiency virus 1"),
        ("influenza a virus",              "Influenza a virus"),
        # Phage names must NOT be truncated.
        ("mycobacterium phage d29",    "Mycobacterium phage d29"),
        ("escherichia phage lambda",   "Escherichia phage lambda"),
    ])
    def test_normalisation(self, raw, expected):
        result = _name_normalizer([raw])
        assert result == [expected], (
            f"{raw!r} normalised to {result[0]!r}, expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# 2. Specificity  (these tests FAIL on current code — they document the bug)
# ---------------------------------------------------------------------------

class TestSpecificity:
    """NCBI Taxon ID keys must only carry interactions explicitly recorded for
    that specific ID; they must not inherit interactions from other strains
    that happen to share the same normalised species name."""

    def test_different_strains_same_species_do_not_share_hosts(self, tmp_path):
        """M. tuberculosis H37Rv (NCBI:83332) is recorded as infecting humans;
        the Mtb type strain (NCBI:1773) is recorded here as infecting mice.
        Both names normalise to "Mycobacterium tuberculosis", but their NCBI
        keys must only point to their own respective hosts.

        This test FAILS on the current implementation (cross-contamination bug).
        """
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            EID2,83332,mycobacterium tuberculosis h37rv,9606,homo sapiens
            EID2,1773,mycobacterium tuberculosis,10090,mus musculus
        """)
        db = _build_interaction_map(csv, tmp_path)

        # H37Rv → human (explicitly recorded, must be present)
        assert "NCBI:9606" in db.get("NCBI:83332", []), (
            "H37Rv (NCBI:83332) should interact with Homo sapiens (NCBI:9606)"
        )
        # H37Rv must NOT inherit mouse from the type-strain row
        assert "NCBI:10090" not in db.get("NCBI:83332", []), (
            "H37Rv (NCBI:83332) must not inherit Mus musculus (NCBI:10090) "
            "from the type strain via name-level merging"
        )
        # Type strain → mouse (explicitly recorded, must be present)
        assert "NCBI:10090" in db.get("NCBI:1773", []), (
            "Type strain (NCBI:1773) should interact with Mus musculus (NCBI:10090)"
        )
        # Type strain must NOT inherit human from the H37Rv row
        assert "NCBI:9606" not in db.get("NCBI:1773", []), (
            "Type strain (NCBI:1773) must not inherit Homo sapiens (NCBI:9606) "
            "from H37Rv via name-level merging"
        )

    def test_host_ncbi_keys_are_not_cross_contaminated(self, tmp_path):
        """Two host NCBI IDs that normalise to the same species name must not
        pool each other's pathogens.

        "homo sapiens neanderthal" → first two words → "Homo sapiens", so
        NCBI:9606 and NCBI:63221 share the same name bucket.  The NCBI-keyed
        entries must still be independent.

        This test FAILS on the current implementation (cross-contamination bug).
        """
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            EID2,1773,mycobacterium tuberculosis,9606,homo sapiens
            EID2,11520,influenza a virus,63221,homo sapiens neanderthal
        """)
        db = _build_interaction_map(csv, tmp_path)

        # Homo sapiens (9606) hosts Mtb — must be present
        assert "NCBI:1773" in db.get("NCBI:9606", []), (
            "Homo sapiens (NCBI:9606) should list Mtb (NCBI:1773)"
        )
        # Homo sapiens (9606) must NOT inherit Influenza A from Neanderthal
        assert "NCBI:11520" not in db.get("NCBI:9606", []), (
            "Homo sapiens (NCBI:9606) must not inherit Influenza A (NCBI:11520) "
            "recorded against Neanderthal (NCBI:63221)"
        )
        # Neanderthal (63221) hosts Influenza A — must be present
        assert "NCBI:11520" in db.get("NCBI:63221", []), (
            "Neanderthal (NCBI:63221) should list Influenza A (NCBI:11520)"
        )
        # Neanderthal (63221) must NOT inherit Mtb from Homo sapiens 9606
        assert "NCBI:1773" not in db.get("NCBI:63221", []), (
            "Neanderthal (NCBI:63221) must not inherit Mtb (NCBI:1773) "
            "recorded against Homo sapiens (NCBI:9606)"
        )


# ---------------------------------------------------------------------------
# 3. Sensitivity
# ---------------------------------------------------------------------------

class TestSensitivity:
    """Normalised species-name keys must still aggregate interactions from all
    strains so that the name-based fallback path (used in join_msa when an
    NCBI ID is absent from the database) retains full recall."""

    def test_species_name_key_aggregates_all_strain_interactions(self, tmp_path):
        """Even though H37Rv (83332) and the type strain (1773) must have
        independent NCBI keys, the shared normalised name
        "Mycobacterium tuberculosis" must cover both hosts."""
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            EID2,83332,mycobacterium tuberculosis h37rv,9606,homo sapiens
            EID2,1773,mycobacterium tuberculosis,10090,mus musculus
        """)
        db = _build_interaction_map(csv, tmp_path)

        name_key = "Mycobacterium tuberculosis"
        interactions = db.get(name_key, [])
        assert "NCBI:9606" in interactions, (
            f"{name_key!r} name key must include Homo sapiens (NCBI:9606) "
            "for sensitivity"
        )
        assert "NCBI:10090" in interactions, (
            f"{name_key!r} name key must include Mus musculus (NCBI:10090) "
            "for sensitivity"
        )

    def test_novel_strain_matched_via_name_fallback(self, tmp_path):
        """A query organism with an NCBI ID absent from the database but whose
        normalised species name is present must be reachable via the name key.

        Scenario: database records Cryptosporidium parvum (NCBI:5807) as
        infecting humans.  A novel isolate (NCBI:9999999) that normalises to
        "Cryptosporidium parvum" should not appear as an independent key (no
        spurious entry), but the name key must exist to enable fallback lookup.
        """
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            EID2,5807,cryptosporidium parvum,9606,homo sapiens
        """)
        db = _build_interaction_map(csv, tmp_path)

        assert "NCBI:9999999" not in db, (
            "NCBI:9999999 was never in the source data and must not appear "
            "as a key"
        )
        name_key = "Cryptosporidium parvum"
        assert name_key in db, (
            f"{name_key!r} must be present as a name key for fallback matching"
        )
        assert "NCBI:9606" in db[name_key], (
            f"{name_key!r} name key must point to Homo sapiens (NCBI:9606)"
        )


# ---------------------------------------------------------------------------
# 4. Virus / phage positive controls (real rows from 20250409_hpi.csv)
# ---------------------------------------------------------------------------

class TestVirusAndPhageInteractions:
    """Virus and phage names are never truncated.  The first two SCBIR rows
    in the actual CSV are used verbatim as positive controls."""

    def test_phage_d29_mycolicibacterium_smegmatis(self, tmp_path):
        """Real SCBIR entry: mycobacterium phage d29 (NCBI:228369) infects
        mycolicibacterium smegmatis (NCBI:1772).

        CSV row: SCBIR,228369,mycobacterium phage d29,1772,mycolicibacterium smegmatis
        """
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            SCBIR,228369,mycobacterium phage d29,1772,mycolicibacterium smegmatis
        """)
        db = _build_interaction_map(csv, tmp_path)

        # Phage NCBI key → host NCBI ID
        assert "NCBI:1772" in db.get("NCBI:228369", []), (
            "Phage D29 (NCBI:228369) must list M. smegmatis (NCBI:1772) as host"
        )
        # Full phage name key (not truncated) → host NCBI ID
        assert "NCBI:1772" in db.get("Mycobacterium phage d29", []), (
            "Phage D29 full name key must list M. smegmatis (NCBI:1772)"
        )
        # Bidirectional: host NCBI key → phage NCBI ID
        assert "NCBI:228369" in db.get("NCBI:1772", []), (
            "M. smegmatis (NCBI:1772) must list phage D29 (NCBI:228369) "
            "bidirectionally"
        )

    def test_phage_lambda_ecoli_k12(self, tmp_path):
        """Real SCBIR entry: escherichia phage lambda (NCBI:2681611) infects
        escherichia coli k-12 (NCBI:83333).

        CSV row: SCBIR,2681611,escherichia phage lambda,83333,escherichia coli k-12
        """
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            SCBIR,2681611,escherichia phage lambda,83333,escherichia coli k-12
        """)
        db = _build_interaction_map(csv, tmp_path)

        # Phage NCBI key → host NCBI ID
        assert "NCBI:83333" in db.get("NCBI:2681611", []), (
            "Lambda phage (NCBI:2681611) must list E. coli K-12 (NCBI:83333)"
        )
        # Full phage name key (not truncated)
        assert "NCBI:83333" in db.get("Escherichia phage lambda", []), (
            "Lambda phage full name key must list E. coli K-12 (NCBI:83333)"
        )
        # E. coli K-12 normalises to "Escherichia coli" (two-word rule)
        assert "NCBI:2681611" in db.get("Escherichia coli", []), (
            "Normalised host name 'Escherichia coli' must list lambda phage "
            "(NCBI:2681611)"
        )
        # Bidirectional: host NCBI key → phage NCBI ID
        assert "NCBI:2681611" in db.get("NCBI:83333", []), (
            "E. coli K-12 (NCBI:83333) must list lambda phage (NCBI:2681611) "
            "bidirectionally"
        )


# ---------------------------------------------------------------------------
# 5. Bidirectional indexing
# ---------------------------------------------------------------------------

class TestBidirectionalIndexing:
    """Every pathogen→host interaction in the source CSV must also appear as
    host→pathogen in the built index."""

    def test_pathogen_host_interaction_is_bidirectional(self, tmp_path):
        csv = dedent("""\
            source,pathogen_taxon_id,pathogen_name,host_taxon_id,host_name
            EID2,1773,mycobacterium tuberculosis,9606,homo sapiens
        """)
        db = _build_interaction_map(csv, tmp_path)

        assert "NCBI:9606" in db.get("NCBI:1773", []), (
            "Pathogen (NCBI:1773) must list host (NCBI:9606)"
        )
        assert "NCBI:1773" in db.get("NCBI:9606", []), (
            "Host (NCBI:9606) must list pathogen (NCBI:1773) bidirectionally"
        )
