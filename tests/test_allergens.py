import pytest

from acc.allergens import FAMILIES, family_of, source_of
from acc.data import DATA_FILE, load_raw, split_columns


def test_source_of():
    assert source_of("Der_p_2") == "Der_p"
    assert source_of("Cor_a_1.0101") == "Cor_a"
    assert source_of("Alpha-Gal") == "Alpha-Gal"
    assert source_of("Chi_spp.") == "Chi_spp"


def test_family_overrides():
    assert family_of("Cor_a_14") == "plant food"
    assert family_of("Cor_a_1.0101") == "tree pollen"
    assert family_of("Equ_c_1") == "animal dander"
    assert family_of("Equ_c_milk") == "animal food"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="run `uv run acc-download` first")
def test_every_ige_column_has_a_family():
    _, ige = split_columns(load_raw())
    assert all(family_of(c) in FAMILIES for c in ige)


def test_iuis_table_covers_every_column():
    from acc.allergens import iuis_table

    table = iuis_table()
    assert len(table) == 241 and table.index.is_unique
    assert set(table["origin"]) <= {"iuis", "iuis_species", "manual"}


def test_names_and_protein_families():
    from acc.allergens import display_name, protein_family_of

    assert display_name("Der_p_2") == "Der p 2 (European house dust mite)"
    assert display_name("Cyn_d") == "Cyn d extract (Bermuda grass)"
    assert protein_family_of("Ara_h_2") == "2S albumin"
    assert protein_family_of("Ara_h_8") == "PR-10"
    assert protein_family_of("Pru_p_3") == "nsLTP"
    assert protein_family_of("Pen_m_1") == "Tropomyosin"
    assert protein_family_of("Cyn_d") == "unknown"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="run `uv run acc-download` first")
def test_iuis_table_matches_data_columns():
    from acc.allergens import iuis_table

    _, ige = split_columns(load_raw())
    assert list(iuis_table().index) == ige
