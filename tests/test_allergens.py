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
