import pytest

from acc.data import DATA_FILE, load_raw, split_columns

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="run `uv run acc-download` first")


def test_shape_and_split():
    df = load_raw()
    clinical, ige = split_columns(df)
    assert df.shape == (4271, 256)
    assert len(clinical) == 15 and len(ige) == 241
    assert clinical[0] == "Patient_ID" and "Chip_Type" in clinical


def test_types():
    df = load_raw()
    _, ige = split_columns(df)
    assert all(df[c].dtype.kind == "f" for c in ige)
    # multi-coded clinical values must survive as text, not be parsed as decimals
    assert "1,3" in set(df["Treatment_of_rhinitis"].dropna())
