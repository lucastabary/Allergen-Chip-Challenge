import numpy as np
import pandas as pd
import pytest

from acc.data import DATA_FILE, load_raw
from acc.features import (
    INTENSITY_LEVELS,
    TARGET_GROUPS,
    encode_intensity,
    food_allergy,
    group_intensity,
)
from acc.profiles import _profile_name

needs_data = pytest.mark.skipif(not DATA_FILE.exists(), reason="run `uv run acc-download` first")


def test_profile_names():
    prevalence = pd.Series(0.1, index=list(TARGET_GROUPS))
    assert _profile_name(prevalence, 0) == "Peu ou pas sensibilisés"
    assert _profile_name(prevalence, 12) == "Polysensibilisés"
    prevalence[["mites", "tree"]] = 0.9
    name = _profile_name(prevalence, 3)
    assert "cariens" in name and "rbres" in name


def test_food_allergy_from_onset_codes():
    df = pd.DataFrame({"Age_of_onsets": ["0", "2", "1,3", "9", None]})
    out = food_allergy(df)
    assert out.iloc[0] == 0 and out.iloc[1] == 1 and out.iloc[2] == 1
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4])


@needs_data
def test_intensity_levels_and_inputs():
    df = load_raw()
    levels = group_intensity(df)
    assert list(levels.columns) == list(TARGET_GROUPS)
    assert levels.isin(range(len(INTENSITY_LEVELS))).all().all()
    # venoms are not on ISAC v2: always negative there
    assert (levels.loc[df["Chip_Type"] == "ISAC_V2", "venoms"] == 0).all()
    x = encode_intensity(df[df["Age"].notna()])
    assert x.shape[1] == 3 * len(TARGET_GROUPS) + 9 + 2 + 3
    # at most one intensity case per group
    for key in TARGET_GROUPS:
        assert (x[[f"int_{key}_{lv}" for lv in (1, 2, 3)]].sum(axis=1) <= 1).all()


def test_profile_allergen_rates():
    from acc.profiles import profile_allergen_rates

    clinical = {f"clin_{i}": ["9"] * 4 for i in range(13)}  # 15 clinical columns in all
    df = pd.DataFrame(
        {
            "Patient_ID": ["a", "b", "c", "d"],
            "Chip_Type": ["ISAC_V1", "ISAC_V2", "ALEX", "ALEX"],
            **clinical,
            "Bet_v_1": [0.0, 2.0, 5.0, 0.1],
            "Der_p_1": [1.0, 0.0, 0.4, 0.5],
        }
    )
    out = profile_allergen_rates(df, np.array([0, 0, 1, 1]))
    # trees come before mites in TARGET_GROUPS
    assert out["columns"] == ["Bet_v_1", "Der_p_1"]
    assert out["groups"] == ["tree", "mites"]
    assert out["cohort"] == [0.5, 0.75]
    assert out["by_profile"][0] == [0.5, 0.5]
    assert out["by_profile"][1] == [0.5, 1.0]
