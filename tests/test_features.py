import pandas as pd
import pytest

from acc.data import DATA_FILE, load_raw, split_columns
from acc.features import (
    EXCLUDED_ALLERGENS,
    TARGET_GROUPS,
    age_band,
    allergen_group,
    clinical_feature_labels,
    common_allergens,
    encode_clinical,
    group_columns,
    sensitization_targets,
    split_codes,
)

needs_data = pytest.mark.skipif(not DATA_FILE.exists(), reason="run `uv run acc-download` first")


def test_split_codes():
    assert split_codes("6, 7") == {"6", "7"}
    assert split_codes("1,3") == {"1", "3"}
    assert split_codes(float("nan")) == {"9"}


def test_age_band_bounds():
    assert age_band(0) == 0 and age_band(2.5) == 1 and age_band(17) == 3
    assert age_band(65) == 7 and age_band(102) == 8


def test_encode_clinical_partial_answers():
    df = pd.DataFrame(
        {
            "Age": [4.0, 40.0],
            "Gender": ["1", None],
            "Rural_or_urban_area": ["9", "1"],
            "Skin_Symptoms": ["1", "9"],
            "Treatment_of_rhinitis": ["9", "1,3"],
            "Treatment_of_athsma": ["9", "0"],
            "Treatment_of_atopic_dematitis": ["2", "9"],
            "Age_of_onsets": ["2", "9"],
            "General_cofactors": ["0", "6, 7"],
        }
    )
    x = encode_clinical(df)
    assert list(x.columns) == list(clinical_feature_labels())
    assert x.loc[0, ["age_1", "male", "skin_1", "ad_2", "onset_2"]].tolist() == [1] * 5
    assert x.loc[0, [c for c in x if c.startswith(("rhin_", "asth_", "urban_"))]].sum() == 0
    assert x.loc[1, ["rhin_1", "rhin_3", "cof_6", "cof_7", "asth_0"]].tolist() == [1] * 5
    assert x.loc[1, ["male", "female"]].sum() == 0


def test_food_split_by_protein_family():
    assert allergen_group("Ara_h_2") == "storage"
    assert allergen_group("Pru_p_3") == "ltp"
    assert allergen_group("Mal_d_1") == "pr10"
    assert allergen_group("Gal_d_1") == "egg"
    assert allergen_group("Gal_d_5") == "meat"
    assert allergen_group("Bos_d_8") == "milk"
    assert allergen_group("MUXF3") is None


@needs_data
def test_every_allergen_is_grouped_or_excluded():
    df = load_raw()
    _, ige = split_columns(df)
    assert len(common_allergens(df)) == 91
    groups = group_columns(ige)
    grouped = [c for cols in groups.values() for c in cols]
    assert len(grouped) == len(set(grouped)) == len(ige) - len(EXCLUDED_ALLERGENS)
    assert set(ige) - set(grouped) == set(EXCLUDED_ALLERGENS)
    assert list(groups) == list(TARGET_GROUPS)


@needs_data
def test_venom_target_undefined_on_isac_v2():
    df = load_raw()
    targets = sensitization_targets(df)
    on_v2 = df["Chip_Type"] == "ISAC_V2"
    assert targets.loc[on_v2, "venoms"].isna().all()
    assert targets.loc[~on_v2, "venoms"].notna().all()


def test_severity_degree_takes_highest_step():
    from acc.features import severity_degrees

    df = pd.DataFrame(
        {
            "Treatment_of_athsma": ["0", "1,3", "9", "6"],
            "Treatment_of_atopic_dematitis": ["5", "1, 2", "0", "9"],
            "Treatment_of_rhinitis": ["2", "9", "1,3", "0"],
        }
    )
    d = severity_degrees(df)
    assert d["asthma"].tolist()[:2] == [0, 2] and pd.isna(d["asthma"][2]) and d["asthma"][3] == 3
    assert d["atopic_dermatitis"].tolist()[:3] == [3, 2, 0]
    assert d["rhinitis"][0] == 2 and d["rhinitis"][2] == 3


@needs_data
def test_profile_inputs_and_centres():
    from acc.features import centre_of, encode_profile, profile_feature_labels

    df = load_raw()
    known = df[df["Age"].notna()]
    x = encode_profile(known)
    assert list(x.columns) == list(profile_feature_labels())
    assert x[[c for c in x if c.startswith("chip_")]].sum(axis=1).eq(1).all()
    assert centre_of(df).nunique() == 11
