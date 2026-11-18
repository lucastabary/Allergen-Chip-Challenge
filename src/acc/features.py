"""Shared feature engineering: IgE positivity, cross-chip allergen set, clinical encoding.

Everything here is used by the notebooks and by the clinical simulator (`acc.atlas`),
so the three of us build on the same definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from acc.data import split_columns

# Usual positivity cut-off of the multiplex chips (kUA/L on ALEX, ISU on ISAC).
POSITIVITY_THRESHOLD = 0.3

NOT_PROVIDED = "9"


def common_allergens(df: pd.DataFrame) -> list[str]:
    """IgE columns measured on all three chips (91 columns).

    Cross-chip analyses must be restricted to these, otherwise structural NaN
    (allergen absent from a chip) would be read as information.
    """
    _, ige = split_columns(df)
    measured = df.groupby("Chip_Type")[ige].agg(lambda s: s.notna().any())
    return [c for c in ige if measured[c].all()]


def positivity(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Boolean table: IgE >= 0.3. Unmeasured or invalid values count as negative."""
    return df[columns].fillna(0) >= POSITIVITY_THRESHOLD


def split_codes(value: object) -> set[str]:
    """Codes of a multi-coded clinical cell: '6, 7' -> {'6', '7'}; missing -> {'9'}."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {NOT_PROVIDED}
    return {part.strip() for part in str(value).split(",") if part.strip()}


# ---------------------------------------------------------------------------
# Clinical codebook
# ---------------------------------------------------------------------------
#
# Labels come from `acc-dictionnaire-final.xls` (French codebook published with the data).
# They are kept in French because they are shown as is in the atlas. Rare codes
# (fewer than ~5 patients) are listed for display but not used as model features.


@dataclass(frozen=True)
class ClinicalField:
    key: str  # short feature prefix
    column: str  # column in the ACC table
    label: str  # French label shown in the atlas
    options: dict[str, str]  # code -> French label (without "9 = not provided")
    multiple: bool = False  # several codes can be ticked at once
    model_codes: tuple[str, ...] = ()  # codes used as model features


CLINICAL_FIELDS: list[ClinicalField] = [
    ClinicalField(
        "urban",
        "Rural_or_urban_area",
        "Type d'habitat",
        {"0": "Rural", "1": "Urbain"},
        model_codes=("0", "1"),
    ),
    ClinicalField(
        "skin",
        "Skin_Symptoms",
        "Symptômes cutanés (peau, urticaire)",
        {"0": "Non", "1": "Oui"},
        model_codes=("0", "1"),
    ),
    ClinicalField(
        "rhin",
        "Treatment_of_rhinitis",
        "Traitement de la rhinite",
        {
            "0": "Aucun",
            "1": "Antihistaminique local",
            "2": "Antihistaminique par voie générale",
            "3": "Antihistaminique + corticoïde local",
            "4": "Corticoïde par voie générale",
        },
        multiple=True,
        model_codes=("0", "1", "2", "3"),
    ),
    ClinicalField(
        "asth",
        "Treatment_of_athsma",
        "Traitement de l'asthme",
        {
            "0": "Aucun",
            "1": "Bronchodilatateur à la demande (SABA)",
            "2": "Corticoïde inhalé faible dose",
            "3": "Corticoïde inhalé + LABA faible dose",
            "4": "Corticoïde inhalé + LABA dose moyenne ou forte",
            "5": "Corticoïde inhalé + LABA forte dose ou au long cours",
            "6": "Biothérapie anti-IL5",
            "7": "Biothérapie anti-IL4/IL13",
            "8": "Non répondeur aux biothérapies",
            "10": "Antileucotriène",
        },
        multiple=True,
        model_codes=("0", "1", "2", "3", "4", "5", "6", "7"),
    ),
    ClinicalField(
        "ad",
        "Treatment_of_atopic_dematitis",
        "Traitement de la dermatite atopique",
        {
            "0": "Aucun",
            "1": "Émollients",
            "2": "Dermocorticoïdes",
            "3": "Tacrolimus topique",
            "4": "Ciclosporine",
            "5": "Dupilumab",
            "6": "Upadacitinib",
            "7": "Baricitinib",
        },
        multiple=True,
        model_codes=("0", "1", "2", "5"),
    ),
    ClinicalField(
        "onset",
        "Age_of_onsets",
        "Allergie alimentaire : âge d'apparition",
        {
            "0": "Pas d'allergie alimentaire",
            "1": "Avant 6 mois",
            "2": "Entre 6 mois et 2 ans",
            "3": "Entre 2 et 3 ans",
            "4": "Entre 3 et 10 ans",
            "5": "Entre 10 et 20 ans",
            "6": "Après 20 ans",
        },
        multiple=True,
        model_codes=("0", "1", "2", "3", "4", "5", "6"),
    ),
    ClinicalField(
        "cof",
        "General_cofactors",
        "Cofacteurs et expositions",
        {
            "1": "Effort, activité sportive",
            "2": "Anti-inflammatoires (AINS)",
            "3": "Alcool",
            "4": "Moisissures au domicile",
            "5": "Acariens (literie ancienne)",
            "6": "Chat au domicile",
            "7": "Chien au domicile",
            "8": "Cheval",
            "10": "Rongeur au domicile",
            "11": "Mucoviscidose",
            "12": "Inhibiteur de la pompe à protons",
        },
        multiple=True,
        # Code 10 is left out of the model: 918 patients carry it, almost all from four
        # centres, which is implausible for pet rodents and looks like a centre artefact.
        model_codes=("1", "3", "4", "5", "6", "7"),
    ),
]

# General_cofactors has no "not provided" code: "0" (none known) is its default value.
CLINICAL_COLUMNS_WITH_MISSING_CODE = [
    f.column for f in CLINICAL_FIELDS if f.column != "General_cofactors"
]

# Age bands (upper bounds included), in years.
AGE_BINS: list[tuple[int, int, str]] = [
    (0, 2, "0 à 2 ans"),
    (3, 5, "3 à 5 ans"),
    (6, 11, "6 à 11 ans"),
    (12, 17, "12 à 17 ans"),
    (18, 29, "18 à 29 ans"),
    (30, 44, "30 à 44 ans"),
    (45, 59, "45 à 59 ans"),
    (60, 69, "60 à 69 ans"),
    (70, 200, "70 ans et plus"),  # only 42 training patients: no finer band
]


def age_band(age: float) -> int:
    """Index in AGE_BINS of an age in years."""
    for i, (_, high, _) in enumerate(AGE_BINS):
        if age <= high:
            return i
    return len(AGE_BINS) - 1


def centre_of(df: pd.DataFrame) -> pd.Series:
    """Recruiting centre: the letters of Patient_ID (11 centres, 'XPW0012' -> 'XPW')."""
    return df["Patient_ID"].str.replace(r"\d+$", "", regex=True)


def n_clinical_known(df: pd.DataFrame) -> pd.Series:
    """Number of clinical variables that are provided (code other than 9), 0 to 6."""
    known = df[CLINICAL_COLUMNS_WITH_MISSING_CODE].fillna(NOT_PROVIDED) != NOT_PROVIDED
    return known.sum(axis=1)


def clinical_feature_labels() -> dict[str, str]:
    """Feature name -> French label, in the column order of `encode_clinical()`."""
    labels = {f"age_{i}": f"Âge : {label}" for i, (_, _, label) in enumerate(AGE_BINS)}
    labels["male"] = "Sexe masculin"
    labels["female"] = "Sexe féminin"
    for field in CLINICAL_FIELDS:
        for code in field.model_codes:
            labels[f"{field.key}_{code}"] = f"{field.label} : {field.options[code]}"
    return labels


def encode_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding of age band, sex and clinical codes.

    A "not provided" answer is encoded as all zeros for its variable, so a model
    can be queried with any subset of answers. Rows must have a known age.
    """
    if df["Age"].isna().any():
        raise ValueError("encode_clinical() needs a known age on every row.")
    features: dict[str, np.ndarray] = {}
    bands = df["Age"].map(age_band).to_numpy()
    for i in range(len(AGE_BINS)):
        features[f"age_{i}"] = (bands == i).astype(int)
    features["male"] = (df["Gender"] == "1").astype(int).to_numpy()
    features["female"] = (df["Gender"] == "0").astype(int).to_numpy()
    for field in CLINICAL_FIELDS:
        codes = df[field.column].map(split_codes)
        for code in field.model_codes:
            features[f"{field.key}_{code}"] = codes.map(lambda s, c=code: int(c in s)).to_numpy()
    return pd.DataFrame(features, index=df.index)


# ---------------------------------------------------------------------------
# Sensitization targets: 18 allergen groups covering 235 of the 241 IgE columns
# ---------------------------------------------------------------------------
#
# Each IgE column belongs to at most one group. Groups follow the source families of
# `acc.allergens`; food is split by protein family because severity depends on it
# (storage proteins and LTP are linked to systemic reactions, PR-10 and profilins to
# mild oral symptoms). A patient's target for a group is "at least one IgE >= 0.3 among
# the group's columns measured on their chip", and is undefined (NaN) when their chip
# measures none of them (venoms on ISAC v2).

RESPIRATORY = "Allergènes respiratoires"
FOOD = "Allergènes alimentaires"
VENOM_LATEX = "Venins et latex"

# key -> (French label, category)
TARGET_GROUPS: dict[str, tuple[str, str]] = {
    "grass": ("Pollens de graminées", RESPIRATORY),
    "tree": ("Pollens d'arbres", RESPIRATORY),
    "weed": ("Pollens d'herbacées", RESPIRATORY),
    "mites": ("Acariens", RESPIRATORY),
    "animals": ("Animaux (poils, squames)", RESPIRATORY),
    "molds": ("Moisissures et levures", RESPIRATORY),
    "insects": ("Insectes (blatte, criquet, grillon, chironome)", RESPIRATORY),
    "storage": ("Aliments végétaux : protéines de stockage", FOOD),
    "ltp": ("Aliments végétaux : protéines de transfert lipidique (LTP)", FOOD),
    "pr10": ("Aliments végétaux : PR-10 et profilines", FOOD),
    "plant_other": ("Aliments végétaux : extraits et autres protéines", FOOD),
    "egg": ("Œuf", FOOD),
    "milk": ("Lait", FOOD),
    "fish": ("Poissons", FOOD),
    "seafood": ("Crustacés, mollusques et anisakis", FOOD),
    "meat": ("Viandes et albumines sériques (dont alpha-gal)", FOOD),
    "venoms": ("Venins d'hyménoptères (abeille, guêpe)", VENOM_LATEX),
    "latex": ("Latex", VENOM_LATEX),
}

_FAMILY_TO_GROUP = {
    "grass pollen": "grass",
    "tree pollen": "tree",
    "weed pollen": "weed",
    "mites": "mites",
    "animal dander": "animals",
    "molds / yeasts": "molds",
    "insects": "insects",
    "venoms": "venoms",
    "latex": "latex",
}
_STORAGE_PROTEINS = {"2S albumin", "7S vicilin", "11S legumin"}
_MILK = {
    "Bos_d_4",
    "Bos_d_5",
    "Bos_d_8",
    "Bos_d_Lactoferrin",
    "Bos_d_milk",
    "Cap_h_milk",
    "Equ_c_milk",
}
_EGG_EXCLUDED = {"Gal_d_5", "Gal_d_meat"}  # chicken serum albumin and meat go to "meat"
_FISH = {"Gad_c", "Gad_m", "Clu_h", "Cyp_c"}
_SEAFOOD = {"Pen_m", "Lit_s", "Cra_c", "Hom_g", "Car_c", "Myt_e", "Ani_s"}

# The 6 columns of the "other" source family are kept out of every group (French reason shown).
EXCLUDED_ALLERGENS: dict[str, str] = {
    "MUXF3": "Marqueur des déterminants glucidiques croisés (CCD) : souvent positif sans "
    "conséquence clinique, et neutralisé par un inhibiteur sur ALEX².",
    "Hom_s_LF": "Lactoferrine humaine : protéine du patient lui-même, pas une source "
    "d'exposition allergénique.",
    "Aca_m": "Extrait non identifié dans la nomenclature WHO/IUIS.",
    "Can_s": "Chanvre : source isolée, mesurée sur ALEX² seulement.",
    "Can_s_3": "Chanvre : source isolée, mesurée sur ALEX² seulement.",
    "Fic_b": "Ficus (plante d'intérieur) : source isolée, mesurée sur ALEX² seulement.",
}


def allergen_group(column: str) -> str | None:
    """Key in TARGET_GROUPS of an IgE column, or None for the excluded columns."""
    from acc.allergens import family_of, protein_family_of, source_of

    family = family_of(column)
    if family in _FAMILY_TO_GROUP:
        return _FAMILY_TO_GROUP[family]
    if family == "plant food":
        protein = protein_family_of(column)
        if protein in _STORAGE_PROTEINS:
            return "storage"
        if protein == "nsLTP":
            return "ltp"
        if protein in ("PR-10", "Profilin"):
            return "pr10"
        return "plant_other"
    if family == "animal food":
        source = source_of(column)
        if column in _MILK:
            return "milk"
        if source == "Gal_d" and column not in _EGG_EXCLUDED:
            return "egg"
        if source in _FISH:
            return "fish"
        if source in _SEAFOOD:
            return "seafood"
        return "meat"
    return None


def group_columns(columns: list[str]) -> dict[str, list[str]]:
    """IgE columns of each group, in TARGET_GROUPS order."""
    groups: dict[str, list[str]] = {key: [] for key in TARGET_GROUPS}
    for column in columns:
        key = allergen_group(column)
        if key is not None:
            groups[key].append(column)
    return groups


def measured_by_chip(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean table chip x IgE column: is the allergen on that chip?"""
    _, ige = split_columns(df)
    return df.groupby("Chip_Type")[ige].agg(lambda s: s.notna().any())


def sensitization_targets(df: pd.DataFrame) -> pd.DataFrame:
    """One column per group: 1.0 / 0.0, or NaN when the patient's chip has none of its allergens."""
    _, ige = split_columns(df)
    measured = measured_by_chip(df)
    out = pd.DataFrame(index=df.index, dtype=float)
    for key, columns in group_columns(ige).items():
        target = pd.Series(np.nan, index=df.index)
        for chip in measured.index:
            on_chip = [c for c in columns if measured.loc[chip, c]]
            rows = df["Chip_Type"] == chip
            if on_chip:
                target[rows] = positivity(df.loc[rows], on_chip).any(axis=1).astype(float)
        out[key] = target
    return out


# ---------------------------------------------------------------------------
# Severity proxy: treatment step of three allergic diseases, as a degree 0 to 3
# ---------------------------------------------------------------------------
#
# The data has no severity label. Guidelines step treatment up with severity, so the
# treatment step is used as a proxy, mapped to 0 = no treatment, 1 = mild, 2 = moderate,
# 3 = severe. A multi-coded cell takes its highest step; code 9 gives NaN.
#   asthma (GINA steps): SABA, low-dose ICS or LTRA = mild; low-dose ICS/LABA = moderate;
#     medium/high-dose ICS/LABA and biologics = severe.
#   rhinitis (ARIA): local antihistamine = mild; oral antihistamine = moderate;
#     antihistamine + nasal corticosteroid, or systemic corticosteroid = severe.
#   atopic dermatitis: emollients = mild; topical anti-inflammatory (corticosteroid,
#     tacrolimus) = moderate; systemic treatment (ciclosporin, dupilumab, JAK inhibitor) = severe.
# Treatment also reflects prescribing habits (centre, period): it is a proxy, not a measure.

SEVERITY_LEVELS = ["Aucun traitement", "Léger", "Modéré", "Sévère"]

# key -> (French label, column, treatment code -> degree)
SEVERITY_TARGETS: dict[str, tuple[str, str, dict[str, int]]] = {
    "asthma": (
        "Asthme",
        "Treatment_of_athsma",
        {"0": 0, "1": 1, "2": 1, "10": 1, "3": 2, "4": 3, "5": 3, "6": 3, "7": 3, "8": 3},
    ),
    "atopic_dermatitis": (
        "Dermatite atopique",
        "Treatment_of_atopic_dematitis",
        {"0": 0, "1": 1, "2": 2, "3": 2, "4": 3, "5": 3, "6": 3, "7": 3},
    ),
    "rhinitis": (
        "Rhinite",
        "Treatment_of_rhinitis",
        {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3},
    ),
}


def _degree(value: object, mapping: dict[str, int]) -> float:
    codes = split_codes(value)
    if NOT_PROVIDED in codes:
        return np.nan
    known = [mapping[c] for c in codes if c in mapping]
    return float(max(known)) if known else np.nan


def severity_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """One column per SEVERITY_TARGETS entry: degree 0 to 3, NaN when not provided."""
    return pd.DataFrame(
        {
            key: df[column].map(lambda v, m=mapping: _degree(v, m))
            for key, (_, column, mapping) in SEVERITY_TARGETS.items()
        },
        index=df.index,
    )


def encode_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs of the severity models: age band, sex, the 18 group flags and the chip.

    A group flag is 1 when the group is positive on the patient's own chip, 0 when it is
    negative or not measured by that chip (the chip inputs let the model tell them apart).
    """
    if df["Age"].isna().any():
        raise ValueError("encode_profile() needs a known age on every row.")
    features: dict[str, np.ndarray] = {}
    bands = df["Age"].map(age_band).to_numpy()
    for i in range(len(AGE_BINS)):
        features[f"age_{i}"] = (bands == i).astype(int)
    features["male"] = (df["Gender"] == "1").astype(int).to_numpy()
    features["female"] = (df["Gender"] == "0").astype(int).to_numpy()
    targets = sensitization_targets(df).fillna(0).astype(int)
    for key in TARGET_GROUPS:
        features[f"grp_{key}"] = targets[key].to_numpy()
    for chip in ("ISAC_V1", "ISAC_V2", "ALEX"):
        features[f"chip_{chip}"] = (df["Chip_Type"] == chip).astype(int).to_numpy()
    return pd.DataFrame(features, index=df.index)


def profile_feature_labels() -> dict[str, str]:
    """Feature name -> French label, in the column order of `encode_profile()`."""
    labels = {f"age_{i}": f"Âge : {label}" for i, (_, _, label) in enumerate(AGE_BINS)}
    labels["male"] = "Sexe masculin"
    labels["female"] = "Sexe féminin"
    for key, (label, _) in TARGET_GROUPS.items():
        labels[f"grp_{key}"] = f"Sensibilisé : {label}"
    for chip, name in (("ISAC_V1", "ISAC v1"), ("ISAC_V2", "ISAC v2"), ("ALEX", "ALEX²")):
        labels[f"chip_{chip}"] = f"Puce {name}"
    return labels


# ---------------------------------------------------------------------------
# IgE intensity per group, food allergy, and the inputs of the IgE-profile models
# ---------------------------------------------------------------------------
#
# Intensity of a group = its highest IgE among the group's allergens on the patient's chip,
# in the three bands used by the ISAC report (0.3-0.9 low, 1-14.9 moderate, >= 15 high ISU),
# applied as is to ALEX (kUA/L): 0 = negative or not measured, 1 = low, 2 = moderate, 3 = high.

INTENSITY_BOUNDS = (POSITIVITY_THRESHOLD, 1.0, 15.0)
INTENSITY_LEVELS = ["Négatif", "Faible", "Modéré", "Fort"]


def group_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Intensity level (0-3) of each of the 18 groups, on each patient's own chip."""
    _, ige = split_columns(df)
    measured = measured_by_chip(df)
    out = pd.DataFrame(0, index=df.index, columns=list(TARGET_GROUPS), dtype=int)
    low, mid, high = INTENSITY_BOUNDS
    for key, columns in group_columns(ige).items():
        for chip in measured.index:
            on_chip = [c for c in columns if measured.loc[chip, c]]
            rows = df["Chip_Type"] == chip
            if on_chip:
                top = df.loc[rows, on_chip].fillna(0).clip(lower=0).max(axis=1)
                out.loc[rows, key] = np.select([top >= high, top >= mid, top >= low], [3, 2, 1], 0)
    return out


def food_allergy(df: pd.DataFrame) -> pd.Series:
    """1 if a food allergy onset age is given, 0 if 'no food allergy', NaN if not provided."""

    def code(value: object) -> float:
        codes = split_codes(value)
        if NOT_PROVIDED in codes:
            return np.nan
        return 0.0 if codes == {"0"} else 1.0

    return df["Age_of_onsets"].map(code)


def encode_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs of the IgE-profile models: one case per group and intensity (low, moderate,
    high; negative = all zeros), age band, sex and chip."""
    if df["Age"].isna().any():
        raise ValueError("encode_intensity() needs a known age on every row.")
    levels = group_intensity(df)
    features: dict[str, np.ndarray] = {}
    for key in TARGET_GROUPS:
        for level in (1, 2, 3):
            features[f"int_{key}_{level}"] = (levels[key] == level).astype(int).to_numpy()
    bands = df["Age"].map(age_band).to_numpy()
    for i in range(len(AGE_BINS)):
        features[f"age_{i}"] = (bands == i).astype(int)
    features["male"] = (df["Gender"] == "1").astype(int).to_numpy()
    features["female"] = (df["Gender"] == "0").astype(int).to_numpy()
    for chip in ("ISAC_V1", "ISAC_V2", "ALEX"):
        features[f"chip_{chip}"] = (df["Chip_Type"] == chip).astype(int).to_numpy()
    return pd.DataFrame(features, index=df.index)
