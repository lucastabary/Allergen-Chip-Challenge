"""Build the simulator page: sensitization profile and severity degree.

Usage:
    uv run acc-atlas            # writes reports/atlas/index.html

For each of the 18 allergen groups of `acc.features.TARGET_GROUPS`, two kinds of
logistic regression are trained on the clinical answers (age, sex, symptoms, treatments...):

- a pooled model on the patients of the three chips, with the chip as an extra input;
- one chip-specific model per chip, on that chip's patients only.

A patient's target always uses every allergen of the group measured on their own chip,
so ALEX² patients are not reduced to the 91 allergens shared with ISAC. Both kinds of
model are evaluated on the same held-out patients (nested cross-validation), so their
F1 scores can be compared overall and chip by chip.

Severity (degree 0 to 3, from the treatment step) of asthma, atopic dermatitis and rhinitis:
centre-adjusted odds ratios link each allergen group to the severity of each disease, and a
multinomial model estimates the degree of the diseases the user left unanswered.

The page is self-contained: data, coefficients and scores are inlined as JSON and the
predictions are computed in the browser.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

from acc.allergens import FAMILIES, display_name, family_of, sort_by_family
from acc.data import PROJECT_ROOT, load_raw, split_columns
from acc.features import (
    AGE_BINS,
    CLINICAL_FIELDS,
    EXCLUDED_ALLERGENS,
    POSITIVITY_THRESHOLD,
    SEVERITY_LEVELS,
    SEVERITY_TARGETS,
    TARGET_GROUPS,
    allergen_group,
    centre_of,
    clinical_feature_labels,
    encode_clinical,
    group_columns,
    measured_by_chip,
    n_clinical_known,
    sensitization_targets,
    severity_degrees,
    split_codes,
)
from acc.modeling import (
    C_REGULARIZATION,
    CHIP_FEATURES,
    CHIP_LABELS,
    CHIPS,
    MIN_CASES,
    N_REPEATS,
    SEED,
    _always_positive_f1,
    _best_threshold,
    _coefficients,
    _fit_with_threshold,
    _model,
    evaluate_group,
    evaluate_new_centre,
)
from acc.profiles import (
    OUTCOMES,
    PROFILE_SHORT,
    allergic_profiles,
    fit_profile_models,
    severity_factors,
)

OUTPUT_FILE = PROJECT_ROOT / "reports" / "atlas" / "index.html"


# French names of the allergen source families (acc.allergens.FAMILIES order).
FAMILY_LABELS_FR = {
    "grass pollen": "Pollens de graminées",
    "tree pollen": "Pollens d'arbres",
    "weed pollen": "Pollens d'herbacées",
    "mites": "Acariens",
    "insects": "Insectes",
    "animal dander": "Animaux",
    "molds / yeasts": "Moisissures et levures",
    "venoms": "Venins",
    "latex": "Latex",
    "plant food": "Aliments végétaux",
    "animal food": "Aliments d'origine animale",
    "other": "Autres",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_group(
    key: str, features: pd.DataFrame, chips: np.ndarray, centres: np.ndarray, target: pd.Series
) -> dict:
    """Evaluate then fit the final pooled and chip-specific models of one group."""
    defined = target.notna().to_numpy()
    x = features.to_numpy()[defined]
    chip = chips[defined]
    x_chip = np.stack([(chip == c).astype(int) for c in CHIPS], axis=1)
    y = target.to_numpy()[defined].astype(int)
    scores = evaluate_group(x, x_chip, y, chip)
    new_centre = evaluate_new_centre(np.hstack([x, x_chip]), y, centres[defined])

    age_sex = [i for i, name in enumerate(features.columns) if name.startswith("age_")]
    age_sex += [list(features.columns).index(n) for n in ("male", "female")]
    baseline = evaluate_group(x[:, age_sex], x_chip, y, chip, n_repeats=2)

    pooled_model, pooled_threshold = _fit_with_threshold(np.hstack([x, x_chip]), y, SEED)
    intercept, coef = _coefficients(pooled_model)
    result = {
        "key": key,
        "label": TARGET_GROUPS[key][0],
        "category": TARGET_GROUPS[key][1],
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 3),
        "pooled": {
            "intercept": intercept,
            "coef": coef,
            "threshold": round(pooled_threshold, 3),
            **scores["pooled"],
            "f1_age_sex": baseline["pooled"]["f1"],
            "f1_always_positive": _always_positive_f1(y),
            "f1_new_centre": new_centre["f1"],
        },
        "chips": {},
    }
    for c in CHIPS:
        rows = chip == c
        if not rows.any():
            result["chips"][c] = None  # the chip measures no allergen of this group
            continue
        entry = {
            "n": int(rows.sum()),
            "n_positive": int(y[rows].sum()),
            "prevalence": round(float(y[rows].mean()), 3),
            "f1_always_positive": _always_positive_f1(y[rows]),
            "pooled": {**scores[f"pooled_{c}"], "f1_age_sex": baseline[f"pooled_{c}"]["f1"]},
            "specific": None,
        }
        if f"specific_{c}" in scores:
            model, threshold = _fit_with_threshold(x[rows], y[rows], SEED)
            intercept, coef = _coefficients(model)
            entry["specific"] = {
                "intercept": intercept,
                "coef": coef,
                "threshold": round(threshold, 3),
                **scores[f"specific_{c}"],
                "f1_age_sex": baseline[f"specific_{c}"]["f1"],
            }
        result["chips"][c] = entry
    return result


def training_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Patients with a known age and at least one clinical answer provided."""
    return df[df["Age"].notna() & (n_clinical_known(df) >= 1)]


def fit_models(df: pd.DataFrame, n_jobs: int = -1) -> dict:
    """All groups, in parallel. Returns the `model` part of the page payload."""
    train = training_patients(df)
    features = encode_clinical(train)
    targets = sensitization_targets(train)
    chips = train["Chip_Type"].to_numpy()
    centres = centre_of(train).to_numpy()
    groups = Parallel(n_jobs=n_jobs)(
        delayed(fit_group)(key, features, chips, centres, targets[key]) for key in TARGET_GROUPS
    )
    counts = features.sum()
    labels = clinical_feature_labels()
    return {
        "n_train": int(len(train)),
        "n_train_by_chip": {c: int((chips == c).sum()) for c in CHIPS},
        "features": list(features.columns),
        "feature_labels": labels,
        "feature_counts": {name: int(counts[name]) for name in features.columns},
        "chip_features": CHIP_FEATURES,
        "c_regularization": C_REGULARIZATION,
        "min_cases": MIN_CASES,
        "n_repeats": N_REPEATS,
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# Severity: sensitization profile -> treatment-step degree
# ---------------------------------------------------------------------------


# Clinical field of each disease's own treatment: left out of that disease's inputs.
DEGREE_FIELD = {"asthma": "asth", "atopic_dermatitis": "ad", "rhinitis": "rhin"}


def _multinomial() -> LogisticRegression:
    return LogisticRegression(C=C_REGULARIZATION, max_iter=4000)


def _macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    """F1 averaged over the 4 degrees, for the most probable degree."""
    return float(
        f1_score(y, proba.argmax(axis=1), average="macro", labels=[0, 1, 2, 3], zero_division=0)
    )


def evaluate_degree(
    x: np.ndarray, y: np.ndarray, centres: np.ndarray, n_repeats: int = N_REPEATS
) -> dict:
    """Macro F1 of the most probable degree: same centres (5 x 5-fold CV) and new centre (LOCO)."""
    runs = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED + repeat)
        runs.append(
            _macro_f1(y, cross_val_predict(_multinomial(), x, y, cv=cv, method="predict_proba"))
        )
    loco = cross_val_predict(
        _multinomial(), x, y, cv=LeaveOneGroupOut(), groups=centres, method="predict_proba"
    )
    return {
        "f1": round(float(np.mean(runs)), 3),
        "f1_sd": round(float(np.std(runs)), 3),
        "f1_new_centre": round(_macro_f1(y, loco), 3),
    }


def fit_degree_target(
    key: str, features: pd.DataFrame, chips: np.ndarray, centres: np.ndarray, degree: pd.Series
) -> dict:
    """Treatment degree (0-3) of one disease from the OTHER clinical answers, age, sex and chip.

    One multinomial logistic regression: it gives the four degree probabilities at once,
    always coherent, and the page shows the most probable degree. Used for the diseases the
    user left "not provided" in the form, so it never predicts an answer it was given.
    """
    known = degree.notna().to_numpy()
    own = f"{DEGREE_FIELD[key]}_"
    cols = [c for c in features.columns if not c.startswith(own)]
    age_sex = [c for c in cols if c.startswith("age_") or c in ("male", "female")]
    x_chip = np.stack([(chips == c).astype(int) for c in CHIPS], axis=1)[known]
    y = degree.to_numpy()[known].astype(int)
    x = np.hstack([features[cols].to_numpy()[known], x_chip])
    x_base = np.hstack([features[age_sex].to_numpy()[known], x_chip])
    scores = evaluate_degree(x, y, centres[known])
    baseline = evaluate_degree(x_base, y, centres[known], n_repeats=2)
    majority = np.full_like(y, np.bincount(y).argmax())
    model = _multinomial().fit(x, y)
    label, column, mapping = SEVERITY_TARGETS[key]
    return {
        "key": key,
        "label": label,
        "field": DEGREE_FIELD[key],
        "codes": dict(mapping),
        "n": int(len(y)),
        "counts": [int((y == k).sum()) for k in range(4)],
        "features": cols + CHIP_FEATURES,
        "intercept": [round(float(v), 4) for v in model.intercept_],
        "coef": [[round(float(v), 4) for v in row] for row in model.coef_],
        **scores,
        "f1_age_sex": baseline["f1"],
        "f1_age_sex_new_centre": baseline["f1_new_centre"],
        "f1_majority": round(
            float(f1_score(y, majority, average="macro", labels=[0, 1, 2, 3], zero_division=0)), 3
        ),
    }


def fit_severity(df: pd.DataFrame, n_jobs: int = -1) -> dict:
    """Degree models for unanswered diseases, and centre-adjusted severity factors."""
    train = training_patients(df)
    features = encode_clinical(train)
    degrees = severity_degrees(train)
    chips = train["Chip_Type"].to_numpy()
    centres = centre_of(train).to_numpy()
    models = Parallel(n_jobs=n_jobs)(
        delayed(fit_degree_target)(key, features, chips, centres, degrees[key])
        for key in SEVERITY_TARGETS
    )
    return {
        "levels": SEVERITY_LEVELS,
        "degree_models": models,
        "factors": severity_factors(df),
    }


# ---------------------------------------------------------------------------
# What does not raise the F1 of the sensitization models
# ---------------------------------------------------------------------------

CEILING_VARIANTS = [
    ("base", "Modèle retenu : réponses cliniques et puce, C = 0,3"),
    ("region", "Ajout de la région de résidence"),
    ("region_month", "Ajout de la région et du mois de prélèvement"),
    ("c_strong", "Régularisation plus forte (C = 0,05)"),
    ("c_weak", "Régularisation plus faible (C = 1)"),
    ("boosting", "Gradient boosting (arbres de décision), au lieu de la régression logistique"),
]


def _ceiling_group(x_sets: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {
        "base": ("base", _model()),
        "region": ("region", _model()),
        "region_month": ("region_month", _model()),
        "c_strong": ("region_month", LogisticRegression(C=0.05, max_iter=3000)),
        "c_weak": ("region_month", LogisticRegression(C=1.0, max_iter=3000)),
        "boosting": (
            "region_month",
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200),
        ),
    }
    out = {}
    for name, (inputs, model) in models.items():
        proba = cross_val_predict(model, x_sets[inputs], y, cv=cv, method="predict_proba")[:, 1]
        out[name] = round(f1_score(y, proba >= _best_threshold(y, proba)), 3)
    return out


def ceiling_experiments(df: pd.DataFrame, n_jobs: int = -1) -> dict:
    """F1 of alternative inputs and models, all on the same 5 folds.

    Thresholds are tuned on the out-of-fold predictions themselves (no nesting), which
    flatters every variant by the same small amount: only the differences matter here.
    """
    train = training_patients(df)
    base = pd.concat(
        [encode_clinical(train), pd.get_dummies(train["Chip_Type"]).astype(int)], axis=1
    )
    region = pd.get_dummies(train["French_Region"], prefix="region").astype(int)
    month = pd.to_numeric(train["Blood_Month_sample"], errors="coerce")
    known = month.notna().astype(int)
    season = pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month.fillna(0) / 12) * known,
            "month_cos": np.cos(2 * np.pi * month.fillna(0) / 12) * known,
        },
        index=train.index,
    )
    sets = {
        "base": base,
        "region": pd.concat([base, region], axis=1),
        "region_month": pd.concat([base, region, season], axis=1),
    }
    targets = sensitization_targets(train)
    keys = list(TARGET_GROUPS)

    def run(key: str) -> dict[str, float]:
        defined = targets[key].notna().to_numpy()
        y = targets[key].to_numpy()[defined].astype(int)
        return _ceiling_group({k: v.to_numpy()[defined] for k, v in sets.items()}, y)

    results = Parallel(n_jobs=n_jobs)(delayed(run)(key) for key in keys)
    return {
        "variants": [
            {
                "key": name,
                "label": label,
                "f1_mean": round(float(np.mean([r[name] for r in results])), 3),
                "f1_by_group": {key: r[name] for key, r in zip(keys, results, strict=True)},
            }
            for name, label in CEILING_VARIANTS
        ]
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def allergen_table(df: pd.DataFrame) -> list[dict]:
    """The 241 IgE columns: name, family, group (or exclusion reason), chips, positive rates."""
    _, ige = split_columns(df)
    measured = measured_by_chip(df)
    rows = []
    for col in sort_by_family(ige):
        rates = []
        for chip in CHIPS:
            if measured.loc[chip, col]:
                values = df.loc[df["Chip_Type"] == chip, col]
                rates.append(round(float((values >= POSITIVITY_THRESHOLD).mean()), 3))
            else:
                rates.append(None)
        rows.append(
            {
                "column": col,
                "name": display_name(col),
                "family": FAMILY_LABELS_FR[family_of(col)],
                "group": allergen_group(col),
                "excluded": EXCLUDED_ALLERGENS.get(col),
                "rates": rates,
            }
        )
    return rows


def build_payload(df: pd.DataFrame) -> dict:
    """Everything the page needs, as a JSON-serializable dict."""
    _, ige = split_columns(df)
    measured = measured_by_chip(df)
    train = training_patients(df)
    targets = sensitization_targets(train)
    groups = group_columns(ige)

    def target_list(key: str) -> list:
        return [None if pd.isna(v) else int(v) for v in targets[key]]

    return {
        "meta": {
            "n_patients": int(len(df)),
            "n_allergens": len(ige),
            "n_groups": len(TARGET_GROUPS),
            "n_excluded": len(EXCLUDED_ALLERGENS),
            "threshold": POSITIVITY_THRESHOLD,
            "chips": CHIPS,
            "chip_labels": CHIP_LABELS,
            "families": [FAMILY_LABELS_FR[f] for f in FAMILIES],
            "age_bins": [list(b) for b in AGE_BINS],
            "group_sizes": {
                key: [int(sum(measured.loc[chip, c] for c in cols)) for chip in CHIPS]
                for key, cols in groups.items()
            },
        },
        "fields": [
            {
                "key": f.key,
                "label": f.label,
                "options": f.options,
                "multiple": f.multiple,
                "model_codes": list(f.model_codes),
            }
            for f in CLINICAL_FIELDS
        ],
        "allergens": allergen_table(df),
        # Training patients only, for the "closest real patients" lookup.
        "patients": {
            "age": [round(float(a), 1) for a in train["Age"]],
            "sex": [{"1": 1, "0": 0}.get(g, -1) for g in train["Gender"]],
            "chip": [CHIPS.index(c) for c in train["Chip_Type"]],
            # Clinical codes as text, e.g. "6,7"; "9" = not provided.
            "clinical": {
                f.key: [",".join(sorted(split_codes(v), key=int)) for v in train[f.column]]
                for f in CLINICAL_FIELDS
            },
            "targets": {key: target_list(key) for key in TARGET_GROUPS},
        },
        "model": fit_models(df),
        "profiles": allergic_profiles(df),
        "ige_models": fit_profile_models(df),
        "outcome_labels": {k: v[0] for k, v in OUTCOMES.items()},
        "group_short": PROFILE_SHORT,
        "severity": fit_severity(df),
        "ceiling": ceiling_experiments(df),
    }


def render(payload: dict, output: Path = OUTPUT_FILE) -> Path:
    """Inline the payload in the HTML template and write the page."""
    template = resources.files("acc").joinpath("resources/atlas_template.html").read_text("utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    marker = "/*__ATLAS_DATA__*/null"
    if marker not in template:
        raise RuntimeError("Data marker not found in atlas_template.html.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(marker, data), encoding="utf-8")
    return output


def main() -> None:
    payload = build_payload(load_raw())
    path = render(payload)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    for t in payload["severity"]["degree_models"]:
        print(t["key"], t["f1"], t["f1_new_centre"], t["f1_majority"])
    print({v["key"]: v["f1_mean"] for v in payload["ceiling"]["variants"]})
    for g in payload["model"]["groups"]:
        per_chip = {
            c: (e["pooled"]["f1"], e["specific"]["f1"] if e["specific"] else None)
            for c, e in g["chips"].items()
            if e
        }
        print(f"{g['key']:12s} pooled F1 {g['pooled']['f1']:.2f}  per chip {per_chip}")


if __name__ == "__main__":
    main()
