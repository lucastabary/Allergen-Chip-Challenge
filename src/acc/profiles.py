"""IgE-profile analyses: allergic profiles, pathology prediction, severity factors.

Answers the three parts of the challenge question:

1. `allergic_profiles()`: groups of patients built from their IgE profile (the 18 allergen
   groups, positive or not and at which intensity, on their own chip), each described by its
   allergens and by its pathologies, compared with the non-sensitized profile at equal
   centre, age and sex.
2. `severity_factors()`: which sensitizations go with treatment and severity of asthma,
   atopic dermatitis and rhinitis, one group at a time, all groups together, and by intensity.
3. `fit_profile_models()`: probability of each pathology from the IgE profile, scored in the
   same centres (nested CV) and in a new centre (leave one centre out).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import multipletests

from acc.features import (
    AGE_BINS,
    INTENSITY_LEVELS,
    SEVERITY_TARGETS,
    TARGET_GROUPS,
    age_band,
    allergen_group,
    centre_of,
    common_allergens,
    encode_intensity,
    food_allergy,
    group_intensity,
    n_clinical_known,
    positivity,
    severity_degrees,
)
from acc.modeling import (
    CHIP_FEATURES,
    CHIPS,
    SEED,
    _always_positive_f1,
    _coefficients,
    _fit_with_threshold,
    evaluate_group,
    evaluate_new_centre,
)

N_PROFILES = 8

# key -> (French label, source, minimum degree); source "food" = declared food allergy
OUTCOMES: dict[str, tuple[str, str, int]] = {
    "asthma_treated": ("Asthme traité", "asthma", 1),
    "asthma_severe": ("Asthme sévère", "asthma", 3),
    "atopic_dermatitis_treated": ("Dermatite atopique traitée", "atopic_dermatitis", 1),
    "atopic_dermatitis_severe": ("Dermatite atopique sévère", "atopic_dermatitis", 3),
    "rhinitis_treated": ("Rhinite traitée", "rhinitis", 1),
    "rhinitis_severe": ("Rhinite sévère", "rhinitis", 3),
    "food": ("Allergie alimentaire déclarée", "food", 1),
}


def outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per OUTCOMES entry, NaN when the underlying answer is not provided."""
    degrees = severity_degrees(df)
    food = food_allergy(df)
    out = {}
    for key, (_, source, level) in OUTCOMES.items():
        values = food if source == "food" else degrees[source]
        out[key] = (values >= level).astype(float).where(values.notna())
    return pd.DataFrame(out, index=df.index)


def _covariates(df: pd.DataFrame, centres: pd.Series) -> pd.DataFrame:
    """Age band (6-11 years as reference), sex and centre (fixed effects)."""
    bands = df["Age"].map(age_band)
    ages = pd.DataFrame(
        {f"age_{i}": (bands == i).astype(int) for i in range(len(AGE_BINS)) if i != 2},
        index=df.index,
    )
    ages["male"] = (df["Gender"] == "1").astype(int)
    return pd.concat(
        [ages, pd.get_dummies(centres, prefix="centre", drop_first=True).astype(int)], axis=1
    )


def _logit(y: pd.Series, design: pd.DataFrame, names: list[str]) -> dict[str, dict]:
    """Odds ratios, 95% CI and p-values of the named columns in a logistic regression."""
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    design = design.astype(float)
    design = sm.add_constant(design.loc[:, design.std() > 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = sm.Logit(y.astype(float), design).fit(disp=0, maxiter=300)
    ci = fit.conf_int()
    return {
        n: {
            "or": float(np.exp(fit.params[n])),
            "low": float(np.exp(ci.loc[n, 0])),
            "high": float(np.exp(ci.loc[n, 1])),
            "p": float(fit.pvalues[n]),
        }
        for n in names
        if n in fit.params
    }


def _varied_centres(y: pd.Series, centres: pd.Series) -> pd.Index:
    """Rows of the centres where the outcome varies (the others carry no comparison)."""
    varied = y.groupby(centres.loc[y.index]).transform(lambda v: 0 < v.mean() < 1)
    return y.index[varied.to_numpy()]


def _round_or(r: dict) -> dict:
    out = {k: round(r[k], 3) for k in ("or", "low", "high")}
    out.update({k: float(f"{r[k]:.3g}") for k in ("p", "q") if k in r})
    return out


# ---------------------------------------------------------------------------
# 1. Allergic profiles
# ---------------------------------------------------------------------------


def _profile_matrix(levels: pd.DataFrame) -> np.ndarray:
    """Clustering space: the 18 group flags, then the 18 intensities scaled to [0, 1]."""
    return np.hstack([(levels > 0).to_numpy(float), levels.to_numpy(float) / 3.0])


PROFILE_SHORT = {
    "grass": "graminées",
    "tree": "arbres",
    "weed": "herbacées",
    "mites": "acariens",
    "animals": "animaux",
    "molds": "moisissures",
    "insects": "insectes",
    "storage": "protéines de stockage",
    "ltp": "LTP",
    "pr10": "PR-10",
    "plant_other": "extraits végétaux",
    "egg": "œuf",
    "milk": "lait",
    "fish": "poissons",
    "seafood": "crustacés",
    "meat": "viandes",
    "venoms": "venins",
    "latex": "latex",
}


def _profile_name(prevalence: pd.Series, median_groups: float) -> str:
    """Name from the groups most patients of the profile are positive to."""
    if median_groups == 0:
        return "Peu ou pas sensibilisés"
    if median_groups >= 10:
        return "Polysensibilisés"
    top = prevalence[prevalence >= 0.6].sort_values(ascending=False).index[:3]
    if len(top) == 0:
        top = prevalence.sort_values(ascending=False).index[:2]
    names = [PROFILE_SHORT[k] for k in top]
    name = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " et " + names[-1]
    return name[0].upper() + name[1:]


def allergic_profiles(df: pd.DataFrame) -> dict:
    """K-means profiles of the IgE profile, described by allergens and pathologies."""
    levels = group_intensity(df)
    x = _profile_matrix(levels)
    runs = [KMeans(N_PROFILES, n_init=20, random_state=SEED + s).fit(x) for s in range(5)]
    raw = runs[0].labels_
    stability = float(np.mean([adjusted_rand_score(raw, r.labels_) for r in runs[1:]]))
    flags = levels > 0
    n_groups = flags.sum(axis=1)

    # stable order: the non-sensitized profile last, the others by decreasing size
    medians = {c: float(np.median(n_groups[raw == c])) for c in range(N_PROFILES)}
    order = sorted(range(N_PROFILES), key=lambda c: (medians[c] == 0, -(raw == c).sum()))
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[c] for c in raw])
    centroids = runs[0].cluster_centers_[order]
    reference = N_PROFILES - 1 if medians[order[-1]] == 0 else None

    outcomes = outcome_table(df)
    centres = centre_of(df)
    known_age = df["Age"].notna()
    profiles = []
    for p in range(N_PROFILES):
        members = labels == p
        sub = df[members]
        prevalence = flags[members].mean()
        positives = int((levels[members] > 0).to_numpy().sum())
        strong = int((levels[members] == 3).to_numpy().sum())
        profiles.append(
            {
                "name": _profile_name(prevalence, medians[order[p]]),
                "n": int(members.sum()),
                "age_median": float(sub["Age"].median()),
                "male": round(float((sub["Gender"] == "1").mean()), 3),
                "chips": [round(float((sub["Chip_Type"] == c).mean()), 3) for c in CHIPS],
                "median_groups": medians[order[p]],
                "prevalence": {k: round(float(prevalence[k]), 3) for k in TARGET_GROUPS},
                "strong_share": round(strong / positives, 3) if positives else 0.0,
                "outcomes": {
                    k: {
                        "rate": round(float(outcomes.loc[members, k].mean()), 3)
                        if outcomes.loc[members, k].notna().any()
                        else None,
                        "n": int(outcomes.loc[members, k].notna().sum()),
                    }
                    for k in OUTCOMES
                },
            }
        )

    # Odds ratio of each outcome for each profile against the reference profile,
    # at equal centre, age and sex; BH correction over all profile x outcome tests.
    if reference is not None:
        dummies = pd.DataFrame(
            {
                f"profile_{p}": (labels == p).astype(int)
                for p in range(N_PROFILES)
                if p != reference
            },
            index=df.index,
        )
        tests = []
        for key in OUTCOMES:
            rows = outcomes.index[outcomes[key].notna() & known_age]
            y = outcomes.loc[rows, key]
            rows = _varied_centres(y, centres)
            design = pd.concat(
                [dummies.loc[rows], _covariates(df.loc[rows], centres.loc[rows])], axis=1
            )
            fitted = _logit(y.loc[rows], design, list(dummies.columns))
            for name, r in fitted.items():
                tests.append((int(name.split("_")[1]), key, r))
        q = multipletests([t[2]["p"] for t in tests], method="fdr_bh")[1]
        for (p, key, r), q_value in zip(tests, q, strict=True):
            profiles[p]["outcomes"][key]["vs_reference"] = _round_or({**r, "q": q_value})

    return {
        "k": N_PROFILES,
        "stability_ari": round(stability, 3),
        "reference": reference,
        "centroids": [[round(float(v), 4) for v in row] for row in centroids],
        "profiles": profiles,
        "allergens": profile_allergen_rates(df, labels),
    }


def profile_allergen_rates(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Positive rate of each common allergen in each profile and in the whole cohort.

    Restricted to the 91 allergens measured on all three chips, so every patient counts for
    every bar; columns are ordered by group (TARGET_GROUPS order), then by name. The page draws
    each profile's rate minus the cohort rate, like the PCA loadings of notebook 02.
    """
    order = list(TARGET_GROUPS)
    columns = sorted(
        (c for c in common_allergens(df) if allergen_group(c) is not None),
        key=lambda c: (order.index(allergen_group(c)), c),
    )
    positive = positivity(df, columns)
    return {
        "columns": columns,
        "groups": [allergen_group(c) for c in columns],
        "cohort": [round(float(v), 4) for v in positive.mean()],
        "by_profile": [
            [round(float(v), 4) for v in positive[labels == p].mean()] for p in range(N_PROFILES)
        ],
    }


# ---------------------------------------------------------------------------
# 2. Severity factors
# ---------------------------------------------------------------------------


def severity_factors(df: pd.DataFrame) -> list[dict]:
    """Which sensitizations go with treatment and severity, within the same centre.

    For each disease and outcome (treated: degree >= 1; severe: degree 3), three analyses,
    all adjusted for age band, sex and centre (fixed effects), on the centres where the
    outcome varies:
    - one logistic regression per group (positive vs negative), plus the number of positive
      groups (polysensitization), BH-corrected over the 18 groups;
    - one regression with the 18 groups together, so that groups that often go together
      (mites and animals) are separated; 95% CI by bootstrap (200 resamples, L2 C = 1);
    - one regression per group with its intensity (low, moderate, high vs negative).
    """
    data = df[df["Age"].notna()]
    levels = group_intensity(data)
    flags = (levels > 0).astype(int)
    degrees = severity_degrees(data)
    centres = centre_of(data)
    groups = list(TARGET_GROUPS)
    out = []
    for key in SEVERITY_TARGETS:
        for level, outcome in ((1, "treated"), (3, "severe")):
            known = degrees[key].notna()
            y = (degrees[key][known] >= level).astype(int)
            rows = _varied_centres(y, centres)
            y = y.loc[rows]
            cov = _covariates(data.loc[rows], centres.loc[rows])

            single = []
            for g in groups:
                r = _logit(
                    y,
                    pd.concat([cov, flags.loc[rows, [g]].rename(columns={g: "x"})], axis=1),
                    ["x"],
                )
                single.append({"exposure": f"grp_{g}", **r["x"]})
            q = multipletests([r["p"] for r in single], method="fdr_bh")[1]
            single = [
                {"exposure": r["exposure"], **_round_or({**r, "q": qv})}
                for r, qv in zip(single, q, strict=True)
            ]
            poly = _logit(
                y, pd.concat([cov, flags.loc[rows].sum(axis=1).rename("x")], axis=1), ["x"]
            )["x"]
            single.append({"exposure": "n_groups", **_round_or({**poly, "q": poly["p"]})})

            joint = _joint_odds_ratios(y, flags.loc[rows, groups], cov)

            dose = []
            for g in groups:
                dummies = pd.DataFrame(
                    {f"l{lv}": (levels.loc[rows, g] == lv).astype(int) for lv in (1, 2, 3)},
                    index=rows,
                )
                r = _logit(y, pd.concat([cov, dummies], axis=1), list(dummies.columns))
                dose.append(
                    {
                        "exposure": f"grp_{g}",
                        "levels": [
                            _round_or(r[f"l{lv}"]) | {"n": int(dummies[f"l{lv}"].sum())}
                            if f"l{lv}" in r
                            else None
                            for lv in (1, 2, 3)
                        ],
                    }
                )
            out.append(
                {
                    "target": key,
                    "outcome": outcome,
                    "n": int(len(y)),
                    "n_positive": int(y.sum()),
                    "n_centres": int(centres.loc[rows].nunique()),
                    "results": single,
                    "joint": joint,
                    "dose": dose,
                }
            )
    return out


def _joint_odds_ratios(
    y: pd.Series, flags: pd.DataFrame, cov: pd.DataFrame, n_boot: int = 200
) -> list[dict]:
    """All 18 groups in one penalized logistic regression; bootstrap percentile CI."""
    x = pd.concat([flags, cov], axis=1).to_numpy(float)
    yy = y.to_numpy()
    k = flags.shape[1]

    def fit(rows: np.ndarray) -> np.ndarray:
        return LogisticRegression(C=1.0, max_iter=4000).fit(x[rows], yy[rows]).coef_[0][:k]

    point = fit(np.arange(len(yy)))
    rng = np.random.default_rng(SEED)
    boots = np.array([fit(rng.integers(0, len(yy), len(yy))) for _ in range(n_boot)])
    low, high = np.percentile(boots, [2.5, 97.5], axis=0)
    return [
        {
            "exposure": f"grp_{g}",
            "or": round(float(np.exp(point[i])), 3),
            "low": round(float(np.exp(low[i])), 3),
            "high": round(float(np.exp(high[i])), 3),
            "significant": bool(low[i] > 0 or high[i] < 0),
        }
        for i, g in enumerate(flags.columns)
    ]


# ---------------------------------------------------------------------------
# 3. Pathologies from the IgE profile
# ---------------------------------------------------------------------------


def _fit_outcome(
    key: str, features: pd.DataFrame, chips: np.ndarray, centres: np.ndarray, y_all: pd.Series
) -> dict:
    defined = y_all.notna().to_numpy()
    base_cols = [c for c in features.columns if not c.startswith("chip_")]
    age_sex = [
        i for i, c in enumerate(base_cols) if c.startswith("age_") or c in ("male", "female")
    ]
    x = features[base_cols].to_numpy()[defined]
    x_chip = features[CHIP_FEATURES].to_numpy()[defined]
    chip, centre = chips[defined], centres[defined]
    y = y_all.to_numpy()[defined].astype(int)
    x_pooled = np.hstack([x, x_chip])
    scores = evaluate_group(x, x_chip, y, chip, specific=False)
    baseline = evaluate_group(x[:, age_sex], x_chip, y, chip, n_repeats=2, specific=False)
    model, threshold = _fit_with_threshold(x_pooled, y, SEED)
    intercept, coef = _coefficients(model)
    return {
        "key": key,
        "label": OUTCOMES[key][0],
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 3),
        "intercept": intercept,
        "coef": coef,
        "threshold": round(threshold, 3),
        **scores["pooled"],
        "f1_by_chip": {c: scores[f"pooled_{c}"]["f1"] for c in CHIPS if f"pooled_{c}" in scores},
        "f1_age_sex": baseline["pooled"]["f1"],
        "auc_age_sex": baseline["pooled"]["auc"],
        "f1_always_positive": _always_positive_f1(y),
        "new_centre": evaluate_new_centre(x_pooled, y, centre),
        "new_centre_age_sex": evaluate_new_centre(np.hstack([x[:, age_sex], x_chip]), y, centre),
    }


def fit_profile_models(df: pd.DataFrame, n_jobs: int = -1) -> dict:
    """One logistic regression per pathology, from group intensities, age, sex and chip."""
    train = df[df["Age"].notna() & (n_clinical_known(df) >= 1)]
    features = encode_intensity(train)
    outcomes = outcome_table(train)
    chips = train["Chip_Type"].to_numpy()
    centres = centre_of(train).to_numpy()
    models = Parallel(n_jobs=n_jobs)(
        delayed(_fit_outcome)(key, features, chips, centres, outcomes[key]) for key in OUTCOMES
    )
    names = [c for c in features.columns if not c.startswith("chip_")] + CHIP_FEATURES
    levels = group_intensity(train)
    return {
        "features": names,
        "intensity_levels": INTENSITY_LEVELS,
        "outcomes": models,
        # training patients, for "load a real patient"
        "patients": {
            "age": [round(float(a), 1) for a in train["Age"]],
            "sex": [{"1": 1, "0": 0}.get(g, -1) for g in train["Gender"]],
            "chip": [CHIPS.index(c) for c in train["Chip_Type"]],
            "levels": levels.to_numpy().tolist(),
            "outcomes": {
                k: [None if pd.isna(v) else int(v) for v in outcomes[k]] for k in OUTCOMES
            },
        },
    }
