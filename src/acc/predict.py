"""Predict clinical outcomes from IgE profiles, with repeated cross-validation.

The question is not "how high can the AUC go" but "does the IgE profile carry information about a
clinical outcome beyond what age, sex, chip and region already tell". So every target is evaluated
with nested feature sets (demographics alone, demographics + an IgE summary, demographics + the full
IgE panel) on the same folds, and the gains are tested with paired, corrected tests.

Usage:
    python -m acc.predict run --out runs/v1 --repeats 10 [--targets a,b] [--models logreg,hgb]
    python -m acc.predict report runs/v1
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from acc.allergens import family_of
from acc.data import DATA_FILE, download, load_raw, split_columns

UNKNOWN = 9  # "Non renseigné" / "sans objet" in the clinical columns
POSITIVITY = 0.3  # kUA/L (ALEX) or ISU (ISAC)
CHIPS = ["ALEX", "ISAC_V1", "ISAC_V2"]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def parse_codes(value: object) -> frozenset[int] | None:
    """Multi-coded value -> set of codes ("1,3", "6, 7" occur); unknown (9) or NaN -> None."""
    if pd.isna(value):
        return None
    codes = frozenset(int(float(x)) for x in str(value).replace(" ", "").split(",") if x)
    if not codes or codes == {UNKNOWN}:
        return None
    return codes


@dataclass(frozen=True)
class Target:
    name: str
    column: str
    description: str
    positive: Callable[[frozenset[int]], bool]
    # restrict to patients whose codes satisfy this (e.g. "among treated patients")
    subset: Callable[[frozenset[int]], bool] | None = None
    # demographic features that would leak the target (e.g. Age for "adult")
    exclude: tuple[str, ...] = field(default_factory=tuple)
    control: str = ""  # "positive" / "negative" control, "" for a clinical target

    def labels(self, df: pd.DataFrame) -> pd.Series:
        """1 / 0 / NaN (unknown or outside the subset)."""
        codes = df[self.column].map(parse_codes)

        def label(c: frozenset[int] | None) -> float:
            if c is None or (self.subset is not None and not self.subset(c)):
                return np.nan
            return float(self.positive(c))

        return codes.map(label).astype(float)


def _any(*values: int) -> Callable[[frozenset[int]], bool]:
    return lambda c: bool(c & set(values))


def _not_none(c: frozenset[int]) -> bool:  # "treated": any code other than 0 = none
    return c != {0}


ANAPHYLAXIS_ONSET = (
    1,
    2,
    3,
    4,
    5,
    6,
)  # Age_of_onsets: age group of the first food anaphylaxis, 0 = none

TARGETS: dict[str, Target] = {
    t.name: t
    for t in [
        Target(
            "food_anaphylaxis",
            "Age_of_onsets",
            "History of food anaphylaxis (Age_of_onsets 1-6 vs 0 = no anaphylaxis)",
            _any(*ANAPHYLAXIS_ONSET),
        ),
        Target(
            "early_food_anaphylaxis",
            "Age_of_onsets",
            "Among food-anaphylaxis patients: first episode before 2 years (1-2) vs later (3-6)",
            lambda c: min(c & set(ANAPHYLAXIS_ONSET)) <= 2,
            subset=_any(*ANAPHYLAXIS_ONSET),
        ),
        Target(
            "skin_symptoms",
            "Skin_Symptoms",
            "Skin symptoms (1 vs 0; 9 = not relevant is excluded)",
            _any(1),
        ),
        Target(
            "asthma_treatment",
            "Treatment_of_athsma",
            "Any asthma treatment (codes 1-8, 10) vs none (0)",
            _not_none,
        ),
        Target(
            "asthma_step3",
            "Treatment_of_athsma",
            "Among treated asthma: GINA step >= 3 (ICS/LABA, biologics: 3-8) vs SABA/low ICS/LTRA",
            _any(3, 4, 5, 6, 7, 8),
            subset=_not_none,
        ),
        Target(
            "rhinitis_treatment",
            "Treatment_of_rhinitis",
            "Any rhinitis treatment (codes 1-4) vs none (0)",
            _not_none,
        ),
        Target(
            "ad_treatment",
            "Treatment_of_atopic_dematitis",
            "Any atopic dermatitis treatment (codes 1-7) vs none (0)",
            _not_none,
        ),
        Target(
            "ad_systemic",
            "Treatment_of_atopic_dematitis",
            "Among treated atopic dermatitis: systemic (ciclosporin, biologic, JAK: 4-7) vs other",
            _any(4, 5, 6, 7),
            subset=_not_none,
        ),
        Target(
            "urban",
            "Rural_or_urban_area",
            "Urban (1) vs rural (0) residence",
            _any(1),
        ),
        Target(
            "adult",
            "Age",
            "Positive control: age >= 18",
            lambda c: True,  # replaced in labels_for()
            exclude=("Age",),
            control="positive",
        ),
        Target(
            "male",
            "Gender",
            "Male sex (not a negative control: sex ratio depends on age in this cohort)",
            _any(1),
            exclude=("Gender",),
        ),
        Target(
            "shuffled",
            "Age_of_onsets",
            "Negative control: food_anaphylaxis labels permuted within chip (AUC must be ~0.5)",
            _any(*ANAPHYLAXIS_ONSET),
            control="negative",
        ),
    ]
}


def labels_for(target: Target, df: pd.DataFrame) -> pd.Series:
    if target.name == "adult":  # Age is numeric, not a code
        return (df["Age"] >= 18).astype(float).where(df["Age"].notna())
    y = target.labels(df)
    if target.name == "shuffled":  # leakage check: any signal left would come from the pipeline
        rng = np.random.default_rng(0)
        known = y.notna()
        for chip in CHIPS:
            m = (known & (df["Chip_Type"] == chip)).to_numpy()
            y.iloc[np.flatnonzero(m)] = rng.permutation(y.to_numpy()[m])
    return y


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def common_panel(df: pd.DataFrame) -> list[str]:
    """IgE columns measured on every chip (91 allergens): comparable across the whole cohort."""
    _, ige = split_columns(df)
    measured = [df.loc[df["Chip_Type"] == chip, ige].notna().mean() > 0.5 for chip in CHIPS]
    return [c for c in ige if all(m[c] for m in measured)]


def demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Age, sex, chip and region (a proxy of the recruiting hospital), one-hot encoded."""
    region = df["French_Region"].fillna("unknown")
    rare = region.map(region.value_counts()) < 50
    region = region.mask(rare, "other")
    out = pd.DataFrame(
        {"Age": df["Age"], "Gender": pd.to_numeric(df["Gender"], errors="coerce")}, index=df.index
    )
    out = out.join(chip_features(df)).join(pd.get_dummies(region, prefix="region", dtype=float))
    return out


def chip_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["Chip_Type"], prefix="chip", dtype=float)


def ige_features(df: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    """log1p IgE of the common panel (NaN = invalid measurement, imputed in the model)."""
    return np.log1p(df[panel]).add_prefix("ige_")


def summary_features(df: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    """Number of positive allergens per source family, total count and maximum level."""
    pos = df[panel] >= POSITIVITY
    families = pd.Series({c: family_of(c) for c in panel})
    out = pd.DataFrame(index=df.index)
    for fam, cols in families.groupby(families).groups.items():
        out[f"npos_{fam}"] = pos[list(cols)].sum(axis=1)
    out["npos_total"] = pos.sum(axis=1)
    out["log_max_ige"] = np.log1p(df[panel].max(axis=1))
    return out


FEATURE_SETS = ["demo", "summary", "ige", "demo+summary", "demo+ige"]


def build_features(df: pd.DataFrame, feature_set: str, panel: list[str]) -> pd.DataFrame:
    """Chip is in every set: the chips do not measure on the same scale."""
    parts = []
    for part in feature_set.split("+"):
        if part == "demo":
            parts.append(demographic_features(df))
        elif part == "summary":
            parts.append(summary_features(df, panel))
        elif part == "ige":
            parts.append(ige_features(df, panel))
        else:
            raise ValueError(feature_set)
    if "demo" not in feature_set:
        parts.append(chip_features(df))
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Models and cross-validation
# ---------------------------------------------------------------------------


def make_model(name: str, seed: int):
    if name == "logreg":  # L2, C chosen by an inner 3-fold CV on the training fold only
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegressionCV(
                Cs=np.logspace(-3, 1, 9), cv=3, scoring="roc_auc", max_iter=5000, n_jobs=1
            ),
        )
    if name == "hgb":  # fixed, conservative settings (no tuning in the first pass)
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(name)


MODELS = ["logreg", "hgb"]


def _fold_metrics(y: np.ndarray, p: np.ndarray, chip: np.ndarray) -> dict[str, float]:
    out = {
        "auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "prevalence": float(y.mean()),
    }
    # AUC within each chip: removes what the chip (and its hospitals, years) explains on its own
    for c in CHIPS:
        m = chip == c
        out[f"auc_{c}"] = roc_auc_score(y[m], p[m]) if m.any() and len(set(y[m])) == 2 else np.nan
    return out


def _run_fold(model_name, X, y, chip, train, test, seed) -> tuple[dict[str, float], np.ndarray]:
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = clone(make_model(model_name, seed))
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
    return _fold_metrics(y[test], p, chip[test]), p


def _run_cell(target_name, feature_set, model_name, X, y, chip, splits, keep_oof):
    """One (target, feature set, model): every fold of every repeat."""
    rows, oof = [], []
    for repeat, fold, train, test in splits:
        t0 = time.time()
        metrics, p = _run_fold(model_name, X, y, chip, train, test, seed=repeat)
        rows.append(
            dict(
                target=target_name,
                features=feature_set,
                model=model_name,
                repeat=repeat,
                fold=fold,
                n_train=len(train),
                n_test=len(test),
                seconds=time.time() - t0,
                **metrics,
            )
        )
        if keep_oof and repeat == 0:
            oof.append(pd.DataFrame({"row": test, "p": p}))
    oof_df = pd.concat(oof) if oof else None
    if oof_df is not None:
        oof_df = oof_df.assign(target=target_name, features=feature_set, model=model_name)
    return rows, oof_df


def run(
    out: Path,
    targets: list[str],
    models: list[str],
    feature_sets: list[str],
    repeats: int,
    folds: int,
    jobs: int,
) -> pd.DataFrame:
    if not DATA_FILE.exists():
        download()
    df = load_raw()
    panel = common_panel(df)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} patients, common panel of {len(panel)} allergens", flush=True)

    tasks, label_table = [], {}
    for name in targets:
        target = TARGETS[name]
        y_all = labels_for(target, df)
        keep = y_all.notna().to_numpy()
        sub = df[keep]
        y = y_all[keep].to_numpy().astype(int)
        chip = sub["Chip_Type"].to_numpy()
        label_table[name] = y_all
        print(f"{name}: n={keep.sum()}, prevalence={y.mean():.3f}", flush=True)
        # stratify on target x chip so that every fold keeps the chip mix
        strata = pd.factorize(pd.Series(y.astype(str)) + "_" + chip)[0]
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
        splits = [
            (i // folds, i % folds, tr, te) for i, (tr, te) in enumerate(cv.split(sub, strata))
        ]
        for fs in feature_sets:
            X = build_features(sub, fs, panel).drop(columns=list(target.exclude), errors="ignore")
            for m in models:
                tasks.append((name, fs, m, X.to_numpy(float), y, chip, splits))

    print(f"{len(tasks)} cells x {repeats * folds} folds", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=jobs, verbose=5)(delayed(_run_cell)(*t, keep_oof=True) for t in tasks)
    rows = [r for cell_rows, _ in results for r in cell_rows]
    folds_df = pd.DataFrame(rows)
    folds_df.to_csv(out / "folds.csv", index=False)
    oof = pd.concat([o for _, o in results if o is not None])
    # map fold-local row numbers back to patient ids
    ids = {name: df.loc[label_table[name].notna(), "Patient_ID"].to_numpy() for name in label_table}
    oof["patient"] = [ids[t][r] for t, r in zip(oof["target"], oof["row"], strict=True)]
    oof.drop(columns="row").to_csv(out / "oof_repeat0.csv", index=False)
    meta = dict(
        targets={n: TARGETS[n].description for n in targets},
        models=models,
        feature_sets=feature_sets,
        repeats=repeats,
        folds=folds,
        panel=panel,
        minutes=(time.time() - t0) / 60,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['minutes']:.1f} min", flush=True)
    return folds_df


# ---------------------------------------------------------------------------
# Comparison and report
# ---------------------------------------------------------------------------


def corrected_ttest(diff: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired per-fold differences.

    Folds of repeated CV share training data, so the naive t-test is anticonservative; variance is
    inflated by n_test / n_train. Returns (t, two-sided p).
    """
    k = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return np.inf * np.sign(diff.mean()), 0.0
    t = diff.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


COMPARISONS = [  # (better candidate, reference): "does the candidate add information?"
    ("demo+ige", "demo"),
    ("demo+summary", "demo"),
    ("demo+ige", "demo+summary"),
    ("ige", "summary"),
]


def compare(folds_df: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    rows = []
    for (target, model), g in folds_df.groupby(["target", "model"]):
        wide = g.pivot_table(index=["repeat", "fold"], columns="features", values=metric)
        n_train, n_test = g["n_train"].mean(), g["n_test"].mean()
        for cand, ref in COMPARISONS:
            if cand not in wide or ref not in wide:
                continue
            d = (wide[cand] - wide[ref]).dropna().to_numpy()
            t, p = corrected_ttest(d, n_train, n_test)
            rows.append(
                dict(
                    target=target,
                    model=model,
                    candidate=cand,
                    reference=ref,
                    delta=d.mean(),
                    delta_sd=d.std(ddof=1),
                    t=t,
                    p=p,
                )
            )
    out = pd.DataFrame(rows)
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]  # BH over every comparison
    return out


def summarize(folds_df: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    g = folds_df.groupby(["target", "model", "features"])[metric]
    return g.agg(["mean", "std", "count"]).reset_index()


def report(out: Path) -> str:
    folds_df = pd.read_csv(out / "folds.csv")
    lines = []
    for metric in ["auc", "ap"]:
        s = summarize(folds_df, metric)
        s["cell"] = s["mean"].map("{:.3f}".format) + " ± " + s["std"].map("{:.3f}".format)
        table = s.pivot_table(
            index=["target", "model"], columns="features", values="cell", aggfunc="first"
        )
        lines += [
            f"## {metric} (mean ± sd over folds)",
            table.reindex(columns=FEATURE_SETS).to_string(),
            "",
        ]
    within = folds_df.assign(auc_within=folds_df[[f"auc_{c}" for c in CHIPS]].mean(axis=1))
    s = summarize(within, "auc_within")
    s["cell"] = s["mean"].map("{:.3f}".format)
    table = s.pivot_table(
        index=["target", "model"], columns="features", values="cell", aggfunc="first"
    )
    lines += [
        "## AUC within chip (mean of the per-chip AUCs)",
        table.reindex(columns=FEATURE_SETS).to_string(),
        "",
    ]
    c = compare(folds_df)
    lines += [
        "## Paired comparisons (AUC, corrected t-test, BH q)",
        c.round(4).to_string(index=False),
    ]
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    c.to_csv(out / "comparisons.csv", index=False)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--targets", default="all")
    r.add_argument("--models", default=",".join(MODELS))
    r.add_argument("--features", default=",".join(FEATURE_SETS))
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--jobs", type=int, default=-1)
    rep = sub.add_parser("report")
    rep.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        targets = list(TARGETS) if args.targets == "all" else args.targets.split(",")
        run(
            args.out,
            targets,
            args.models.split(","),
            args.features.split(","),
            args.repeats,
            args.folds,
            args.jobs,
        )
        print(report(args.out))
    else:
        print(report(args.out))


if __name__ == "__main__":
    main()
