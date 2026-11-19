"""A simple storage-protein score for food anaphylaxis: discrimination, calibration, thresholds.

The score is the number of positive (>= 0.3) seed storage proteins on the panel common to the three
chips. Notebook 04 found 2S albumins and 11S legumins to carry most of the IgE information about a
history of food anaphylaxis; this module asks whether a count that a clinician can read off a chip
report is a usable, calibrated marker.

Caveat: the score was suggested by an analysis of this same cohort. The cross-validation below
protects the fitted risks, not the choice of the score; leave-one-region-out is a partial external
check, a new cohort is the real one.

    python -m acc.score --out runs/score [--repeats 10] [--boot 2000] [--jobs -1]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acc.allergens import protein_family_of
from acc.data import DATA_FILE, download, load_raw
from acc.predict import (
    CHIPS,
    POSITIVITY,
    TARGETS,
    build_features,
    common_panel,
    labels_for,
    make_model,
)

TARGET = "food_anaphylaxis"
CAP = 3  # score levels 0, 1, 2, 3+: few patients above, and the risk is flat there

# Extra stable markers of food allergy that start early (notebook 04, early_food_anaphylaxis):
# casein (cow's milk) and ovomucoid (egg). Exploratory variant only.
EARLY_MARKERS = ["Bos_d_8", "Gal_d_1"]


def score_components(panel: list[str]) -> dict[str, list[str]]:
    fam = {c: protein_family_of(c) for c in panel}
    s2 = [c for c in panel if fam[c] == "2S albumin"]
    s7 = [c for c in panel if fam[c] == "7S vicilin"]
    s11 = [c for c in panel if fam[c] == "11S legumin"]
    return {
        "2S": s2,
        "2S+11S": s2 + s11,  # primary score
        "storage": s2 + s7 + s11,
        "storage+milk+egg": s2 + s7 + s11 + [c for c in EARLY_MARKERS if c in panel],
    }


PRIMARY = "2S+11S"


def compute_scores(df: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    pos = df[panel] >= POSITIVITY
    return pd.DataFrame(
        {name: pos[cols].sum(axis=1) for name, cols in score_components(panel).items()}
    )


# ---------------------------------------------------------------------------
# Descriptive: risk by score level, threshold metrics
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def risk_table(score: pd.Series, y: np.ndarray, chip: np.ndarray) -> pd.DataFrame:
    rows = []
    level = score.clip(upper=CAP).to_numpy()
    for group in ["all", *CHIPS]:
        m = np.ones(len(y), bool) if group == "all" else chip == group
        for lv in range(CAP + 1):
            sel = m & (level == lv)
            k, n = int(y[sel].sum()), int(sel.sum())
            lo, hi = wilson(k, n)
            rows.append(
                dict(chip=group, level=lv, n=n, cases=k, risk=k / n if n else np.nan, lo=lo, hi=hi)
            )
    return pd.DataFrame(rows)


def _threshold_metrics(y: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
    fn, tn = np.sum(~pos & (y == 1)), np.sum(~pos & (y == 0))
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return dict(
        sensitivity=sens,
        specificity=spec,
        ppv=tp / (tp + fp) if tp + fp else np.nan,
        npv=tn / (tn + fn) if tn + fn else np.nan,
        lr_pos=sens / (1 - spec) if spec < 1 else np.inf,
        lr_neg=(1 - sens) / spec,
        positive_rate=pos.mean(),
    )


def threshold_table(
    scores: pd.DataFrame, y: np.ndarray, n_boot: int, seed: int = 0
) -> pd.DataFrame:
    """Score >= k as a test, with percentile bootstrap CIs (patients resampled)."""
    rng = np.random.default_rng(seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # lr+ is infinite when spec = 1
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    rows = []
    for name in scores:
        s = scores[name].to_numpy()
        for k in (1, 2, 3):
            point = _threshold_metrics(y, s >= k)
            boot = pd.DataFrame([_threshold_metrics(y[i], s[i] >= k) for i in idx])
            for metric, value in point.items():
                lo, hi = np.nanpercentile(boot[metric].replace(np.inf, np.nan), [2.5, 97.5])
                rows.append(dict(score=name, threshold=k, metric=metric, value=value, lo=lo, hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-validated models: discrimination and calibration
# ---------------------------------------------------------------------------


def score_design(scores: pd.DataFrame, name: str, chip: pd.Series) -> pd.DataFrame:
    """Score levels as categories (no linearity assumption) + chip."""
    level = scores[name].clip(upper=CAP)
    X = pd.get_dummies(
        level.astype("category").cat.set_categories(range(CAP + 1)), prefix="lv", dtype=float
    )
    X = X.drop(columns="lv_0")
    return X.join(pd.get_dummies(chip, prefix="chip", dtype=float).drop(columns="chip_ALEX"))


def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration intercept (in the large, slope fixed at 1) and slope (logistic recalibration)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e6).fit(lp[:, None], y).coef_[0, 0]
    # intercept with offset: one-parameter Newton iterations on y ~ a + 1 * lp
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        step = np.sum(y - q) / np.sum(q * (1 - q))
        a += step
        if abs(step) < 1e-10:
            break
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(
        abs(y[bins == b].mean() - p[bins == b].mean()) * np.mean(bins == b) for b in np.unique(bins)
    )
    return dict(cal_intercept=a, cal_slope=slope, ece=ece)


MODELS = {  # name -> (feature builder, estimator)
    "score": "logit",
    "demo+score": "logit",
    "demo": "logit",
    "demo+pfam": "hgb",
}


def _features(
    name: str, sub: pd.DataFrame, scores: pd.DataFrame, panel: list[str], score_name: str
):
    if name == "score":
        return score_design(scores, score_name, sub["Chip_Type"])
    if name == "demo+score":
        demo = build_features(sub, "demo", panel).drop(columns=[c for c in ["Gender"]])
        return demo.join(score_design(scores, score_name, sub["Chip_Type"]).filter(like="lv_"))
    return build_features(sub, name, panel).drop(columns=["Gender"])


def _make(kind: str, seed: int):
    if kind == "hgb":
        return make_model("hgb", seed)
    # Not acc.predict's LogisticRegressionCV: choosing C by AUC can shrink the probabilities towards
    # the prevalence without changing the AUC, which would wreck the calibration we want to measure.
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)
    )


def _fit_predict(kind, X, y, train, test, seed):
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = clone(_make(kind, seed)).fit(X[train], y[train])
        return m.predict_proba(X[test])[:, 1]


def _cv_cell(model, score_name, X, y, chip, splits):
    rows, oof = [], np.full(len(y), np.nan)
    for repeat, fold, tr, te in splits:
        p = _fit_predict(MODELS[model], X, y, tr, te, repeat)
        if repeat == 0:
            oof[te] = p
        rows.append(
            dict(
                model=model,
                score=score_name,
                repeat=repeat,
                fold=fold,
                n_train=len(tr),
                n_test=len(te),
                auc=roc_auc_score(y[te], p),
                brier=brier_score_loss(y[te], p),
                log_loss=log_loss(y[te], p),
                **calibration_stats(y[te], p),
                **{
                    f"auc_{c}": roc_auc_score(y[te][chip[te] == c], p[chip[te] == c])
                    for c in CHIPS
                    if len(set(y[te][chip[te] == c])) == 2
                },
            )
        )
    return rows, (model, score_name, oof)


def cross_validate(sub, y, scores, panel, repeats, jobs):
    chip = sub["Chip_Type"].to_numpy()
    strata = pd.factorize(pd.Series(y.astype(str)) + "_" + chip)[0]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)  # = acc.predict
    splits = [(i // 5, i % 5, tr, te) for i, (tr, te) in enumerate(cv.split(sub, strata))]
    cells = [("score", s) for s in scores] + [
        ("demo+score", PRIMARY),
        ("demo", "-"),
        ("demo+pfam", "-"),
    ]
    tasks = [
        (
            m,
            s,
            _features(m, sub, scores, panel, s if s != "-" else PRIMARY).to_numpy(float),
            y,
            chip,
            splits,
        )
        for m, s in cells
    ]
    out = Parallel(n_jobs=jobs, verbose=5)(delayed(_cv_cell)(*t) for t in tasks)
    folds = pd.DataFrame([r for rows, _ in out for r in rows])
    oof = pd.DataFrame({f"{m}|{s}": p for _, (m, s, p) in out})
    return folds, oof


# ---------------------------------------------------------------------------
# Leave-one-region-out: does the calibration travel?
# ---------------------------------------------------------------------------


def transport(sub, y, scores, min_test=100):
    region = sub["French_Region"].fillna("unknown").to_numpy()
    X = score_design(scores, PRIMARY, sub["Chip_Type"]).to_numpy(float)
    rows = []
    for r, n in pd.Series(region).value_counts().items():
        te = np.flatnonzero(region == r)
        if n < min_test or len(set(y[te])) < 2 or y[te].sum() < 10:
            continue
        tr = np.flatnonzero(region != r)
        p = _fit_predict("logit", X, y, tr, te, 0)
        cal = calibration_stats(y[te], p)
        # "Model updating" scenario: the centre re-estimates only the intercept from its own base
        # rate. It uses the held-out labels, so it is an upper bound of what a local update gives.
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        p_upd = 1 / (1 + np.exp(-(lp + cal["cal_intercept"])))
        level = scores[PRIMARY].to_numpy()[te]
        by_level = {}
        for name, m in [("0", level == 0), ("1", level == 1), ("2plus", level >= 2)]:
            by_level[f"n_{name}"] = int(m.sum())
            by_level[f"risk_{name}"] = y[te][m].mean() if m.any() else np.nan
        rows.append(
            dict(
                region=r,
                n_test=len(te),
                prevalence=y[te].mean(),
                mean_predicted=p.mean(),
                auc=roc_auc_score(y[te], p),
                brier=brier_score_loss(y[te], p),
                brier_intercept_updated=brier_score_loss(y[te], p_upd),
                **cal,
                **by_level,
            )
        )
    return pd.DataFrame(rows)


def run(out: Path, repeats: int, n_boot: int, jobs: int) -> None:
    if not DATA_FILE.exists():
        download()
    df = load_raw()
    panel = common_panel(df)
    y_all = labels_for(TARGETS[TARGET], df)
    sub = df[y_all.notna()]
    y = y_all.dropna().to_numpy().astype(int)
    scores = compute_scores(sub, panel)
    chip = sub["Chip_Type"].to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(y)} patients, prevalence {y.mean():.3f}", flush=True)

    risk = risk_table(scores[PRIMARY], y, chip)
    risk.to_csv(out / "risk_table.csv", index=False)
    print(risk.round(3).to_string(index=False), flush=True)

    thr = threshold_table(scores, y, n_boot)
    thr.to_csv(out / "thresholds.csv", index=False)

    folds, oof = cross_validate(sub, y, scores, panel, repeats, jobs)
    folds.to_csv(out / "folds.csv", index=False)
    oof.assign(
        y=y,
        chip=chip,
        patient=sub["Patient_ID"].to_numpy(),
        **{f"score_{k}": v.to_numpy() for k, v in scores.items()},
    ).to_csv(out / "oof_repeat0.csv", index=False)

    tr = transport(sub, y, scores)
    tr.to_csv(out / "transport.csv", index=False)

    summary = folds.groupby(["model", "score"])[
        ["auc", "brier", "log_loss", "cal_intercept", "cal_slope", "ece"]
    ].mean()
    print(summary.round(3).to_string(), flush=True)
    print(tr.round(3).to_string(index=False), flush=True)
    meta = dict(
        target=TARGET,
        components=score_components(panel),
        primary=PRIMARY,
        cap=CAP,
        repeats=repeats,
        n_boot=n_boot,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def corrected_ci(x, n_train, n_test, level=0.95):
    """Mean and Nadeau-Bengio corrected CI of a repeated-CV quantity (see acc.predict)."""
    x = np.asarray(x)
    k = len(x)
    se = np.sqrt((1 / k + n_test / n_train) * x.var(ddof=1))
    t = stats.t.ppf(0.5 + level / 2, k - 1)
    return x.mean(), x.mean() - t * se, x.mean() + t * se


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    run(args.out, args.repeats, args.boot, args.jobs)


if __name__ == "__main__":
    main()
