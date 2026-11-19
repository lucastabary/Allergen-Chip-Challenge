"""Follow-up experiments on the clinical targets of `acc.predict`.

python -m acc.experiments screen --out runs/screen
    One logistic regression per (target, allergen), on the patients whose chip measures that
    allergen (full 241 panel): y ~ positive(allergen) + age + age^2 + chip + region ("demo"), and
    the same + log(1 + number of other positive allergens) ("demo+npos"), which separates a
    specific marker from general polysensitization. Odds ratios, Wald 95 % CI, BH q per target.
python -m acc.experiments importance --out runs/imp --targets food_anaphylaxis [--repeats 3]
    Which allergens the demo+ige gradient boosting relies on: permutation importance (drop in
    held-out AUC when one IgE column is shuffled) over repeated 5-fold CV.
python -m acc.experiments transport --out runs/transport [--targets ...]
    Leave-one-region-out: train on every other region, test on the held-out one (region is a
    proxy of the recruiting hospital). Does the IgE gain survive a change of centre?
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from acc.allergens import family_of
from acc.data import DATA_FILE, download, load_raw, split_columns
from acc.predict import (
    POSITIVITY,
    TARGETS,
    build_features,
    common_panel,
    demographic_features,
    labels_for,
    make_model,
)

CLINICAL = [n for n, t in TARGETS.items() if not t.control and n != "male"]


def _load() -> pd.DataFrame:
    if not DATA_FILE.exists():
        download()
    return load_raw()


def _target_frame(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = labels_for(TARGETS[name], df)
    keep = y.notna()
    return df[keep], y[keep].to_numpy().astype(int)


# ---------------------------------------------------------------------------
# Univariate adjusted screening
# ---------------------------------------------------------------------------


def _screen_one(name, allergen, sub, y, min_pos, npos, adjust):
    measured = sub[allergen].notna().to_numpy()
    x = (sub[allergen] >= POSITIVITY).to_numpy()[measured].astype(float)
    yy = y[measured]
    n_pos = int(x.sum())
    if n_pos < min_pos or measured.sum() - n_pos < min_pos or len(set(yy)) < 2:
        return None
    covars = demographic_features(sub[measured]).drop(columns=list(TARGETS[name].exclude))
    covars = covars.drop(columns=["Gender"])  # ~60 missing, not a confounder of interest here
    covars["Age2"] = (covars["Age"] / 10) ** 2
    covars["Age"] = covars["Age"] / 10
    covars = covars.loc[:, covars.std() > 0]
    # drop one dummy per one-hot block to avoid collinearity with the intercept
    for prefix in ("chip_", "region_"):
        cols = [c for c in covars if c.startswith(prefix)]
        if cols:
            covars = covars.drop(columns=cols[0])
    if adjust == "demo+npos":
        covars["log_npos_other"] = np.log1p(npos[measured] - x)
    X = sm.add_constant(pd.concat([pd.Series(x, index=covars.index, name="pos"), covars], axis=1))
    ok = X.notna().all(axis=1).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.Logit(yy[ok], X[ok]).fit(disp=0, maxiter=200)
        except Exception:  # perfect separation, singular matrix
            return None
    b, se = fit.params["pos"], fit.bse["pos"]
    if not np.isfinite(se) or se > 10:
        return None
    return dict(
        target=name,
        adjustment=adjust,
        allergen=allergen,
        family=family_of(allergen),
        n=int(ok.sum()),
        n_pos=n_pos,
        prev_pos=float(yy[ok][x[ok] == 1].mean()),
        prev_neg=float(yy[ok][x[ok] == 0].mean()),
        log_or=b,
        se=se,
        odds_ratio=float(np.exp(b)),
        ci_low=float(np.exp(b - 1.96 * se)),
        ci_high=float(np.exp(b + 1.96 * se)),
        p=float(fit.pvalues["pos"]),
        chips=",".join(sorted(sub.loc[sub[allergen].notna(), "Chip_Type"].unique())),
    )


def screen(out: Path, targets: list[str], min_pos: int, jobs: int) -> pd.DataFrame:
    df = _load()
    _, ige = split_columns(df)
    panel = common_panel(df)
    tasks = []
    for name in targets:
        sub, y = _target_frame(df, name)
        # positives on the common panel: the same 91 allergens whatever the chip
        npos = (sub[panel] >= POSITIVITY).sum(axis=1).to_numpy().astype(float)
        for adjust in ("demo", "demo+npos"):
            tasks += [(name, a, sub, y, min_pos, npos, adjust) for a in ige]
    rows = Parallel(n_jobs=jobs)(delayed(_screen_one)(*t) for t in tasks)
    res = pd.DataFrame([r for r in rows if r is not None])
    res["q"] = res.groupby(["target", "adjustment"])["p"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1]
    )
    res = res.sort_values(["target", "adjustment", "p"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "screen.csv", index=False)
    print(res.groupby(["target", "adjustment"])["q"].apply(lambda q: (q < 0.05).sum()))
    for (name, adjust), g in res.groupby(["target", "adjustment"]):
        print(f"\n## {name}, adjusted for {adjust}: top allergens (q < 0.05)")
        cols = ["allergen", "family", "n_pos", "odds_ratio", "ci_low", "ci_high", "q"]
        print(g[g.q < 0.05].head(15)[cols].round(4).to_string(index=False))
    return res


# ---------------------------------------------------------------------------
# Permutation importance under CV
# ---------------------------------------------------------------------------


def _importance_fold(name, X, y, cols, train, test, seed, n_perm):
    rng = np.random.default_rng(seed)
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = clone(make_model("hgb", seed)).fit(X[train], y[train])
        base = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
        rows = []
        for j, col in enumerate(cols):
            drops = []
            for _ in range(n_perm):
                Xp = X[test].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - roc_auc_score(y[test], model.predict_proba(Xp)[:, 1]))
            rows.append(dict(target=name, feature=col, seed=seed, drop=float(np.mean(drops))))
    return rows


def importance(out: Path, targets: list[str], repeats: int, n_perm: int, jobs: int):
    df = _load()
    panel = common_panel(df)
    tasks = []
    for name in targets:
        sub, y = _target_frame(df, name)
        Xdf = build_features(sub, "demo+ige", panel).drop(columns=list(TARGETS[name].exclude))
        X, cols = Xdf.to_numpy(float), list(Xdf.columns)
        strata = pd.factorize(pd.Series(y.astype(str)) + "_" + sub["Chip_Type"].to_numpy())[0]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
        for i, (tr, te) in enumerate(cv.split(X, strata)):
            tasks.append((name, X, y, cols, tr, te, i, n_perm))
    rows = Parallel(n_jobs=jobs, verbose=5)(delayed(_importance_fold)(*t) for t in tasks)
    res = pd.DataFrame([r for fold in rows for r in fold])
    summary = (
        res.groupby(["target", "feature"])["drop"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["target", "mean"], ascending=[True, False])
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "importance_folds.csv", index=False)
    summary.to_csv(out / "importance.csv", index=False)
    for name, g in summary.groupby("target"):
        print(f"\n## {name}: top features (drop in AUC when permuted)")
        print(g.head(20).round(4).to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Leave-one-region-out transport
# ---------------------------------------------------------------------------


def _transport_one(name, fs, model_name, X, y, train, test, region):
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = clone(make_model(model_name, 0)).fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
    return dict(
        target=name,
        features=fs,
        model=model_name,
        region=region,
        n_test=len(test),
        prevalence=float(y[test].mean()),
        auc=roc_auc_score(y[test], p),
    )


def transport(out: Path, targets: list[str], min_test: int, jobs: int) -> pd.DataFrame:
    """Region dummies are dropped from the demographic set: a held-out region has no coefficient."""
    df = _load()
    panel = common_panel(df)
    tasks = []
    for name in targets:
        sub, y = _target_frame(df, name)
        region = sub["French_Region"].fillna("unknown").to_numpy()
        held = [
            r
            for r, n in pd.Series(region).value_counts().items()
            if n >= min_test and len(set(y[region == r])) == 2 and (y[region == r] == 1).sum() >= 10
        ]
        for fs in ["demo", "demo+summary", "demo+ige", "ige"]:
            Xdf = build_features(sub, fs, panel).drop(columns=list(TARGETS[name].exclude))
            Xdf = Xdf.drop(columns=[c for c in Xdf if c.startswith("region_")])
            X = Xdf.to_numpy(float)
            for m in ["logreg", "hgb"]:
                for r in held:
                    test = np.flatnonzero(region == r)
                    train = np.flatnonzero(region != r)
                    tasks.append((name, fs, m, X, y, train, test, r))
    rows = Parallel(n_jobs=jobs, verbose=5)(delayed(_transport_one)(*t) for t in tasks)
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "transport.csv", index=False)
    wide = res.pivot_table(index=["target", "model", "region"], columns="features", values="auc")
    print(wide.round(3).to_string())
    gain = (wide["demo+ige"] - wide["demo"]).groupby(level=[0, 1]).agg(["mean", "min", "count"])
    print("\n## demo+ige - demo, over held-out regions")
    print(gain.round(3).to_string())
    return res


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ["screen", "importance", "transport"]:
        p = sub.add_parser(cmd)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--targets", default=",".join(CLINICAL))
        p.add_argument("--jobs", type=int, default=-1)
        if cmd == "screen":
            p.add_argument("--min-pos", type=int, default=20)
        if cmd == "importance":
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--n-perm", type=int, default=5)
        if cmd == "transport":
            p.add_argument("--min-test", type=int, default=100)
    args = parser.parse_args(argv)
    targets = args.targets.split(",")
    if args.cmd == "screen":
        screen(args.out, targets, args.min_pos, args.jobs)
    elif args.cmd == "importance":
        importance(args.out, targets, args.repeats, args.n_perm, args.jobs)
    else:
        transport(args.out, targets, args.min_test, args.jobs)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}))


if __name__ == "__main__":
    main()
