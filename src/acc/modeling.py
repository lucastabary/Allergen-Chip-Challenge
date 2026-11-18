"""Shared modelling helpers: logistic regression, F1-optimal threshold, nested CV.

Used by `acc.atlas` (clinical simulator) and `acc.profiles` (IgE-profile models).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

SEED = 0
N_REPEATS = 5  # outer cross-validation repeats (5 folds each)
C_REGULARIZATION = 0.3  # inverse L2 strength (scikit-learn's C), not tuned
MIN_CASES = 30  # a chip-specific model needs >= 30 positives and >= 30 negatives
CHIPS = ["ISAC_V1", "ISAC_V2", "ALEX"]
CHIP_LABELS = ["ISAC v1", "ISAC v2", "ALEX²"]
CHIP_FEATURES = [f"chip_{c}" for c in CHIPS]


def _model() -> LogisticRegression:
    return LogisticRegression(C=C_REGULARIZATION, max_iter=3000)


def _best_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Decision threshold (5% to 95%, step 2.5%) with the best F1 on these predictions."""
    grid = np.arange(0.05, 0.96, 0.025)
    return float(grid[np.argmax([f1_score(y, proba >= t, zero_division=0) for t in grid])])


def _fit_with_threshold(
    x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[LogisticRegression, float]:
    """Fit on (x, y); threshold tuned on out-of-fold predictions of these rows only."""
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(_model(), x, y, cv=inner, method="predict_proba")[:, 1]
    return _model().fit(x, y), _best_threshold(y, oof)


def _scores(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    auc = roc_auc_score(y, proba) if 0 < y.sum() < len(y) else np.nan
    return f1_score(y, pred, zero_division=0), auc


def _eligible(y: np.ndarray) -> bool:
    return bool(y.sum() >= MIN_CASES and (len(y) - y.sum()) >= MIN_CASES)


def evaluate_group(
    x: np.ndarray,
    x_chip: np.ndarray,
    y: np.ndarray,
    chips: np.ndarray,
    n_repeats: int = N_REPEATS,
    specific: bool = True,
) -> dict:
    """Nested cross-validation of the pooled model and of the chip-specific models.

    Rows must all have a defined target. Each repeat splits the patients into 5 folds,
    stratified on chip x target. For every held-out fold, the pooled model is trained on
    the other folds of all chips, and each chip-specific model on the other folds of its
    chip; thresholds are tuned on those training rows only. Every patient is therefore
    predicted once per repeat by both kinds of model, and F1 / AUC are computed on these
    held-out predictions, overall and per chip. Mean and sd are taken over repeats.
    """
    x_pooled = np.hstack([x, x_chip])
    present = [c for c in CHIPS if (chips == c).any()]
    specific_ok = {c: specific and _eligible(y[chips == c]) for c in present}
    runs: dict[str, list] = {"pooled": [], **{f"pooled_{c}": [] for c in present}}
    runs.update({f"specific_{c}": [] for c in present if specific_ok[c]})

    for repeat in range(n_repeats):
        seed = SEED + repeat
        proba_p, pred_p = np.zeros(len(y)), np.zeros(len(y), dtype=bool)
        proba_s, pred_s = np.zeros(len(y)), np.zeros(len(y), dtype=bool)
        strata = np.char.add(chips.astype(str), y.astype(int).astype(str))
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in outer.split(x, strata):
            model, threshold = _fit_with_threshold(x_pooled[train], y[train], seed)
            proba_p[test] = model.predict_proba(x_pooled[test])[:, 1]
            pred_p[test] = proba_p[test] >= threshold
            for chip in present:
                if not specific_ok[chip]:
                    continue
                tr, te = train[chips[train] == chip], test[chips[test] == chip]
                model, threshold = _fit_with_threshold(x[tr], y[tr], seed)
                proba_s[te] = model.predict_proba(x[te])[:, 1]
                pred_s[te] = proba_s[te] >= threshold
        runs["pooled"].append(_scores(y, proba_p, pred_p))
        for chip in present:
            rows = chips == chip
            runs[f"pooled_{chip}"].append(_scores(y[rows], proba_p[rows], pred_p[rows]))
            if specific_ok[chip]:
                runs[f"specific_{chip}"].append(_scores(y[rows], proba_s[rows], pred_s[rows]))

    def summary(values: list) -> dict:
        arr = np.array(values, dtype=float)
        return {
            "f1": round(float(arr[:, 0].mean()), 3),
            "f1_sd": round(float(arr[:, 0].std()), 3),
            "auc": round(float(np.nanmean(arr[:, 1])), 3),
        }

    return {key: summary(values) for key, values in runs.items()}


def _always_positive_f1(y: np.ndarray) -> float:
    """F1 of the trivial rule "everyone is sensitized": 2p / (1 + p), p = prevalence."""
    return round(float(f1_score(y, np.ones_like(y))), 3) if y.sum() else 0.0


def _coefficients(model: LogisticRegression) -> tuple[float, list[float]]:
    return round(float(model.intercept_[0]), 4), [round(float(c), 4) for c in model.coef_[0]]


def evaluate_new_centre(x: np.ndarray, y: np.ndarray, centres: np.ndarray) -> dict:
    """Leave-one-centre-out: each centre is predicted by a model that never saw it.

    The threshold is tuned on the training centres only (inner 5-fold CV). This measures
    how a model would do in a new hospital, free of the centre's own habits.
    """
    proba, pred = np.zeros(len(y)), np.zeros(len(y), dtype=bool)
    for train, test in LeaveOneGroupOut().split(x, y, centres):
        if y[train].sum() < 5 or y[train].sum() > len(train) - 5:
            continue
        model, threshold = _fit_with_threshold(x[train], y[train], SEED)
        proba[test] = model.predict_proba(x[test])[:, 1]
        pred[test] = proba[test] >= threshold
    f1, auc = _scores(y, proba, pred)
    return {"f1": round(float(f1), 3), "auc": round(float(auc), 3)}
