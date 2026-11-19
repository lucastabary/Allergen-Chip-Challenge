import numpy as np
import pandas as pd
import pytest

from acc.data import DATA_FILE, load_raw
from acc.predict import TARGETS, common_panel, corrected_ttest, labels_for, parse_codes


def test_parse_codes():
    assert parse_codes("1,3") == {1, 3}
    assert parse_codes("6, 7") == {6, 7}
    assert parse_codes("9") is None
    assert parse_codes(np.nan) is None
    assert parse_codes("0") == {0}


def test_target_labels_from_codes():
    df = pd.DataFrame({"Age_of_onsets": ["0", "2", "9", "2, 3", "5"], "Chip_Type": ["ALEX"] * 5})
    y = labels_for(TARGETS["food_anaphylaxis"], df)
    assert y.tolist()[:2] == [0.0, 1.0] and np.isnan(y[2]) and y[3] == 1.0
    early = labels_for(TARGETS["early_food_anaphylaxis"], df)
    assert np.isnan(early[0]) and early[1] == 1.0 and early[4] == 0.0


def test_corrected_ttest_is_more_conservative():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 50)
    _, p = corrected_ttest(d, n_train=800, n_test=200)
    naive = 2 * __import__("scipy").stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(50))), 49)
    assert p > naive


@pytest.mark.skipif(not DATA_FILE.exists(), reason="run `uv run acc-download` first")
def test_common_panel_and_shuffled_target():
    df = load_raw()
    assert len(common_panel(df)) == 91
    y, s = labels_for(TARGETS["food_anaphylaxis"], df), labels_for(TARGETS["shuffled"], df)
    # same labels per chip, different order
    assert (y.groupby(df["Chip_Type"]).mean() == s.groupby(df["Chip_Type"]).mean()).all()
    assert (y.dropna() != s.dropna()).any()


def test_score_calibration_stats_on_calibrated_data():
    from acc.score import calibration_stats, wilson

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    cal = calibration_stats(y, p)
    assert abs(cal["cal_intercept"]) < 0.05 and abs(cal["cal_slope"] - 1) < 0.05
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
