import numpy as np
import pytest

from acc import atlas, modeling


def test_template_has_data_marker(tmp_path):
    payload = {"hello": "</script>"}
    page = atlas.render(payload, tmp_path / "index.html").read_text("utf-8")
    assert "/*__ATLAS_DATA__*/" not in page
    assert '"<\\/script>"' in page  # a closing tag in the data must not end the script block


def test_best_threshold_prefers_recall_on_rare_target():
    y = np.array([0] * 80 + [1] * 20)
    proba = np.r_[np.linspace(0.0, 0.3, 80), np.linspace(0.2, 0.5, 20)]
    assert modeling._best_threshold(y, proba) < 0.5


@pytest.mark.parametrize("n_pos, ok", [(29, False), (30, True)])
def test_specific_model_needs_enough_cases(n_pos, ok):
    y = np.array([1] * n_pos + [0] * 100)
    assert modeling._eligible(y) is ok
