"""
Smoke tests for the pure-function validators in meld_validation.

These tests use small synthetic arrays (no PHI) and verify that each
function returns values in the expected range. They are *not* statistical
unit tests — for that, consult the original references cited in each
equation's docstring.

Run with:
    pytest -q tests/
or simply:
    python tests/test_validators.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Allow running directly without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meld_validation import (  # noqa: E402
    cronbach_alpha,
    kr20,
    bootstrap_ci,
    prevalence_bias,
    mape,
    bland_altman_loa,
    lin_ccc,
    standardized_mean_difference,
    smd_binary,
    cohens_kappa,
    mann_kendall,
    cusum_control,
    calibration_slope_intercept,
    brier_score,
    auroc,
    mahalanobis_flags,
    fellegi_sunter_weights,
    fs_decision,
    rubin_combine,
)

RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# Block A — internal consistency
# ---------------------------------------------------------------------------
def test_cronbach_alpha_high_for_correlated_items():
    base = RNG.normal(size=500)
    items = np.column_stack([base + RNG.normal(scale=0.3, size=500) for _ in range(5)])
    a = cronbach_alpha(items)
    assert 0.80 <= a <= 1.0, f"expected high alpha, got {a:.3f}"


def test_kr20_range():
    bin_items = (RNG.random((300, 6)) > 0.5).astype(int)
    v = kr20(bin_items)
    assert -1.0 <= v <= 1.0


def test_bootstrap_ci_returns_ordered_pair():
    x = RNG.normal(size=200)
    lo, hi = bootstrap_ci(x, np.mean, n_boot=200, seed=1)
    assert lo < hi


# ---------------------------------------------------------------------------
# Block B — concurrent validity
# ---------------------------------------------------------------------------
def test_prevalence_bias_zero_when_identical():
    a = np.array([0.10, 0.20, 0.30])
    assert abs(prevalence_bias(a, a)) < 1e-9


def test_mape_zero_when_identical():
    a = np.array([0.10, 0.20, 0.30])
    assert mape(a, a) == 0.0


def test_bland_altman_loa_order():
    x = RNG.normal(size=300)
    y = x + RNG.normal(scale=0.1, size=300)
    mean_diff, lo, hi = bland_altman_loa(x, y)
    assert lo < mean_diff < hi


def test_lin_ccc_close_to_one_for_identical():
    x = RNG.normal(size=300)
    assert lin_ccc(x, x) > 0.99


# ---------------------------------------------------------------------------
# Block D — representativeness
# ---------------------------------------------------------------------------
def test_smd_zero_for_same_distribution():
    a = RNG.normal(size=500)
    b = RNG.normal(size=500)
    assert abs(standardized_mean_difference(a, b)) < 0.2


def test_smd_binary_zero_when_equal():
    assert smd_binary(0.4, 0.4) == 0.0


# ---------------------------------------------------------------------------
# Block E — Rubin's rules
# ---------------------------------------------------------------------------
def test_rubin_combine_returns_finite():
    ests = [1.0, 1.1, 0.9, 1.05, 0.95]
    ses = [0.2, 0.21, 0.19, 0.2, 0.2]
    out = rubin_combine(ests, ses)
    assert np.isfinite(out["estimate"])
    assert np.isfinite(out["se"])
    assert out["se"] > 0


# ---------------------------------------------------------------------------
# Block F — record linkage
# ---------------------------------------------------------------------------
def test_fs_weights_and_decision():
    m = {"name": 0.95, "dob": 0.99, "zip": 0.90}
    u = {"name": 0.01, "dob": 0.02, "zip": 0.05}
    w = fellegi_sunter_weights(m, u)
    agree = {"name": 1, "dob": 1, "zip": 1}
    score = sum(w[f]["agree"] for f in w)
    assert fs_decision(score, lower=0, upper=5) == "match"


# ---------------------------------------------------------------------------
# Block G — kappa
# ---------------------------------------------------------------------------
def test_kappa_perfect_agreement():
    a = np.array([1, 0, 1, 1, 0, 0, 1])
    assert cohens_kappa(a, a) == 1.0


# ---------------------------------------------------------------------------
# Block H — temporal stability
# ---------------------------------------------------------------------------
def test_mann_kendall_detects_trend():
    x = np.arange(50) + RNG.normal(scale=0.1, size=50)
    out = mann_kendall(x)
    assert out["tau"] > 0.5
    assert out["p_value"] < 0.05


def test_cusum_output_shape():
    x = RNG.normal(size=100)
    c = cusum_control(x)
    assert c.shape == (100, 2) or c.shape == (100,)


# ---------------------------------------------------------------------------
# Block I — predictive validity
# ---------------------------------------------------------------------------
def test_calibration_perfect_on_linear_logits():
    # Perfectly calibrated: y_prob equals true probability
    p = RNG.uniform(0.05, 0.95, size=1000)
    y = (RNG.uniform(size=1000) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert 0.5 < slope < 1.5


def test_brier_and_auroc_ranges():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    b = brier_score(y, p)
    assert 0 <= b <= 1
    assert 0.5 <= auroc(y, p) <= 1.0


# ---------------------------------------------------------------------------
# Multivariate outliers
# ---------------------------------------------------------------------------
def test_mahalanobis_flags_shape():
    X = RNG.normal(size=(200, 4))
    flags = mahalanobis_flags(X)
    assert flags.shape == (200,)
    # Most points should not be flagged
    assert flags.sum() < 50


# ---------------------------------------------------------------------------
# Allow direct execution without pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import inspect
    funcs = [f for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"[PASS] {f.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {f.__name__}: {e}")
    print(f"\n{passed}/{len(funcs)} tests passed")
    sys.exit(0 if passed == len(funcs) else 1)
