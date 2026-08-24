"""
tanager_ice.uncertainty
========================
Calibrated per-pixel uncertainty via split conformal prediction -- the project's
core differentiator. Two modes:

    regression      -> distribution-free prediction intervals for the retrieval
                       proxies (grain-size / melt / sediment).
    classification  -> prediction SETS with guaranteed marginal coverage for the
                       class map (pixels get {sea ice, dirty ice} when ambiguous
                       rather than a false single label).

HONESTY NOTE (state this in the write-up): conformal guarantees coverage
*relative to the calibration distribution*. With no field ground truth,
calibration residuals come from held-out hand-labelled pixels, so coverage is
conditional on those labels being representative of the scene. Conformal makes
the uncertainty honest and reproducible; it does not manufacture field truth.
"""
from __future__ import annotations
import numpy as np

__all__ = ["conformal_quantile", "regression_interval",
           "classification_sets", "empirical_coverage"]


def conformal_quantile(scores, alpha):
    """Finite-sample conformal quantile of nonconformity scores."""
    scores = np.asarray(scores, float)
    n = scores.size
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def regression_interval(cal_pred, cal_true, alpha=0.1):
    """Split-conformal half-width for absolute-residual nonconformity.

    Prediction interval for a new pixel is yhat +/- qhat (marginal 1-alpha
    coverage). Returns qhat.
    """
    res = np.abs(np.asarray(cal_true, float) - np.asarray(cal_pred, float))
    return conformal_quantile(res, alpha)


def classification_sets(cal_probs, cal_labels, test_probs, alpha=0.1):
    """Least-ambiguous-set conformal classification (Sadinle et al.).

    Nonconformity on calibration = 1 - p(true class). Threshold qhat; a test
    pixel's set = {classes c : p_c >= 1 - qhat}. Guarantees marginal coverage
    >= 1-alpha. `cal_labels` are integer class indices into the prob columns.

    Returns (qhat, sets) where sets is a boolean (n_test, n_classes) mask.
    """
    cal_probs = np.asarray(cal_probs, float)
    cal_labels = np.asarray(cal_labels, int)
    p_true = cal_probs[np.arange(cal_probs.shape[0]), cal_labels]
    qhat = conformal_quantile(1.0 - p_true, alpha)
    sets = np.asarray(test_probs, float) >= (1.0 - qhat)
    # guarantee non-empty sets: fall back to argmax where the rule excludes all
    empty = ~sets.any(1)
    if empty.any():
        sets[empty, np.asarray(test_probs)[empty].argmax(1)] = True
    return qhat, sets


def empirical_coverage(sets_or_lo_hi, truth, mode="class"):
    """Diagnostic: realised coverage on a labelled test set.

    mode='class': sets_or_lo_hi is boolean (n, n_classes), truth int indices.
    mode='reg'  : sets_or_lo_hi is (lo, hi) arrays, truth continuous.
    """
    if mode == "class":
        sets = np.asarray(sets_or_lo_hi, bool)
        truth = np.asarray(truth, int)
        hit = sets[np.arange(sets.shape[0]), truth]
        return float(hit.mean())
    lo, hi = sets_or_lo_hi
    truth = np.asarray(truth, float)
    return float(((truth >= lo) & (truth <= hi)).mean())
