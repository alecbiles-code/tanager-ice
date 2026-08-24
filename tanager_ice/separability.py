"""
tanager_ice.separability
=========================
Class separability for the hand-labelled Task-2 step: prove the classes
(iceberg ice / sea ice / lead-water / dirty ice / wet snow) are spectrally
distinct BEFORE any modelling. If they aren't, that reshapes the project.

Metrics:
    spectral_angle  - SAM between class means; gain-insensitive (radiance-safe).
    jeffries_matusita - JM in [0,2]; >~1.9 = well separated. Uses covariance,
                      so with 426 bands and few labelled pixels the covariance
                      is singular -> we shrink (diagonal loading) and/or reduce
                      with PCA first. Both handled here with loud defaults.
"""
from __future__ import annotations
import numpy as np

__all__ = ["class_spectra", "spectral_angle", "jeffries_matusita",
           "pairwise_jm", "pca_reduce"]


def class_spectra(X, y):
    """Return {label: {'mean':(bands,), 'std':(bands,), 'n':int}}."""
    X = np.asarray(X, float); y = np.asarray(y)
    out = {}
    for lab in np.unique(y):
        m = X[y == lab]
        out[lab] = {"mean": m.mean(0), "std": m.std(0), "n": int(m.shape[0])}
    return out


def spectral_angle(a, b):
    """Spectral Angle Mapper distance (radians) between two spectra."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    num = float(a @ b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.arccos(np.clip(num / den, -1.0, 1.0)))


def pca_reduce(X, k):
    """Mean-centre and project onto top-k principal components. Returns (Z, P, mu)."""
    X = np.asarray(X, float)
    mu = X.mean(0)
    Xc = X - mu
    # SVD is stable for tall-thin and fat matrices alike
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:k].T
    return Xc @ P, P, mu


def _shrunk_cov(Xc, shrink):
    """Diagonal-loading (Ledoit-Wolf-lite) covariance to stay invertible."""
    n, d = Xc.shape
    S = (Xc.T @ Xc) / max(n - 1, 1)
    mu = np.trace(S) / d
    return (1 - shrink) * S + shrink * mu * np.eye(d)


def jeffries_matusita(Xa, Xb, pca_k=None, shrink=0.1):
    """JM distance between two labelled sets (each (n_i, bands)).

    JM = 2(1 - exp(-B)) with Bhattacharyya B between Gaussians. With many bands
    and few samples, pass pca_k to reduce first (recommended, e.g. 6-10) and/or
    rely on `shrink` to keep covariances invertible.
    """
    Xa = np.asarray(Xa, float); Xb = np.asarray(Xb, float)
    if pca_k is not None:
        Z, P, mu = pca_reduce(np.vstack([Xa, Xb]), pca_k)
        Xa = (Xa - mu) @ P
        Xb = (Xb - mu) @ P
    ma, mb = Xa.mean(0), Xb.mean(0)
    Ca = _shrunk_cov(Xa - ma, shrink)
    Cb = _shrunk_cov(Xb - mb, shrink)
    C = 0.5 * (Ca + Cb)
    dm = (ma - mb).reshape(-1, 1)
    Cinv = np.linalg.pinv(C)
    term1 = 0.125 * float((dm.T @ Cinv @ dm).item())
    # log-det ratio via slogdet for stability
    _, ld_C = np.linalg.slogdet(C)
    _, ld_a = np.linalg.slogdet(Ca)
    _, ld_b = np.linalg.slogdet(Cb)
    term2 = 0.5 * (ld_C - 0.5 * (ld_a + ld_b))
    B = term1 + term2
    return float(2.0 * (1.0 - np.exp(-max(B, 0.0))))


def pairwise_jm(X, y, **kw):
    """Return (labels, JM_matrix) for all class pairs."""
    X = np.asarray(X, float); y = np.asarray(y)
    labs = list(np.unique(y))
    n = len(labs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = jeffries_matusita(X[y == labs[i]], X[y == labs[j]], **kw)
            M[i, j] = M[j, i] = v
    return labs, M
