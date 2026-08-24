"""
tanager_ice.spectral
=====================
Spectral-shape retrieval primitives for Tanager TOA radiance.

Design principle (from the project's honesty constraints): retrievals are
computed as *relative, shape-based* quantities that are robust to a flat
multiplicative atmospheric transmission, and reported as gradients rather than
absolute physical numbers. The only place an atmospheric *correction* is applied
is the VNIR sediment slope, where an additive Rayleigh path-radiance term
genuinely biases the shape (DOS-Rayleigh below).

All functions operate on arrays shaped (..., n_bands) with a 1-D `wl` in
nanometres, so they work identically on a single spectrum (n_bands,), a table of
labelled pixels (N, n_bands), or a reshaped cube (n_pixels, n_bands).

Atmospheric confounds, per retrieval:
    grain size (1030 nm) : clean window, Rayleigh small -> NO correction.
    melt (~970 nm)       : sits under the 940 nm water-VAPOUR band -> estimate
                           column water vapour (CIBR) and flag/deweight, do NOT
                           "correct" (the confound is gas, not path radiance).
    sediment (VNIR slope): Rayleigh path radiance (lambda^-4) biases the slope
                           -> DOS-Rayleigh additive removal before the index.
"""
from __future__ import annotations
import numpy as np

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

__all__ = [
    "nearest_index", "continuum_removed", "band_depth", "scaled_band_area",
    "cibr", "pwv_proxy", "dos_rayleigh_estimate", "dos_rayleigh_correct",
    "vnir_slope_index", "grain_size_index", "melt_index", "sediment_index",
]


def nearest_index(wl: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wl) - target_nm)))


def continuum_removed(spec: np.ndarray, wl: np.ndarray,
                      lo_nm: float, hi_nm: float):
    """Clark & Roush continuum removal over [lo_nm, hi_nm].

    The continuum is the straight line (in radiance) joining the values at the
    two shoulder wavelengths; CR = spec / continuum within the window. A flat
    multiplicative gain divides out, which is what makes this radiance-safe.

    Returns (wl_window, cr_window) where cr==1 means "on the continuum".
    """
    spec = np.asarray(spec, float); wl = np.asarray(wl, float)
    i_lo, i_hi = nearest_index(wl, lo_nm), nearest_index(wl, hi_nm)
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo
    sl = slice(i_lo, i_hi + 1)
    w = wl[sl]
    y = spec[..., sl]
    y_lo = spec[..., i_lo][..., None]
    y_hi = spec[..., i_hi][..., None]
    frac = (w - w[0]) / (w[-1] - w[0] + 1e-12)
    cont = y_lo + (y_hi - y_lo) * frac
    cr = y / np.where(cont == 0, np.nan, cont)
    return w, cr


def band_depth(spec, wl, center_nm, lo_nm, hi_nm):
    """1 - CR at the feature centre (0 = no absorption)."""
    w, cr = continuum_removed(spec, wl, lo_nm, hi_nm)
    j = int(np.argmin(np.abs(w - center_nm)))
    return 1.0 - cr[..., j]


def scaled_band_area(spec, wl, lo_nm, hi_nm):
    """Integral of (1 - CR) over the window (Nolin-Dozier style band area).

    Monotonic proxy for grain size at the 1030 nm ice feature: deeper/wider ->
    larger area -> coarser/older ice. Relative only; absolute grain radius needs
    a radiative-transfer LUT and is deliberately NOT claimed here.
    """
    w, cr = continuum_removed(spec, wl, lo_nm, hi_nm)
    return _trapz(1.0 - cr, w, axis=-1)


def cibr(spec, wl, abs_nm=940.0, win_lo_nm=865.0, win_hi_nm=1020.0):
    """Continuum Interpolated Band Ratio at a gas absorption band.

    CIBR = L(abs) / [linear interp of the two window bands at abs].
    <1 for absorption; smaller -> more water vapour. Used to isolate the
    ATMOSPHERIC vapour signal (940 nm) so it can be separated from surface
    liquid-water absorption (~970 nm) in the melt channel.
    """
    spec = np.asarray(spec, float); wl = np.asarray(wl, float)
    ia, i1, i2 = (nearest_index(wl, abs_nm), nearest_index(wl, win_lo_nm),
                  nearest_index(wl, win_hi_nm))
    la, l1, l2 = wl[ia], wl[i1], wl[i2]
    f = (la - l1) / (l2 - l1 + 1e-12)
    cont = spec[..., i1] * (1 - f) + spec[..., i2] * f
    return spec[..., ia] / np.where(cont == 0, np.nan, cont)


def pwv_proxy(spec, wl, **kw):
    """Monotonic-in-PWV proxy = -ln(CIBR) (relative, not centimetres)."""
    return -np.log(np.clip(cibr(spec, wl, **kw), 1e-6, None))


# --- VNIR Rayleigh path-radiance (DOS) --------------------------------------
def dos_rayleigh_estimate(cube_pix, wl, ref_nm=440.0, dark_pct=1.0,
                          dark_mask=None):
    """Estimate the additive path-radiance amplitude A at ref_nm.

    Dark-object subtraction: over dark targets (open water / shadow) the TOA
    blue radiance is dominated by atmospheric path radiance. We take a low
    percentile of the reference-band radiance as A, then model the additive
    term's spectral shape as Rayleigh (lambda^-4). Self-contained: needs no
    external solar-irradiance or aerosol tables. Coarse by construction.

    cube_pix : (N, n_bands) radiance for N pixels (pass the whole scene).
    dark_mask: optional boolean (N,) selecting water/shadow pixels for the anchor.
    Returns scalar A (radiance units at ref_nm).

    Known bias: DOS reads path radiance + the dark target's residual surface
    radiance, so A is an UPPER bound on true path radiance (over-correction in
    the blue). Choose the darkest available target and a low percentile; fold
    the residual into the sediment-index uncertainty.
    """
    cube_pix = np.asarray(cube_pix, float)
    iref = nearest_index(wl, ref_nm)
    col = cube_pix[:, iref]
    if dark_mask is not None:
        col = col[np.asarray(dark_mask, bool)]
    col = col[np.isfinite(col)]
    if col.size == 0:
        return 0.0
    return float(np.percentile(col, dark_pct))


def dos_rayleigh_correct(spec, wl, A, ref_nm=440.0):
    """Subtract A*(ref/lambda)^4 additive path term (lambda in nm)."""
    wl = np.asarray(wl, float)
    add = A * (ref_nm / wl) ** 4
    return np.asarray(spec, float) - add


# --- packaged indices -------------------------------------------------------
def grain_size_index(spec, wl, center_nm=1030.0, lo_nm=960.0, hi_nm=1080.0):
    """Relative grain-size proxy (scaled band area at the 1030 nm ice feature)."""
    return scaled_band_area(spec, wl, lo_nm, hi_nm)


def melt_index(spec, wl, center_nm=970.0, lo_nm=930.0, hi_nm=1050.0,
               pwv_flag_thresh=None):
    """Relative liquid-water/melt proxy + a water-vapour flag.

    Returns (depth, pwv). `depth` is the continuum-removed band depth over the
    liquid-water region; `pwv` is the relative water-vapour proxy from the
    940 nm CIBR. Where pwv is high, `depth` is confounded and should be
    deweighted/flagged (the caller decides the threshold from the scene).
    """
    depth = band_depth(spec, wl, center_nm, lo_nm, hi_nm)
    pwv = pwv_proxy(spec, wl)
    if pwv_flag_thresh is not None:
        flag = pwv > pwv_flag_thresh
        return depth, pwv, flag
    return depth, pwv


def sediment_index(spec, wl, blue_nm=490.0, red_nm=680.0,
                   A=None, ref_nm=440.0):
    """Relative impurity/'dirty ice' proxy: VNIR reddening slope.

    If A (DOS-Rayleigh amplitude) is given, the additive path term is removed
    first so the slope reflects the SURFACE, not the atmosphere. Normalised
    difference (red-blue)/(red+blue): higher -> redder/darker -> more impurity.
    """
    s = np.asarray(spec, float)
    if A is not None:
        s = dos_rayleigh_correct(s, wl, A, ref_nm)
    ib, ir = nearest_index(wl, blue_nm), nearest_index(wl, red_nm)
    b, r = s[..., ib], s[..., ir]
    return (r - b) / (r + b + 1e-12)
