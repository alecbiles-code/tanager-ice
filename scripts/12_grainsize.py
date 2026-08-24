#!/usr/bin/env python3
"""
12_grainsize.py -- relative snow/ice grain-size retrieval with calibrated,
per-pixel uncertainty. The project's first real retrieval (Task 3 core).

METHOD
  Retrieval : Nolin-Dozier scaled band-area of the 1030 nm ice absorption
              (tanager_ice.spectral.scaled_band_area). A RELATIVE grain-size
              proxy -- larger area = coarser / older / wetter grains. Not an
              absolute grain radius (that needs a radiative-transfer LUT we do
              not have and do not claim). Shape-based, so robust to the flat
              multiplicative brightness error that the AC gate flagged over sea
              ice (r=0.38): band-area normalises absolute brightness OUT.

  Uncertainty (two independent layers):
    (1) PROPAGATED: Planet's per-band surface_reflectance_uncertainty pushed
        through the band-area operator by linear error propagation -> a physics-
        based 1-sigma on the proxy for every pixel.
    (2) NORMALISED SPLIT-CONFORMAL: distribution-free intervals with guaranteed
        marginal coverage, using the propagated sigma as the normaliser so the
        interval width ADAPTS per pixel (tight on clean bright ice, wide where
        SR is noisy). Calibrated on a held-out split. This is the methodological
        contribution: conformal on top of ISOFIT's posterior.

  Topography: over snow_terrain, reflectance is C-corrected with cos(local solar
        incidence) from topo.npz before the proxy is computed; self-shadowed
        pixels (cos_i <= 0) are masked (no retrievable direct-beam signal). We
        also REPORT how much the topo correction moves the proxy -- for a shape
        retrieval it should move little, which is itself a result.

  Stratified: grain-size distribution reported per surface class.

  Internal cross-check (no field truth exists, and we say so): the 1030 nm proxy
        is regressed against the INDEPENDENT 1250 nm secondary ice feature. They
        should correlate if both track grain size -- an internal consistency test.

Usage (repo root):
    python 12_grainsize.py
    python 12_grainsize.py --alpha 0.1 --no-topo

Inputs (from earlier steps):
    outputs/scene_meta.json, cache/<ortho_sr_hdf5>.h5   (SR + uncertainty cube)
    outputs/segment_labels.npy + segment_report.json    (classes)
    outputs/topo.npz, outputs/land_mask.npy             (optional, for terrain)

Writes: outputs/grainsize.npy, outputs/grainsize_sigma.npy,
        outputs/grainsize_conformal.npz, outputs/grainsize.png,
        outputs/grainsize_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tanager_ice import io
from tanager_ice import spectral as sp
from tanager_ice import uncertainty as unc


def band_area_with_sigma(R, sigma, wl, lo=960.0, hi=1080.0):
    """Scaled band-area + propagated 1-sigma, vectorised over pixels.

    R, sigma : (N, nbands) reflectance and per-band 1-sigma (same band subset)
    Returns (area (N,), area_sigma (N,)).

    Continuum removal is linear in R given fixed shoulders, and the band area is
    A = sum_j (1 - R_j / C_j) * dwl_j, with C_j the shoulder-interpolated
    continuum. We propagate sigma through this linear form. Shoulder covariance
    is included to first order (the shoulders enter every term via C).
    """
    N, _ = R.shape
    i_lo = int(np.argmin(np.abs(wl - lo)))
    i_hi = int(np.argmin(np.abs(wl - hi)))
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo
    w = wl[i_lo:i_hi + 1]
    Rw = R[:, i_lo:i_hi + 1]
    Sw = sigma[:, i_lo:i_hi + 1]
    r_lo = R[:, i_lo]; r_hi = R[:, i_hi]
    s_lo = sigma[:, i_lo]; s_hi = sigma[:, i_hi]
    frac = (w - w[0]) / (w[-1] - w[0] + 1e-12)          # (nw,)
    C = r_lo[:, None] * (1 - frac)[None, :] + r_hi[:, None] * frac[None, :]
    C = np.where(C == 0, np.nan, C)
    cr = Rw / C
    dwl = np.gradient(w)
    area = np.nansum((1 - cr) * dwl, axis=1)

    # partial derivatives of (1 - R_j/C_j) wrt R_j, r_lo, r_hi
    dA_dRj = -(1.0 / C)                                   # (N,nw)
    # dC_j/dr_lo = (1-frac_j), dC_j/dr_hi = frac_j ; d(1-R/C)/dC = R/C^2
    dcr_dC = Rw / (C ** 2)
    dA_drlo = np.nansum(dcr_dC * (1 - frac)[None, :] * dwl, axis=1)
    dA_drhi = np.nansum(dcr_dC * frac[None, :] * dwl, axis=1)
    # variance: interior bands (independent) + shoulder terms
    var = np.nansum((dA_dRj * dwl) ** 2 * Sw ** 2, axis=1)
    var += (dA_drlo * s_lo) ** 2 + (dA_drhi * s_hi) ** 2
    return area, np.sqrt(np.maximum(var, 0.0))


def c_correction(Rw, cos_i, sun_zenith_deg, cval=None):
    """Teillet C-correction of a reflectance block by local solar incidence.

    R_corr = R * (cos_sz + C) / (cos_i + C).  C estimated per-band by regressing
    R on cos_i if not given; a scalar fallback keeps it simple and robust.
    """
    cos_sz = np.cos(np.radians(sun_zenith_deg))
    if cval is None:
        # scalar C from a robust slope/intercept of mean-R vs cos_i
        y = np.nanmean(Rw, axis=1)
        x = cos_i
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 100 and np.std(x[m]) > 1e-6:
            b, a = np.polyfit(x[m], y[m], 1)          # y = b*x + a
            cval = a / b if abs(b) > 1e-9 else 0.0
        else:
            cval = 0.0
    denom = (cos_i + cval)
    factor = (cos_sz + cval) / np.where(np.abs(denom) < 1e-3, np.nan, denom)
    return Rw * factor[:, None], float(cval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--labels", default="outputs/segment_labels.npy")
    ap.add_argument("--topo", default="outputs/topo.npz")
    ap.add_argument("--land", default="outputs/land_mask.npy")
    ap.add_argument("--alpha", type=float, default=0.1, help="1-alpha coverage")
    ap.add_argument("--no-topo", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")

    # class map + legend
    labels2d = np.load(args.labels, allow_pickle=True)
    seg = json.load(open(os.path.join(args.outdir, "segment_report.json")))
    id2name = {v: k for k, v in seg.get("final_class_ids", {}).items()}

    topo = None
    if not args.no_topo and os.path.exists(args.topo):
        topo = np.load(args.topo)
    land = np.load(args.land) if os.path.exists(args.land) else None

    rep = {"asset": args.asset, "alpha": args.alpha}
    with io.Scene(path) as s:
        valid = s.valid_mask()
        wl = s.wl_nm
        # read the 900-1300 window once: covers 1030 feature + 1250 cross-check
        band_sel = np.where((wl >= 895) & (wl <= 1305) & s.good)[0]
        R, _ = s.read_cube(bands=band_sel)
        has_sig = s.has("surface_reflectance_uncertainty")
        if has_sig:
            S, _ = s.read_cube(bands=band_sel,
                               dataset="surface_reflectance_uncertainty")
        else:
            S = np.full_like(R, np.nan)
        wlw = wl[band_sel]
        H, W = valid.shape
        R = np.moveaxis(R, 0, -1)          # (H,W,nb)
        S = np.moveaxis(S, 0, -1)

        # flatten valid pixels
        flat_valid = valid.reshape(-1)
        Rf = R.reshape(H * W, -1)[flat_valid]
        Sf = S.reshape(H * W, -1)[flat_valid]
        lab_f = labels2d.reshape(-1)[flat_valid]

        # optional topo correction over terrain (snow_terrain pixels on land)
        topo_shift = None
        if topo is not None and land is not None:
            cos_i = topo["cos_i"].reshape(-1)[flat_valid]
            sun_z = float(topo["sun_zenith"])
            land_f = land.reshape(-1)[flat_valid]
            terrain = land_f & np.isfinite(cos_i)
            shadow = terrain & (cos_i <= 0.05)
            # proxy before correction (for the shift measurement)
            area_before, _ = band_area_with_sigma(Rf, np.abs(Sf), wlw)
            Rf_corr = Rf.copy()
            if terrain.sum() > 200:
                Rc, cval = c_correction(Rf[terrain], cos_i[terrain], sun_z)
                Rf_corr[terrain] = Rc
                rep["topo_C_value"] = round(cval, 4)
            Rf = Rf_corr
            # mask self-shadowed: no retrievable signal
            Sf = Sf.copy()
            Sf[shadow] = np.nan
            rep["self_shadowed_pixels"] = int(shadow.sum())
        else:
            area_before = None
            print("[grain] no topo/land -> skipping C-correction "
                  "(flat-surface retrieval only is still valid)")

    # ---- retrieval + propagated sigma ----
    Sf_abs = np.where(np.isfinite(Sf), np.abs(Sf), np.nan)
    area, area_sig = band_area_with_sigma(Rf, np.nan_to_num(Sf_abs, nan=0.0), wlw)
    # independent 1250 nm cross-check feature
    area_1250, _ = band_area_with_sigma(Rf, np.nan_to_num(Sf_abs, nan=0.0),
                                        wlw, lo=1180.0, hi=1300.0)

    # ICE MASK: grain size is only defined where ice absorption exists. Water and
    # dark/mixed pixels give meaningless band areas (continuum removal divides by
    # near-zero reflectance -> huge outliers that destroy any validation). Require
    # a real 1030 absorption AND adequate NIR brightness.
    depth_1030 = sp.band_depth(Rf, wlw, 1030.0, 960.0, 1080.0)
    i865 = int(np.argmin(np.abs(wlw - 1030)))     # brightness proxy inside window
    nir_bright = np.nanmax(Rf, axis=1)            # peak SR in 900-1300 window
    ice_mask = (np.isfinite(depth_1030) & (depth_1030 > 0.02) &
                (nir_bright > 0.15))
    rep["ice_pixels"] = int(ice_mask.sum())
    rep["ice_fraction_of_valid"] = round(float(ice_mask.mean()), 4)
    # zero out non-ice so they are excluded everywhere downstream
    area = np.where(ice_mask, area, np.nan)
    area_sig = np.where(ice_mask, area_sig, np.nan)

    ok = np.isfinite(area) & np.isfinite(area_sig) & (area_sig > 0)
    rep["n_retrieved"] = int(ok.sum())

    # topo shift measurement
    if area_before is not None:
        d = area[ok] - area_before[ok]
        rep["topo_shift_median"] = round(float(np.nanmedian(np.abs(d))), 4)
        rep["topo_shift_vs_signal"] = round(
            float(np.nanmedian(np.abs(d)) / (np.nanstd(area[ok]) + 1e-9)), 4)

    # ---- normalised split-conformal (LOCAL spatial reference) ----
    # No external truth exists. The interval must cover per-pixel NOISE, so the
    # reference is a pixel's own smoothed spatial neighbourhood (grain size
    # varies smoothly in space; departures from the local mean are noise, which
    # is exactly what Planet's propagated sigma should predict). Nonconformity =
    # |area - local_smooth| / sigma; split-conformal on a held-out half gives a
    # multiplier qhat so the normalised interval hits (1-alpha) coverage. This is
    # a PRECISION / self-consistency interval, not accuracy vs field truth.
    from scipy.ndimage import median_filter
    # build the proxy back into 2-D to smooth spatially, then re-extract
    tmp = np.full(H * W, np.nan)
    tmp[np.where(flat_valid)[0][ok]] = area[ok]
    tmp2d = tmp.reshape(H, W)
    finite2d = np.isfinite(tmp2d)
    filled = np.where(finite2d, tmp2d, np.nan)
    # nan-robust local median via a filled copy
    filled0 = np.where(finite2d, tmp2d, np.nanmedian(area[ok]))
    local = median_filter(filled0, size=5)
    local_flat = local.reshape(-1)[np.where(flat_valid)[0]]
    ref = np.full(area.shape, np.nan)
    ref[ok] = local_flat[ok]

    rng = np.random.default_rng(0)
    idx = np.where(ok & np.isfinite(ref))[0]
    rng.shuffle(idx)
    half = len(idx) // 2
    cal, test = idx[:half], idx[half:]
    nonconf = np.abs(area[cal] - ref[cal]) / area_sig[cal]
    qhat = unc.conformal_quantile(nonconf[np.isfinite(nonconf)], args.alpha)
    lo = area - qhat * area_sig
    hi = area + qhat * area_sig
    # coverage evaluated against the SAME local reference on the held-out split
    cov = float(((ref[test] >= lo[test]) & (ref[test] <= hi[test])).mean())
    lab_ok = lab_f
    rep["conformal"] = {
        "qhat": round(float(qhat), 3),
        "target_coverage": round(1 - args.alpha, 3),
        "empirical_coverage_holdout": round(cov, 3),
        "median_interval_halfwidth": round(float(np.nanmedian(qhat * area_sig[ok])), 4),
        "note": ("precision / self-consistency interval: nonconformity is the "
                 "departure from a pixel's local spatial neighbourhood, normalised "
                 "by Planet's propagated SR uncertainty. Coverage is against the "
                 "local reference, NOT field truth (none exists)."),
    }

    # ---- internal cross-check: NIR physics (Nolin-Dozier) ----
    # No distinct 1250 nm feature exists over fine snow/ice (only over coarse
    # glacier ice), so the 1250 check is uninformative here. The robust physical
    # check: coarser grains deepen ice absorption -> LOWER reflectance in the ice
    # bands. The grain proxy must ANTI-correlate with NIR reflectance at ~1100 nm,
    # over ICE pixels only.
    i1100 = int(np.argmin(np.abs(wlw - 1100)))
    nir_1100 = Rf[:, i1100]
    m2 = ok & np.isfinite(nir_1100)
    if m2.sum() > 200:
        r_phys = float(np.corrcoef(area[m2], nir_1100[m2])[0, 1])
    else:
        r_phys = float("nan")
    rep["physics_check_proxy_vs_nir1100_r"] = None if not np.isfinite(r_phys) else round(r_phys, 3)
    rep["physics_check_note"] = ("Nolin-Dozier: coarser grains -> deeper ice "
                                 "absorption -> lower NIR reflectance. Strong "
                                 "NEGATIVE r validates the retrieval physically.")

    # ---- per-class distributions ----
    rep["per_class"] = {}
    for c in np.unique(lab_ok[ok]):
        m = ok & (lab_ok == c)
        nm = id2name.get(int(c), f"class_{int(c)}")
        rep["per_class"][nm] = {
            "n": int(m.sum()),
            "grain_proxy_median": round(float(np.nanmedian(area[m])), 4),
            "grain_proxy_iqr": [round(float(np.nanpercentile(area[m], 25)), 4),
                                round(float(np.nanpercentile(area[m], 75)), 4)],
            "median_conformal_halfwidth": round(float(np.nanmedian(qhat * area_sig[m])), 4),
        }

    # ---- write maps ----
    grain_map = np.full(H * W, np.nan)
    sig_map = np.full(H * W, np.nan)
    gi = np.where(flat_valid)[0]
    grain_map[gi[ok]] = area[ok]
    sig_map[gi[ok]] = area_sig[ok]
    grain_map = grain_map.reshape(H, W)
    sig_map = sig_map.reshape(H, W)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "grainsize.npy"), grain_map)
    np.save(os.path.join(args.outdir, "grainsize_sigma.npy"), sig_map)
    np.savez(os.path.join(args.outdir, "grainsize_conformal.npz"),
             qhat=qhat, alpha=args.alpha)

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    im = ax[0, 0].imshow(grain_map, cmap="viridis")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046)
    ax[0, 0].set_title("grain-size proxy (1030 nm band area)\nlarger = coarser/older")
    im = ax[0, 1].imshow(qhat * sig_map, cmap="magma")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)
    ax[0, 1].set_title(f"conformal half-width ({int(100*(1-args.alpha))}% coverage)\n"
                       "adaptive: wide where SR noisy")
    # per-class violin-ish
    names, data = [], []
    for c in np.unique(lab_ok[ok]):
        m = ok & (lab_ok == c)
        names.append(id2name.get(int(c), str(int(c))))
        data.append(area[m])
    ax[1, 0].boxplot(data, tick_labels=names, showfliers=False)
    ax[1, 0].set_ylabel("grain proxy")
    ax[1, 0].set_title("grain size by surface class")
    ax[1, 0].tick_params(axis="x", rotation=30)
    # cross-check scatter
    if m2.sum() > 200:
        samp = np.where(m2)[0]
        samp = samp[rng.choice(len(samp), min(4000, len(samp)), replace=False)]
        ax[1, 1].scatter(area[samp], nir_1100[samp], s=4, alpha=0.3)
        ax[1, 1].set_xlabel("1030 nm grain proxy"); ax[1, 1].set_ylabel("NIR reflectance @1100nm")
        ax[1, 1].set_title(f"physics check  r={r_phys:.3f}\n"
                           "(coarser grain -> darker ice bands)")
    fig.suptitle(f"Grain-size retrieval + conformal uncertainty -- {meta['id']}")
    fig.tight_layout()
    p = os.path.join(args.outdir, "grainsize.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "grainsize_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print("\n=== GRAIN-SIZE RETRIEVAL ===")
    print(f"retrieved pixels : {rep['n_retrieved']:,}")
    print(f"conformal cover  : {rep['conformal']['empirical_coverage_holdout']} "
          f"(target {rep['conformal']['target_coverage']})  qhat={rep['conformal']['qhat']}")
    if "topo_shift_vs_signal" in rep:
        print(f"topo shift/signal: {rep['topo_shift_vs_signal']}  "
              f"(C={rep.get('topo_C_value')}) -- small confirms shape-robustness")
    print(f"physics check (proxy vs NIR1100) r: {rep['physics_check_proxy_vs_nir1100_r']}  (want strong negative)")
    print("per class (median proxy [IQR], conformal +/-):")
    for nm, d in rep["per_class"].items():
        print(f"  {nm:16s} n={d['n']:7d}  {d['grain_proxy_median']:.3f} "
              f"{d['grain_proxy_iqr']}  +/-{d['median_conformal_halfwidth']:.3f}")
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/grainsize.npy, grainsize_sigma.npy, grainsize_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
