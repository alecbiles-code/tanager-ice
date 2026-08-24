#!/usr/bin/env python3
"""
25_emit_notch_crosscheck.py -- is the 941 nm vapor-core notch Tanager-specific,
or generic to the ISOFIT-family vapor stage? Runs the identical decomposition on
an EMIT L2A reflectance granule over bright snow/ice.

EMIT L2A RFL uses the same algorithmic lineage (ISOFIT optimal estimation) at
lower latitudes. If EMIT bright-snow scenes show a comparable one-signed 941 nm
core loading, the finding is fleet-wide (a property of the vapor stage over
bright surfaces); if EMIT is clean, it is Tanager- or scene-specific. Either
answer sharpens the memo's claim.

Get a granule (free, Earthdata login): search EMIT_L2A_RFL over a bright snow
target (e.g. Sierra Nevada, Andes, Alps in spring) and download the .nc file.

Usage: python 25_emit_notch_crosscheck.py --granule path/to/EMIT_L2A_RFL_*.nc
Writes: outputs/emit_crosscheck.png, outputs/emit_crosscheck.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import h5py
except ImportError:
    sys.exit("h5py required")
from tanager_ice import spectral as sp

K_WL=[895.0, 900.0, 905.0, 910.0, 915.0, 920.0, 925.0, 930.0, 935.0, 940.0, 945.0, 950.0, 955.0, 960.0, 965.0, 970.0, 975.0, 980.0, 985.0, 990.0, 995.0, 1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0, 1035.0, 1040.0, 1045.0, 1050.0, 1055.0, 1060.0, 1065.0, 1070.0, 1075.0, 1080.0, 1085.0, 1090.0, 1095.0, 1100.0, 1105.0, 1110.0]
A_ICE=[5.7005, 5.8643, 5.9985, 6.1313, 6.3038, 6.4744, 6.6907, 6.9047, 7.1501, 7.3928, 7.6794, 7.9631, 8.928, 9.8829, 10.9451, 11.9963, 13.185, 14.3616, 15.6282, 16.8821, 18.6285, 20.3575, 22.632, 24.8839, 26.3089, 27.7199, 28.0751, 28.4268, 28.2895, 28.1535, 27.0568, 25.9705, 24.5967, 23.2359, 22.2419, 21.2571, 20.7491, 20.2458, 20.0946, 19.9448, 19.6816, 19.4208, 19.674, 19.9251]
A_WATER=[6.4709, 6.8207, 7.1037, 7.8981, 9.5047, 11.1867, 14.6931, 19.3623, 23.4598, 29.3293, 34.7223, 38.3279, 41.9793, 44.0822, 44.8918, 45.3099, 44.8516, 43.7121, 42.4117, 41.4211, 39.6802, 37.6964, 35.4006, 33.1741, 31.2337, 29.3124, 26.9862, 24.585, 22.4661, 20.3934, 18.6045, 16.9189, 16.1002, 15.3629, 15.0497, 14.863, 15.2057, 15.6675, 16.5816, 17.534, 18.6487, 19.8853, 21.6406, 23.6139]


def anchored(v, w):
    line = v[0] + (v[-1] - v[0]) * (w - w[0]) / (w[-1] - w[0])
    return v - line


def find_reflectance(h5):
    """Locate the reflectance cube + wavelengths in an EMIT L2A file, with
    fallbacks; prints the tree and exits if the layout is unrecognized."""
    cand = None
    def visit(name, obj):
        nonlocal cand
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and "reflect" in name.lower():
            cand = name
    h5.visititems(visit)
    wl = None
    for wname in ("sensor_band_parameters/wavelengths", "wavelengths",
                  "sensor_band_parameters/radiance_lambda"):
        if wname in h5:
            wl = np.asarray(h5[wname][:], float).ravel()
            break
    if cand is None or wl is None:
        print("Unrecognized EMIT layout. File tree:")
        h5.visititems(lambda n, o: print("  ", n, getattr(o, "shape", "")))
        sys.exit("could not locate reflectance cube and wavelengths")
    if np.nanmax(wl) < 10:
        wl = wl * 1000.0
    return cand, wl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", required=True)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    with h5py.File(args.granule, "r") as h5:
        dsname, wl = find_reflectance(h5)
        ds = h5[dsname]
        # bands may be first or last axis
        bax = int(np.argmin(np.abs(np.array(ds.shape) - wl.size)))
        cube = np.asarray(ds[:], np.float32)
        cube = np.moveaxis(cube, bax, -1)
        cube[cube <= -0.005] = np.nan          # EMIT fill / invalid
        H, W, Bn = cube.shape
    sel = (wl >= 895) & (wl <= 1110)
    if sel.sum() < 15:
        sys.exit("too few bands in 895-1110 nm window")
    wlw = wl[sel]
    Rf = cube[:, :, sel].reshape(H * W, -1)
    fin = np.isfinite(Rf).all(1)
    Rf = Rf[fin]

    depth1030 = sp.band_depth(Rf, wlw, 1030, 958, 1082)
    bright = np.nanmean(Rf, axis=1)
    ice = np.isfinite(depth1030) & (depth1030 > 0.02) & (bright > 0.10)
    rep = {"granule": os.path.basename(args.granule),
           "n_bands_window": int(sel.sum()), "n_snow_px": int(ice.sum())}
    if ice.sum() < 2000:
        rep["verdict"] = ("insufficient bright-snow pixels (%d) in this granule; "
                          "choose a snowier target" % int(ice.sum()))
        json.dump(rep, open(os.path.join(args.outdir, "emit_crosscheck.json"), "w"), indent=2)
        print(rep["verdict"]); return 0
    X = Rf[ice]
    if X.shape[0] > 60000:
        X = X[np.random.default_rng(0).choice(X.shape[0], 60000, replace=False)]

    wcr, CR = sp.continuum_removed(X, wlw, 900, 1100)
    ai = anchored(np.interp(wcr, K_WL, A_ICE), wcr)
    aw = anchored(np.interp(wcr, K_WL, A_WATER), wcr)
    gv = anchored(np.exp(-0.5 * ((wcr - 941.0) / 13.0) ** 2), wcr)
    # signed transform: equals the (ln CR)^2 linearization in the absorption
    # regime (CR<1) but PRESERVES over-correction bumps (CR>1) as negative
    # excursions instead of clipping them to zero -- essential here, because a
    # clean-snow scene sits at CR ~ 1 near 941 nm and a Baffin-style
    # over-correction pushes CR ABOVE 1.
    lnCR = np.log(np.clip(CR, 1e-4, None))
    y = -lnCR * np.abs(lnCR)
    x = (wcr - wcr.mean()) / (wcr.max() - wcr.min())
    G = np.column_stack([np.ones_like(wcr), x, ai, aw, gv])

    def fit5(Y):
        coef, *_ = np.linalg.lstsq(G, Y.T, rcond=None)
        resid = Y - (G @ coef).T
        rms = np.sqrt(np.nanmean(resid ** 2, axis=1))
        GtGinv = np.linalg.inv(G.T @ G)
        return coef, rms * np.sqrt(GtGinv[4, 4]), rms * np.sqrt(GtGinv[3, 3])

    coef, sV, sB = fit5(y)
    V, B = coef[4], coef[3]

    # SELF-NULL BIAS CALIBRATION. Narrow-template fits on coarse grids carry a
    # deterministic bias from linearization misfit, dependent on the granule's
    # own spectral shapes. The matched null: fill each pixel's 918-967 nm
    # region by smooth interpolation from its own surrounding bands (erasing
    # any 941 nm structure while preserving everything else), re-run the
    # IDENTICAL fit, and use the null loading as this granule's bias floor.
    gap = (wcr > 918) & (wcr < 967)
    keep = ~gap
    CRn = CR.copy()
    CRn[:, gap] = np.array([np.interp(wcr[gap], wcr[keep], row) for row in CR[:, keep]])
    lnn = np.log(np.clip(CRn, 1e-4, None))
    yn = -lnn * np.abs(lnn)
    coefn, sVn, _ = fit5(yn)
    Vn = coefn[4]

    Vs = float(np.nanmedian(V) / (np.nanmedian(sV) + 1e-12))
    Vs_null = float(np.nanmedian(Vn) / (np.nanmedian(sV) + 1e-12))
    Vcorr = Vs - Vs_null
    rep["V_median"] = round(float(np.nanmedian(V)), 6)
    rep["V_over_pixel_sigma"] = round(Vs, 2)
    rep["V_null_bias_floor"] = round(Vs_null, 2)
    rep["V_corrected_over_sigma"] = round(Vcorr, 2)
    rep["B_over_pixel_sigma"] = round(float(np.nanmedian(B) / (np.nanmedian(sB) + 1e-12)), 2)
    print("\n[EMIT crosscheck] %s: %d snow px, %d bands in window" % (
        rep["granule"], rep["n_snow_px"], rep["n_bands_window"]))
    print("  941 nm loading V      : %+0.4f  (%.1fx per-pixel sigma)" % (np.nanmedian(V), Vs))
    print("  self-null bias floor  : %.1fx  (941-region smoothly infilled)" % Vs_null)
    print("  bias-corrected V      : %.1fx sigma  <-- verdict statistic" % Vcorr)

    if Vcorr < -2:
        verdict = ("FLEET-WIDE CANDIDATE: after subtracting its own smooth-infill "
                   "bias floor, this EMIT granule shows a 941 nm core deficit "
                   "(%.1fx sigma) -- same sign as the Tanager finding, consistent "
                   "with a generic property of the ISOFIT-family vapor stage over "
                   "bright surfaces." % Vcorr)
    elif Vcorr > 2:
        verdict = ("OPPOSITE-SIGNED 941 nm anomaly (%.1fx sigma, bias-corrected): "
                   "vapor-stage residual present but with opposite sign to "
                   "Tanager -- processing-version dependent; report both." % Vcorr)
    else:
        verdict = ("EMIT CLEAN (bias-corrected 941 nm loading %.1fx sigma): no "
                   "anomaly beyond this granule's own bias floor -- the Tanager "
                   "finding is product- or scene-specific as far as this "
                   "cross-check can see." % Vcorr)
    rep["verdict"] = verdict
    print("\nVERDICT: %s" % verdict)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    meany = np.nanmean(y, 0)
    cm, *_ = np.linalg.lstsq(G, meany, rcond=None)
    ax[0].plot(wcr, meany, "k-", lw=1.6, label="EMIT observed (ln CR)^2")
    ax[0].plot(wcr, G @ cm, "b--", lw=1.3, label="ice+water+vapor model")
    ax[0].plot(wcr, cm[4] * gv, "r-", lw=1.2, label="941 vapor component")
    ax[0].legend(fontsize=8); ax[0].set_xlabel("nm"); ax[0].set_ylabel("(ln CR)^2")
    ax[0].set_title("EMIT mean bright-snow decomposition")
    dv = (V - Vn)[np.isfinite(V - Vn)]
    ax[1].hist(dv, bins=70, color="tab:red", alpha=0.8)
    ax[1].axvline(0, color="k")
    ax[1].set_xlabel("941 loading minus self-null"); ax[1].set_ylabel("count")
    ax[1].set_title("bias-corrected distribution (%.1fx sigma)" % Vcorr)
    fig.suptitle("Same decomposition, EMIT L2A: is the notch fleet-wide?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "emit_crosscheck.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    with open(os.path.join(args.outdir, "emit_crosscheck.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\nwrote %s and %s/emit_crosscheck.json" % (p, args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
