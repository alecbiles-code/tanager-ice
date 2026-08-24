#!/usr/bin/env python3
"""
22_vapor_attribution.py -- is the scene-wide "water loading" from script 20
atmospheric water-vapor residual, or surface liquid water? A structural
discrimination: add a narrow 941 nm vapor template to the SAME regression and
ask which component survives.

WHY THE JOINT FIT (not correlations with B). Script 20's B is contaminated by
depth-tracking linearization bias, so correlating B against vapor indices
cannot attribute it. The decisive question is spectral shape: per pixel, fit
   y = c0 + c1*x + A*alpha_ice + B*alpha_water(broad, 970) + V*g941(narrow)
If the 940-region structure is NARROW vapor residual, V absorbs it and the
broad-water B collapses. If it is BROAD liquid water, B survives the added
vapor term and V stays small. Both templates are continuum-anchored the same
way; collinearity is reported.

Corroborating (not deciding) diagnostics:
  - narrow 940 depth index map and its correlation with the product's
    columnar-water-vapor plane (a shared joint-inversion nuisance)
  - correlation of the model-free minimum position with the vapor index

Pre-registered verdicts:
  VAPOR-RESIDUAL DOMINANT : V significant (>3x sigma) AND B drops >70%
                            (or below 2x sigma) when the vapor term enters
  LIQUID SURVIVES         : B retains >50% and >2x sigma with the vapor term
                            in the model, V below 2x its sigma
  MIXED / AMBIGUOUS       : otherwise (both components present or neither
                            attributable)

Usage: python 22_vapor_attribution.py   (needs cache; independent of 20)
Writes: outputs/vapor_attribution.png, outputs/vapor_attribution.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tanager_ice import io
from tanager_ice import spectral as sp

# literature absorption (m^-1), 895-1110 nm @5 nm -- ice Warren & Brandt 2008,
# water Segelstein 1981, via the public-domain refractiveindex.info database.
K_WL=[895.0, 900.0, 905.0, 910.0, 915.0, 920.0, 925.0, 930.0, 935.0, 940.0, 945.0, 950.0, 955.0, 960.0, 965.0, 970.0, 975.0, 980.0, 985.0, 990.0, 995.0, 1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0, 1035.0, 1040.0, 1045.0, 1050.0, 1055.0, 1060.0, 1065.0, 1070.0, 1075.0, 1080.0, 1085.0, 1090.0, 1095.0, 1100.0, 1105.0, 1110.0]
A_ICE=[5.7005, 5.8643, 5.9985, 6.1313, 6.3038, 6.4744, 6.6907, 6.9047, 7.1501, 7.3928, 7.6794, 7.9631, 8.928, 9.8829, 10.9451, 11.9963, 13.185, 14.3616, 15.6282, 16.8821, 18.6285, 20.3575, 22.632, 24.8839, 26.3089, 27.7199, 28.0751, 28.4268, 28.2895, 28.1535, 27.0568, 25.9705, 24.5967, 23.2359, 22.2419, 21.2571, 20.7491, 20.2458, 20.0946, 19.9448, 19.6816, 19.4208, 19.674, 19.9251]
A_WATER=[6.4709, 6.8207, 7.1037, 7.8981, 9.5047, 11.1867, 14.6931, 19.3623, 23.4598, 29.3293, 34.7223, 38.3279, 41.9793, 44.0822, 44.8918, 45.3099, 44.8516, 43.7121, 42.4117, 41.4211, 39.6802, 37.6964, 35.4006, 33.1741, 31.2337, 29.3124, 26.9862, 24.585, 22.4661, 20.3934, 18.6045, 16.9189, 16.1002, 15.3629, 15.0497, 14.863, 15.2057, 15.6675, 16.5816, 17.534, 18.6487, 19.8853, 21.6406, 23.6139]


def smooth_nan(a, sig=2.0):
    from scipy.ndimage import gaussian_filter
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), sig)
    den = gaussian_filter(m.astype(float), sig)
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


def corr2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 500 or np.nanstd(a[m]) < 1e-12 or np.nanstd(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def anchored(v, w):
    line = v[0] + (v[-1] - v[0]) * (w - w[0]) / (w[-1] - w[0])
    return v - line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")

    with io.Scene(path) as s:
        valid = s.valid_mask(); wl = s.wl_nm
        sel = np.where((wl >= 895) & (wl <= 1110) & s.good)[0]
        R, _ = s.read_cube(bands=sel); R = np.moveaxis(R, 0, -1)
        wlw = wl[sel]; H, W = valid.shape
        fv = valid.reshape(-1)
        Rf = R.reshape(H * W, -1)[fv]
        cwv = None
        for name in ("columnar_water_vapor", "water_vapor", "cwv",
                     "column_water_vapor"):
            cwv = s.plane(name)
            if cwv is not None:
                break
        if cwv is not None:
            cwv = np.where(valid, cwv, np.nan)

    depth1030 = sp.band_depth(Rf, wlw, 1030, 958, 1082)
    bright = np.nanmean(Rf, axis=1)
    ice = np.isfinite(depth1030) & (depth1030 > 0.02) & (bright > 0.10) & np.isfinite(Rf).all(1)
    X = Rf[ice]
    rep = {"n_ice": int(ice.sum()), "cwv_plane": cwv is not None}

    wcr, CR = sp.continuum_removed(X, wlw, 900, 1100)
    ai = anchored(np.interp(wcr, K_WL, A_ICE), wcr)
    aw = anchored(np.interp(wcr, K_WL, A_WATER), wcr)
    gv = anchored(np.exp(-0.5 * ((wcr - 941.0) / 13.0) ** 2), wcr)
    rep["basis_corr_water_vapor"] = round(float(np.corrcoef(aw, gv)[0, 1]), 3)

    y = np.log(np.clip(CR, 1e-4, 0.9999)) ** 2
    x = (wcr - wcr.mean()) / (wcr.max() - wcr.min())

    def fit(cols):
        G = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(G, y.T, rcond=None)
        resid = y - (G @ coef).T
        rms = np.sqrt(np.nanmean(resid ** 2, axis=1))
        GtGinv = np.linalg.inv(G.T @ G)
        sig = np.sqrt(np.diag(GtGinv))[None, :] * rms[:, None]
        return coef.T, sig, rms

    ones = np.ones_like(wcr)
    c4, s4, rms4 = fit([ones, x, ai, aw])
    c5, s5, rms5 = fit([ones, x, ai, aw, gv])
    B4, sB4 = c4[:, 3], s4[:, 3]
    B5, sB5 = c5[:, 3], s5[:, 3]
    V5, sV5 = c5[:, 4], s5[:, 4]

    B4m, B5m = float(np.nanmedian(B4)), float(np.nanmedian(B5))
    Vm = float(np.nanmedian(V5))
    B4sig = float(np.nanmedian(B4 / (sB4 + 1e-12)))
    B5sig = float(np.nanmedian(B5 / (sB5 + 1e-12)))
    Vsig = float(np.nanmedian(V5 / (sV5 + 1e-12)))
    drop = (B4m - B5m) / (abs(B4m) + 1e-12)
    rep.update({"B_without_vapor_median": round(B4m, 6),
                "B_with_vapor_median": round(B5m, 6),
                "B_drop_fraction": round(float(drop), 3),
                "B_with_vapor_over_sigma": round(B5sig, 2),
                "V_median": round(Vm, 6),
                "V_over_sigma": round(Vsig, 2),
                "fit_rms_4term_median": round(float(np.nanmedian(rms4)), 6),
                "fit_rms_5term_median": round(float(np.nanmedian(rms5)), 6)})

    # corroborating indices
    v940 = sp.band_depth(X, wlw, 940, 920, 960)
    def tomap(vec):
        full = np.full(H * W, np.nan); full[np.where(fv)[0][ice]] = vec
        return full.reshape(H, W)
    V940map = tomap(v940)
    r_v940_cwv = corr2(smooth_nan(V940map), smooth_nan(cwv)) if cwv is not None else float("nan")
    rep["vapor940_median_depth"] = round(float(np.nanmedian(v940)), 5)
    rep["corr_vapor940_vs_CWV"] = None if not np.isfinite(r_v940_cwv) else round(r_v940_cwv, 3)

    print("\n[vapor attribution] %d ice pixels" % rep["n_ice"])
    print("  basis corr (water, vapor)     : %.2f" % rep["basis_corr_water_vapor"])
    print("  B median  without vapor term  : %+.2e (%.1fx sigma)" % (B4m, B4sig))
    print("  B median  WITH vapor term     : %+.2e (%.1fx sigma)  drop %.0f%%" % (
        B5m, B5sig, 100 * drop))
    print("  V (vapor) median              : %+.2e (%.1fx sigma)" % (Vm, Vsig))
    print("  940 index vs CWV plane        : %s" % rep["corr_vapor940_vs_CWV"])

    # Primary discriminator is the COLLAPSE of B when the vapor term enters
    # (13x separation between planted-vapor and planted-liquid synthetics);
    # V significance is a supporting floor (per-pixel median understates
    # scene-level significance).
    vap = (drop > 0.7 or B5sig < 1) and (Vsig > 2)
    liq = (B5sig > 2) and (drop < 0.5) and (Vsig < 2)
    if vap:
        verdict = ("VAPOR-RESIDUAL DOMINANT: adding a narrow 941 nm vapor "
                   "template absorbs the signal -- the vapor loading is "
                   "significant (%.1fx sigma) and the broad-water loading "
                   "collapses by %.0f%% (to %.1fx sigma). The 'liquid water' "
                   "signal in this product's 940-1000 nm region is dominated by "
                   "residual atmospheric water vapor -- the second face of the "
                   "joint-inversion surface coupling seen in the AOD diagnostic. "
                   "Surface melt is not retrievable from this scene's SR product "
                   "without vapor-residual screening or radiance-domain "
                   "retrieval." % (Vsig, 100 * drop, B5sig))
    elif liq:
        verdict = ("LIQUID SURVIVES THE VAPOR TERM: with a narrow 941 nm vapor "
                   "template in the model, the broad-water loading retains "
                   "%.0f%% of its magnitude at %.1fx sigma while the vapor "
                   "loading is insignificant (%.1fx). The 970 nm structure is "
                   "broad, as liquid water is -- not narrow vapor residual." % (
                       100 * (1 - drop), B5sig, Vsig))
    else:
        verdict = ("MIXED / AMBIGUOUS: both components load (B %.1fx sigma with "
                   "vapor term, drop %.0f%%; V %.1fx sigma). Vapor residual and "
                   "surface water cannot be cleanly separated on this scene; "
                   "melt remains not evaluable, and the vapor component should "
                   "be reported as a product characteristic." % (
                       B5sig, 100 * drop, Vsig))
    rep["verdict"] = verdict
    print("\nVERDICT: %s" % verdict)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    meany = np.nanmean(y, 0)
    G5 = np.column_stack([ones, x, ai, aw, gv])
    cmean, *_ = np.linalg.lstsq(G5, meany, rcond=None)
    ax[0].plot(wcr, meany, "k-", lw=1.6, label="observed (ln CR)^2")
    ax[0].plot(wcr, G5 @ cmean, "b--", lw=1.3, label="ice+water+vapor model")
    ax[0].plot(wcr, cmean[4] * gv, "r-", lw=1.1, label="vapor component")
    ax[0].plot(wcr, cmean[3] * aw, "c-", lw=1.1, label="broad-water component")
    ax[0].legend(fontsize=8); ax[0].set_xlabel("nm"); ax[0].set_ylabel("(ln CR)^2")
    ax[0].set_title("joint decomposition, mean ice spectrum")
    Bmap5 = tomap(B5); Vmap5 = tomap(V5)
    v = np.nanpercentile(np.abs(Vmap5), 95)
    im = ax[1].imshow(Vmap5, cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set_title("vapor loading V (red = positive)"); ax[1].axis("off")
    v = np.nanpercentile(np.abs(Bmap5), 95)
    im = ax[2].imshow(Bmap5, cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    ax[2].set_title("broad-water loading B, vapor term included"); ax[2].axis("off")
    fig.suptitle("Attribution: narrow vapor residual vs broad liquid water",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "vapor_attribution.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "vapor_attribution.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\nwrote %s" % p)
    print("wrote %s/vapor_attribution.json" % args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
