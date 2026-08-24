#!/usr/bin/env python3
"""
18_melt_positive_control.py -- is the 970 nm liquid-water signal spectrally REAL
and distinct from the grain feature, or an artifact? A non-circular control.

THE OBJECTION. Over open water the melt index reads like pure ice, so the
independence of melt from water fraction (r=-0.09) might mean the index responds
to NOTHING. We need positive evidence that the 970 nm absorption is real.

WHY A DOSE-RESPONSE WON'T WORK HONESTLY. On a single snapshot with no ground
truth there is no independent "wetness" label to rank pixels by: every spectral
wetness proxy (1030 grain depth, 1250 ice depth, visible brightness) either
measures a different property or co-varies with 970 by construction. Ranking by
any of them and then reading 970 would be the same circularity the grain check
was criticised for. And a 970 band depth is undefined over open water (no
reflected NIR signal to absorb from), so open water is the wrong endmember.

THE NON-CIRCULAR TEST. Ask a purely spectral question: does a 970 nm absorption
exist OVER AND ABOVE the grain/ice feature? Procedure, per ice pixel:
  1. Model the 900-1100 nm shape from the ICE feature alone: a smooth continuum
     plus the broad ice absorption anchored at 1030, fit using bands OUTSIDE a
     protected 955-985 nm window (so the fit never sees 970).
  2. Predict reflectance at 970 from that ice-only model; the RESIDUAL
     (observed - predicted) at 970 is absorption unexplained by grain/ice.
  3. A systematic NEGATIVE residual (extra absorption) with spatial STRUCTURE
     exceeding noise = a liquid-water signature spectrally independent of grain.
     Pure-noise residual = no melt signal.
This never uses the 970 value to define the thing it measures, so a positive
result is real evidence, not a tautology.

Usage: python 18_melt_positive_control.py
Writes: outputs/melt_control.png, outputs/melt_control.json,
        outputs/melt_residual970.npy
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--labels", default="outputs/segment_labels.npy")
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
        wlw = wl[sel]
        H, W = valid.shape
        fv = valid.reshape(-1)
        Rf = R.reshape(H * W, -1)[fv]

    depth1030 = sp.band_depth(Rf, wlw, 1030, 958, 1082)
    win_bright = np.nanmean(Rf, axis=1)
    ice = np.isfinite(depth1030) & (depth1030 > 0.02) & (win_bright > 0.10) & np.isfinite(Rf).all(1)
    X = Rf[ice]
    rep = {"n_ice": int(ice.sum())}

    PROT_LO, PROT_HI = 955.0, 985.0
    fit_bands = (wlw < PROT_LO) | (wlw > PROT_HI)
    i970 = int(np.argmin(np.abs(wlw - 970)))
    wl_fit = wlw[fit_bands]

    def design(w):
        w0 = (w - 1000.0) / 100.0
        ice_feat = np.exp(-0.5 * ((w - 1030.0) / 55.0) ** 2)
        return np.column_stack([np.ones_like(w), w0, w0 ** 2, ice_feat])
    Afit = design(wl_fit)
    Aall = design(wlw)
    Yfit = X[:, fit_bands].T
    coef, *_ = np.linalg.lstsq(Afit, Yfit, rcond=None)
    pred_all = (Aall @ coef).T
    resid = X - pred_all
    resid970 = resid[:, i970]

    fit_resid = resid[:, fit_bands]
    noise = float(np.nanstd(fit_resid))
    rep["fit_region_noise"] = round(noise, 5)
    rep["resid970_median"] = round(float(np.nanmedian(resid970)), 5)
    rep["resid970_mean"] = round(float(np.nanmean(resid970)), 5)
    rep["resid970_over_noise"] = round(float(abs(np.nanmedian(resid970)) / (noise + 1e-9)), 2)
    frac_absorb = float(np.mean(resid970 < -noise))
    rep["frac_pixels_extra_absorption"] = round(frac_absorb, 3)

    print("\n[control] %d ice pixels" % int(ice.sum()))
    print("  fit-region noise (1-sigma)     : %.4f" % noise)
    print("  median 970 residual            : %+.4f  (%.2fx noise)" % (
        np.nanmedian(resid970), rep["resid970_over_noise"]))
    print("  fraction with extra absorption : %.0f%% (residual < -1 sigma)" % (100 * frac_absorb))

    full = np.full(H * W, np.nan)
    gi = np.where(fv)[0][ice]
    full[gi] = resid970
    r2d = full.reshape(H, W)
    from scipy.ndimage import uniform_filter
    m = np.isfinite(r2d)
    filled = np.where(m, r2d, 0.0)
    nb = (uniform_filter(filled, 3) * 9 - filled) / 8.0
    both = m & np.isfinite(nb)
    if both.sum() > 500 and np.std(r2d[both]) > 1e-9:
        spatial_r = float(np.corrcoef(r2d[both], nb[both])[0, 1])
    else:
        spatial_r = float("nan")
    rep["spatial_autocorr"] = round(spatial_r, 3)
    print("  spatial autocorrelation        : %.3f (high = structured)" % spatial_r)

    real = (rep["resid970_over_noise"] > 2 and rep["resid970_median"] < 0
            and np.isfinite(spatial_r) and spatial_r > 0.3 and frac_absorb > 0.3)
    if real:
        verdict = ("970 nm SIGNAL IS REAL AND DISTINCT FROM GRAIN: after modelling "
                   "each ice spectrum from the 1030 ice feature alone (excluding the "
                   "955-985 nm window), a systematic EXTRA absorption remains at 970 "
                   "(%.1fx noise, in %.0f%% of ice pixels) with spatial structure "
                   "(autocorr %.2f). A liquid-water signature the grain feature "
                   "cannot explain -- positive, non-circular evidence." % (
                       rep["resid970_over_noise"], 100 * frac_absorb, spatial_r))
    elif rep["resid970_over_noise"] < 1.2:
        verdict = ("NO DISTINCT 970 SIGNAL: the residual at 970 after the ice-only "
                   "model is at noise level. The 970 index carries no absorption "
                   "beyond the grain feature here; the melt claim is NOT supported "
                   "and should be dropped or reframed as inconclusive.")
    else:
        verdict = ("PARTIAL: a 970 residual exists but is weak or spatially noisy. "
                   "Report as suggestive; keep the melt claim cautious.")
    rep["verdict"] = verdict
    print("\nVERDICT: %s" % verdict)

    np.save(os.path.join(args.outdir, "melt_residual970.npy"), r2d)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    meanX = np.nanmean(X, 0)
    cfit, *_ = np.linalg.lstsq(Afit, meanX[fit_bands], rcond=None)
    ax[0].plot(wlw, meanX, "k-", lw=1.6, label="observed (mean ice)")
    ax[0].plot(wlw, Aall @ cfit, "b--", lw=1.4, label="ice-only model (970 excluded)")
    ax[0].axvspan(PROT_LO, PROT_HI, color="green", alpha=0.15, label="protected 970 window")
    ax[0].axvline(970, color="green", ls=":")
    ax[0].set_xlabel("nm"); ax[0].set_ylabel("reflectance")
    ax[0].set_title("Ice-only model vs observed\n(gap at 970 = extra absorption)")
    ax[0].legend(fontsize=8)
    ax[1].hist(resid970[np.isfinite(resid970)], bins=80, color="tab:purple", alpha=0.8)
    ax[1].axvline(0, color="k"); ax[1].axvline(-noise, color="r", ls="--", label="-1 sigma noise")
    ax[1].set_xlabel("970 nm residual (obs - ice model)"); ax[1].set_ylabel("count")
    ax[1].set_title("extra absorption at 970\nmedian %+.3f (%.1fx noise)" % (
        rep["resid970_median"], rep["resid970_over_noise"])); ax[1].legend(fontsize=8)
    im = ax[2].imshow(r2d, cmap="RdBu", vmin=np.nanpercentile(r2d, 5), vmax=np.nanpercentile(r2d, 95))
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    ax[2].set_title("970 residual map (blue = extra absorption)\nspatial autocorr %.2f" % spatial_r)
    ax[2].axis("off")
    fig.suptitle("Melt control: is the 970 nm signal real and distinct from grain?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "melt_control.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "melt_control.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\nwrote %s" % p)
    print("wrote %s/melt_control.json, melt_residual970.npy" % args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
