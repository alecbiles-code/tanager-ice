#!/usr/bin/env python3
"""
06_stratified_ac.py -- AC diagnostics done correctly: STRATIFIED by surface type.

Supersedes 05_ac_diagnostics.py, which was wrong in three ways:

  (1) POOLED CORRELATIONS (Simpson's paradox). The scene is strongly bimodal:
      bright ice (SR@650 ~ 0.9-1.0) and dark leads/water (SR@650 ~ 0-0.1).
      Correlating any two quantities across the POOLED population measures the
      separation between those two clusters, not any relationship within either.
      05's "CWV aliased with melt, r=+0.41" was manufactured this way. The tell
      was test2b: CWV vs brightness r = -0.689, i.e. CWV reads high exactly
      where the surface is dark -- a dark-target retrieval failure, not aliasing.

  (2) RATIO BLOW-UP. The cirrus index L(1380)/L(1240) divides by a quantity that
      goes to ~0 over leads, so the "cirrus" filaments were just the lead
      network. Fixed here by evaluating only where the denominator is safely
      above noise, and by reporting the index over ice pixels only.

  (3) INVALID CIRRUS ASSUMPTION IN DRY POLAR AIR. The 1380 nm test assumes water
      vapour renders the atmosphere opaque so only high cloud returns signal --
      calibrated for 1-5 g/cm^2 CWV. This scene has CWV ~0.54-0.67 g/cm^2
      (extremely dry). At that column, 1380 nm is NOT opaque and surface signal
      leaks through, so the test cannot distinguish cirrus from surface. It is
      reported here as INCONCLUSIVE-BY-CONSTRUCTION rather than as evidence.

What this script actually answers:
  * Is the scene bimodal, and where is the ice/water split? (Otsu)
  * WITHIN ICE ONLY: does AOD track surface brightness? (real AC contamination test)
  * WITHIN ICE ONLY: is CWV entangled with the 970 nm melt feature? (real aliasing test)
  * WITHIN WATER ONLY: how badly do the AC aux retrievals fail over dark targets?
    (a reportable Planet-product finding in its own right)

Usage:
    python 06_stratified_ac.py
    python 06_stratified_ac.py --ice-thresh 0.5      # override Otsu

Writes: outputs/stratified_ac.png, outputs/stratified_ac.json
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


def otsu(x, nbins=256):
    """Otsu threshold on finite values of x."""
    v = x[np.isfinite(x)]
    if v.size < 100:
        return np.nan
    lo, hi = np.percentile(v, [0.5, 99.5])
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    hist = hist.astype(float)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(mids[k])


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan, int(m.sum())
    if np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def verdict(r, subject):
    if not np.isfinite(r):
        return "UNDEFINED (insufficient variance / samples)"
    a = abs(r)
    if a > 0.4:
        return f"STRONG coupling -> {subject} is contaminated; report it"
    if a < 0.2:
        return f"WEAK coupling -> {subject} looks clean within this stratum"
    return f"MODERATE coupling -> {subject}: caveat, do not lean on it"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--sr-asset", default="ortho_sr_hdf5")
    ap.add_argument("--toa-asset", default="ortho_radiance_hdf5")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--ice-thresh", type=float, default=None)
    args = ap.parse_args()

    meta = io.load_meta(args.meta)

    def cached(asset):
        a = meta.get("assets", {}).get(asset)
        if not a:
            return None
        p = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
        return p if os.path.exists(p) else None

    sr_path = cached(args.sr_asset)
    if sr_path is None:
        sys.exit(f"{args.sr_asset} not cached")
    rep = {}

    with io.Scene(sr_path) as s:
        valid = s.valid_mask()

        b650 = int(np.argmin(np.abs(s.wl_nm - 650)))
        bright, _ = s.read_cube(bands=[b650]); bright = bright[0]
        bright = np.where(valid, bright, np.nan)

        # ---- stratify -------------------------------------------------------
        thr = args.ice_thresh if args.ice_thresh is not None else otsu(bright)
        ice = valid & (bright > thr)
        water = valid & (bright <= thr)
        rep["stratification"] = {
            "method": "otsu on SR@650" if args.ice_thresh is None else "manual",
            "threshold": round(float(thr), 4),
            "ice_fraction_of_valid": round(float(ice.sum() / valid.sum()), 4),
            "water_fraction_of_valid": round(float(water.sum() / valid.sum()), 4)}
        print(f"[strata] Otsu threshold SR@650 = {thr:.3f}")
        print(f"         ice   : {100*ice.sum()/valid.sum():5.1f}% of valid")
        print(f"         water : {100*water.sum()/valid.sum():5.1f}% of valid  "
              f"<- 05's '8.97% cirrus' should be compared to this")

        aod = np.where(valid, s.plane("aerosol_optical_depth"), np.nan)
        cwv = np.where(valid, s.plane("column_water_vapour"), np.nan)

        lo, hi = 930.0, 1050.0
        bmask = (s.wl_nm >= lo - 1) & (s.wl_nm <= hi + 1) & s.good
        win, widx = s.read_cube(bands=np.where(bmask)[0])
        depth = sp.band_depth(np.moveaxis(win, 0, -1), s.wl_nm[widx], 970.0, lo, hi)
        depth = np.where(valid, depth, np.nan)
        del win

        # ---- correlations, pooled vs stratified -----------------------------
        print("\n[corr] pearson r, pooled vs within-stratum:")
        print(f"{'pair':38s} {'POOLED':>9s} {'ICE':>9s} {'WATER':>9s}")
        results = {}
        for name, X, Y in [("AOD vs brightness", aod, bright),
                           ("CWV vs 970nm melt depth", cwv, depth),
                           ("CWV vs brightness", cwv, bright)]:
            rp, _ = corr(X[valid].ravel(), Y[valid].ravel())
            ri, ni = corr(X[ice].ravel(), Y[ice].ravel())
            rw, nw = corr(X[water].ravel(), Y[water].ravel())
            results[name] = {"pooled": _r(rp), "ice": _r(ri), "water": _r(rw),
                             "n_ice": ni, "n_water": nw}
            print(f"{name:38s} {_f(rp):>9s} {_f(ri):>9s} {_f(rw):>9s}")
        rep["correlations"] = results

        print("\n[verdicts] the ones that matter are WITHIN ICE:")
        r_aod_ice = results["AOD vs brightness"]["ice"]
        r_cwv_ice = results["CWV vs 970nm melt depth"]["ice"]
        v_aod = verdict(r_aod_ice, "Planet AOD over ice")
        v_cwv = verdict(r_cwv_ice, "Planet CWV vs surface melt, over ice")
        print(f"  AOD  : {v_aod}")
        print(f"  CWV  : {v_cwv}")
        rep["verdict_aod_over_ice"] = v_aod
        rep["verdict_cwv_over_ice"] = v_cwv

        r_cwv_water = results["CWV vs brightness"]["water"]
        if np.isfinite(r_cwv_water) and abs(r_cwv_water) > 0.3:
            f = ("Planet's CWV retrieval degrades over dark targets (leads/open "
                 "water) -- reportable as a product finding, and a reason to mask "
                 "water before using any AC aux layer.")
            print(f"  WATER: {f}")
            rep["finding_cwv_over_water"] = f

        # ---- plots ----------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        v = bright[np.isfinite(bright)]
        axes[0, 0].hist(v, bins=200, color="steelblue")
        axes[0, 0].axvline(thr, color="r", ls="--", label=f"Otsu {thr:.3f}")
        axes[0, 0].set_yscale("log"); axes[0, 0].legend()
        axes[0, 0].set_xlabel("SR @ 650 nm"); axes[0, 0].set_title(
            "scene is BIMODAL: ice vs water\n(pooling these was the 05 error)")

        for ax, (m, lab, cmap) in zip(
                [axes[0, 1], axes[0, 2]],
                [(ice, "ICE only", "magma"), (water, "WATER only", "viridis")]):
            x, y = depth[m].ravel(), cwv[m].ravel()
            k = np.isfinite(x) & np.isfinite(y)
            if k.sum() > 100:
                ax.hexbin(x[k], y[k], gridsize=55, bins="log", cmap=cmap)
            rr = results["CWV vs 970nm melt depth"]["ice" if lab.startswith("ICE") else "water"]
            ax.set_title(f"CWV vs melt depth -- {lab}   r={_f(rr)}")
            ax.set_xlabel("970 nm band depth"); ax.set_ylabel("CWV (g/cm$^2$)")

        im = axes[1, 0].imshow(np.where(ice, 1, np.where(water, 0, np.nan)),
                               cmap="coolwarm")
        axes[1, 0].set_title("stratification (red=ice, blue=water)")

        im = axes[1, 1].imshow(np.where(ice, depth, np.nan), cmap="magma")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].set_title("970 nm melt proxy, ICE ONLY\n(water masked: feature undefined there)")

        im = axes[1, 2].imshow(np.where(ice, cwv, np.nan), cmap="viridis")
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        axes[1, 2].set_title("Planet CWV, ICE ONLY")

    # ---- cirrus, honestly -----------------------------------------------
    toa_path = cached(args.toa_asset)
    if toa_path:
        with io.Scene(toa_path) as t:
            i1380 = int(np.argmin(np.abs(t.wl_nm - 1380)))
            i1240 = int(np.argmin(np.abs(t.wl_nm - 1240)))
            cir, _ = t.read_cube(bands=[i1380, i1240])
            num, den = cir[0], cir[1]
            # only evaluate where the denominator is well above noise
            safe = np.isfinite(den) & (den > np.nanpercentile(den, 60))
            ratio = np.where(safe, num / np.maximum(den, 1e-6), np.nan)
            fin = ratio[np.isfinite(ratio)]
            rep["cirrus_over_bright_only"] = {
                "p50": round(float(np.percentile(fin, 50)), 5),
                "p99": round(float(np.percentile(fin, 99)), 5),
                "note": ("INCONCLUSIVE BY CONSTRUCTION: CWV ~0.6 g/cm^2 means "
                         "1380 nm is not opaque in this dry polar atmosphere, so "
                         "surface signal leaks through and the standard cirrus "
                         "test cannot separate cloud from surface here. Reported "
                         "for completeness only; visual inspection of the RGB is "
                         "the better evidence, and it shows no cloud.")}
            print(f"\n[cirrus] over bright pixels only: p50={np.percentile(fin,50):.5f} "
                  f"p99={np.percentile(fin,99):.5f}")
            print("         INCONCLUSIVE BY CONSTRUCTION (dry polar air: 1380 nm "
                  "not opaque).")
            print("         05's 'cirrus present' verdict is RETRACTED.")

    os.makedirs(args.outdir, exist_ok=True)
    fig.suptitle("Stratified AC diagnostics -- ice and water analysed separately")
    fig.tight_layout()
    p = os.path.join(args.outdir, "stratified_ac.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    with open(os.path.join(args.outdir, "stratified_ac.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/stratified_ac.json")
    return 0


def _r(x):
    return None if not np.isfinite(x) else round(float(x), 4)


def _f(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


if __name__ == "__main__":
    sys.exit(main())
