#!/usr/bin/env python3
"""
05_ac_diagnostics.py -- is Planet's SR atmospheric correction trustworthy over ice?

Three tests, each of which is a reportable result for the submission:

TEST 1 -- AOD vs surface brightness.
    Dark-target AOD retrieval is ill-posed over bright snow. Planet's AOD here
    IS spatially varying (std 0.028), so it was not defaulted -- but "varying"
    is not "correct". If AOD correlates with surface brightness, the retrieval
    is picking up the SURFACE, not the aerosol, and the SR built on it inherits
    that error.

TEST 2 -- CWV vs the surface liquid-water feature.  *** the important one ***
    Planet's column_water_vapour comes from the ~940 nm vapour band. Surface
    liquid water absorbs at ~970 nm. These OVERLAP. If the CWV retrieval
    attributes surface melt absorption to atmospheric vapour, then:
      (a) CWV is biased high over wet ice,
      (b) CWV partly IS the melt signal, so using it to "deweight" the melt
          index would remove the very thing we are retrieving -- circular,
      (c) the SR itself is mis-corrected wherever this happens.
    High correlation here => do NOT use Planet CWV as an independent melt
    confound control; use the CIBR cross-check and say so.

TEST 3 -- undetected thin cirrus (needs ortho_radiance_hdf5).
    beta_cloud_mask reports 0.0% cloud. In Fram Strait in May that is not
    credible on its face -- cloud/ice discrimination over bright ice is a
    classic hard problem and the mask is flagged 'beta'. The ~1380 nm band is
    THE standard cirrus channel (Landsat-8 B9, Sentinel-2 B10): water vapour
    absorbs the surface signal, so anything bright there is high-altitude ice
    cloud. Note that 1380 nm sits inside the SR product's flagged-bad span
    (1342-1438 nm) -- so this test is only possible on the TOA radiance file,
    using bands the SR product discards.

Usage (repo root):
    python 05_ac_diagnostics.py
    python 05_ac_diagnostics.py --no-cirrus     # skip if TOA not downloaded

Writes: outputs/ac_diagnostics.png, outputs/ac_diagnostics.json
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


def corr(a, b):
    """Pearson r over jointly finite samples."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def subsample(*arrays, n=200_000, seed=0):
    """Thin big flat arrays for correlation/plotting."""
    k = arrays[0].size
    if k <= n:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(k, n, replace=False)
    return tuple(a[idx] for a in arrays)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--sr-asset", default="ortho_sr_hdf5")
    ap.add_argument("--toa-asset", default="ortho_radiance_hdf5")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-cirrus", action="store_true")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    rep = {}

    def cached(asset):
        href = meta["assets"][asset]["href"]
        p = os.path.join("cache", os.path.basename(href.split("?")[0]))
        return p if os.path.exists(p) else None

    sr_path = cached(args.sr_asset)
    if sr_path is None:
        sys.exit(f"{args.sr_asset} not cached -- run 04_quicklook.py first")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    with io.Scene(sr_path) as s:
        valid = s.valid_mask()
        print(f"[scene] {s.rows}x{s.cols}, valid {100*valid.mean():.1f}%")

        # ---------- pull the planes we need ----------
        aod = s.plane("aerosol_optical_depth")
        cwv = s.plane("column_water_vapour")

        # surface brightness at 650 nm (clean of BRF>1 issues; frac_gt1 = 0 there)
        b650 = int(np.argmin(np.abs(s.wl_nm - 650)))
        bright, _ = s.read_cube(bands=[b650])
        bright = bright[0]

        # melt proxy: continuum-removed depth at ~970 nm, computed per pixel.
        # read only the window of bands we need (930-1050 nm) -> cheap.
        lo, hi = 930.0, 1050.0
        bmask = (s.wl_nm >= lo - 1) & (s.wl_nm <= hi + 1) & s.good
        win, widx = s.read_cube(bands=np.where(bmask)[0])
        wl_win = s.wl_nm[widx]
        # move bands last -> (rows, cols, nb) for the vectorised primitive
        cube_last = np.moveaxis(win, 0, -1)
        depth = sp.band_depth(cube_last, wl_win, 970.0, lo, hi)
        del win, cube_last

        # ---------- TEST 1: AOD vs surface brightness ----------
        a = np.where(valid, aod, np.nan).ravel()
        b = np.where(valid, bright, np.nan).ravel()
        a_s, b_s = subsample(a, b)
        r_aod, n1 = corr(a_s, b_s)
        rep["test1_aod_vs_brightness"] = {"pearson_r": round(r_aod, 4), "n": n1,
                                          "aod_std": round(float(np.nanstd(a)), 5)}
        print(f"\n[TEST 1] AOD vs surface brightness(650nm): r = {r_aod:+.3f}  (n={n1:,})")
        v1 = ("UNDEFINED: insufficient variance in AOD or brightness to correlate"
              if not np.isfinite(r_aod) else
              "AOD tracks the surface -> retrieval is contaminated; SR inherits it"
              if abs(r_aod) > 0.4 else
              "weak coupling -> AOD field is plausibly atmospheric" if abs(r_aod) < 0.2
              else "moderate coupling -> treat AOD (and SR) with some caution")
        print(f"         {v1}")
        rep["test1_verdict"] = v1

        axes[0, 0].hexbin(b_s[np.isfinite(b_s) & np.isfinite(a_s)],
                          a_s[np.isfinite(b_s) & np.isfinite(a_s)],
                          gridsize=60, bins="log", cmap="magma")
        axes[0, 0].set_xlabel("SR @ 650 nm (surface brightness)")
        axes[0, 0].set_ylabel("aerosol_optical_depth")
        axes[0, 0].set_title(f"TEST 1: AOD vs brightness  r={r_aod:+.3f}")

        # ---------- TEST 2: CWV vs surface melt feature ----------
        c = np.where(valid, cwv, np.nan).ravel()
        d = np.where(valid, depth, np.nan).ravel()
        c_s, d_s = subsample(c, d)
        r_cwv, n2 = corr(c_s, d_s)
        rep["test2_cwv_vs_melt_feature"] = {"pearson_r": round(r_cwv, 4), "n": n2}
        print(f"\n[TEST 2] CWV vs 970nm surface depth: r = {r_cwv:+.3f}  (n={n2:,})")
        v2 = ("UNDEFINED: insufficient variance to correlate"
              if not np.isfinite(r_cwv) else
              "ALIASED: Planet's CWV and the surface melt feature are entangled. "
              "Do NOT use CWV to deweight melt (circular). Use the CIBR cross-check "
              "and report the aliasing as a finding."
              if abs(r_cwv) > 0.4 else
              "CWV looks largely independent of the surface feature -> usable as a "
              "melt confound control" if abs(r_cwv) < 0.2 else
              "partial entanglement -> use CWV only as a flag, never a correction")
        print(f"         {v2}")
        rep["test2_verdict"] = v2

        m2 = np.isfinite(c_s) & np.isfinite(d_s)
        axes[0, 1].hexbin(d_s[m2], c_s[m2], gridsize=60, bins="log", cmap="viridis")
        axes[0, 1].set_xlabel("continuum-removed depth @ 970 nm (surface melt proxy)")
        axes[0, 1].set_ylabel("column_water_vapour (g/cm$^2$)")
        axes[0, 1].set_title(f"TEST 2: CWV vs melt feature  r={r_cwv:+.3f}")

        # also: CWV vs brightness (a second contamination route)
        c2, b2 = subsample(c, b, seed=1)
        r_cb, _ = corr(c2, b2)
        rep["test2b_cwv_vs_brightness_r"] = round(r_cb, 4)
        print(f"         (CWV vs brightness: r = {r_cb:+.3f})")

        im = axes[0, 2].imshow(np.where(valid, depth, np.nan), cmap="magma")
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        axes[0, 2].set_title("970 nm band depth (melt proxy)")

        im = axes[1, 0].imshow(np.where(valid, cwv, np.nan), cmap="viridis")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        axes[1, 0].set_title("Planet column_water_vapour")

        im = axes[1, 1].imshow(np.where(valid, aod, np.nan), cmap="cividis")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].set_title("Planet aerosol_optical_depth")

    # ---------- TEST 3: cirrus from TOA 1380 nm ----------
    toa_path = None if args.no_cirrus else cached(args.toa_asset)
    if toa_path is None:
        print(f"\n[TEST 3] skipped ({args.toa_asset} not cached). "
              f"Download it to test the 0.0% cloud claim.")
        axes[1, 2].text(0.5, 0.5, "cirrus test skipped\n(TOA radiance not cached)",
                        ha="center", va="center"); axes[1, 2].axis("off")
    else:
        with io.Scene(toa_path) as t:
            valid_t = t.valid_mask()
            i1380 = int(np.argmin(np.abs(t.wl_nm - 1380)))
            i1240 = int(np.argmin(np.abs(t.wl_nm - 1240)))   # clear window reference
            cir, _ = t.read_cube(bands=[i1380, i1240])
            # relative cirrus index: no solar-irradiance table needed, and a flat
            # multiplicative gain cancels.
            ratio = cir[0] / (cir[1] + 1e-9)
            ratio = np.where(valid_t, ratio, np.nan)
            fin = ratio[np.isfinite(ratio)]
            p50, p99 = float(np.percentile(fin, 50)), float(np.percentile(fin, 99))
            hi_frac = float((fin > (p50 + 5 * (np.percentile(fin, 84) - p50))).mean())
            rep["test3_cirrus"] = {
                "band_1380_nm": round(float(t.wl_nm[i1380]), 2),
                "band_1240_nm": round(float(t.wl_nm[i1240]), 2),
                "ratio_p50": round(p50, 5), "ratio_p99": round(p99, 5),
                "high_ratio_fraction": round(hi_frac, 5)}
            print(f"\n[TEST 3] cirrus index L(1380)/L(1240): "
                  f"median {p50:.4f}, p99 {p99:.4f}, "
                  f"anomalous fraction {100*hi_frac:.2f}%")
            v3 = ("cirrus signal present -> the 0.0%% beta cloud mask is missing "
                  "thin cloud; screen before labelling"
                  if hi_frac > 0.01 or p99 > 5 * max(p50, 1e-6) else
                  "no strong cirrus signature -> the clear-sky claim survives this test")
            print(f"         {v3}")
            rep["test3_verdict"] = v3
            im = axes[1, 2].imshow(ratio, cmap="inferno",
                                   vmin=np.nanpercentile(ratio, 2),
                                   vmax=np.nanpercentile(ratio, 98))
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
            axes[1, 2].set_title("cirrus index  L(1380)/L(1240)")

    fig.suptitle("Planet SR atmospheric-correction diagnostics over Fram Strait ice")
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "ac_diagnostics.png")
    fig.savefig(p, dpi=125); plt.close(fig)
    with open(os.path.join(args.outdir, "ac_diagnostics.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/ac_diagnostics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
