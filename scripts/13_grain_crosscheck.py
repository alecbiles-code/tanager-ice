#!/usr/bin/env python3
"""
13_grain_crosscheck.py -- diagnose the grain-size retrieval's validation.

The 1030-vs-1250 cross-check gave r=0.99 on synthetic but r=0.05 on the real
scene. Before trusting (or discarding) the grain map, LOOK at the real spectra
and test several independent checks -- don't guess which is right.

Checks computed (all against the 1030 nm scaled band-area grain proxy):
  A. 1250 nm band area   -- the original weak second feature (why did it fail?)
  B. 1030 nm band DEPTH  -- OPERATOR sanity: depth & area of the SAME feature
                            must correlate ~1 if the retrieval is coded right.
  C. NIR level @ ~1100nm -- PHYSICS: Nolin-Dozier says coarser grains -> deeper
                            ice absorption -> LOWER ice-band reflectance. Proxy
                            should ANTI-correlate with 1100 nm reflectance. This
                            is the robust physical check.
  D. continuum-removed mean spectra per class, plotted, so we SEE whether a
     1250 feature even exists over this ice and where its shoulders belong.

Outputs a verdict on which check validates the retrieval, and per-class means.

Usage: python 13_grain_crosscheck.py
Writes: outputs/grain_crosscheck.png, outputs/grain_crosscheck.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--labels", default="outputs/segment_labels.npy")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--sample", type=int, default=40000)
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    labels2d = np.load(args.labels, allow_pickle=True)
    seg = json.load(open(os.path.join(args.outdir, "segment_report.json")))
    id2name = {v: k for k, v in seg.get("final_class_ids", {}).items()}

    with io.Scene(path) as s:
        valid = s.valid_mask()
        wl = s.wl_nm
        sel = np.where((wl >= 900) & (wl <= 1350) & s.good)[0]
        R, _ = s.read_cube(bands=sel)
        R = np.moveaxis(R, 0, -1)
        wlw = wl[sel]
        H, W = valid.shape
        fv = valid.reshape(-1)
        Rf = R.reshape(H * W, -1)[fv]
        lab = labels2d.reshape(-1)[fv]

    # subsample for speed
    rng = np.random.default_rng(0)
    keep = np.isfinite(Rf).all(1)
    Rf, lab = Rf[keep], lab[keep]
    if len(Rf) > args.sample:
        pick = rng.choice(len(Rf), args.sample, replace=False)
        Rf, lab = Rf[pick], lab[pick]

    # the four quantities
    area_1030 = sp.scaled_band_area(Rf, wlw, 960, 1080)
    area_1250 = sp.scaled_band_area(Rf, wlw, 1180, 1300)
    depth_1030 = sp.band_depth(Rf, wlw, 1030, 960, 1080)
    i1100 = int(np.argmin(np.abs(wlw - 1100)))
    nir_1100 = Rf[:, i1100]

    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    checks = {
        "A_1250_area": round(corr(area_1030, area_1250), 3),
        "B_1030_depth_operator": round(corr(area_1030, depth_1030), 3),
        "C_nir1100_physics": round(corr(area_1030, nir_1100), 3),
    }
    rep = {"checks": checks}

    print("\n=== GRAIN CROSS-CHECKS (vs 1030 nm area proxy) ===")
    print(f"  A. 1250 nm area (weak 2nd band)   r = {checks['A_1250_area']:+.3f}")
    print(f"  B. 1030 nm depth (operator sanity) r = {checks['B_1030_depth_operator']:+.3f}"
          "   expect ~+1")
    print(f"  C. 1100 nm NIR level (physics)     r = {checks['C_nir1100_physics']:+.3f}"
          "   expect strongly NEGATIVE")

    verdict = []
    if abs(checks["B_1030_depth_operator"]) > 0.9:
        verdict.append("OPERATOR SOUND: area and depth of the 1030 feature agree "
                       "-> the retrieval is coded correctly.")
    else:
        verdict.append("OPERATOR SUSPECT: area and depth disagree -> bug in the "
                       "band-area computation, investigate before anything else.")
    if checks["C_nir1100_physics"] < -0.5:
        verdict.append("PHYSICS CONFIRMED: proxy anti-correlates with 1100 nm "
                       "reflectance as Nolin-Dozier predicts (coarser->darker ice "
                       "bands). This is the real validation; the 1250 check was "
                       "just too weak/noisy to serve.")
    elif checks["C_nir1100_physics"] > 0.5:
        verdict.append("PHYSICS INVERTED: proxy tracks NIR the wrong way -- the "
                       "proxy may be picking up brightness, not grain. Serious.")
    else:
        verdict.append("PHYSICS WEAK: NIR anti-correlation is weak; grain signal "
                       "may be marginal over this ice.")
    rep["verdict"] = verdict
    for v in verdict:
        print(f"  -> {v}")

    # per-class continuum-removed mean spectra
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    import matplotlib.cm as cm
    classes = [c for c in np.unique(lab)]
    colors = cm.tab10(np.linspace(0, 1, len(classes)))
    for c, col in zip(classes, colors):
        m = lab == c
        if m.sum() < 50:
            continue
        mean = np.nanmean(Rf[m], 0)
        nm = id2name.get(int(c), str(int(c)))
        ax[0].plot(wlw, mean, color=col, label=f"{nm} (n={m.sum()})")
        w_cr, cr = sp.continuum_removed(mean, wlw, 900, 1350)
        ax[1].plot(w_cr, cr, color=col, label=nm)
    for a_ in ax:
        for x in (1030, 1250):
            a_.axvline(x, color="k", ls=":", alpha=0.4)
    ax[0].set_title("per-class mean spectra (900-1350 nm)")
    ax[0].set_xlabel("nm"); ax[0].set_ylabel("SR"); ax[0].legend(fontsize=8)
    ax[1].set_title("continuum-removed (does a 1250 feature exist?)")
    ax[1].set_xlabel("nm"); ax[1].set_ylabel("CR"); ax[1].legend(fontsize=8)
    fig.suptitle("Grain retrieval cross-check diagnostics")
    fig.tight_layout()
    p = os.path.join(args.outdir, "grain_crosscheck.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    # per-class proxy + physics-check values
    rep["per_class"] = {}
    for c in classes:
        m = lab == c
        if m.sum() < 50:
            continue
        rep["per_class"][id2name.get(int(c), str(int(c)))] = {
            "n": int(m.sum()),
            "grain_1030_area": round(float(np.nanmedian(area_1030[m])), 3),
            "depth_1030": round(float(np.nanmedian(depth_1030[m])), 4),
            "nir_1100": round(float(np.nanmedian(nir_1100[m])), 4),
        }

    with open(os.path.join(args.outdir, "grain_crosscheck.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/grain_crosscheck.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
