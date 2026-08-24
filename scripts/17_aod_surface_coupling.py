#!/usr/bin/env python3
"""
17_aod_surface_coupling.py -- quantify how Planet/ISOFIT's retrieved aerosol
optical depth couples to the surface over Arctic snow and ice. Converts the
reviewer's biggest liability into a headline, Planet-relevant finding.

THE OBSERVATION. Retrieved AOD over this June Baffin scene averages ~0.70, where
climatological Arctic summer AOD is ~0.05-0.15 (MODIS/AERONET). And the AOD map
traces surface features -- individual valleys -- which a true aerosol field
(smooth over tens of km) cannot do.

WHY IT MATTERS. ISOFIT is an optimal-estimation joint retrieval of atmosphere
AND surface; over bright, spectrally flat snow this inversion is known to be
ill-posed, so surface structure can leak into the atmospheric terms. That leak
lands in the 940-1030 nm region where the grain and melt retrievals live -- so
this is not a footnote, it is a direct control on which scenes are trustworthy
to release. That is exactly the decision Planet's prize governs.

THE GUARD (don't overclaim). A high AOD could be real June smoke. The tell is
not the magnitude but the SPATIAL COUPLING to the surface WITHIN a class:
aerosol cannot correlate with surface brightness inside a single surface type,
nor step sharply at the coastline. So we report:
  (a) per-class AOD-vs-brightness correlation (within-class coupling),
  (b) the cross-class AOD step (does AOD jump at the land/sea boundary?),
  (c) the AOD magnitude vs a cited climatology range.
We frame magnitude as "anomalous vs climatology," not "wrong," and state
plainly that ill-posedness over bright snow is a known property, not a defect
unique to this product.

Usage: python 17_aod_surface_coupling.py
Writes: outputs/aod_coupling.png, outputs/aod_coupling.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tanager_ice import io

# cited climatology anchor (Arctic summer background AOD, MODIS/AERONET range)
CLIM_LO, CLIM_HI = 0.05, 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--labels", default="outputs/segment_labels.npy")
    ap.add_argument("--land", default="outputs/land_mask.npy")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")
    labels2d = np.load(args.labels, allow_pickle=True)
    seg = json.load(open(os.path.join(args.outdir, "segment_report.json")))
    id2name = {v: k for k, v in seg.get("final_class_ids", {}).items()}
    land = np.load(args.land) if os.path.exists(args.land) else None

    with io.Scene(path) as s:
        valid = s.valid_mask(); wl = s.wl_nm
        b650 = int(np.argmin(np.abs(wl - 650)))
        bright, _ = s.read_cube(bands=[b650]); bright = np.where(valid, bright[0], np.nan)
        aod = s.plane("aerosol_optical_depth")
        aod = np.where(valid, aod, np.nan) if aod is not None else None
        if aod is None:
            sys.exit("no aerosol_optical_depth layer in this product")
        H, W = valid.shape

    rep = {"aod_mean": round(float(np.nanmean(aod)), 3),
           "aod_median": round(float(np.nanmedian(aod)), 3),
           "aod_p5_p95": [round(float(np.nanpercentile(aod, 5)), 3),
                          round(float(np.nanpercentile(aod, 95)), 3)],
           "climatology_range": [CLIM_LO, CLIM_HI]}
    rep["anomaly_factor_vs_climatology"] = round(float(np.nanmean(aod)) / ((CLIM_LO + CLIM_HI) / 2), 1)

    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    # (a) per-class AOD vs brightness coupling
    lab = labels2d
    rep["per_class"] = {}
    print(f"\n{'class':16s} {'n':>8s} {'AODmean':>8s} {'AOD~bright r':>12s}")
    print("-" * 50)
    couplings = {}
    for cid, name in id2name.items():
        m = valid & (lab == cid)
        if m.sum() < 300:
            continue
        r = corr(aod[m], bright[m])
        couplings[name] = r
        rep["per_class"][name] = {"n": int(m.sum()),
                                  "aod_mean": round(float(np.nanmean(aod[m])), 3),
                                  "aod_vs_brightness_r": None if not np.isfinite(r) else round(r, 3)}
        print(f"{name:16s} {int(m.sum()):8d} {np.nanmean(aod[m]):8.3f} {r:12.3f}")

    # (b) cross-class AOD step (land vs sea, if land mask present)
    if land is not None:
        aod_land = float(np.nanmean(aod[valid & land]))
        aod_sea = float(np.nanmean(aod[valid & ~land]))
        rep["aod_land_mean"] = round(aod_land, 3)
        rep["aod_sea_mean"] = round(aod_sea, 3)
        rep["land_sea_aod_step"] = round(abs(aod_land - aod_sea), 3)
        print(f"\nland AOD {aod_land:.3f}  sea AOD {aod_sea:.3f}  "
              f"step {abs(aod_land-aod_sea):.3f}")

    # ---- verdict ----
    max_coupling = np.nanmax([abs(v) for v in couplings.values() if np.isfinite(v)]) \
        if couplings else float("nan")
    step = rep.get("land_sea_aod_step", np.nan)
    rep["max_within_class_coupling"] = round(float(max_coupling), 3) if np.isfinite(max_coupling) else None

    surface_driven = (np.isfinite(max_coupling) and max_coupling > 0.3) or \
                     (np.isfinite(step) and step > 0.2)
    if surface_driven:
        verdict = ("ISOFIT AOD is partly SURFACE-DRIVEN over this scene: it "
                   f"averages {rep['aod_mean']} (~{rep['anomaly_factor_vs_climatology']}x "
                   "Arctic-summer climatology) and couples to surface brightness "
                   f"within classes (max |r|={max_coupling:.2f})"
                   + (f" and steps {step:.2f} across the coastline" if np.isfinite(step) else "")
                   + ". Consistent with the known ill-posedness of joint atmosphere/"
                   "surface retrieval over bright, spectrally flat snow. Implication: "
                   "absolute reflectance carries an AC-dependent term here, so "
                   "retrievals should be SHAPE-based (as ours are), and this scene's "
                   "AC quality is a factor in release/interpretation decisions.")
    else:
        verdict = ("AOD is elevated but NOT strongly surface-coupled: consistent "
                   "with genuine elevated aerosol (e.g. transported smoke). The AC "
                   "auxiliary layers can be used with normal caution.")
    rep["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    im = ax[0, 0].imshow(aod, cmap="inferno"); plt.colorbar(im, ax=ax[0, 0], fraction=0.046)
    ax[0, 0].set_title(f"retrieved AOD (mean {rep['aod_mean']})\n"
                       f"~{rep['anomaly_factor_vs_climatology']}x climatology "
                       f"({CLIM_LO}-{CLIM_HI})"); ax[0, 0].axis("off")
    im = ax[0, 1].imshow(bright, cmap="gray"); plt.colorbar(im, ax=ax[0, 1], fraction=0.046)
    ax[0, 1].set_title("surface brightness (650 nm)\ncompare structure to AOD"); ax[0, 1].axis("off")
    names = list(couplings.keys()); vals = [couplings[n] for n in names]
    colors = ["tab:red" if abs(v) > 0.3 else "tab:blue" for v in vals]
    ax[1, 0].barh(names, vals, color=colors)
    ax[1, 0].axvline(0, color="k", lw=0.8)
    ax[1, 0].axvline(0.3, color="grey", ls="--"); ax[1, 0].axvline(-0.3, color="grey", ls="--")
    ax[1, 0].set_xlabel("AOD vs brightness r (within class)")
    ax[1, 0].set_title("within-class coupling\n(|r|>0.3 = surface leaking into AOD)")
    # scatter for the most-coupled class
    if couplings:
        worst = max(couplings, key=lambda k: abs(couplings[k]) if np.isfinite(couplings[k]) else 0)
        wid = [k for k, v in id2name.items() if v == worst][0]
        m = valid & (lab == wid)
        ax[1, 1].scatter(bright[m].ravel()[::5], aod[m].ravel()[::5], s=3, alpha=0.2)
        ax[1, 1].set_xlabel("surface brightness"); ax[1, 1].set_ylabel("AOD")
        ax[1, 1].set_title(f"'{worst}': AOD vs brightness  r={couplings[worst]:.2f}")
    fig.suptitle("Does ISOFIT's AOD track the surface? (AC trust for release decisions)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "aod_coupling.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "aod_coupling.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/aod_coupling.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
