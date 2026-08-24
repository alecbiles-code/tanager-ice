#!/usr/bin/env python3
"""
10_segment.py -- k-way surface segmentation + per-class SR trust gate.

Two jobs, one script (they share feature extraction):

  JOB 1  SEGMENT. The Baffin scene has 4+ surface types (snow terrain, sea ice,
         melt-blue ice, open water, coast, cloud). A single Otsu ice/water split
         -- which 06/08 use -- would mis-stratify it. Here we cluster on a few
         PHYSICALLY chosen features (not raw 426 bands: curse of dimensionality
         + 1.6 GB) with a Gaussian mixture, then NAME the clusters from their
         feature signatures. Every downstream script gets a real class map.

  JOB 2  AC TRUST GATE. This scene's AOD is huge (mean 0.70 vs Fram 0.06) and
         shows a step at the coastline in the map. Either it is real June haze
         (fine) or ISOFIT AOD absorbed a surface/BRDF signal (SR then suspect).
         The discriminator is per-CLASS: within a single surface class, aerosol
         should NOT correlate with surface brightness. If it does, and if AOD
         steps across class boundaries, the SR is contaminated. Reassuring prior:
         the bright SR spectrum is spectrally clean (0% BRF>1, strong ice
         features), which argues for real haze -- but we test, not assume.

Features (all robust to a flat atmospheric gain; chosen from what the RGB showed):
    f1 R650                          brightness      (snow/ice hi, water/land lo)
    f2 (R550-R865)/(R550+R865)       NDWI-like       (water vs solid)
    f3 1030 nm ice band depth        ice presence    (land has none)
    f4 970 nm melt band depth        surface wetness (melt-ice vs dry)
    f5 (R650-R550)/(R650+R550)       VNIR slope      (snow flat, land red)

Usage (repo root):
    python 10_segment.py                     # auto k via BIC (3..7)
    python 10_segment.py --k 5               # fixed k
    python 10_segment.py --asset ortho_sr_hdf5

Writes: outputs/segmentation.png, outputs/segment_labels.npy (H x W int),
        outputs/segment_report.json, outputs/class_seeds.csv (labels.csv seed)

Deps: h5py, numpy, scikit-learn, scipy, matplotlib
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

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("scikit-learn required: conda install -c conda-forge scikit-learn")


FEATURE_NAMES = ["R650", "NDWI", "ice1030", "melt970", "vnir_slope", "roughness"]


def _local_std(a, w=2):
    """Local standard deviation in a (2w+1) window, nan-aware, cheap."""
    from scipy.ndimage import uniform_filter
    m = np.isfinite(a)
    x = np.where(m, a, 0.0)
    n = uniform_filter(m.astype(float), size=2 * w + 1)
    mean = uniform_filter(x, size=2 * w + 1) / np.maximum(n, 1e-6)
    mean2 = uniform_filter(x * x, size=2 * w + 1) / np.maximum(n, 1e-6)
    var = np.maximum(mean2 - mean * mean, 0.0)
    out = np.sqrt(var)
    out[~m] = np.nan
    return out


def build_features(scene, valid):
    """Return (H*W, 5) feature matrix (nan where invalid) + the 2-D shapes."""
    wl = scene.wl_nm
    need = [450, 550, 650, 865]
    # read the handful of single bands we need for f1/f2/f5 (cheap)
    singles = {w: int(np.argmin(np.abs(wl - w))) for w in need}
    planes = {}
    cube, idx = scene.read_cube(bands=list(singles.values()))
    for w, b in zip(need, idx):
        planes[w] = np.where(valid, cube[list(idx).index(b)], np.nan)

    # ice + melt band depths need small windows (read once each)
    def depth(center, lo, hi):
        m = (wl >= lo - 1) & (wl <= hi + 1) & scene.good
        win, widx = scene.read_cube(bands=np.where(m)[0])
        d = sp.band_depth(np.moveaxis(win, 0, -1), wl[widx], center, lo, hi)
        return np.where(valid, d, np.nan)

    ice = depth(1030.0, 960.0, 1080.0)
    melt = depth(970.0, 930.0, 1050.0)

    R550, R650, R865 = planes[550], planes[650], planes[865]
    ndwi = (R550 - R865) / (R550 + R865 + 1e-9)
    slope = (R650 - R550) / (R650 + R550 + 1e-9)
    rough = _local_std(R650, w=2)      # dendritic terrain rough; floe interiors smooth

    feats = np.stack([R650, ndwi, ice, melt, slope, rough], axis=-1)  # (H,W,6)
    H, W, _ = feats.shape
    return feats.reshape(H * W, 6), (H, W)


def choose_k(X, kmin=3, kmax=7, seed=0):
    """Suggest k from the BIC 'knee', but this is only a HINT.

    Raw BIC on imagery keeps improving with k (homogeneous regions subdivide),
    so the minimum is unreliable. We instead find the knee: the k after which
    added components stop buying much BIC improvement. The caller shows the
    segmentation figure so the human can override with --k after LOOKING. That
    is more honest than a heuristic that cannot self-validate on unseen scenes.
    """
    fits, bic = {}, {}
    for k in range(kmin, kmax + 1):
        gm = GaussianMixture(k, covariance_type="full", random_state=seed,
                             n_init=3, reg_covar=1e-5).fit(X)
        fits[k], bic[k] = gm, gm.bic(X)
    ks = sorted(bic)
    vals = np.array([bic[k] for k in ks])
    # normalised improvement per step; knee = last k with >15% of the max drop
    drops = -np.diff(vals)                       # positive where BIC improves
    if drops.max() <= 0:
        knee = kmin
    else:
        rel = drops / drops.max()
        signif = [ks[i + 1] for i in range(len(drops)) if rel[i] > 0.15]
        knee = max(signif) if signif else kmin
    return fits[knee], {k: float(v) for k, v in bic.items()}


def name_clusters(gm, scaler):
    """Assign human names from each cluster's feature signature (unscaled means)."""
    means = scaler.inverse_transform(gm.means_)   # (k,6)
    names = []
    # roughness threshold = median of the per-cluster roughness means; terrain
    # clusters sit above it, smooth ice/water below.
    rough_thresh = float(np.median(means[:, 5]))
    for m in means:
        r650, ndwi, ice, melt, slope, rough = m
        if r650 < 0.15:
            name = "open_water"        # dark is water/shadow, whatever NDWI does
        elif ice > 0.06 and r650 > 0.3 and rough > rough_thresh:
            name = "snow_terrain"      # bright + ice feature + ROUGH = snow on land
        elif ice > 0.15 and r650 > 0.55 and melt > 0.05:
            name = "melt_ice"
        elif ice > 0.13 and r650 > 0.7:
            name = "snow_ice"          # bright, strong ice feature, smooth
        elif ice > 0.06 and r650 > 0.3:
            name = "sea_ice"           # intermediate, ice feature, smooth
        elif slope > 0.03 and r650 >= 0.15:
            name = "land_snowfree"     # reddened, no ice feature = exposed ground
        else:
            name = "mixed_or_other"
    # de-duplicate names by suffixing
        names.append(name)
    # ensure uniqueness
    seen = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out, means


def spatial_relabel(labels2d, names, valid):
    """Split spectrally-identical snow_terrain vs sea_ice by SPATIAL context.

    Texture alone cannot separate dendritic terrain from a broken brash/floe
    field (both are rough at 33 m). But land is one CONTIGUOUS mass touching the
    scene, while sea ice is patches embedded in water. We derive the land region
    morphologically -- no coastline file -- then rename solid pixels by whether
    they fall inside the land mass or on the water side.

    Returns (per_pixel_name_array, land_mask) or None if no coherent landmass.
    """
    try:
        from scipy.ndimage import (binary_fill_holes, binary_closing,
                                    label as cc_label)
    except ImportError:
        return None

    k = len(names)
    name_of = {i: names[i] for i in range(k)}
    water_ids = [i for i, n in name_of.items() if n.startswith("open_water")]
    solid = valid & ~np.isin(labels2d, water_ids) & (labels2d >= 0)

    solid_closed = binary_fill_holes(binary_closing(solid, iterations=3))
    lab_cc, ncc = cc_label(solid_closed)
    if ncc == 0:
        return None
    sizes = np.bincount(lab_cc.ravel()); sizes[0] = 0
    land_mask = (lab_cc == int(np.argmax(sizes)))
    if land_mask.sum() < 0.05 * valid.sum():
        return None

    out = np.empty(labels2d.shape, dtype=object); out[:] = ""
    for i in range(k):
        out[labels2d == i] = name_of[i]

    ambiguous = {"snow_terrain", "sea_ice", "melt_ice", "land_snowfree", "snow_ice"}
    for i in range(k):
        nm = name_of[i]
        if nm not in ambiguous:
            continue
        cls = (labels2d == i)
        if nm == "melt_ice":
            out[cls & land_mask] = "snow_terrain"
            out[cls & ~land_mask] = "melt_ice"
        elif "terrain" in nm or "land" in nm:
            out[cls & land_mask] = "snow_terrain"
            out[cls & ~land_mask] = "sea_ice"      # rough ICE mislabelled as land
        else:                                       # sea_ice / snow_ice
            out[cls & land_mask] = "snow_terrain"   # snow on land
            out[cls & ~land_mask] = nm
    return out, land_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--sample", type=int, default=60000,
                    help="pixels used to fit the mixture")
    ap.add_argument("--land-mask", default=None,
                    help="path to land_mask.npy from 11_dem_landmask.py; finalises "
                         "snow_terrain vs sea_ice from DEM elevation (authoritative)")
    ap.add_argument("--spatial-relabel", action="store_true",
                    help="attempt morphological land/sea relabel (OFF by default; "
                         "unreliable where landfast ice welds sea to coast -- use "
                         "--land-mask from a DEM instead)")

    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached -- run 04_quicklook.py first")

    rep = {"asset": args.asset}
    with io.Scene(path) as s:
        valid = s.valid_mask()
        print(f"[seg] scene {s.rows}x{s.cols}, valid {100*valid.mean():.1f}%")
        X_all, (H, W) = build_features(s, valid)
        aod = np.where(valid, s.plane("aerosol_optical_depth"), np.nan).ravel()

        good_rows = np.isfinite(X_all).all(1)
        Xg = X_all[good_rows]
        print(f"[seg] {good_rows.sum():,} valid feature pixels")

        # fit on a sample, predict all
        rng = np.random.default_rng(0)
        samp = Xg[rng.choice(len(Xg), min(args.sample, len(Xg)), replace=False)]
        scaler = StandardScaler().fit(samp)
        samp_s = scaler.transform(samp)

        if args.k:
            gm = GaussianMixture(args.k, covariance_type="full", random_state=0,
                                 n_init=3, reg_covar=1e-5).fit(samp_s)
            bics = {args.k: float(gm.bic(samp_s))}
        else:
            gm, bics = choose_k(samp_s)
        k = gm.n_components
        print(f"[seg] suggested k = {k} (BIC knee; a HINT -- confirm from the "
              f"figure, override with --k). BICs: " +
              ", ".join(f"{kk}:{vv:.0f}" for kk, vv in bics.items()))

        names, means = name_clusters(gm, scaler)
        lab_g = gm.predict(scaler.transform(Xg))
        labels = np.full(H * W, -1, int)
        labels[good_rows] = lab_g
        labels2d = labels.reshape(H, W)

        # naming resolution priority: DEM land mask > morphological relabel > raw
        name_map2d = None
        dem_land = None
        if args.land_mask and os.path.exists(args.land_mask):
            dem_land = np.load(args.land_mask)
            if dem_land.shape != labels2d.shape:
                print(f"[seg] WARNING land mask shape {dem_land.shape} != scene "
                      f"{labels2d.shape}; ignoring")
                dem_land = None
        if dem_land is not None:
            # authoritative: land pixels -> terrain names, sea pixels -> ice/water
            name_map2d = np.empty(labels2d.shape, dtype=object)
            name_map2d[:] = ""
            for i in range(k):
                name_map2d[labels2d == i] = names[i]
            ambiguous = {"snow_terrain", "sea_ice", "melt_ice", "snow_ice",
                         "land_snowfree"}
            for i in range(k):
                nm = names[i]
                if nm not in ambiguous:
                    continue
                cls = labels2d == i
                # on land: snow/ice spectra -> snow_terrain; reddened -> land_snowfree
                on_land = cls & dem_land
                on_sea = cls & ~dem_land & valid
                if "land" in nm:
                    name_map2d[on_land] = nm            # keep land_snowfree
                else:
                    name_map2d[on_land] = "snow_terrain"
                # on sea: keep melt distinction, else sea_ice
                name_map2d[on_sea] = "melt_ice" if nm == "melt_ice" else "sea_ice"
            resolved = sorted(set(name_map2d[name_map2d != ""].tolist()))
            print(f"[seg] names finalised from DEM land mask "
                  f"({100*dem_land[valid].mean():.0f}% land)")
            print(f"[seg] resolved classes: {resolved}")
        elif args.spatial_relabel:
            relabel = spatial_relabel(labels2d, names, valid)
            if relabel is not None:
                name_map2d, land_mask = relabel
                print(f"[seg] morphological relabel; land = "
                      f"{100*land_mask.sum()/valid.sum():.1f}% of valid")
        else:
            print("[seg] provisional spectral names only; pass --land-mask "
                  "outputs/land_mask.npy to finalise terrain vs sea-ice.")

        # ---- report per FINAL class + the AC gate --------------------------
        # group pixels by RESOLVED name (post spatial-relabel), not raw cluster,
        # so the gate measures the corrected classes.
        if name_map2d is not None:
            final_name_flat = name_map2d.reshape(-1)[good_rows]
        else:
            final_name_flat = np.array([names[c] for c in lab_g], dtype=object)
        aod_g = aod[good_rows]
        r650_g = Xg[:, 0]

        print(f"\n{'class':16s} {'n':>8s} {'%':>6s} {'R650':>6s} {'ice':>6s} "
              f"{'melt':>6s} {'AODmean':>7s} {'AOD~bright r':>12s}")
        print("-" * 74)
        classes = {}
        gate_flags = {}
        for nm in sorted(set(final_name_flat.tolist())):
            if nm == "":
                continue
            m = final_name_flat == nm
            n = int(m.sum())
            frac = n / len(final_name_flat)
            aod_c = aod_g[m]; r650_c = r650_g[m]
            ice_c = Xg[m, 2]; melt_c = Xg[m, 3]
            fin = np.isfinite(aod_c) & np.isfinite(r650_c)
            if fin.sum() > 200 and np.std(aod_c[fin]) > 1e-6 and np.std(r650_c[fin]) > 1e-6:
                r = float(np.corrcoef(aod_c[fin], r650_c[fin])[0, 1])
            else:
                r = float("nan")
            gate_flags[nm] = r
            classes[nm] = {
                "n": n, "frac": round(frac, 4),
                "R650": round(float(np.nanmean(r650_c)), 4),
                "ice1030": round(float(np.nanmean(ice_c)), 4),
                "melt970": round(float(np.nanmean(melt_c)), 4),
                "aod_mean": round(float(np.nanmean(aod_c)), 4),
                "aod_vs_brightness_r": None if not np.isfinite(r) else round(r, 4),
            }
            print(f"{nm:16s} {n:8d} {100*frac:5.1f}% {np.nanmean(r650_c):6.2f} "
                  f"{np.nanmean(ice_c):6.3f} {np.nanmean(melt_c):+6.3f} "
                  f"{np.nanmean(aod_c):7.3f} {r:+12.3f}")
        rep["k"] = k
        rep["bic"] = bics
        rep["classes"] = classes

        # ---- AC trust verdict ----------------------------------------------
        # KEY: judge FLAT classes separately from terrain. Terrain illumination
        # legitimately couples brightness<->AOD; that is NOT an SR failure, it is
        # a topographic-correction requirement. The SR-trust question is really:
        # are the FLAT surfaces (sea ice, melt ice, water) clean?
        FLAT = {"sea_ice", "melt_ice", "open_water", "mixed_or_other",
                "mixed_or_other_2", "mixed_or_other_3"}
        TERRAIN = {"snow_terrain", "land_snowfree"}
        flat_r = [abs(gate_flags[n]) for n in gate_flags
                  if n in FLAT and np.isfinite(gate_flags[n])]
        terr_r = [abs(gate_flags[n]) for n in gate_flags
                  if n in TERRAIN and np.isfinite(gate_flags[n])]
        max_flat = max(flat_r) if flat_r else float("nan")
        max_terr = max(terr_r) if terr_r else float("nan")
        class_aod = [classes[n]["aod_mean"] for n in classes]
        aod_spread = float(np.nanmax(class_aod) - np.nanmin(class_aod))
        rep["ac_gate"] = {
            "max_flat_class_aod_brightness_r": None if not np.isfinite(max_flat) else round(max_flat, 4),
            "max_terrain_class_aod_brightness_r": None if not np.isfinite(max_terr) else round(max_terr, 4),
            "class_mean_aod_spread": round(aod_spread, 4),
        }
        print("\n" + "=" * 74)
        print("AC TRUST GATE (does AOD corrupt the SR?)  -- flat vs terrain")
        print("=" * 74)
        print(f"  max |AOD~bright r| over FLAT ice/water : {max_flat:.3f}")
        print(f"  max |AOD~bright r| over TERRAIN        : {max_terr:.3f}")
        if np.isfinite(max_flat) and max_flat < 0.25:
            flat_verdict = ("FLAT SURFACES CLEAN: SR trustworthy over sea ice / "
                            "melt ice / water. Retrievals proceed directly there.")
        else:
            flat_verdict = ("FLAT SURFACES SHOW COUPLING: investigate before "
                            "trusting SR even on flat ice.")
        if np.isfinite(max_terr) and max_terr > 0.3:
            terr_verdict = ("TERRAIN COUPLED (expected): brightness<->illumination"
                            "<->AOD entangled over slopes. Grain size on terrain "
                            "REQUIRES topographic correction (DEM) before trust.")
        else:
            terr_verdict = "terrain coupling weak."
        print(f"  -> {flat_verdict}")
        print(f"  -> {terr_verdict}")
        rep["ac_gate"]["flat_verdict"] = flat_verdict
        rep["ac_gate"]["terrain_verdict"] = terr_verdict
        verdict = flat_verdict + " | " + terr_verdict
        rep["ac_gate"]["verdict"] = verdict

        # ---- figures --------------------------------------------------------
        # remap to an integer field over FINAL names for display
        final_names = sorted(set(classes.keys()))
        name_to_int = {nm: i for i, nm in enumerate(final_names)}
        disp = np.full(labels2d.shape, np.nan)
        if name_map2d is not None:
            for nm, i in name_to_int.items():
                disp[name_map2d == nm] = i
        else:
            for c in range(k):
                disp[labels2d == c] = name_to_int.get(names[c], np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        import matplotlib.colors as mcolors
        base = list(mcolors.TABLEAU_COLORS.values())
        K = len(final_names)
        cmap = mcolors.ListedColormap(base[:K])
        im = axes[0].imshow(disp, cmap=cmap, vmin=0, vmax=K - 1)
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, ticks=range(K))
        cbar.ax.set_yticklabels(final_names)
        axes[0].set_title(f"surface segmentation (k={k}, spatially relabelled)")

        aod2d = np.where(valid, s.plane("aerosol_optical_depth"), np.nan)
        im = axes[1].imshow(aod2d, cmap="inferno")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title(f"AOD (mean {np.nanmean(aod2d):.2f})")
        fig.suptitle(f"Baffin {meta['id']}: segmentation + AC gate")
        fig.tight_layout()
        p = os.path.join(args.outdir, "segmentation.png")
        fig.savefig(p, dpi=130); plt.close(fig)

        # save the FINAL integer label map + its legend
        np.save(os.path.join(args.outdir, "segment_labels.npy"), disp)
        rep["final_class_ids"] = name_to_int

        # class seeds for labels.csv: sample interior pixels per class
        try:
            from scipy.ndimage import binary_erosion
            with open(os.path.join(args.outdir, "class_seeds.csv"), "w") as f:
                f.write("row,col,class\n")
                for nm in final_names:
                    m = (disp == name_to_int[nm])
                    me = binary_erosion(m, iterations=2)
                    if me.sum() < 10:
                        me = m
                    rr, cc = np.where(me)
                    if len(rr) == 0:
                        continue
                    pick = rng.choice(len(rr), min(80, len(rr)), replace=False)
                    for r_, c_ in zip(rr[pick], cc[pick]):
                        f.write(f"{r_},{c_},{nm}\n")
            print(f"\nwrote {args.outdir}/class_seeds.csv  (seed for labels.csv)")
        except Exception as e:
            print(f"[warn] class_seeds skipped: {e}")

    with open(os.path.join(args.outdir, "segment_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {p}")
    print(f"wrote {args.outdir}/segment_labels.npy")
    print(f"wrote {args.outdir}/segment_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
