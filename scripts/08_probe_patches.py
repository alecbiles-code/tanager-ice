#!/usr/bin/env python3
"""
08_probe_patches.py -- what ARE the grey patches? dirty ice, wet ice, or thin ice?

This decides the class list, the narrative, and the headline, so it runs before
any labelling.

THE THREE HYPOTHESES make different predictions on different spectral axes.
No single index separates them; the JOINT pattern does:

    candidate     1030 ice feature   970 melt    VNIR slope   NIR level
    ------------------------------------------------------------------
    DIRTY ice     PRESERVED          unchanged   REDDENED     high
    WET ice       deeper             POSITIVE    ~unchanged   moderate
    THIN ice      WEAKENED           unchanged   blue-steep   LOW
    (clean floe)  strong             negative    flat         high

Reasoning: sediment sits ON the ice, so ice absorption survives while VNIR
reddens. Meltwater is IN the ice, so 970 nm liquid-water absorption appears.
Thin ice has less ice path length, so absorption features weaken and the
spectrum drifts toward the dark, blue-dominated water endmember.

*** WET ICE IS NOT GREY IN THE RED. *** Liquid water absorbs in the NIR; at
650 nm wet snow stays bright (~0.95). So anything that looks GREY in an RGB
composite cannot be simply wet ice -- darkening at 650 nm needs absorbing
impurities (sediment) or water mixing (thin ice). Melt is therefore hunted as a
SEPARATE group ('wet_candidate': interior ice, top decile of 970 nm depth,
still bright in red), because the grey filter would miss it entirely.

CRITICAL DESIGN POINT -- grey pixels at floe EDGES are just mixed ice/water
pixels and carry no information about material. Grey pixels deep INSIDE floes
are real material. So candidates are selected using the distance transform:
intermediate brightness AND far from any water. Without that, this probe would
simply resample the edge population (which is what made 06/07 misleading).

BRF caveat: BRF>1 at 451 nm affects ABSOLUTE reflectance. But sun_zenith varies
only ~0.23 deg and sensor_zenith ~3 deg across the scene, so the BRDF geometry
is near-identical for every group -- RELATIVE comparison between groups (what
this script does) is valid. Absolute values still carry the caveat.

Usage (repo root):
    python 08_probe_patches.py
    python 08_probe_patches.py --min-dist 4 --grey-lo 0.30 --grey-hi 0.75
    python 08_probe_patches.py --points 660,520 730,560 700,800   # manual probe

Writes: outputs/probe_patches.png, outputs/probe_patches.json,
        outputs/patch_candidates.csv   (row,col,group -> seed for labels.csv)
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
    from scipy.ndimage import distance_transform_edt, label as cc_label
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def read_scattered(scene, rows, cols, dataset=None):
    """Spectra at scattered points WITHOUT reading their bounding box.

    read_labeled_pixels() reads the bbox of all points -- if points are spread
    across the scene that is the whole 1.6 GB cube. Here we read one band plane
    at a time (~3.7 MB peak) and index the points out of it.
    """
    d = scene.data[dataset] if dataset else scene._cube
    idx = np.where(scene.good)[0]
    X = np.empty((len(rows), len(idx)), np.float32)
    for i, b in enumerate(idx):
        plane = d[int(b)]
        X[:, i] = plane[rows, cols]
    X[X == scene.fill] = np.nan
    return X, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--grey-lo", type=float, default=0.30,
                    help="lower SR@650 bound for 'grey'")
    ap.add_argument("--grey-hi", type=float, default=0.75,
                    help="upper SR@650 bound for 'grey'")
    ap.add_argument("--min-dist", type=int, default=3,
                    help="min px from water (rejects edge mixed pixels)")
    ap.add_argument("--n-sample", type=int, default=400,
                    help="pixels sampled per group")
    ap.add_argument("--points", nargs="*", default=None,
                    help="manual probe points as row,col")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")
    if not HAVE_SCIPY:
        sys.exit("scipy required: conda install -c conda-forge scipy")

    rng = np.random.default_rng(0)
    rep = {}

    with io.Scene(path) as s:
        valid = s.valid_mask()
        b650 = int(np.argmin(np.abs(s.wl_nm - 650)))
        bright, _ = s.read_cube(bands=[b650])
        bright = np.where(valid, bright[0], np.nan)

        # ---- strata (same Otsu split as 06/07) --------------------------
        v = bright[np.isfinite(bright)]
        lo, hi = np.percentile(v, [0.5, 99.5])
        hist, edges = np.histogram(v, bins=256, range=(lo, hi))
        p = hist / hist.sum(); omega = np.cumsum(p)
        mids = (edges[:-1] + edges[1:]) / 2; mu = np.cumsum(p * mids)
        den = omega * (1 - omega); den[den == 0] = np.nan
        thr = float(mids[int(np.nanargmax((mu[-1] * omega - mu) ** 2 / den))])
        ice = valid & (bright > thr)
        water = valid & (bright <= thr)

        # distance from water, in pixels -- the key to rejecting edge mixing
        dist = distance_transform_edt(ice)
        interior = ice & (dist >= args.min_dist)
        print(f"[strata] Otsu {thr:.3f} | ice {100*ice.mean():.1f}% | "
              f"interior(>= {args.min_dist}px from water) {100*interior.mean():.1f}%")

        # ---- define groups ------------------------------------------------
        groups = {}
        if args.points:
            pts = [tuple(int(x) for x in p.split(",")) for p in args.points]
            rr = np.array([p[0] for p in pts]); cc = np.array([p[1] for p in pts])
            groups["manual_probe"] = (rr, cc)
        else:
            # GREY = intermediate brightness, in floe interiors (NOT edges)
            grey = interior & (bright > args.grey_lo) & (bright < args.grey_hi)
            # CLEAN = brightest interior ice
            clean = interior & (bright > np.nanpercentile(bright[interior], 90))
            # WATER = dark, away from ice
            wdist = distance_transform_edt(water)
            openw = water & (wdist >= args.min_dist)
            # EDGE-GREY = grey but AT edges -> included as a contrast control,
            # to show the interior grey is not the same population
            edge_grey = ice & (dist < args.min_dist) & \
                        (bright > args.grey_lo) & (bright < args.grey_hi)
            # WET candidates: wet ice is NOT grey in the red (liquid water
            # absorbs in the NIR; R650 stays high), so the grey filter would
            # miss melt entirely. Hunt it separately: interior ice in the top
            # decile of 970 nm band depth.
            bmask_w = (s.wl_nm >= 929) & (s.wl_nm <= 1051) & s.good
            win_w, widx_w = s.read_cube(bands=np.where(bmask_w)[0])
            depth_map = sp.band_depth(np.moveaxis(win_w, 0, -1), s.wl_nm[widx_w],
                                      970.0, 930.0, 1050.0)
            del win_w
            depth_map = np.where(interior, depth_map, np.nan)
            if np.isfinite(depth_map).any():
                d90 = np.nanpercentile(depth_map, 90)
                wet = interior & (depth_map > d90) & (bright > args.grey_hi)
            else:
                wet = np.zeros_like(interior)

            for name, m in [("grey_interior", grey), ("clean_floe", clean),
                            ("lead_water", openw), ("wet_candidate", wet),
                            ("grey_edge_control", edge_grey)]:
                n = int(m.sum())
                print(f"         {name:20s} {n:8d} px  ({100*n/max(valid.sum(),1):5.2f}% of valid)")
                if n < 20:
                    print(f"           -> too few, SKIPPED (group will be absent)")
                    continue
                ridx, cidx = np.where(m)
                pick = rng.choice(len(ridx), min(args.n_sample, len(ridx)), replace=False)
                groups[name] = (ridx[pick], cidx[pick])
            rep["group_sizes"] = {k: int(m.sum()) for k, m in
                                  [("grey_interior", grey), ("clean_floe", clean),
                                   ("lead_water", openw), ("wet_candidate", wet),
                                   ("grey_edge_control", edge_grey)]}
            if int(grey.sum()) < 20:
                print("\n[!] NO interior grey found in the brightness window "
                      f"[{args.grey_lo}, {args.grey_hi}]. Either the grey in the RGB "
                      "is all floe-edge mixing, or the window needs widening "
                      "(--grey-lo/--grey-hi). The RGB was percentile-stretched, so "
                      "display-grey != reflectance-grey.")

        if "grey_interior" in rep.get("group_sizes", {}) and \
                rep["group_sizes"]["grey_interior"] < 50:
            print("\n[!] very little INTERIOR grey -- the grey seen in the RGB may be")
            print("    mostly floe-edge mixing. Interpret with care.")

        # ---- pull spectra --------------------------------------------------
        print("\n[read] extracting spectra (band-plane at a time, low memory)...")
        spectra, sigmas = {}, {}
        for name, (rr, cc) in groups.items():
            X, gidx = read_scattered(s, rr, cc)
            spectra[name] = X
            if s.has("surface_reflectance_uncertainty"):
                U, _ = read_scattered(s, rr, cc,
                                      dataset="surface_reflectance_uncertainty")
                sigmas[name] = U
            print(f"       {name:20s} {X.shape}")
        wl = s.wl_nm[np.where(s.good)[0]]

    # ---- diagnostic indices -------------------------------------------
    def idx_of(w):
        return int(np.argmin(np.abs(wl - w)))

    print("\n" + "=" * 78)
    print("DIAGNOSTIC INDICES (mean +/- std across sampled pixels)")
    print("=" * 78)
    hdr = f"{'group':20s} {'R450':>8s} {'R650':>8s} {'R865':>8s} " \
          f"{'ice1030':>9s} {'melt970':>9s} {'vnir_sl':>8s} {'nir/blu':>8s}"
    print(hdr); print("-" * 78)

    table = {}
    for name, X in spectra.items():
        m = np.nanmean(X, 0)
        ice_area = sp.scaled_band_area(X, wl, 960, 1080)
        melt_d = sp.band_depth(X, wl, 970, 930, 1050)
        r450, r650, r865 = X[:, idx_of(450)], X[:, idx_of(650)], X[:, idx_of(865)]
        # slope anchored at 550 (not 450) -- 451 nm is the BRF>1 zone
        r550 = X[:, idx_of(550)]
        vnir_slope = (r650 - r550) / (r650 + r550 + 1e-9)
        nir_blue = r865 / (r450 + 1e-9)
        table[name] = {
            "n": int(X.shape[0]),
            "R450": _ms(r450), "R550": _ms(r550), "R650": _ms(r650),
            "R865": _ms(r865),
            "ice_band_area_1030": _ms(ice_area),
            "melt_depth_970": _ms(melt_d),
            "vnir_slope_550_650": _ms(vnir_slope),
            "nir_blue_ratio": _ms(nir_blue),
        }
        print(f"{name:20s} {np.nanmean(r450):8.3f} {np.nanmean(r650):8.3f} "
              f"{np.nanmean(r865):8.3f} {np.nanmean(ice_area):9.2f} "
              f"{np.nanmean(melt_d):+9.3f} {np.nanmean(vnir_slope):+8.4f} "
              f"{np.nanmean(nir_blue):8.3f}")
    rep["indices"] = table

    # ---- verdict --------------------------------------------------------
    if "grey_interior" in table and "clean_floe" in table:
        g, c = table["grey_interior"], table["clean_floe"]
        d_ice = g["ice_band_area_1030"][0] / max(c["ice_band_area_1030"][0], 1e-9)
        d_melt = g["melt_depth_970"][0] - c["melt_depth_970"][0]
        d_slope = g["vnir_slope_550_650"][0] - c["vnir_slope_550_650"][0]
        d_nir = g["R865"][0] / max(c["R865"][0], 1e-9)
        rep["grey_vs_clean"] = {
            "ice_feature_ratio": round(float(d_ice), 3),
            "melt_depth_delta": round(float(d_melt), 4),
            "vnir_slope_delta": round(float(d_slope), 4),
            "nir_level_ratio": round(float(d_nir), 3)}

        print("\n" + "=" * 78)
        print("GREY vs CLEAN FLOE")
        print("=" * 78)
        print(f"  1030 nm ice feature : {d_ice:6.3f} x   "
              f"({'PRESERVED' if d_ice > 0.8 else 'WEAKENED' if d_ice < 0.6 else 'reduced'})")
        print(f"  970 nm melt depth   : {d_melt:+6.3f}    "
              f"({'MORE liquid water' if d_melt > 0.02 else 'no melt signal'})")
        print(f"  VNIR slope (550-650): {d_slope:+6.4f}    "
              f"({'REDDENED' if d_slope > 0.01 else 'not reddened'})")
        print(f"  NIR level (865)     : {d_nir:6.3f} x   "
              f"({'much darker' if d_nir < 0.6 else 'similar'})")

        votes = []
        if d_ice > 0.8 and d_slope > 0.01 and d_nir > 0.6:
            votes.append("DIRTY ICE (sediment): ice features intact, VNIR reddened, "
                         "still bright in NIR")
        if d_melt > 0.02 and d_ice > 0.8:
            votes.append("WET ICE (melt): 970 nm liquid-water absorption deepened")
        if d_ice < 0.6 and d_nir < 0.6:
            votes.append("THIN ICE (nilas): ice features weakened and NIR collapsed "
                         "-> drifting toward the water endmember")
        if not votes:
            votes.append("AMBIGUOUS: no hypothesis fits cleanly. Inspect the spectra "
                         "plot; may be a mixture, or the grey may be edge artifact.")
        rep["verdict"] = votes
        print("\n  VERDICT:")
        for v in votes:
            print(f"    -> {v}")

        if "grey_edge_control" in table:
            e = table["grey_edge_control"]
            same = abs(e["ice_band_area_1030"][0] - g["ice_band_area_1030"][0]) < \
                0.15 * max(g["ice_band_area_1030"][0], 1e-9)
            match_txt = "MATCHES" if same else "DIFFERS from"
            concl = ("interior grey may just be mixing" if same
                     else "interior grey is a DISTINCT population from edge mixing")
            print("\n  edge-grey control : ice feature "
                  f"{match_txt} interior grey")
            print(f"    -> {concl}")
            rep["edge_control_matches_interior"] = bool(same)

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"grey_interior": "tab:orange", "clean_floe": "tab:blue",
              "lead_water": "tab:cyan", "grey_edge_control": "tab:red", "wet_candidate": "tab:purple",
              "manual_probe": "tab:green"}

    ax = axes[0, 0]
    for name, X in spectra.items():
        m = np.nanmean(X, 0); sd = np.nanstd(X, 0)
        ax.plot(wl, m, lw=1.3, label=f"{name} (n={X.shape[0]})",
                color=colors.get(name), ls="--" if "control" in name else "-")
        ax.fill_between(wl, m - sd, m + sd, alpha=0.15, lw=0, color=colors.get(name))
    for w, lab in [(970, "melt"), (1030, "grain"), (650, "sed")]:
        ax.axvline(w, color="k", ls=":", alpha=0.35)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("surface reflectance")
    ax.set_title("mean spectra (+/- 1 sd across pixels)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, X in spectra.items():
        m = np.nanmean(X, 0)
        k = idx_of(865)
        ax.plot(wl, m / max(m[k], 1e-9), lw=1.3, color=colors.get(name),
                ls="--" if "control" in name else "-", label=name)
    ax.set_xlim(380, 1300); ax.axvline(970, color="k", ls=":", alpha=0.35)
    ax.axvline(1030, color="k", ls=":", alpha=0.35)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("reflectance / R(865)")
    ax.set_title("SHAPE only (normalised at 865 nm)\n"
                 "separates 'darker' from 'different material'"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, X in spectra.items():
        w_win, cr = sp.continuum_removed(np.nanmean(X, 0), wl, 900, 1100)
        ax.plot(w_win, cr, lw=1.4, color=colors.get(name),
                ls="--" if "control" in name else "-", label=name)
    ax.axvline(970, color="k", ls=":", alpha=0.35)
    ax.axvline(1030, color="k", ls=":", alpha=0.35)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("continuum-removed")
    ax.set_title("ice absorption 900-1100 nm\n(preserved=dirty, deeper=wet, "
                 "weak=thin)"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, X in spectra.items():
        ice_area = sp.scaled_band_area(X, wl, 960, 1080)
        r550, r650 = X[:, idx_of(550)], X[:, idx_of(650)]
        slope = (r650 - r550) / (r650 + r550 + 1e-9)
        ax.scatter(slope, ice_area, s=6, alpha=0.45, color=colors.get(name),
                   label=name)
    ax.set_xlabel("VNIR slope (550-650)  ->  redder = more impurity")
    ax.set_ylabel("1030 nm ice band area  ->  more ice")
    ax.set_title("the discriminating plane"); ax.legend(fontsize=8)

    fig.suptitle("What are the grey patches? Fram Strait MIZ, 2025-05-16")
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "probe_patches.png")
    fig.savefig(p, dpi=130); plt.close(fig)

    with open(os.path.join(args.outdir, "probe_patches.json"), "w") as f:
        json.dump(rep, f, indent=2)
    if not args.points:
        with open(os.path.join(args.outdir, "patch_candidates.csv"), "w") as f:
            f.write("row,col,group\n")
            for name, (rr, cc) in groups.items():
                for r_, c_ in zip(rr, cc):
                    f.write(f"{r_},{c_},{name}\n")
        print(f"\nwrote {args.outdir}/patch_candidates.csv  (seed for labels.csv)")
    print(f"wrote {p}")
    print(f"wrote {args.outdir}/probe_patches.json")
    return 0


def _ms(x):
    return [round(float(np.nanmean(x)), 5), round(float(np.nanstd(x)), 5)]


if __name__ == "__main__":
    sys.exit(main())
