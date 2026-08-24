#!/usr/bin/env python3
"""
04_quicklook.py -- health check the SR cube and produce a labelling basemap.

Answers, before any modelling:
  1. Is the -9999 sample just ortho corner-fill, or is the scene actually empty?
  2. Are SR values physical over bright ice (0-1, no negatives)? Negative SR in
     the VNIR over snow is the classic signature of an over-corrected
     atmosphere (dark-target AOD retrieval failing on a bright surface).
  3. Which bands does Planet flag bad (good_wavelengths)?
  4. Did the AC actually retrieve AOD here, or fall back to a default? A
     spatially-constant AOD over a 36 km scene = fallback = trust SR less.
     This is the direct test of the "AC tuned for land" worry.
  5. What is column water vapour doing (the melt-retrieval confound)?
  6. An RGB quicklook + grid overlay so you can pick label pixels by row/col.

Usage (repo root):
    python 04_quicklook.py                          # ortho_sr_hdf5
    python 04_quicklook.py --asset ortho_radiance_hdf5
    python 04_quicklook.py --stride 2               # finer preview (slower)

Writes: outputs/quicklook_<asset>.png     RGB + grid for labelling
        outputs/diagnostics_<asset>.png   AOD / CWV / sun / valid maps
        outputs/spectra_<asset>.png       sample spectra from bright+dark pixels
        outputs/quicklook_report.json     the numbers
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


def pct(x):
    return f"{100*x:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--stride", type=int, default=1,
                    help="spatial subsample for the RGB preview")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    href = meta["assets"][args.asset]["href"]
    path = os.path.join("cache", os.path.basename(href.split("?")[0]))
    if not os.path.exists(path):
        print(f"[fetch] {args.asset} not cached; downloading ~1 GB ...")
        path = io.download(href)
    os.makedirs(args.outdir, exist_ok=True)

    rep = {"asset": args.asset, "file": path}
    with io.Scene(path) as s:
        print(repr(s))
        rep.update(cube=s.cube_name, is_reflectance=bool(s.is_reflectance),
                   shape=[s.n_bands, s.rows, s.cols], epsg=int(s.epsg) if s.epsg else None,
                   strip_id=str(s.strip_id), fields=s.fields())

        # ---- 3. band quality ------------------------------------------------
        nbad = int((~s.good).sum())
        bad_wl = s.wl_nm[~s.good]
        rep["bands_good"] = int(s.good.sum()); rep["bands_bad"] = nbad
        print(f"\n[bands] good {int(s.good.sum())}/{s.n_bands}  (dropped {nbad})")
        if nbad:
            # summarise contiguous bad spans
            idx = np.where(~s.good)[0]
            spans, start = [], idx[0]
            for a, b in zip(idx[:-1], idx[1:]):
                if b != a + 1:
                    spans.append((s.wl_nm[start], s.wl_nm[a])); start = b
            spans.append((s.wl_nm[start], s.wl_nm[idx[-1]]))
            rep["bad_spans_nm"] = [[round(a, 1), round(b, 1)] for a, b in spans]
            for a, b in spans:
                print(f"        bad span: {a:7.1f} - {b:7.1f} nm")
        for label, tgt in [("grain 1030", 1030), ("melt 970", 970),
                           ("sediment 650", 650), ("swir 2300", 2300)]:
            j = int(np.argmin(np.abs(s.wl_nm - tgt)))
            print(f"        {label:13s} -> band {j:3d} @ {s.wl_nm[j]:7.2f} nm  "
                  f"good={bool(s.good[j])}")
            rep.setdefault("key_bands", {})[label] = {
                "band": j, "wl_nm": round(float(s.wl_nm[j]), 2),
                "good": bool(s.good[j])}

        # ---- 1. validity ----------------------------------------------------
        valid = s.valid_mask()
        rep["valid_fraction"] = round(float(valid.mean()), 4)
        nodata = s.mask("nodata_pixels")
        cloud = s.mask("beta_cloud_mask")
        cirrus = s.mask("beta_cirrus_mask")
        print(f"\n[valid] usable pixels: {pct(valid.mean())} of {s.rows}x{s.cols}")
        for nm, m in [("nodata", nodata), ("cloud", cloud), ("cirrus", cirrus)]:
            if m is not None:
                f = float((m != 0).mean())
                rep[f"frac_{nm}"] = round(f, 4)
                print(f"        {nm:7s}: {pct(f)}")

        # ---- 2. SR physicality over valid pixels ---------------------------
        probe_bands = [int(np.argmin(np.abs(s.wl_nm - t)))
                       for t in (450, 550, 650, 865, 1030, 1250, 1600, 2200)]
        cube, bidx = s.read_cube(bands=probe_bands)
        stats = {}
        print(f"\n[values] per-band stats over valid pixels "
              f"({'reflectance 0-1 expected' if s.is_reflectance else 'radiance'}):")
        for k, b in enumerate(bidx):
            v = cube[k][valid]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            neg = float((v < 0).mean()); hi = float((v > 1).mean())
            stats[int(b)] = {"wl_nm": round(float(s.wl_nm[b]), 1),
                             "min": round(float(v.min()), 4),
                             "p50": round(float(np.median(v)), 4),
                             "max": round(float(v.max()), 4),
                             "frac_negative": round(neg, 4),
                             "frac_gt1": round(hi, 4)}
            flag = ""
            if s.is_reflectance and neg > 0.01:
                flag = "  <-- NEGATIVE SR: atmospheric over-correction"
            elif s.is_reflectance and hi > 0.01:
                flag = "  <-- SR > 1"
            print(f"         {s.wl_nm[b]:7.1f} nm  min={v.min():8.4f} "
                  f"med={np.median(v):7.4f} max={v.max():8.4f} "
                  f"neg={pct(neg):>6s}{flag}")
        rep["band_stats"] = stats

        # ---- 4/5. AC diagnostics -------------------------------------------
        print("\n[atmos] correction inputs:")
        aux = {}
        for nm, unit in [("aerosol_optical_depth", ""),
                         ("column_water_vapour", "g/cm^2"),
                         ("sun_zenith", "deg"), ("sensor_zenith", "deg")]:
            a = s.plane(nm)
            if a is None:
                continue
            v = a[valid]; v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            uniq = int(np.unique(np.round(v, 6)).size)
            aux[nm] = {"min": round(float(v.min()), 4),
                       "mean": round(float(v.mean()), 4),
                       "max": round(float(v.max()), 4),
                       "std": round(float(v.std()), 6),
                       "n_unique": uniq}
            note = ""
            if nm == "aerosol_optical_depth":
                if uniq <= 2 or v.std() < 1e-6:
                    note = "  <-- CONSTANT: AOD not retrieved, AC used a fallback"
                else:
                    note = "  <-- spatially varying: AOD genuinely retrieved"
            print(f"        {nm:28s} mean={v.mean():9.4f} std={v.std():9.6f} "
                  f"uniq={uniq:6d} {unit}{note}")
        rep["atmos"] = aux

        # ---- 6. RGB quicklook ----------------------------------------------
        st = max(1, args.stride)
        rgb_bands = [int(np.argmin(np.abs(s.wl_nm - t))) for t in (650, 550, 450)]
        rgbc, _ = s.read_cube(bands=rgb_bands)
        rgb = np.stack([rgbc[i][::st, ::st] for i in range(3)], -1).astype(float)
        fin = np.isfinite(rgb)
        if fin.any():
            lo, hi = np.nanpercentile(rgb[fin], [2, 98])
            rgbn = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
        else:
            rgbn = np.zeros_like(rgb)
        rgbn[~fin] = 0.0

        fig, ax = plt.subplots(figsize=(11, 10))
        ax.imshow(rgbn, interpolation="nearest")
        step = max(50, (s.cols // st) // 12)
        ax.set_xticks(np.arange(0, rgbn.shape[1], step))
        ax.set_xticklabels((np.arange(0, rgbn.shape[1], step) * st))
        ax.set_yticks(np.arange(0, rgbn.shape[0], step))
        ax.set_yticklabels((np.arange(0, rgbn.shape[0], step) * st))
        ax.grid(color="yellow", alpha=0.35, lw=0.5)
        ax.set_xlabel("col"); ax.set_ylabel("row")
        ax.set_title(f"{args.asset}  RGB (650/550/450 nm), 2-98% stretch\n"
                     f"use these row/col for labels.csv")
        fig.tight_layout()
        q = os.path.join(args.outdir, f"quicklook_{args.asset}.png")
        fig.savefig(q, dpi=130); plt.close(fig)
        print(f"\nwrote {q}")

        # ---- diagnostics panel ---------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for axx, nm in zip(axes.ravel(),
                           ["aerosol_optical_depth", "column_water_vapour",
                            "sun_zenith", None]):
            if nm is None:
                axx.imshow(valid[::st, ::st], cmap="gray")
                axx.set_title(f"valid mask ({pct(valid.mean())} usable)")
                continue
            a = s.plane(nm)
            if a is None:
                axx.axis("off"); continue
            a = np.where(valid, a, np.nan)[::st, ::st]
            im = axx.imshow(a, cmap="viridis")
            plt.colorbar(im, ax=axx, fraction=0.046)
            axx.set_title(nm)
        fig.suptitle(f"{args.asset} atmospheric-correction diagnostics")
        fig.tight_layout()
        d = os.path.join(args.outdir, f"diagnostics_{args.asset}.png")
        fig.savefig(d, dpi=120); plt.close(fig)
        print(f"wrote {d}")

        # ---- sample spectra -------------------------------------------------
        bright_idx = np.argmax(np.where(valid, cube[4], -np.inf))  # 1030 nm plane
        dark_idx = np.argmin(np.where(valid, cube[4], np.inf))
        br, bc = np.unravel_index(bright_idx, valid.shape)
        dr, dc = np.unravel_index(dark_idx, valid.shape)
        rows = np.array([br, dr]); cols = np.array([bc, dc])
        X, gidx = s.read_labeled_pixels(rows, cols)
        U, _ = (s.read_labeled_pixels(rows, cols,
                dataset="surface_reflectance_uncertainty")
                if s.has("surface_reflectance_uncertainty") else (None, None))
        wlg = s.wl_nm[gidx]
        fig, ax = plt.subplots(figsize=(11, 5))
        for k, (lab, rc) in enumerate([("brightest px", (br, bc)),
                                       ("darkest px", (dr, dc))]):
            ax.plot(wlg, X[k], lw=1.1, label=f"{lab} (r{rc[0]},c{rc[1]})")
            if U is not None:
                ax.fill_between(wlg, X[k]-U[k], X[k]+U[k], alpha=0.25, lw=0)
        for w, nm in [(1030, "grain"), (970, "melt"), (650, "sed")]:
            ax.axvline(w, color="k", ls=":", alpha=0.4)
            ax.text(w, ax.get_ylim()[1]*0.96, nm, fontsize=8, rotation=90, va="top")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("surface reflectance" if s.is_reflectance else "TOA radiance")
        ax.set_title(f"{args.asset}: sample spectra (shaded = Planet's per-pixel "
                     f"uncertainty)" if U is not None else args.asset)
        ax.legend(); fig.tight_layout()
        sp = os.path.join(args.outdir, f"spectra_{args.asset}.png")
        fig.savefig(sp, dpi=130); plt.close(fig)
        print(f"wrote {sp}")
        rep["sample_pixels"] = {"bright": [int(br), int(bc)], "dark": [int(dr), int(dc)]}

    rp = os.path.join(args.outdir, "quicklook_report.json")
    with open(rp, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {rp}")

    print("\n=================== VERDICT ===================")
    v = rep["valid_fraction"]
    print(f"usable pixels     : {pct(v)}  "
          f"{'-> corner fill only, scene is fine' if v > 0.5 else '-> INVESTIGATE'}")
    aod = rep.get("atmos", {}).get("aerosol_optical_depth", {})
    if aod:
        const = aod.get("std", 1) < 1e-6 or aod.get("n_unique", 99) <= 2
        print(f"AOD retrieval     : {'FALLBACK (constant)' if const else 'retrieved'}"
              f"  -> {'treat SR with caution; lean on TOA-shape arm' if const else 'SR AC had real aerosol input'}")
    negs = [b["frac_negative"] for b in rep.get("band_stats", {}).values()]
    if negs:
        print(f"negative SR (max) : {pct(max(negs))}  "
              f"{'-> over-correction present' if max(negs) > 0.01 else '-> clean'}")
    print("==============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
