#!/usr/bin/env python3
"""
run_separability.py  --  Task 2 on-ramp: prove the classes are distinct.

Workflow this supports (all before any modelling):
    1. run 01_scene_recon.py -> outputs/scene_meta.json (+ data_asset_href)
    2. download & open the HDF5, reshape to a pixel table
    3. hand-label representative pixels for the five classes (see LABELS below);
       provide them as a CSV: row,col,label   (or edit the stub loader)
    4. this script: class-mean spectra + variability, pairwise Jeffries-Matusita
       and spectral-angle matrices, and a separability verdict.

If classes are NOT separable, that reshapes the project -- and this is where
you find out, cheaply, before spending effort on retrievals.

Usage:
    python run_separability.py --labels labels.csv
    python run_separability.py --demo        # runs on synthetic data, no scene needed
"""
from __future__ import annotations
import argparse, csv, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tanager_ice import spectral as sp
from tanager_ice import separability as sep

LABELS = ["iceberg_ice", "sea_ice", "lead_water", "dirty_ice", "wet_snow"]


def load_scene_pixels(meta_path):
    """Open the real scene and return (X pixels, wl, row/col idx)."""
    from tanager_ice import io
    meta = io.load_meta(meta_path)
    h5 = io.download(meta["data_asset_href"])
    scene = io.open_scene(h5)
    wl = scene["wl_nm"]
    if not np.isfinite(wl).all():          # STAC carries wavelengths if HDF5 didn't
        wl = np.array([b for b in _wl_from_meta(meta)])
    X, idx, _ = io.to_pixel_table(scene, valid_only=True)
    return X, wl, idx


def _wl_from_meta(meta):
    # scene_meta.json records spectral_range + count; exact centres live in the
    # STAC item's band list. Simplest: refetch item bands if needed. Stub here.
    lo, hi = meta["product"]["spectral_range_nm"]
    n = meta["product"]["n_spectral_bands"]
    return np.linspace(lo, hi, n)


def load_labels(path, X_full, idx_full):
    """CSV with header row,col,label -> (X_labeled, y)."""
    lut = {tuple(rc): i for i, rc in enumerate(map(tuple, idx_full.tolist()))}
    Xs, ys = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["row"]), int(r["col"]))
            if key in lut:
                Xs.append(X_full[lut[key]]); ys.append(r["label"])
    return np.array(Xs), np.array(ys)


def demo_data(wl, n_per=120, rng=None):
    """Synthetic five-class scene for a dry run of the pipeline."""
    rng = rng or np.random.default_rng(1)
    def base(): return np.ones_like(wl) * 20.0
    protos = {}
    c1030 = sp.nearest_index(wl, 1030)
    for lab in LABELS:
        s = base()
        if lab in ("iceberg_ice", "sea_ice", "wet_snow"):     # ice absorption depth varies
            depth = {"iceberg_ice": 0.35, "sea_ice": 0.20, "wet_snow": 0.45}[lab]
            s -= np.clip(1 - np.abs(np.arange(len(wl)) - c1030) / 10, 0, 1) * depth * 20
        if lab == "lead_water":
            s *= np.clip((wl.max() - wl) / (wl.max() - wl.min()), 0.02, 1)  # dark, blue-ish
        if lab == "dirty_ice":
            s[wl < 700] *= 0.6                                  # reddened VNIR
        if lab == "wet_snow":
            s -= np.exp(-0.5 * ((wl - 970) / 25) ** 2) * 3      # extra liquid-water dip
        protos[lab] = s
    X = np.vstack([protos[l] + rng.normal(0, 0.6, (n_per, len(wl))) for l in LABELS])
    y = np.array([l for l in LABELS for _ in range(n_per)])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--labels")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--pca-k", type=int, default=8)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    wl = np.arange(380, 2500, 5.0)
    if args.demo:
        X, y = demo_data(wl)
    else:
        Xfull, wl, idx = load_scene_pixels(args.meta)
        if not args.labels:
            sys.exit("Provide --labels labels.csv (row,col,label) or use --demo.")
        X, y = load_labels(args.labels, Xfull, idx)

    cs = sep.class_spectra(X, y)
    labs, JM = sep.pairwise_jm(X, y, pca_k=args.pca_k)
    # spectral-angle matrix on class means
    means = {l: cs[l]["mean"] for l in cs}
    order = list(means.keys())
    SAM = np.array([[sep.spectral_angle(means[a], means[b]) for b in order] for a in order])

    print("\n=============== TASK 2: CLASS SEPARABILITY ===============")
    print("class            n     mean@1030   grain  melt  sediment")
    for l in order:
        m = cs[l]["mean"]
        g = sp.grain_size_index(m, wl)
        md, _ = sp.melt_index(m, wl)
        sed = sp.sediment_index(m, wl)
        print(f"{l:15s} {cs[l]['n']:4d}   {m[sp.nearest_index(wl,1030)]:8.2f}"
              f"   {g:5.1f} {md:5.2f} {sed:+.3f}")
    print("\nJeffries-Matusita (0..2; >1.9 = well separated):")
    print("            " + "".join(f"{l[:9]:>10s}" for l in labs))
    for i, a in enumerate(labs):
        print(f"{a[:11]:11s} " + "".join(f"{JM[i,j]:10.2f}" for j in range(len(labs))))
    worst = min(JM[i, j] for i in range(len(labs)) for j in range(i + 1, len(labs)))
    print(f"\nweakest pair JM = {worst:.2f}  ->  "
          f"{'SEPARABLE: proceed to retrievals' if worst > 1.6 else 'WEAK: classes overlap; rethink class set / features'}")
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "separability.npz"),
             labels=np.array(order), JM=JM, SAM=SAM,
             means=np.array([means[l] for l in order]), wl=wl)
    print(f"wrote {args.outdir}/separability.npz")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
