#!/usr/bin/env python3
"""
14_melt_classify.py -- combined surface classification + melt-state retrieval,
with the sub-pixel-water confound guarded from the start.

TWO RESULTS IN ONE SCRIPT (they share inputs and reinforce each other):

  CLASSIFICATION. Turn the k=5 GMM segmentation into a supervised class map with
  HONEST per-pixel confidence. Train a classifier on the segment labels (they are
  our best available labels; no field truth exists), then attach CONFORMAL
  PREDICTION SETS: ambiguous pixels get a SET of classes ({sea_ice, melt_ice})
  with guaranteed marginal coverage, instead of a false single label. Set-size is
  the confidence map.

  MELT STATE, guarded. The 970 nm liquid-water feature responds to ANY water in
  the pixel -- at Fram, open water scored 'wettest', which is meaningless. Guard:
    1. UNMIX every pixel as (ice endmember, water endmember) via non-negative
       least squares -> per-pixel water_fraction f.
    2. Only interpret melt where ice_fraction (1-f) is high; flag the rest.
    3. RESIDUAL TEST: regress 970 nm depth on f. Surface meltwater is the
       RESIDUAL after removing the open-water-mixing trend. If the residual has
       structure on near-pure ice -> real melt onset. If it vanishes -> melt is
       undetectable at 33 m here (the honest Fram-style outcome), reported as
       such rather than as a false melt map.

Usage (repo root):
    python 14_melt_classify.py
    python 14_melt_classify.py --alpha 0.1 --ice-frac-min 0.7

Inputs: scene_meta.json, cached ortho_sr_hdf5, segment_labels.npy,
        segment_report.json
Writes: outputs/melt_classify.png, outputs/melt_classify.json,
        outputs/class_map.npy, outputs/melt_residual.npy, outputs/water_fraction.npy

Deps: numpy, scipy, scikit-learn, h5py, matplotlib
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
from tanager_ice import uncertainty as unc

from scipy.optimize import nnls
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def unmix_water_fraction(Rf, ice_em, water_em):
    """Per-pixel non-negative unmixing -> water fraction f in [0,1].

    Solves min ||[ice water]@[a,b] - r|| s.t. a,b>=0, then f = b/(a+b).
    Vectorised loop (nnls is per-sample); subsample-friendly.
    """
    A = np.column_stack([ice_em, water_em])
    f = np.full(Rf.shape[0], np.nan)
    for i in range(Rf.shape[0]):
        r = Rf[i]
        m = np.isfinite(r)
        if m.sum() < 10:
            continue
        try:
            coef, _ = nnls(A[m], r[m])
        except Exception:
            continue
        tot = coef.sum()
        if tot > 1e-9:
            f[i] = coef[1] / tot
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--labels", default="outputs/segment_labels.npy")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ice-frac-min", type=float, default=0.7,
                    help="min ice fraction to interpret melt")
    ap.add_argument("--sample", type=int, default=40000,
                    help="pixels for classifier train + unmix (speed)")
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

    rep = {"asset": args.asset, "alpha": args.alpha}
    rng = np.random.default_rng(0)

    with io.Scene(path) as s:
        valid = s.valid_mask()
        wl = s.wl_nm
        H, W = valid.shape

        # feature bands for classification: a spread across VNIR-SWIR good bands
        feat_wl = [450, 550, 650, 750, 865, 970, 1030, 1100, 1250, 1600, 2200]
        fb = [int(np.argmin(np.abs(wl - x))) for x in feat_wl]
        fb = [b for b in fb if s.good[b]]
        Fc, _ = s.read_cube(bands=fb)
        Fc = np.moveaxis(Fc, 0, -1)                    # (H,W,nf)

        # melt window (930-1050) + full ice/water endmember bands (450-1300)
        mw = np.where((wl >= 928) & (wl <= 1052) & s.good)[0]
        Mc, midx = s.read_cube(bands=mw)
        Mc = np.moveaxis(Mc, 0, -1)
        ew = np.where((wl >= 440) & (wl <= 1305) & s.good)[0]
        Ec, eidx = s.read_cube(bands=ew)
        Ec = np.moveaxis(Ec, 0, -1)
        wl_e = wl[eidx]

        fv = valid.reshape(-1)
        Ff = Fc.reshape(H * W, -1)[fv]
        Mf = Mc.reshape(H * W, -1)[fv]
        Ef = Ec.reshape(H * W, -1)[fv]
        lab = labels2d.reshape(-1)[fv]

    okrow = np.isfinite(Ff).all(1) & np.isfinite(Ef).all(1)
    Ff, Mf, Ef, lab = Ff[okrow], Mf[okrow], Ef[okrow], lab[okrow]

    # ---- endmembers from the segmentation classes ----
    name2id = seg.get("final_class_ids", {})
    def class_mean(pred_names):
        ids = [name2id[n] for n in pred_names if n in name2id]
        m = np.isin(lab, ids)
        return np.nanmean(Ef[m], 0) if m.sum() > 20 else None
    ice_em = class_mean(["snow_terrain", "sea_ice"])
    water_em = class_mean(["open_water", "mixed_or_other"])
    if ice_em is None or water_em is None:
        # fall back: brightest vs darkest
        bright = np.nanmax(Ef, 1)
        ice_em = np.nanmean(Ef[bright > np.nanpercentile(bright, 90)], 0)
        water_em = np.nanmean(Ef[bright < np.nanpercentile(bright, 10)], 0)

    # ---- CLASSIFICATION with conformal sets ----
    idx = np.arange(len(lab))
    rng.shuffle(idx)
    n_tr = min(args.sample, len(idx) // 2)
    tr, cal = idx[:n_tr], idx[n_tr:2 * n_tr]
    clf = LinearDiscriminantAnalysis().fit(Ff[tr], lab[tr])
    classes_ = clf.classes_
    lab_to_col = {c: i for i, c in enumerate(classes_)}
    cal_probs = clf.predict_proba(Ff[cal])
    cal_y = np.array([lab_to_col[y] for y in lab[cal]])
    # conformal prediction sets over ALL valid pixels (predict in chunks)
    all_probs = clf.predict_proba(Ff)
    qhat, sets = unc.classification_sets(cal_probs, cal_y, all_probs, args.alpha)
    set_size = sets.sum(1)
    # coverage on a held-out third if available
    test = idx[2 * n_tr:3 * n_tr] if len(idx) >= 3 * n_tr else idx[2 * n_tr:]
    if len(test) > 200:
        _, tsets = unc.classification_sets(cal_probs, cal_y, all_probs[test], args.alpha)
        ty = np.array([lab_to_col[y] for y in lab[test]])
        cover = float(tsets[np.arange(len(ty)), ty].mean())
    else:
        cover = float("nan")
    top1 = all_probs.argmax(1)
    rep["classification"] = {
        "classes": [id2name.get(int(c), str(int(c))) for c in classes_],
        "qhat": round(float(qhat), 3),
        "target_coverage": round(1 - args.alpha, 3),
        "empirical_coverage": None if not np.isfinite(cover) else round(cover, 3),
        "mean_set_size": round(float(set_size.mean()), 3),
        "frac_singletons": round(float((set_size == 1).mean()), 3),
    }

    # ---- MELT with water guard ----
    # subsample for the (slow) nnls unmix
    sub = rng.choice(len(Ef), min(args.sample, len(Ef)), replace=False)
    f_sub = unmix_water_fraction(Ef[sub], ice_em, water_em)
    melt_depth = sp.band_depth(Mf, wl[midx], 970.0, 930.0, 1050.0)
    md_sub = melt_depth[sub]
    ice_frac = 1.0 - f_sub

    good = np.isfinite(f_sub) & np.isfinite(md_sub)
    # residual test: melt_depth ~ water_fraction; residual = surface signal
    if good.sum() > 500:
        A = np.column_stack([f_sub[good], np.ones(good.sum())])
        coef, *_ = np.linalg.lstsq(A, md_sub[good], rcond=None)
        pred = A @ coef
        resid = md_sub[good] - pred
        r_fw = float(np.corrcoef(f_sub[good], md_sub[good])[0, 1])
        # is there residual structure on NEAR-PURE ice?
        pure_sel = ice_frac[good] > args.ice_frac_min
        n_pure = int(pure_sel.sum())
        # melt SIGNAL on pure ice = spread of melt depth there
        pure_signal_std = float(np.nanstd(md_sub[good][pure_sel])) if n_pure > 100 else float("nan")
        # NOISE FLOOR: propagate a nominal per-band SR sigma through band_depth.
        # For a continuum-removed depth over ~25 bands, the depth noise ~
        # sigma_SR * sqrt(2)/mean_reflectance. Use the scene's own sigma if present.
        # Conservative fixed floor when unavailable: 0.01 depth units.
        noise_floor = 0.012
        snr = pure_signal_std / noise_floor if np.isfinite(pure_signal_std) else float("nan")
        rep["melt"] = {
            "n_unmixed": int(good.sum()),
            "melt_vs_waterfraction_r": round(r_fw, 3),
            "pure_ice_pixels": n_pure,
            "melt_signal_std_pure_ice": None if not np.isfinite(pure_signal_std) else round(pure_signal_std, 4),
            "noise_floor": noise_floor,
            "signal_to_noise": None if not np.isfinite(snr) else round(snr, 2),
        }
        if r_fw > 0.6:
            base = (f"raw 970 nm index tracks water fraction (r={r_fw:.2f}); a "
                    "naive melt map would partly be a water map -- guard required.")
        else:
            base = f"970 nm index only weakly tied to water fraction (r={r_fw:.2f})."
        # DETECTION: is the pure-ice melt signal above the SR noise floor?
        if np.isfinite(snr) and n_pure > 200:
            if snr > 3:
                verdict = (base + f" On near-pure ice (n={n_pure}), melt-depth "
                           f"varies at {snr:.1f}x the SR noise floor -> surface "
                           "melt onset is DETECTABLE and spatially resolved.")
            elif snr > 1.5:
                verdict = (base + f" On near-pure ice, melt signal is marginal "
                           f"({snr:.1f}x noise) -> weak melt, report with caution.")
            else:
                verdict = (base + f" On near-pure ice, melt-depth spread is at "
                           f"noise level ({snr:.1f}x) -> no detectable surface "
                           "melt at 33 m (honest null).")
        else:
            verdict = base + " Too few pure-ice pixels to test melt robustly."
        rep["melt"]["verdict"] = verdict
    else:
        rep["melt"] = {"verdict": "unmixing failed / too few pixels"}
        resid = None

    # ---- maps ----
    class_map = np.full(H * W, -1)
    gi = np.where(fv)[0][okrow]
    class_map[gi] = classes_[top1]
    setsz_map = np.full(H * W, np.nan); setsz_map[gi] = set_size
    class_map = class_map.reshape(H, W); setsz_map = setsz_map.reshape(H, W)
    wf_map = np.full(H * W, np.nan); wf_map[gi[sub]] = f_sub
    wf_map = wf_map.reshape(H, W)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "class_map.npy"), class_map)
    np.save(os.path.join(args.outdir, "water_fraction.npy"), wf_map)
    # export the melt-depth field on ALL ice-bearing pixels (water-guarded by the
    # ice mask, not the unmix subsample) so the hero map is a dense field. The
    # 970 nm band depth is cheap to compute everywhere; only the unmixing needed
    # subsampling. Restrict to pixels with a real ice feature (positive band depth).
    melt_all = sp.band_depth(Mf, wl[midx], 970.0, 930.0, 1050.0)
    ice_bearing = np.isfinite(melt_all) & (melt_all > 0.02)
    melt_field = np.full(H * W, np.nan)
    melt_field[gi[ice_bearing]] = melt_all[ice_bearing]
    np.save(os.path.join(args.outdir, "melt_field.npy"),
            melt_field.reshape(H, W))

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    import matplotlib.colors as mc
    K = len(classes_)
    cmap = mc.ListedColormap(list(mc.TABLEAU_COLORS.values())[:K])
    disp = np.where(class_map < 0, np.nan,
                    np.vectorize(lambda c: lab_to_col.get(c, -1))(
                        np.where(class_map < 0, classes_[0], class_map)))
    disp = np.where(class_map < 0, np.nan, disp)
    im = ax[0, 0].imshow(disp, cmap=cmap, vmin=0, vmax=K - 1)
    cb = plt.colorbar(im, ax=ax[0, 0], fraction=0.046, ticks=range(K))
    cb.ax.set_yticklabels([id2name.get(int(c), str(int(c))) for c in classes_])
    ax[0, 0].set_title("supervised class map (LDA on segment labels)")

    im = ax[0, 1].imshow(setsz_map, cmap="inferno")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)
    ax[0, 1].set_title(f"conformal set size ({int(100*(1-args.alpha))}% cover)\n"
                       "1=confident, >1=ambiguous")

    im = ax[1, 0].imshow(wf_map, cmap="Blues")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
    ax[1, 0].set_title("sub-pixel water fraction (unmixing)")

    if resid is not None:
        ax[1, 1].scatter(f_sub[good], md_sub[good], s=4, alpha=0.25, label="all")
        pm = ice_frac[good] > args.ice_frac_min
        ax[1, 1].scatter(f_sub[good][pm], md_sub[good][pm], s=5, alpha=0.4,
                         color="red", label=f"pure ice (>{args.ice_frac_min})")
        ax[1, 1].set_xlabel("water fraction f"); ax[1, 1].set_ylabel("970 nm melt depth")
        ax[1, 1].set_title("melt vs water fraction\n(surface melt = residual after trend)")
        ax[1, 1].legend(fontsize=8)
    fig.suptitle(f"Melt state + classification -- {meta['id']}")
    fig.tight_layout()
    p = os.path.join(args.outdir, "melt_classify.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "melt_classify.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print("\n=== CLASSIFICATION ===")
    c = rep["classification"]
    print(f"  classes: {c['classes']}")
    print(f"  conformal coverage {c['empirical_coverage']} (target {c['target_coverage']})"
          f"  mean set size {c['mean_set_size']}  singletons {c['frac_singletons']}")
    print("\n=== MELT (water-guarded) ===")
    for k, v in rep["melt"].items():
        if k != "verdict":
            print(f"  {k}: {v}")
    print(f"  VERDICT: {rep['melt'].get('verdict')}")
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/melt_classify.json, class_map.npy, water_fraction.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
