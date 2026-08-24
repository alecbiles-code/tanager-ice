#!/usr/bin/env python3
"""
16_albedo_stratified_grain.py -- is the grain-size proxy real, or is it
brightness in disguise? Resolves the reviewer's Tier-1 objection.

THE OBJECTION. The grain proxy correlates POSITIVELY with 1100 nm reflectance
(r=+0.38). Nolin-Dozier physics predicts the OPPOSITE sign (coarser grains
absorb more -> lower NIR reflectance). A positive sign is the signature of
BRIGHTNESS driving the proxy, not grain size. If true, the whole grain result
is a brightness map wearing a grain label.

THE TEST (and its trap). The obvious fix is to regress proxy vs NIR within
albedo bins and, if the correlation vanishes, declare the confound removed. But
that is ambiguous: a near-zero within-bin correlation is ALSO what you get if
the proxy has NO grain signal once brightness is removed. Same number, opposite
conclusions.

THE DISCRIMINATOR. Within a FIXED albedo bin, do the surface CLASSES still
separate in the proxy? Two ice types at the SAME brightness but DIFFERENT grain
size must differ in the proxy IF the proxy tracks grain. So we report, per
albedo bin:
  (a) within-bin proxy-vs-NIR correlation  (is brightness still driving it?)
  (b) within-bin class separation           (does a material signal survive
                                              at fixed brightness?)
Verdicts:
  - class separation SURVIVES at fixed albedo -> genuine material signal;
    the global +0.38 was brightness co-variation across classes, now removed.
  - class separation COLLAPSES at fixed albedo -> the proxy WAS brightness;
    grain size is not independently retrievable here. Report honestly and
    demote grain in the submission.

Also emits the residual-grain map (proxy minus its brightness trend) that the
reviewer asked for.

Usage: python 16_albedo_stratified_grain.py
Writes: outputs/albedo_stratified.png, outputs/albedo_stratified.json,
        outputs/grain_residual.npy
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
    ap.add_argument("--nbins", type=int, default=6)
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

    with io.Scene(path) as s:
        valid = s.valid_mask(); wl = s.wl_nm
        sel = np.where((wl >= 895) & (wl <= 1305) & s.good)[0]
        R, _ = s.read_cube(bands=sel); R = np.moveaxis(R, 0, -1)
        wlw = wl[sel]
        # broadband albedo proxy: mean visible reflectance (450-650), INDEPENDENT
        # of the 1030 feature the grain proxy uses
        vsel = np.where((wl >= 450) & (wl <= 680) & s.good)[0]
        V, _ = s.read_cube(bands=vsel); V = np.moveaxis(V, 0, -1)
        H, W = valid.shape
        fv = valid.reshape(-1)
        Rf = R.reshape(H * W, -1)[fv]
        albedo = np.nanmean(V.reshape(H * W, -1)[fv], axis=1)
        lab = labels2d.reshape(-1)[fv]
        i1100 = int(np.argmin(np.abs(wlw - 1100)))
        nir = Rf[:, i1100]

    grain = sp.scaled_band_area(Rf, wlw, 960, 1080)
    depth = sp.band_depth(Rf, wlw, 1030, 960, 1080)
    ice = np.isfinite(grain) & (depth > 0.02) & (nir > 0.15) & np.isfinite(albedo)
    grain, albedo, nir, lab = grain[ice], albedo[ice], nir[ice], lab[ice]
    rep = {"n_ice": int(ice.sum())}

    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    rep["global_proxy_vs_nir_r"] = round(corr(grain, nir), 3)
    rep["global_proxy_vs_albedo_r"] = round(corr(grain, albedo), 3)

    # the two ice classes the submission is about
    n2i = seg.get("final_class_ids", {})
    ice_classes = [n2i[n] for n in ("sea_ice", "snow_terrain") if n in n2i]

    # ---- stratify by albedo bin ----
    edges = np.nanpercentile(albedo, np.linspace(0, 100, args.nbins + 1))
    bins = []
    print(f"\n{'albedo bin':>18s} {'n':>7s} {'proxy~NIR r':>11s} "
          f"{'seaice med':>11s} {'snow med':>10s} {'class sep':>10s}")
    print("-" * 72)
    for b in range(args.nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (albedo >= lo) & (albedo < hi if b < args.nbins - 1 else albedo <= hi)
        if m.sum() < 200:
            continue
        r_in = corr(grain[m], nir[m])
        meds = {}
        for c in ice_classes:
            cm = m & (lab == c)
            if cm.sum() > 30:
                meds[id2name.get(int(c), str(c))] = float(np.nanmedian(grain[cm]))
        sep = (abs(meds.get("sea_ice", np.nan) - meds.get("snow_terrain", np.nan))
               if len(meds) == 2 else float("nan"))
        # normalise separation by within-class spread (pooled IQR) for context
        spreads = []
        for c in ice_classes:
            cm = m & (lab == c)
            if cm.sum() > 30:
                spreads.append(np.subtract(*np.nanpercentile(grain[cm], [75, 25])))
        pooled = np.nanmean(spreads) if spreads else np.nan
        sep_norm = sep / pooled if (np.isfinite(sep) and pooled and pooled > 0) else np.nan
        bins.append({"albedo_lo": round(float(lo), 3), "albedo_hi": round(float(hi), 3),
                     "n": int(m.sum()), "proxy_vs_nir_r": None if not np.isfinite(r_in) else round(r_in, 3),
                     "class_medians": {k: round(v, 3) for k, v in meds.items()},
                     "class_separation": None if not np.isfinite(sep) else round(float(sep), 3),
                     "separation_over_iqr": None if not np.isfinite(sep_norm) else round(float(sep_norm), 3)})
        print(f"  [{lo:.2f},{hi:.2f}] {m.sum():7d} {r_in:11.3f} "
              f"{meds.get('sea_ice', float('nan')):11.2f} "
              f"{meds.get('snow_terrain', float('nan')):10.2f} "
              f"{sep:10.2f}")
    rep["albedo_bins"] = bins

    # ---- verdict ----
    within_r = [b["proxy_vs_nir_r"] for b in bins if b["proxy_vs_nir_r"] is not None]
    seps = [b["class_separation"] for b in bins if b["class_separation"] is not None]
    sep_norms = [b["separation_over_iqr"] for b in bins if b["separation_over_iqr"] is not None]
    mean_within_r = float(np.nanmean(within_r)) if within_r else float("nan")
    mean_sep_norm = float(np.nanmean(sep_norms)) if sep_norms else float("nan")
    rep["mean_within_bin_proxy_nir_r"] = round(mean_within_r, 3)
    rep["mean_within_bin_class_sep_over_iqr"] = round(mean_sep_norm, 3)

    brightness_gone = abs(mean_within_r) < 0.2
    class_survives = np.isfinite(mean_sep_norm) and mean_sep_norm > 0.5
    if class_survives and brightness_gone:
        verdict = ("GRAIN SIGNAL IS REAL: at fixed albedo the proxy-vs-brightness "
                   "correlation collapses (mean r={:.2f}) yet the sea-ice/snow "
                   "class separation SURVIVES ({:.1f}x within-class IQR). The global "
                   "+0.38 was brightness co-variation ACROSS classes; the material "
                   "signal is independent of it.").format(mean_within_r, mean_sep_norm)
    elif class_survives and not brightness_gone:
        verdict = ("MIXED: class separation survives at fixed albedo ({:.1f}x IQR) "
                   "but a residual brightness coupling remains (mean r={:.2f}). Real "
                   "signal present; report the residual and keep the brightness "
                   "caveat.").format(mean_sep_norm, mean_within_r)
    else:
        verdict = ("PROXY IS BRIGHTNESS-DOMINATED: at fixed albedo the classes no "
                   "longer separate (sep {:.1f}x IQR). The proxy does not carry an "
                   "independent grain signal on this scene; grain size should be "
                   "DEMOTED and the melt/AOD/gap results should lead.").format(mean_sep_norm)
    rep["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    # ---- residual-grain map (proxy minus brightness trend) ----
    m = np.isfinite(grain) & np.isfinite(albedo)
    A = np.column_stack([albedo[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, grain[m], rcond=None)
    resid = np.full(grain.shape, np.nan); resid[m] = grain[m] - A @ coef
    rep["brightness_trend_slope"] = round(float(coef[0]), 3)
    rep["residual_std_over_raw_std"] = round(float(np.nanstd(resid) / (np.nanstd(grain) + 1e-9)), 3)

    # residual back to a map
    full = np.full(H * W, np.nan)
    gi = np.where(fv)[0][ice]
    full[gi[m]] = resid[m]
    np.save(os.path.join(args.outdir, "grain_residual.npy"), full.reshape(H, W))

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    ax[0, 0].scatter(nir, grain, s=3, alpha=0.2, c=albedo, cmap="cividis")
    ax[0, 0].set_xlabel("1100 nm reflectance"); ax[0, 0].set_ylabel("grain proxy")
    ax[0, 0].set_title(f"global proxy vs NIR  r={rep['global_proxy_vs_nir_r']}\n"
                       "(colour = albedo; the confound)")
    xs = [f"[{b['albedo_lo']:.2f},{b['albedo_hi']:.2f}]" for b in bins]
    ax[0, 1].bar(range(len(bins)), [b["proxy_vs_nir_r"] or 0 for b in bins], color="tab:red", alpha=0.7)
    ax[0, 1].axhline(0, color="k", lw=0.8); ax[0, 1].axhline(0.38, color="grey", ls="--", label="global +0.38")
    ax[0, 1].set_xticks(range(len(bins))); ax[0, 1].set_xticklabels(xs, rotation=40, fontsize=7)
    ax[0, 1].set_ylabel("proxy~NIR r"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("within-albedo-bin brightness coupling\n(should collapse toward 0)")
    ax[1, 0].bar(range(len(bins)), [b["separation_over_iqr"] or 0 for b in bins], color="tab:blue", alpha=0.7)
    ax[1, 0].axhline(0.5, color="grey", ls="--", label="0.5x IQR threshold")
    ax[1, 0].set_xticks(range(len(bins))); ax[1, 0].set_xticklabels(xs, rotation=40, fontsize=7)
    ax[1, 0].set_ylabel("sea_ice/snow sep / IQR"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("within-bin class separation\n(survives => real material signal)")
    rmap = np.load(os.path.join(args.outdir, "grain_residual.npy"))
    im = ax[1, 1].imshow(rmap, cmap="RdBu_r",
                         vmin=np.nanpercentile(rmap, 5), vmax=np.nanpercentile(rmap, 95))
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    ax[1, 1].set_title("grain residual after removing brightness trend"); ax[1, 1].axis("off")
    fig.suptitle("Is the grain proxy real, or brightness? Albedo-stratified test",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "albedo_stratified.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "albedo_stratified.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/albedo_stratified.json, grain_residual.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
