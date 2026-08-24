#!/usr/bin/env python3
"""
21_negative_area_autopsy.py -- characterize the unphysical negative-band-area
pixels the reviewer flagged (a low-NIR cluster around proxy ~ -2), determine
their cause, and define a defensible mask rule.

A scaled band AREA can only go negative if the continuum-removed spectrum
rises ABOVE its continuum inside the window -- i.e., the 960/1080 shoulder
endpoints sit BELOW the interior. Physically plausible causes, distinguishable
by their spectra and context:
  (a) water-adjacent / thin-ice pixels: NIR reflectance is collapsing toward
      zero across the window, so the 1080 endpoint is darker than the
      interior -- the "feature" is inverted by the falling continuum;
  (b) shadowed / low-signal pixels where noise dominates a near-zero signal;
  (c) genuine retrieval breakdown on valid bright ice (would be concerning).

The script maps the negative cluster, pulls its mean spectrum against a
normal-ice reference, reports its brightness/water-fraction/class context, and
emits the mask rule with the fraction of scene it affects.

Usage: python 21_negative_area_autopsy.py
Writes: outputs/negative_autopsy.png, outputs/negative_autopsy.json
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
    ap.add_argument("--grain", default="outputs/grainsize.npy")
    ap.add_argument("--threshold", type=float, default=-0.5,
                    help="band-area below this = negative cluster")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")
    if not os.path.exists(args.grain):
        sys.exit("run 12_grainsize.py first (need grainsize.npy)")
    grain = np.load(args.grain)

    with io.Scene(path) as s:
        valid = s.valid_mask(); wl = s.wl_nm
        sel = np.where((wl >= 895) & (wl <= 1305) & s.good)[0]
        R, _ = s.read_cube(bands=sel); R = np.moveaxis(R, 0, -1)
        wlw = wl[sel]; H, W = valid.shape

    neg = valid & np.isfinite(grain) & (grain < args.threshold)
    pos = valid & np.isfinite(grain) & (grain > 1.0)
    rep = {"threshold": args.threshold,
           "n_negative": int(neg.sum()),
           "n_valid_retrieved": int((valid & np.isfinite(grain)).sum()),
           "fraction_of_retrieved": round(float(neg.sum() / max((valid & np.isfinite(grain)).sum(), 1)), 4)}
    print("\n[autopsy] %d negative-area pixels (%.1f%% of retrieved)" % (
        rep["n_negative"], 100 * rep["fraction_of_retrieved"]))
    if neg.sum() < 20:
        rep["verdict"] = "Negative cluster negligible (<20 px); no action needed."
        json.dump(rep, open(os.path.join(args.outdir, "negative_autopsy.json"), "w"), indent=2)
        print(rep["verdict"]); return 0

    spec_neg = np.nanmean(R[neg], axis=0)
    spec_pos = np.nanmean(R[pos], axis=0)
    b1100 = R[:, :, int(np.argmin(np.abs(wlw - 1100)))]
    i960, i1080 = int(np.argmin(np.abs(wlw - 960))), int(np.argmin(np.abs(wlw - 1080)))
    interior = R[:, :, (wlw > 990) & (wlw < 1060)].mean(axis=2)
    shoulder_lo = R[:, :, i960]; shoulder_hi = R[:, :, i1080]
    inverted = shoulder_hi < interior  # falling continuum signature

    rep["negative_cluster"] = {
        "mean_1100nm_reflectance": round(float(np.nanmean(b1100[neg])), 4),
        "normal_ice_1100nm": round(float(np.nanmean(b1100[pos])), 4),
        "frac_with_inverted_continuum": round(float(np.nanmean(inverted[neg])), 3),
        "frac_inverted_in_normal_ice": round(float(np.nanmean(inverted[pos])), 3)}
    nc = rep["negative_cluster"]
    print("  mean 1100nm reflectance   : %.3f (normal ice %.3f)" % (
        nc["mean_1100nm_reflectance"], nc["normal_ice_1100nm"]))
    print("  inverted-continuum frac   : %.0f%% (normal ice %.0f%%)" % (
        100 * nc["frac_with_inverted_continuum"], 100 * nc["frac_inverted_in_normal_ice"]))

    dark = nc["mean_1100nm_reflectance"] < 0.5 * nc["normal_ice_1100nm"]
    inv = nc["frac_with_inverted_continuum"] > 0.6
    if dark and inv:
        cause = ("water-adjacent / low-NIR pixels: the continuum falls across the "
                 "window (1080 shoulder darker than interior), inverting the "
                 "band-area sign. A retrieval artifact of the falling continuum, "
                 "not a grain measurement.")
        mask_rule = ("mask pixels with 1100 nm reflectance < %.2f OR shoulder_hi < "
                     "interior mean" % (0.5 * nc["normal_ice_1100nm"]))
    elif inv:
        cause = ("inverted-continuum pixels at normal brightness -- spectral shape "
                 "anomaly (mixed surface or shadowing); band area undefined there.")
        mask_rule = "mask pixels where the 1080 nm shoulder is darker than the window interior"
    else:
        cause = ("negative areas occur on bright, normal-continuum ice -- genuine "
                 "retrieval breakdown; investigate before trusting the grain map.")
        mask_rule = "no simple rule; flag for investigation"
    rep["cause"] = cause; rep["mask_rule"] = mask_rule
    rep["verdict"] = ("Negative-area cluster (%.1f%% of retrieved) diagnosed: %s "
                      "Mask rule: %s." % (100 * rep["fraction_of_retrieved"], cause, mask_rule))
    print("\nVERDICT: %s" % rep["verdict"])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    show = np.where(valid & np.isfinite(grain), grain, np.nan)
    im = ax[0].imshow(show, cmap="viridis", vmin=-3, vmax=12)
    ax[0].contour(neg.astype(float), levels=[0.5], colors="red", linewidths=0.8)
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    ax[0].set_title("grain proxy; negative cluster outlined red"); ax[0].axis("off")
    ax[1].plot(wlw, spec_pos, "k-", lw=1.6, label="normal ice (proxy > 1)")
    ax[1].plot(wlw, spec_neg, "r-", lw=1.6, label="negative cluster")
    ax[1].axvline(960, ls=":", color="grey"); ax[1].axvline(1080, ls=":", color="grey")
    ax[1].set_xlabel("nm"); ax[1].set_ylabel("reflectance")
    ax[1].set_title("mean spectra: the falling continuum"); ax[1].legend(fontsize=8)
    ax[2].hist(grain[valid & np.isfinite(grain)].ravel(), bins=100, color="tab:blue", alpha=0.8)
    ax[2].axvline(args.threshold, color="red", ls="--", label="threshold")
    ax[2].set_yscale("log"); ax[2].set_xlabel("band area"); ax[2].legend(fontsize=8)
    ax[2].set_title("band-area distribution (log count)")
    fig.suptitle("Negative band-area autopsy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "negative_autopsy.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "negative_autopsy.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\nwrote %s" % p)
    print("wrote %s/negative_autopsy.json" % args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
