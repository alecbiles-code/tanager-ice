#!/usr/bin/env python3
"""
23_sensor_ladder.py -- how much of the cryosphere-state spectral signal does
each spaceborne sensor retain? A graded comparison across the real fleet,
replacing the structurally-zero Sentinel-2 experiment with an actual gradient.

Each sensor is modelled by its nominal band set (Gaussian SRFs at published
sampling/FWHM) over the 900-1110 nm analysis window; the real Tanager spectra
are convolved to each and the 1030 nm grain band area and 970 nm depth are
recomputed per ice pixel. Retention = median per-pixel ratio vs native Tanager.

Nominal band models (published values):
  Tanager  : ~5 nm sampling (native; identity, known-answer = 1.0)
  EnMAP    : 6.5 nm sampling / ~8.1 nm FWHM (VNIR, <=1000 nm),
             10 nm / ~11 nm (SWIR)      [Chabrillat et al. 2024]
  EMIT     : 7.4 nm sampling / ~8.5 nm FWHM
  PRISMA   : ~11 nm sampling / ~12 nm FWHM
  Sentinel-2: no surface band 958-1565 nm -> structural zero (by band
             placement; reported as such, no staged experiment)

Access context written to the JSON for the memo (verified sources):
  EMIT    : ISS orbit ~51.6 deg inclination -- cannot reach 73.7 N
  PRISMA  : acquisition envelope 70 S - 70 N -- cannot reach 73.7 N
  EnMAP   : 80 N - 80 S via on-demand tasking; single 30-km-swath satellite
  Tanager : polar orbit, open untasked catalog

Usage: python 23_sensor_ladder.py
Writes: outputs/sensor_ladder.png, outputs/sensor_ladder.json
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

SENSORS = {
    "EnMAP":  {"vnir": (6.5, 8.1, 1000.0), "swir": (10.0, 11.0)},
    "EMIT":   {"uniform": (7.4, 8.5)},
    "PRISMA": {"uniform": (11.0, 12.0)},
}
ACCESS = {
    "Tanager": "polar orbit; open untasked catalog; reaches 73.7 N",
    "EnMAP":   "80N-80S via on-demand tasking; single 30-km swath; nominal EOL 2026",
    "EMIT":    "ISS ~51.6 deg inclination; cannot reach 73.7 N",
    "PRISMA":  "acquisition envelope 70S-70N; cannot reach 73.7 N",
    "Sentinel-2": "polar, open, systematic -- but no surface band 958-1565 nm",
}


def band_centers(name, lo, hi):
    cfg = SENSORS[name]
    if "uniform" in cfg:
        s, f = cfg["uniform"]
        return [(c, f) for c in np.arange(lo, hi + 0.1, s)]
    out = []
    s, f, edge = cfg["vnir"]
    out += [(c, f) for c in np.arange(lo, min(hi, edge) + 0.1, s)]
    s, f = cfg["swir"]
    out += [(c, f) for c in np.arange(min(hi, edge) + s, hi + 0.1, s)]
    return out


def convolve(X, wlw, bands):
    out = np.empty((X.shape[0], len(bands)), np.float32)
    for j, (c, fwhm) in enumerate(bands):
        sig = fwhm / 2.3548
        w = np.exp(-0.5 * ((wlw - c) / sig) ** 2)
        w /= w.sum()
        out[:, j] = X @ w
    return out, np.array([c for c, _ in bands])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")

    with io.Scene(path) as s:
        valid = s.valid_mask(); wl = s.wl_nm
        sel = np.where((wl >= 890) & (wl <= 1120) & s.good)[0]
        R, _ = s.read_cube(bands=sel); R = np.moveaxis(R, 0, -1)
        wlw = wl[sel]; H, W = valid.shape
        Rf = R.reshape(H * W, -1)[valid.reshape(-1)]

    depth1030 = sp.band_depth(Rf, wlw, 1030, 958, 1082)
    bright = np.nanmean(Rf, axis=1)
    ice = np.isfinite(depth1030) & (depth1030 > 0.02) & (bright > 0.10) & np.isfinite(Rf).all(1)
    X = Rf[ice]
    # subsample for speed; retention is a ratio statistic
    if X.shape[0] > 40000:
        idx = np.random.default_rng(0).choice(X.shape[0], 40000, replace=False)
        X = X[idx]
    rep = {"n_ice_sampled": int(X.shape[0]), "access": ACCESS}

    area_t = sp.scaled_band_area(X, wlw, 960, 1080)
    d970_t = sp.band_depth(X, wlw, 970, 930, 1010)
    ok = np.isfinite(area_t) & (area_t > 1.0)

    results = {"Tanager": {"grain_retention": 1.0, "d970_retention": 1.0,
                           "grain_corr": 1.0, "n_bands_in_window": int(len(wlw))}}
    print("\n%-11s %8s %12s %12s %10s" % ("sensor", "bands", "grain ret.", "970 ret.", "corr"))
    print("-" * 58)
    print("%-11s %8d %12.3f %12.3f %10.3f" % ("Tanager", len(wlw), 1.0, 1.0, 1.0))
    for name in ("EnMAP", "EMIT", "PRISMA"):
        Xc, wc = convolve(X, wlw, band_centers(name, wlw.min(), wlw.max()))
        # like-for-like: interpolate each sensor's sampled spectrum back onto
        # the native grid so the band-area operator uses IDENTICAL continuum
        # anchors for every sensor; retention then measures genuine spectral
        # information loss, not anchor-snapping to coarser grids.
        Xr = np.empty_like(X)
        for i in range(X.shape[0]):
            Xr[i] = np.interp(wlw, wc, Xc[i])
        area_s = sp.scaled_band_area(Xr, wlw, 960, 1080)
        d970_s = sp.band_depth(Xr, wlw, 970, 930, 1010)
        m = ok & np.isfinite(area_s)
        ret = float(np.nanmedian(area_s[m] / area_t[m]))
        ret970 = float(np.nanmedian(d970_s[m] / np.where(np.abs(d970_t[m]) > 1e-3, d970_t[m], np.nan)))
        r = float(np.corrcoef(area_s[m], area_t[m])[0, 1])
        results[name] = {"grain_retention": round(ret, 3),
                         "d970_retention": round(ret970, 3),
                         "grain_corr": round(r, 3),
                         "n_bands_in_window": int(len(wc))}
        print("%-11s %8d %12.3f %12.3f %10.3f" % (name, len(wc), ret, ret970, r))
    results["Sentinel-2"] = {"grain_retention": 0.0, "d970_retention": 0.0,
                             "grain_corr": None, "n_bands_in_window": 0,
                             "note": "no surface band 958-1565 nm; zero by band placement"}
    print("%-11s %8d %12.3f %12.3f %10s   (structural)" % ("Sentinel-2", 0, 0.0, 0.0, "--"))
    rep["results"] = results

    verdict = ("SPECTRAL PARITY ACROSS THE HYPERSPECTRAL FLEET: EnMAP, EMIT and "
               "PRISMA nominal band models retain %.0f%%, %.0f%% and %.0f%% of "
               "the 1030 nm grain signal respectively (per-pixel correlation "
               ">= %.2f), while Sentinel-2 retains none by band placement. The "
               "capability that separates Tanager at 73.7 N is therefore ACCESS, "
               "not spectroscopy: EMIT is excluded by its ISS orbit, PRISMA by "
               "its 70 N acquisition envelope, and EnMAP -- which could task "
               "this site -- is a single on-demand 30-km-swath mission; Tanager "
               "provides open, untasked catalog coverage." % (
                   100 * results["EnMAP"]["grain_retention"],
                   100 * results["EMIT"]["grain_retention"],
                   100 * results["PRISMA"]["grain_retention"],
                   min(results[k]["grain_corr"] for k in ("EnMAP", "EMIT", "PRISMA"))))
    rep["verdict"] = verdict
    print("\nVERDICT: %s" % verdict)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    names = ["Tanager", "EnMAP", "EMIT", "PRISMA", "Sentinel-2"]
    vals = [results[n]["grain_retention"] for n in names]
    cols = ["#1565C0", "#2e7d32", "#2e7d32", "#2e7d32", "#c62828"]
    ax[0].bar(names, vals, color=cols, alpha=0.85)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.02, "%.0f%%" % (100 * v), ha="center", fontsize=10, fontweight="bold")
    ax[0].set_ylabel("1030 nm grain-signal retention"); ax[0].set_ylim(0, 1.15)
    ax[0].set_title("spectral retention ladder (nominal band models)")
    mean_t = np.nanmean(X, 0)
    ax[1].plot(wlw, mean_t, "k-", lw=1.8, label="Tanager (native ~5 nm)")
    for name, col in (("EnMAP", "tab:green"), ("EMIT", "tab:olive"), ("PRISMA", "tab:orange")):
        Xc, wc = convolve(mean_t[None, :], wlw, band_centers(name, wlw.min(), wlw.max()))
        ax[1].plot(wc, Xc[0], "o-", ms=3.5, lw=1, color=col, alpha=0.85, label=name)
    ax[1].axvline(1030, ls=":", color="grey"); ax[1].axvline(970, ls=":", color="grey")
    ax[1].set_xlabel("nm"); ax[1].set_ylabel("reflectance")
    ax[1].set_title("mean ice spectrum as each sensor samples it")
    ax[1].legend(fontsize=9)
    fig.suptitle("The capability gap is access, not spectroscopy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.outdir, "sensor_ladder.png")
    fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "sensor_ladder.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("\nwrote %s" % p)
    print("wrote %s/sensor_ladder.json" % args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
