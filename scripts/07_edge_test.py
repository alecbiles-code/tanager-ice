#!/usr/bin/env python3
"""
07_edge_test.py -- is the residual CWV/melt coupling an EDGE artifact?

Context. 06_stratified_ac.py found, within the ice stratum:
    CWV vs 970 nm melt depth : r = -0.366   ("moderate")
The sign is NEGATIVE, which already refutes the aliasing mechanism originally
feared (surface absorption misread as vapour would give a POSITIVE r). And the
CWV map shows elevated values in thin rings hugging every floe/lead boundary --
the signature of mixed pixels and adjacency effects, not of an atmospheric field.

Hypothesis: the coupling lives entirely in ice pixels NEAR WATER. Pure floe
interiors should show no coupling at all.

Test: erode the ice mask away from the water boundary in steps, and recompute
the correlation at each erosion distance. If |r| decays toward 0 with distance,
the coupling is an edge artifact and Planet's CWV over pure ice is clean.
That is a decisive, falsifiable answer rather than an argument.

Consequences either way:
  * r -> 0 with erosion  => CWV over pure ice is fine; the finding becomes
    "Planet's AC aux layers degrade within ~N pixels of a floe edge and over
    open water" (useful to Planet, and a labelling rule for Task 2: keep
    labels >= N px from any edge).
  * r stays flat          => something real couples CWV and the melt feature
    over ice; investigate before trusting the melt retrieval.

Usage:
    python 07_edge_test.py
    python 07_edge_test.py --max-erode 12

Writes: outputs/edge_test.png, outputs/edge_test.json
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
    from scipy.ndimage import binary_erosion, distance_transform_edt
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def _erode_once(mask):
    """4-connected erosion without scipy: a pixel survives if all neighbours are set."""
    m = mask
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def erode(mask, n):
    if n <= 0:
        return mask
    if HAVE_SCIPY:
        return binary_erosion(mask, iterations=n, border_value=0)
    m = mask.copy()
    for _ in range(n):
        m = _erode_once(m)
    return m


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--ice-thresh", type=float, default=None)
    ap.add_argument("--max-erode", type=int, default=10)
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")
    if not HAVE_SCIPY:
        print("[note] scipy not found; using a slower pure-numpy erosion "
              "(conda install -c conda-forge scipy to speed up)")

    rep = {"asset": args.asset}
    with io.Scene(path) as s:
        valid = s.valid_mask()
        b650 = int(np.argmin(np.abs(s.wl_nm - 650)))
        bright, _ = s.read_cube(bands=[b650])
        bright = np.where(valid, bright[0], np.nan)

        thr = args.ice_thresh
        if thr is None:
            # same Otsu split as 06 so the strata are comparable
            v = bright[np.isfinite(bright)]
            lo, hi = np.percentile(v, [0.5, 99.5])
            hist, edges = np.histogram(v, bins=256, range=(lo, hi))
            p = hist / hist.sum()
            omega = np.cumsum(p)
            mids = (edges[:-1] + edges[1:]) / 2
            mu = np.cumsum(p * mids)
            den = omega * (1 - omega); den[den == 0] = np.nan
            thr = float(mids[int(np.nanargmax((mu[-1] * omega - mu) ** 2 / den))])
        ice = valid & (bright > thr)
        rep["ice_threshold"] = round(float(thr), 4)
        print(f"[strata] ice threshold {thr:.3f}, ice = "
              f"{100*ice.sum()/valid.sum():.1f}% of valid")

        cwv = np.where(valid, s.plane("column_water_vapour"), np.nan)
        lo_w, hi_w = 930.0, 1050.0
        bmask = (s.wl_nm >= lo_w - 1) & (s.wl_nm <= hi_w + 1) & s.good
        win, widx = s.read_cube(bands=np.where(bmask)[0])
        depth = sp.band_depth(np.moveaxis(win, 0, -1), s.wl_nm[widx], 970.0, lo_w, hi_w)
        depth = np.where(valid, depth, np.nan)
        del win

        gsd = 35.77
        print(f"\n[erosion] correlation of CWV vs 970nm depth, ice pixels only,")
        print(f"          as we retreat from the ice/water boundary "
              f"({gsd:.0f} m per step):\n")
        print(f"{'erode px':>9s} {'dist m':>8s} {'n pixels':>10s} {'r':>8s}")
        curve = []
        for k in range(0, args.max_erode + 1):
            m = erode(ice, k)
            if m.sum() < 500:
                print(f"{k:9d} {k*gsd:8.0f} {int(m.sum()):10d}   (too few, stop)")
                break
            r, n = corr(cwv[m].ravel(), depth[m].ravel())
            curve.append({"erode_px": k, "dist_m": round(k * gsd, 1),
                          "n": n, "r": None if not np.isfinite(r) else round(r, 4)})
            print(f"{k:9d} {k*gsd:8.0f} {n:10d} {r:+8.3f}")
        rep["erosion_curve"] = curve

        rs = [c["r"] for c in curve if c["r"] is not None]
        if len(rs) >= 3:
            r0, rend = abs(rs[0]), abs(rs[-1])
            decay = (r0 - rend) / max(r0, 1e-9)
            rep["r_at_0px"] = round(rs[0], 4)
            rep["r_at_max_erode"] = round(rs[-1], 4)
            rep["fractional_decay"] = round(float(decay), 4)
            print()
            if rend < 0.15 or decay > 0.6:
                v = ("EDGE ARTIFACT CONFIRMED: the coupling decays away from the "
                     "boundary. Planet's CWV over pure floe interiors is clean. "
                     f"Rule for Task 2: keep labels >= {curve[-1]['erode_px']} px "
                     f"(~{curve[-1]['dist_m']:.0f} m) from any ice/water edge.")
            elif decay > 0.3:
                v = ("PARTIAL edge effect: coupling weakens but does not vanish. "
                     "Erode labels AND keep the CWV caveat.")
            else:
                v = ("NOT an edge effect: coupling persists into floe interiors. "
                     "Investigate before trusting the melt retrieval.")
            rep["verdict"] = v
            print(f"[verdict] {v}")

        # ---------------- figures ----------------
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        ax = axes[0, 0]
        xs = [c["dist_m"] for c in curve]
        ys = [abs(c["r"]) if c["r"] is not None else np.nan for c in curve]
        ax.plot(xs, ys, "o-", color="crimson")
        ax.axhline(0.2, ls="--", color="grey", label="weak-coupling threshold")
        ax.set_xlabel("distance eroded from ice/water boundary (m)")
        ax.set_ylabel("|r|  (CWV vs 970 nm depth)")
        ax.set_title("Does the coupling survive away from floe edges?")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ns = [c["n"] for c in curve]
        ax.plot(xs, ns, "s-", color="steelblue")
        ax.set_xlabel("distance eroded (m)"); ax.set_ylabel("ice pixels remaining")
        ax.set_title("sample size vs erosion")
        ax.grid(alpha=0.3)

        # distance-to-water map
        if HAVE_SCIPY:
            dist = distance_transform_edt(ice) * gsd
            dist = np.where(ice, dist, np.nan)
            im = axes[1, 0].imshow(dist, cmap="cividis")
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, label="m from water")
            axes[1, 0].set_title("distance of each ice pixel from open water")
        else:
            axes[1, 0].axis("off")

        core = erode(ice, min(args.max_erode, len(curve) - 1))
        im = axes[1, 1].imshow(np.where(core, cwv, np.nan), cmap="viridis")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].set_title(f"Planet CWV, PURE floe interiors only\n"
                             f"(eroded {len(curve)-1} px)")

        fig.suptitle("Edge test: is the CWV/melt coupling a mixed-pixel artifact?")
        fig.tight_layout()
        os.makedirs(args.outdir, exist_ok=True)
        p = os.path.join(args.outdir, "edge_test.png")
        fig.savefig(p, dpi=125); plt.close(fig)

    with open(os.path.join(args.outdir, "edge_test.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {p}")
    print(f"wrote {args.outdir}/edge_test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
