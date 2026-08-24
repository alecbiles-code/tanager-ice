#!/usr/bin/env python3
"""
09_survey_catalog.py -- walk the whole Tanager Open-Data STAC and rank scenes
for our cryosphere hunt, so we choose from a sorted shortlist instead of
clicking through 150+ scenes.

Why this is the fast path: the catalog is a static STAC tree (no auth). We read
only the item JSONs -- kilobytes each, NOT the gigabyte HDF5 cubes -- so the
whole survey is a few minutes of metadata fetches. Downloading a cube is the
LAST step, for the 2-3 finalists only.

Scoring is transparent and tunable (weights below). Each scene gets:
    * geometry-derived: centroid lat/lon, EMIT +/-52 gap flag
    * time-derived: month, hemisphere-aware melt-season score
    * sun elevation (retrieval-quality proxy)
    * collection/theme (snow-ice etc.)
    * an ASSET check: does it have ortho surface reflectance? (needed for our pipeline)
    * a TARGET score aimed at the rubric's 30-pt Environmental Application box:
      melt-season high-latitude scenes, glacier-margin candidates, etc.

What it CANNOT know from metadata alone: whether a given snow-ice scene actually
contains glacier ice / algae / debris vs plain sea ice. That needs the thumbnail
or a peek at the cube. So the script also downloads each finalist's small
thumbnail PNG (kilobytes) and assembles a contact sheet for eyeball triage.

Usage (repo root, needs internet; no Planet auth required):
    python 09_survey_catalog.py
    python 09_survey_catalog.py --themes snow-ice            # restrict themes
    python 09_survey_catalog.py --top 15 --thumbnails        # + contact sheet
    python 09_survey_catalog.py --lat-min 55                 # only high latitude

Writes: outputs/catalog_survey.csv     every scene, all fields, sortable
        outputs/catalog_survey.json    same, structured
        outputs/survey_shortlist.txt    top-N human-readable
        outputs/thumbs/                 finalist thumbnails (with --thumbnails)
        outputs/contact_sheet.png       (with --thumbnails)

Deps: pystac-client OR pystac, requests, shapely (optional), matplotlib+PIL
      (only for --thumbnails)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys

import requests

CATALOG = "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"
EMIT_LAT = 52.0

# cryosphere target keywords we hope to find in id / collection / description
GLACIER_HINTS = ("glacier", "ice-sheet", "icesheet", "greenland", "iceland",
                 "alaska", "himalaya", "andes", "alps", "patagonia", "svalbard",
                 "norway", "baffin", "ellesmere")
# scoring weights (tune freely)
W = {"emit_gap": 25, "season": 30, "sun": 15, "sr_asset": 15,
     "glacier_hint": 15}


def get_items(catalog_url):
    """Yield STAC items from the static catalog. Prefer pystac; fall back to manual walk."""
    try:
        import pystac
        root = pystac.Catalog.from_file(catalog_url)
        for it in root.get_items(recursive=True):
            yield it.to_dict(), (it.get_self_href() or "")
        return
    except Exception as e:
        print(f"[survey] pystac unavailable or failed ({e}); using manual walk")

    # manual recursive walk of a static catalog
    session = requests.Session()

    def base(u):
        return u.rsplit("/", 1)[0]

    def resolve(href, parent_url):
        if href.startswith("http"):
            return href
        # relative link
        return os.path.normpath(os.path.join(base(parent_url), href)).replace(
            "https:/", "https://")

    seen = set()
    stack = [catalog_url]
    while stack:
        url = stack.pop()
        if url in seen:
            continue
        seen.add(url)
        try:
            doc = session.get(url, timeout=60).json()
        except Exception as ex:
            print(f"  [warn] failed {url}: {ex}")
            continue
        typ = doc.get("type")
        if typ == "Feature":                       # a STAC Item
            yield doc, url
            continue
        for lk in doc.get("links", []):
            rel = lk.get("rel")
            if rel in ("child", "item"):
                stack.append(resolve(lk["href"], url))


def centroid(geom):
    """Rough centroid of a polygon's exterior ring."""
    if not geom or geom.get("type") != "Polygon":
        b = geom
        return None, None
    ring = geom["coordinates"][0]
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    n = len(pts)
    a = cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        a += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr
    if abs(a) < 1e-12:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return sum(xs) / n, sum(ys) / n
    a *= 0.5
    return cx / (6 * a), cy / (6 * a)


def melt_season_score(month, lat):
    """1.0 at peak melt onset for the hemisphere, decaying away from it.

    N. hemisphere cryosphere melt onset ~ May-Aug (peak Jun-Jul).
    S. hemisphere ~ Nov-Feb. Pre-onset cold snow scores low (our Fram problem:
    mid-May at 76N was too early).
    """
    if lat is None:
        return 0.0
    peak = 6.5 if lat >= 0 else 12.5              # month of peak melt
    # circular distance in months
    d = abs(month - peak)
    d = min(d, 12 - d)
    return max(0.0, 1.0 - d / 4.0)                # zero >4 months from peak


def score_scene(rec):
    s = 0.0
    parts = {}
    lat = rec["lat"]
    if lat is not None and abs(lat) > EMIT_LAT:
        parts["emit_gap"] = W["emit_gap"]
    else:
        parts["emit_gap"] = 0
    parts["season"] = W["season"] * melt_season_score(rec["month"], lat) \
        if rec["month"] else 0
    # sun elevation: reward 20-45 deg (usable, not washed out, not too low)
    se = rec.get("sun_elev")
    if se is not None:
        parts["sun"] = W["sun"] * max(0.0, 1.0 - abs(se - 33) / 33)
    else:
        parts["sun"] = 0
    parts["sr_asset"] = W["sr_asset"] if rec.get("has_ortho_sr") else 0
    hint = any(h in (rec["id"] + " " + rec["collection"] + " " +
                     rec.get("desc", "")).lower() for h in GLACIER_HINTS)
    parts["glacier_hint"] = W["glacier_hint"] if hint else 0
    rec["score_parts"] = parts
    return round(sum(parts.values()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--themes", nargs="*", default=None,
                    help="restrict to collections whose id contains any of these")
    ap.add_argument("--lat-min", type=float, default=None,
                    help="drop scenes with |lat| below this")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--thumbnails", action="store_true",
                    help="download finalist thumbnails + build a contact sheet")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[survey] walking {args.catalog}")
    recs = []
    n = 0
    for item, self_href in get_items(args.catalog):
        n += 1
        props = item.get("properties", {})
        geom = item.get("geometry")
        lon, lat = centroid(geom) if geom else (None, None)
        coll = item.get("collection", "") or ""
        if args.themes and not any(t.lower() in coll.lower() for t in args.themes):
            continue
        if args.lat_min and (lat is None or abs(lat) < args.lat_min):
            continue
        dtstr = props.get("datetime", "") or ""
        month = None
        if dtstr:
            try:
                month = dt.datetime.fromisoformat(dtstr.replace("Z", "+00:00")).month
            except Exception:
                pass
        assets = item.get("assets", {})
        se = props.get("view:sun_elevation")
        if se is None and props.get("view:sun_zenith") is not None:
            se = 90 - props["view:sun_zenith"]
        rec = {
            "id": item.get("id", "?"),
            "collection": coll,
            "datetime": dtstr,
            "month": month,
            "lat": None if lat is None else round(lat, 3),
            "lon": None if lon is None else round(lon, 3),
            "sun_elev": None if se is None else round(float(se), 1),
            "has_ortho_sr": "ortho_sr_hdf5" in assets,
            "has_ortho_rad": "ortho_radiance_hdf5" in assets,
            "thumb": assets.get("thumbnail", {}).get("href"),
            "self_href": self_href,
            "desc": (props.get("description", "") or "")[:200],
        }
        rec["emit_gap"] = (rec["lat"] is not None and abs(rec["lat"]) > EMIT_LAT)
        rec["score"] = score_scene(rec)
        recs.append(rec)

    print(f"[survey] walked {n} items; kept {len(recs)} after filters")
    if not recs:
        sys.exit("no scenes matched filters")

    recs.sort(key=lambda r: r["score"], reverse=True)

    # ---- write full table ----
    cols = ["score", "id", "collection", "datetime", "lat", "lon", "sun_elev",
            "emit_gap", "has_ortho_sr", "has_ortho_rad", "month", "self_href"]
    with open(os.path.join(args.outdir, "catalog_survey.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)
    with open(os.path.join(args.outdir, "catalog_survey.json"), "w") as f:
        json.dump(recs, f, indent=2)

    # ---- shortlist ----
    top = recs[: args.top]
    lines = [f"TANAGER CATALOG SURVEY -- top {len(top)} of {len(recs)} scenes",
             f"scored for cryosphere target potential (melt season x EMIT gap x "
             f"sun x SR asset x glacier hint)", ""]
    hdr = (f"{'score':>5s}  {'lat':>7s} {'mon':>3s} {'sun':>4s} {'SR':>3s} "
           f"{'EMIT':>4s}  {'collection':<16s} id")
    lines.append(hdr); lines.append("-" * len(hdr))
    for r in top:
        lines.append(
            f"{r['score']:5.1f}  {(_num(r['lat'])):>7s} {(_num(r['month'],0)):>3s} "
            f"{(_num(r['sun_elev'])):>4s} {'Y' if r['has_ortho_sr'] else '-':>3s} "
            f"{'Y' if r['emit_gap'] else '-':>4s}  {r['collection'][:16]:<16s} {r['id']}")
    lines.append("")
    lines.append("score breakdown for the top 5:")
    for r in top[:5]:
        p = r["score_parts"]
        lines.append(f"  {r['id']}: " + ", ".join(f"{k}={v:g}" for k, v in p.items()))
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.outdir, "survey_shortlist.txt"), "w") as f:
        f.write(report)

    # ---- thumbnails / contact sheet ----
    if args.thumbnails:
        make_contact_sheet(top, args.outdir)

    print(f"\nwrote {args.outdir}/catalog_survey.csv  (full, sortable)")
    print(f"wrote {args.outdir}/survey_shortlist.txt")
    print("\nNEXT: eyeball the top few. Metadata cannot tell sea ice from glacier;")
    print("      the thumbnail can. Re-run with --thumbnails for a contact sheet,")
    print("      then run 01_scene_recon.py --item <self_href> on your pick.")


def make_contact_sheet(top, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        import io as _io
    except ImportError:
        print("[thumbs] need matplotlib + pillow: "
              "conda install -c conda-forge matplotlib pillow")
        return
    tdir = os.path.join(outdir, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    imgs = []
    for r in top:
        if not r["thumb"]:
            imgs.append((r, None)); continue
        try:
            b = requests.get(r["thumb"], timeout=60).content
            with open(os.path.join(tdir, r["id"] + ".png"), "wb") as f:
                f.write(b)
            imgs.append((r, Image.open(_io.BytesIO(b)).convert("RGB")))
        except Exception as ex:
            print(f"  [warn] thumb {r['id']}: {ex}")
            imgs.append((r, None))
    ncol = 4
    nrow = math.ceil(len(imgs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    for ax, (r, im) in zip(axes.ravel(), imgs):
        if im is not None:
            ax.imshow(im)
        ax.set_title(f"{r['score']:.0f} | {r['id'][:18]}\n"
                     f"lat {r['lat']} m{r['month']} sun{r['sun_elev']}",
                     fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(imgs):]:
        ax.axis("off")
    fig.suptitle("Tanager finalists -- eyeball for glacier ice / algae / debris "
                 "vs plain sea ice")
    fig.tight_layout()
    p = os.path.join(outdir, "contact_sheet.png")
    fig.savefig(p, dpi=110)
    print(f"[thumbs] wrote {p}")


def _num(x, nd=1):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    sys.exit(main())
