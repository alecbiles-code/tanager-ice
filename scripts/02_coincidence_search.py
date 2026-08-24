#!/usr/bin/env python3
"""
02_coincidence_search.py  --  find cross-sensor validation partners for the scene.

Reads outputs/scene_meta.json (from 01_scene_recon.py) and searches for
near-coincident acquisitions that can validate the Tanager scene:

    detection truth   : Sentinel-1 GRD (SAR, all-weather, orthogonal physics)
    "what HS adds"     : Sentinel-2 L2A + Landsat C2 L2 (Landsat 30 m ~= Tanager)
    melt corroboration : Landsat TIRS ST / (Sentinel-3 SLSTR -- see note)
    height/freeboard   : ICESat-2 ATL03/07/10 (the coincidence bottleneck)

Writes outputs/coincidence_report.json: every partner sorted by |time offset|,
plus a per-track gate verdict and an overall project-viability verdict.

Auth / environment (this script must run where these are reachable):
    pip install pystac-client planetary-computer earthaccess shapely
    earthaccess: needs a (free) NASA Earthdata login -> earthaccess.login()
    Planetary Computer: anonymous search; assets signed via planetary_computer.sign

Design choices:
    * time window is widened in stages (default +/-3d, then +/-10d) because the
      ICESat-2 track crossing a ~20 km footprint is the main risk.
    * S1/S2/Landsat almost always have partners at this latitude; ICESat-2 is the
      gate. If ICESat-2 misses, the report says so plainly and flags the
      degradation experiment (Task 4c) as the guaranteed-to-work fallback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any

# ---- optional imports kept lazy so the file imports/compiles without them ----
def _need(mod: str):
    try:
        return __import__(mod)
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency '{mod}'. Install: "
            f"pip install pystac-client planetary-computer earthaccess shapely"
        ) from e


PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_COLLECTIONS = {
    "sentinel-1": "sentinel-1-grd",
    "sentinel-2": "sentinel-2-l2a",
    "landsat":    "landsat-c2-l2",   # OLI/TIRS surface reflectance + surface temp
}
IS2_SHORT_NAMES = ["ATL03", "ATL07", "ATL10"]  # photons, sea-ice height, freeboard


def load_meta(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def parse_dt(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def offset_days(a: dt.datetime, b: dt.datetime) -> float:
    return abs((a - b).total_seconds()) / 86400.0


def search_pc(footprint_geojson: dict, t0: dt.datetime, days: float) -> list[dict]:
    pystac_client = _need("pystac_client")
    planetary_computer = _need("planetary_computer")
    lo = (t0 - dt.timedelta(days=days)).date().isoformat()
    hi = (t0 + dt.timedelta(days=days)).date().isoformat()
    cat = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    results = []
    for label, coll in PC_COLLECTIONS.items():
        try:
            search = cat.search(collections=[coll], intersects=footprint_geojson,
                                datetime=f"{lo}/{hi}", max_items=50)
            for it in search.items():
                t = parse_dt(it.properties["datetime"])
                results.append({
                    "sensor": label, "collection": coll, "id": it.id,
                    "datetime": it.properties["datetime"],
                    "offset_days": round(offset_days(t, t0), 3),
                    "platform": it.properties.get("platform"),
                    "eo:cloud_cover": it.properties.get("eo:cloud_cover"),
                    "sar:polarizations": it.properties.get("sar:polarizations"),
                    "self_href": it.get_self_href(),
                })
        except Exception as e:  # keep going; one collection outage shouldn't kill the run
            results.append({"sensor": label, "collection": coll, "error": str(e)})
    return results


def search_icesat2(bbox: list[float], t0: dt.datetime, days: float) -> list[dict]:
    earthaccess = _need("earthaccess")
    earthaccess.login(strategy="environment", persist=True)  # needs EDL creds
    lo = (t0 - dt.timedelta(days=days)).date().isoformat()
    hi = (t0 + dt.timedelta(days=days)).date().isoformat()
    out = []
    for short in IS2_SHORT_NAMES:
        try:
            grans = earthaccess.search_data(
                short_name=short, bounding_box=tuple(bbox),
                temporal=(lo, hi), count=100)
            for g in grans:
                try:
                    ts = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
                    off = round(offset_days(parse_dt(ts), t0), 3)
                except Exception:
                    ts, off = None, None
                out.append({
                    "sensor": "icesat-2", "product": short,
                    "granule": g["umm"].get("GranuleUR"),
                    "datetime": ts, "offset_days": off,
                })
        except Exception as e:
            out.append({"sensor": "icesat-2", "product": short, "error": str(e)})
    return out


def gate_verdict(partners: list[dict]) -> dict:
    def best(sensor_key, field="sensor"):
        cand = [p for p in partners
                if p.get(field) == sensor_key and p.get("offset_days") is not None]
        return min((p["offset_days"] for p in cand), default=None)
    s1 = best("sentinel-1"); s2 = best("sentinel-2")
    ls = best("landsat");    is2 = best("icesat-2")
    return {
        "sentinel1_best_offset_days": s1,
        "sentinel2_best_offset_days": s2,
        "landsat_best_offset_days": ls,
        "icesat2_best_offset_days": is2,
        "detection_axis_ok": s1 is not None,                # SAR cross-check
        "hs_added_value_axis_ok": (s2 is not None or ls is not None),
        "freeboard_axis_ok": is2 is not None,               # the bottleneck
        "verdict": (
            "FULL: SAR + optical + ICESat-2 all present"
            if (s1 is not None and (s2 or ls) and is2 is not None) else
            "CORE: SAR + optical present; ICESat-2 opportunistic/missing "
            "-> lead with degradation experiment (Task 4c) for the reproducible core"
            if (s1 is not None and (s2 or ls)) else
            "THIN: re-check footprint/window; some primary partners missing"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--windows", default="3,10",
                    help="comma-sep day half-windows to try, in order")
    ap.add_argument("--skip-icesat2", action="store_true",
                    help="skip NASA Earthdata query (e.g. no creds available)")
    args = ap.parse_args()

    meta = load_meta(args.meta)
    t0 = parse_dt(meta["datetime"])
    bbox = meta["footprint_bbox"]
    # rebuild a footprint geojson for intersects= (recon wrote it separately too)
    fp_path = os.path.join(args.outdir, "scene_footprint.geojson")
    footprint = None
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            fc = json.load(f)
        footprint = next((ft["geometry"] for ft in fc["features"]
                          if ft["properties"].get("role") == "footprint"), None)
    if footprint is None:  # fall back to bbox polygon
        footprint = {"type": "Polygon", "coordinates": [[
            [bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]],
            [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]}

    windows = [float(x) for x in args.windows.split(",")]
    partners: list[dict] = []
    used_window = None
    for w in windows:
        print(f"[coincidence] searching +/-{w:g} d ...")
        partners = search_pc(footprint, t0, w)
        if not args.skip_icesat2:
            partners += search_icesat2(bbox, t0, w)
        used_window = w
        v = gate_verdict(partners)
        # stop early if we already have SAR + optical (+ ideally ICESat-2)
        if v["detection_axis_ok"] and v["hs_added_value_axis_ok"]:
            break

    partners_sorted = sorted(
        [p for p in partners if "error" not in p],
        key=lambda p: (p.get("offset_days") is None, p.get("offset_days", 9e9)))
    errors = [p for p in partners if "error" in p]

    report = {
        "scene_id": meta["id"],
        "scene_datetime": meta["datetime"],
        "search_window_days": used_window,
        "n_partners": len(partners_sorted),
        "gate": gate_verdict(partners),
        "partners_sorted_by_time_offset": partners_sorted,
        "errors": errors,
    }
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "coincidence_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    g = report["gate"]
    print("\n============== COINCIDENCE REPORT ==============")
    print(f"scene       : {report['scene_id']}  @ {report['scene_datetime']}")
    print(f"window       : +/-{used_window:g} d   partners: {report['n_partners']}")
    print(f"Sentinel-1   : best offset {g['sentinel1_best_offset_days']} d")
    print(f"Sentinel-2   : best offset {g['sentinel2_best_offset_days']} d")
    print(f"Landsat      : best offset {g['landsat_best_offset_days']} d")
    print(f"ICESat-2     : best offset {g['icesat2_best_offset_days']} d")
    print(f"VERDICT      : {g['verdict']}")
    if errors:
        print(f"({len(errors)} collection/query errors -- see report['errors'])")
    print(f"\nwrote {out_path}")
    print("===============================================\n")
    return 0


# NOTE: Sentinel-3 SLSTR LST is not in the Planetary Computer collection set used
# here; pull it from CDSE/EUMETSAT if you want the thermal melt corroboration
# layer. Landsat C2 L2 already provides a surface-temperature band (ST_B10),
# which covers the melt-corroboration axis at 30/100 m without a second provider.
if __name__ == "__main__":
    sys.exit(main())
