#!/usr/bin/env python3
"""
01_scene_recon.py  --  Tanager snow/ice scene reconnaissance.

Pulls a Tanager Open-Data STAC *item*, and writes:
    outputs/scene_meta.json        - flattened metadata we actually need downstream
    outputs/scene_footprint.geojson - footprint polygon (+ centroid) for map/overlay work

It reports, against the actual item metadata (never the scene ID):
    - exact footprint, centroid latitude/longitude, and the EMIT +/-52 deg check
    - sun elevation (and the low-sun retrieval caveat)
    - band coverage at the wavelengths that matter for snow/ice retrievals
    - which data assets exist and at what processing level (basic vs ortho)

IMPORTANT REALITY CHECK (confirmed from the live STAC item, 2025):
    The open Tanager product is TOA RADIANCE in an HDF5 asset
    ('basic_radiance_hdf5', bands 'toa_radiance_B000'..'B425',
    units W/(m^2 sr um)) -- NOT surface reflectance, and NOT a
    reflectance COG. The only COG in the item is the usable-data/cloud
    mask ('basic_beta_udm'). Plan retrievals accordingly (see notes at
    bottom of this file).

Deps: requests  (pystac optional; we parse JSON directly to avoid a hard dep)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any

import requests

# --- wavelengths that matter for this project (nanometres) -------------------
# Kept explicit so the report is self-documenting.
KEY_WAVELENGTHS_NM = {
    "grain_size_1030": 1030.0,   # Nolin-Dozier scaled band-area (ice absorption)
    "grain_size_1250": 1250.0,   # secondary grain-size feature
    "melt_980":         980.0,   # liquid-water / continuum-removal region (~900-1050)
    "sediment_vnir":    650.0,   # VNIR slope/absorption for impurity/"dirty" ice
    "swir_snr_2300":   2300.0,   # high-SNR SWIR region (sensor spec check)
}
EMIT_LAT_LIMIT = 52.0  # EMIT acquires only roughly within +/-52 deg latitude


def browser_to_raw(url: str) -> str:
    """STAC Browser URL -> raw item JSON URL (remove the '/browser' segment)."""
    return url.replace("/data/stac/browser/", "/data/stac/", 1)


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def polygon_centroid(coords: list) -> tuple[float, float]:
    """Planar centroid of the exterior ring (good enough for a ~20 km footprint)."""
    ring = coords[0]
    # drop repeated closing vertex if present
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-12:  # degenerate -> fall back to vertex mean
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return sum(xs) / n, sum(ys) / n
    a *= 0.5
    return cx / (6 * a), cy / (6 * a)


def find_radiance_asset(assets: dict) -> tuple[str, dict]:
    """Return (key, asset) for the primary hyperspectral cube.

    Release 2 ships BOTH surface reflectance ('*_sr_hdf5', roles include
    'reflectance') and TOA radiance ('*_radiance_hdf5'), in ortho and basic
    framings. Prefer ortho SR; fall back through ortho radiance -> basic SR ->
    basic radiance. Use --asset on the CLI to force a specific one (the SR-vs-
    TOA comparison needs both).
    """
    preference = [
        "ortho_sr_hdf5", "ortho_reflectance", "reflectance",
        "ortho_radiance_hdf5", "ortho_radiance",
        "basic_sr_hdf5",
        "basic_radiance_hdf5", "basic_radiance", "radiance",
    ]
    for key in preference:
        if key in assets:
            return key, assets[key]
    # last resort: any asset that carries per-band eo:center_wavelength
    for key, a in assets.items():
        if any("eo:center_wavelength" in b for b in a.get("bands", [])):
            return key, a
    raise SystemExit("No hyperspectral radiance/reflectance asset found in item.")


def spectral_axis_nm(asset: dict) -> list[tuple[str, float, float]]:
    """Return [(band_name, center_nm, fwhm_nm), ...] for spectral bands only."""
    out = []
    for b in asset.get("bands", []):
        cw = b.get("eo:center_wavelength")
        if cw is None:
            continue  # geometry/mask bands (sun_zenith, masks, ...) have no wavelength
        fwhm = b.get("eo:full_width_half_max")
        # STAC eo:* wavelengths are micrometres -> convert to nm
        out.append((b.get("name", "?"), cw * 1000.0, (fwhm or 0.0) * 1000.0))
    out.sort(key=lambda t: t[1])
    return out


def nearest_band(axis, target_nm):
    return min(axis, key=lambda t: abs(t[1] - target_nm))


def sun_elevation_from_props(props: dict):
    """Scalar sun elevation if present. Returns (value_or_None, source_str)."""
    if "view:sun_elevation" in props:
        return float(props["view:sun_elevation"]), "properties.view:sun_elevation"
    if "view:sun_zenith" in props:
        return 90.0 - float(props["view:sun_zenith"]), "90 - properties.view:sun_zenith"
    return None, ("per-pixel only: read 'sun_zenith' band from the HDF5 and use "
                  "elevation = 90 - mean(sun_zenith)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Tanager scene reconnaissance")
    ap.add_argument(
        "--item",
        default=("https://www.planet.com/data/stac/browser/tanager-core-imagery/"
                 "snow-ice/20250516_132954_74_4001/20250516_132954_74_4001.json"),
        help="STAC item URL (browser or raw form both accepted)",
    )
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--asset", default=None,
                    help="force a specific asset key, e.g. ortho_radiance_hdf5")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_url = browser_to_raw(args.item)
    print(f"[recon] fetching item: {raw_url}")
    item = fetch_json(raw_url)

    props = item.get("properties", {})
    geom = item.get("geometry")
    bbox = item.get("bbox")
    if geom is None or bbox is None:
        raise SystemExit("Item is missing geometry/bbox; cannot proceed.")

    lon_c, lat_c = polygon_centroid(geom["coordinates"])
    lat_min, lat_max = bbox[1], bbox[3]

    if args.asset:
        if args.asset not in item["assets"]:
            raise SystemExit(f"asset {args.asset!r} not in item; have: "
                             f"{sorted(item['assets'])}")
        rad_key, rad_asset = args.asset, item["assets"][args.asset]
    else:
        rad_key, rad_asset = find_radiance_asset(item["assets"])
    axis = spectral_axis_nm(rad_asset)
    _roles = [r.lower() for r in (rad_asset.get("roles") or [])]
    is_reflectance = ("reflectance" in _roles or "reflect" in rad_key.lower()
                      or re.search(r"(^|_)sr(_|$)", rad_key.lower()) is not None)

    sun_elev, sun_src = sun_elevation_from_props(props)

    # --- band coverage report ------------------------------------------------
    coverage = {}
    for label, tgt in KEY_WAVELENGTHS_NM.items():
        if not axis:
            coverage[label] = None
            continue
        name, cw, fwhm = nearest_band(axis, tgt)
        coverage[label] = {
            "target_nm": tgt, "nearest_band": name,
            "center_nm": round(cw, 2), "fwhm_nm": round(fwhm, 2),
            "offset_nm": round(cw - tgt, 2),
        }
    # median sampling interval as a spec check
    sampling = None
    if len(axis) > 1:
        deltas = [axis[i + 1][1] - axis[i][1] for i in range(len(axis) - 1)]
        deltas.sort()
        sampling = round(deltas[len(deltas) // 2], 3)

    # --- asset / processing-level inventory ---------------------------------
    asset_inventory = {}
    for k, a in item["assets"].items():
        asset_inventory[k] = {
            "href": a.get("href"),
            "type": a.get("type"),
            "roles": a.get("roles"),
            "proj:code": a.get("proj:code"),
            "proj:shape": a.get("proj:shape"),
        }
    has_ortho = any(k.startswith("ortho") for k in item["assets"])
    sr_keys = [k for k, a in item["assets"].items()
               if "reflectance" in [r.lower() for r in (a.get("roles") or [])]]
    rad_keys = [k for k in item["assets"] if "radiance" in k]

    meta = {
        "id": item.get("id"),
        "collection": item.get("collection"),
        "datetime": props.get("datetime"),
        "platform": props.get("platform"),
        "constellation": props.get("constellation"),
        "gsd_m": props.get("gsd") or rad_asset.get("bands", [{}])[0]
                  .get("raster:spatial_resolution"),
        "footprint_bbox": bbox,
        "centroid_lonlat": [round(lon_c, 5), round(lat_c, 5)],
        "lat_min": lat_min, "lat_max": lat_max,
        "emit_gap": {
            "limit_deg": EMIT_LAT_LIMIT,
            "centroid_clears": abs(lat_c) > EMIT_LAT_LIMIT,
            "entire_footprint_clears": min(abs(lat_min), abs(lat_max)) > EMIT_LAT_LIMIT,
        },
        "sun_elevation_deg": sun_elev,
        "sun_elevation_source": sun_src,
        "view": {k: props.get(k) for k in props if k.startswith("view:")},
        "product": {
            "radiance_asset_key": rad_key,
            "is_surface_reflectance": is_reflectance,
            "radiometric_units": rad_asset.get("bands", [{}])[-1].get("unit"),
            "n_spectral_bands": len(axis),
            "spectral_range_nm": [round(axis[0][1], 2), round(axis[-1][1], 2)] if axis else None,
            "median_sampling_nm": sampling,
            "has_ortho_or_reflectance_asset": has_ortho,
            "surface_reflectance_assets": sr_keys,
            "radiance_assets": rad_keys,
        },
        "band_coverage": coverage,
        "assets": asset_inventory,
        "data_asset_href": rad_asset.get("href"),
    }

    meta_path = os.path.join(args.outdir, "scene_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"id": item.get("id"), "role": "footprint"},
             "geometry": geom},
            {"type": "Feature", "properties": {"id": item.get("id"), "role": "centroid"},
             "geometry": {"type": "Point", "coordinates": [lon_c, lat_c]}},
        ],
    }
    fp_path = os.path.join(args.outdir, "scene_footprint.geojson")
    with open(fp_path, "w") as f:
        json.dump(geojson, f, indent=2)

    # --- human-readable report ----------------------------------------------
    print("\n================= TANAGER SCENE RECON =================")
    print(f"id            : {meta['id']}")
    print(f"datetime      : {meta['datetime']}")
    print(f"centroid      : lat {lat_c:.4f}, lon {lon_c:.4f}")
    print(f"bbox lat span : {lat_min:.4f} .. {lat_max:.4f}")
    clears = meta["emit_gap"]["entire_footprint_clears"]
    print(f"EMIT +/-52    : entire footprint clears? {clears}  "
          f"(EMIT structurally cannot see this scene => {'HOLDS' if clears else 'CHECK'})")
    if sun_elev is not None:
        caveat = "  <-- LOW SUN: absolute retrievals high-uncertainty" if sun_elev < 25 else ""
        print(f"sun elevation : {sun_elev:.2f} deg{caveat}")
    else:
        print(f"sun elevation : {sun_src}")
    print(f"GSD           : {meta['gsd_m']} m")
    p = meta["product"]
    print(f"product       : key='{p['radiance_asset_key']}'  "
          f"reflectance={p['is_surface_reflectance']}  units={p['radiometric_units']}")
    print(f"spectral      : {p['n_spectral_bands']} bands, "
          f"{p['spectral_range_nm']} nm, ~{p['median_sampling_nm']} nm sampling")
    print(f"ortho?        : {p['has_ortho_or_reflectance_asset']}")
    print(f"SR assets     : {p['surface_reflectance_assets'] or 'NONE'}")
    print(f"radiance      : {p['radiance_assets'] or 'NONE'}")
    if p['surface_reflectance_assets'] and p['radiance_assets']:
        print("                -> both SR and TOA available: run the AC-sensitivity comparison")
    print("band coverage :")
    for label, c in coverage.items():
        if c:
            print(f"   {label:16s} -> {c['nearest_band']:16s} "
                  f"{c['center_nm']:8.2f} nm (off {c['offset_nm']:+.2f}, fwhm {c['fwhm_nm']:.2f})")
    print(f"\nwrote {meta_path}")
    print(f"wrote {fp_path}")
    print("=======================================================\n")
    return 0


# ---------------------------------------------------------------------------
# DOWNSTREAM NOTE (read before Task 2/3):
#   * Data is TOA radiance, not reflectance. Options, in order of defensibility:
#       (a) relative/spectral-shape retrievals directly on radiance (continuum
#           removal and band-area ratios are largely gain-robust) + report as
#           gradients, not absolute grain radius / LWC;
#       (b) run ISOFIT/6S atmospheric correction to surface reflectance yourself
#           (hard over bright ice + low polar sun; carry the added uncertainty);
#       (c) a hybrid: TOA continuum-removal for indices, plus a coarse Rayleigh/
#           gas correction for the VNIR sediment slope.
#   * "basic" = georeferenced but UNPROJECTED (native sensor geometry). The HDF5
#     carries a Planet_Ortho_Framing (epsg+geotransform) and per-pixel sun/sensor
#     geometry bands; the only ready-to-overlay COG is the cloud/UDM mask
#     (EPSG:4326). All cross-sensor overlay work (S1/S2/ICESat-2) must resolve
#     this geometry first. Check scene_meta.json['product']['has_ortho...'] --
#     if an ortho asset is present, prefer it for Task 4.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
