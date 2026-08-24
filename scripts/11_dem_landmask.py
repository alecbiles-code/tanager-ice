#!/usr/bin/env python3
"""
11_dem_landmask.py -- fetch a DEM, coregister to the Tanager grid, emit:
    * elevation on the exact scene grid
    * a land/sea mask (elevation > threshold)   <- finalises snow_terrain vs sea_ice
    * slope + aspect + local solar incidence     <- inputs for topo correction (C-corr)

Why this exists: the terrain-vs-sea-ice naming cannot be done reliably from the
imagery alone (texture confuses brash ice with ridges; connectivity fails where
landfast ice welds sea to coast). Elevation is unambiguous: land > 0 m, sea ~ 0.
And we need the DEM anyway for topographic correction of the grain-size retrieval
over terrain -- so this single step unblocks BOTH.

DEM source: Copernicus GLO-30 (30 m, ~matches Tanager 33 m) via the Microsoft
Planetary Computer STAC (anonymous read; assets signed by planetary_computer).
Falls back to a user-provided --dem-file (any GDAL-readable raster) if PC is
unreachable.

Coregistration: the Tanager ortho grid geotransform + EPSG come from the scene
HDF5 (io.Scene.framing / .epsg). We warp the DEM into that exact grid with
rasterio so every DEM pixel lines up 1:1 with a reflectance pixel.

Usage (repo root; needs internet + `pip install planetary-computer pystac-client
rasterio`):
    python 11_dem_landmask.py
    python 11_dem_landmask.py --dem-file mydem.tif      # skip PC, use local raster
    python 11_dem_landmask.py --sea-level 5             # m threshold for 'land'

Writes: outputs/dem_on_grid.npy         elevation (H x W) metres
        outputs/land_mask.npy           bool (H x W), True = land
        outputs/topo.npz                slope, aspect, cos_i (local solar incidence)
        outputs/dem_landmask.png        quicklook
        outputs/dem_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from tanager_ice import io


def fetch_dem_pc(bbox_wgs84, cache_dir="cache"):
    """Return a local path to a DEM covering bbox via Planetary Computer, or None."""
    try:
        import pystac_client
        import planetary_computer
        import rasterio
        from rasterio.merge import merge
    except ImportError as e:
        print(f"[dem] PC path unavailable ({e}); use --dem-file")
        return None
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)
    for coll in ("cop-dem-glo-30", "cop-dem-glo-90", "nasadem"):
        try:
            search = cat.search(collections=[coll], bbox=bbox_wgs84)
            items = list(search.items())
            if not items:
                continue
            print(f"[dem] {coll}: {len(items)} tile(s)")
            srcs = []
            os.makedirs(cache_dir, exist_ok=True)
            for it in items:
                # DEM asset key varies by collection
                akey = "data" if "data" in it.assets else list(it.assets)[0]
                href = it.assets[akey].href
                srcs.append(rasterio.open(href))
            mosaic, transform = merge(srcs)
            out_path = os.path.join(cache_dir, f"dem_{coll}.tif")
            meta = srcs[0].meta.copy()
            meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                        transform=transform)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic)
            for s_ in srcs:
                s_.close()
            print(f"[dem] wrote {out_path}")
            return out_path
        except Exception as ex:
            print(f"[dem] {coll} failed: {ex}")
    return None


def warp_to_grid(dem_path, framing, epsg, H, W):
    """Warp a DEM raster into the Tanager grid (framing geotransform + epsg)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine

    gt = framing["geotransform"]           # (ulx, xres, 0, uly, 0, yres)
    dst_transform = Affine.from_gdal(*gt)
    dst = np.full((H, W), np.nan, np.float32)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=f"EPSG:{epsg}",
            resampling=Resampling.bilinear,
            dst_nodata=np.nan)
    return dst


def slope_aspect(dem, xres, yres):
    """Slope + aspect (radians) from a DEM via Horn's method."""
    dzdx = np.gradient(dem, axis=1) / xres
    dzdy = np.gradient(dem, axis=0) / abs(yres)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdy, dzdx)          # 0=E, math convention
    return slope, aspect


def cos_incidence(slope, aspect, sun_zenith_deg, sun_azimuth_deg):
    """Cosine of local solar incidence angle (for cosine / C-correction)."""
    sz = np.radians(sun_zenith_deg)
    sa = np.radians(sun_azimuth_deg)
    # convert aspect(math,0=E,CCW) to azimuth(0=N,CW) to match sun azimuth
    aspect_az = (np.pi / 2 - aspect) % (2 * np.pi)
    cos_i = (np.cos(sz) * np.cos(slope) +
             np.sin(sz) * np.sin(slope) * np.cos(sa - aspect_az))
    return np.clip(cos_i, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--dem-file", default=None, help="local DEM raster (skips PC)")
    ap.add_argument("--sea-level", type=float, default=5.0,
                    help="elevation (m) below which a pixel is 'sea'")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    meta = io.load_meta(args.meta)
    a = meta["assets"][args.asset]
    path = os.path.join("cache", os.path.basename(a["href"].split("?")[0]))
    if not os.path.exists(path):
        sys.exit(f"{args.asset} not cached")

    with io.Scene(path) as s:
        H, W = s.rows, s.cols
        framing = s.framing
        epsg = int(s.epsg)
        if framing is None:
            sys.exit("scene has no StructMetadata framing; cannot coregister DEM")
        gt = framing["geotransform"]
        xres, yres = gt[1], gt[5]
        sun_z = float(np.nanmean(s.plane("sun_zenith")))
        sun_a = float(np.nanmean(s.plane("sun_azimuth")))
        valid = s.valid_mask()
        bbox = meta["footprint_bbox"]

    dem_path = args.dem_file or fetch_dem_pc(bbox)
    if dem_path is None:
        sys.exit("no DEM available: provide --dem-file (GLO-30 GeoTIFF) and re-run")

    print(f"[dem] warping {dem_path} to Tanager grid {H}x{W} EPSG:{epsg}")
    dem = warp_to_grid(dem_path, framing, epsg, H, W)
    dem = np.where(valid, dem, np.nan)

    land = np.isfinite(dem) & (dem > args.sea_level)
    slope, aspect = slope_aspect(np.nan_to_num(dem, nan=0.0), xres, yres)
    cos_i = cos_incidence(slope, aspect, sun_z, sun_a)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "dem_on_grid.npy"), dem)
    np.save(os.path.join(args.outdir, "land_mask.npy"), land)
    np.savez(os.path.join(args.outdir, "topo.npz"),
             slope=slope, aspect=aspect, cos_i=cos_i,
             sun_zenith=sun_z, sun_azimuth=sun_a)

    rep = {
        "dem_source": os.path.basename(dem_path),
        "sea_level_m": args.sea_level,
        "land_fraction_of_valid": round(float(land[valid].mean()), 4),
        "elev_min": float(np.nanmin(dem)), "elev_max": float(np.nanmax(dem)),
        "elev_mean_land": round(float(np.nanmean(dem[land])), 1) if land.any() else None,
        "sun_zenith": round(sun_z, 2), "sun_azimuth": round(sun_a, 2),
        "cos_i_range": [round(float(np.nanmin(cos_i[land])), 3),
                        round(float(np.nanmax(cos_i[land])), 3)] if land.any() else None,
    }
    with open(os.path.join(args.outdir, "dem_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 6))
        im = ax[0].imshow(dem, cmap="terrain"); plt.colorbar(im, ax=ax[0], fraction=0.046)
        ax[0].set_title("DEM on Tanager grid (m)")
        ax[1].imshow(land, cmap="gray"); ax[1].set_title(
            f"land mask (elev > {args.sea_level} m): "
            f"{100*land[valid].mean():.0f}% land")
        im = ax[2].imshow(np.where(land, cos_i, np.nan), cmap="magma")
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        ax[2].set_title("cos(local solar incidence)\n-> topo correction input")
        fig.suptitle(f"DEM coregistration -- {meta['id']}")
        fig.tight_layout()
        p = os.path.join(args.outdir, "dem_landmask.png")
        fig.savefig(p, dpi=125); plt.close(fig)
        print(f"[dem] wrote {p}")
    except Exception as e:
        print(f"[dem] plot skipped: {e}")

    print("\n=== DEM SUMMARY ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {args.outdir}/dem_on_grid.npy, land_mask.npy, topo.npz")
    print("NEXT: re-run 10_segment naming with this land mask, then grain-size "
          "retrieval (flat now, terrain after C-correction using topo.npz).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
