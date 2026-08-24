#!/usr/bin/env python3
"""
03_inspect_h5.py -- download a Tanager HDF5 asset and dump its actual structure.

Why this exists: tanager_ice/io.py was written to the *documented* schema for the
release-1 'basic_radiance' product. Release 2 ships surface-reflectance cubes
('ortho_sr_hdf5') whose field names are NOT known here. Rather than guess, dump
the real tree once and adapt the loader to what is actually there.

Usage (from the repo root):
    python 03_inspect_h5.py                          # ortho_sr_hdf5 (default)
    python 03_inspect_h5.py --asset ortho_radiance_hdf5
    python 03_inspect_h5.py --no-download            # inspect an already-cached file
    python 03_inspect_h5.py --max-depth 3

Writes: outputs/h5_structure_<asset>.txt   (paste this back)
        cache/<filename>.h5                (resumable download)

Deps: h5py, numpy, requests
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import requests

try:
    import h5py
except ImportError:
    sys.exit("h5py required:  conda install -c conda-forge h5py")


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def download(href: str, cache_dir: str = "cache") -> str:
    """Resumable streaming download with a progress line."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(href.split("?")[0]))

    head = requests.head(href, timeout=60, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0))
    have = os.path.getsize(dest) if os.path.exists(dest) else 0

    if total and have == total:
        print(f"[cache] already complete: {dest} ({human(total)})")
        return dest
    if have and total and have < total:
        print(f"[cache] resuming {dest} at {human(have)} / {human(total)}")
        headers = {"Range": f"bytes={have}-"}
        mode = "ab"
    else:
        have, headers, mode = 0, {}, "wb"
        print(f"[fetch] {href}\n[fetch] size {human(total) if total else 'unknown'}")

    with requests.get(href, stream=True, timeout=600, headers=headers) as r:
        r.raise_for_status()
        done = have
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 22):   # 4 MB
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r[fetch] {human(done)} / {human(total)}  {pct:5.1f}%",
                          end="", flush=True)
                else:
                    print(f"\r[fetch] {human(done)}", end="", flush=True)
    print()
    return dest


def fmt_attrs(obj, indent: str) -> list[str]:
    out = []
    for k, v in obj.attrs.items():
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8", "replace")
        s = str(v)
        if len(s) > 220:
            s = s[:220] + f" ... [truncated, len={len(str(v))}]"
        out.append(f"{indent}  @{k} = {s}")
    return out


def dump(h5: h5py.File, max_depth: int, sample_bands: int) -> list[str]:
    lines: list[str] = []
    band_like: list[str] = []

    def walk(name: str, obj):
        depth = name.count("/")
        if depth > max_depth:
            return
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            lines.append(f"{indent}[G] /{name}")
            lines.extend(fmt_attrs(obj, indent))
        else:
            lines.append(f"{indent}[D] /{name}  shape={obj.shape} dtype={obj.dtype}"
                         f" size={human(obj.nbytes)}")
            lines.extend(fmt_attrs(obj, indent))
            base = name.split("/")[-1]
            if any(t in base.lower() for t in
                   ("radiance", "reflect", "_b0", "_b1", "_b2", "_b3", "_b4")):
                band_like.append(name)

    h5.visititems(walk)

    lines.append("\n--- ROOT ATTRIBUTES ---")
    lines.extend(fmt_attrs(h5, ""))

    if band_like:
        lines.append(f"\n--- BAND-LIKE DATASETS: {len(band_like)} found ---")
        show = band_like[:sample_bands] + (["..."] if len(band_like) > sample_bands else [])
        for n in show:
            lines.append(f"    {n}")
        # inspect one for wavelength metadata + value range
        d = h5[band_like[0]]
        lines.append(f"\n--- SAMPLE DATASET: /{band_like[0]} ---")
        lines.append(f"    shape={d.shape} dtype={d.dtype}")
        lines.extend(fmt_attrs(d, "  "))
        try:
            # sample the CENTRE, not the corner: ortho grids are rotated within
            # their bounding box, so corners are legitimately -9999 fill.
            if d.ndim == 2:
                r, c = d.shape
                sub = d[max(0, r//2-25):r//2+25, max(0, c//2-25):c//2+25]
            elif d.ndim == 3:
                _, r, c = d.shape
                sub = d[d.shape[0]//2,
                        max(0, r//2-25):r//2+25, max(0, c//2-25):c//2+25]
            else:
                sub = d[: min(2500, d.size)]
            sub = np.asarray(sub, float)
            fin = sub[np.isfinite(sub) & (sub != -9999.0)]
            nfill = int((sub == -9999.0).sum())
            if fin.size:
                lines.append(f"    value range (50x50 CENTRE sample, fill excluded): "
                             f"min={fin.min():.4g} max={fin.max():.4g} "
                             f"mean={fin.mean():.4g}")
                lines.append(f"    fill pixels in sample: {nfill} / {sub.size}")
                lines.append("    NOTE: reflectance is typically 0-1 (or 0-10000 scaled);")
                lines.append("          radiance in W/(m^2 sr um) is typically ~1-200.")
            else:
                lines.append(f"    centre sample is ALL fill ({nfill}/{sub.size}) "
                             f"-- investigate, this is not just corner nodata")
        except Exception as e:
            lines.append(f"    (could not sample values: {e})")
    else:
        lines.append("\n--- NO BAND-LIKE DATASETS FOUND (check names above) ---")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/scene_meta.json")
    ap.add_argument("--asset", default="ortho_sr_hdf5")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--sample-bands", type=int, default=8)
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    assets = meta.get("assets", {})
    if args.asset not in assets:
        sys.exit(f"asset {args.asset!r} not in scene_meta.json; have: {sorted(assets)}")
    href = assets[args.asset]["href"]

    if args.no_download:
        path = os.path.join("cache", os.path.basename(href.split("?")[0]))
        if not os.path.exists(path):
            sys.exit(f"not cached: {path} (drop --no-download)")
    else:
        path = download(href)

    print(f"[open] {path} ({human(os.path.getsize(path))})")
    with h5py.File(path, "r") as h5:
        lines = dump(h5, args.max_depth, args.sample_bands)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"h5_structure_{args.asset}.txt")
    header = [f"# {args.asset}", f"# file: {path}",
              f"# size: {human(os.path.getsize(path))}", ""]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines))

    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"\n... [{len(lines)-60} more lines]")
    print(f"\nwrote {out}  <-- paste this back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
