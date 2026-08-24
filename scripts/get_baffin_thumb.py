#!/usr/bin/env python3
"""Fetch and display one Tanager scene thumbnail by scene id or item URL."""
import sys, requests, io, os, json

URL = ("https://www.planet.com/data/stac/tanager-core-imagery/snow-ice/"
       "20250606_181248_58_4001/20250606_181248_58_4001.json")

def main():
    item_url = sys.argv[1] if len(sys.argv) > 1 else URL
    print(f"[fetch] item {item_url}")
    item = requests.get(item_url, timeout=60).json()
    thumb = item["assets"].get("thumbnail", {}).get("href")
    vis   = item["assets"].get("ortho_visual", {}).get("href")
    props = item.get("properties", {})
    geom  = item.get("geometry", {})
    # quick facts
    import statistics
    lons = [c[0] for c in geom["coordinates"][0]]
    lats = [c[1] for c in geom["coordinates"][0]]
    print(f"[facts] lat {min(lats):.3f}..{max(lats):.3f}  lon {min(lons):.3f}..{max(lons):.3f}")
    print(f"[facts] datetime {props.get('datetime')}")
    se = props.get('view:sun_elevation')
    if se is None and props.get('view:sun_zenith') is not None:
        se = 90 - props['view:sun_zenith']
    print(f"[facts] sun elevation {se}")
    print(f"[facts] assets: {sorted(item['assets'])}")

    os.makedirs("outputs", exist_ok=True)
    if thumb:
        b = requests.get(thumb, timeout=60).content
        p = "outputs/baffin_june_thumb.png"
        open(p, "wb").write(b)
        print(f"[thumb] wrote {p}  ({len(b)//1024} KB)  <-- OPEN THIS")
    else:
        print("[thumb] no thumbnail asset; try ortho_visual:", vis)

if __name__ == "__main__":
    main()
