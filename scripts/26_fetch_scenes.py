#!/usr/bin/env python3
"""
26_fetch_scenes.py -- fetch ortho_sr_hdf5 assets for open-catalog STAC items
into cache/, with resumable transfers so a dropped multi-GB download continues
instead of restarting.

Resolves each scene's STAC item JSON, reads the ortho_sr_hdf5 asset href, and
streams it to cache/<basename of href>. Re-running skips files that are already
complete and resumes partial ones via HTTP Range. Also writes a per-scene meta
JSON so the analysis scripts can be pointed at --metas instead of --cubes.

Usage:
  python 26_fetch_scenes.py --ids 20250515_202305_00_4001 20250511_011730_87_4001
  python 26_fetch_scenes.py --ids <id> --dry-run          (sizes only, no download)
  python 26_fetch_scenes.py --items <full STAC item URL> ...

After it finishes it prints the exact ready-to-paste survey command listing
every cube in cache/, so filename conventions never have to be guessed.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import urllib.request, urllib.error

STAC_BASE = "https://www.planet.com/data/stac/tanager-core-imagery"
ASSET_KEY = "ortho_sr_hdf5"
UA = {"User-Agent": "tanager-ice/1.0"}


def get_json(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:
            if k == tries - 1:
                raise
            print("   retry %d/%d after %s" % (k + 1, tries - 1, exc))
            time.sleep(2 * (k + 1))


def remote_size(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def human(n):
    if n is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def download(url, target, total):
    """Resumable stream to target. Returns True if the file is complete."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if total is not None and have == total:
        print("   already complete (%s)" % human(total))
        return True
    if total is not None and have > total:
        print("   local file larger than remote; re-fetching from scratch")
        os.remove(target); have = 0

    headers = dict(UA)
    mode = "wb"
    if have > 0:
        headers["Range"] = "bytes=%d-" % have
        mode = "ab"
        print("   resuming at %s" % human(have))

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:            # range beyond end -> already complete
            print("   server reports complete")
            return True
        raise
    if have > 0 and resp.status != 206:
        print("   server ignored Range; restarting from 0")
        have = 0; mode = "wb"

    t0 = time.time(); done = have
    with open(target, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total:
                pct = 100.0 * done / total
                rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                sys.stdout.write("\r   %5.1f%%  %s / %s  (%.1f MB/s)" % (
                    pct, human(done), human(total), rate))
            else:
                sys.stdout.write("\r   %s" % human(done))
            sys.stdout.flush()
    resp.close()
    print("")
    if total is not None and os.path.getsize(target) != total:
        print("   WARNING: size mismatch (%s vs %s) -- re-run to resume" % (
            human(os.path.getsize(target)), human(total)))
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--items", nargs="*", default=[], help="full STAC item JSON URLs")
    ap.add_argument("--collection", default="snow-ice")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    urls = list(args.items)
    for sid in args.ids:
        urls.append("%s/%s/%s/%s.json" % (STAC_BASE, args.collection, sid, sid))
    if not urls:
        sys.exit("pass --ids <scene id> ... or --items <item json url> ...")
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    ok, failed = [], []
    for i, url in enumerate(urls, 1):
        print("\n[%d/%d] %s" % (i, len(urls), url.rsplit("/", 1)[-1]))
        try:
            item = get_json(url)
        except Exception as exc:
            print("   FAILED to read STAC item: %s" % exc); failed.append(url); continue
        assets = item.get("assets", {})
        if ASSET_KEY not in assets:
            print("   no %s asset (has: %s)" % (ASSET_KEY, ", ".join(sorted(assets)[:6])))
            failed.append(url); continue
        href = assets[ASSET_KEY]["href"]
        clean = href.split("?")[0]
        fname = os.path.basename(clean)
        if not fname.endswith((".h5", ".hdf5", ".he5")):
            fname = "%s_%s.h5" % (item.get("id", "scene"), ASSET_KEY)
        target = os.path.join(args.cache, fname)
        total = remote_size(href)
        print("   asset: %s  (%s)" % (fname, human(total)))
        if args.dry_run:
            ok.append(target); continue
        try:
            complete = download(href, target, total)
        except Exception as exc:
            print("   FAILED: %s" % exc); failed.append(url); continue
        if not complete:
            failed.append(url); continue
        meta = {"id": item.get("id"), "datetime": item.get("properties", {}).get("datetime"),
                "assets": {ASSET_KEY: {"href": href}}}
        mpath = os.path.join(args.outdir, "%s_meta.json" % item.get("id", "scene"))
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2)
        print("   wrote %s and %s" % (target, mpath))
        ok.append(target)

    print("\n%d ok, %d failed" % (len(ok), len(failed)))
    if failed:
        print("re-run the same command to resume/retry the failures.")
    allh5 = sorted(f for f in os.listdir(args.cache)
                   if f.endswith((".h5", ".hdf5", ".he5")))
    # the survey needs SURFACE REFLECTANCE cubes; radiance cubes in the same
    # cache would be scanned and then skipped on the pixel threshold, which
    # clutters the report -- filter them out of the suggested command.
    cubes = [os.path.join(args.cache, f) for f in allh5 if "radiance" not in f.lower()]
    other = [f for f in allh5 if "radiance" in f.lower()]
    if other:
        print("\nnote: excluding %d radiance cube(s) from the survey command: %s" % (
            len(other), ", ".join(other)))
    if cubes:
        print("\nReady-to-paste survey command (%d SR cubes in %s):\n" % (len(cubes), args.cache))
        print("python scripts\\24_multiscene_ac_survey.py --cubes " + " ".join(cubes))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
