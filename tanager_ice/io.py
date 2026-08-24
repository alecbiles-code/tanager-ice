"""
tanager_ice.io
==============
Load a Tanager Open-Data (release 2) scene.

Schema below is the REAL one, confirmed by dumping an ortho_sr_hdf5 file --
not the documented release-1 layout:

    /HDFEOS/GRIDS/HYP/                       <- GRIDS (not SWATHS)
        @epsg_code, @strip_id, @created_at
        Data Fields/                         <- NOTE: space, not underscore
            surface_reflectance              (426, 935, 1007) float32
                @wavelengths       (426,) nm
                @fwhm              (426,) nm
                @good_wavelengths  (426,) 1=usable
                @_FillValue = -9999.0
            surface_reflectance_uncertainty  (426, 935, 1007) float32
            aerosol_optical_depth            (935, 1007) float32
            column_water_vapour              (935, 1007) g/cm^2
            sun_zenith / sun_azimuth / sensor_zenith / sensor_azimuth
            sensor_to_ground_path_length / time
            beta_cloud_mask / beta_cirrus_mask / nodata_pixels  uint8
    /HDFEOS INFORMATION/StructMetadata.0     <- grid corners -> geotransform

TOA radiance files (ortho_radiance_hdf5) use the same layout with a
'toa_radiance' dataset instead; this module handles both.

MEMORY: the cube is ~1.6 GB uncompressed (3.2 GB with uncertainty). Never read
it whole by default -- use `window=` / `bands=` on read_cube(), or
read_labeled_pixels() which reads only the bounding window of the labels.

Deps: h5py, numpy. rasterio optional (geotransform fallback via the UDM COG).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

FILL = -9999.0
_GRID_PATHS = ("/HDFEOS/GRIDS/HYP", "HDFEOS/GRIDS/HYP",
               "/HDFEOS/SWATHS/HYP", "HDFEOS/SWATHS/HYP")
_DATA_GROUPS = ("Data Fields", "Data_Fields")
_CUBE_NAMES = ("surface_reflectance", "toa_radiance", "radiance", "reflectance")


def load_meta(path="outputs/scene_meta.json"):
    with open(path) as f:
        return json.load(f)


def download(href, cache_dir="cache"):
    """Resumable stream download; returns local path."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(href.split("?")[0]))
    head = requests.head(href, timeout=60, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0))
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if total and have == total:
        return dest
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(href, stream=True, timeout=900, headers=headers) as r:
        r.raise_for_status()
        with open(dest, "ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return dest


# --------------------------------------------------------------------------
# structure discovery
# --------------------------------------------------------------------------
def _data_group(h5):
    """Return (data_group, grid_group), tolerating GRIDS/SWATHS and space/underscore."""
    for gp in _GRID_PATHS:
        if gp in h5:
            g = h5[gp]
            for dn in _DATA_GROUPS:
                if dn in g:
                    return g[dn], g
            return g, g
    found = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in _CUBE_NAMES:
            found.setdefault("p", "/".join(name.split("/")[:-1]))

    h5.visititems(visit)
    if "p" in found:
        grp = h5[found["p"]]
        return grp, grp.parent
    raise KeyError("Could not locate the HYP data group; run 03_inspect_h5.py.")


def _cube_dataset(data_grp):
    """Return (name, dataset) for the main hyperspectral cube."""
    for n in _CUBE_NAMES:
        if n in data_grp:
            return n, data_grp[n]
    for k in data_grp:
        d = data_grp[k]
        if isinstance(d, h5py.Dataset) and d.ndim == 3 and not k.endswith("uncertainty"):
            return k, d
    raise KeyError(f"No 3-D cube found among: {list(data_grp)}")


def _attr(obj, *names, default=None):
    for n in names:
        if n in obj.attrs:
            v = obj.attrs[n]
            if isinstance(v, (bytes, np.bytes_)):
                v = v.decode("utf-8", "replace")
            return v
    return default


def parse_structmetadata(h5):
    """Grid corners from HDF-EOS StructMetadata.0 -> geotransform.

    Returns dict with a GDAL 6-tuple (ulx, xres, 0, uly, 0, yres), or None.
    """
    key = None
    for k in ("/HDFEOS INFORMATION/StructMetadata.0",
              "HDFEOS INFORMATION/StructMetadata.0"):
        if k in h5:
            key = k
            break
    if key is None:
        return None
    raw = h5[key][()]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8", "replace")
    txt = str(raw)

    def num_pair(tag):
        m = re.search(tag + r"\s*=\s*\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", txt)
        return (float(m.group(1)), float(m.group(2))) if m else None

    def num(tag):
        m = re.search(tag + r"\s*=\s*([-\d]+)", txt)
        return int(m.group(1)) if m else None

    ul, lr = num_pair("UpperLeftPointMtrs"), num_pair("LowerRightMtrs")
    xdim, ydim = num("XDim"), num("YDim")
    if not (ul and lr and xdim and ydim):
        return None
    xres = (lr[0] - ul[0]) / xdim
    yres = (lr[1] - ul[1]) / ydim
    return {"geotransform": (ul[0], xres, 0.0, ul[1], 0.0, yres),
            "xdim": xdim, "ydim": ydim,
            "upper_left": ul, "lower_right": lr}


def geotransform_from_udm(meta, asset_key="ortho_beta_udm", cache_dir="cache"):
    """Fallback: take the transform from the same-grid UDM COG (needs rasterio)."""
    import rasterio
    href = meta["assets"][asset_key]["href"]
    path = download(href, cache_dir)
    with rasterio.open(path) as ds:
        return {"geotransform": ds.transform.to_gdal(), "crs": str(ds.crs),
                "xdim": ds.width, "ydim": ds.height}


def _slice(window):
    if window is None:
        return None
    r0, r1, c0, c1 = window
    return (slice(r0, r1), slice(c0, c1))


def _wshape(sl, rows, cols):
    if sl is None:
        return (rows, cols)
    return (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)


# --------------------------------------------------------------------------
# main reader
# --------------------------------------------------------------------------
class Scene:
    """Lazy handle on a Tanager HDF5. Use as a context manager."""

    def __init__(self, path):
        if h5py is None:
            raise ImportError("h5py required: conda install -c conda-forge h5py")
        self.path = path
        self._h5 = h5py.File(path, "r")
        self.data, self.grid = _data_group(self._h5)
        self.cube_name, self._cube = _cube_dataset(self.data)
        self.n_bands, self.rows, self.cols = self._cube.shape

        wl = _attr(self._cube, "wavelengths", "center_wavelengths")
        self.wl_nm = (np.asarray(wl, float).ravel() if wl is not None
                      else np.full(self.n_bands, np.nan))
        if np.isfinite(self.wl_nm).all() and np.nanmax(self.wl_nm) < 10:
            self.wl_nm = self.wl_nm * 1000.0        # micrometres -> nm
        fw = _attr(self._cube, "fwhm")
        self.fwhm_nm = np.asarray(fw, float).ravel() if fw is not None else None
        gw = _attr(self._cube, "good_wavelengths")
        self.good = (np.asarray(gw).ravel().astype(bool) if gw is not None
                     else np.ones(self.n_bands, bool))
        self.fill = float(_attr(self._cube, "_FillValue", default=FILL))
        self.is_reflectance = "reflect" in self.cube_name
        self.epsg = _attr(self.grid, "epsg_code")
        self.strip_id = _attr(self.grid, "strip_id")
        self.framing = parse_structmetadata(self._h5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._h5.close()
        except Exception:
            pass

    def __repr__(self):
        kind = "SR" if self.is_reflectance else "TOA"
        return (f"<Scene {kind} {self.cube_name} {self.n_bands}x{self.rows}x{self.cols} "
                f"epsg={self.epsg} good={int(self.good.sum())}/{self.n_bands}>")

    # -- fields ------------------------------------------------------------
    def has(self, name):
        return name in self.data

    def fields(self):
        return sorted(self.data.keys())

    def plane(self, name, window=None):
        """2-D auxiliary field (aod, cwv, sun_zenith, ...) with fill -> nan."""
        if name not in self.data:
            return None
        d = self.data[name]
        sl = _slice(window)
        a = np.asarray(d[sl] if sl else d[:], float)
        fv = float(_attr(d, "_FillValue", default=FILL))
        return np.where(a == fv, np.nan, a)

    def mask(self, name, window=None):
        """uint8 mask as stored (255 = fill)."""
        if name not in self.data:
            return None
        d = self.data[name]
        sl = _slice(window)
        return np.asarray(d[sl] if sl else d[:])

    # -- cube --------------------------------------------------------------
    def read_cube(self, bands=None, window=None, good_only=False, dataset=None):
        """(n_sel_bands, rows, cols) with fill -> nan.

        bands    : indices or boolean mask over bands (None = all)
        window   : (r0, r1, c0, c1); None = full scene (1.6 GB!)
        good_only: intersect with good_wavelengths
        dataset  : e.g. 'surface_reflectance_uncertainty'
        """
        d = self.data[dataset] if dataset else self._cube
        idx = np.arange(self.n_bands)
        if bands is not None:
            b = np.asarray(bands)
            idx = np.where(b)[0] if b.dtype == bool else b
        if good_only:
            idx = idx[self.good[idx]]
        sl = _slice(window)
        out = np.empty((len(idx), *_wshape(sl, self.rows, self.cols)), np.float32)
        for i, b in enumerate(idx):          # band-at-a-time = chunk friendly
            out[i] = d[int(b)][sl] if sl else d[int(b)]
        fv = float(_attr(d, "_FillValue", default=FILL))
        out[out == fv] = np.nan
        return out, idx

    def valid_mask(self, window=None):
        """Boolean (rows, cols): not nodata / cloud / cirrus."""
        sl = _slice(window)
        keep = np.ones(_wshape(sl, self.rows, self.cols), bool)
        for m in ("nodata_pixels", "beta_cloud_mask", "beta_cirrus_mask"):
            a = self.mask(m, window)
            if a is not None:
                keep &= (a == 0)
        return keep

    def read_labeled_pixels(self, rows, cols, good_only=True, dataset=None):
        """Spectra at scattered (row, col) points; reads only their bbox.

        Returns (X (n_points, n_sel_bands), band_idx).
        """
        rows = np.asarray(rows, int)
        cols = np.asarray(cols, int)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        cube, idx = self.read_cube(window=(r0, r1, c0, c1),
                                   good_only=good_only, dataset=dataset)
        X = cube[:, rows - r0, cols - c0].T
        return X, idx


def open_scene(path):
    """Convenience: returns a Scene (usable as a context manager)."""
    return Scene(path)


def to_pixel_table(cube, valid=None):
    """(bands, rows, cols) -> ((n_pix, bands), (n_pix, 2) row/col, keep mask)."""
    b, r, c = cube.shape
    X = cube.reshape(b, r * c).T
    rr, cc = np.meshgrid(np.arange(r), np.arange(c), indexing="ij")
    idx = np.stack([rr.ravel(), cc.ravel()], 1)
    keep = np.isfinite(X).all(1)
    if valid is not None:
        keep &= valid.reshape(-1)
    return X[keep], idx[keep], keep
