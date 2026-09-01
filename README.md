# tanager-ice

Tanager hyperspectral snow/ice characterisation with conformal per-pixel uncertainty.
Submission work for the Planet Tanager Open Data Competition (due 2026-08-31).

## Companion website

A readable version of the memo, with every figure and a browsable view of this
pipeline, is published via GitHub Pages:

> **https://alecbiles-code.github.io/tanager-ice/**

The site source lives in [`docs/`](docs/) (`docs/index.html` is the memo,
`docs/code.html` is the code browser).

## Layout

    tanager_ice/     pip-installable core: spectral primitives, separability,
                     conformal uncertainty, Tanager HDF5 io
    scripts/         numbered analysis pipeline (01-26), run from the repo root.
                     Named entry points:
                       01_scene_recon.py        -> outputs/scene_meta.json, scene_footprint.geojson
                       02_coincidence_search.py -> outputs/coincidence_report.json
                     Scripts 03-26 cover HDF5 inspection, atmospheric-correction
                     diagnostics, segmentation, DEM/land masking, the grain-size and
                     melt retrievals, per-floe aggregation, the Sentinel-2 capability
                     gap, and the sensor-degradation experiment; make_hero_figures.py
                     builds the two hero PNGs.
    examples/        run_separability.py (Task 2 driver; --demo works with no data)
    tests/           test_primitives.py (14 known-answer checks, no network)
    outputs/         generated products (gitignored)
    cache/           downloaded scene assets (gitignored)
    notebooks/       (placeholder for exploratory notebooks)

    environmental_framing.md                          competition write-up
    hero_arctic_locator.png, hero_capability_gap.png  summary figures

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[io,search,model]"

## Run (from the repo root -- scripts write to ./outputs)

    python tests/test_primitives.py            # verify the maths: expect 14/14
    python examples/run_separability.py --demo # dry-run the Task 2 pipeline
    python scripts/01_scene_recon.py           # confirm footprint / EMIT gap / bands
    python scripts/02_coincidence_search.py    # needs NASA Earthdata creds for ICESat-2

## Notes

* The open Tanager product is **TOA radiance** (HDF5), not surface reflectance.
  Retrievals are relative/shape-based by design; see `tanager_ice/spectral.py`.
* `tanager_ice/io.py` is written to Planet's documented HDFEOS schema but has not
  been run against a live .h5 -- verify field names on first real run.
