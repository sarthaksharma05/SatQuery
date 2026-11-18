"""Build real-satellite SatQuery demo files from Sentinel-2 imagery you download yourself.

WHY THIS SCRIPT EXISTS
-----------------------
SatQuery's bundled demo_data/ is intentionally SYNTHETIC (see demo_data/manifest.json's
own "note" field and satquery/demo/synth.py) -- it's generated with known ground truth so
the documented accuracy numbers can be checked automatically. It was never meant to be
shown as "real" imagery.

This script takes REAL Sentinel-2 bands you download from Copernicus Browser (free, no
card, ~5 minutes) and repackages them into the exact GeoTIFF shape SatQuery expects:
12 bands (B01..B12), correctly named/described, tagged with SATQUERY_MODALITY and
ACQUISITION_DATE, same CRS/grid for both dates so the bi-temporal pair validates cleanly.

HOW TO GET THE INPUT FILES (one-time, ~5 min)
-----------------------------------------------
1. Go to https://browser.dataspace.copernicus.eu  (free account, no card needed)
2. Search your area of interest (zoom/pan the map). Pick somewhere with visible change
   between two dates -- e.g. a growing urban edge, a reservoir, a construction site.
3. Pick a cloud-free date -> "Download" -> "Analytical" -> select ALL bands, format
   GeoTIFF, resolution 10m (or "as acquired"). This gives you one .tif per band
   (B01.tif ... B12.tif) or sometimes a single stacked file depending on the UI version --
   both are handled below.
4. Repeat for a second date, same location, ~6-18 months apart.
5. Put each date's bands in its own folder, e.g.:
     before/B01.tif ... before/B12.tif
     after/B01.tif  ... after/B12.tif

USAGE
-----
    python make_real_demo.py \
        --before-dir before --before-date 2024-06-01 \
        --after-dir  after  --after-date  2025-06-05 \
        --out-dir demo_data --prefix real_bitemporal

This writes:
    demo_data/real_bitemporal_before.tif
    demo_data/real_bitemporal_after.tif
and prints a manifest.json snippet to paste in (or add --update-manifest to do it for you).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

S2_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def _find_band_file(folder: Path, band: str) -> Path | None:
    for p in folder.glob(f"*{band}*.tif*"):
        return p
    for p in folder.glob(f"*{band.lower()}*.tif*"):
        return p
    return None


def _load_stack(folder: Path) -> tuple[np.ndarray, dict]:
    """Load 12 single-band files, or one already-stacked 12+ band file, from `folder`."""
    stacked = [p for p in folder.glob("*.tif*")]
    multi = next((p for p in stacked if rasterio.open(p).count >= 12), None)
    if multi is not None:
        with rasterio.open(multi) as ds:
            arr = ds.read(list(range(1, 13))).astype(np.uint16)
            profile = ds.profile.copy()
        return arr, profile

    bands = []
    ref_profile = None
    for b in S2_BAND_ORDER:
        f = _find_band_file(folder, b)
        if f is None:
            raise SystemExit(f"Could not find a file for band {b} in {folder} "
                              f"(expected a filename containing '{b}')")
        with rasterio.open(f) as ds:
            if ref_profile is None:
                ref_profile = ds.profile.copy()
                arr0 = ds.read(1)
                target_shape = arr0.shape
                bands.append(arr0)
            else:
                if ds.shape == (ref_profile["height"], ref_profile["width"]):
                    bands.append(ds.read(1))
                else:  # resample this band onto the reference grid (S2 bands are at 10/20/60m)
                    dst = np.empty(target_shape, dtype=ds.dtypes[0])
                    reproject(rasterio.band(ds, 1), dst, dst_transform=ref_profile["transform"],
                             dst_crs=ref_profile["crs"], resampling=Resampling.bilinear)
                    bands.append(dst)
    return np.stack(bands).astype(np.uint16), ref_profile


def build(folder: Path, date: str, out_path: Path) -> None:
    arr, profile = _load_stack(folder)
    profile.update(count=12, dtype="uint16", compress="deflate", tiled=False)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = tuple(S2_BAND_ORDER)
        dst.update_tags(ACQUISITION_DATE=date, SATQUERY_MODALITY="multispectral",
                         SOURCE="Real Sentinel-2 L2A, downloaded via Copernicus Browser")
    print(f"wrote {out_path}  ({arr.shape[1]}x{arr.shape[2]}, 12 bands, {profile['crs']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-dir", required=True, type=Path)
    ap.add_argument("--before-date", required=True)
    ap.add_argument("--after-dir", required=True, type=Path)
    ap.add_argument("--after-date", required=True)
    ap.add_argument("--out-dir", default=Path("demo_data"), type=Path)
    ap.add_argument("--prefix", default="bitemporal_real")
    ap.add_argument("--update-manifest", action="store_true",
                    help="append a scenario entry to demo_data/manifest.json")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.out_dir / f"{args.prefix}_before.tif"
    after_path = args.out_dir / f"{args.prefix}_after.tif"
    build(args.before_dir, args.before_date, before_path)
    build(args.after_dir, args.after_date, after_path)

    entry = {
        "id": args.prefix,
        "title": f"Real Sentinel-2 bi-temporal pair ({args.before_date} -> {args.after_date})",
        "files": [before_path.name, after_path.name],
        "queries": [
            "What changed between these two dates, and where did the change occur?",
            "Has the built-up area increased, decreased, or remained unchanged?",
            "How much of the area changed?",
        ],
    }
    manifest_path = args.out_dir / "manifest.json"
    if args.update_manifest and manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        m["scenarios"] = [s for s in m["scenarios"] if s["id"] != args.prefix] + [entry]
        manifest_path.write_text(json.dumps(m, indent=2))
        print(f"updated {manifest_path} with scenario '{args.prefix}'")
    else:
        print("\nAdd this to demo_data/manifest.json's \"scenarios\" list "
              "(or rerun with --update-manifest):\n")
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
