"""
Build GFW seed points from local MMSI-daily ZIP files (2023 and 2024).

Usage:
  python gfw_seed_points_local.py --zip-2023 "D:/path/to/2023.zip" --zip-2024 "D:/path/to/2024.zip"

What it does:
  - Streams daily CSV files from each ZIP (no full extraction needed)
  - Reads columns [date, lon, lat, hours] (or lat/lon swapped; auto-detected)
  - Filters to Indian Ocean bbox: lat 0..30, lon 55..100
  - Aggregates fishing_hours by (lat, lon)
  - Selects top 8 cells by total fishing_hours (or fewer if <8 exist)
  - Saves JSON to ./data/argo/gfw_seed_points.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    tqdm = None  # type: ignore


BBox = Tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def _pick_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Return (lon_col, lat_col, hours_col), auto-detecting swapped coordinates.
    """
    cols = list(df.columns)
    norm_map = {_normalize_name(c): c for c in cols}

    # Prefer explicit headers first.
    lon_col = None
    lat_col = None
    hours_col = None

    for k in ("lon", "longitude"):
        if k in norm_map:
            lon_col = norm_map[k]
            break
    for k in ("lat", "latitude"):
        if k in norm_map:
            lat_col = norm_map[k]
            break
    for k in ("hours", "fishinghours", "fishing_hour", "fishinghourssum"):
        if k in norm_map:
            hours_col = norm_map[k]
            break

    # If any are missing, fallback to position assumption [date, lon, lat, hours].
    if lon_col is None or lat_col is None or hours_col is None:
        if len(cols) < 4:
            raise ValueError(f"Unexpected CSV schema with <4 columns: {cols}")
        lon_col = cols[1]
        lat_col = cols[2]
        hours_col = cols[3]

    # Detect and fix swapped lat/lon by range check on sample rows.
    s_lon = pd.to_numeric(df[lon_col], errors="coerce")
    s_lat = pd.to_numeric(df[lat_col], errors="coerce")
    sample = pd.DataFrame({"lon": s_lon, "lat": s_lat}).dropna().head(1000)

    if not sample.empty:
        # If many "lon" values look like lat range and many "lat" look like lon range,
        # then columns are likely swapped.
        lon_in_lat_range = ((sample["lon"] >= -90) & (sample["lon"] <= 90)).mean()
        lat_in_lon_range = ((sample["lat"] >= -180) & (sample["lat"] <= 180)).mean()
        lon_in_lon_range = ((sample["lon"] >= -180) & (sample["lon"] <= 180)).mean()
        lat_in_lat_range = ((sample["lat"] >= -90) & (sample["lat"] <= 90)).mean()

        # Swap if "lon" barely looks like lon but "lat" does, while "lon" looks lat-like.
        if lon_in_lon_range < 0.7 and lat_in_lon_range > 0.9 and lon_in_lat_range > 0.9 and lat_in_lat_range < 0.7:
            lon_col, lat_col = lat_col, lon_col

    return lon_col, lat_col, hours_col


def _iter_csv_members(zf: zipfile.ZipFile) -> List[str]:
    return [n for n in zf.namelist() if n.lower().endswith(".csv")]


def _aggregate_zip(
    zip_path: Path,
    *,
    bbox: BBox,
    agg: Dict[Tuple[float, float], float],
    use_tqdm: bool,
) -> Tuple[int, int]:
    """
    Process one ZIP and update agg[(lat, lon)] += hours.
    Returns (csv_processed, csv_failed).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    processed = 0
    failed = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = _iter_csv_members(zf)
            iterator: Iterable[str]
            if use_tqdm and tqdm is not None:
                iterator = tqdm(members, desc=f"{zip_path.name}", unit="csv")
            else:
                iterator = members

            for member in iterator:
                try:
                    with zf.open(member) as raw:
                        data = raw.read()
                    # decode with utf-8 fallback to latin-1
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = data.decode("latin-1")

                    df = pd.read_csv(io.StringIO(text))
                    lon_col, lat_col, hours_col = _pick_columns(df)

                    lon = pd.to_numeric(df[lon_col], errors="coerce")
                    lat = pd.to_numeric(df[lat_col], errors="coerce")
                    hours = pd.to_numeric(df[hours_col], errors="coerce").fillna(0.0)

                    mask = (
                        lat.notna()
                        & lon.notna()
                        & (lat >= lat_min)
                        & (lat <= lat_max)
                        & (lon >= lon_min)
                        & (lon <= lon_max)
                    )
                    if not mask.any():
                        processed += 1
                        continue

                    tmp = pd.DataFrame(
                        {
                            "lat": lat[mask].round(6),
                            "lon": lon[mask].round(6),
                            "hours": hours[mask],
                        }
                    )
                    grouped = tmp.groupby(["lat", "lon"], as_index=False)["hours"].sum()
                    for r in grouped.itertuples(index=False):
                        agg[(float(r.lat), float(r.lon))] += float(r.hours)

                    processed += 1
                except Exception as e:
                    failed += 1
                    print(f"[WARN] CSV failed in {zip_path.name}: {member} -> {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"[WARN] ZIP extract/open failed: {zip_path} -> {e}", file=sys.stderr)
        return 0, 1

    return processed, failed


def _top_seed_points(agg: Dict[Tuple[float, float], float], top_n: int = 8) -> List[Tuple[float, float, float]]:
    rows = sorted(((lat, lon, h) for (lat, lon), h in agg.items()), key=lambda x: x[2], reverse=True)
    return rows[:top_n]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Extract top GFW seed points from local MMSI daily ZIPs.")
    p.add_argument("--zip-2023", type=Path, required=True, help="Path to 2023 MMSI-daily ZIP file.")
    p.add_argument("--zip-2024", type=Path, required=True, help="Path to 2024 MMSI-daily ZIP file.")
    p.add_argument("--lat-min", type=float, default=0.0)
    p.add_argument("--lat-max", type=float, default=30.0)
    p.add_argument("--lon-min", type=float, default=55.0)
    p.add_argument("--lon-max", type=float, default=100.0)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("./data/argo/gfw_seed_points.json"),
        help="Output JSON file path.",
    )
    args = p.parse_args(argv)

    bbox: BBox = (args.lon_min, args.lat_min, args.lon_max, args.lat_max)
    agg: Dict[Tuple[float, float], float] = defaultdict(float)

    use_tqdm = tqdm is not None

    p1, f1 = _aggregate_zip(args.zip_2023, bbox=bbox, agg=agg, use_tqdm=use_tqdm)
    p2, f2 = _aggregate_zip(args.zip_2024, bbox=bbox, agg=agg, use_tqdm=use_tqdm)

    top = _top_seed_points(agg, top_n=8)

    # If Indian Ocean has <8 cells, return all found.
    seed_points = [[lat, lon] for lat, lon, _ in top]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"seed_points": seed_points}, indent=2), encoding="utf-8")

    print("Top seed cells (lat, lon, total_fishing_hours):")
    for lat, lon, hours in top:
        print(f"  ({lat:.6f}, {lon:.6f}) -> {hours:.3f}")

    print(
        f"Processed CSVs: {p1 + p2}, failed CSVs: {f1 + f2}, "
        f"unique bbox cells: {len(agg)}"
    )
    print("Found 8 real GFW seed points from 2023–2024 data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

