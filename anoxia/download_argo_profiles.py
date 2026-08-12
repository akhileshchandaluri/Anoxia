"""
Download BGC-Argo profiles (Indian Ocean) and extract DOXY near 50 m.

Primary method:
  - ERDDAP via erddapy (tries NOAA CoastWatch first, then IFREMER fallback)

Output:
  - ./data/argo/argo_profiles.json
  - list of dicts:
      {"lat": float, "lon": float, "date": "YYYY-MM-DD", "do": float}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
try:
    from erddapy import ERDDAP
except Exception:
    ERDDAP = None  # type: ignore


SERVERS_DEFAULT = [
    "https://erddap.ifremer.fr/erddap",  # IFREMER has ArgoFloats datasets
    "https://coastwatch.pfeg.noaa.gov/erddap",  # NOAA fallback (may not have all datasets)
]

DATASET_CANDIDATES = [
    "ArgoFloats-synthetic-BGC",
    "ArgoFloats",
    "ArgoFloats-reference",
]


def _rename_if_present(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handle common ERDDAP variants and column names with units.
    mapping = {
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "TIME": "time",
        "JULD": "time",
        "PRES": "pres",
        "DOXY": "doxy",
        "PLATFORM_NUMBER": "platform_number",
        "CYCLE_NUMBER": "cycle_number",
        # Handle columns with units (e.g., "longitude (degrees_east)")
        "latitude (degrees_north)": "latitude",
        "latitude (degree_north)": "latitude",
        "longitude (degrees_east)": "longitude",
        "longitude (degree_east)": "longitude",
        "time (UTC)": "time",
        "time (Utc)": "time",
        "time (utc)": "time",
        "pres (decibar)": "pres",
        "pres (dbar)": "pres",
        "doxy (micromole/kg)": "doxy",
        "doxy (umol/kg)": "doxy",
        "platform_number (no dimension)": "platform_number",
        "cycle_number (no dimension)": "cycle_number",
    }
    df = _rename_if_present(df, mapping)
    return df


def _try_query_dataset(
    *,
    server: str,
    dataset_id: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start_date: Optional[str],
    end_date: Optional[str],
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Try multiple variable and constraint name variants to handle schema differences.
    """
    var_sets = [
        ["longitude", "latitude", "time", "pres", "doxy", "platform_number", "cycle_number"],
        ["LONGITUDE", "LATITUDE", "TIME", "PRES", "DOXY", "PLATFORM_NUMBER", "CYCLE_NUMBER"],
        ["longitude", "latitude", "time", "pres", "doxy"],
        ["LONGITUDE", "LATITUDE", "TIME", "PRES", "DOXY"],
    ]

    constraint_sets = [
        ("longitude", "latitude", "time"),
        ("LONGITUDE", "LATITUDE", "TIME"),
    ]

    for lon_key, lat_key, time_key in constraint_sets:
        constraints = {
            f"{lon_key}>=": lon_min,
            f"{lon_key}<=": lon_max,
            f"{lat_key}>=": lat_min,
            f"{lat_key}<=": lat_max,
        }
        if start_date:
            constraints[f"{time_key}>="] = f"{start_date}T00:00:00Z"
        if end_date:
            constraints[f"{time_key}<="] = f"{end_date}T23:59:59Z"

        for variables in var_sets:
            try:
                e = ERDDAP(
                    server=server,
                    protocol="tabledap",
                    response="csv",
                )
                e.dataset_id = dataset_id  # Set dataset_id as attribute, not parameter
                e.constraints = constraints
                e.variables = variables
                df = e.to_pandas(parse_dates=True)
                if df is not None and not df.empty:
                    if verbose:
                        print(f"✓ Successfully queried {server} / {dataset_id}", file=sys.stderr)
                    return _standardize_columns(df)
            except Exception as ex:
                if verbose:
                    error_msg = str(ex)
                    print(f"  ✗ {server} / {dataset_id} with {variables}: {type(ex).__name__}: {error_msg[:100]}", file=sys.stderr)
                continue
    return None


def fetch_argo_dataframe(
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start_date: Optional[str],
    end_date: Optional[str],
    verbose: bool = False,
) -> pd.DataFrame:
    if ERDDAP is None:
        raise RuntimeError(
            "erddapy is required for this script.\nInstall with:\n  pip install erddapy"
        )
    
    # Try first with date constraints
    if verbose:
        print("Attempting query WITH date constraints...", file=sys.stderr)
    for server in SERVERS_DEFAULT:
        for dsid in DATASET_CANDIDATES:
            df = _try_query_dataset(
                server=server,
                dataset_id=dsid,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose,
            )
            if df is not None and not df.empty:
                return df
    
    # If that fails, try without date constraints for any available data
    if verbose:
        print("Attempting query WITHOUT date constraints...", file=sys.stderr)
    for server in SERVERS_DEFAULT:
        for dsid in DATASET_CANDIDATES:
            df = _try_query_dataset(
                server=server,
                dataset_id=dsid,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                start_date=None,
                end_date=None,
                verbose=verbose,
            )
            if df is not None and not df.empty:
                # Filter to requested date range if query succeeded
                if "time" in df.columns and (start_date or end_date):
                    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
                    if start_date:
                        df = df[df["time"] >= pd.Timestamp(start_date, tz="UTC")]
                    if end_date:
                        df = df[df["time"] <= pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)]
                    if df.empty:
                        continue  # No data in requested range
                return df
    
    raise RuntimeError(
        "Could not query Argo DOXY data from configured ERDDAP servers/datasets. "
        "Try adjusting date range or server settings."
    )


def build_profile_records(
    df: pd.DataFrame,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    target_depth_m: float = 50.0,
    depth_tolerance_m: float = 20.0,
) -> List[Dict[str, object]]:
    required = {"latitude", "longitude", "time", "pres", "doxy"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    work = df.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["pres"] = pd.to_numeric(work["pres"], errors="coerce")
    work["doxy"] = pd.to_numeric(work["doxy"], errors="coerce")
    work["time"] = pd.to_datetime(work["time"], errors="coerce", utc=True)

    work = work.dropna(subset=["latitude", "longitude", "pres", "doxy", "time"])
    work = work[
        (work["latitude"] >= lat_min)
        & (work["latitude"] <= lat_max)
        & (work["longitude"] >= lon_min)
        & (work["longitude"] <= lon_max)
    ]

    if work.empty:
        return []

    # Approximate depth using pressure (dbar ~ m in upper ocean).
    work["depth_diff"] = (work["pres"] - target_depth_m).abs()
    work = work[work["depth_diff"] <= depth_tolerance_m]
    if work.empty:
        return []

    profile_keys: List[str] = []
    if "platform_number" in work.columns:
        profile_keys.append("platform_number")
    if "cycle_number" in work.columns:
        profile_keys.append("cycle_number")
    profile_keys.append("time")

    idx = work.groupby(profile_keys)["depth_diff"].idxmin()
    best = work.loc[idx].copy()
    best = best.sort_values("time")

    out: List[Dict[str, object]] = []
    for r in best.itertuples(index=False):
        out.append(
            {
                "lat": float(r.latitude),
                "lon": float(r.longitude),
                "date": pd.Timestamp(r.time).strftime("%Y-%m-%d"),
                "do": float(r.doxy),
            }
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download Indian Ocean Argo DOXY profiles near 50 m.")
    p.add_argument("--lat-min", type=float, default=0.0)
    p.add_argument("--lat-max", type=float, default=30.0)
    p.add_argument("--lon-min", type=float, default=55.0)
    p.add_argument("--lon-max", type=float, default=100.0)
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date YYYY-MM-DD for ERDDAP query.",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date YYYY-MM-DD for ERDDAP query.",
    )
    p.add_argument("--target-depth", type=float, default=50.0, help="Target depth in meters (default: 50).")
    p.add_argument(
        "--depth-tolerance",
        type=float,
        default=20.0,
        help="Allowed |depth-target| in meters (default: 20).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./data/argo/argo_profiles.json"),
        help="Output JSON path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed query attempts.",
    )
    args = p.parse_args(argv)

    try:
        df = fetch_argo_dataframe(
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            start_date=args.start_date,
            end_date=args.end_date,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Failed to fetch Argo data: {e}", file=sys.stderr)
        return 1

    profiles = build_profile_records(
        df,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        target_depth_m=args.target_depth,
        depth_tolerance_m=args.depth_tolerance,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    print(f"Downloaded {len(profiles)} Argo profiles with DOXY measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

