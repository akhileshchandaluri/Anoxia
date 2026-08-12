"""
Download ~2 years of MODIS Aqua 8-day chlorophyll composites (L3, 4km).

Defaults:
  - bbox: lat 0..30N, lon 55..100E
  - time window: last 730 days (inclusive-ish)
  - target: MODISA_L3m_CHL collection, version 2022.0

Auth:
  Uses EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables via
  earthaccess' "environment" login strategy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import earthaccess


def _parse_yyyy_mm_dd(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}', expected YYYY-MM-DD") from e


def _ensure_env_creds_present() -> None:
    # earthaccess will error later, but this makes failures clearer.
    missing = [k for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD") if not os.getenv(k)]
    if missing:
        raise SystemExit(
            "Missing required env var(s): "
            + ", ".join(missing)
            + "\nSet them before running, e.g. (PowerShell):"
            + "\n  $env:EARTHDATA_USERNAME='...'; $env:EARTHDATA_PASSWORD='...'"
        )


def _granule_id(g) -> str:
    # DataGranule behaves like a mapping; GranuleUR is usually the most readable ID.
    try:
        umm = g.get("umm", {})  # type: ignore[attr-defined]
        return umm.get("GranuleUR") or str(g)
    except Exception:
        return str(g)


def _filter_8day_4km_chl(granules: Iterable) -> List:
    """
    Keep only L3m 8-day chlorophyll-a 4km granules.

    Typical GranuleUR patterns include:
      AQUA_MODIS.<start>_<end>.L3m.8D.CHL.chlor_a.4km.nc
    """
    out = []
    for g in granules:
        gid = _granule_id(g)
        s = gid.lower()
        if ".l3m." not in s:
            continue
        if ".8d." not in s:
            continue
        if ".chl." not in s:
            continue
        if ".chlor_a." not in s:
            continue
        if ".4km" not in s:
            continue
        out.append(g)
    return out


def search_granules(
    *,
    start_date: dt.date,
    end_date: dt.date,
    bbox: Tuple[float, float, float, float],
    short_name: str = "MODISA_L3m_CHL",
    version: str = "2022.0",
    provider: str = "OB_CLOUD",
    count: int = 500,
) -> List:
    temporal = (start_date.isoformat(), end_date.isoformat())
    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        provider=provider,
        temporal=temporal,
        bounding_box=bbox,
        count=count,
    )
    return list(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Download MODIS Aqua L3 8-day 4km chlorophyll composites using earthaccess."
        )
    )
    p.add_argument(
        "--start",
        type=_parse_yyyy_mm_dd,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 730 days ago (UTC).",
    )
    p.add_argument(
        "--end",
        type=_parse_yyyy_mm_dd,
        default=None,
        help="End date (YYYY-MM-DD). Default: today (UTC).",
    )
    p.add_argument("--lon-min", type=float, default=55.0)
    p.add_argument("--lat-min", type=float, default=0.0)
    p.add_argument("--lon-max", type=float, default=100.0)
    p.add_argument("--lat-max", type=float, default=30.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./data/modis"),
        help="Output directory (default: ./data/modis).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matched granules, do not download.",
    )
    args = p.parse_args(argv)

    _ensure_env_creds_present()

    end = args.end or dt.datetime.utcnow().date()
    start = args.start or (end - dt.timedelta(days=730))
    if start > end:
        print("Error: --start is after --end", file=sys.stderr)
        return 2

    bbox = (args.lon_min, args.lat_min, args.lon_max, args.lat_max)

    auth = earthaccess.login(strategy="environment")
    try:
        authenticated = bool(getattr(auth, "authenticated", True))
    except Exception:
        authenticated = True
    if not authenticated:
        print("Earthaccess login failed (check env vars).", file=sys.stderr)
        return 2

    granules = search_granules(start_date=start, end_date=end, bbox=bbox)
    granules = _filter_8day_4km_chl(granules)

    if not granules:
        print(
            "No matching granules found.\n"
            "Notes:\n"
            "  - This script targets collection short_name=MODISA_L3m_CHL, version=2022.0\n"
            "  - Filtering expects GranuleUR containing: L3m.8D.CHL.chlor_a.4km\n"
            "  - Try widening dates or removing the filter logic if naming differs for your region/time.\n",
            file=sys.stderr,
        )
        return 1

    # Sort for stable output (best-effort).
    granules_sorted = sorted(granules, key=_granule_id)

    print(f"Matched {len(granules_sorted)} granules:")
    for g in granules_sorted:
        print(f"  {_granule_id(g)}")

    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    downloaded = earthaccess.download(granules_sorted, local_path=str(args.out))

    # earthaccess.download returns local file paths (strings/Paths) for successes.
    n = 0
    if downloaded is None:
        n = 0
    else:
        try:
            n = len(downloaded)  # type: ignore[arg-type]
        except TypeError:
            n = 0

    print(f"Downloaded {n} files to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

