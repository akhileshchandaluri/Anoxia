"""
Download 10 consecutive days of HYCOM GOFS 3.1 surface currents via OPeNDAP.

Source: HYCOM THREDDS (OPeNDAP)
  https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0

What it saves:
  - One NetCDF per day: ./data/hycom/hycom_YYYYMMDD.nc
  - Variables: water_u, water_v
  - Subset: lat 0..30N, lon 55..100E, depth index 0 (surface)
  - Time: all HYCOM timesteps that fall within each day (typically 3-hourly)

Robustness:
  - If a network request times out (>30s), retry up to 3 times with 5s wait.

Validation:
  - Loads one day's u/v slice and constructs RegularGridInterpolator-friendly arrays;
    prints u/v grid shapes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, TypeVar

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


T = TypeVar("T")


# HYCOM GOFS 3.1 datasets - try multiple URLs as they change over time
OPENDAP_URLS = [
    "https://tds.hycom.org/thredds/dodsC/GOFS3.1/latest",  # Latest operational
    "https://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_93.1",  # Alternative high-res
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0",  # Original default
]

OPENDAP_URL_DEFAULT = OPENDAP_URLS[0]


def _parse_yyyy_mm_dd(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}', expected YYYY-MM-DD") from e


def _is_timeout_exc(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(
        tok in msg
        for tok in (
            "timed out",
            "timeout",
            "operation timed out",
            "curl error",
            "i/o failure",
            "errno -68",
            "could not connect",
            "connection",
        )
    )


def _with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    wait_seconds: float = 5.0,
) -> T:
    last: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except BaseException as e:
            last = e
            if attempt >= max_retries or not _is_timeout_exc(e):
                raise
            print(f"Timeout on attempt {attempt}/{max_retries}; retrying in {wait_seconds:.0f}s...")
            time.sleep(wait_seconds)
    assert last is not None
    raise last


def _set_netcdf4_timeout(seconds: int = 30) -> None:
    """
    netCDF4-python uses libcurl underneath for OPeNDAP. If available, set a
    default timeout so xarray opens/loads won't hang indefinitely.
    """
    try:
        import netCDF4  # type: ignore

        if hasattr(netCDF4, "set_default_timeout"):
            netCDF4.set_default_timeout(seconds)  # type: ignore[attr-defined]
    except Exception:
        # If netCDF4 isn't installed or doesn't support this, we still rely on retries.
        pass


def _requests_session():
    """
    Create a requests Session suitable for OPeNDAP (pydap backend).
    """
    import requests

    s = requests.Session()
    # Keep defaults; users may be behind proxies, etc.
    return s


def _find_working_url(urls: list, *, backend: str = "pydap") -> Optional[str]:
    """
    Try multiple URLs and return the first one that responds.
    """
    for url in urls:
        try:
            print(f"Testing URL: {url}", file=sys.stderr)
            ds = _open_hycom(url, backend=backend)
            print(f"✓ URL works: {url}", file=sys.stderr)
            ds.close()
            return url
        except Exception as e:
            print(f"✗ URL failed: {str(e)[:80]}", file=sys.stderr)
            continue
    return None


def _open_hycom(url: str, *, backend: str) -> xr.Dataset:
    """
    Open the remote HYCOM dataset. We keep it lazy and only load subsets.
    """
    _set_netcdf4_timeout(30)

    def _open_one(u: str) -> xr.Dataset:
        # HYCOM provides a problematic "tau" variable with units "hours since analysis"
        # which breaks xarray's automatic CF time decoding. We disable global decoding
        # and manually decode only the real "time" coordinate.
        if backend == "pydap":
            try:
                import pydap  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "pydap is required for reliable OPeNDAP timeouts on Windows.\n"
                    "Install it with:\n"
                    "  pip install pydap requests\n"
                ) from e

            session = _requests_session()
            ds = xr.open_dataset(
                u,
                engine="pydap",
                decode_times=False,
                mask_and_scale=True,
                backend_kwargs={
                    "session": session,
                    "timeout": 30,
                },
            )
        elif backend == "netcdf4":
            ds = xr.open_dataset(
                u,
                decode_times=False,
                mask_and_scale=True,
            )
        else:
            raise ValueError("--backend must be 'pydap' or 'netcdf4'")
        if "time" in ds.coords or "time" in ds.variables:
            try:
                units = ds["time"].attrs.get("units", "")
                # Expected: "hours since 2000-01-01 00:00:00"
                if isinstance(units, str) and units.lower().startswith("hours since "):
                    ref = units.split("since", 1)[1].strip()
                    ref_dt = np.datetime64(dt.datetime.fromisoformat(ref))
                    hours = np.asarray(ds["time"].values, dtype="float64")
                    seconds = np.rint(hours * 3600.0).astype("int64")
                    decoded = ref_dt + seconds.astype("timedelta64[s]")
                    ds = ds.assign_coords(time=decoded)
            except Exception:
                # If decoding fails, keep numeric time; downstream selection will fail loudly.
                pass
        return ds

    # Some Windows netCDF4/libcurl builds fail on https for OPeNDAP; HYCOM serves http too.
    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url.removeprefix("https://"))

    last: Optional[BaseException] = None
    for u in candidates:
        try:
            return _with_retries(lambda: _open_one(u))
        except BaseException as e:
            last = e
            continue
    assert last is not None
    raise last


def _subset_day(
    ds: xr.Dataset,
    *,
    day: dt.date,
    bbox: Tuple[float, float, float, float],
) -> xr.Dataset:
    lon_min, lat_min, lon_max, lat_max = bbox

    day_start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
    day_end = day_start + dt.timedelta(days=1) - dt.timedelta(microseconds=1)

    # Subset using bounds; try different approaches depending on what works
    try:
        # First try with simple slice (no method argument)
        subset = ds[["water_u", "water_v"]].sel(
            time=slice(day_start, day_end),
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )
    except Exception as e:
        # If that fails, try with index-based slicing
        try:
            # Get all time points for the day
            subset = ds[["water_u", "water_v"]].sel(time=slice(day_start, day_end))
            
            # Do spatial subsetting with isel (index-based) instead of sel (label-based)
            if "lat" in ds.coords and "lon" in ds.coords:
                lat_vals = ds.coords["lat"].values
                lon_vals = ds.coords["lon"].values
                lat_idx = np.where((lat_vals >= lat_min) & (lat_vals <= lat_max))[0]
                lon_idx = np.where((lon_vals >= lon_min) & (lon_vals <= lon_max))[0]
                
                if len(lat_idx) > 0 and len(lon_idx) > 0:
                    subset = subset.isel(lat=slice(lat_idx[0], lat_idx[-1] + 1),
                                         lon=slice(lon_idx[0], lon_idx[-1] + 1))
                else:
                    raise ValueError(f"No spatial data found in bbox: {bbox}")
            else:
                raise ValueError("Dataset does not have 'lat' and 'lon' coordinates")
        except Exception as e2:
            raise RuntimeError(f"Failed to subset data: {e2}")

    # Depth index 0 is surface.
    if "depth" in subset.dims:
        subset = subset.isel(depth=0)
    elif "depth" in subset.coords:
        subset = subset.isel(depth=0)

    return subset


def _write_daily_nc(ds_day: xr.Dataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure CF-ish output (xarray will keep coords/attrs).
    encoding = {}
    for v in ("water_u", "water_v"):
        if v in ds_day:
            encoding[v] = {"zlib": True, "complevel": 4}

    def _write() -> None:
        # Prefer netcdf4 engine if available; fall back to scipy.
        try:
            ds_day.to_netcdf(out_path, mode="w", engine="netcdf4", encoding=encoding)
        except Exception:
            ds_day.to_netcdf(out_path, mode="w", engine="scipy")

    _with_retries(_write)


def _validate_grids(ds_day: xr.Dataset) -> None:
    """
    Build interpolators for u and v on (lat, lon) for one timestep.
    This is just a sanity check that the grids are regular and load correctly.
    """
    if "time" not in ds_day.dims or ds_day.sizes.get("time", 0) == 0:
        raise RuntimeError("No timesteps found in the selected day; cannot validate grids.")

    lat = ds_day["lat"].values
    lon = ds_day["lon"].values

    u2 = ds_day["water_u"].isel(time=0).values
    v2 = ds_day["water_v"].isel(time=0).values

    print(f"u grid shape (time0): {u2.shape}  (lat={lat.shape}, lon={lon.shape})")
    print(f"v grid shape (time0): {v2.shape}  (lat={lat.shape}, lon={lon.shape})")

    # RegularGridInterpolator expects monotonic 1D points and an N-D values array
    # ordered as (lat, lon) here.
    _ = RegularGridInterpolator((lat, lon), u2, bounds_error=False, fill_value=np.nan)
    _ = RegularGridInterpolator((lat, lon), v2, bounds_error=False, fill_value=np.nan)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Download 10 days of HYCOM GOFS 3.1 surface currents via OPeNDAP."
    )
    p.add_argument("--url", default=OPENDAP_URL_DEFAULT, help="OPeNDAP dataset URL.")
    p.add_argument(
        "--backend",
        choices=("pydap", "netcdf4"),
        default="pydap",
        help="xarray backend for OPeNDAP (default: pydap).",
    )
    p.add_argument(
        "--start",
        type=_parse_yyyy_mm_dd,
        default=None,
        help="First day to download (YYYY-MM-DD). Default: 9 days ago (UTC).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of consecutive days to download (default: 10).",
    )
    p.add_argument("--lon-min", type=float, default=55.0)
    p.add_argument("--lat-min", type=float, default=0.0)
    p.add_argument("--lon-max", type=float, default=100.0)
    p.add_argument("--lat-max", type=float, default=30.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./data/hycom"),
        help="Output directory (default: ./data/hycom).",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write files.")
    args = p.parse_args(argv)

    if args.days <= 0:
        raise SystemExit("--days must be > 0")

    bbox = (args.lon_min, args.lat_min, args.lon_max, args.lat_max)

    # Detect if URL is from the list or custom; try alternatives if needed
    urls_to_try = [args.url]
    if args.url == OPENDAP_URL_DEFAULT:
        urls_to_try = OPENDAP_URLS
    
    print(f"Connecting to HYCOM OPeNDAP server...", file=sys.stderr)
    working_url = _find_working_url(urls_to_try, backend=args.backend)
    
    if working_url is None:
        print(f"ERROR: Could not connect to any HYCOM server:", file=sys.stderr)
        for url in urls_to_try:
            print(f"  - {url}", file=sys.stderr)
        print(f"Please check your internet connection or try again later.", file=sys.stderr)
        return 1
    
    print(f"", file=sys.stderr)

    ds = _open_hycom(working_url, backend=args.backend)

    if args.start is None:
        # Choose the latest available day in the dataset (HYCOM expt_93.0 is not "today").
        try:
            last_time = np.asarray(ds["time"].values).ravel()[-1]
            end = dt.date.fromisoformat(str(last_time)[:10])
        except Exception:
            end = dt.datetime.utcnow().date()
        start = end - dt.timedelta(days=args.days - 1)
    else:
        start = args.start
        end = start + dt.timedelta(days=args.days - 1)

    written = 0
    validated = False
    for i in range(args.days):
        day = start + dt.timedelta(days=i)
        out_path = args.out / f"hycom_{day:%Y%m%d}.nc"

        ds_day = _subset_day(ds, day=day, bbox=bbox)

        # Force actual network read for just this day's subset.
        def _load() -> xr.Dataset:
            try:
                return ds_day.load()
            except Exception as e:
                # OPeNDAP servers sometimes have issues with constraints; print diagnostic info
                print(f"Warning: Full data load failed, trying selective variable load...", file=sys.stderr)
                if "water_u" in ds_day.data_vars:
                    u = ds_day["water_u"].load()
                if "water_v" in ds_day.data_vars:
                    v = ds_day["water_v"].load()
                # Re-try the full load
                return ds_day.load()

        try:
            ds_day_loaded = _with_retries(_load)
        except Exception as e:
            print(f"✗ Failed to load data for {day:%Y-%m-%d}: {str(e)[:120]}", file=sys.stderr)
            print(f"  Skipping this day...", file=sys.stderr)
            continue

        if not validated and ds_day_loaded.sizes.get("time", 0) > 0:
            _validate_grids(ds_day_loaded)
            validated = True
        elif ds_day_loaded.sizes.get("time", 0) == 0:
            raise RuntimeError(
                f"No timesteps found for {day:%Y-%m-%d}. "
                "Pick a start date within the dataset coverage (or omit --start to use latest available)."
            )

        if args.dry_run:
            print(f"[dry-run] Would write {out_path}")
        else:
            _write_daily_nc(ds_day_loaded, out_path)
            written += 1

    print(f"Downloaded date range: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    print(f"File count: {written}" if not args.dry_run else f"File count: {args.days} (dry-run)")
    if written > 0 and not args.dry_run:
        print(f"Data directory: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

