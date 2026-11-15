"""
SMAP L4 soil moisture for a coordinate, requested by API (no local file needed).

Same calling style as the teammate's NASApowerAPI.py / PRISMAPI.py:
validated inputs -> one function call -> a tidy DataFrame
(Timestamp, Latitude, Longitude, sm_surface, sm_rootzone), timestamps in UTC.

Unlike POWER/PRISM, SMAP has no keyless JSON endpoint, so the "request" goes to
NASA Earthdata (search the catalogue, then read just this pixel of each granule over
HTTPS). It needs a free Earthdata login (saved once in ~/_netrc by the first run).
Results are cached in data/smap_cache.csv, shared with smap_timeseries.py.

    from smap_lookup import fetch_smap_data, get_soil_moisture
    df = fetch_smap_data(40.7, -74.0, "2026-11-01", "2026-11-07", interval=6)
    get_soil_moisture(40.7, -74.0)                       # latest-ish snapshot -> dict

    python smap_lookup.py 40.7 -74.0                     # one snapshot
    python smap_lookup.py 40.7 -74.0 --start 2026-11-01 --end 2026-11-07 --interval 6
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# EASE-Grid 2.0 global cylindrical equal-area projection (EPSG:6933), used by SMAP L4.
_R = 6378137.0                      # WGS84 semi-major axis (m)
_E2 = 0.00669437999014              # WGS84 eccentricity squared
_E = np.sqrt(_E2)
_LAT_TS = np.deg2rad(30.0)          # standard parallel


def _q(sin_phi):
    return (1 - _E2) * (sin_phi / (1 - _E2 * sin_phi**2)
                        - (1 / (2 * _E)) * np.log((1 - _E * sin_phi) / (1 + _E * sin_phi)))


_K0 = np.cos(_LAT_TS) / np.sqrt(1 - _E2 * np.sin(_LAT_TS) ** 2)


def _lonlat_to_xy(lon, lat):
    """lon/lat in degrees -> EASE-Grid 2.0 x/y in meters."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x = _R * _K0 * np.deg2rad(lon)
    y = _R * _q(np.sin(np.deg2rad(lat))) / (2 * _K0)
    return x, y


VALID_INTERVALS = [3, 6, 12, 24]    # hours between observations (SMAP L4 is 3-hourly)
MAX_REQUESTS = 400                   # safety limit: each timestamp is one remote file read


def _slot_times(start, end, interval):
    """UTC granule-centre times (01:30, 04:30, ..., 22:30) from start..end, every `interval` hours."""
    step = interval // 3
    out = []
    day = pd.Timestamp(start).normalize()
    while day <= pd.Timestamp(end):
        for k in range(8):
            if k % step == 0:
                out.append((day + pd.Timedelta(hours=1.5 + 3 * k)).tz_localize("UTC"))
        day += pd.Timedelta(days=1)
    return out


def fetch_smap_data(latitude, longitude, start_date, end_date, interval=3,
                    workers=8, use_cache=True, max_requests=MAX_REQUESTS):
    """
    Retrieve SMAP L4 surface and root-zone soil moisture (m3/m3) for one location.

    interval : hours between observations, one of 3, 6, 12, 24 (keeps every 1st, 2nd,
               4th or 8th 3-hour granule of each day, starting at 01:30 UTC).
    Weeks/days that are not published yet, and pixels without data (ocean, ice), give NaN.
    """
    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")
    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid date.")
    if start > end:
        raise ValueError("Invalid date range.")
    if start < pd.Timestamp("2015-03-31"):
        raise ValueError("SMAP L4 data starts on 2015-03-31.")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid sampling interval; choose one of {VALID_INTERVALS}.")

    times = _slot_times(start, end, interval)
    if len(times) > max_requests:
        raise ValueError(f"{len(times)} observations requested (limit {max_requests}); "
                         "shorten the date range or use a larger interval.")

    import smap_timeseries as sts    # imported here: smap_timeseries imports the projection above
    df = sts.fetch_times(times, latitude, longitude, workers, use_cache)

    out = pd.DataFrame({
        "Timestamp": df["time"].dt.tz_convert("UTC").dt.tz_localize(None),
        "Latitude": latitude,
        "Longitude": longitude,
        "sm_surface": df["sm_surface"],
        "sm_rootzone": df["sm_rootzone"],
    })
    return out.reset_index(drop=True)


def get_soil_moisture(lat, lon, when=None):
    """
    One snapshot -> {'time', 'sm_surface', 'sm_rootzone'}.
    when: UTC datetime/'YYYY-MM-DD HH:MM'; default = the newest 3-hour slot that is
    normally published (about 4 days ago).
    """
    t = pd.Timestamp(when) if when is not None else pd.Timestamp(
        datetime.now(timezone.utc) - timedelta(days=4)).tz_localize(None)
    t = t.tz_localize(None) if t.tzinfo else t
    day = t.normalize()
    cand = _slot_times(day, day, 3)
    slot = min(cand, key=lambda c: abs(c.tz_localize(None) - t))     # nearest 3-hour slot
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Invalid latitude/longitude.")
    import smap_timeseries as sts
    row = sts.fetch_times([slot], lat, lon).iloc[0]          # exactly one granule
    return {"time": slot.tz_localize(None), "sm_surface": float(row["sm_surface"]),
            "sm_rootzone": float(row["sm_rootzone"])}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default: single snapshot)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default = start)")
    p.add_argument("--interval", type=int, default=3, help=f"hours between observations {VALID_INTERVALS}")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    try:
        if a.start is None:
            r = get_soil_moisture(a.lat, a.lon)
            print(f"Location: {a.lat}, {a.lon}   time: {r['time']} UTC")
            print(f"Surface soil moisture:   {r['sm_surface']:.4f} m3/m3")
            print(f"Root-zone soil moisture: {r['sm_rootzone']:.4f} m3/m3")
        else:
            df = fetch_smap_data(a.lat, a.lon, a.start, a.end or a.start, a.interval, a.workers)
            print(df.to_string(index=False))
            if a.csv:
                df.to_csv(a.csv, index=False)
                print(f"saved {a.csv}")
    except EOFError:
        raise SystemExit("LOGIN NEEDED: run `python smap_timeseries.py 40.7 -74.0 3` once in a "
                         "terminal to save your Earthdata login.")
