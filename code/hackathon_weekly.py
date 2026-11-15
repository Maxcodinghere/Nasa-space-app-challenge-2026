"""
NASAHackathon2026 data on the SAME weekly grid as the SMAP soil moisture:
one row per week, always the same weekday and UTC time (default 22:30), 520 weeks (10 years).

Uses that folder's own programs (not modified):
    APIs/NASApowerAPI.py  hourly NASA POWER (2-year chunks, POWER rejects longer requests)
    APIs/PRISMAPI.py      daily PRISM (ACIS grid 21)
    APIs/NOAAAPI.py       monthly NAO index
    APIs/ERA5API.py       hourly ERA5 (needs a CDS token; if none is set up it is skipped and the
                          ERA5 hours already in NASAHackathon2026/data/combined_dataset.csv are
                          used where they exist)
Whatever a source cannot provide stays empty (NaN): you get as much as exists inside the window.

How the hourly/daily data is lined up with a SMAP week (time T, e.g. Thursday 22:30 UTC):
  * hourly variables .. mean of the 3 hourly values 21:00, 22:00, 23:00 UTC, i.e. SMAP's own 3-hour
                         averaging window. (ERA5 precipitation and ERA5 solar J/m2 are hourly totals:
                         summed over the 3 hours. Wind direction is rebuilt from the averaged u/v.)
  * *_prev7d_mm ....... precipitation total of the 7 days BEFORE T (a 3-hour rain value is almost
                         always 0, so the weekly total is the useful rain feature).
  * PRISM_* ........... the daily value for the calendar date of T (PRISM's day ends 12:00 UTC, before T).
  * NAO_index ......... the monthly index of T's month, as in the folder's pipeline;
    NAO_index_prev_month the previous month (fully known at T, so it cannot leak the future).

    python hackathon_weekly.py 48.4187 -104.4876 520 --end 2026-11-05
    -> data/hackathon_weekly_48.4187_-104.4876.csv
"""
import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HACK = os.path.join(ROOT, "NASAHackathon2026")
sys.path.insert(0, HERE)
sys.path.insert(0, HACK)

DATA = os.path.join(ROOT, "data")
COMBINED = os.path.join(HACK, "data", "combined_dataset.csv")
POWER_PARAMS = ["T2M", "T2MDEW", "T2MWET", "RH2M", "QV2M", "PS", "WS10M", "WD10M", "U10M", "V10M",
                "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
PRISM_VARS = ["tmin", "tmax", "tmean", "ppt"]
POWER_MAX_DAYS = 730
ERA5_SUM = ["ERA5_total_precipitation_mm", "ERA5_solar_radiation_J_m2"]     # hourly totals -> summed


# ----------------------------------------------------------- alignment ---
def _offsets(t_naive):
    """The three hourly stamps of SMAP's 3-hour window centred on hh:30: (hh-1):00, hh:00, (hh+1):00."""
    base = t_naive.floor("h")
    return [base + pd.Timedelta(hours=h) for h in (-1, 0, 1)]


def window_values(hourly, times):
    """Hourly table (Timestamp + numeric columns) -> one row per SMAP time (naive UTC `times`).
    A value needs all three hours, otherwise it is NaN."""
    h = hourly.drop_duplicates("Timestamp").set_index("Timestamp").sort_index()
    h = h.select_dtypes("number").drop(columns=["Latitude", "Longitude", "latitude", "longitude"], errors="ignore")
    stamps = [pd.DatetimeIndex([o for o in _offsets(pd.Timestamp(t))]) for t in times]
    out = {}
    for col in h.columns:
        vals = np.array([h[col].reindex(s).to_numpy(float) for s in stamps])      # (n_times, 3)
        full = ~np.isnan(vals).any(axis=1)
        agg = np.full(len(times), np.nan)
        agg[full] = (vals[full].sum(axis=1) if col in ERA5_SUM else vals[full].mean(axis=1))
        out[col] = agg
    res = pd.DataFrame(out, index=pd.DatetimeIndex(times))
    # a mean of angles is wrong: rebuild the direction (wind comes FROM it, clockwise from north)
    for d, u, v in (("WD10M", "U10M", "V10M"), ("ERA5_wind_direction_10m_deg", "ERA5_u_wind_10m_m_s",
                                                "ERA5_v_wind_10m_m_s")):
        if {d, u, v} <= set(res.columns):
            res[d] = (180 + np.degrees(np.arctan2(res[u], res[v]))) % 360
    for sp, u, v in (("ERA5_wind_speed_10m_m_s", "ERA5_u_wind_10m_m_s", "ERA5_v_wind_10m_m_s"),):
        if {sp, u, v} <= set(res.columns):
            res[sp] = np.sqrt(res[u] ** 2 + res[v] ** 2)
    return res


def prev7d_total(hourly, col, times, min_frac=0.8):
    """Sum of an hourly series over the 7 days BEFORE each SMAP time (hours that end by hh:00 <= T)."""
    s = hourly.drop_duplicates("Timestamp").set_index("Timestamp").sort_index()[col]
    roll = s.rolling("168h", min_periods=int(168 * min_frac)).sum()
    idx = pd.DatetimeIndex([pd.Timestamp(t).floor("h") - pd.Timedelta(hours=1) for t in times])
    return roll.reindex(idx).to_numpy(float)


def align_weekly(times, power=None, prism=None, nao=None, era5=None):
    """
    times : tz-aware UTC SMAP times (one per week, same weekday and hour).  Each source is optional.
    Returns a DataFrame with `time` (tz-aware UTC, same as the soil-moisture table) + all columns.
    """
    t_naive = pd.DatetimeIndex([pd.Timestamp(t).tz_convert("UTC").tz_localize(None) for t in times])
    out = pd.DataFrame({"time": list(times)})
    parts = []
    if power is not None and len(power):
        w = window_values(power, t_naive)
        if "PRECTOTCORR" in power.columns:
            w["PRECTOTCORR_prev7d_mm"] = prev7d_total(power, "PRECTOTCORR", t_naive)
        parts.append(w)
    if era5 is not None and len(era5):
        w = window_values(era5, t_naive)
        if "ERA5_total_precipitation_mm" in era5.columns:
            w["ERA5_total_precipitation_prev7d_mm"] = prev7d_total(era5, "ERA5_total_precipitation_mm", t_naive)
        parts.append(w)
    if prism is not None and len(prism):
        p = prism.copy()
        p["Date"] = pd.to_datetime(p["Date"]).dt.normalize()
        p = p.drop_duplicates("Date").set_index("Date").sort_index()
        days = t_naive.normalize()
        w = pd.DataFrame(index=t_naive)
        for c in [c for c in p.columns if c.startswith("PRISM_")]:
            w[c] = p[c].reindex(days).to_numpy(float)
        if "PRISM_ppt" in p.columns:
            roll = p["PRISM_ppt"].asfreq("D").rolling(7, min_periods=6).sum()
            w["PRISM_ppt_prev7d_mm"] = roll.reindex(days).to_numpy(float)
        parts.append(w)
    if nao is not None and len(nao):
        n = nao.copy()
        n["m"] = pd.to_datetime(n["Timestamp"]).dt.to_period("M")
        n = n.drop_duplicates("m").set_index("m")["NAO_index"]
        w = pd.DataFrame(index=t_naive)
        w["NAO_index"] = n.reindex(t_naive.to_period("M")).to_numpy(float)
        w["NAO_index_prev_month"] = n.reindex(t_naive.to_period("M") - 1).to_numpy(float)
        parts.append(w)
    for w in parts:
        w.index = out.index
        out = pd.concat([out, w.reset_index(drop=True)], axis=1)
    return out


# -------------------------------------------------------------- sources ---
def _fetch_power(lat, lon, start, end):
    from APIs.NASApowerAPI import fetch_nasa_data
    parts, s = [], start
    while s <= end:
        e = min(s + timedelta(days=POWER_MAX_DAYS - 1), end)
        parts.append(fetch_nasa_data(lat, lon, s, e, POWER_PARAMS, time_standard="UTC"))
        s = e + timedelta(days=1)
    df = pd.concat(parts, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(None)
    df = df.drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True)
    # POWER marks "no data" with -999; make those NaN
    num = df.select_dtypes("number").columns.difference(["Latitude", "Longitude"])
    df[num] = df[num].where(df[num] > -990)
    return df


def _cached(name, lat, lon, start, end, fetch):
    """Raw downloads are cached in data/hackathon_cache so a rerun does not hit the web again."""
    os.makedirs(os.path.join(DATA, "hackathon_cache"), exist_ok=True)
    path = os.path.join(DATA, "hackathon_cache", f"{name}_{lat:.4f}_{lon:.4f}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Timestamp" if "Timestamp" in pd.read_csv(path, nrows=0).columns
                                            else "Date"])
        col = "Timestamp" if "Timestamp" in df.columns else "Date"
        if df[col].min() <= pd.Timestamp(start) + pd.Timedelta(days=2) and \
           df[col].max() >= pd.Timestamp(end) - pd.Timedelta(days=45):
            return df
    df = fetch()
    df.to_csv(path, index=False)
    return df


def build(lat, lon, n_weeks=520, end="2026-11-05", slot="22:30", log=print):
    import smap_timeseries as sts
    times = sts._target_times(n_weeks, end, slot)               # newest first, tz-aware UTC
    times = sorted(times)
    t0, t1 = times[0].tz_localize(None) if hasattr(times[0], "tz_localize") else times[0].replace(tzinfo=None), \
        times[-1].replace(tzinfo=None)
    start, end_d = (t0 - timedelta(days=8)).date(), t1.date()
    status = {}

    def attempt(name, fn):
        try:
            df = fn()
            status[name] = f"{len(df)} rows"
            return df
        except Exception as e:                                    # keep going: give as much as possible
            status[name] = f"unavailable ({type(e).__name__}: {str(e)[:120]})"
            return None

    log("NASA POWER hourly ...")
    power = attempt("NASA POWER", lambda: _cached("power", lat, lon, start, end_d,
                                                  lambda: _fetch_power(lat, lon, start, end_d)))
    log("PRISM daily ...")

    def prism_fetch():
        from APIs.PRISMAPI import fetch_prism_data
        d = fetch_prism_data(lat, lon, start, end_d, PRISM_VARS)
        return d
    prism = attempt("PRISM", lambda: _cached("prism", lat, lon, start, end_d, prism_fetch))
    log("NOAA NAO monthly ...")
    from APIs.NOAAAPI import fetch_nao_data
    nao = attempt("NOAA NAO", lambda: fetch_nao_data(start - timedelta(days=40), end_d))
    log("ERA5 ...")

    def era5_fetch():
        from APIs.ERA5API import fetch_era5_data
        return fetch_era5_data(lat, lon, start, end_d, os.path.join(DATA, "hackathon_cache", "era5"))
    era5 = attempt("ERA5 (folder program)", era5_fetch)
    if era5 is None and os.path.exists(COMBINED):                  # fall back to what the folder already holds
        era5 = _era5_from_combined(lat, lon)
        status["ERA5 (combined_dataset.csv)"] = ("none (different location)" if era5 is None
                                                 else f"{len(era5)} hourly rows, {era5['Timestamp'].min():%Y-%m-%d} to "
                                                      f"{era5['Timestamp'].max():%Y-%m-%d}")
    out = align_weekly(times, power, prism, nao, era5)
    out.insert(1, "latitude", lat)
    out.insert(2, "longitude", lon)
    return out, status


def _era5_from_combined(lat, lon):
    """ERA5 hours already stored in NASAHackathon2026/data/combined_dataset.csv, if that file is for
    the same 9 km SMAP cell as (lat, lon)."""
    import smap_harmony as sh
    d = pd.read_csv(COMBINED, parse_dates=["Timestamp"])
    if sh.cell_for(float(d["Latitude"].iloc[0]), float(d["Longitude"].iloc[0])) != sh.cell_for(lat, lon):
        return None
    cols = ["Timestamp"] + [c for c in d.columns if c.startswith("ERA5_")]
    return d[cols]


def report(df):
    cols = [c for c in df.columns if c not in ("time", "latitude", "longitude")]
    rows = []
    for c in cols:
        ok = df.loc[df[c].notna(), "time"]
        rows.append((c, len(ok), f"{ok.min():%Y-%m-%d}" if len(ok) else "-", f"{ok.max():%Y-%m-%d}" if len(ok) else "-"))
    return pd.DataFrame(rows, columns=["column", f"weeks with data (of {len(df)})", "first", "last"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("n_weeks", type=int, nargs="?", default=520)
    p.add_argument("--end", default="2026-11-05")
    p.add_argument("--slot", default="22:30")
    a = p.parse_args()
    df, status = build(a.lat, a.lon, a.n_weeks, a.end, a.slot)
    out = os.path.join(DATA, f"hackathon_weekly_{a.lat}_{a.lon}.csv")
    df.to_csv(out, index=False)
    print("\nSources:")
    for k, v in status.items():
        print(f"  {k}: {v}")
    print(f"\nSaved {len(df)} weekly rows ({df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d}) -> {out}\n")
    pd.set_option("display.width", 200)
    print(report(df).to_string(index=False))
