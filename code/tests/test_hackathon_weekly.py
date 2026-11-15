"""Offline tests for hackathon_weekly (synthetic data, no network)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hackathon_weekly as hw  # noqa: E402

TIMES = [pd.Timestamp("2026-10-15 22:30", tz="UTC") + pd.Timedelta(weeks=k) for k in range(3)]
STAMPS = pd.date_range("2026-10-01", "2026-11-10", freq="h")      # naive UTC hours


def hourly(**cols):
    return pd.DataFrame({"Timestamp": STAMPS, **{k: v(STAMPS) for k, v in cols.items()}})


def test_window_is_the_three_hours_around_the_smap_time():
    h = hourly(T2M=lambda s: s.hour.astype(float))                 # value = the hour of day
    w = hw.window_values(h, pd.DatetimeIndex([t.tz_localize(None) for t in TIMES]))
    assert w["T2M"].tolist() == [22.0, 22.0, 22.0]                  # mean of 21, 22, 23


def test_window_needs_all_three_hours():
    h = hourly(T2M=lambda s: np.ones(len(s)))
    h.loc[h.Timestamp == pd.Timestamp("2026-10-15 23:00"), "T2M"] = np.nan
    w = hw.window_values(h, pd.DatetimeIndex([t.tz_localize(None) for t in TIMES]))
    assert np.isnan(w["T2M"].iloc[0]) and w["T2M"].iloc[1] == 1.0


def test_prev7d_total_uses_only_hours_before_the_smap_time():
    h = hourly(PRECTOTCORR=lambda s: np.ones(len(s)))
    t = pd.DatetimeIndex([TIMES[1].tz_localize(None)])
    assert hw.prev7d_total(h, "PRECTOTCORR", t)[0] == 168.0         # 7 days x 24 h
    h.loc[h.Timestamp >= pd.Timestamp("2026-10-22 22:00"), "PRECTOTCORR"] = 1000.0   # at/after T: must not count
    assert hw.prev7d_total(h, "PRECTOTCORR", t)[0] == 168.0


def test_wind_direction_comes_from_averaged_components_not_averaged_angles():
    h = hourly(U10M=lambda s: np.full(len(s), 1.0), V10M=lambda s: np.zeros(len(s)),
               WD10M=lambda s: np.where(s.hour == 22, 350.0, 10.0))  # averaging these angles would give ~123
    w = hw.window_values(h, pd.DatetimeIndex([TIMES[0].tz_localize(None)]))
    assert abs(w["WD10M"].iloc[0] - 270.0) < 1e-6                    # u=+1, v=0 -> wind from the west


def test_weekly_table_keeps_the_soil_grid_and_missing_sources_stay_nan():
    power = hourly(T2M=lambda s: np.full(len(s), 5.0))
    prism = pd.DataFrame({"Date": pd.date_range("2026-10-01", "2026-11-10"), "PRISM_ppt": 2.0})
    nao = pd.DataFrame({"Timestamp": pd.to_datetime(["2026-09-01", "2026-10-01"]), "NAO_index": [0.5, 1.5]})
    times = TIMES + [pd.Timestamp("2026-11-05 22:30", tz="UTC")]       # 3 October weeks + 1 November week
    out = hw.align_weekly(times, power=power, prism=prism, nao=nao, era5=None)
    assert out["time"].tolist() == times                              # identical timestamps to the SMAP table
    assert out["T2M"].tolist() == [5.0] * 4
    assert out["PRISM_ppt"].tolist() == [2.0] * 4 and out["PRISM_ppt_prev7d_mm"].iloc[1] == 14.0
    assert out["NAO_index"].iloc[0] == 1.5 and out["NAO_index_prev_month"].iloc[0] == 0.5   # previous month: known at T
    assert np.isnan(out["NAO_index"].iloc[3])                         # November not published -> stays empty
    assert out["NAO_index_prev_month"].iloc[3] == 1.5                 # but October is known by then
    assert not any(c.startswith("ERA5_") for c in out.columns)        # absent source adds nothing
