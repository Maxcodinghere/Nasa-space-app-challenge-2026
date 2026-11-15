import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from smap_timeseries import weekly_series, VALID_SLOTS

if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Get the data with the same fetcher the new model uses (smap_timeseries.weekly_series):
    # cached weeks come from data/smap_cache.csv, missing weeks are downloaded.
    #   python original_xgboost.py 40.8 -74.0 52 [--target sm_rootzone] [--end YYYY-MM-DD]
    # ---------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("lat", type=float, nargs="?", default=40.8)
    p.add_argument("lon", type=float, nargs="?", default=-74.0)
    p.add_argument("n_weeks", type=int, nargs="?", default=52, help="weeks of history")
    p.add_argument("--target", default="sm_surface", choices=["sm_surface", "sm_rootzone"])
    p.add_argument("--end", default=None, help="most recent date, YYYY-MM-DD")
    p.add_argument("--slot", default="22:30", help=f"one of {VALID_SLOTS}")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    LAT, LON, TARGET = args.lat, args.lon, args.target

    raw = weekly_series(LAT, LON, args.n_weeks, args.end, args.slot, args.workers)
    if raw["sm_surface"].notna().sum() < 12:
        raise SystemExit(f"Only {raw['sm_surface'].notna().sum()} weeks with data for lat={LAT}, lon={LON} "
                         "(ocean pixel or missing granules?). Use more weeks or another location.")

    # one row per week, oldest first; a skipped week is already a NaN row, so row steps = 1 week
    raw = raw.set_index("time").sort_index()

    # X_raw = the CSV data (all measured columns); y_raw = the column we want to predict ahead
    X_raw = raw[["sm_surface", "sm_rootzone"]].to_numpy(dtype=float)
    y_raw = raw[TARGET].to_numpy(dtype=float)
    print(f"Loaded {len(X_raw)} weekly rows for ({LAT}, {LON}); predicting {TARGET}")

    n_lags = 4
    horizon = 1   # how many rows ahead the hidden target is

    def make_causal_features(X, n_lags, lead):
        n, d = X.shape
        blocks = []
        for k in range(lead, lead + n_lags):
            shifted = np.full((n, d), np.nan)
            shifted[k:] = X[:n - k]
            blocks.append(shifted)
        return np.concatenate(blocks, axis=1)

    X_all = make_causal_features(X_raw, n_lags, lead=0)

    first_valid = n_lags - 1
    X = X_all[first_valid:len(X_all) - horizon]
    y = y_raw[first_valid + horizon:]

    d = X_raw.shape[1]
    feature_names = [f"f{j}_lag{k}" for k in range(n_lags) for j in range(d)]

    t0 = 10
    assert np.allclose(X_all[t0, :d], X_raw[t0], equal_nan=True)
    assert len(X) == len(y)
    assert np.allclose(y[0], y_raw[first_valid + horizon], equal_nan=True)

    # weeks with no measurement can't be a training target: drop those rows
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]

    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.075,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=4.0,
        objective="reg:squarederror",
        eval_metric="rmse",
    )

    # Causal split: SLIDING window validation (fixed-size training window moves forward in time)
    gap = 2
    def sliding_splits(n, gap, min_splits=3):
        """
        Sliding-window validation. A FIXED-size training window moves forward in time:
        train on rows [s, s+train), skip `gap` rows, test on the next `test` rows, then
        slide everything forward by `test`. Unlike an expanding split, old data drops out
        of the training window, and the validation windows never overlap.
        Yields (train_idx, val_idx).
        """
        test = max(3, n // 12)
        train = max(8, min(n // 2, n - gap - min_splits * test))
        train = min(train, n - gap - test)            # always leave room for >= 1 window
        start = 0
        while start + train + gap + test <= n:
            yield (np.arange(start, start + train),
                   np.arange(start + train + gap, start + train + gap + test))
            start += test


    preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(sliding_splits(len(y), gap)):
        assert train_idx.max() + gap < val_idx.min()

        model = XGBRegressor(**xgb_params)
        model.fit(X[train_idx], y[train_idx])
        preds[val_idx] = model.predict(X[val_idx])

        mae = mean_absolute_error(y[val_idx], preds[val_idx])
        rmse = mean_squared_error(y[val_idx], preds[val_idx]) ** 0.5
        print(f"Window {fold}: train rows {train_idx.min()}-{train_idx.max()}, "
              f"test rows {val_idx.min()}-{val_idx.max()}, "
              f"MAE = {mae:.3f}, RMSE = {rmse:.3f}")

    # Overall performance on predicted rows
    scored = ~np.isnan(preds)
    mse_model = mean_squared_error(y[scored], preds[scored])

    print(f"\nScored rows: {scored.sum()} of {len(y)}")
    print(f"MAE: {mean_absolute_error(y[scored], preds[scored]):.3f}")
    print(f"RMSE: {mse_model ** 0.5:.3f}")
    print(f"Correlation: {np.corrcoef(y[scored], preds[scored])[0, 1]:.3f}")

    # Final model and feature importance
    final_model = XGBRegressor(**xgb_params)
    final_model.fit(X, y)

    # ---------------------------------------------------------------------------
    # Test the predictions: every scored week, the model's guess vs what really happened
    # (each guess came from a model trained only on weeks BEFORE it, see the causal split)
    # ---------------------------------------------------------------------------
    target_dates = raw.index[first_valid + horizon:][ok]       # the week each y refers to
    naive = X_raw[first_valid:len(X_raw) - horizon, 0 if TARGET == "sm_surface" else 1][ok]  # "same as now"
    test_table = pd.DataFrame({"week": target_dates[scored], "actual": y[scored],
                               "predicted": preds[scored], "naive_same_as_now": naive[scored]})
    test_table["error"] = test_table["predicted"] - test_table["actual"]
    print(f"\nPredicted vs actual ({TARGET}, {horizon} week(s) ahead), last 10 tested weeks:")
    print(test_table.tail(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nModel   MAE {test_table['error'].abs().mean():.4f}")
    print(f"Naive   MAE {(test_table['naive_same_as_now'] - test_table['actual']).abs().mean():.4f}"
          "   (guess = last measured value; the model should beat this)")

    # Real forecast: the newest row (latest week + earlier) predicts `horizon` week(s) past the data
    next_week = raw.index[-1] + pd.Timedelta(weeks=horizon)
    print(f"\nForecast for {next_week:%Y-%m-%d}: {final_model.predict(X_all[-1:])[0]:.4f}  ({TARGET})")

    imp = (pd.Series(final_model.feature_importances_, index=feature_names)
             .sort_values(ascending=False))
    print("\nTop features:")
    print(imp.head(10))

    try:
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(final_model, num_trees=0, ax=ax)
        plt.show()
    except Exception as e:
        print("\nGraphviz not available, showing tree 0 as a table instead.")
        print(f"({type(e).__name__})")
        tree_df = final_model.get_booster().trees_to_dataframe()
        print(tree_df[tree_df["Tree"] == 0])