"""Market reaction: cumulative abnormal return (CAR) around the call date."""
import os
import numpy as np
import pandas as pd


def car_from_series(stock, bench, event_date, window=(0, 2), max_gap_days=5):
    """Sum of (stock return - benchmark return) over trading days [window]
    relative to the first trading day on/after event_date. Returns NaN if the
    window falls outside the data."""
    rets = pd.concat([stock.pct_change(), bench.pct_change()], axis=1, keys=["s", "b"]).dropna()
    ab = rets["s"] - rets["b"]
    ev = pd.Timestamp(event_date)
    idx = ab.index.searchsorted(ev)
    if idx >= len(ab) or (ab.index[idx] - ev).days > max_gap_days:
        return np.nan
    lo, hi = idx + window[0], idx + window[1]
    if lo < 0 or hi >= len(ab):
        return np.nan
    return float(ab.iloc[lo:hi + 1].sum())


def fetch_close(ticker, start, end, cache_dir="data/price_cache"):
    """Download adjusted close prices via yfinance, cached to CSV."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if h.empty:
        return pd.Series(dtype=float)
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.to_csv(path)
    return s


def attach_car(events, window=(0, 2), benchmark="SPY", cache_dir="data/price_cache"):
    """Add a 'car' column. If events has a boolean 'after_close' column, the
    window starts one trading day later (the market reacts the next morning)."""
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    start = (ev["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (ev["date"].max() + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    bench = fetch_close(benchmark, start, end, cache_dir)
    cache, cars = {}, []
    for _, r in ev.iterrows():
        t = r["ticker"]
        if t not in cache:
            try:
                cache[t] = fetch_close(t, start, end, cache_dir)
            except Exception as e:  # network / delisted ticker
                print(f"  WARNING could not fetch {t}: {e}")
                cache[t] = pd.Series(dtype=float)
        s = cache[t]
        shift = 1 if bool(r.get("after_close", False)) else 0
        w = (window[0] + shift, window[1] + shift)
        cars.append(np.nan if s.empty else car_from_series(s, bench, r["date"], w))
    ev["car"] = cars
    return ev
