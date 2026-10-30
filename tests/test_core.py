import numpy as np
import pandas as pd
from src import features as F, prices


def test_split_sections():
    text = "Prepared words here. " * 20 + "Operator: question-and-answer session. Q words."
    prep, qa = F.split_sections(text)
    assert "Prepared" in prep and "Q words" in qa


def test_tone_sign():
    pos = F.text_features("strong growth and excellent record results")
    neg = F.text_features("weak decline and difficult adverse loss")
    assert pos["all_pos"] > pos["all_neg"] and neg["all_neg"] > neg["all_pos"]


def test_car_known_values():
    idx = pd.bdate_range("2024-01-01", periods=20)
    bench = pd.Series(100.0, index=idx)
    stock = pd.Series(100.0, index=idx)
    stock.iloc[10:] = 110.0                     # +10% jump on day index 10, then flat
    car = prices.car_from_series(stock, bench, idx[10], window=(0, 2))
    assert abs(car - 0.10) < 1e-9


def test_car_out_of_range_is_nan():
    idx = pd.bdate_range("2024-01-01", periods=5)
    s = pd.Series(range(1, 6), index=idx, dtype=float)
    assert np.isnan(prices.car_from_series(s, s, idx[-1], window=(0, 2)))
