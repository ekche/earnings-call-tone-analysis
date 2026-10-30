"""Convert the Hugging Face S&P 500 transcript dataset into data/calls.csv.

  python make_dataset.py                          # 100 random tickers, 2016-2024
  python make_dataset.py --tickers 200 --start 2012 --end 2024
"""
import argparse
import numpy as np
import pandas as pd
from src.features import QA_TOKEN


def build_text(turns):
    """Join speaker turns into one string, inserting QA_TOKEN where Q&A begins.

    Q&A is assumed to start at the first Operator turn that mentions 'question'
    AFTER a non-Operator (management) turn has already spoken."""
    parts, seen_mgmt, marked = [], False, False
    for t in (turns if turns is not None else []):
        speaker = str(t.get("speaker", "")).strip()
        text = str(t.get("text", "")).strip()
        is_op = speaker.lower().startswith("operator")
        if is_op and seen_mgmt and not marked and "question" in text.lower():
            parts.append(QA_TOKEN)
            marked = True
        if not is_op:
            seen_mgmt = True
        parts.append(f"{speaker}: {text}")
    return " ".join(parts), marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=100)
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/calls.csv")
    a = ap.parse_args()

    from datasets import load_dataset
    print("Loading dataset (cached after the first time)...")
    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")

    dates = pd.to_datetime(pd.Series(ds["date"]))
    syms = pd.Series(ds["symbol"])
    in_range = (dates.dt.year >= a.start) & (dates.dt.year <= a.end)
    counts = syms[in_range].value_counts()
    eligible = counts[counts >= 20].index.to_numpy()
    rng = np.random.default_rng(a.seed)
    chosen = rng.choice(eligible, size=min(a.tickers, len(eligible)), replace=False)
    idx = np.where(in_range & syms.isin(chosen))[0]
    print(f"Selected {len(chosen)} tickers, {len(idx)} calls from {a.start} to {a.end}")

    df = ds.select(idx).select_columns(["symbol", "date", "structured_content"]).to_pandas()
    df["dt"] = pd.to_datetime(df["date"])
    built = [build_text(t) for t in df["structured_content"]]
    df["text"] = [b[0] for b in built]
    df["has_qa"] = [b[1] for b in built]
    print(f"Q&A section detected in {df['has_qa'].mean():.1%} of calls (others dropped)")
    df = df[df["has_qa"]].copy()

    out = pd.DataFrame({
        "ticker": df["symbol"].str.replace(".", "-", regex=False),   # Yahoo uses BRK-B
        "date": df["dt"].dt.strftime("%Y-%m-%d"),
        "text": df["text"],
        "after_close": df["dt"].dt.hour >= 16,    # assumes timestamps are US Eastern
    })
    print("Call start hour distribution (check this looks like US Eastern business hours):")
    print(df["dt"].dt.hour.value_counts().sort_index().to_string())
    print(f"after_close share: {out['after_close'].mean():.1%}")
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} calls to {a.out}")


if __name__ == "__main__":
    main()
