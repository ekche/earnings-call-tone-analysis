"""Command-line entry point.

  python run.py demo                          # synthetic data, no internet needed
  python run.py run --transcripts calls.csv   # real data (needs internet for prices)
"""
import argparse
import pandas as pd
from src import features as F, pipeline, prices, synthetic


def main():
    ap = argparse.ArgumentParser(description="Earnings call tone vs. stock reaction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run on synthetic data")
    d.add_argument("--n", type=int, default=800)
    d.add_argument("--out", default="outputs/demo")
    r = sub.add_parser("run", help="run on real transcripts")
    r.add_argument("--transcripts", required=True, help="CSV with columns: ticker,date,text[,after_close]")
    r.add_argument("--lm-dict", help="path to Loughran-McDonald master dictionary CSV")
    r.add_argument("--finbert", action="store_true", help="also score with FinBERT")
    r.add_argument("--window-end", type=int, default=2, help="last day of event window (default +2)")
    r.add_argument("--out", default="outputs/real")
    a = ap.parse_args()

    if a.cmd == "demo":
        df = synthetic.make_dataset(n=a.n)
        pipeline.run_pipeline(df, a.out, demo=True)
    else:
        df = pd.read_csv(a.transcripts)
        missing = {"ticker", "date", "text"} - set(df.columns)
        if missing:
            raise SystemExit(f"CSV is missing columns: {missing}")
        print("Fetching prices (first run is slow; cached after)...")
        df = prices.attach_car(df, window=(0, a.window_end))
        print(f"  {df['car'].notna().sum()} / {len(df)} calls have price data")
        lex = F.load_lm_dictionary(a.lm_dict) if a.lm_dict else None
        pipeline.run_pipeline(df, a.out, lex=lex, use_finbert=a.finbert)


if __name__ == "__main__":
    main()
