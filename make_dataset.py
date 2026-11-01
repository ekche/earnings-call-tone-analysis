"""Convert the Hugging Face S&P 500 transcript dataset into data/calls.csv.

Also splits each call's Q&A into management answers vs analyst questions.

  python make_dataset.py                          # 100 random tickers, 2016-2024
  python make_dataset.py --tickers 200 --start 2012 --end 2024
"""
import argparse
import os
import numpy as np
import pandas as pd
from src.features import QA_TOKEN


def parse_call(turns):
    """Return dict: text (with QA_TOKEN), has_qa, analyst_qa, mgmt_qa, roles.

    Q&A starts at the first Operator turn that mentions 'question' after a
    non-Operator turn has spoken. Roles: anyone who spoke before the Q&A is
    management; anyone the Operator hands the floor to is an analyst; others
    count as analysts only if at least 25% of their turns contain a question mark."""
    norm = []
    for t in (turns if turns is not None else []):
        sp = str(t.get("speaker", "") or "").strip()
        tx = str(t.get("text", "") or "").strip()
        norm.append((sp, tx, sp.lower().startswith("operator")))
    qa_idx, seen_mgmt = None, False
    for i, (sp, tx, is_op) in enumerate(norm):
        if is_op and seen_mgmt and "question" in tx.lower():
            qa_idx = i
            break
        if not is_op:
            seen_mgmt = True
    parts = []
    for i, (sp, tx, _) in enumerate(norm):
        if i == qa_idx:
            parts.append(QA_TOKEN)
        parts.append(f"{sp}: {tx}")
    out = {"text": " ".join(parts), "has_qa": qa_idx is not None,
           "analyst_qa": "", "mgmt_qa": "", "roles": {}}
    if qa_idx is None:
        return out
    prep_mgmt = {sp.lower() for sp, tx, is_op in norm[:qa_idx] if not is_op and sp}
    after_op, turns_by, q_by = set(), {}, {}
    prev_op = False
    for sp, tx, is_op in norm[qa_idx:]:
        if is_op:
            prev_op = True
            continue
        key = sp.lower()
        if sp:
            turns_by[key] = turns_by.get(key, 0) + 1
            q_by[key] = q_by.get(key, 0) + (1 if "?" in tx else 0)
            if prev_op:
                after_op.add(key)
        prev_op = False

    def is_analyst(key):
        if key in prep_mgmt:
            return False
        if key in after_op:
            return True
        return q_by.get(key, 0) / max(turns_by.get(key, 1), 1) >= 0.25

    analyst_txt, mgmt_txt = [], []
    for sp, tx, is_op in norm[qa_idx + 1:]:
        if is_op or not sp:
            continue
        if is_analyst(sp.lower()):
            analyst_txt.append(tx)
            out["roles"][sp] = "analyst"
        else:
            mgmt_txt.append(tx)
            out["roles"][sp] = "management"
    out["analyst_qa"] = " ".join(analyst_txt)
    out["mgmt_qa"] = " ".join(mgmt_txt)
    return out


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
    parsed = [parse_call(t) for t in df["structured_content"]]
    for k in ("text", "has_qa", "analyst_qa", "mgmt_qa"):
        df[k] = [p[k] for p in parsed]
    print(f"Q&A section detected in {df['has_qa'].mean():.1%} of calls (others dropped)")

    print("\nSanity check: who was labelled what in one example call:")
    for p in parsed:
        if p["roles"]:
            for sp, role in list(p["roles"].items())[:10]:
                print(f"  {sp[:40]:40s} -> {role}")
            break

    df = df[df["has_qa"]].copy()
    a_w = df["analyst_qa"].str.split().str.len()
    m_w = df["mgmt_qa"].str.split().str.len()
    print(f"\nMedian Q&A words: analysts {a_w.median():.0f}, management {m_w.median():.0f}")
    print(f"Calls with >=30 words from both groups: {((a_w >= 30) & (m_w >= 30)).mean():.1%}")

    out = pd.DataFrame({
        "ticker": df["symbol"].str.replace(".", "-", regex=False),   # Yahoo uses BRK-B
        "date": df["dt"].dt.strftime("%Y-%m-%d"),
        "text": df["text"],
        "after_close": df["dt"].dt.hour >= 16,    # assumes timestamps are US Eastern
        "analyst_qa": df["analyst_qa"],
        "mgmt_qa": df["mgmt_qa"],
    })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nWrote {len(out)} calls to {a.out}")


if __name__ == "__main__":
    main()
