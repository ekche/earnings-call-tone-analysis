import os
import pandas as pd
from . import analysis, features as F


def run_pipeline(df, outdir="outputs", lex=None, use_finbert=False, demo=False):
    os.makedirs(outdir, exist_ok=True)
    print(f"[1/4] Building text features for {len(df)} calls...")
    df = F.build_features(df, lex)
    feats = list(analysis.FEATURES)
    if use_finbert:
        print("[1b] Scoring with FinBERT (slow on CPU)...")
        df["finbert"] = F.finbert_scores(df["text"].tolist())
        feats.append("finbert")
    df.drop(columns=["text"]).to_csv(f"{outdir}/features.csv", index=False)

    print("[2/4] Regression...")
    res, n = analysis.run_regression(df, feats)
    open(f"{outdir}/regression.txt", "w").write(str(res.summary()))

    print("[3/4] Quintile sort + out-of-sample test...")
    tab, spread = analysis.quintile_table(df, "net_tone")
    tab.to_csv(f"{outdir}/quintiles.csv")
    oos = analysis.out_of_sample(df, feats)

    print("[4/4] Plots + report...")
    analysis.make_plots(df, outdir)
    coef = pd.DataFrame({"coef": res.params, "t": res.tvalues, "p": res.pvalues}).drop("Intercept")
    coef["VIF"] = analysis.vif_table(df, feats)
    lines = ["# Results", ""]
    if demo:
        lines += ["> **DEMO MODE: synthetic data.** The relationship below was planted by the "
                  "data generator. It shows the pipeline works, not that tone predicts returns.", ""]
    lines += [f"Sample: {n} calls with complete data.", "",
              "## 1. Regression (standardised features, ticker-clustered SEs)",
              "Coefficient = change in 3-day abnormal return per 1 SD of the feature.", "",
              coef.round(4).to_markdown(), "",
              f"R-squared: {res.rsquared:.4f}  (VIF above ~5 would flag collinearity)", "",
              "## 2. Quintile sort on net tone",
              tab.round(4).to_markdown(), "",
              f"Top-minus-bottom spread: {spread['spread']*100:.2f}% (Welch t = {spread['t']:.2f}, p = {spread['p']:.3f})", "",
              "## 3. Out-of-sample (train on earliest 70%, test on latest 30%)",
              f"- Test calls: {oos['n_test']}",
              f"- Out-of-sample R-squared: {oos['oos_r2']:.4f} (above 0 means it beat guessing the training mean)",
              f"- Direction hit rate: {oos['hit_rate']:.1%}",
              f"- Long-short (top vs bottom third of predictions), mean abnormal return: {oos['long_short_test']*100:.2f}%", "",
              "## Limitations to discuss in interviews",
              "- Abnormal return uses a simple market-adjusted model, not Fama-French factors.",
              "- Earnings surprise (actual vs. consensus EPS) is not controlled for; tone may just proxy for it.",
              "- Lexicon methods ignore context and negation.",
              "- No transaction costs; a long-short spread is not a tradable strategy by itself.",
              "- Multiple features tested, so some significance may be chance."]
    open(f"{outdir}/report.md", "w").write("\n".join(lines))
    print(f"\nDone. See {outdir}/report.md")
    return df, res, tab, oos
