"""Statistical tests: does call language predict the stock reaction?"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import Ridge

# Prepared-remarks tone and Q&A tone are entered separately. Overall net_tone and
# tone_gap are linear combinations of them, so including all four is collinear.
FEATURES = ["net_tone_prep", "net_tone_qa", "qa_uncertainty", "qa_hedge_rate",
            "qa_evasive_rate", "log_words"]


def zscore(df, cols, ref=None):
    ref = df if ref is None else ref
    return (df[cols] - ref[cols].mean()) / ref[cols].std(ddof=0).replace(0, 1)


def clean(df, features, target="car"):
    d = df.dropna(subset=features + [target]).copy()
    # winsorise the target so one wild day doesn't dominate
    lo, hi = d[target].quantile([0.01, 0.99])
    d[target] = d[target].clip(lo, hi)
    return d


def run_regression(df, features, target="car"):
    """OLS of CAR on standardised features; SEs clustered by ticker when possible."""
    d = clean(df, features, target)
    d[features] = zscore(d, features)
    model = smf.ols(f"{target} ~ " + " + ".join(features), data=d)
    if d["ticker"].nunique() >= 10:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": d["ticker"].astype("category").cat.codes})
    else:
        res = model.fit(cov_type="HC1")
    return res, len(d)


def vif_table(df, features, target="car"):
    """Variance inflation factors; values above ~5 signal troublesome collinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = clean(df, features, target)
    X = zscore(d, features).assign(const=1.0)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(len(features))],
                     index=features, name="VIF")


def quintile_table(df, signal="net_tone", target="car"):
    d = clean(df, [signal], target)
    d["quintile"] = pd.qcut(d[signal].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    tab = d.groupby("quintile", observed=True)[target].agg(["mean", "std", "count"])
    top, bot = d[d.quintile == 5][target], d[d.quintile == 1][target]
    t, p = stats.ttest_ind(top, bot, equal_var=False)
    return tab, {"spread": float(top.mean() - bot.mean()), "t": float(t), "p": float(p)}


def out_of_sample(df, features, target="car", train_frac=0.7, alpha=50.0):
    """Train on the earliest 70% of calls, test on the latest 30% (no look-ahead)."""
    d = clean(df, features, target).sort_values("date").reset_index(drop=True)
    cut = int(len(d) * train_frac)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    Xtr, Xte = zscore(tr, features), zscore(te, features, ref=tr)
    m = Ridge(alpha=alpha).fit(Xtr, tr[target])
    pred = m.predict(Xte)
    sse = ((te[target] - pred) ** 2).sum()
    sst = ((te[target] - tr[target].mean()) ** 2).sum()
    n3 = max(len(te) // 3, 1)
    order = np.argsort(pred)
    ls = te[target].values[order[-n3:]].mean() - te[target].values[order[:n3]].mean()
    return {"n_train": len(tr), "n_test": len(te), "oos_r2": float(1 - sse / sst),
            "hit_rate": float((np.sign(pred - pred.mean()) == np.sign(te[target] - tr[target].mean())).mean()),
            "long_short_test": float(ls),
            "coefs": dict(zip(features, m.coef_.round(5)))}


def make_plots(df, outdir, signal="net_tone", target="car"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab, _ = quintile_table(df, signal, target)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tab.index.astype(str), tab["mean"] * 100, color="#3b6ea5")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(f"{signal} quintile (1 = most negative)")
    ax.set_ylabel("Mean abnormal return, % (days 0 to +2)")
    ax.set_title("Market reaction by call tone")
    fig.tight_layout(); fig.savefig(f"{outdir}/quintiles.png", dpi=150); plt.close(fig)

    d = clean(df, [signal], target)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d[signal], d[target] * 100, s=8, alpha=0.4)
    b = np.polyfit(d[signal], d[target] * 100, 1)
    xs = np.linspace(d[signal].min(), d[signal].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="crimson")
    ax.set_xlabel(signal); ax.set_ylabel("Abnormal return, %")
    fig.tight_layout(); fig.savefig(f"{outdir}/scatter.png", dpi=150); plt.close(fig)
