"""Synthetic transcripts + returns so the pipeline runs end to end with no data.

NOTE: The 'finding' in demo mode is baked in by construction (a weak link between
tone and returns). It demonstrates the pipeline, NOT anything about real markets.
"""
import numpy as np
import pandas as pd

NEUTRAL = ["Revenue for the quarter was reported across our three segments.",
           "Operating expenses were allocated according to the plan we outlined.",
           "We completed the integration of two small acquisitions this period.",
           "Cash flow from operations was recorded in line with the prior period.",
           "The board approved the capital allocation framework for next year.",
           "Our regional teams continued executing across North America and Europe."]
POS = ["We delivered strong growth and excellent margin improvement.",
       "Momentum was robust and we exceeded our targets with record results.",
       "We are confident in the opportunity ahead and achieved healthy efficiency gains."]
NEG = ["We faced difficult headwinds and a significant decline in demand.",
       "Weak results and adverse pressure led to a shortfall versus plan.",
       "Challenging conditions caused a loss and a disappointing slowdown."]
UNC = ["It is uncertain how conditions may develop and results could vary.",
       "We believe the outcome depends on risks that are unclear at this point."]
Q = ["Operator: Our next question comes from an analyst at a major bank.",
     "Analyst: Can you talk about the outlook for next quarter and margins?"]
HEDGE = ["I think it's too early to say, and we're not going to guide beyond that.",
         "It's kind of hard to say, it depends, and we'll see how it plays out.",
         "As we've said before, we don't break out that detail."]
ANS_NEUTRAL = ["We continue to execute on the plan across the business.",
               "Thanks for the question, the team is focused on the segments we discussed."]


def _pick(rng, pool, k):
    return [pool[i] for i in rng.integers(0, len(pool), size=k)]


def make_transcript(rng, tone, anxiety):
    p_pos = 1 / (1 + np.exp(-1.5 * tone))
    prep = []
    for _ in range(30):
        r = rng.random()
        if r < 0.45:
            prep += _pick(rng, NEUTRAL, 1)
        elif r < 0.45 + 0.55 * p_pos:
            prep += _pick(rng, POS, 1)
        else:
            prep += _pick(rng, NEG, 1)
    p_unc = 1 / (1 + np.exp(-(anxiety - 1.0)))
    qa = []
    for _ in range(20):
        qa += _pick(rng, Q, 1)
        r = rng.random()
        if r < 0.15:
            qa += _pick(rng, POS if rng.random() < p_pos else NEG, 1)
        elif r < 0.15 + 0.35 * p_unc:
            qa += _pick(rng, HEDGE, 1)
        elif r < 0.15 + 0.55 * p_unc + 0.2:
            qa += _pick(rng, UNC, 1)
        else:
            qa += _pick(rng, ANS_NEUTRAL, 1)
    return " ".join(prep) + " Operator: We will now begin the question-and-answer session. " + " ".join(qa)


def make_dataset(n=800, n_tickers=60, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
    dates = pd.to_datetime("2018-01-15") + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D")
    rows = []
    for d in sorted(dates):
        tone, anxiety = rng.normal(), rng.normal()
        car = 0.006 * tone - 0.004 * anxiety + rng.normal(0, 0.035)   # baked-in weak signal
        rows.append({"ticker": tickers[rng.integers(0, n_tickers)], "date": d.date().isoformat(),
                     "text": make_transcript(rng, tone, anxiety), "car": car})
    return pd.DataFrame(rows)
