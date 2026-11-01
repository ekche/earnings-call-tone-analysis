# Results

> **DEMO MODE: synthetic data.** The relationship below was planted by the data generator. It shows the pipeline works, not that tone predicts returns.

Sample: 800 calls with complete data.

## 1. Regression (standardised features, ticker-clustered SEs)
Coefficient = change in 3-day abnormal return per 1 SD of the feature.

|                 |    coef |       t |      p |    VIF |
|:----------------|--------:|--------:|-------:|-------:|
| net_tone_prep   |  0.0011 |  0.8042 | 0.4213 | 1.6586 |
| net_tone_qa     |  0.003  |  1.9628 | 0.0497 | 1.6366 |
| qa_uncertainty  | -0.0018 | -1.3977 | 0.1622 | 1.2921 |
| qa_hedge_rate   | -0.0021 | -1.3189 | 0.1872 | 1.7827 |
| qa_evasive_rate | -0.0007 | -0.5253 | 0.5994 | 1.1731 |
| log_words       |  0.0017 |  1.208  | 0.2271 | 1.3627 |

R-squared: 0.0224  (VIF above ~5 would flag collinearity)

## Quintile sort on overall net tone
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0083 | 0.035  |     160 |
|          2 | -0.0048 | 0.0358 |     160 |
|          3 | -0.0009 | 0.0311 |     160 |
|          4 |  0.0012 | 0.0378 |     160 |
|          5 |  0.0001 | 0.0333 |     160 |

Top-minus-bottom spread: 0.84% (Welch t = 2.21, p = 0.028)

## Out-of-sample (train on earliest 70%, test on latest 30%)
- Test calls: 240
- Out-of-sample R-squared: -0.0164 (above 0 means it beat guessing the training mean)
- Direction hit rate: 52.9%
- Long-short (top vs bottom third of predictions), mean abnormal return: 0.01%

## Limitations to discuss in interviews
- Abnormal return uses a simple market-adjusted model, not Fama-French factors.
- Earnings surprise (actual vs. consensus EPS) is not controlled for; tone may just proxy for it.
- Lexicon methods ignore context and negation.
- Management vs analyst is inferred (who spoke before the Q&A), so some executives who only speak in the Q&A are counted as analysts.
- No transaction costs; a long-short spread is not a tradable strategy by itself.
- Multiple features tested, so some significance may be chance.