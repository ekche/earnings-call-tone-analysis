# Results

Sample: 3106 calls with complete data.

## 1. Regression (standardised features, ticker-clustered SEs)
Coefficient = change in 3-day abnormal return per 1 SD of the feature.

|                 |    coef |       t |      p |    VIF |
|:----------------|--------:|--------:|-------:|-------:|
| net_tone_prep   |  0.0017 |  2.1165 | 0.0343 | 1.0901 |
| net_tone_qa     |  0.0039 |  3.9005 | 0.0001 | 1.1862 |
| qa_uncertainty  |  0.0015 |  1.413  | 0.1576 | 1.1714 |
| qa_hedge_rate   | -0.0003 | -0.3225 | 0.7471 | 1.0635 |
| qa_evasive_rate |  0.0002 |  0.1869 | 0.8517 | 1.0032 |
| log_words       | -0.0009 | -0.9248 | 0.3551 | 1.0627 |

R-squared: 0.0065  (VIF above ~5 would flag collinearity)

## 2. Quintile sort on net tone
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0075 | 0.0527 |     622 |
|          2 | -0.0014 | 0.051  |     621 |
|          3 | -0.0024 | 0.0529 |     621 |
|          4 |  0.0031 | 0.0514 |     621 |
|          5 |  0.0039 | 0.0518 |     621 |

Top-minus-bottom spread: 1.13% (Welch t = 3.82, p = 0.000)

## 3. Out-of-sample (train on earliest 70%, test on latest 30%)
- Test calls: 932
- Out-of-sample R-squared: 0.0001 (above 0 means it beat guessing the training mean)
- Direction hit rate: 51.8%
- Long-short (top vs bottom third of predictions), mean abnormal return: 0.38%

## Limitations to discuss in interviews
- Abnormal return uses a simple market-adjusted model, not Fama-French factors.
- Earnings surprise (actual vs. consensus EPS) is not controlled for; tone may just proxy for it.
- Lexicon methods ignore context and negation.
- No transaction costs; a long-short spread is not a tradable strategy by itself.
- Multiple features tested, so some significance may be chance.