# Results

Sample: 3106 calls with complete data.

## 1. Regression (standardised features, ticker-clustered SEs)
Coefficient = change in 3-day abnormal return per 1 SD of the feature.

|                 |    coef |       t |      p |    VIF |
|:----------------|--------:|--------:|-------:|-------:|
| net_tone_prep   |  0.0011 |  1.3164 | 0.188  | 1.0791 |
| net_tone_qa     |  0.0039 |  3.8425 | 0.0001 | 1.102  |
| qa_uncertainty  |  0.0014 |  1.3603 | 0.1737 | 1.065  |
| qa_hedge_rate   |  0.0006 |  0.6792 | 0.497  | 1.1264 |
| qa_evasive_rate |  0.0002 |  0.1387 | 0.8897 | 1.0032 |
| log_words       | -0.0007 | -0.7543 | 0.4507 | 1.0222 |

R-squared: 0.0064  (VIF above ~5 would flag collinearity)

## 2. Quintile sort on net tone
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0066 | 0.0554 |     622 |
|          2 | -0.0016 | 0.0557 |     621 |
|          3 |  0.0001 | 0.051  |     621 |
|          4 | -0.0008 | 0.0482 |     621 |
|          5 |  0.0046 | 0.0492 |     621 |

Top-minus-bottom spread: 1.12% (Welch t = 3.77, p = 0.000)

## 3. Out-of-sample (train on earliest 70%, test on latest 30%)
- Test calls: 932
- Out-of-sample R-squared: 0.0036 (above 0 means it beat guessing the training mean)
- Direction hit rate: 50.1%
- Long-short (top vs bottom third of predictions), mean abnormal return: 0.68%

## Limitations to discuss in interviews
- Abnormal return uses a simple market-adjusted model, not Fama-French factors.
- Earnings surprise (actual vs. consensus EPS) is not controlled for; tone may just proxy for it.
- Lexicon methods ignore context and negation.
- No transaction costs; a long-short spread is not a tradable strategy by itself.
- Multiple features tested, so some significance may be chance.