# Results

Speaker split: analysts = voices the operator hands the floor to (or who mostly ask questions); management = everyone else who speaks, including executives who only join the Q&A.
Correlation between management tone and analyst tone: 0.31

Sample: 3060 calls with complete data.

## 1. Regression (standardised features, ticker-clustered SEs)
Coefficient = change in 3-day abnormal return per 1 SD of the feature.

|                   |    coef |       t |      p |    VIF |
|:------------------|--------:|--------:|-------:|-------:|
| net_tone_prep     |  0.002  |  2.3505 | 0.0187 | 1.1519 |
| net_tone_mgmt     |  0.0009 |  0.7303 | 0.4652 | 1.3215 |
| net_tone_analyst  |  0.0047 |  3.432  | 0.0006 | 1.1243 |
| mgmt_unc          |  0.0008 |  0.7955 | 0.4263 | 1.1649 |
| mgmt_hedge_rate   | -0      | -0.0067 | 0.9946 | 1.0407 |
| mgmt_evasive_rate | -0.0001 | -0.108  | 0.914  | 1.0063 |
| log_words         | -0.0007 | -0.6941 | 0.4876 | 1.0306 |

R-squared: 0.0106  (VIF above ~5 would flag collinearity)

## Quintile sort on management answers in the Q&A
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0052 | 0.053  |     615 |
|          2 | -0.0007 | 0.0527 |     614 |
|          3 | -0.0015 | 0.0477 |     614 |
|          4 |  0.0017 | 0.0534 |     614 |
|          5 |  0.0017 | 0.0538 |     615 |

Top-minus-bottom spread: 0.69% (Welch t = 2.26, p = 0.024)

## Quintile sort on analyst questions in the Q&A
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0088 | 0.058  |     617 |
|          2 | -0.0031 | 0.0552 |     617 |
|          3 | -0.0002 | 0.0521 |     617 |
|          4 |  0.0025 | 0.0481 |     617 |
|          5 |  0.0049 | 0.0453 |     617 |

Top-minus-bottom spread: 1.37% (Welch t = 4.63, p = 0.000)

## Out-of-sample (train on earliest 70%, test on latest 30%)
- Test calls: 918
- Out-of-sample R-squared: 0.0067 (above 0 means it beat guessing the training mean)
- Direction hit rate: 51.5%
- Long-short (top vs bottom third of predictions), mean abnormal return: 1.10%

## Limitations to discuss in interviews
- Abnormal return uses a simple market-adjusted model, not Fama-French factors.
- Earnings surprise (actual vs. consensus EPS) is not controlled for; tone may just proxy for it.
- Lexicon methods ignore context and negation.
- Management vs analyst is inferred from speaking order and question marks, so some turns may be mislabelled.
- No transaction costs; a long-short spread is not a tradable strategy by itself.
- Multiple features tested, so some significance may be chance.