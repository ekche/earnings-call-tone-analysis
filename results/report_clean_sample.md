# Results

Speaker split: analysts = voices the operator hands the floor to (or who mostly ask questions); management = everyone else who speaks, including executives who only join the Q&A.
Correlation between management tone and analyst tone: 0.30

Sample: 2638 calls with complete data.

## 1. Regression (standardised features, ticker-clustered SEs)
Coefficient = change in 3-day abnormal return per 1 SD of the feature.

|                   |    coef |       t |      p |    VIF |
|:------------------|--------:|--------:|-------:|-------:|
| net_tone_prep     |  0.0018 |  2.1247 | 0.0336 | 1.1519 |
| net_tone_mgmt     |  0.0013 |  1.0092 | 0.3129 | 1.3049 |
| net_tone_analyst  |  0.0041 |  2.8622 | 0.0042 | 1.1099 |
| mgmt_unc          |  0.0003 |  0.2498 | 0.8027 | 1.1631 |
| mgmt_hedge_rate   | -0.0001 | -0.0759 | 0.9395 | 1.0415 |
| mgmt_evasive_rate | -0.0002 | -0.1989 | 0.8424 | 1.0027 |
| log_words         | -0.0005 | -0.3903 | 0.6963 | 1.0305 |

R-squared: 0.0099  (VIF above ~5 would flag collinearity)

## Quintile sort on management answers in the Q&A
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0067 | 0.0517 |     530 |
|          2 | -0.001  | 0.0518 |     530 |
|          3 |  0.0006 | 0.0481 |     529 |
|          4 |  0.0003 | 0.0508 |     530 |
|          5 |  0.0029 | 0.0548 |     530 |

Top-minus-bottom spread: 0.96% (Welch t = 2.92, p = 0.004)

## Quintile sort on analyst questions in the Q&A
|   quintile |    mean |    std |   count |
|-----------:|--------:|-------:|--------:|
|          1 | -0.0075 | 0.0551 |     533 |
|          2 | -0.0032 | 0.0566 |     532 |
|          3 | -0.0008 | 0.0516 |     532 |
|          4 |  0.0026 | 0.0473 |     532 |
|          5 |  0.0042 | 0.0452 |     532 |

Top-minus-bottom spread: 1.17% (Welch t = 3.78, p = 0.000)

## Out-of-sample (train on earliest 70%, test on latest 30%)
- Test calls: 792
- Out-of-sample R-squared: 0.0058 (above 0 means it beat guessing the training mean)
- Direction hit rate: 51.6%
- Long-short (top vs bottom third of predictions), mean abnormal return: 0.92%

## Limitations to discuss in interviews
- Abnormal return uses a simple market-adjusted model, not Fama-French factors.
- Earnings surprise (actual vs. consensus EPS) is not controlled for; tone may just proxy for it.
- Lexicon methods ignore context and negation.
- Management vs analyst is inferred from speaking order and question marks, so some turns may be mislabelled.
- No transaction costs; a long-short spread is not a tradable strategy by itself.
- Multiple features tested, so some significance may be chance.