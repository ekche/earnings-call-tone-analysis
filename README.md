# Do earnings-call words predict the stock reaction?

Pipeline that scores earnings-call transcripts (tone, uncertainty, hedging, evasiveness),
measures the 3-day market-adjusted return after each call, and tests whether the language
predicts the move with a regression, a quintile sort, and a time-split out-of-sample test.

## Run it
    pip install -r requirements.txt
    python -m pytest -q tests
    python run.py demo
    python run.py run --transcripts data/calls.csv --lm-dict data/LM_MasterDictionary.csv

Demo mode uses synthetic data with a planted relationship. It shows the pipeline works,
not that tone predicts returns. Real transcripts need a CSV with columns ticker,date,text.

## Limitations
Market-adjusted returns only (no factor model), earnings surprise not controlled for,
lexicon methods ignore context, no transaction costs.
## Limitations
- **Association, not forecast.** Many calls start after the earnings release, so day 0 can already include the reaction to the numbers.
- **Earnings surprise is not controlled for.** Tone may simply proxy for good or bad results.
- **Q&A text includes analysts' questions,** not only management's answers.
- **Survivorship bias.** The data covers current S&P 500 constituents; 233 calls had no price data (mostly acquired or delisted companies).
- **Simple benchmark.** Market-adjusted returns, not a Fama-French factor model.
- **Multiple testing.** Six features were tested; lexicon methods ignore context and negation.
- Tone is not a tradable strategy: no transaction costs, and the out-of-sample test shows no reliable edge.

## Reproduce
    pip install -r requirements.txt
    python -m pytest -q tests
    python patch_features.py                  # one-time: enables Q&A split marker
    python make_dataset.py                    # builds data/calls.csv from Hugging Face
    python run.py run --transcripts data/calls.csv --lm-dict data/LM_MasterDictionary.csv

Download the Loughran-McDonald Master Dictionary CSV from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and save it as data/LM_MasterDictionary.csv (free for non-commercial research; not redistributed here). Add --finbert to also score with FinBERT (needs torch and transformers). `python run.py demo` runs on synthetic data to check the pipeline; its planted relationship is not a finding.

## Next steps
Separate management answers from analyst questions using speaker labels, control for earnings surprise, test a post-call drift window (days +1 to +3), and add Fama-French factors.
