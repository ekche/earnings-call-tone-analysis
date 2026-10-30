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
