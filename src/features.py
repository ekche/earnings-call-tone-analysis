"""Text features for earnings-call transcripts.

Default word lists are SMALL, for demo purposes. For real results, download the
Loughran-McDonald Master Dictionary (free, Notre Dame SRAF) and pass its CSV
path with --lm-dict; see load_lm_dictionary().
"""
import re
import numpy as np
import pandas as pd

POSITIVE = set("""strong growth improved improvement improving excellent record momentum
gain gains gained success successful successfully confident opportunity opportunities
benefit benefits outperform outperformed exceeded favorable robust healthy efficient
efficiency leadership innovative best delivered achieve achieved profitable""".split())

NEGATIVE = set("""decline declined declining weak weakness difficult headwind headwinds loss
losses challenging challenges adverse impairment restructuring shortfall missed miss
downturn pressure pressures deteriorate deteriorated disappointing litigation default
unfavorable slowdown softness""".split())

UNCERTAIN = set("""uncertain uncertainty uncertainties may might could possibly perhaps
approximately unclear unpredictable depends depend fluctuate fluctuations risk risks
assume believe anticipate volatile""".split())

# Phrases managers use to soften or dodge. Counted per 1,000 words.
HEDGE_PHRASES = [r"\bi think\b", r"\bkind of\b", r"\bsort of\b", r"\bto some extent\b",
                 r"\bwe'll see\b", r"\bit depends\b", r"\bhard to say\b", r"\bat this point\b"]
EVASIVE_PHRASES = [r"too early to say", r"not going to (guide|comment|speculate)",
                   r"we don't (comment|break out|provide)", r"as we('ve| have) said before",
                   r"get back to you", r"not (something )?we('re| are) prepared to"]

QA_MARKER = re.compile(r"question[- ]and[- ]answer|q\s*&\s*a session|\bq&a\b", re.I)

DEFAULT_LEXICON = {"pos": POSITIVE, "neg": NEGATIVE, "unc": UNCERTAIN}


def load_lm_dictionary(path):
    """Load the Loughran-McDonald Master Dictionary CSV into a lexicon dict."""
    d = pd.read_csv(path)
    d["Word"] = d["Word"].str.lower()
    return {"pos": set(d.loc[d["Positive"] > 0, "Word"]),
            "neg": set(d.loc[d["Negative"] > 0, "Word"]),
            "unc": set(d.loc[d["Uncertainty"] > 0, "Word"])}


def tokenize(text):
    return re.findall(r"[a-z']+", str(text).lower())


QA_TOKEN = "[[QA_START]]"   # inserted by make_dataset.py where the Q&A begins


def split_sections(text):
    """Split into (prepared remarks, Q&A). Q&A is '' if no marker is found."""
    text = str(text)
    if QA_TOKEN in text:
        prep, qa = text.split(QA_TOKEN, 1)
        return prep, qa
    m = QA_MARKER.search(text)
    if m and m.start() > 0.05 * len(text):
        return text[:m.start()], text[m.end():]
    return text, ""


def _phrase_rate(text, patterns, n_words):
    low = str(text).lower()
    hits = sum(len(re.findall(p, low)) for p in patterns)
    return 1000.0 * hits / max(n_words, 1)


def text_features(text, lex=None):
    lex = lex or DEFAULT_LEXICON
    prepared, qa = split_sections(text)
    out = {}
    for name, part in (("prep", prepared), ("qa", qa), ("all", str(text).replace(QA_TOKEN, " "))):
        toks = tokenize(part)
        n = len(toks)
        out[f"{name}_words"] = n
        if n == 0:
            out.update({f"{name}_{k}": np.nan for k in ("pos", "neg", "unc")})
            continue
        for key in ("pos", "neg", "unc"):
            out[f"{name}_{key}"] = sum(t in lex[key] for t in toks) / n
    qa_n = out["qa_words"]
    out["qa_hedge_rate"] = _phrase_rate(qa, HEDGE_PHRASES, qa_n) if qa_n else np.nan
    out["qa_evasive_rate"] = _phrase_rate(qa, EVASIVE_PHRASES, qa_n) if qa_n else np.nan
    return out


def build_features(df, lex=None):
    """Add engineered features to a DataFrame with a 'text' column."""
    feats = pd.DataFrame([text_features(t, lex) for t in df["text"]], index=df.index)
    feats["net_tone"] = feats["all_pos"] - feats["all_neg"]
    feats["net_tone_prep"] = feats["prep_pos"] - feats["prep_neg"]
    feats["net_tone_qa"] = feats["qa_pos"] - feats["qa_neg"]
    feats["tone_gap"] = feats["net_tone_prep"] - feats["net_tone_qa"]   # scripted vs. spontaneous
    feats["qa_uncertainty"] = feats["qa_unc"]
    feats["qa_share"] = feats["qa_words"] / feats["all_words"].clip(lower=1)
    feats["log_words"] = np.log1p(feats["all_words"])
    return pd.concat([df, feats], axis=1)


def finbert_scores(texts, model_name="ProsusAI/finbert", chunk_words=200,
                   max_chunks=40, batch_size=16):
    """Mean (P(positive) - P(negative)) over transcript chunks using FinBERT.

    Needs: pip install torch transformers. First call downloads ~440MB.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}
    pos_i = next(i for i, l in id2label.items() if l == "positive")
    neg_i = next(i for i, l in id2label.items() if l == "negative")
    scores = []
    for text in texts:
        words = str(text).split()
        chunks = [" ".join(words[i:i + chunk_words])
                  for i in range(0, len(words), chunk_words)][:max_chunks]
        vals = []
        for i in range(0, len(chunks), batch_size):
            enc = tok(chunks[i:i + batch_size], truncation=True, max_length=512,
                      padding=True, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)
            vals.extend((probs[:, pos_i] - probs[:, neg_i]).tolist())
        scores.append(float(np.mean(vals)) if vals else np.nan)
    return scores
