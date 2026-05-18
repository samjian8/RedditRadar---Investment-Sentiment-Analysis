## Reddit Subreddit Sentiment Analyzer

Fetch recent subreddit posts, run a transformer sentiment model, and summarize results (positive / negative / neutral) plus links to highlighted posts. Provides a CLI and a small Flask web endpoint/UI.

## Inspiration
The goal was to summarize community sentiment around financial instruments (stocks, crypto, etc.) without manually reading large volumes of posts. Subreddits provide focused discussion, so this tool aggregates posts from a chosen subreddit (with desired filters) and produces sentiment breakdowns with quantifiable confidence levels for less subjectivity.

## Features (Brief)
- Fetch posts by category (hot, new, top, rising, controversial, gilded)
- Clean & combine title + body (URL removal, whitespace normalization, length cap)
- Batch sentiment with `ProsusAI/finbert` (selected via the benchmark below)
- Two overall signals: weighted score + percentage distribution
- Highlights: top positive / negative posts by confidence

## Structure
```
cli.py                       # Interactive CLI entrypoint
src/
  data/
    reddit_fetcher.py        # PRAW integration
    data_cleaner.py          # Text normalization
  models/
    model_analysis.py        # Sentiment pipeline
  utils/
    sentiment_utils.py       # Aggregation logic
web/
  app.py                     # Flask web app & /analyze endpoint
  templates/
    index.html               # HTML template
```

## Setup
1. Clone repository
2. (Optional) create venv
3. Install dependencies
4. Add `.env` with Reddit credentials

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` example available as "example.env"

## Usage
CLI:
```bash
python cli.py
```

Web server (run from project root):
```bash
python -m web.app
# visit http://127.0.0.1:5000
```

## Sentiment Method (Short)
- Model returns label + probability (positive / neutral / negative)
- Weighted overall score = (sum positive confidences − sum negative confidences)/N
  - > +0.15 => Bullish, < -0.15 => Bearish, else Mixed
- Percentage method compares positive vs negative share

## Benchmark Evaluation
The sentiment pipeline was evaluated against the **FinancialPhraseBank** (Malo et al., 2014) `sentences_allagree` split (N=2264) using the in-repo preprocessing. Three models were compared, picked to span different training domains relative to the Reddit production input:

| Model | Training domain | Accuracy | Macro-F1 |
|-------|-----------------|----------|----------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | general social media | 0.7147 | 0.5750 |
| **`ProsusAI/finbert`** (current default) | formal financial news | **0.9717** | **0.9625** |
| `zhayunduo/roberta-base-stocktwits-finetuned` | financial social media (binary) | 0.3847 | 0.4006 |

Each model fails differently. The Twitter-trained model collapses understated finance phrasing into `neutral` (under-claims sentiment). FinBERT is in-distribution on this benchmark and rarely errs. The StockTwits-finetuned model is binary (`Positive` / `Negative`); after threshold-mapping low-confidence predictions to neutral, it over-claims sentiment on neutral news prose — the mirror image of the Twitter model. FinancialPhraseBank tests only formal financial news, so it favors FinBERT by construction and does not settle which model is best for Reddit. **FinBERT remains the production default** because it is the only model of the three that this benchmark can fairly evaluate, and switching to a model that scores far worse on the one signal available is hard to justify without a Reddit-labeled measurement (may hand-label a dataset in the future for further testing and tuning).

Reproduce with `python -m src.eval.evaluate_phrasebank`. Full per-class metrics, confusion matrices, failure-mode analysis, and limitations: [EVALUATION.md](EVALUATION.md).

## Troubleshooting
| Problem | Suggestion |
| ------- | ---------- |
| Empty list | Check subreddit name & category |
| Auth errors | Verify `.env` values |
| Slow first run | Model download is occurring |
| Memory issues | Lower `limit` or use smaller model |

## Next Steps (Optional Ideas)
- Add tests
- Dockerize
- Comment sentiment
- Hand-label Reddit dataset
- Model tuning

## Disclaimer
Sentiment output is approximate. Not financial advice.


