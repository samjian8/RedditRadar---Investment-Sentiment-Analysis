## Reddit Subreddit Sentiment Analyzer

Fetch recent subreddit posts, run a transformer sentiment model, and summarize results (positive / negative / neutral) plus links to highlighted posts. Provides a CLI and a small Flask web endpoint/UI.

## Inspiration
The goal was to summarize community sentiment around financial instruments (stocks, crypto, etc.) without manually reading large volumes of posts. Subreddits provide focused discussion, so this tool aggregates posts from a chosen subreddit (with desired filters) and produces sentiment breakdowns with quantifiable confidence levels for less subjectivity.

## Features (Brief)
- Fetch posts by category (hot, new, top, rising, controversial, gilded)
- Clean & combine title + body (URL removal, whitespace normalization, length cap)
- Batch sentiment with `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Two overall signals: weighted score + percentage distribution
- Highlights: top positive / negative posts by confidence

## Structure
```
app.py            # Flask web app & /analyze endpoint
cli.py            # Interactive CLI
reddit_fetcher.py # PRAW integration
data_cleaner.py   # Text normalization
model_analysis.py # Sentiment pipeline
sentiment_utils.py# Aggregation logic
templates/        # HTML template
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

`.env` example:
```
CLIENT_ID=...
CLIENT_SECRET=...
USER_AGENT=reddit-sentiment-analyzer/0.1 by u/<username>
```

## Usage
CLI:
```bash
python cli.py
```

Web server:
```bash
python app.py
# visit http://127.0.0.1:5000
```

## Sentiment Method (Short)
- Model returns label + probability (positive / neutral / negative)
- Weighted overall score = (sum positive confidences − sum negative confidences)/N
  - > +0.15 => Bullish, < -0.15 => Bearish, else Mixed
- Percentage method compares positive vs negative share

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
- Model tuning

## Disclaimer
Sentiment output is approximate. Not financial advice.


