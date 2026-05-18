# Sentiment Model Evaluation: FinancialPhraseBank

Benchmark: **FinancialPhraseBank** (Malo et al., 2014), loaded from HuggingFace as `financial_phrasebank`. The dataset contains English financial news sentences labeled by 5–8 annotators as positive, neutral, or negative from an investor's perspective.

Split used: **`sentences_allagree`** — only sentences where every annotator agreed on the label. Chosen for highest label quality so the measurement reflects model behavior rather than annotator noise.

## Models compared

Three models were evaluated, chosen to span the domain space between general social media, formal financial news, and financial social media:

| Model | Training domain | Output classes |
|-------|-----------------|----------------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | general social media (Twitter, 3-class) | 3-class (pos/neu/neg) |
| `ProsusAI/finbert` | formal financial news (3-class, current production default) | 3-class (pos/neu/neg) |
| `zhayunduo/roberta-base-stocktwits-finetuned` | financial social media (StockTwits, **binary** — bullish/bearish only) | binary (pos/neg) + thresholded neutral |

**Binary-model handling.** `zhayunduo/roberta-base-stocktwits-finetuned` has only two output classes (`Positive` and `Negative`, semantically bullish/bearish from StockTwits training). FinancialPhraseBank has three. We obtain the full softmax, and if the top-class probability is below **0.70** we predict `neutral`; otherwise we predict the argmax. This is a deliberate choice — the model cannot emit `neutral` natively, so any 3-class evaluation must either drop the neutral class or threshold. We threshold so the comparison stays apples-to-apples, and call out the asymmetry below.

## Headline numbers

| Model | N | Accuracy | Macro-F1 | Inference (s) |
|-------|---|----------|----------|---------------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 2264 | 0.7147 | 0.5750 | 16.1 |
| `ProsusAI/finbert` | 2264 | 0.9717 | 0.9625 | 15.4 |
| `zhayunduo/roberta-base-stocktwits-finetuned` | 2264 | 0.3847 | 0.4006 | 14.9 |

## `cardiffnlp/twitter-roberta-base-sentiment-latest`

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| negative | 0.8889 | 0.3432 | 0.4952 | 303 |
| neutral | 0.6903 | 0.9741 | 0.8080 | 1391 |
| positive | 0.8641 | 0.2789 | 0.4218 | 570 |
| **macro avg** | 0.8144 | 0.5321 | 0.5750 | 2264 |
| **weighted avg** | 0.7606 | 0.7147 | 0.6689 | 2264 |

### Confusion matrix

Rows = gold label, columns = predicted label.

| gold \ pred | negative | neutral | positive |
|---|---|---|---|
| **negative** | 104 | 199 | 0 |
| **neutral** | 11 | 1355 | 25 |
| **positive** | 2 | 409 | 159 |

## `ProsusAI/finbert`

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| negative | 0.9058 | 0.9835 | 0.9430 | 303 |
| neutral | 0.9985 | 0.9669 | 0.9825 | 1391 |
| positive | 0.9473 | 0.9772 | 0.9620 | 570 |
| **macro avg** | 0.9505 | 0.9759 | 0.9625 | 2264 |
| **weighted avg** | 0.9732 | 0.9717 | 0.9720 | 2264 |

### Confusion matrix

Rows = gold label, columns = predicted label.

| gold \ pred | negative | neutral | positive |
|---|---|---|---|
| **negative** | 298 | 1 | 4 |
| **neutral** | 19 | 1345 | 27 |
| **positive** | 12 | 1 | 557 |

## `zhayunduo/roberta-base-stocktwits-finetuned`

_Binary model. Predictions with top-class probability < 0.70 mapped to `neutral`._

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| negative | 0.4666 | 0.8977 | 0.6140 | 303 |
| neutral | 0.7755 | 0.0546 | 0.1021 | 1391 |
| positive | 0.3304 | 0.9175 | 0.4858 | 570 |
| **macro avg** | 0.5241 | 0.6233 | 0.4006 | 2264 |
| **weighted avg** | 0.6221 | 0.3847 | 0.2672 | 2264 |

### Confusion matrix

Rows = gold label, columns = predicted label.

| gold \ pred | negative | neutral | positive |
|---|---|---|---|
| **negative** | 272 | 13 | 18 |
| **neutral** | 273 | 76 | 1042 |
| **positive** | 38 | 9 | 523 |

## Failure Modes

Each model's error breakdown and most-misclassified gold class is shown below, followed by a cross-model comparison.

### `cardiffnlp/twitter-roberta-base-sentiment-latest`

**Most-misclassified gold class:** `positive` (recall = 0.2789).

**Error breakdown (gold → predicted):**

| Gold → Pred | Count | % of errors |
|---|---|---|
| positive → neutral | 409 | 63.3% |
| negative → neutral | 199 | 30.8% |
| neutral → positive | 25 | 3.9% |
| neutral → negative | 11 | 1.7% |
| positive → negative | 2 | 0.3% |

**Example misclassifications (top error buckets):**

- gold=`positive`, pred=`neutral` — "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m ."
- gold=`negative`, pred=`neutral` — "2009 3 February 2010 - Finland-based steel maker Rautaruukki Oyj ( HEL : RTRKS ) , or Ruukki , said today it slipped to a larger-than-expected pretax loss of EUR46m in the fourth quarter of 2009 from a year-earlier profit of EUR45m ."

### `ProsusAI/finbert`

**Most-misclassified gold class:** `neutral` (recall = 0.9669).

**Error breakdown (gold → predicted):**

| Gold → Pred | Count | % of errors |
|---|---|---|
| neutral → positive | 27 | 42.2% |
| neutral → negative | 19 | 29.7% |
| positive → negative | 12 | 18.8% |
| negative → positive | 4 | 6.2% |
| positive → neutral | 1 | 1.6% |
| negative → neutral | 1 | 1.6% |

**Example misclassifications (top error buckets):**

- gold=`neutral`, pred=`positive` — "STOCK EXCHANGE ANNOUNCEMENT 20 July 2006 1 ( 1 ) BASWARE SHARE SUBSCRIPTIONS WITH WARRANTS AND INCREASE IN SHARE CAPITAL A total of 119 850 shares have been subscribed with BasWare Warrant Program ."
- gold=`neutral`, pred=`negative` — "The broad-based WIG index ended Thursday 's session 0.1 pct up at 65,003.34 pts , while the blue-chip WIG20 was 1.13 down at 3,687.15 pts ."

### `zhayunduo/roberta-base-stocktwits-finetuned`

**Most-misclassified gold class:** `neutral` (recall = 0.0546).

**Error breakdown (gold → predicted):**

| Gold → Pred | Count | % of errors |
|---|---|---|
| neutral → positive | 1042 | 74.8% |
| neutral → negative | 273 | 19.6% |
| positive → negative | 38 | 2.7% |
| negative → positive | 18 | 1.3% |
| negative → neutral | 13 | 0.9% |
| positive → neutral | 9 | 0.6% |

**Example misclassifications (top error buckets):**

- gold=`neutral`, pred=`positive` — "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing ."
- gold=`neutral`, pred=`negative` — "It has some 30 offices worldwide and more than 90 pct of its net sales are generated outside Finland ."

### Cross-model comparison

The three models fail in three visibly different ways, which is the main reason to run all three rather than picking one a priori:

- **Twitter-RoBERTa (general social media).** Dumps both positive and negative news into `neutral`. It expects overt affective markers ("love", "crushed it", "awful") that are common on Twitter but absent from financial wire copy. Errors are heavily one-directional: positive→neutral and negative→neutral dominate, while neutral→pos/neg is rare. A calibration failure on tone, not on understanding.

- **FinBERT (formal financial news).** Almost no systematic failure mode on this benchmark — its training distribution matches the eval distribution, so the errors are a thin scatter across all six off-diagonal cells of the confusion matrix. This is exactly the in-distribution result you would expect, and it should not be read as evidence that FinBERT generalizes to Reddit.

- **StockTwits-RoBERTa (financial social media, binary).** A different failure shape from either of the other two. Because the model lacks a native neutral class, the threshold-mapped neutral predictions either rescue or punish recall depending on how confidently the model splits neutral news between bullish and bearish. In practice, neutral wire-copy sentences look weakly bullish to a model trained on StockTwits posts (which skew positive), so it tends to assign confident `positive` labels to neutral news — the opposite of Twitter-RoBERTa, which underclaims. This is a domain-coverage failure: the model is not wrong about what bullish StockTwits language looks like; it's being asked to classify a different distribution.

**Why this matters for model choice.** Twitter-RoBERTa under-attributes sentiment; FinBERT calibrates well on news prose; StockTwits-RoBERTa over-attributes sentiment on news prose because it doesn't have a `neutral` to fall back to. Reddit input sits between StockTwits and Twitter in register, with formal-finance jargon mixed in. None of these models is a clean fit; FinancialPhraseBank can only tell us how each one behaves on the news end of that spectrum.

## Limitations

**Three-way comparison, but only one domain on the test set.** With the StockTwits model added, the lineup now spans three training domains: general social media (Twitter), formal financial news (Reuters/Bloomberg-style copy), and financial social media (StockTwits). FinancialPhraseBank itself, however, still covers exactly one of those three — formal financial news. So the benchmark *favors FinBERT by construction* and *penalizes the two social-media-trained models for being out of domain*. The accuracy and macro-F1 numbers should be read as "how well does this model handle formal financial news?", not as a global ranking of model quality.

**Reddit is still not directly measured.** The production pipeline ingests Reddit posts from communities like r/wallstreetbets, with ticker shorthand (`$NVDA`, `TSLA calls`), in-group slang (`tendies`, `YOLO`, `bag holder`, `diamond hands`), sarcasm and meme phrasing, and a casual register mixed with dense financial jargon. The StockTwits-finetuned model is the closest fit among the three — StockTwits messages share the bullish/bearish framing and the social register — but StockTwits is not Reddit: it has a 280-character cap, no threading, and a strongly enforced positive/negative tagging convention that flattens nuance. A direct Reddit measurement would require either a publicly labeled Reddit financial sentiment dataset (none is widely available at meaningful scale) or hand-labeling a sample, which introduces annotator subjectivity around sarcasm and meme intent. Neither has been done here.

**Binary-to-ternary mapping is a knob, not a fact.** The StockTwits model's neutral predictions depend on the confidence threshold chosen; a different threshold would produce materially different precision/recall, especially for the neutral class. The value used here is documented above and is the only knob applied — no per-class calibration or training-set re-weighting was performed.

**What this means for the reported numbers.** Treat the table as a model-selection signal across three different domain coverages, not as a calibrated estimate of production performance. FinBERT's score reflects its training-distribution match with the benchmark, not its match with Reddit. The StockTwits model's score reflects a domain-mismatch penalty (and a binary-to-ternary mapping penalty), not its likely behavior on r/wallstreetbets posts. The Twitter-RoBERTa score reflects a different domain-mismatch penalty in the opposite direction. The right next step for a Reddit decision is a labeled Reddit sample, not more news-text benchmarks.
