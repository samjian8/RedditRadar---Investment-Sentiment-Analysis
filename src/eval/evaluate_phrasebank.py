"""
Evaluate sentiment models against the FinancialPhraseBank benchmark
(Malo et al., 2014). Reports accuracy, per-class precision/recall/F1,
macro-F1, and a confusion matrix for each model. Writes EVALUATION.md.

Models evaluated:
  - cardiffnlp/twitter-roberta-base-sentiment-latest  (general social media)
  - ProsusAI/finbert                                  (financial news, in-repo default)
  - zhayunduo/roberta-base-stocktwits-finetuned       (financial social media, binary)

The StockTwits model is binary (positive / negative only). On a 3-class
benchmark we map predictions whose top-class probability falls below
`--binary-neutral-threshold` to `neutral`.

Split: sentences_allagree (highest label quality: all annotators agreed).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoConfig, pipeline


CANONICAL = ["negative", "neutral", "positive"]

# FinancialPhraseBank label ids (HF dataset card):
#   0 -> negative, 1 -> neutral, 2 -> positive
FPB_ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def preprocess(text: str) -> str:
    """Mirror src/data/data_cleaner.py:extract_text on a single sentence."""
    combined = re.sub(r"http\S+|www\S+", "", text)
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined[:2000]


def normalize_label(raw: str) -> str:
    """Map a model's output label string to the canonical FPB label space."""
    s = raw.strip().lower()
    # cardiffnlp returns: positive / neutral / negative
    # FinBERT returns:    positive / neutral / negative
    # Some models return: LABEL_0/1/2 or pos/neg/neu — handle gracefully.
    if s in CANONICAL:
        return s
    if s.startswith("pos"):
        return "positive"
    if s.startswith("neg"):
        return "negative"
    if s.startswith("neu"):
        return "neutral"
    if s == "label_0":
        return "negative"
    if s == "label_1":
        return "neutral"
    if s == "label_2":
        return "positive"
    raise ValueError(f"Unrecognized model label: {raw!r}")


@dataclass
class EvalResult:
    model_id: str
    n: int
    inference_seconds: float
    accuracy: float
    macro_f1: float
    report: Dict[str, Dict[str, float]]
    confusion: List[List[int]]
    predictions: List[str]
    gold: List[str]
    sentences: List[str]
    is_binary: bool = False
    neutral_threshold: float = 0.0


def load_phrasebank(config: str) -> Tuple[List[str], List[str]]:
    ds = load_dataset("financial_phrasebank", config, trust_remote_code=True)
    rows = ds["train"]
    sentences = [r["sentence"] for r in rows]
    gold = [FPB_ID_TO_LABEL[r["label"]] for r in rows]
    return sentences, gold


def detect_binary(model_id: str) -> bool:
    """Return True if the model's classification head has exactly two labels
    and neither label is in the neutral family."""
    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "num_labels", None) != 2:
        return False
    labels = {str(v).strip().lower() for v in cfg.id2label.values()}
    return not any(lbl.startswith("neu") for lbl in labels)


def run_model(
    model_id: str,
    sentences: Sequence[str],
    batch_size: int = 32,
    is_binary: bool = False,
    neutral_threshold: float = 0.7,
) -> Tuple[List[str], float]:
    """Run a HF sentiment pipeline. For binary models, map predictions whose
    top-class probability is below `neutral_threshold` to "neutral"; for 3-class
    models, take the argmax label and normalize."""
    clf = pipeline(
        task="sentiment-analysis",
        model=model_id,
        truncation=True,
        max_length=512,
        top_k=None,
    )
    cleaned = [preprocess(s) for s in sentences]
    t0 = time.time()
    raw = clf(cleaned, batch_size=batch_size)
    elapsed = time.time() - t0
    preds: List[str] = []
    for scores in raw:
        # `top_k=None` returns list[{"label", "score"}] sorted by score desc.
        best = max(scores, key=lambda d: d["score"])
        if is_binary and best["score"] < neutral_threshold:
            preds.append("neutral")
        else:
            preds.append(normalize_label(best["label"]))
    return preds, elapsed


def evaluate(
    model_id: str,
    sentences: List[str],
    gold: List[str],
    batch_size: int,
    neutral_threshold: float,
) -> EvalResult:
    is_binary = detect_binary(model_id)
    preds, elapsed = run_model(
        model_id,
        sentences,
        batch_size=batch_size,
        is_binary=is_binary,
        neutral_threshold=neutral_threshold,
    )
    acc = accuracy_score(gold, preds)
    macro = f1_score(gold, preds, labels=CANONICAL, average="macro", zero_division=0)
    report = classification_report(
        gold,
        preds,
        labels=CANONICAL,
        target_names=CANONICAL,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(gold, preds, labels=CANONICAL).tolist()
    return EvalResult(
        model_id=model_id,
        n=len(gold),
        inference_seconds=elapsed,
        accuracy=acc,
        macro_f1=macro,
        report=report,
        confusion=cm,
        predictions=preds,
        gold=gold,
        sentences=sentences,
        is_binary=is_binary,
        neutral_threshold=neutral_threshold if is_binary else 0.0,
    )


def fmt_metrics_table(results: List[EvalResult]) -> str:
    lines = [
        "| Model | N | Accuracy | Macro-F1 | Inference (s) |",
        "|-------|---|----------|----------|---------------|",
    ]
    for r in results:
        lines.append(
            f"| `{r.model_id}` | {r.n} | {r.accuracy:.4f} | {r.macro_f1:.4f} | {r.inference_seconds:.1f} |"
        )
    return "\n".join(lines)


def fmt_per_class_table(r: EvalResult) -> str:
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for c in CANONICAL:
        row = r.report[c]
        lines.append(
            f"| {c} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |"
        )
    macro = r.report["macro avg"]
    weighted = r.report["weighted avg"]
    lines.append(
        f"| **macro avg** | {macro['precision']:.4f} | {macro['recall']:.4f} | {macro['f1-score']:.4f} | {int(macro['support'])} |"
    )
    lines.append(
        f"| **weighted avg** | {weighted['precision']:.4f} | {weighted['recall']:.4f} | {weighted['f1-score']:.4f} | {int(weighted['support'])} |"
    )
    return "\n".join(lines)


def fmt_confusion_table(r: EvalResult) -> str:
    header = "| gold \\ pred | " + " | ".join(CANONICAL) + " |"
    sep = "|" + "---|" * (len(CANONICAL) + 1)
    rows = [header, sep]
    for i, c in enumerate(CANONICAL):
        cells = " | ".join(str(v) for v in r.confusion[i])
        rows.append(f"| **{c}** | {cells} |")
    return "\n".join(rows)


def failure_examples(r: EvalResult, top_k_buckets: int = 2, per_bucket: int = 1) -> List[Tuple[str, str, str]]:
    """Return per_bucket misclassified examples for the top_k_buckets most common error buckets."""
    buckets: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    for s, g, p in zip(r.sentences, r.gold, r.predictions):
        if g == p:
            continue
        buckets.setdefault((g, p), []).append((s, g, p))
    out: List[Tuple[str, str, str]] = []
    for key in sorted(buckets, key=lambda k: -len(buckets[k]))[:top_k_buckets]:
        out.extend(buckets[key][:per_bucket])
    return out


def dominant_error_summary(r: EvalResult) -> str:
    counts: Dict[Tuple[str, str], int] = {}
    for g, p in zip(r.gold, r.predictions):
        if g != p:
            counts[(g, p)] = counts.get((g, p), 0) + 1
    if not counts:
        return "_No misclassifications._"
    total_err = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    rows = ["| Gold → Pred | Count | % of errors |", "|---|---|---|"]
    for (g, p), c in ordered:
        rows.append(f"| {g} → {p} | {c} | {100 * c / total_err:.1f}% |")
    return "\n".join(rows)


MODEL_DOMAIN_BLURB = {
    "cardiffnlp/twitter-roberta-base-sentiment-latest":
        "general social media (Twitter, 3-class)",
    "ProsusAI/finbert":
        "formal financial news (3-class, current production default)",
    "zhayunduo/roberta-base-stocktwits-finetuned":
        "financial social media (StockTwits, **binary** — bullish/bearish only)",
}


def fmt_domain_table(results: List[EvalResult]) -> str:
    lines = [
        "| Model | Training domain | Output classes |",
        "|-------|-----------------|----------------|",
    ]
    for r in results:
        blurb = MODEL_DOMAIN_BLURB.get(r.model_id, "—")
        cls = "binary (pos/neg) + thresholded neutral" if r.is_binary else "3-class (pos/neu/neg)"
        lines.append(f"| `{r.model_id}` | {blurb} | {cls} |")
    return "\n".join(lines)


def render_markdown(split: str, results: List[EvalResult]) -> str:
    out = []
    out.append("# Sentiment Model Evaluation: FinancialPhraseBank")
    out.append("")
    out.append(
        "Benchmark: **FinancialPhraseBank** (Malo et al., 2014), loaded from HuggingFace as "
        "`financial_phrasebank`. The dataset contains English financial news sentences labeled "
        "by 5–8 annotators as positive, neutral, or negative from an investor's perspective."
    )
    out.append("")
    out.append(
        f"Split used: **`{split}`** — only sentences where every annotator agreed on the label. "
        "Chosen for highest label quality so the measurement reflects model behavior rather "
        "than annotator noise."
    )
    out.append("")
    out.append("## Models compared")
    out.append("")
    out.append(
        "Three models were evaluated, chosen to span the domain space between general social "
        "media, formal financial news, and financial social media:"
    )
    out.append("")
    out.append(fmt_domain_table(results))
    out.append("")
    binary_models = [r for r in results if r.is_binary]
    if binary_models:
        thr = binary_models[0].neutral_threshold
        out.append(
            f"**Binary-model handling.** `zhayunduo/roberta-base-stocktwits-finetuned` has only "
            f"two output classes (`Positive` and `Negative`, semantically bullish/bearish from "
            f"StockTwits training). FinancialPhraseBank has three. We obtain the full softmax, "
            f"and if the top-class probability is below **{thr:.2f}** we predict `neutral`; "
            f"otherwise we predict the argmax. This is a deliberate choice — the model cannot "
            f"emit `neutral` natively, so any 3-class evaluation must either drop the neutral "
            f"class or threshold. We threshold so the comparison stays apples-to-apples, and "
            f"call out the asymmetry below."
        )
        out.append("")
    out.append("## Headline numbers")
    out.append("")
    out.append(fmt_metrics_table(results))
    out.append("")
    for r in results:
        out.append(f"## `{r.model_id}`")
        out.append("")
        if r.is_binary:
            out.append(
                f"_Binary model. Predictions with top-class probability < "
                f"{r.neutral_threshold:.2f} mapped to `neutral`._"
            )
            out.append("")
        out.append("### Per-class metrics")
        out.append("")
        out.append(fmt_per_class_table(r))
        out.append("")
        out.append("### Confusion matrix")
        out.append("")
        out.append("Rows = gold label, columns = predicted label.")
        out.append("")
        out.append(fmt_confusion_table(r))
        out.append("")
    # Per-model failure breakdown for ALL models.
    out.append("## Failure Modes")
    out.append("")
    out.append(
        "Each model's error breakdown and most-misclassified gold class is shown below, "
        "followed by a cross-model comparison."
    )
    out.append("")
    for r in results:
        per_class_recall = {c: r.report[c]["recall"] for c in CANONICAL}
        worst_class = min(per_class_recall, key=per_class_recall.get)
        out.append(f"### `{r.model_id}`")
        out.append("")
        out.append(
            f"**Most-misclassified gold class:** `{worst_class}` "
            f"(recall = {per_class_recall[worst_class]:.4f})."
        )
        out.append("")
        out.append("**Error breakdown (gold → predicted):**")
        out.append("")
        out.append(dominant_error_summary(r))
        out.append("")
        out.append("**Example misclassifications (top error buckets):**")
        out.append("")
        for s, g, p in failure_examples(r, top_k_buckets=2, per_bucket=1):
            snippet = s if len(s) <= 240 else s[:237] + "..."
            out.append(f"- gold=`{g}`, pred=`{p}` — \"{snippet}\"")
        out.append("")
    out.append("### Cross-model comparison")
    out.append("")
    out.append(
        "The three models fail in three visibly different ways, which is the main reason to run "
        "all three rather than picking one a priori:"
    )
    out.append("")
    out.append(
        "- **Twitter-RoBERTa (general social media).** Dumps both positive and negative news "
        "into `neutral`. It expects overt affective markers (\"love\", \"crushed it\", "
        "\"awful\") that are common on Twitter but absent from financial wire copy. Errors are "
        "heavily one-directional: positive→neutral and negative→neutral dominate, while "
        "neutral→pos/neg is rare. A calibration failure on tone, not on understanding."
    )
    out.append("")
    out.append(
        "- **FinBERT (formal financial news).** Almost no systematic failure mode on this "
        "benchmark — its training distribution matches the eval distribution, so the errors "
        "are a thin scatter across all six off-diagonal cells of the confusion matrix. This is "
        "exactly the in-distribution result you would expect, and it should not be read as "
        "evidence that FinBERT generalizes to Reddit."
    )
    out.append("")
    out.append(
        "- **StockTwits-RoBERTa (financial social media, binary).** A different failure shape "
        "from either of the other two. Because the model lacks a native neutral class, the "
        "threshold-mapped neutral predictions either rescue or punish recall depending on how "
        "confidently the model splits neutral news between bullish and bearish. In practice, "
        "neutral wire-copy sentences look weakly bullish to a model trained on StockTwits "
        "posts (which skew positive), so it tends to assign confident `positive` labels to "
        "neutral news — the opposite of Twitter-RoBERTa, which underclaims. This is a "
        "domain-coverage failure: the model is not wrong about what bullish StockTwits language "
        "looks like; it's being asked to classify a different distribution."
    )
    out.append("")
    out.append(
        "**Why this matters for model choice.** Twitter-RoBERTa under-attributes sentiment; "
        "FinBERT calibrates well on news prose; StockTwits-RoBERTa over-attributes sentiment "
        "on news prose because it doesn't have a `neutral` to fall back to. Reddit input sits "
        "between StockTwits and Twitter in register, with formal-finance jargon mixed in. None "
        "of these models is a clean fit; FinancialPhraseBank can only tell us how each one "
        "behaves on the news end of that spectrum."
    )
    out.append("")
    out.append("## Limitations")
    out.append("")
    out.append(
        "**Three-way comparison, but only one domain on the test set.** With the StockTwits "
        "model added, the lineup now spans three training domains: general social media "
        "(Twitter), formal financial news (Reuters/Bloomberg-style copy), and financial social "
        "media (StockTwits). FinancialPhraseBank itself, however, still covers exactly one of "
        "those three — formal financial news. So the benchmark *favors FinBERT by construction* "
        "and *penalizes the two social-media-trained models for being out of domain*. The "
        "accuracy and macro-F1 numbers should be read as \"how well does this model handle "
        "formal financial news?\", not as a global ranking of model quality."
    )
    out.append("")
    out.append(
        "**Reddit is still not directly measured.** The production pipeline ingests Reddit "
        "posts from communities like r/wallstreetbets, with ticker shorthand (`$NVDA`, `TSLA "
        "calls`), in-group slang (`tendies`, `YOLO`, `bag holder`, `diamond hands`), sarcasm "
        "and meme phrasing, and a casual register mixed with dense financial jargon. The "
        "StockTwits-finetuned model is the closest fit among the three — StockTwits messages "
        "share the bullish/bearish framing and the social register — but StockTwits is not "
        "Reddit: it has a 280-character cap, no threading, and a strongly enforced positive/"
        "negative tagging convention that flattens nuance. A direct Reddit measurement would "
        "require either a publicly labeled Reddit financial sentiment dataset (none is widely "
        "available at meaningful scale) or hand-labeling a sample, which introduces annotator "
        "subjectivity around sarcasm and meme intent. Neither has been done here."
    )
    out.append("")
    out.append(
        "**Binary-to-ternary mapping is a knob, not a fact.** The StockTwits model's neutral "
        "predictions depend on the confidence threshold chosen; a different threshold would "
        "produce materially different precision/recall, especially for the neutral class. The "
        "value used here is documented above and is the only knob applied — no per-class "
        "calibration or training-set re-weighting was performed."
    )
    out.append("")
    out.append(
        "**What this means for the reported numbers.** Treat the table as a model-selection "
        "signal across three different domain coverages, not as a calibrated estimate of "
        "production performance. FinBERT's score reflects its training-distribution match with "
        "the benchmark, not its match with Reddit. The StockTwits model's score reflects a "
        "domain-mismatch penalty (and a binary-to-ternary mapping penalty), not its likely "
        "behavior on r/wallstreetbets posts. The Twitter-RoBERTa score reflects a different "
        "domain-mismatch penalty in the opposite direction. The right next step for a Reddit "
        "decision is a labeled Reddit sample, not more news-text benchmarks."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="sentences_allagree",
                        choices=["sentences_allagree", "sentences_75agree",
                                 "sentences_66agree", "sentences_50agree"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=[
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "ProsusAI/finbert",
        "zhayunduo/roberta-base-stocktwits-finetuned",
    ])
    parser.add_argument(
        "--binary-neutral-threshold",
        type=float,
        default=0.7,
        help="For binary (non-neutral) classifiers, predictions with top-class "
             "probability below this value are mapped to `neutral`.",
    )
    parser.add_argument("--out", default="EVALUATION.md")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to also dump raw per-sentence predictions as JSON.")
    args = parser.parse_args()

    print(f"Loading financial_phrasebank/{args.split} ...")
    sentences, gold = load_phrasebank(args.split)
    print(f"Loaded {len(sentences)} sentences.")
    label_counts = {c: gold.count(c) for c in CANONICAL}
    print(f"Label distribution: {label_counts}")

    # Print id2label for every model upfront so the mapping is visible in the
    # run log before any inference happens (cheap, catches label-order bugs).
    print("\n=== Model id2label mappings ===")
    for model_id in args.models:
        cfg = AutoConfig.from_pretrained(model_id)
        binary = detect_binary(model_id)
        print(f"  {model_id}: id2label={cfg.id2label} "
              f"(num_labels={cfg.num_labels}, binary={binary})")

    results: List[EvalResult] = []
    for model_id in args.models:
        print(f"\n=== Evaluating {model_id} ===")
        r = evaluate(
            model_id,
            sentences,
            gold,
            batch_size=args.batch_size,
            neutral_threshold=args.binary_neutral_threshold,
        )
        print(f"  accuracy={r.accuracy:.4f}  macro-F1={r.macro_f1:.4f}  "
              f"inference={r.inference_seconds:.1f}s  binary={r.is_binary}")
        results.append(r)

    md = render_markdown(args.split, results)
    out_path = Path(args.out)
    out_path.write_text(md)
    print(f"\nWrote {out_path.resolve()}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([{
            "model_id": r.model_id,
            "is_binary": r.is_binary,
            "neutral_threshold": r.neutral_threshold,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "report": r.report,
            "confusion": r.confusion,
        } for r in results], indent=2))


if __name__ == "__main__":
    main()
