from typing import List, Dict, Optional, Literal, Tuple
import re
import logging

# For preprocessing social media text
import emoji

# Hugging Face transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_posts(clean_posts):
    sentiments = classifier(clean_posts)
    return [
        {
            "text": text,
            "sentiment_label": sent["label"],
            "sentiment_score": sent["score"]
        }
        for text, sent in zip(clean_posts, sentiments)
    ]

