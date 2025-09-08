from typing import List, Dict, Optional, Literal, Tuple, Union
import re
import logging

# Hugging Face transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline(
    task="sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512
)

def analyze_posts(clean_posts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """
    Analyze sentiment of cleaned Reddit posts using a pre-trained transformer model.
    
    Uses batch processing for efficiency with cardiffnlp/twitter-roberta-base-sentiment-latest
    model, which is optimized for social media text.
    
    Args:
        clean_posts: List of cleaned text strings from Reddit posts
        Each string should be title + selftext combined and normalized
                    
    Returns:
        List of dictionaries containing original text and sentiment analysis results
        Each dict has keys: 'text', 'sentiment_label', 'sentiment_score'
        
    Example:
        >>> texts = ["I love this stock!", "This is terrible"]
        >>> analyze_posts(texts)
        [
            {
                'text': 'I love this stock!',
                'sentiment_label': 'LABEL_2', 
                'sentiment_score': 0.9998
            },
            {
                'text': 'This is terrible',
                'sentiment_label': 'LABEL_0',
                'sentiment_score': 0.9991
            }
        ]
    """
    sentiments = classifier(clean_posts)  # Batch process all texts at once
    return [
        {
            "text": text,
            "sentiment_label": sent["label"],  # LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
            "sentiment_score": sent["score"]   # Confidence score (0.0 to 1.0)
        }
        for text, sent in zip(clean_posts, sentiments)
    ]