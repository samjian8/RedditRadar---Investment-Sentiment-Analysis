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

def analyze_posts(clean_posts: List[Dict[str, str]]) -> List[Dict[str, Union[str, float]]]:
    """
    Analyze sentiment of cleaned Reddit posts using a pre-trained transformer model.
    
    Uses batch processing for efficiency with cardiffnlp/twitter-roberta-base-sentiment-latest
    model, which is optimized for social media text.
    
    Args:
        clean_posts: List of dictionaries with 'text', 'title', 'url' keys
        Each dict should contain cleaned text ready for analysis
                    
    Returns:
        List of dictionaries containing post metadata and sentiment analysis results
        Each dict has keys: 'url', 'title', 'text', 'sentiment_label', 'sentiment_score'
        
    Example:
        >>> posts = [{'title': 'GME rocks!', 'url': 'reddit.com/...', 'text': 'GME rocks! Great stock'}]
        >>> analyze_posts(posts)
        [
            {
                'title': 'GME rocks!',
                'url': 'reddit.com/...',
                'text': 'GME rocks! Great stock',
                'sentiment_label': 'postive', 
                'sentiment_score': 0.9998
            }
        ]
    """
    # Extract just the text for sentiment analysis
    texts = [post["text"] for post in clean_posts]
    sentiments = classifier(texts)  # Batch process all texts at once
    
    # Combine original post data with sentiment results
    return [
        {
            "title": post["title"],
            "url": post["url"],
            "text": post["text"],
            "sentiment_label": sent["label"],
            "sentiment_score": sent["score"]
        }
        for post, sent in zip(clean_posts, sentiments)
    ]