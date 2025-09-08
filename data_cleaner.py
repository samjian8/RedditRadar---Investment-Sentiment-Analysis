from typing import List, Dict, Any
import re
import emoji

def extract_text(posts: List[Dict[str, Any]]) -> List[str]:
    """
    Extract and clean text content from Reddit posts for sentiment analysis.
    
    Args:
        posts: List of Reddit post dictionaries containing 'title' and 'selftext' fields
               Expected format: [{'title': str, 'selftext': str, ...}, ...]
    
    Returns:
        List of cleaned text strings combining title and selftext
        Each string is limited to 2000 characters and has URLs/whitespace normalized
        
    Example:
        >>> posts = [{'title': 'GME to moon!', 'selftext': 'This stock...'}]
        >>> extract_text(posts)
        ['GME to moon! This stock...']
    """
    
    texts = []
    for post in posts:
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        
        combined = f"{title} {selftext}".strip() # Combine and basic cleanup
        combined = re.sub(r'http\S+|www\S+', '', combined) # Remove URLs
        combined = re.sub(r'\s+', ' ', combined).strip() # Normalize whitespace
        
        # Adding length limit to text (roughly 500 tokens = 2000 characters)
        combined = combined[:2000]

        texts.append(combined)
    
    return texts



