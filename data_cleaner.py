from typing import List, Dict, Any
import re
import emoji

def extract_text(posts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract and clean text content from Reddit posts for sentiment analysis.
    
    Combines title and selftext, removes URLs, normalizes whitespace, and applies
    character limits to prepare text for transformer-based sentiment analysis.
    
    Args:
        posts: List of Reddit post dictionaries containing 'title' and 'selftext' fields
               Expected format: [{'title': str, 'selftext': str, 'url': str, ...}, ...]
               Missing fields are handled gracefully with empty string defaults
    
    Returns:
        List of dictionaries containing cleaned text and metadata
        Each dictionary has keys:
        - 'title': Original post title
        - 'url': Original post URL
        - 'text': Combined and cleaned title + selftext (max 2000 chars)
        
    Example:
        >>> posts = [
        ...     {'title': 'GME to moon!', 'selftext': 'This stock is amazing   https://example.com', 'url': 'reddit.com/...'}
        ... ]
        >>> extract_text(posts)
        [{'title': 'GME to moon!', 'url': 'reddit.com/...', 'text': 'GME to moon! This stock is amazing'}]
    """

    submissions = []
    for post in posts:
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        url = post.get('url', '')

        combined = f"{title} {selftext}".strip() # Combine and basic cleanup
        combined = re.sub(r'http\S+|www\S+', '', combined) # Remove URLs
        combined = re.sub(r'\s+', ' ', combined).strip() # Normalize whitespace
        
        # Adding length limit to text (roughly 500 tokens = 2000 characters)
        combined = combined[:2000]

        submissions.append(
            {
                "title": title,
                "url": url,
                "text": combined
            }
        )
    
    return submissions



