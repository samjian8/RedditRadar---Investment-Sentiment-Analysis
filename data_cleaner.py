import re

def extract_text(posts):

    texts = []
    for post in posts:
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        
        combined = f"{title} {selftext}".strip() # Combine and basic cleanup
        combined = re.sub(r'http\S+|www\S+', '', combined) # Remove URLs
        combined = re.sub(r'\s+', ' ', combined).strip() # Normalize whitespace
        
        texts.append(combined)
    
    return texts



