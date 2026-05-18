# Note that the sentiment score is a measure of how confident the model is in the given rating

def get_most_positive_posts(posts, top_n = 5):
    postive_posts = [post for post in posts if post['sentiment_label'].lower() == 'positive']
    top_n = min(top_n, len(postive_posts))
    sorted_posts = sorted(postive_posts, key=lambda x: x['sentiment_score'], reverse=True)
    return sorted_posts[:top_n]

def get_most_negative_posts(posts, top_n = 5):
    negative_posts = [post for post in posts if post['sentiment_label'].lower() == 'negative']
    top_n = min(top_n, len(negative_posts))
    sorted_posts = sorted(negative_posts, key=lambda x: x['sentiment_score'], reverse=True)
    return sorted_posts[:top_n]

def calculate_overall_sentiment_weighted(posts):
    """Calculate overall sentiment weighted by confidence scores"""
    if not posts:
        return "Unknown", 0.0
    
    total_score = 0
    for post in posts:
        if post['sentiment_label'].lower() == 'positive':
            total_score += post['sentiment_score']
        elif post['sentiment_label'].lower() == 'negative':
            total_score -= post['sentiment_score']
        # neutral posts contribute 0
    
    avg_score = total_score / len(posts)
    
    if avg_score > 0.15:
        return "Bullish", avg_score
    elif avg_score < -0.15:
        return "Bearish", avg_score
    else:
        return "Mixed", avg_score

def calculate_overall_sentiment_percentage(posts):
    """Calculate overall sentiment based on percentages"""
    if not posts:
        return {"sentiment": "Unknown", "positive_percentage": 0, "negative_percentage": 0, "neutral_percentage": 0}
    
    total_posts = len(posts)
    positive_count = len([p for p in posts if p['sentiment_label'].lower() == 'positive'])
    negative_count = len([p for p in posts if p['sentiment_label'].lower() == 'negative'])
    neutral_count = len([p for p in posts if p['sentiment_label'].lower() == 'neutral'])
    
    positive_pct = (positive_count / total_posts) * 100
    negative_pct = (negative_count / total_posts) * 100
    neutral_pct = (neutral_count / total_posts) * 100
    
    overall = "Bullish" if positive_pct > negative_pct + 10 else "Bearish" if negative_pct > positive_pct + 10 else "Mixed"
    
    return {
        'sentiment': overall,
        'positive_percentage': round(positive_pct, 1),
        'negative_percentage': round(negative_pct, 1),
        'neutral_percentage': round(neutral_pct, 1)
    }