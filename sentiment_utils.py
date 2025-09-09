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