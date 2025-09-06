import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

def fetch_posts(subreddit: str, category: str, limit=10):
    # Category can be one of the following: new, hot, top, rising, controversial, gilded
    submissions = []
    for submission in getattr(reddit.subreddit(subreddit), category)(limit=limit):
        submissions.append({
            "title": submission.title,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_utc": submission.created_utc,
            "url": submission.url,
            "upvote_ratio": submission.upvote_ratio,
            "selftext": submission.selftext
        })
    return submissions

def test_connection():
    submissions = fetch_posts("test", "new", 10)
    for submission in submissions:
        title = submission["title"]
        selftext = submission["selftext"]
        print(title)
        print(selftext)
        print("-" * 100)

# Test it
test_connection()