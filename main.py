import reddit_fetcher
import data_cleaner
import model_analysis

subreddit = input("Enter subreddit name:")
category = input("Please enter one of gilded, new, hot, top, rising, controversial: ")
limit = input("Number of posts: ")

if subreddit and category and limit:
    raw_data = reddit_fetcher.fetch_posts(subreddit, category, int(limit))

clean_posts = data_cleaner.extract_text(raw_data) # Clean posts have title and selftext appended together

texts_with_sentiment = model_analysis.analyze_posts(clean_posts)
print(texts_with_sentiment)
