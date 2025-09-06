import reddit_fetcher

subreddit = input("Enter subreddit name:")
category = input("Please enter one of gilded, new, hot, top, rising, controversial: ")
limit = input("Number of posts: ")

if subreddit and category and limit:
    raw_data = reddit_fetcher.fetch_posts(subreddit, category, int(limit))
print(raw_data)
