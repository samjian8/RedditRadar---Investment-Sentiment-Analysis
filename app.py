from flask import Flask, render_template, request, jsonify
import reddit_fetcher
import data_cleaner 
import model_analysis
import sentiment_utils

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/analyze', methods=['POST'])
def analyze():
    subreddit = request.form.get('subreddit')
    category = request.form.get('category')
    limit = request.form.get('limit', type=int)

    if not subreddit or not category or not limit:
        return jsonify({"Error": "Please provide subreddit, category, and limit"}), 400

    try:
        raw_data = reddit_fetcher.fetch_posts(subreddit, category, limit)
        clean_posts = data_cleaner.extract_text(raw_data)
        texts_with_sentiment = model_analysis.analyze_posts(clean_posts)

        # Get most confidently positive and negative posts
        most_positive = sentiment_utils.get_most_positive_posts(texts_with_sentiment, top_n=5)
        most_negative = sentiment_utils.get_most_negative_posts(texts_with_sentiment, top_n=5)

        # Create summary stats
        positive_count = len([post for post in texts_with_sentiment if post['sentiment_label'].lower() == 'positive'])
        negative_count = len([post for post in texts_with_sentiment if post['sentiment_label'].lower() == 'negative'])
        neutral_count = len([post for post in texts_with_sentiment if post['sentiment_label'].lower() == 'neutral'])

        # Calculate overall sentiment (two methods)
        overall_sentiment_weighted = sentiment_utils.calculate_overall_sentiment_weighted(texts_with_sentiment)
        overall_sentiment_percentage = sentiment_utils.calculate_overall_sentiment_percentage(texts_with_sentiment)

        return jsonify({
            'all_posts': texts_with_sentiment,
            'highlights': {
                'most_positive': most_positive,
                'most_negative': most_negative
            },
            'summary': {
                'total_posts': len(texts_with_sentiment),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count
            },
            'subreddit': subreddit,
            'category': category,
            'overall_sentiment': {
                'weighted': overall_sentiment_weighted,
                'percentage': overall_sentiment_percentage
            }
        })
    
    except Exception as e:
        return jsonify({"Error": f"Analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)