from flask import Flask, render_template, request, jsonify
import reddit_fetcher
import data_cleaner 
import model_analysis

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
        return jsonify({"error": "Please provide subreddit, category, and limit"}), 400

    raw_data = reddit_fetcher.fetch_posts(subreddit, category, limit)
    clean_posts = data_cleaner.extract_text(raw_data)
    texts_with_sentiment = model_analysis.analyze_posts(clean_posts)

    return jsonify(texts_with_sentiment)

if __name__ == '__main__':
    app.run(debug=True)