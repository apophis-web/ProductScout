from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    data = request.json
    results = sentiment_pipeline(data)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug = False)
