from api import app, nlp_module
from flask import request, jsonify


@app.route('/', methods=['GET'])
def index():
    return "everything it's fine", 200

@app.route('/tweet', methods=['POST'])
def tweet_route():
    data = request.get_json()
    response_data =  nlp_module.detect_tweet(data)
    return jsonify(response_data), 200