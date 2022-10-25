import json
import sys
from typing import Tuple

import requests
from flask import Flask, request

from config import Config

sys.path.insert(0, "../news_traveler_document_similarity")
sys.path.insert(0, "../news_traveler_sentiment_analysis")

app = Flask(__name__)


def send_newsapi_request(keyword: str) -> Tuple[int, str]:
    params = {"q": keyword, "sortBy": "publishedAt", "apiKey": Config.newsapi_api_key}
    response = requests.get(url=Config.newsapi_url, params=params, timeout=5)
    if response.ok:
        return response.status_code, response.content.decode("utf-8")
    return response.status_code, ""


@app.route("/search", methods=["GET"])
def search() -> str:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
        if "keyword" not in request_content:
            raise ValueError
        keyword = request_content["keyword"]
        status_code, newsapi_response = send_newsapi_request(keyword)
        if status_code != 200:
            return f"newsAPI status_code {status_code}", 500
        newsapi_response = json.loads(newsapi_response)
        response = {"totalResults": newsapi_response["totalResults"], "results": []}
        for news in newsapi_response["articles"]:
            response["results"].append(
                {
                    "semanticInfo": {},
                    "source": news["source"],
                    "author": news["author"],
                    "title": news["title"],
                    "description": news["description"],
                    "url": news["url"],
                    "urlToImage": news["urlToImage"],
                    "publishedAt": news["publishedAt"],
                    "content": news["content"],
                }
            )
        return json.dumps(response), 200
    except ValueError:
        return 400, "Invalid json format"
