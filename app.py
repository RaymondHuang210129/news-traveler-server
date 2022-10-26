import json
import os
import sys
from typing import Dict, Tuple

import requests
from flask import Flask, request
from newsdataapi import NewsDataApiClient

sys.path.insert(0, "../news_traveler_document_similarity")
sys.path.insert(0, "../news_traveler_sentiment_analysis")


def send_newsapi_request(keyword: str) -> Tuple[int, str]:
    api_key = os.getenv("NEWSAPI_KEY")
    params = {"q": keyword, "sortBy": "publishedAt", "apiKey": api_key}
    response = requests.get(
        url="https://newsapi.org/v2/everything", params=params, timeout=5
    )
    if response.ok:
        return response.status_code, response.content.decode("utf-8")
    return response.status_code, ""


def parse_newsapi_response(newsapi_response: str) -> Tuple[str, int]:
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


def send_newsdataapi_request(keyword: str) -> Tuple[int, Dict]:
    api_key = os.getenv("NEWSDATAAPI_KEY")
    api = NewsDataApiClient(apikey=api_key)
    print(api_key)
    response = api.news_api(q=keyword, country="us")
    if response["status"] != "success":
        return 400, response
    return 200, response


def parse_newsdataapi_response(newsdataapi_response: Dict) -> Tuple[str, int]:
    response = {"totalResults": newsdataapi_response["totalResults"], "results": []}
    for news in newsdataapi_response["results"]:
        response["results"].append(
            {
                "semanticInfo": {},
                "source": news["source_id"],
                "author": news["creator"],
                "title": news["title"],
                "description": news["description"],
                "url": news["link"],
                "urlToImage": news["image_url"],
                "publishedAt": news["pubDate"],
                "content": news["content"],
            }
        )
    return json.dumps(response), 200


app = Flask(__name__)


@app.route("/search", methods=["GET"])
def search() -> str:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
        if "keyword" not in request_content:
            raise ValueError
        keyword = request_content["keyword"]
        # status_code, response = send_newsapi_request(keyword)
        status_code, response = send_newsdataapi_request(keyword)
        if status_code != 200:
            return f"newsAPI status_code {status_code}", 500
        # return parse_newsapi_response(response)
        return parse_newsdataapi_response(response)
    except ValueError:
        return 400, "Invalid json format"
