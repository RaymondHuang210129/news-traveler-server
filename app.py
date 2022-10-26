import json
import os
from typing import Dict, List, Tuple

import requests
from flask import Flask, request
from newsdataapi import NewsDataApiClient

from news_traveler_sentiment_analysis import sentiment_analysis


def send_newsapi_request(keyword: str) -> Tuple[Dict, int]:
    api_key = os.getenv("NEWSAPI_KEY")
    params = {"q": keyword, "sortBy": "publishedAt", "apiKey": api_key}
    response = requests.get(
        url="https://newsapi.org/v2/everything", params=params, timeout=5
    )
    if response.ok:
        return json.loads(response.content.decode("utf-8")), response.status_code
    return {}, response.status_code


def parse_newsapi_response(newsapi_response: Dict) -> str:
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
    return json.dumps(response)


def send_newsdataapi_request(keyword: str) -> Tuple[Dict, int]:
    api_key = os.getenv("NEWSDATAAPI_KEY")
    api = NewsDataApiClient(apikey=api_key)
    print(api_key)
    response = api.news_api(q=keyword, country="us")
    if response["status"] != "success":
        return response, 400
    return response, 200


def parse_newsdataapi_response(newsdataapi_response: Dict) -> str:
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
    return json.dumps(response)


def send_biasapi_request(article: str) -> Tuple[Dict, int]:
    return {}, 200


def send_toneapi_request(articles: List[str]) -> Tuple[List, int]:
    print(articles[0])
    return sentiment_analysis.process_sentiment_analysis(articles), 200


def parse_semantic_response(biasapi_response: Dict, tone_response: Dict) -> str:
    tone = {}
    if tone_response["label"] == "POS":
        tone["class"] = "positive"
    elif tone_response["label"] == "NEU":
        tone["class"] = "neural"
    else:
        tone["class"] = "negative"
    tone["confidence"] = tone_response["score"]
    return json.dumps({"bias": "center", "tone": tone})


app = Flask(__name__)


@app.route("/get-news-semantic", methods=["GET"])
def get_news_semantic() -> str:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
        if "selectedNews" not in request_content:
            raise ValueError
        news_content = request_content["selectedNews"]["content"]
        biasapi_response, status_code = send_biasapi_request(news_content)
        if status_code != 200:
            return f"bias API status_code {status_code}", 500
        toneapi_response, status_code = send_toneapi_request([news_content])
        if status_code != 200:
            return f"tone API status_code {status_code}", 500
        return parse_semantic_response(biasapi_response, toneapi_response[0]), 200
    except ValueError:
        return "Invalid json format", 400
    except RuntimeError:
        return "Internal error", 500


@app.route("/search", methods=["GET"])
def search() -> str:
    try:
        request_content = json.loads(request.data.decode("utf-8"))
        if "keyword" not in request_content:
            raise ValueError
        keyword = request_content["keyword"]
        # status_code, response = send_newsapi_request(keyword)
        response, status_code = send_newsdataapi_request(keyword)
        if status_code != 200:
            return f"newsAPI status_code {status_code}", 500
        # return parse_newsapi_response(response)
        return parse_newsdataapi_response(response), 200
    except ValueError:
        return "Invalid json format", 400
    except RuntimeError:
        return "Internal error", 500
