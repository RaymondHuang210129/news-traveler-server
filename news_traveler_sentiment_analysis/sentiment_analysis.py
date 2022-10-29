from multiprocessing import Pool, cpu_count

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def process_sentiment_analysis(documents: list) -> list:
    with Pool(cpu_count()) as p:
        results = list(p.map(sentiment_analysis_per_document, documents))
    return results


def sentiment_analysis_per_document(document: str) -> dict:
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(document)

    result = {}

    if sentiment_dict["compound"] >= 0.05:
        result["label"] = "POS"
        result["score"] = sentiment_dict["pos"]
    elif sentiment_dict["compound"] <= -0.05:
        result["label"] = "NEG"
        result["score"] = sentiment_dict["neg"]
    else:
        result["label"] = "NEU"
        result["score"] = sentiment_dict["neu"]

    return result
