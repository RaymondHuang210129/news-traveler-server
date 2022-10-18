from news_traveler_sentiment_analysis.sentiment_analysis import (
    process_sentiment_analysis,
)


def test_process_sentiment_analysis():
    documents = ["I love you", "I hate you"]

    result = process_sentiment_analysis(documents)

    assert len(result) == 2
