from transformers import pipeline


def process_sentiment_analysis(documents: list) -> list:
    sentiment_pipeline = pipeline(
        model="finiteautomata/bertweet-base-sentiment-analysis"
    )
    return sentiment_pipeline(documents)
