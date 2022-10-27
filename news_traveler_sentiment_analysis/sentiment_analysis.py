from transformers import pipeline


def process_sentiment_analysis(documents: list) -> list:
    classifier = pipeline("sentiment-analysis")

    return classifier(documents, truncation=True)
