import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def process_tfidf_similarity(
    base_document: str, documents: list, threshold: float
) -> npt.NDArray[np.int_]:
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform([base_document] + documents)

    cosine_similarities = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:]
    ).flatten()

    return np.where(cosine_similarities > threshold)[0]
