"""
Feature Extraction Module

Two approaches:
1. TF-IDF (for your colleague's model)
2. Word Embeddings - Word2Vec/GloVe (for your model)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import gensim.downloader as api
from typing import Tuple, Optional
import os

from preprocess import preprocess_to_string, preprocess


class TFIDFFeatureExtractor:
    """
    TF-IDF based feature extraction.
    This is what your COLLEAGUE will use.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        self.is_fitted = False

    def fit(self, texts: pd.Series) -> 'TFIDFFeatureExtractor':
        """Fit the TF-IDF vectorizer on training texts."""
        # Preprocess texts
        processed = texts.apply(lambda x: preprocess_to_string(str(x)))
        self.vectorizer.fit(processed)
        self.is_fitted = True
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        processed = texts.apply(lambda x: preprocess_to_string(str(x)))
        return self.vectorizer.transform(processed).toarray()

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit and transform texts to TF-IDF features."""
        processed = texts.apply(lambda x: preprocess_to_string(str(x)))
        self.is_fitted = True
        return self.vectorizer.fit_transform(processed).toarray()

    def get_feature_names(self) -> list:
        """Get feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out().tolist()


class EmbeddingFeatureExtractor:
    """
    Word Embedding based feature extraction.
    This is what YOU will use.

    Uses pre-trained Word2Vec or GloVe embeddings.
    Document embedding = average of word embeddings.
    """

    def __init__(self, model_name: str = 'glove-wiki-gigaword-100', embedding_dim: int = 100):
        """
        Args:
            model_name: Name of pre-trained model from gensim
                Options: 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
                        'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300',
                        'word2vec-google-news-300'
            embedding_dim: Dimension of embeddings (must match model)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.word_vectors = None

    def load_model(self) -> None:
        """Load pre-trained word vectors."""
        print(f"Loading {self.model_name}... (this may take a while on first run)")
        self.word_vectors = api.load(self.model_name)
        print(f"Loaded! Vocabulary size: {len(self.word_vectors)}")

    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a single word."""
        if self.word_vectors is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if word in self.word_vectors:
            return self.word_vectors[word]
        return None

    def get_document_embedding(self, tokens: list) -> np.ndarray:
        """
        Get document embedding as average of word embeddings.

        Args:
            tokens: List of preprocessed tokens

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if self.word_vectors is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        embeddings = []
        for token in tokens:
            if token in self.word_vectors:
                embeddings.append(self.word_vectors[token])

        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            # Return zero vector if no words found
            return np.zeros(self.embedding_dim)

    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts to embedding features.

        Args:
            texts: Series of raw texts

        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        if self.word_vectors is None:
            self.load_model()

        embeddings = []
        for text in texts:
            # Preprocess to get tokens
            tokens = preprocess(str(text), remove_sw=True, remove_emojis=True)
            # Get document embedding
            doc_emb = self.get_document_embedding(tokens)
            embeddings.append(doc_emb)

        return np.array(embeddings)


class CombinedFeatureExtractor:
    """
    Combines multiple feature types:
    - Word embeddings (or TF-IDF)
    - Emotion scores from NRC Lexicon

    This can give better results than using either alone.
    """

    def __init__(self, use_embeddings: bool = True, use_emotions: bool = True,
                 embedding_model: str = 'glove-wiki-gigaword-100'):
        """
        Args:
            use_embeddings: Whether to include word embeddings
            use_emotions: Whether to include NRC emotion scores
            embedding_model: Which embedding model to use
        """
        self.use_embeddings = use_embeddings
        self.use_emotions = use_emotions

        if use_embeddings:
            self.embedding_extractor = EmbeddingFeatureExtractor(embedding_model)

        if use_emotions:
            from emotion_scoring import get_emotion_vector
            self.get_emotion_vector = get_emotion_vector

    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts to combined features.
        """
        features = []

        if self.use_embeddings:
            emb_features = self.embedding_extractor.transform(texts)
            features.append(emb_features)

        if self.use_emotions:
            emotion_features = np.array([self.get_emotion_vector(str(t)) for t in texts])
            features.append(emotion_features)

        if features:
            return np.hstack(features)
        else:
            raise ValueError("At least one feature type must be enabled")


if __name__ == "__main__":
    # Test feature extractors
    test_texts = pd.Series([
        "I am so happy and excited about this wonderful news!",
        "This makes me really angry and frustrated!",
        "I'm scared and worried about what might happen.",
        "What a pleasant surprise! I didn't expect that."
    ])

    print("=" * 60)
    print("FEATURE EXTRACTION DEMO")
    print("=" * 60)

    # Test TF-IDF
    print("\n1. TF-IDF Features:")
    tfidf = TFIDFFeatureExtractor(max_features=100)
    tfidf_features = tfidf.fit_transform(test_texts)
    print(f"   Shape: {tfidf_features.shape}")
    print(f"   Sample features: {tfidf.get_feature_names()[:10]}")

    # Test Embeddings (uncomment to test - downloads model)
    # print("\n2. Word Embedding Features:")
    # emb = EmbeddingFeatureExtractor()
    # emb_features = emb.transform(test_texts)
    # print(f"   Shape: {emb_features.shape}")