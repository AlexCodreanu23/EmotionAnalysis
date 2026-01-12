"""
Models Module

Contains two model implementations:
1. YOUR MODEL: Logistic Regression + Word Embeddings (+ optional emotion features)
2. COLLEAGUE'S MODEL: Logistic Regression + TF-IDF
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from typing import Dict, Any, Optional

from feature_extraction import TFIDFFeatureExtractor, EmbeddingFeatureExtractor


class BaseEmotionClassifier:
    """Base class for emotion classifiers."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Encode string labels to integers."""
        return self.label_encoder.fit_transform(labels)

    def decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(encoded)

    def get_classes(self) -> list:
        """Get list of class labels."""
        return self.label_encoder.classes_.tolist()

    def save(self, filepath: str) -> None:
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'name': self.name
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.name = data['name']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


class TFIDFLogisticClassifier(BaseEmotionClassifier):
    """
    COLLEAGUE'S MODEL

    TF-IDF features + Logistic Regression
    Simple but effective baseline.
    """

    def __init__(self, max_features: int = 5000, C: float = 1.0):
        """
        Args:
            max_features: Max TF-IDF features
            C: Regularization strength for Logistic Regression
        """
        super().__init__(name="TF-IDF + Logistic Regression")

        self.feature_extractor = TFIDFFeatureExtractor(max_features=max_features)
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        )

    def fit(self, texts: pd.Series, labels: pd.Series) -> 'TFIDFLogisticClassifier':
        """
        Train the model.

        Args:
            texts: Series of raw text data
            labels: Series of emotion labels
        """
        print(f"Training {self.name}...")

        # Extract features
        X = self.feature_extractor.fit_transform(texts)
        y = self.encode_labels(labels)

        # Train model
        self.model.fit(X, y)
        self.is_fitted = True

        print(f"Training complete! Classes: {self.get_classes()}")
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        """Predict emotion labels for texts."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")

        X = self.feature_extractor.transform(texts)
        y_pred = self.model.predict(X)
        return self.decode_labels(y_pred)

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")

        X = self.feature_extractor.transform(texts)
        return self.model.predict_proba(X)

    def get_features(self, texts: pd.Series) -> np.ndarray:
        """Get TF-IDF features for texts."""
        if not self.feature_extractor.is_fitted:
            raise ValueError("Feature extractor not fitted.")
        return self.feature_extractor.transform(texts)


class EmbeddingLogisticClassifier(BaseEmotionClassifier):
    """
    YOUR MODEL

    Word Embeddings (GloVe/Word2Vec) + Logistic Regression
    Can optionally include emotion features from NRC Lexicon.
    """

    def __init__(self, embedding_model: str = 'glove-wiki-gigaword-100',
                 include_emotions: bool = True, C: float = 1.0):
        """
        Args:
            embedding_model: Pre-trained embedding model name
            include_emotions: Whether to include NRC emotion scores as features
            C: Regularization strength for Logistic Regression
        """
        super().__init__(name="Word Embeddings + Logistic Regression")

        self.include_emotions = include_emotions
        if include_emotions:
            self.name += " (+ Emotion Features)"

        self.feature_extractor = EmbeddingFeatureExtractor(model_name=embedding_model)
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        )

        if include_emotions:
            from emotion_scoring import get_emotion_vector
            self.get_emotion_vector = get_emotion_vector

    def _extract_features(self, texts: pd.Series) -> np.ndarray:
        """Extract combined features."""
        # Get embedding features
        emb_features = self.feature_extractor.transform(texts)

        if self.include_emotions:
            # Add emotion features
            emotion_features = np.array([
                self.get_emotion_vector(str(t)) for t in texts
            ])
            return np.hstack([emb_features, emotion_features])

        return emb_features

    def fit(self, texts: pd.Series, labels: pd.Series) -> 'EmbeddingLogisticClassifier':
        """
        Train the model.

        Args:
            texts: Series of raw text data
            labels: Series of emotion labels
        """
        print(f"Training {self.name}...")

        # Load embeddings if not loaded
        if self.feature_extractor.word_vectors is None:
            self.feature_extractor.load_model()

        # Extract features
        X = self._extract_features(texts)
        y = self.encode_labels(labels)

        print(f"Feature shape: {X.shape}")

        # Train model
        self.model.fit(X, y)
        self.is_fitted = True

        print(f"Training complete! Classes: {self.get_classes()}")
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        """Predict emotion labels for texts."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._extract_features(texts)
        y_pred = self.model.predict(X)
        return self.decode_labels(y_pred)

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")

        X = self._extract_features(texts)
        return self.model.predict_proba(X)

    def get_features(self, texts: pd.Series) -> np.ndarray:
        """Get combined features for texts."""
        return self._extract_features(texts)


# Alternative models you could also try
class AlternativeModels:
    """
    Collection of alternative classifiers you could experiment with.
    """

    @staticmethod
    def get_svm(C: float = 1.0):
        """Support Vector Machine classifier."""
        return SVC(C=C, kernel='rbf', probability=True, random_state=42)

    @staticmethod
    def get_random_forest(n_estimators: int = 100):
        """Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42
        )

    @staticmethod
    def get_mlp(hidden_layers: tuple = (100, 50)):
        """Multi-layer Perceptron (simple neural network)."""
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=500,
            random_state=42
        )


if __name__ == "__main__":
    # Quick demo
    print("=" * 60)
    print("MODELS DEMO")
    print("=" * 60)

    # Sample data
    texts = pd.Series([
        "I am so happy and excited!",
        "This makes me angry!",
        "I'm scared about this.",
        "What a pleasant surprise!",
        "I feel sad today.",
        "I trust you completely."
    ])
    labels = pd.Series(['joy', 'anger', 'fear', 'surprise', 'sadness', 'trust'])

    # Test TF-IDF model (colleague's model)
    print("\n1. Testing TF-IDF + Logistic Regression:")
    tfidf_model = TFIDFLogisticClassifier(max_features=100)
    tfidf_model.fit(texts, labels)
    predictions = tfidf_model.predict(texts)
    print(f"   Predictions: {predictions}")

    print("\n2. Embedding model would require downloading GloVe vectors...")
    print("   Run the full training script to test it.")