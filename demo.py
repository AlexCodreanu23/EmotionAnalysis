import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

from preprocess import preprocess, preprocess_to_string
from emotion_scoring import (
    get_emotion_scores,
    get_dominant_emotion,
    analyze_text_emotions,
    PLUTCHIK_EMOTIONS
)
from feature_extraction import TFIDFFeatureExtractor
from sklearn.linear_model import LogisticRegression


def demo_preprocessing():
    print("=" * 60)
    print("1. PREPROCESSING DEMO")
    print("=" * 60)

    test_texts = [
        "WOW!! I can't believe this 😱! https://news.com",
        "Feeling SO happy today!!! #blessed",
        "This is TERRIBLE service, never again!"
    ]

    for text in test_texts:
        processed = preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Processed: {processed}")


def demo_emotion_scoring():
    """Demonstrate emotion scoring with NRC Lexicon."""
    print("\n" + "=" * 60)
    print("2. EMOTION SCORING DEMO (NRC Lexicon)")
    print("=" * 60)

    test_texts = [
        "I am so happy and excited about this wonderful news!",
        "This makes me really angry and frustrated!",
        "I'm scared and worried about what might happen.",
        "What a pleasant surprise! I didn't expect that.",
        "I feel sad and disappointed about the results.",
    ]

    for text in test_texts:
        analyze_text_emotions(text)


def demo_tfidf_model():
    print("\n" + "=" * 60)
    print("3. TF-IDF MODEL DEMO (Colleague's Model)")
    print("=" * 60)

    # Load data
    df = pd.read_csv('data/sentimentdataset.csv')
    df['Text'] = df['Text'].astype(str).str.strip()

    print(f"\nDataset size: {len(df)} samples")

    # Create emotion labels using NRC Lexicon
    print("\nCreating emotion labels using NRC Lexicon...")
    df['emotion_label'] = df['Text'].apply(get_dominant_emotion)

    print("\nEmotion distribution:")
    print(df['emotion_label'].value_counts())

    # Prepare data
    X = df['Text']
    y = df['emotion_label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Extract TF-IDF features
    print("\nExtracting TF-IDF features...")
    tfidf = TFIDFFeatureExtractor(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"Feature shape: {X_train_tfidf.shape}")

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'=' * 40}")
    print("RESULTS: TF-IDF + Logistic Regression")
    print(f"{'=' * 40}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"F1 (macro):   {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, f1_macro


def demo_emotion_features_model():
    print("\n" + "=" * 60)
    print("4. EMOTION FEATURES MODEL DEMO")
    print("=" * 60)
    print("(Alternative approach using NRC scores as features)")

    # Load data
    df = pd.read_csv('data/sentimentdataset.csv')
    df['Text'] = df['Text'].astype(str).str.strip()

    # Extract emotion scores as features
    print("\nExtracting emotion features...")
    emotion_features = []
    for text in df['Text']:
        scores = get_emotion_scores(text, normalize=True)
        features = [scores[emo] for emo in PLUTCHIK_EMOTIONS + ['positive', 'negative']]
        emotion_features.append(features)

    X = np.array(emotion_features)
    y = df['Text'].apply(get_dominant_emotion)

    print(f"Feature shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"\n{'=' * 40}")
    print("RESULTS: Emotion Features + Logistic Regression")
    print(f"{'=' * 40}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"F1 (macro):   {f1_macro:.4f}")

    return accuracy, f1_macro


def main():
    print("\n" + "=" * 60)
    print("   EMOTION ANALYSIS - SIMPLE DEMO")
    print("=" * 60)

    # Run demos
    demo_preprocessing()
    demo_emotion_scoring()
    tfidf_acc, tfidf_f1 = demo_tfidf_model()
    emo_acc, emo_f1 = demo_emotion_features_model()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n{:<35} {:>10} {:>10}".format("Model", "Accuracy", "F1 Macro"))
    print("-" * 55)
    print("{:<35} {:>10.4f} {:>10.4f}".format("TF-IDF + LogReg (Colleague)", tfidf_acc, tfidf_f1))
    print("{:<35} {:>10.4f} {:>10.4f}".format("Emotion Features + LogReg", emo_acc, emo_f1))

    print("\n" + "=" * 60)
    print("To run the full comparison with Word Embeddings:")
    print("  python train.py")
    print("(This will download GloVe embeddings - ~100MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()