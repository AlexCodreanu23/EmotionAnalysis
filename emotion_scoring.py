"""
Emotion Scoring Module using NRC Emotion Lexicon

This module uses the NRCLex library to extract Plutchik's 8 basic emotions:
- anger, anticipation, disgust, fear, joy, sadness, surprise, trust

Plus positive/negative sentiment scores.
"""

import pandas as pd
import numpy as np
from nrclex import NRCLex
from collections import defaultdict
from preprocess import preprocess_to_string, preprocess

# Plutchik's 8 basic emotions
PLUTCHIK_EMOTIONS = [
    'anger', 'anticipation', 'disgust', 'fear',
    'joy', 'sadness', 'surprise', 'trust'
]

# All NRC categories (emotions + sentiment)
ALL_NRC_CATEGORIES = PLUTCHIK_EMOTIONS + ['positive', 'negative']


def get_emotion_scores(text: str, normalize: bool = True) -> dict:
    """
    Extract emotion scores for a single text using NRC Lexicon.

    Args:
        text: Input text string
        normalize: If True, normalize scores by total emotion words

    Returns:
        Dictionary with emotion scores
    """
    # Create NRCLex object
    emotion = NRCLex(text)

    # Get raw frequencies
    raw_scores = emotion.raw_emotion_scores

    # Initialize all emotions to 0
    scores = {emo: 0.0 for emo in ALL_NRC_CATEGORIES}

    # Update with actual scores
    for emo, count in raw_scores.items():
        if emo in scores:
            scores[emo] = count

    # Normalize if requested
    if normalize:
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

    return scores


def get_dominant_emotion(text: str) -> str:
    """
    Get the dominant Plutchik emotion for a text.

    Returns:
        Name of the emotion with highest score, or 'neutral' if no emotions detected
    """
    scores = get_emotion_scores(text, normalize=False)

    # Only consider Plutchik emotions (not positive/negative)
    emotion_scores = {k: v for k, v in scores.items() if k in PLUTCHIK_EMOTIONS}

    if sum(emotion_scores.values()) == 0:
        return 'neutral'

    return max(emotion_scores, key=emotion_scores.get)


def extract_emotion_features(df: pd.DataFrame, text_column: str = 'Text') -> pd.DataFrame:
    """
    Extract emotion features for entire DataFrame.

    Args:
        df: DataFrame with text data
        text_column: Name of column containing text

    Returns:
        DataFrame with added emotion score columns
    """
    print(f"Extracting emotion features for {len(df)} texts...")

    # Initialize lists to store results
    emotion_data = []
    dominant_emotions = []

    for idx, text in enumerate(df[text_column]):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")

        text_str = str(text) if pd.notna(text) else ""

        # Get emotion scores
        scores = get_emotion_scores(text_str, normalize=True)
        emotion_data.append(scores)

        # Get dominant emotion
        dominant = get_dominant_emotion(text_str)
        dominant_emotions.append(dominant)

    # Create emotion DataFrame
    emotion_df = pd.DataFrame(emotion_data)

    # Add dominant emotion
    emotion_df['dominant_emotion'] = dominant_emotions

    # Combine with original DataFrame
    result_df = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)

    print("Done!")
    return result_df


def create_emotion_labels(df: pd.DataFrame, text_column: str = 'Text') -> pd.DataFrame:
    """
    Create emotion labels for supervised learning.
    Uses dominant emotion as the label.

    Returns DataFrame with 'emotion_label' column.
    """
    df = df.copy()
    df['emotion_label'] = df[text_column].apply(lambda x: get_dominant_emotion(str(x)))
    return df


def get_emotion_vector(text: str) -> np.ndarray:
    """
    Get emotion scores as a numpy vector.
    Useful for ML models.

    Returns:
        numpy array of shape (10,) with emotion scores
    """
    scores = get_emotion_scores(text, normalize=True)
    return np.array([scores[cat] for cat in ALL_NRC_CATEGORIES])


def analyze_text_emotions(text: str) -> None:
    """
    Print detailed emotion analysis for a single text.
    """
    print(f"\nText: {text[:100]}..." if len(text) > 100 else f"\nText: {text}")
    print("-" * 50)

    scores = get_emotion_scores(text, normalize=False)
    normalized_scores = get_emotion_scores(text, normalize=True)
    dominant = get_dominant_emotion(text)

    print("\nEmotion Scores (raw | normalized):")
    for emotion in PLUTCHIK_EMOTIONS:
        bar = "█" * int(normalized_scores[emotion] * 20)
        print(f"  {emotion:12}: {scores[emotion]:3.0f} | {normalized_scores[emotion]:.3f} {bar}")

    print(f"\nSentiment:")
    print(f"  Positive: {scores['positive']:.0f} | {normalized_scores['positive']:.3f}")
    print(f"  Negative: {scores['negative']:.0f} | {normalized_scores['negative']:.3f}")

    print(f"\nDominant Emotion: {dominant.upper()}")


if __name__ == "__main__":
    # Test emotion extraction
    test_texts = [
        "I am so happy and excited about this wonderful news!",
        "This makes me really angry and frustrated!",
        "I'm scared and worried about what might happen.",
        "What a pleasant surprise! I didn't expect that.",
        "I feel sad and disappointed about the results.",
        "That's disgusting behavior, I can't believe it.",
        "I trust you completely, you're very reliable.",
        "I can't wait for the concert next week!"
    ]

    print("=" * 60)
    print("EMOTION ANALYSIS DEMO")
    print("=" * 60)

    for text in test_texts:
        analyze_text_emotions(text)
        print()