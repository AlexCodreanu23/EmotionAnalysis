"""
Main Training Script for Emotion Analysis

This script:
1. Loads and preprocesses data
2. Creates emotion labels using NRC Lexicon
3. Trains both models (yours and colleague's)
4. Evaluates and compares them
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from emotion_scoring import extract_emotion_features, create_emotion_labels, PLUTCHIK_EMOTIONS
from models import TFIDFLogisticClassifier, EmbeddingLogisticClassifier
from evaluate import ModelEvaluator, compare_models


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset and prepare it for training.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")

    # Clean text column
    df['Text'] = df['Text'].astype(str).str.strip()

    # Show sample
    print("\nSample texts:")
    for i in range(min(3, len(df))):
        print(f"  {i + 1}. {df['Text'].iloc[i][:60]}...")

    return df


def create_emotion_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create emotion labels using NRC Lexicon.
    """
    print("\n" + "=" * 60)
    print("CREATING EMOTION LABELS")
    print("=" * 60)

    # Extract emotion features and labels
    df = extract_emotion_features(df, text_column='Text')

    # Show emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df['dominant_emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:12}: {count:4} ({count / len(df) * 100:.1f}%)")

    return df


def train_tfidf_model(X_train: pd.Series, y_train: pd.Series,
                      X_test: pd.Series, y_test: pd.Series) -> dict:
    """
    Train and evaluate TF-IDF + Logistic Regression model.
    This is your COLLEAGUE's model.
    """
    print("\n" + "=" * 60)
    print("COLLEAGUE'S MODEL: TF-IDF + Logistic Regression")
    print("=" * 60)

    # Train model
    model = TFIDFLogisticClassifier(max_features=3000, C=1.0)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\nResults:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'model': model,
        'name': 'TF-IDF + LogReg',
        'predictions': y_pred,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def train_embedding_model(X_train: pd.Series, y_train: pd.Series,
                          X_test: pd.Series, y_test: pd.Series,
                          include_emotions: bool = True) -> dict:
    """
    Train and evaluate Word Embedding + Logistic Regression model.
    This is YOUR model.
    """
    print("\n" + "=" * 60)
    print("YOUR MODEL: Word Embeddings + Logistic Regression")
    if include_emotions:
        print("            (with NRC Emotion Features)")
    print("=" * 60)

    # Train model
    model = EmbeddingLogisticClassifier(
        embedding_model='glove-wiki-gigaword-100',  # 100-dim GloVe vectors
        include_emotions=include_emotions,
        C=1.0
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\nResults:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_name = 'Embeddings + LogReg'
    if include_emotions:
        model_name += ' + Emotions'

    return {
        'model': model,
        'name': model_name,
        'predictions': y_pred,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def visualize_results(results: list, y_test: pd.Series, save_path: str = None):
    """
    Create visualization of model comparison results.
    """
    print("\n" + "=" * 60)
    print("VISUALIZING RESULTS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar chart comparing metrics
    ax1 = axes[0, 0]
    model_names = [r['name'] for r in results]
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    x = np.arange(len(model_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax1.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

    ax1.set_ylabel('Score')
    ax1.set_title('Model Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Confusion matrix for first model
    ax2 = axes[0, 1]
    cm = confusion_matrix(y_test, results[0]['predictions'])
    labels = sorted(y_test.unique())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_title(f"Confusion Matrix: {results[0]['name']}")
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    # 3. Confusion matrix for second model (if exists)
    ax3 = axes[1, 0]
    if len(results) > 1:
        cm2 = confusion_matrix(y_test, results[1]['predictions'])
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax3,
                    xticklabels=labels, yticklabels=labels)
        ax3.set_title(f"Confusion Matrix: {results[1]['name']}")
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
    else:
        ax3.axis('off')

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    for r in results:
        table_data.append([
            r['name'],
            f"{r['accuracy']:.4f}",
            f"{r['f1_macro']:.4f}",
            f"{r['f1_weighted']:.4f}"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'F1 (Macro)', 'F1 (Weighted)'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Results Summary', y=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def main():
    """Main training pipeline."""

    # Configuration
    DATA_PATH = 'data/sentimentdataset.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # 1. Load data
    df = load_and_prepare_data(DATA_PATH)

    # 2. Create emotion labels
    df = create_emotion_dataset(df)

    # 3. Prepare features and labels
    X = df['Text']
    y = df['dominant_emotion']

    # Filter out 'neutral' if too many samples (optional)
    # df = df[df['dominant_emotion'] != 'neutral']

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # 5. Train models
    results = []

    # Colleague's model: TF-IDF
    tfidf_results = train_tfidf_model(X_train, y_train, X_test, y_test)
    results.append(tfidf_results)

    # Your model: Embeddings (this will download GloVe vectors - ~100MB)
    print("\n[Note: First run will download GloVe embeddings (~100MB)]")
    embedding_results = train_embedding_model(
        X_train, y_train, X_test, y_test,
        include_emotions=True
    )
    results.append(embedding_results)

    # 6. Compare models
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print("\n{:<30} {:>10} {:>10} {:>12}".format(
        "Model", "Accuracy", "F1 Macro", "F1 Weighted"
    ))
    print("-" * 65)

    for r in results:
        print("{:<30} {:>10.4f} {:>10.4f} {:>12.4f}".format(
            r['name'], r['accuracy'], r['f1_macro'], r['f1_weighted']
        ))

    # 7. Visualize
    visualize_results(results, y_test, save_path='results/model_comparison.png')

    # 8. Save models
    print("\nSaving models...")
    results[0]['model'].save('results/tfidf_model.joblib')
    results[1]['model'].save('results/embedding_model.joblib')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import os

    os.makedirs('results', exist_ok=True)
    results = main()