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
from models import TFIDFLogisticClassifier, EmbeddingLogisticClassifier, MLPNeuralNetworkClassifier
from evaluate import ModelEvaluator, compare_models


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")

    df['Text'] = df['Text'].astype(str).str.strip()

    print("\nSample texts:")
    for i in range(min(3, len(df))):
        print(f"  {i + 1}. {df['Text'].iloc[i][:60]}...")

    return df


def create_emotion_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("CREATING EMOTION LABELS")
    print("=" * 60)

    df = extract_emotion_features(df, text_column='Text')

    print("\nEmotion distribution:")
    emotion_counts = df['dominant_emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:12}: {count:4} ({count / len(df) * 100:.1f}%)")

    return df


def train_tfidf_model(X_train: pd.Series, y_train: pd.Series,
                      X_test: pd.Series, y_test: pd.Series) -> dict:

    print("\n" + "=" * 60)
    print("COLLEAGUE'S MODEL: TF-IDF + Logistic Regression")
    print("=" * 60)

    model = TFIDFLogisticClassifier(max_features=3000, C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

    print("\n" + "=" * 60)
    print("YOUR MODEL: Word Embeddings + Logistic Regression")
    if include_emotions:
        print("            (with NRC Emotion Features)")
    print("=" * 60)

    model = EmbeddingLogisticClassifier(
        embedding_model='glove-wiki-gigaword-100',
        include_emotions=include_emotions,
        C=1.0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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


def train_mlp_model(X_train: pd.Series, y_train: pd.Series,
                    X_test: pd.Series, y_test: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print("NEURAL NETWORK: MLP (Multi-Layer Perceptron)")
    print("=" * 60)

    model = MLPNeuralNetworkClassifier(
        max_features=3000,
        hidden_layers=(256, 128, 64),  # 3 hidden layers
        learning_rate=0.001,
        max_iter=500
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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
        'name': 'MLP Neural Network',
        'predictions': y_pred,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'loss_curve': model.get_training_curve()
    }


def visualize_results(results: list, y_test: pd.Series, save_path: str = None):
    print("\n" + "=" * 60)
    print("VISUALIZING RESULTS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

    ax2 = axes[0, 1]
    cm = confusion_matrix(y_test, results[0]['predictions'])
    labels = sorted(y_test.unique())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_title(f"Confusion Matrix: {results[0]['name']}")
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

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
    DATA_PATH = 'data/sentimentdataset.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    df = load_and_prepare_data(DATA_PATH)

    df = create_emotion_dataset(df)

    X = df['Text']
    y = df['dominant_emotion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    results = []

    tfidf_results = train_tfidf_model(X_train, y_train, X_test, y_test)
    results.append(tfidf_results)

    print("\n[Note: First run will download GloVe embeddings (~100MB)]")
    embedding_results = train_embedding_model(
        X_train, y_train, X_test, y_test,
        include_emotions=True
    )
    results.append(embedding_results)

    mlp_results = train_mlp_model(X_train, y_train, X_test, y_test)
    results.append(mlp_results)

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

    visualize_results(results, y_test, save_path='results/model_comparison.png')

    print("\nSaving models...")

    results[0]['model'].save('results/tfidf_model.joblib')
    results[1]['model'].save('results/embedding_model.joblib')
    results[2]['model'].save('results/mlp_model.joblib')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import os

    os.makedirs('results', exist_ok=True)
    results = main()