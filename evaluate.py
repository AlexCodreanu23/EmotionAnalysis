"""
Evaluation Module for Emotion Analysis

Contains:
- ModelEvaluator class for comprehensive evaluation
- Comparison functions between models
- Visualization utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """

    def __init__(self, model, model_name: str):
        """
        Args:
            model: Trained model with predict() method
            model_name: Name for display
        """
        self.model = model
        self.model_name = model_name
        self.results = {}

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """
        Run full evaluation on test set.

        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = self.model.predict(X_test)

        # Basic metrics
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        # Per-class metrics
        classes = sorted(y_test.unique())
        precision_per_class = precision_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)

        self.results['per_class'] = {
            cls: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
            for i, cls in enumerate(classes)
        }

        # Store predictions and ground truth
        self.results['predictions'] = y_pred
        self.results['ground_truth'] = y_test.values
        self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred, labels=classes).tolist()
        self.results['classes'] = classes

        return self.results

    def cross_validate(self, X: pd.Series, y: pd.Series,
                       cv: int = 5, scoring: str = 'f1_macro') -> Dict[str, float]:
        """
        Run cross-validation.

        Returns:
            Dictionary with mean and std of CV scores
        """
        # Get features
        if hasattr(self.model, 'get_features'):
            X_features = self.model.get_features(X)
        else:
            # Assume model has feature extractor
            X_features = self.model.feature_extractor.transform(X)

        y_encoded = self.model.label_encoder.transform(y)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model.model, X_features, y_encoded,
                                 cv=skf, scoring=scoring)

        return {
            'cv_mean': float(np.mean(scores)),
            'cv_std': float(np.std(scores)),
            'cv_scores': scores.tolist()
        }

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION REPORT: {self.model_name}")
        print(f"{'=' * 60}")

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:         {self.results['accuracy']:.4f}")
        print(f"  Precision (macro): {self.results['precision_macro']:.4f}")
        print(f"  Recall (macro):    {self.results['recall_macro']:.4f}")
        print(f"  F1 (macro):        {self.results['f1_macro']:.4f}")
        print(f"  F1 (weighted):     {self.results['f1_weighted']:.4f}")

        print(f"\nPer-Class Metrics:")
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-' * 44}")

        for cls, metrics in self.results['per_class'].items():
            print(f"  {cls:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")

    def plot_confusion_matrix(self, figsize: tuple = (8, 6),
                              save_path: Optional[str] = None) -> None:
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=figsize)

        cm = np.array(self.results['confusion_matrix'])
        classes = self.results['classes']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)

        plt.title(f'Confusion Matrix: {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.results.items()
        }

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to {filepath}")


def compare_models(results_list: List[Dict[str, Any]],
                   save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        results_list: List of dictionaries with model results
        save_path: Optional path to save comparison plot

    Returns:
        DataFrame with comparison
    """
    comparison_data = []

    for r in results_list:
        comparison_data.append({
            'Model': r['name'],
            'Accuracy': r['accuracy'],
            'Precision (Macro)': r.get('precision_macro', 0),
            'Recall (Macro)': r.get('recall_macro', 0),
            'F1 (Macro)': r['f1_macro'],
            'F1 (Weighted)': r['f1_weighted']
        })

    df = pd.DataFrame(comparison_data)

    # Print comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)']
    x = np.arange(len(df))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Model'], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, v in enumerate(df[metric]):
            ax.text(j + i * width, v + 0.02, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()

    return df


def statistical_significance_test(scores1: List[float], scores2: List[float],
                                  model1_name: str, model2_name: str) -> Dict[str, float]:
    """
    Perform paired t-test to check if difference is statistically significant.

    Args:
        scores1, scores2: Cross-validation scores for both models
        model1_name, model2_name: Names for display

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Effect size (Cohen's d)
    diff = scores1 - scores2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    print(f"\nStatistical Significance Test: {model1_name} vs {model2_name}")
    print("-" * 50)
    print(f"  {model1_name} mean: {np.mean(scores1):.4f} (±{np.std(scores1):.4f})")
    print(f"  {model2_name} mean: {np.mean(scores2):.4f} (±{np.std(scores2):.4f})")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    if p_value < 0.05:
        winner = model1_name if np.mean(scores1) > np.mean(scores2) else model2_name
        print(f"  Result: Significant difference (p < 0.05). {winner} performs better.")
    else:
        print(f"  Result: No significant difference (p >= 0.05).")

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Use ModelEvaluator class to evaluate models.")
    print("Use compare_models() to compare multiple models.")