"""Evaluation metrics for classification models."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from intent_classification.config import RESULTS_DIR


def evaluate_metrics(y_true, y_pred):
    """
    Calculate and print classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, and f1 scores
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def print_experiment_summary(train_metrics, dev_metrics, experiment_name):
    """
    Print final experiment summary with all results.

    Parameters
    ----------
    train_metrics : dict
        Training set metrics
    dev_metrics : dict
        Development set metrics
    experiment_name : str
        Name of the experiment for display
    """
    print('=' * 70)
    print(f'EXPERIMENT SUMMARY: {experiment_name}')
    print('=' * 70)
    print(f'Train - Accuracy: {train_metrics["accuracy"]:.4f}, F1: {train_metrics["f1"]:.4f}')
    print(f'Dev   - Accuracy: {dev_metrics["accuracy"]:.4f}, F1: {dev_metrics["f1"]:.4f}')
    print('=' * 70)


def model_slug(model_name):
    """
    Flatten a model ID into a single path segment.

    Parameters
    ----------
    model_name : str
        Model ID as reported by the client (HuggingFace ID or endpoint model name)

    Returns
    -------
    str
        Lowercase model ID with slashes replaced, e.g. 'Qwen/Qwen3.5-9B' -> 'qwen-qwen3.5-9b'
    """
    return model_name.replace('/', '-').lower()


def save_predictions(texts, y_true, y_pred, experiment_name, split_name, model_name):
    """Save predictions to the results directory.

    Parameters
    ----------
    texts : array-like
        Input texts.
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    experiment_name : str
        Name of the experiment.
    split_name : str
        Split name (train/dev/test).
    model_name : str
        Model ID that produced the predictions, recorded in the `model` column.

    """
    experiment_dir = RESULTS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_path = experiment_dir / f'{split_name}.csv'

    df = pd.DataFrame(
        {'model': model_name, 'text': list(texts), 'y_true': list(y_true), 'y_pred': list(y_pred)}
    )
    df.to_csv(output_path, index=False)
    print(f'Saved predictions to {output_path}')
