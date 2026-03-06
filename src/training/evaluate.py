"""Model evaluation metrics for GPON router failure prediction.

This module provides reusable metric functions built on top of
``scikit-learn.metrics`` for evaluating the XGBoost binary classifier.
All functions accept NumPy-array-like inputs (arrays, lists, or pandas
Series) and return a single ``float`` score.
"""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def calculate_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
) -> float:
    """Compute the Area Under the ROC Curve (AUC-ROC).

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_score: Predicted probabilities for the positive class
            (i.e. the output of ``model.predict_proba(X)[:, 1]``).

    Returns:
        AUC-ROC score in the range [0.0, 1.0].

    Raises:
        ValueError: If inputs have mismatched lengths or contain
            invalid values.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true={y_true.shape[0]}, "
            f"y_score={y_score.shape[0]}"
        )

    return float(roc_auc_score(y_true, y_score))

def calculate_precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: Union[int, float] = 0.0,
) -> float:
    """Compute precision for the positive class.

    Precision = TP / (TP + FP).  Measures how many predicted failures
    are actual failures.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        zero_division: Value to return when there are no positive
            predictions. Defaults to ``0.0``.

    Returns:
        Precision score in the range [0.0, 1.0].

    Raises:
        ValueError: If inputs have mismatched lengths or contain
            invalid values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true={y_true.shape[0]}, "
            f"y_pred={y_pred.shape[0]}"
        )

    return float(
        precision_score(y_true, y_pred, zero_division=zero_division)
    )

def calculate_recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: Union[int, float] = 0.0,
) -> float:
    """Compute recall (sensitivity) for the positive class.

    Recall = TP / (TP + FN).  Measures how many actual failures are
    correctly identified.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        zero_division: Value to return when there are no positive
            ground-truth labels. Defaults to ``0.0``.

    Returns:
        Recall score in the range [0.0, 1.0].

    Raises:
        ValueError: If inputs have mismatched lengths or contain
            invalid values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true={y_true.shape[0]}, "
            f"y_pred={y_pred.shape[0]}"
        )

    return float(
        recall_score(y_true, y_pred, zero_division=zero_division)
    )

def calculate_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: Union[int, float] = 0.0,
) -> float:
    """Compute the F1 score for the positive class.

    F1 = 2 * (precision * recall) / (precision + recall).  Provides a
    single balanced measure combining precision and recall -- critical
    for the imbalanced GPON failure-prediction task.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        zero_division: Value to return when there are no positive
            predictions or ground-truth labels. Defaults to ``0.0``.

    Returns:
        F1 score in the range [0.0, 1.0].

    Raises:
        ValueError: If inputs have mismatched lengths or contain
            invalid values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true={y_true.shape[0]}, "
            f"y_pred={y_pred.shape[0]}"
        )

    return float(
        f1_score(y_true, y_pred, zero_division=zero_division)
    )
