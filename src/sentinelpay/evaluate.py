"""Évaluation adaptée à la fraude (mission M2)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluer(pipeline, X_test, y_test) -> dict:
    """Métriques adaptées à un problème très déséquilibré.

    L'exactitude (accuracy) n'y figure pas : elle vaudrait 99,8 % pour un
    modèle qui ne prédit jamais de fraude.
    """
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    # TODO 6 : métriques adaptées à la fraude
    return {
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
        "n_test": int(len(y_test)),
        "n_fraudes_test": int(np.sum(y_test)),
    }