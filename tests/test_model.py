"""Tests de modèle (mission M2) : le modèle entraîné est-il acceptable ?"""

import pytest

from sentinelpay.data import charger, separer
from sentinelpay.evaluate import evaluer
from sentinelpay.pipeline import construire_pipeline

SEUIL_RECALL = 0.75
SEUIL_ROC_AUC = 0.90


@pytest.fixture(scope="module")
def resultats():
    df = charger()
    X_train, X_test, y_train, y_test = separer(df)
    pipeline = construire_pipeline().fit(X_train, y_train)
    return pipeline, evaluer(pipeline, X_test, y_test), X_test


def test_rappel_minimal(resultats):
    _, metriques, _ = resultats
    assert metriques["recall"] >= SEUIL_RECALL


def test_auc_minimale(resultats):
    _, metriques, _ = resultats
    assert metriques["roc_auc"] >= SEUIL_ROC_AUC


def test_le_pipeline_predit_des_classes_binaires(resultats):
    pipeline, _, X_test = resultats
    predictions = set(pipeline.predict(X_test.head(100)))
    assert predictions <= {0, 1}


def test_entrainement_reproductible():
    # TODO 8 : entraîner deux fois le pipeline sur la même séparation et
    #          vérifier que les métriques sont strictement identiques.
    assert False, "TODO 8"
