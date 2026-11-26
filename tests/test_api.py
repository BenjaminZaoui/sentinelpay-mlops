"""Tests de l'API (mission M4) : exécutés avec le modèle @production du registry local."""

import pytest
from fastapi.testclient import TestClient

from api.app import app, COLONNES


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # déclenche lifespan : chargement du modèle
        yield c


def transaction_valide():
    t = {c: 0.0 for c in COLONNES}
    t["Amount"] = 42.5
    t["Time"] = 1000.0
    return t


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_transaction_valide(client):
    r = client.post("/predict", json=transaction_valide())
    assert r.status_code == 200
    corps = r.json()
    assert set(corps) >= {"fraude", "score", "seuil", "version"}
    assert 0.0 <= corps["score"] <= 1.0


def test_predict_colonne_manquante(client):
    t = transaction_valide()
    del t["V14"]
    assert client.post("/predict", json=t).status_code == 422


def test_predict_montant_negatif(client):
    t = transaction_valide()
    t["Amount"] = -5
    assert client.post("/predict", json=t).status_code == 422


def test_predict_type_invalide(client):
    t = transaction_valide()
    t["V1"] = "beaucoup"
    assert client.post("/predict", json=t).status_code == 422
