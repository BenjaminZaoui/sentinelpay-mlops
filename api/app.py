"""API de prédiction SentinelPay (mission M4).

Lancement, depuis la racine du dépôt (le registry MLflow local doit contenir
un modèle `detection_fraude` avec l'alias `production`, cf. mission M3) :

    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

MODELE_URI accepte un alias du registry (models:/detection_fraude@production,
par défaut) ou un dossier de modèle MLflow (ex. /app/model dans l'image Docker).

Documentation interactive : http://127.0.0.1:8000/docs
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODELE_URI = os.environ.get("MODELE_URI", "models:/detection_fraude@production")
SEUIL = float(os.environ.get("SEUIL_DECISION", "0.5"))
COLONNES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sentinelpay.api")

# Schéma d'entrée : une transaction = 30 champs flottants, Amount >= 0.
# TODO 1 : un modèle Pydantic "Transaction" avec un champ float obligatoire par
#          colonne de COLONNES, et Amount >= 0 (Field(..., ge=0)). Astuce :
#          create_model("Transaction", **{c: (float, Field(...)) for c in COLONNES}).
Transaction = ...


class Prediction(BaseModel):
    fraude: bool
    score: float = Field(..., ge=0, le=1, description="probabilité de fraude")
    seuil: float
    modele: str
    version: str
    duree_ms: float


etat: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle UNE fois au démarrage, pas à chaque requête."""
    mlflow.set_tracking_uri(TRACKING_URI)
    # TODO 2 : charger le modèle UNE fois ici, avec la flavor sklearn
    #          (mlflow.sklearn.load_model(MODELE_URI)) pour disposer de predict_proba.
    etat["modele"] = ...
    etat["version"] = "inconnue"
    try:
        if MODELE_URI.startswith("models:/"):      # alias du registry -> numéro de version
            nom, alias = MODELE_URI.removeprefix("models:/").split("@")
            etat["version"] = f"v{mlflow.MlflowClient().get_model_version_by_alias(nom, alias).version}"
        else:                                      # dossier local (modèle copié dans l'image) -> run d'origine
            etat["version"] = "run-" + str(mlflow.models.Model.load(MODELE_URI).run_id)[:8]
    except Exception:
        pass
    log.info("modèle chargé : %s (%s)", MODELE_URI, etat["version"])
    yield
    etat.clear()


app = FastAPI(title="SentinelPay — détection de fraude", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Sonde de vie : le service tourne et un modèle est chargé."""
    return {"status": "ok" if "modele" in etat else "degraded", "modele": MODELE_URI, "version": etat.get("version")}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    """Score une transaction et applique le seuil de décision."""
    if "modele" not in etat:
        raise HTTPException(status_code=503, detail="modèle non chargé")
    debut = time.perf_counter()
    # TODO 3 : construire un DataFrame d'une ligne (colonnes dans l'ordre COLONNES)
    #          à partir de transaction.model_dump(), puis calculer la probabilité de
    #          fraude avec predict_proba(...)[0, 1].
    score = ...
    duree_ms = (time.perf_counter() - debut) * 1000
    reponse = Prediction(fraude=score >= SEUIL, score=score, seuil=SEUIL,
                         modele=MODELE_URI, version=etat["version"], duree_ms=round(duree_ms, 2))
    # Journal des prédictions : indispensable au monitoring (chapitre G)
    log.info("predict amount=%.2f score=%.4f fraude=%s version=%s %.1fms",
             transaction.Amount, score, reponse.fraude, etat["version"], duree_ms)
    return reponse
