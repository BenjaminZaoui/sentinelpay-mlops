"""Suivi des expériences avec MLflow (mission M3, partie 1).

Usage, depuis la racine du dépôt :
    python -m sentinelpay.tracking              # trois régressions logistiques (C = 0.1, 1, 10)
    python -m sentinelpay.tracking --rf         # ajoute une Random Forest (extension)
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from sentinelpay.data import charger, separer
from sentinelpay.evaluate import evaluer
from sentinelpay.monitoring import kpi_fraude
from sentinelpay.pipeline import construire_pipeline, RANDOM_STATE

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")  # base locale par défaut, serveur partagé si la variable est définie
EXPERIENCE = "detection_fraude"


def entrainer_et_journaliser(X_train, X_test, y_train, y_test, nom_modele: str, params: dict) -> str:
    """Un run MLflow complet pour une configuration ; renvoie l'identifiant du run."""
    with mlflow.start_run(run_name=f"{nom_modele}_{params}") as run:
        # 1. Paramètres
        # TODO 1 : journaliser le nom du modèle (log_param), les hyperparamètres
        #          (log_params) et un tag "mission" = "M3" (set_tag).
        ...

        # 2. Pipeline
        if nom_modele == "logreg":
            pipeline = construire_pipeline(C=params["C"])
        elif nom_modele == "rf":
            pipeline = construire_pipeline()
            pipeline.set_params(modele=RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **params))
        else:
            raise ValueError(nom_modele)
        pipeline.fit(X_train, y_train)

        # 3. Métriques statistiques + KPI métier
        metriques = evaluer(pipeline, X_test, y_test)
        kpi = kpi_fraude(y_test, pipeline.predict(X_test), X_test["Amount"])
        # TODO 2 : journaliser les métriques flottantes de `metriques` (log_metrics)
        #          et les KPI métier préfixés "kpi_" (ex. kpi_gain_net).
        ...

        # 4. Artefact : matrice de confusion
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, pipeline.predict(X_test), ax=ax)
        fig.savefig("confusion.png", bbox_inches="tight")
        plt.close(fig)
        # TODO 3 : journaliser confusion.png comme artefact du run.
        ...

        # 5. Modèle avec signature et exemple d'entrée
        exemple = X_test.head(5)
        # TODO 4 : déduire la signature (infer_signature sur exemple et la prédiction)
        #          puis journaliser le pipeline avec mlflow.sklearn.log_model(
        #          pipeline, name="model", signature=..., input_example=exemple,
        #          skops_trusted_types=["sklearn.tree._tree.Tree"] if nom_modele == "rf" else None).
        ...

        print(f"{nom_modele} {params} -> recall={metriques['recall']:.3f} "
              f"precision={metriques['precision']:.3f} AP={metriques['average_precision']:.3f} "
              f"gain_net={kpi['gain_net']:.0f} €")
        return run.info.run_id


def main(avec_rf: bool) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIENCE)
    X_train, X_test, y_train, y_test = separer(charger())

    for C in (0.1, 1.0, 10.0):
        entrainer_et_journaliser(X_train, X_test, y_train, y_test, "logreg", {"C": C})
    if avec_rf:
        entrainer_et_journaliser(X_train, X_test, y_train, y_test, "rf",
                                 {"n_estimators": 100, "max_depth": 8})
    print("\nInterface : mlflow ui --backend-store-uri sqlite:///mlflow.db  (http://127.0.0.1:5000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf", action="store_true", help="ajouter une Random Forest")
    main(parser.parse_args().rf)
