"""Préparer un jeu de réentraînement et comparer candidat / production (mission M7, étape c).

    python monitoring/reentrainer.py --jusqu-a 2026-05          # 1. jeu enrichi : historique + lots étiquetés jusqu'au mois donné
    python pipelines/pipeline_sentinelpay.py --chemin-csv data/retrain/creditcard_2026-05.csv   # 2. pipeline M5 -> @candidate
    python monitoring/reentrainer.py --comparer 2026-06          # 3. production vs candidat sur un mois tenu à l'écart
"""

import os
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from sentinelpay.monitoring import kpi_fraude

PROD = Path("data/prod")
RETRAIN = Path("data/retrain")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")  # base locale par défaut, serveur partagé si la variable est définie


def construire_jeu(jusqu_a: str) -> Path:
    """Historique + tous les lots de production étiquetés jusqu'au mois `jusqu_a` inclus."""
    parties = [pd.read_csv("data/creditcard.csv")]
    for lot in sorted(PROD.glob("prod_*.csv")):
        mois = lot.stem.replace("prod_", "")
        if mois <= jusqu_a and (PROD / f"labels_{mois}.csv").exists():
            x = pd.read_csv(lot)
            y = pd.read_csv(PROD / f"labels_{mois}.csv").set_index("transaction_id").loc[x["transaction_id"], "Class"].values
            x = x.drop(columns=["transaction_id"]); x["Class"] = y
            parties.append(x)
            print(f"  + {mois} : {len(x)} transactions, {int(y.sum())} fraudes")
    jeu = pd.concat(parties, ignore_index=True)
    RETRAIN.mkdir(exist_ok=True)
    cible = RETRAIN / f"creditcard_{jusqu_a}.csv"
    jeu.to_csv(cible, index=False)
    print(f"{cible} : {len(jeu)} lignes, {int(jeu['Class'].sum())} fraudes")
    return cible


def comparer(mois: str) -> None:
    """Le modèle @production et le modèle @candidate sur un mois étiqueté non vu à l'entraînement."""
    mlflow.set_tracking_uri(TRACKING_URI)
    lot = pd.read_csv(PROD / f"prod_{mois}.csv")
    X = lot.drop(columns=["transaction_id"])
    y = pd.read_csv(PROD / f"labels_{mois}.csv").set_index("transaction_id").loc[lot["transaction_id"], "Class"].values
    print(f"Mois {mois} : {len(X)} transactions, {int(y.sum())} fraudes")
    for alias in ("production", "candidate"):
        try:
            modele = mlflow.sklearn.load_model(f"models:/detection_fraude@{alias}")
        except Exception as exc:
            print(f"  @{alias} : indisponible ({exc.__class__.__name__})"); continue
        score = modele.predict_proba(X)[:, 1]; y_pred = (score >= 0.5).astype(int)
        kpi = kpi_fraude(y, y_pred, X["Amount"].values)
        print(f"  @{alias:10s} recall={recall_score(y, y_pred):.3f} precision={precision_score(y, y_pred, zero_division=0):.3f} "
              f"AP={average_precision_score(y, score):.3f} gain_net={kpi['gain_net']:.0f} € faux_positifs={kpi['faux_positifs']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jusqu-a", help="construire le jeu enrichi jusqu'à ce mois inclus (AAAA-MM)")
    p.add_argument("--comparer", help="comparer @production et @candidate sur ce mois (AAAA-MM)")
    a = p.parse_args()
    if a.jusqu_a: construire_jeu(a.jusqu_a)
    if a.comparer: comparer(a.comparer)
    if not (a.jusqu_a or a.comparer): p.print_help()
