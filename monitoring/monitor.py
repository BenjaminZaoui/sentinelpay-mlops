"""Surveillance mensuelle de la production SentinelPay (mission M7).

    python monitoring/monitor.py                      # tous les lots de data/prod/
    python monitoring/monitor.py --mois 2026-04       # un seul mois
    python monitoring/monitor.py --seuil-psi 0.25 --seuil-recall 0.75

Pour chaque lot mensuel prod_AAAA-MM.csv :
  1. dérive des entrées : PSI (maison) sur Amount et sur les variables clés,
     rapport Evidently (DataDriftPreset) en HTML ;
  2. prédictions du modèle @production ;
  3. si les étiquettes du mois sont disponibles (labels_AAAA-MM.csv) :
     performance réelle (rappel, précision, AP) et KPI métier ;
  4. tout est journalisé dans MLflow (expérience "monitoring", un run par mois)
     et résumé dans monitoring/reports/resume.csv ;
  5. décision à deux niveaux : alerte si dérive forte des entrées ; réentraînement
     recommandé si la performance réelle passe sous le seuil.
Puis un tableau de bord (monitoring/reports/dashboard.png).
"""

import os
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import average_precision_score, precision_score, recall_score

from sentinelpay.data import charger, separer
from sentinelpay.monitoring import calcul_psi, kpi_fraude

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")  # base locale par défaut, serveur partagé si la variable est définie
MODELE_URI = "models:/detection_fraude@production"
PROD = Path("data/prod")
RAPPORTS = Path("monitoring/reports")
VARIABLES_SUIVIES = ["Amount", "V14", "V17", "V12", "V10", "V4"]


def charger_reference() -> pd.DataFrame:
    """La référence est le jeu d'ENTRAÎNEMENT du modèle (même séparation qu'en M2)."""
    X_train, _, _, _ = separer(charger())
    return X_train


def surveiller_un_mois(mois: str, reference: pd.DataFrame, modele, seuil_psi: float, seuil_recall: float) -> dict:
    lot = pd.read_csv(PROD / f"prod_{mois}.csv")
    X = lot.drop(columns=["transaction_id"])
    resultat = {"mois": mois, "transactions": len(lot)}

    # 1. Dérive des entrées
    # TODO 1 : pour chaque colonne de VARIABLES_SUIVIES, calculer le PSI entre la
    #          référence et le lot (calcul_psi) et le ranger dans resultat["psi_<col>"].
    ...
    rapport = Report([DataDriftPreset()])
    evaluation = rapport.run(X[VARIABLES_SUIVIES], reference[VARIABLES_SUIVIES])
    RAPPORTS.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(str(RAPPORTS / f"drift_{mois}.html"))
    # Part de colonnes en dérive selon Evidently (métrique "DriftedColumnsCount")
    try:
        d = json.loads(evaluation.json())
        drift = [m for m in d["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")][0]["value"]
        resultat["evidently_colonnes_en_derive"] = float(drift["share"])
    except Exception:
        resultat["evidently_colonnes_en_derive"] = float("nan")

    # 2. Prédictions
    # TODO 2 : probabilité de fraude (predict_proba[:, 1]), décision au seuil 0,5,
    #          et taux d'alertes du mois dans resultat["taux_alertes"].
    score = ...
    y_pred = ...
    resultat["taux_alertes"] = ...

    # 3. Performance réelle et KPI, si les étiquettes sont arrivées
    chemin_labels = PROD / f"labels_{mois}.csv"
    if chemin_labels.exists():
        y = pd.read_csv(chemin_labels).set_index("transaction_id").loc[lot["transaction_id"], "Class"].values
        # TODO 3 : performance réelle (recall, precision, average_precision sur score)
        #          et KPI métier (kpi_fraude, clés préfixées "kpi_") dans resultat.
        ...

    # 5. Décision à deux niveaux :
    #    - ALERTE  : dérive forte des entrées (PSI ou Evidently) — disponible tout de suite,
    #                sans étiquettes ; on enquête, on prépare un candidat ;
    #    - RÉENTRAÎNER : performance réelle sous le seuil — disponible quand les
    #                étiquettes arrivent ; c'est le seul signal qui voit une dérive de concept.
    # TODO 4 : derive_forte = PSI sur Amount >= seuil_psi OU part de colonnes en dérive
    #          (Evidently) >= 0,5 ; perf_degradee = étiquettes disponibles ET rappel < seuil_recall.
    derive_forte = ...
    perf_degradee = ...
    resultat["alerte_derive"] = bool(derive_forte)
    resultat["reentrainer"] = bool(perf_degradee)

    # 4. Journalisation MLflow
    with mlflow.start_run(run_name=f"monitoring_{mois}"):
        mlflow.set_tags({"mois": mois, "modele": MODELE_URI})
        mlflow.log_metrics({k: float(v) for k, v in resultat.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_params({"alerte_derive": resultat["alerte_derive"], "reentrainer": resultat["reentrainer"]})
        mlflow.log_artifact(str(RAPPORTS / f"drift_{mois}.html"))
    return resultat


def tableau_de_bord(resume: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(resume["mois"], resume["psi_Amount"], marker="o"); axes[0].axhline(0.25, color="red", ls="--"); axes[0].set_title("PSI sur Amount (dérive des entrées)")
    if "recall" in resume:
        axes[1].plot(resume["mois"], resume["recall"], marker="o", label="rappel"); axes[1].plot(resume["mois"], resume["precision"], marker="s", label="précision")
        axes[1].axhline(0.75, color="red", ls="--"); axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].set_title("Performance réelle (étiquettes reçues)")
        axes[2].bar(resume["mois"], resume["kpi_gain_net"], color=["green" if g > 0 else "red" for g in resume["kpi_gain_net"]]); axes[2].set_title("Gain net métier (€)")
    for ax in axes: ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(RAPPORTS / "dashboard.png", dpi=120); plt.close(fig)


def main(mois_voulu: str | None, seuil_psi: float, seuil_recall: float) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("monitoring")
    modele = mlflow.sklearn.load_model(MODELE_URI)
    reference = charger_reference()
    mois_dispo = sorted(p.stem.replace("prod_", "") for p in PROD.glob("prod_*.csv"))
    if mois_voulu:
        mois_dispo = [m for m in mois_dispo if m == mois_voulu]
    lignes = [surveiller_un_mois(m, reference, modele, seuil_psi, seuil_recall) for m in mois_dispo]
    resume = pd.DataFrame(lignes)
    resume.to_csv(RAPPORTS / "resume.csv", index=False)
    tableau_de_bord(resume)
    colonnes = [c for c in ["mois", "psi_Amount", "evidently_colonnes_en_derive", "taux_alertes", "recall", "precision", "kpi_gain_net", "alerte_derive", "reentrainer"] if c in resume]
    print(resume[colonnes].to_string(index=False))
    alertes = resume[resume["alerte_derive"]]["mois"].tolist()
    a_reentrainer = resume[resume["reentrainer"]]["mois"].tolist()
    print("\nAlerte dérive des entrées :", ", ".join(alertes) if alertes else "aucune")
    print("Réentraînement recommandé à partir de :", a_reentrainer[0] if a_reentrainer else "aucun mois")
    print(f"Rapports : {RAPPORTS}/drift_<mois>.html, resume.csv, dashboard.png ; runs MLflow dans l'expérience « monitoring ».")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mois", default=None)
    p.add_argument("--seuil-psi", type=float, default=0.25)
    p.add_argument("--seuil-recall", type=float, default=0.75)
    a = p.parse_args()
    main(a.mois, a.seuil_psi, a.seuil_recall)
