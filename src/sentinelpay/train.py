"""Entraînement reproductible du modèle SentinelPay (mission M2).

Usage, depuis la racine du dépôt :
    python -m sentinelpay.train            # C = 1.0
    python -m sentinelpay.train --C 0.1
"""

import argparse
import json
from pathlib import Path

import joblib

from sentinelpay.data import charger, separer
from sentinelpay.evaluate import evaluer
from sentinelpay.pipeline import construire_pipeline

ARTEFACTS = Path("artefacts")


def main(C: float) -> dict:
    df = charger()
    X_train, X_test, y_train, y_test = separer(df)

    pipeline = construire_pipeline(C=C)
    pipeline.fit(X_train, y_train)

    metriques = evaluer(pipeline, X_test, y_test)
    metriques["C"] = C

    ARTEFACTS.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTEFACTS / "modele.joblib")
    (ARTEFACTS / "metriques.json").write_text(json.dumps(metriques, indent=2))

    for cle, valeur in metriques.items():
        print(f"{cle:18s} : {valeur:.4f}" if isinstance(valeur, float) else f"{cle:18s} : {valeur}")
    return metriques


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="inverse de la régularisation")
    args = parser.parse_args()
    main(args.C)
