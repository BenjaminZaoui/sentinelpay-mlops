"""Interroger le modèle servi par MLflow (mission M3, partie 2).

Dans un second terminal :
    mlflow models serve -m "models:/detection_fraude@production" --port 5001 --no-conda
Puis :
    python -m sentinelpay.interroger [--port 5001] [--n 5]
"""

import argparse
import time

import requests

from sentinelpay.data import charger


def interroger(port: int, n: int) -> None:
    df = charger()
    lignes = df.drop(columns=["Class"]).head(n)
    payload = {"dataframe_split": {"columns": list(lignes.columns), "data": lignes.values.tolist()}}
    debut = time.perf_counter()
    reponse = requests.post(f"http://127.0.0.1:{port}/invocations", json=payload, timeout=30)
    duree_ms = (time.perf_counter() - debut) * 1000
    reponse.raise_for_status()
    print("Réponse :", reponse.json())
    print(f"Étiquettes réelles : {df['Class'].head(n).tolist()}  ({duree_ms:.0f} ms pour {n} transactions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    interroger(args.port, args.n)
