"""Client de test de l'API SentinelPay : latence et cohérence avec le modèle local.

    python api/client.py --n 100 [--url http://127.0.0.1:8000]
"""

import argparse
import statistics
import time

import mlflow
import mlflow.sklearn
import pandas as pd
import requests

from sentinelpay.data import charger


def main(url: str, n: int) -> None:
    df = charger()
    lignes = df.drop(columns=["Class"]).head(n)
    durees, scores = [], []
    for _, t in lignes.iterrows():
        debut = time.perf_counter()
        r = requests.post(f"{url}/predict", json=t.to_dict(), timeout=10)
        durees.append((time.perf_counter() - debut) * 1000)
        r.raise_for_status()
        scores.append(r.json()["score"])
    print(f"{n} requêtes : médiane {statistics.median(durees):.1f} ms, "
          f"p95 {sorted(durees)[int(0.95 * len(durees)) - 1]:.1f} ms, max {max(durees):.1f} ms")

    # Cohérence : le modèle local doit donner les mêmes scores que l'API
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    modele = mlflow.sklearn.load_model("models:/detection_fraude@production")
    local = modele.predict_proba(lignes)[:, 1]
    ecart = max(abs(float(a) - float(b)) for a, b in zip(local, scores))
    print(f"écart maximal API / modèle local : {ecart:.2e}  ->", "cohérent" if ecart < 1e-6 else "INCOHÉRENT")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=100)
    a = p.parse_args()
    main(a.url, a.n)
