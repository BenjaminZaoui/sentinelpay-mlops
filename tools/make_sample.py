#!/usr/bin/env python3
# ============================================================
#  Échantillon pour l'intégration continue — SentinelPay (mission M6)
# ------------------------------------------------------------
#  Les exécuteurs GitHub Actions n'ont pas accès à data/creditcard.csv
#  (150 Mo, versionné par DVC dans un remote local). On commite donc un
#  petit échantillon stratifié : toutes les fraudes + 30 000 légitimes
#  (≈ 7 Mo, taux de fraude ≈ 1,6 %), suffisant pour tester et entraîner
#  un modèle de validation. Déterministe.
#
#  Usage : python tools/make_sample.py        -> data/sample/creditcard_sample.csv
# ============================================================
from pathlib import Path

import pandas as pd

SOURCE = Path("data/creditcard.csv")
CIBLE = Path("data/sample/creditcard_sample.csv")
N_LEGIT = 30_000

if __name__ == "__main__":
    df = pd.read_csv(SOURCE)
    fraudes = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=N_LEGIT, random_state=42)
    echantillon = pd.concat([fraudes, legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    CIBLE.parent.mkdir(parents=True, exist_ok=True)
    echantillon.to_csv(CIBLE, index=False)
    print(f"{CIBLE} : {len(echantillon)} lignes, {len(fraudes)} fraudes ({len(fraudes)/len(echantillon):.2%}), "
          f"{CIBLE.stat().st_size/1e6:.1f} Mo")
