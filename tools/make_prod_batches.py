#!/usr/bin/env python3
# ============================================================
#  Générateur de lots de production mensuels — SentinelPay
# ------------------------------------------------------------
#  À partir de data/creditcard.csv (jeu historique étiqueté), produit
#  six lots mensuels qui simulent la production de SentinelPay :
#
#    data/prod/prod_2026-01.csv ... prod_2026-06.csv   (transactions SANS étiquette)
#    data/prod/labels_2026-01.csv ... labels_2026-06.csv (étiquettes "livrées un mois plus tard")
#
#  Scénario de dérive (configurable ci-dessous) :
#    - mois 1 à 3 : production stable, proche des données d'entraînement ;
#    - mois 4 à 6 : data drift (les montants augmentent progressivement)
#                   ET concept drift (une partie des fraudes change de
#                   profil : leurs variables V* se rapprochent des
#                   transactions légitimes, le modèle les détecte moins bien) ;
#    - le taux de fraude augmente légèrement à partir du mois 4.
#
#  Le générateur est déterministe (graine fixe) : tout le monde obtient
#  les mêmes lots. Vous l'UTILISEZ (missions M6, M7, M8), vous n'avez pas
#  à le modifier.
#
#  Usage (depuis la racine du dépôt) :
#     python tools/make_prod_batches.py                # lots par défaut
#     python tools/make_prod_batches.py --rows 15000   # lots plus petits
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("data/creditcard.csv")
OUTDIR = Path("data/prod")
SEED = 2026

# Un scénario par mois : facteur sur Amount, part des fraudes "nouvelles",
# intensité du rapprochement vers les légitimes, taux de fraude visé.
SCENARIO = [
    # mois,      amount_factor, new_fraud_share, alpha, fraud_rate
    ("2026-01",  1.00,          0.00,            0.0,   0.0017),
    ("2026-02",  1.02,          0.00,            0.0,   0.0017),
    ("2026-03",  1.05,          0.00,            0.0,   0.0018),
    ("2026-04",  1.35,          0.40,            0.35,  0.0028),
    ("2026-05",  1.65,          0.55,            0.50,  0.0033),
    ("2026-06",  1.90,          0.70,            0.60,  0.0038),
]

# Variables les plus discriminantes de la fraude sur ce jeu de données :
# ce sont elles que les "nouveaux fraudeurs" apprennent à masquer.
MASKED_FEATURES = ["V14", "V17", "V12", "V10", "V16", "V3", "V7", "V11", "V4"]


def main(rows: int) -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE} introuvable. Placez creditcard.csv dans data/ et lancez "
            "ce script depuis la racine du dépôt."
        )
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(SOURCE)
    legit = df[df["Class"] == 0].reset_index(drop=True)
    fraud = df[df["Class"] == 1].reset_index(drop=True)
    legit_mean = legit[MASKED_FEATURES].mean()
    features = [c for c in df.columns if c != "Class"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    next_id = 1_000_000
    for month, amount_factor, new_share, alpha, fraud_rate in SCENARIO:
        n_fraud = max(int(round(rows * fraud_rate)), 1)
        n_legit = rows - n_fraud
        lot_legit = legit.sample(n=n_legit, replace=False, random_state=int(rng.integers(1e9)))
        lot_fraud = fraud.sample(n=n_fraud, replace=True, random_state=int(rng.integers(1e9)))
        lot_fraud = lot_fraud.reset_index(drop=True)

        # Concept drift : une part des fraudes se "déguise" en légitime
        n_new = int(round(n_fraud * new_share))
        if n_new > 0:
            idx = lot_fraud.index[:n_new]
            for col in MASKED_FEATURES:
                lot_fraud.loc[idx, col] = (
                    lot_fraud.loc[idx, col] + alpha * (legit_mean[col] - lot_fraud.loc[idx, col])
                    + rng.normal(0, 0.1, n_new)
                )

        lot = pd.concat([lot_legit, lot_fraud]).sample(frac=1.0, random_state=int(rng.integers(1e9)))
        lot = lot.reset_index(drop=True)

        # Data drift : inflation des montants + bruit
        lot["Amount"] = (lot["Amount"] * amount_factor + rng.normal(0, 3.0, len(lot))).clip(lower=0).round(2)
        # Time : secondes depuis le début de la fenêtre de collecte, même
        # plage que le jeu historique (≈ 2 jours) pour ne pas créer de
        # dérive artificielle sur cette colonne.
        lot["Time"] = np.sort(rng.uniform(0, df["Time"].max(), len(lot))).round(0)

        lot.insert(0, "transaction_id", np.arange(next_id, next_id + len(lot)))
        next_id += len(lot)

        lot[["transaction_id"] + features].to_csv(OUTDIR / f"prod_{month}.csv", index=False)
        lot[["transaction_id", "Class"]].to_csv(OUTDIR / f"labels_{month}.csv", index=False)
        print(f"{month} : {len(lot):6d} transactions, {int(lot['Class'].sum()):3d} fraudes "
              f"(dont {n_new} de profil nouveau), facteur montants x{amount_factor}")

    print(f"\nLots écrits dans {OUTDIR}/ (prod_*.csv sans étiquette, labels_*.csv avec).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère les lots de production mensuels de SentinelPay.")
    parser.add_argument("--rows", type=int, default=20000, help="transactions par mois (défaut 20000)")
    args = parser.parse_args()
    main(args.rows)
