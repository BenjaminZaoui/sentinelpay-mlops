"""Chargement et séparation des données SentinelPay (mission M2)."""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
# Chemin des données : surchargeable (CI : échantillon commité dans data/sample/)
DATA_PATH = Path(os.environ.get("SENTINELPAY_DATA", "data/creditcard.csv"))
TARGET = "Class"
COLONNES_ATTENDUES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", TARGET]


def charger(chemin: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le jeu historique et vérifie son schéma."""
    if not chemin.exists():
        raise FileNotFoundError(
            f"{chemin} introuvable : placez creditcard.csv dans data/ et lancez "
            "les commandes depuis la racine du dépôt."
        )
    df = pd.read_csv(chemin)
    # TODO 1 : vérification du schéma
    manquantes = [c for c in COLONNES_ATTENDUES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")
    return df


def separer(df: pd.DataFrame, test_size: float = 0.3, random_state: int = RANDOM_STATE):
    """Sépare X / y puis train / test, de façon stratifiée et reproductible.

    La séparation se fait AVANT tout prétraitement : c'est la condition
    pour qu'aucune information du jeu de test ne fuie dans l'entraînement.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # TODO 2 : split stratifié et reproductible
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )