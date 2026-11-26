"""Construction du pipeline scikit-learn SentinelPay (mission M2)."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
COLONNES_A_NORMALISER = ["Amount", "Time"]


def construire_pipeline(C: float = 1.0, random_state: int = RANDOM_STATE) -> Pipeline:
    """Prétraitement + modèle dans un seul objet.

    - Seules `Amount` et `Time` sont normalisées : les V1..V28 sont déjà
      issues d'une ACP (centrées, réduites).
    - `class_weight="balanced"` compense le déséquilibre (0,17 % de fraudes).
    - Le même objet sert à l'entraînement et à l'inférence : le prétraitement
      est donc identique des deux côtés (pas de training-serving skew).
    """
    # TODO 3 : normalisation de Amount et Time uniquement
    pretraitement = ColumnTransformer(
        [("scaler", StandardScaler(), COLONNES_A_NORMALISER)],
        remainder="passthrough",
    )
    # TODO 4 : régression logistique équilibrée
    modele = LogisticRegression(
        C=C, max_iter=1000, class_weight="balanced", random_state=random_state
    )
    # TODO 5 : assemblage
    return Pipeline([
        ("pretraitement", pretraitement),
        ("modele", modele),
    ])