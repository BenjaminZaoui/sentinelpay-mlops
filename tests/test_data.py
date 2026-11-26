"""Tests de données (mission M2) : le jeu a-t-il la forme attendue ?"""

import pytest

from sentinelpay.data import COLONNES_ATTENDUES, charger, separer


@pytest.fixture(scope="module")
def df():
    return charger()


def test_colonnes_presentes(df):
    assert list(df.columns) == COLONNES_ATTENDUES


def test_aucune_valeur_manquante(df):
    assert df.isna().sum().sum() == 0


def test_taux_de_fraude_dans_la_plage_connue(df):
    # TODO 7 : le taux de fraude doit rester dans une plage plausible
    #          (entre 0,05 % et 2 %) ; sinon les données ne sont pas celles attendues.
    taux = df["Class"].mean()
    assert False, "TODO 7"


def test_separation_stratifiee_et_reproductible(df):
    X1, _, y1, _ = separer(df)
    X2, _, y2, _ = separer(df)
    assert X1.index.equals(X2.index)                     # même graine, même découpage
    assert abs(y1.mean() - df["Class"].mean()) < 0.001   # stratification
