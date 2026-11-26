# ============================================================
#  sentinelpay.monitoring — fonctions fournies
#  - calcul_psi  : indice de stabilité de population (drift)
#  - kpi_fraude  : indicateurs métier de la détection de fraude
#  Vous les UTILISEZ (missions M7 et M8), vous ne les réécrivez pas.
# ============================================================

import numpy as np


def calcul_psi(reference, courant, n_bins=10, epsilon=1e-6):
    """Indice de stabilite de population entre deux distributions.

    Compare la distribution `courant` (production) a la distribution
    `reference` (entrainement) sur une variable continue.

    Les bornes des intervalles sont definies par les quantiles de la
    reference. Un `epsilon` evite les divisions par zero et les log(0)
    lorsque certains intervalles sont vides.

    Reperes usuels :
        PSI < 0.10            -> distribution stable
        0.10 <= PSI < 0.25    -> derive moderee
        PSI >= 0.25           -> derive forte

    Renvoie un float (le PSI).
    """
    reference = np.asarray(reference, dtype=float)
    courant = np.asarray(courant, dtype=float)

    quantiles = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(courant, bins=quantiles)

    ref_prop = ref_counts / ref_counts.sum()
    cur_prop = cur_counts / cur_counts.sum()

    # gestion des intervalles vides / proportions nulles
    ref_prop = np.clip(ref_prop, epsilon, None)
    cur_prop = np.clip(cur_prop, epsilon, None)

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def kpi_fraude(y_true, y_pred, montants, cout_traitement=5.0):
    """Indicateurs metier de la detection de fraude.

    Parametres
    ----------
    y_true : etiquettes reelles (1 = fraude, 0 = legitime)
    y_pred : predictions du modele (1 = fraude predite)
    montants : montant de chaque transaction (colonne Amount)
    cout_traitement : cout suppose d'une fausse alerte (en euros)

    Hypothese de cout : une fraude detectee evite la perte de son
    montant ; une fausse alerte coute `cout_traitement` (verification
    manuelle, gene client).

    Renvoie un dictionnaire de KPI.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    montants = np.asarray(montants, dtype=float)

    vp = (y_pred == 1) & (y_true == 1)   # vraie fraude detectee
    fp = (y_pred == 1) & (y_true == 0)   # fausse alerte
    fn = (y_pred == 0) & (y_true == 1)   # fraude ratee

    montant_evite = montants[vp].sum()
    cout_fp = fp.sum() * cout_traitement

    return {
        "taux_fraude_detectee": round(float(vp.sum() / max(y_true.sum(), 1)), 3),
        "faux_positifs": int(fp.sum()),
        "fraudes_ratees": int(fn.sum()),
        "cout_fraude_evite": round(float(montant_evite), 2),
        "cout_faux_positifs": round(float(cout_fp), 2),
        "gain_net": round(float(montant_evite - cout_fp), 2),
    }
