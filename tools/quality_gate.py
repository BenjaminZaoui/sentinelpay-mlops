#!/usr/bin/env python3
# ============================================================
#  Porte de qualité — SentinelPay (mission M6)
#  Lit artefacts/metriques.json (produit par `python -m sentinelpay.train`),
#  compare aux seuils, écrit un résumé Markdown (GitHub : $GITHUB_STEP_SUMMARY)
#  et sort en erreur (code 1) si le modèle n'est pas acceptable.
#
#  Usage : python tools/quality_gate.py [--recall 0.75] [--auc 0.90] [--ap 0.5]
# ============================================================
import argparse
import json
import os
import sys
from pathlib import Path

METRIQUES = Path("artefacts/metriques.json")


def main(seuils: dict) -> int:
    if not METRIQUES.exists():
        print(f"{METRIQUES} introuvable : lancer d'abord python -m sentinelpay.train", file=sys.stderr)
        return 2
    m = json.loads(METRIQUES.read_text())
    lignes, ok_global = [], True
    for cle, seuil in seuils.items():
        valeur = m.get(cle)
        ok = valeur is not None and valeur >= seuil
        ok_global &= ok
        lignes.append(f"| `{cle}` | {valeur:.4f} | ≥ {seuil} | {'✅' if ok else '❌'} |")
    resume = "\n".join([
        f"## Porte de qualité — {'ACCEPTÉ' if ok_global else 'REFUSÉ'}",
        "", "| Métrique | Valeur | Seuil | Verdict |", "|---|---|---|---|", *lignes, "",
        f"C = {m.get('C')} · {m.get('n_test')} transactions de test, {m.get('n_fraudes_test')} fraudes",
    ])
    print(resume)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(resume + "\n", encoding="utf-8")
    return 0 if ok_global else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recall", type=float, default=0.75)
    p.add_argument("--auc", type=float, default=0.90)
    p.add_argument("--ap", type=float, default=0.50, help="average precision minimale")
    a = p.parse_args()
    sys.exit(main({"recall": a.recall, "roc_auc": a.auc, "average_precision": a.ap}))
