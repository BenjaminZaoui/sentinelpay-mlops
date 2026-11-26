#!/usr/bin/env python3
# ============================================================
#  Vérification de l'environnement — module MLOps SentinelPay
#  À lancer depuis la racine du dépôt, dans le venv activé :
#     python tools/check_env.py
# ============================================================

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

OK, KO = "OK ", "KO "
problemes = 0


def ligne(etat, texte):
    global problemes
    if etat == KO:
        problemes += 1
    print(f"[{etat}] {texte}")


# 1. Python
v = sys.version_info
ligne(OK if (3, 11) <= (v.major, v.minor) <= (3, 13) else KO,
      f"Python {v.major}.{v.minor}.{v.micro} (3.11 à 3.13 attendu)")
ligne(OK if sys.prefix != sys.base_prefix else KO,
      "environnement virtuel activé" if sys.prefix != sys.base_prefix
      else "aucun environnement virtuel actif (python -m venv .venv puis activation)")

# 2. Paquets Python
for module, nom in [("pandas", "pandas"), ("sklearn", "scikit-learn"), ("mlflow", "mlflow"),
                    ("kfp", "kfp"), ("evidently", "evidently"), ("dvc", "dvc"),
                    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("pytest", "pytest")]:
    try:
        m = importlib.import_module(module)
        ligne(OK, f"{nom} {getattr(m, '__version__', '')}")
    except ImportError:
        ligne(KO, f"{nom} absent : pip install -r requirements.txt")

# 3. Outils en ligne de commande
for outil, obligatoire in [("git", True), ("docker", False)]:
    chemin = shutil.which(outil)
    if chemin:
        try:
            version = subprocess.run([outil, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            version = "présent"
        ligne(OK, version)
    else:
        ligne(KO if obligatoire else OK, f"{outil} introuvable" + ("" if obligatoire else " (optionnel : repli sans Docker prévu)"))

# 4. Données
csv = Path("data/creditcard.csv")
if csv.exists():
    taille = csv.stat().st_size / 1e6
    ligne(OK if taille > 100 else KO, f"data/creditcard.csv présent ({taille:.0f} Mo)")
else:
    ligne(KO, "data/creditcard.csv absent : le récupérer dans « les fichiers de la classe » et le placer dans data/")

print()
print("Environnement prêt." if problemes == 0 else f"{problemes} point(s) à corriger avant la première séance.")
