# SentinelPay — dépôt de départ du module MLOps

Ce dépôt est le point de départ du fil rouge **SentinelPay** (module « Gestion et déploiement des pipelines MLOps », M2). Vous le faites évoluer du premier au dernier jour : chaque mission y ajoute une brique, et le projet final est l'assemblage de l'ensemble. Les énoncés des missions sont distribués séparément (documents Word).

> **Python 3.11, 3.12 ou 3.13 obligatoire** (pas 3.14 : plusieurs bibliothèques du module, dont `pyarrow`, n'y sont pas encore disponibles et `pip` tente alors de les compiler, ce qui échoue). Windows, macOS et Linux.
>
> **Placez le dépôt hors d'un dossier synchronisé** (Google Drive, OneDrive, iCloud, Dropbox) : l'environnement virtuel contient des dizaines de milliers de fichiers, que la synchronisation ralentit ou abîme. Par exemple `~/projets/sentinelpay-mlops`.

---

## 1. Mise en place (avant la première séance)

Toutes les commandes se lancent **depuis la racine du dépôt**.

```bash
# 0. Vérifier la version de Python qui va créer l'environnement : elle doit afficher 3.11, 3.12 ou 3.13
python3.13 --version        # Windows : py -3.13 --version
#    Absente ? Installez Python 3.13 depuis python.org (ou : brew install python@3.13)

# 1. Environnement virtuel, créé explicitement avec Python 3.13
python3.13 -m venv .venv    # Windows : py -3.13 -m venv .venv
# Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# macOS / Linux :
source .venv/bin/activate

# 2. Dépendances (compter 5 à 10 minutes)
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .          # installe le package src/sentinelpay en mode éditable

# 3. Données : placez creditcard.csv (fichiers de la classe) dans data/

# 4. Vérification
python tools/check_env.py
```

**En cas d'erreur `cmake` / « Failed building wheel for pyarrow »** pendant l'installation : votre environnement a été créé avec un Python trop récent. Supprimez-le (`rm -rf .venv`, ou supprimez le dossier `.venv` sous Windows) et recommencez à l'étape 0. `python --version`, une fois l'environnement activé, doit afficher 3.11 à 3.13.

Prérequis hors Python : **Git** (obligatoire), un **compte GitHub** personnel, **Docker Desktop** (recommandé à partir du jour 3 ; un repli sans Docker est prévu dans les énoncés).

---

## 2. Structure du dépôt

<table style="width:100%; border-collapse:collapse; text-align:left;">
<thead style="background:#bfbfbf;">
<tr><th style="padding:8px;">Dossier / fichier</th><th style="padding:8px;">Rôle</th><th style="padding:8px;">Rempli par</th></tr>
</thead>
<tbody>
<tr style="background:white;"><td style="padding:8px;"><code>data/</code></td><td style="padding:8px;">Données historiques et lots de production (jamais dans Git)</td><td style="padding:8px;">vous (M2 : DVC)</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>notebooks/notebook_initial_sale.py</code></td><td style="padding:8px;">Le script « artisanal » hérité de l'équipe risque : point de départ à assainir</td><td style="padding:8px;">fourni (M2)</td></tr>
<tr style="background:white;"><td style="padding:8px;"><code>src/sentinelpay/</code></td><td style="padding:8px;">Code du projet : <code>data.py</code>, <code>pipeline.py</code>, <code>evaluate.py</code> (TODO 1-6, M2), <code>train.py</code> (fourni), <code>monitoring.py</code> (fourni)</td><td style="padding:8px;">vous (M2 à M7)</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>tests/</code></td><td style="padding:8px;">Tests de données et de modèle (pytest, TODO 7-8 en M2)</td><td style="padding:8px;">vous (M2, M6)</td></tr>
<tr style="background:white;"><td style="padding:8px;"><code>api/</code></td><td style="padding:8px;">API de prédiction FastAPI et <code>Dockerfile</code></td><td style="padding:8px;">vous (M4)</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>pipelines/</code></td><td style="padding:8px;">Pipeline d'entraînement Kubeflow (composants, pipeline, YAML compilé)</td><td style="padding:8px;">vous (M5)</td></tr>
<tr style="background:white;"><td style="padding:8px;"><code>.github/workflows/</code></td><td style="padding:8px;">Chaîne CI / CT / CD GitHub Actions</td><td style="padding:8px;">vous (M6)</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>monitoring/</code></td><td style="padding:8px;">Surveillance des lots de production, rapports de dérive, KPI</td><td style="padding:8px;">vous (M7)</td></tr>
<tr style="background:white;"><td style="padding:8px;"><code>tools/</code></td><td style="padding:8px;"><code>check_env.py</code> (vérification), <code>make_prod_batches.py</code> (générateur de lots de production)</td><td style="padding:8px;">fournis</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>docs/</code></td><td style="padding:8px;">Vos comptes rendus de mission (<code>m3.md</code>…), schémas, runbook</td><td style="padding:8px;">vous</td></tr>
</tbody>
</table>

---

## 3. MLflow : un seul suivi, en base SQLite

Pour que l'interface, vos scripts, le pipeline et le serving utilisent le **même** suivi, le module fixe le tracking sur une base SQLite locale et les artefacts dans `mlartifacts/` :

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

Interface (port 5000) :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Depuis MLflow 3, le stockage « fichiers » (`./mlruns`) est en maintenance et refuse les nouveaux usages : utilisez bien la base SQLite.

---

## 4. Lots de production

Les missions M6 à M8 utilisent des lots mensuels qui simulent la production de SentinelPay (dérive des montants, nouveaux profils de fraude à partir du 4e mois, étiquettes livrées un mois plus tard). Ils se génèrent une fois, de façon déterministe :

```bash
python tools/make_prod_batches.py
```

---

## 5. Repartir de zéro

Supprimer `mlflow.db`, `mlruns/`, `mlartifacts/` et `local_outputs/` réinitialise MLflow et les exécutions locales Kubeflow. Les données et le code ne sont pas touchés.

---

## 6. Problèmes fréquents

<table style="width:100%; border-collapse:collapse; text-align:left;">
<thead style="background:#bfbfbf;">
<tr><th style="padding:8px;">Symptôme</th><th style="padding:8px;">Cause probable / solution</th></tr>
</thead>
<tbody>
<tr style="background:white;"><td style="padding:8px;"><code>ModuleNotFoundError</code></td><td style="padding:8px;">venv non activé, ou <code>pip install -r requirements.txt</code> non fait dedans</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>creditcard.csv introuvable</code></td><td style="padding:8px;">Fichier absent de <code>data/</code>, ou commande lancée hors de la racine</td></tr>
<tr style="background:white;"><td style="padding:8px;"><code>filesystem tracking backend … maintenance mode</code></td><td style="padding:8px;">Un script utilise encore <code>file:./mlruns</code> : passer à <code>sqlite:///mlflow.db</code></td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>Address already in use</code></td><td style="padding:8px;">Port occupé : <code>--port 5002</code> (MLflow) ou <code>--port 8001</code> (API)</td></tr>
<tr style="background:white;"><td style="padding:8px;">Expérience ou modèle introuvable</td><td style="padding:8px;">Mission précédente non exécutée, mauvais dossier, ou alias <code>production</code> absent</td></tr>
<tr style="background:#f5f5f5;"><td style="padding:8px;"><code>could not get source code</code> (Kubeflow)</td><td style="padding:8px;">Les composants <code>@dsl.component</code> doivent être dans un fichier <code>.py</code>, pas tapés dans l'interpréteur</td></tr>
<tr style="background:white;"><td style="padding:8px;">Erreur d'exécution de la politique (PowerShell)</td><td style="padding:8px;"><code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</code> puis réactiver le venv</td></tr>
</tbody>
</table>
