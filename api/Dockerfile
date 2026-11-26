# Image de l'API SentinelPay (mission M4, partie 2)
# 1. Exporter le modèle @production dans api/model/ (dossier ignoré par Git) :
#      mlflow artifacts download -u "models:/detection_fraude@production" -d api/model
# 2. Construire, depuis la racine du dépôt :
#      docker build -f api/Dockerfile -t sentinelpay/api:v1 .
# 3. Exécuter :
#      docker run --rm -p 8000:8000 sentinelpay/api:v1
# En production, on remplacerait la copie du modèle par MODELE_URI=models:/...@production
# et MLFLOW_TRACKING_URI=http://<serveur mlflow> (modèle chargé au démarrage).
FROM python:3.11-slim

WORKDIR /app

# 1. Dépendances figées (couche mise en cache tant que le lock ne change pas)
# TODO 4 : copier requirements.lock puis installer les dépendances (pip install --no-cache-dir -r ...)


# 2. Code
# TODO 5 : copier pyproject.toml, src/ et api/, puis installer le package (pip install --no-cache-dir --no-deps -e .)


# 3. Le modèle, exporté depuis le registry (stratégie « modèle copié dans l'image »)
COPY api/model/ model/

# 4. Configuration par variables d'environnement (surchargeables au run)
ENV MODELE_URI=/app/model \
    SEUIL_DECISION=0.5

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
# TODO 6 : la commande de démarrage : uvicorn api.app:app, hôte 0.0.0.0, port 8000 (forme JSON)
CMD [...]
