"""Pipeline d'entraînement SentinelPay : Kubeflow orchestre, MLflow trace (mission M5, partie 2).

    python pipelines/pipeline_sentinelpay.py --C 1.0            # exécution locale + compilation
    python pipelines/pipeline_sentinelpay.py --C 0.1 --seuil-recall 0.8

Cinq composants :
  preparer  -> jeux d'entraînement et de test (Dataset x2), version des données
  entrainer -> ouvre un run MLflow, journalise paramètres et modèle signé (Model + run_id)
  evaluer   -> métriques et KPI journalisés dans le même run (Metrics)
  valider   -> compare au seuil et au modèle @production ; décision (bool)
  enregistrer -> si valide : version dans le registry + alias "candidate"

Les composants importent le package `sentinelpay` : en local (SubprocessRunner,
use_venv=False) il est disponible ; sur un cluster, on construirait une image de
base qui le contient (base_image="sentinelpay/pipeline:v1").
"""

import argparse
import os

from kfp import compiler, dsl, local
from kfp.dsl import Dataset, Input, Metrics, Model, Output

IMAGE = "python:3.11"  # sur un cluster : une image contenant le package sentinelpay


@dsl.component(base_image=IMAGE)
def preparer(chemin_csv: str, train: Output[Dataset], test: Output[Dataset]) -> str:
    """Charge, vérifie le schéma, sépare (stratifié, graine fixe) ; renvoie la version des données."""
    import hashlib
    from pathlib import Path
    import pandas as pd
    from sentinelpay.data import charger, separer

    df = charger(Path(chemin_csv))
    X_train, X_test, y_train, y_test = separer(df)
    pd.concat([X_train, y_train], axis=1).to_parquet(train.path)
    pd.concat([X_test, y_test], axis=1).to_parquet(test.path)
    # Version des données : le md5 du pointeur DVC s'il existe, sinon celui du fichier.
    pointeur = Path(chemin_csv + ".dvc")
    if pointeur.exists():
        version = [l.split(":")[1].strip() for l in pointeur.read_text().splitlines() if "md5:" in l][0]
    else:
        version = hashlib.md5(Path(chemin_csv).read_bytes()).hexdigest()
    train.metadata["lignes"] = len(X_train)
    test.metadata["lignes"] = len(X_test)
    return version


@dsl.component(base_image=IMAGE)
def entrainer(train: Input[Dataset], C: float, version_donnees: str, tracking_uri: str,
              experience: str, modele: Output[Model]) -> str:
    """Entraîne et journalise dans MLflow ; renvoie l'identifiant du run."""
    import joblib
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from mlflow.models import infer_signature
    from sentinelpay.pipeline import construire_pipeline

    df = pd.read_parquet(train.path)
    X_train, y_train = df.drop(columns=["Class"]), df["Class"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experience)
    with mlflow.start_run(run_name=f"kfp_logreg_C={C}") as run:
        mlflow.log_params({"modele": "logreg", "C": C, "version_donnees": version_donnees})
        mlflow.set_tags({"mission": "M5", "orchestrateur": "kubeflow-local"})
        pipeline = construire_pipeline(C=C).fit(X_train, y_train)
        exemple = X_train.head(5)
        mlflow.sklearn.log_model(pipeline, name="model",
                                 signature=infer_signature(exemple, pipeline.predict(exemple)),
                                 input_example=exemple)
        joblib.dump(pipeline, modele.path)
        modele.metadata["run_id"] = run.info.run_id
        return run.info.run_id


@dsl.component(base_image=IMAGE)
def evaluer(test: Input[Dataset], modele: Input[Model], run_id: str, tracking_uri: str,
            metriques: Output[Metrics]) -> float:
    """Évalue sur le jeu de test, journalise métriques et KPI dans le run ; renvoie le rappel."""
    import joblib
    import mlflow
    import pandas as pd
    from sentinelpay.evaluate import evaluer as evaluer_pipeline
    from sentinelpay.monitoring import kpi_fraude

    df = pd.read_parquet(test.path)
    X_test, y_test = df.drop(columns=["Class"]), df["Class"]
    pipeline = joblib.load(modele.path)
    resultats = evaluer_pipeline(pipeline, X_test, y_test)
    kpi = kpi_fraude(y_test, pipeline.predict(X_test), X_test["Amount"])

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):          # on complète le run ouvert par `entrainer`
        mlflow.log_metrics({k: v for k, v in resultats.items() if isinstance(v, float)})
        mlflow.log_metrics({f"kpi_{k}": float(v) for k, v in kpi.items()})
    for cle, valeur in resultats.items():
        if isinstance(valeur, float):
            metriques.log_metric(cle, valeur)
    metriques.log_metric("kpi_gain_net", float(kpi["gain_net"]))
    return resultats["recall"]


@dsl.component(base_image=IMAGE)
def valider(run_id: str, seuil_recall: float, tracking_uri: str, nom_modele: str) -> bool:
    """Porte de qualité : rappel >= seuil ET average_precision >= celle du modèle @production (à 0,02 près)."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    candidat = client.get_run(run_id).data.metrics
    ap_prod = None
    try:
        version_prod = client.get_model_version_by_alias(nom_modele, "production")
        ap_prod = client.get_run(version_prod.run_id).data.metrics.get("average_precision")
    except Exception:
        pass  # pas encore de modèle en production : la comparaison est ignorée
    # TODO 3 : deux conditions — le rappel du candidat atteint le seuil, ET son
    #          average_precision est >= celle du modèle en production moins 0,02
    #          (tolérance : les jeux de test diffèrent), si ap_prod n'est pas None.
    ok_recall = ...
    ok_ap = ...
    print(f"recall={candidat['recall']:.3f} (seuil {seuil_recall}) -> {ok_recall} ; "
          f"AP={candidat['average_precision']:.3f} vs production {ap_prod} -> {ok_ap}")
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("validation", "ok" if ok_recall and ok_ap else "refuse")
    return bool(ok_recall and ok_ap)


@dsl.component(base_image=IMAGE)
def enregistrer(run_id: str, valide: bool, tracking_uri: str, nom_modele: str) -> str:
    """Si valide : nouvelle version dans le registry + alias candidate. Renvoie la version ou 'aucune'."""
    import mlflow
    from mlflow import MlflowClient

    if not valide:
        print("Modèle refusé par la validation : pas d'enregistrement.")
        return "aucune"
    mlflow.set_tracking_uri(tracking_uri)
    # TODO 4 : enregistrer le modèle du run (runs:/<run_id>/model) sous nom_modele,
    #          puis poser l'alias "candidate" sur la version créée.
    version = ...
    client = MlflowClient()
    ...
    client.update_model_version(nom_modele, version.version, description=f"Produit par le pipeline Kubeflow (run {run_id[:8]})")
    print(f"{nom_modele} v{version.version} enregistré avec l'alias candidate")
    return str(version.version)


@dsl.pipeline(name="sentinelpay-entrainement",
              description="preparer -> entrainer -> evaluer -> valider -> enregistrer (MLflow)")
def pipeline_sentinelpay(chemin_csv: str, tracking_uri: str, C: float = 1.0, seuil_recall: float = 0.75,
                         experience: str = "detection_fraude", nom_modele: str = "detection_fraude") -> str:
    t_prep = preparer(chemin_csv=chemin_csv)
    t_train = entrainer(train=t_prep.outputs["train"], C=C, version_donnees=t_prep.outputs["Output"],
                        tracking_uri=tracking_uri, experience=experience)
    # TODO 5 : enchaîner evaluer (jeu de test, modèle, run_id, tracking_uri),
    #          valider (run_id, seuil_recall, tracking_uri, nom_modele) — avec .after(t_eval)
    #          car valider lit les métriques que evaluer a journalisées —,
    #          puis enregistrer (run_id, valide=sortie de valider, tracking_uri, nom_modele) ;
    #          renvoyer la sortie d'enregistrer.
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seuil-recall", type=float, default=0.75)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--chemin-csv", default="data/creditcard.csv",
                        help="jeu d'entraînement (M7 : jeu enrichi des lots de production étiquetés)")
    args = parser.parse_args()

    compiler.Compiler().compile(pipeline_sentinelpay, "pipelines/pipeline_sentinelpay.yaml")
    print("Compilé : pipelines/pipeline_sentinelpay.yaml")
    if not args.compile_only:
        # Les composants tournent dans des sous-processus : chemins absolus obligatoires.
        local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root="./local_outputs")
        execution = pipeline_sentinelpay(
            chemin_csv=os.path.abspath(args.chemin_csv),
            tracking_uri="sqlite:///" + os.path.abspath("mlflow.db"),
            C=args.C, seuil_recall=args.seuil_recall,
        )
        print("Version enregistrée :", execution.outputs["Output"])
