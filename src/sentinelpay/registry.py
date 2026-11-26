"""Model Registry : enregistrer, promouvoir, recharger (mission M3, partie 2).

Usage, depuis la racine du dépôt, après `python -m sentinelpay.tracking` :
    python -m sentinelpay.registry                     # meilleur run par average_precision
    python -m sentinelpay.registry --critere kpi_gain_net
"""

import argparse

import mlflow
from mlflow import MlflowClient

from sentinelpay.data import charger
from sentinelpay.tracking import EXPERIENCE, TRACKING_URI

NOM_MODELE = "detection_fraude"


def meilleur_run(client: MlflowClient, critere: str):
    """Le run dont la métrique `critere` est la plus élevée.

    On ne sélectionne PAS sur le rappel seul : un modèle qui classe tout en
    fraude obtient un rappel de 1 et une précision catastrophique.
    """
    experience = client.get_experiment_by_name(EXPERIENCE)
    if experience is None:
        raise RuntimeError("Expérience introuvable : lancez d'abord python -m sentinelpay.tracking")
    # TODO 5 : client.search_runs(...) trié par `metrics.<critere>` décroissant,
    #          max_results=1 ; lever RuntimeError s'il n'y a aucun run.
    raise NotImplementedError("TODO 5")


def main(critere: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    run = meilleur_run(client, critere)
    print(f"Meilleur run selon {critere} : {run.info.run_id} "
          f"({run.data.params.get('modele')} {dict((k, v) for k, v in run.data.params.items() if k != 'modele')}) "
          f"| recall={run.data.metrics['recall']:.3f} precision={run.data.metrics['precision']:.3f} "
          f"AP={run.data.metrics['average_precision']:.3f}")

    # 1. Enregistrer le modèle du run dans le registry (nouvelle version)
    # TODO 6 : enregistrer le modèle du run (URI runs:/<run_id>/model) sous le nom
    #          NOM_MODELE avec mlflow.register_model ; afficher le numéro de version.
    version = ...

    # 2. Poser l'alias production sur cette version (et documenter)
    # TODO 7 : poser l'alias "production" sur cette version (set_registered_model_alias).
    ...
    client.set_model_version_tag(NOM_MODELE, version.version, "critere_de_selection", critere)
    client.update_model_version(NOM_MODELE, version.version,
                                description=f"Sélectionné sur {critere} (run {run.info.run_id[:8]})")

    # 3. Recharger depuis le registry par alias et prédire
    # TODO 8 : recharger le modèle par son alias (URI models:/<nom>@production).
    modele = ...
    echantillon = charger().drop(columns=["Class"]).head(5)
    print("Prédictions du modèle @production :", modele.predict(echantillon).tolist())

    # 4. Inventaire des versions
    for v in client.search_model_versions(f"name='{NOM_MODELE}'"):
        detail = client.get_model_version(NOM_MODELE, v.version)   # search_model_versions ne renvoie pas les alias
        print(f"  v{v.version}  aliases={detail.aliases}  run={v.run_id[:8]}  {v.description or ''}")
    print("\nServing : mlflow models serve -m \"models:/detection_fraude@production\" --port 5001 --no-conda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--critere", default="average_precision",
                        help="métrique de sélection (average_precision, recall, kpi_gain_net…)")
    main(parser.parse_args().critere)
