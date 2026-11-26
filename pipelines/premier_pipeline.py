"""Mon premier pipeline Kubeflow, exécuté en local (mission M5, partie 1).

    python pipelines/premier_pipeline.py            # exécute en local puis compile premier_pipeline.yaml

Trois composants Python « légers » qui s'échangent des artefacts :
charger (Dataset) -> entrainer (Model) -> evaluer (Metrics).
"""

from kfp import compiler, dsl, local
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn"])
def charger(chemin_csv: str, jeu: Output[Dataset]):
    """Lit le CSV et écrit un artefact Dataset (un fichier Parquet)."""
    import pandas as pd
    df = pd.read_csv(chemin_csv)
    df.to_parquet(jeu.path)
    jeu.metadata["lignes"] = len(df)
    jeu.metadata["fraudes"] = int(df["Class"].sum())


@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn"])
def entrainer(jeu: Input[Dataset], C: float, modele: Output[Model]):
    """Entraîne le pipeline scikit-learn et écrit un artefact Model (joblib)."""
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(jeu.path)
    X, y = df.drop(columns=["Class"]), df["Class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("pretraitement", ColumnTransformer([("scale", StandardScaler(), ["Amount", "Time"])], remainder="passthrough")),
        ("modele", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=42)),
    ]).fit(X_train, y_train)
    joblib.dump(pipeline, modele.path)
    modele.metadata["C"] = C
    modele.metadata["framework"] = "scikit-learn"


@dsl.component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn"])
def evaluer(jeu: Input[Dataset], modele: Input[Model], metriques: Output[Metrics]) -> float:
    """Évalue sur le jeu de test (même séparation) et écrit un artefact Metrics ; renvoie le rappel."""
    import joblib
    import pandas as pd
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(jeu.path)
    X, y = df.drop(columns=["Class"]), df["Class"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipeline = joblib.load(modele.path)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    resultats = {
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
    }
    # TODO 1 : journaliser chaque métrique dans l'artefact Metrics (metriques.log_metric)
    #          et renvoyer le rappel.
    ...


@dsl.pipeline(name="sentinelpay-premier-pipeline", description="charger -> entrainer -> evaluer")
def premier_pipeline(chemin_csv: str = "data/creditcard.csv", C: float = 1.0) -> float:
    t_charger = charger(chemin_csv=chemin_csv)
    # TODO 2 : appeler entrainer avec le Dataset produit par charger (t_charger.outputs["jeu"]) et C,
    #          puis evaluer avec le même Dataset et le Model produit par entrainer ;
    #          renvoyer la sortie d'evaluer (t_evaluer.outputs["Output"]).
    ...


if __name__ == "__main__":
    import os
    # 1. Exécution locale, sans cluster : chaque composant tourne dans un sous-processus.
    local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root="./local_outputs")
    execution = premier_pipeline(chemin_csv=os.path.abspath("data/creditcard.csv"), C=1.0)
    print("Rappel renvoyé par le pipeline :", execution.outputs["Output"])

    # 2. Compilation : le même pipeline, décrit en YAML, prêt pour un cluster Kubeflow.
    compiler.Compiler().compile(premier_pipeline, "pipelines/premier_pipeline.yaml")
    print("Compilé : pipelines/premier_pipeline.yaml")
