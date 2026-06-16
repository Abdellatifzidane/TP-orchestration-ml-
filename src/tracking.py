"""Configuration partagee du suivi MLflow.

Seance 5 - TP MLflow Tracking (suite)
    Centralise la configuration du tracking pour eviter de la dupliquer dans
    chaque script d'entrainement, et ajoute la tracabilite des donnees
    (dataset lineage).

    Une fois ces fonctions disponibles, les scripts (train, train_models)
    appellent `setup_experiment()` au lieu de repeter `set_tracking_uri` +
    `set_experiment`, et `log_dataset()` pour rattacher le jeu de donnees au
    run courant.
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience.

    - pointe vers `MLFLOW_TRACKING_URI` (uniquement s'il est defini ; sinon on
      laisse MLflow utiliser le stockage local par fichiers ``./mlruns``) ;
    - selectionne (ou cree) l'experience `MLFLOW_EXPERIMENT` ;
    - applique la description (tag special `mlflow.note.content`) et les tags
      `MLFLOW_EXPERIMENT_TAGS` a l'experience.

    L'operation est idempotente (re-appelable sans erreur).
    """
    # S5-8
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)

    uri = MLFLOW_TRACKING_URI or "stockage local (./mlruns)"
    logger.info("Suivi MLflow : %s (experience: %s)", uri, MLFLOW_EXPERIMENT)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant (tracabilite donnees -> modele).

    Rattache au run la source des donnees, le schema et un profil, visibles
    dans l'onglet "Datasets" de l'UI MLflow.

    Parameters
    ----------
    df : pandas.DataFrame
        Donnees a referencer (features + cible).
    context : str
        Role du dataset dans le run, par exemple "training" ou "evaluation".
    name : str, optional
        Nom logique du dataset, par defaut "dataset".
    """
    # S5-9
    dataset = mlflow.data.from_pandas(
        df, source=str(DATA_PATH), targets=TARGET, name=name
    )
    mlflow.log_input(dataset, context=context)
