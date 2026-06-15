"""Outils d'evaluation partages : graphiques loggues comme artefacts MLflow."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _feature_names(preprocessor, n_features: int) -> list[str]:
    """Recupere les noms de colonnes apres pre-processing, avec repli robuste.

    `preprocessor` est un Pipeline (engineer + columns) : on tente d'abord
    `get_feature_names_out`, puis le ColumnTransformer interne, sinon des noms
    generiques.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        try:
            return list(preprocessor.named_steps["columns"].get_feature_names_out())
        except Exception:
            return [f"f{i}" for i in range(n_features)]


def log_shap_summary(pipeline: Pipeline, x_test, name: str, max_samples: int = 200) -> None:
    """Logger un summary plot SHAP comme artefact MLflow ``shap_summary.png``.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline entraine, avec les etapes ``preprocessor`` et ``clf``.
    x_test : pandas.DataFrame
        Jeu de test utilise pour estimer les valeurs SHAP.
    name : str
        Nom du modele, utilise dans le titre du graphique.
    max_samples : int, optional
        Nombre maximal d'observations utilisees pour le calcul, par defaut 200.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    sample = transformed[:max_samples]
    feature_names = _feature_names(preprocessor, transformed.shape[1])

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample)
    except Exception:  # pragma: no cover - modeles non supportes par TreeExplainer
        logger.warning("SHAP TreeExplainer indisponible pour %s, artefact ignore", name)
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Importance des variables (SHAP) : {name}")
    mlflow.log_figure(fig, "shap_summary.png")
    plt.close(fig)
