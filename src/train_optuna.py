"""Optimisation d'hyperparametres avec Optuna.

Optimise trois familles de modeles (Random Forest, XGBoost, LightGBM) avec
Optuna (sampler TPE), compare leurs performances et persiste la meilleure dans
``models/model.joblib``.

Chaque famille est suivie dans MLflow (un run par famille, imbrique sous un run
parent ``optuna-compare``, plus un run par essai Optuna). La configuration du
tracking et la tracabilite des donnees sont mutualisees dans ``tracking.py``
(``setup_experiment`` / ``log_dataset``). Le meilleur modele est enregistre dans
le Model Registry sous ``MODEL_NAME`` uniquement si un serveur de tracking
http(s) est configure.

Lancement :
    PYTHONPATH=src python src/train_optuna.py             # make train-optuna
    PYTHONPATH=src python src/train_optuna.py --n-trials 50 --cv 3
    PYTHONPATH=src python src/train_optuna.py --no-mlflow  # desactive le suivi MLflow
"""
from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import optuna.samplers
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config
from data import load_data, split
from evaluation import log_shap_summary
from features import build_preprocessor
from tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Le ColumnTransformer renvoie un tableau numpy sans noms de colonnes lors du
# scoring interne de la validation croisee : on neutralise l'avertissement
# correspondant, sans incidence sur les predictions.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass
class ModelSpec:
    """Specification d'une famille de modeles a optimiser avec Optuna."""

    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des familles de modeles a optimiser (RF / XGB / LGBM)."""
    rs = config.RANDOM_STATE

    def suggest_rf(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }

    def build_rf(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            RandomForestClassifier(
                random_state=rs, class_weight="balanced", n_jobs=-1, **params
            ),
        )

    def suggest_xgb(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

    def build_xgb(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            XGBClassifier(random_state=rs, eval_metric="logloss", n_jobs=-1, **params),
        )

    def suggest_lgbm(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }

    def build_lgbm(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            LGBMClassifier(
                random_state=rs, verbose=-1, class_weight="balanced", n_jobs=-1, **params
            ),
        )

    return [
        ModelSpec("random_forest", suggest_rf, build_rf),
        ModelSpec("xgboost", suggest_xgb, build_xgb),
        ModelSpec("lightgbm", suggest_lgbm, build_lgbm),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assembler le preprocessing et un classifieur dans un pipeline."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : ROC AUC moyen de validation croisee (a maximiser)."""
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)
    scores = cross_val_score(pipeline, x_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    return float(scores.mean())


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    """Lancer l'etude Optuna (sampler TPE) pour une famille de modeles."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
    )
    return study


@dataclass
class FamilyResult:
    """Resultat d'optimisation d'une famille de modeles."""

    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray


def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    """Optimiser une famille de modeles avec Optuna et l'evaluer sur le test."""
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.3f  test_roc_auc=%.3f  params=%s",
        spec.name,
        study.best_value,
        test_roc_auc,
        study.best_params,
    )
    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=test_roc_auc,
        preds=preds,
    )


def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    """Logger une famille de modeles dans un run MLflow imbrique."""
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        for trial in result.study.trials:
            with mlflow.start_run(run_name=f"{result.spec.name}-trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_roc_auc", trial.value)

        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", result.study.best_value)
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        if register_as and model_info.registered_model_version:
            describe_registered_version(
                register_as,
                model_info.registered_model_version,
                result,
                n_trials,
                cv,
            )


def describe_registered_version(
    name: str,
    version: str,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    """Documenter une version enregistree dans le Model Registry (description + tags)."""
    client = mlflow.MlflowClient()
    description = (
        f"Famille {result.spec.name} optimisee par Optuna (TPE, {n_trials} essais, cv={cv}). "
        f"Meilleurs hyperparametres : {result.study.best_params}. "
        f"cv_roc_auc={result.study.best_value:.3f}, test_roc_auc={result.test_roc_auc:.3f}."
    )
    client.update_model_version(name, version, description=description)

    tags = {
        "model_family": result.spec.name,
        "search_method": "optuna-tpe",
        "n_trials": str(n_trials),
        "cv": str(cv),
        "cv_roc_auc": f"{result.study.best_value:.4f}",
        "test_roc_auc": f"{result.test_roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)


def optimize(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> list[FamilyResult]:
    """Optimiser RF / XGBoost / LightGBM avec Optuna et sauvegarder le meilleur.

    Le meilleur modele (selon le ROC AUC de test) est persiste dans
    ``models/model.joblib``. Avec ``use_mlflow``, chaque famille est suivie dans
    un run imbrique sous ``optuna-compare`` ; le meilleur est enregistre dans le
    Model Registry uniquement si un serveur de tracking http(s) est configure
    (le stockage local par fichiers ne supporte pas le registry).
    """
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    registry_available = False
    if use_mlflow:
        setup_experiment()
        registry_available = config.MLFLOW_TRACKING_URI.startswith(("http://", "https://"))

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleure famille : %s (test_roc_auc=%.3f)", best.spec.name, best.test_roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            log_dataset(df, context="training")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            for result in results:
                register_as = config.MODEL_NAME if (result is best and registry_available) else None
                log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=register_as)
        if registry_available:
            logger.info("Meilleur modele enregistre dans le registry sous '%s'", config.MODEL_NAME)
        else:
            logger.info("Model Registry ignore (pas de serveur de tracking http configure)")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, config.MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", config.MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description="Optimisation Optuna + suivi MLflow")
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre d'essais Optuna par famille de modeles"
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
