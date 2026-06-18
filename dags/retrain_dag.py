"""DAG Airflow - pipeline de re-entrainement du modele.

Pipeline : preparation / verification des donnees -> entrainement (baseline
LogisticRegression de ``src/train.py``) -> controle qualite via XCom.

La tache ``train`` pousse la metrique f1 dans XCom ; ``check_quality`` la
relit et fait echouer le DAG si elle passe sous ``QUALITY_THRESHOLD``.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# f1 minimal pour considerer l'entrainement comme acceptable.
QUALITY_THRESHOLD = 0.30

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    """Verifie que le dataset est present (pas de regeneration dans ce projet)."""
    from config import DATA_PATH  # noqa: PLC0415

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")
    logger.info("Dataset trouve : %s", DATA_PATH)


def task_train(**context) -> None:
    """Entraine le modele baseline et pousse f1 dans XCom."""
    from train import train  # noqa: PLC0415

    metrics = train(use_mlflow=True)
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])
    logger.info("Metriques : %s", metrics)


def task_check_quality(**context) -> None:
    """Fait echouer le DAG si f1 < QUALITY_THRESHOLD."""
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None:
        raise ValueError("f1 introuvable dans XCom (la tache train a-t-elle bien tourne ?)")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(f"f1={f1:.3f} < seuil {QUALITY_THRESHOLD} : modele rejete")
    logger.info("Qualite OK : f1=%.3f >= seuil %.2f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine la baseline LogReg et controle sa qualite",
    schedule="0 18 * * *",  # tous les jours a 18h00 (Europe/Paris)
    start_date=pendulum.datetime(2024, 1, 1, tz="Europe/Paris"),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
