"""DAG Airflow - trafic de previsions quotidien.

Planifie l'envoi quotidien d'un lot de previsions a l'API : chaque jour a 10h,
on echantillonne ``N_PREDICTIONS`` lignes du dataset (sans la cible) et on les
envoie en POST /predict pour simuler un flux de production.

L'API est jointe via la variable d'environnement ``API_URL`` (defaut
``http://api:8000`` dans la stack Docker).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """Echantillonne ``N_PREDICTIONS`` lignes et les envoie a POST /predict."""
    import httpx  # noqa: PLC0415

    from config import TARGET  # noqa: PLC0415
    from data import load_data  # noqa: PLC0415

    api_url = os.environ.get("API_URL", "http://api:8000")
    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=N_PREDICTIONS)

    with httpx.Client(base_url=api_url, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()

    logger.info("%d previsions envoyees a %s", N_PREDICTIONS, api_url)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=pendulum.datetime(2024, 1, 1, tz="Europe/Paris"),
    schedule="30 17 * * *",  # tous les jours a 17h30 (Europe/Paris)
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
