"""Configuration de la baseline de classification.

Seul fichier a adapter pour brancher le jeu de donnees : data.py et features.py
lisent leurs colonnes via ces constantes.

Problematique : predire l'acceptation (1) ou le refus (0) d'une demande de carte
de credit a partir des caracteristiques du demandeur.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/config.py -> parents[1] = racine du depot
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Jeu de donnees prepare (fusion des CSV Kaggle + nettoyage).
DATA_PATH = ROOT / "data" / "dataset.csv"

# Dossier de sortie des modeles entraines.
MODEL_DIR = ROOT / "models"

# Colonne cible binaire : 1 = demande acceptee, 0 = refusee.
TARGET = "target"

NUMERIC_FEATURES: list[str] = [
    "CHILDREN",
    "Annual_income",
    "Birthday_count",
    "Employed_days",
    "Work_Phone",
    "Phone",
    "EMAIL_ID",
    "Family_Members",
]

CATEGORICAL_FEATURES: list[str] = [
    "GENDER",
    "Car_Owner",
    "Propert_Owner",
    "Type_Income",
    "EDUCATION",
    "Marital_status",
    "Housing_type",
    "Type_Occupation",
]

RANDOM_STATE = 42

# Suivi d'experiences MLflow (surcouche via variables d'environnement).
# MLFLOW_TRACKING_URI vide => stockage local par defaut (dossier ./mlruns).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit-card-approval")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-card-approval-classifier")

# Metadonnees de l'experience MLflow (description + tags), appliquees par
# tracking.setup_experiment(). La description alimente le tag special
# `mlflow.note.content` visible en tete de l'experience dans l'UI.
MLFLOW_EXPERIMENT_DESCRIPTION = (
    "Classification binaire de l'acceptation (1) ou du refus (0) d'une demande "
    "de carte de credit a partir des caracteristiques du demandeur."
)
MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project": "credit-card-approval",
    "task": "binary-classification",
    "course": "mlops-iabd-esgi",
}
