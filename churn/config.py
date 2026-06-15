"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher un autre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs colonnes
via ces constantes.

Problematique : predire l'acceptation (1) ou le refus (0) d'une demande de carte
de credit a partir des caracteristiques du demandeur.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# churn/config.py -> parents[1] = racine du depot
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Chemin vers le fichier de donnees prepare (genere par `churn.prepare_data`).
DATA_PATH = ROOT / "data" / "dataset.csv"
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

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit-card-approval-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-card-approval-classifier")
