"""Configuration de la baseline de classification.

Seul fichier a adapter pour brancher le jeu de donnees : data.py et features.py
lisent leurs colonnes via ces constantes.

Problematique : predire l'acceptation (1) ou le refus (0) d'une demande de carte
de credit a partir des caracteristiques du demandeur.
"""
from __future__ import annotations

from pathlib import Path

# src/config.py -> parents[1] = racine du depot
ROOT = Path(__file__).resolve().parents[1]

# Jeu de donnees prepare (fusion des CSV Kaggle + nettoyage).
DATA_PATH = ROOT / "data" / "dataset.csv"

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
