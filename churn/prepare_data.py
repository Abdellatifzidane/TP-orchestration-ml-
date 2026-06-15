"""Preparation du jeu de donnees.

Fusionne les deux CSV bruts de Kaggle (features + label), nettoie les valeurs
manquantes et la colonne constante, puis ecrit le fichier final attendu par le
pipeline : data/dataset.csv (voir churn.config -> DATA_PATH).

Source : https://www.kaggle.com/datasets/rohitudageri/credit-card-details
Usage  : uv run python -m churn.prepare_data
"""
from __future__ import annotations

import pandas as pd

from churn.config import ROOT

DATA_DIR = ROOT / "data"
FEATURES_CSV = DATA_DIR / "Credit_card.csv"
LABEL_CSV = DATA_DIR / "Credit_card_label.csv"
OUTPUT_CSV = DATA_DIR / "dataset.csv"

# Colonnes inutiles : identifiant + flag constant (une seule valeur => aucune info)
DROP_COLUMNS = ["Ind_ID", "Mobile_phone"]

# Imputation : mediane pour le numerique, "Unknown" pour le categoriel
NUMERIC_IMPUTE = ["Annual_income", "Birthday_count"]
CATEGORICAL_IMPUTE = ["GENDER", "Type_Occupation"]


def main() -> None:
    features = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABEL_CSV)

    df = features.merge(labels, on="Ind_ID", how="inner")
    df = df.rename(columns={"label": "target"})
    df = df.drop(columns=DROP_COLUMNS)

    for col in NUMERIC_IMPUTE:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_IMPUTE:
        df[col] = df[col].fillna("Unknown")

    assert df.isna().sum().sum() == 0, "Il reste des valeurs manquantes."

    df.to_csv(OUTPUT_CSV, index=False)
    n_pos = int(df["target"].sum())
    print(f"Ecrit {OUTPUT_CSV} : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"target=1 : {n_pos} ({n_pos / len(df):.1%})  |  target=0 : {len(df) - n_pos}")


if __name__ == "__main__":
    main()
