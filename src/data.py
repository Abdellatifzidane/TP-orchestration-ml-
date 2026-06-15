"""Chargement et decoupage des donnees (baseline)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, RANDOM_STATE, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Charge le jeu de donnees prepare."""
    return pd.read_csv(path)


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Separe features/cible puis train/test (stratifie sur la cible)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
