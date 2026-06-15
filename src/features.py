"""Pre-processing de la baseline.

Feature engineering minimal (traitement de la sentinelle Employed_days) puis
standardisation des colonnes numeriques + encodage one-hot des categorielles.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Valeur sentinelle de Employed_days : retraite / non-employe (pas une duree reelle).
EMPLOYED_SENTINEL = 365243


def add_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Traite la sentinelle de Employed_days.

    - cree un flag binaire `is_employed` (0 = retraite / non-employe) ;
    - remet Employed_days a 0 pour ces lignes, sinon la valeur ~365243 fausse
      completement la standardisation.
    """
    df = df.copy()
    sentinel = df["Employed_days"] == EMPLOYED_SENTINEL
    df["is_employed"] = (~sentinel).astype(int)
    df.loc[sentinel, "Employed_days"] = 0
    return df


def build_preprocessor() -> Pipeline:
    columns = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("flag", "passthrough", ["is_employed"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("engineer", FunctionTransformer(add_employment_features)),
            ("columns", columns),
        ]
    )
