"""Tests de base de la baseline de classification."""
from __future__ import annotations

import pandas as pd

import config
from features import EMPLOYED_SENTINEL, add_employment_features, build_preprocessor


def test_config_target_and_features():
    assert config.TARGET == "target"
    assert config.NUMERIC_FEATURES, "au moins une feature numerique attendue"
    assert config.CATEGORICAL_FEATURES, "au moins une feature categorielle attendue"
    # La cible ne doit pas figurer parmi les features.
    assert config.TARGET not in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def test_employment_feature_engineering():
    df = pd.DataFrame({"Employed_days": [EMPLOYED_SENTINEL, -1953, 0]})
    out = add_employment_features(df)
    assert list(out["is_employed"]) == [0, 1, 1]
    assert list(out["Employed_days"]) == [0, -1953, 0]


def test_build_preprocessor():
    pre = build_preprocessor()
    ct = pre.named_steps["columns"]
    cols = [name for _, _, name in ct.transformers]
    assert config.NUMERIC_FEATURES in cols
    assert config.CATEGORICAL_FEATURES in cols


def test_model_specs():
    from train_models import build_model_specs

    specs = build_model_specs()
    assert [s.name for s in specs] == ["random_forest", "xgboost", "lightgbm"]
    # Chaque grille cible bien l'etape `clf` du pipeline.
    for spec in specs:
        assert all(key.startswith("clf__") for key in spec.param_grid)
