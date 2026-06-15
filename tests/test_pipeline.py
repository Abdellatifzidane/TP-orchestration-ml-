"""Tests de base du pipeline de classification."""
from __future__ import annotations

from churn import config
from churn.features import build_preprocessor


def test_config_target_and_features():
    assert config.TARGET == "target"
    assert config.NUMERIC_FEATURES, "au moins une feature numerique attendue"
    assert config.CATEGORICAL_FEATURES, "au moins une feature categorielle attendue"
    # La cible ne doit pas figurer parmi les features.
    assert config.TARGET not in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def test_build_preprocessor():
    pre = build_preprocessor()
    cols = [name for _, _, name in pre.transformers]
    assert config.NUMERIC_FEATURES in cols
    assert config.CATEGORICAL_FEATURES in cols
