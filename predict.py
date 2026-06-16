"""Prediction par lot a partir du modele entraine.

Script utilitaire (hors du package src/) : charge ``models/model.joblib`` et
applique le pipeline complet (pre-processing + classifieur) a un fichier
d'entree CSV ou JSON, puis affiche ou ecrit les predictions (classe + proba).

Le modele est un Pipeline scikit-learn dont l'etape de feature engineering est
definie dans ``src/features.py`` : il faut donc ``src`` sur le PYTHONPATH pour
le deserialiser.

Usage :
    PYTHONPATH=src python predict.py --input data/dataset.csv
    PYTHONPATH=src python predict.py --input demandeur.json --output predictions.csv
    PYTHONPATH=src python predict.py --input data/dataset.csv --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config import MODEL_DIR, TARGET


def load_input(path: Path) -> pd.DataFrame:
    """Charger les donnees a predire depuis un CSV ou un JSON (objet ou liste)."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    return pd.read_csv(path)


def predict(
    input_path: str | Path,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predire sur un fichier d'entree et renvoyer les donnees + prediction/probabilite."""
    model_path = Path(model_path) if model_path else MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Modele introuvable : {model_path} (lancez 'make train' d'abord)")
    model = joblib.load(model_path)

    df = load_input(Path(input_path))
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    proba = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["probability"] = proba.round(4)
    out["prediction"] = (proba >= threshold).astype(int)
    return out


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description="Prediction par lot (modele entraine)")
    parser.add_argument("--input", required=True, help="Fichier d'entree (.csv ou .json)")
    parser.add_argument(
        "--model", default=None, help="Chemin du modele (defaut: models/model.joblib)"
    )
    parser.add_argument("--output", default=None, help="CSV de sortie (sinon affichage stdout)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de decision (defaut 0.5)")
    args = parser.parse_args()

    out = predict(args.input, args.model, args.threshold)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"{len(out)} predictions ecrites dans {args.output}")
    else:
        print(out[["prediction", "probability"]].to_string())


if __name__ == "__main__":
    main()
