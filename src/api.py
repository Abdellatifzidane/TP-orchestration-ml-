"""API d'inference FastAPI pour le modele de classification.

Sert le(s) modele(s) entraine(s) presents dans ``models/`` pour predire
l'acceptation (1) ou le refus (0) d'une demande de carte de credit. Tous les
``*.joblib`` du dossier sont charges une seule fois au demarrage (lifespan) :
``model.joblib`` est le defaut, les autres (``random_forest.joblib``,
``xgboost.joblib``, ``lightgbm.joblib``, ...) sont selectionnables via
``?model=<nom>`` sur ``/predict``.

Lancement :
    PYTHONPATH=src uv run uvicorn api:app --reload   # make api
    -> documentation interactive sur http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "default"

ml: dict = {"models": {}}


def _load_models() -> dict:
    """Charge tous les ``*.joblib`` de ``MODEL_DIR`` en memoire."""
    models: dict = {}
    if not MODEL_DIR.exists():
        return models
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        name = DEFAULT_MODEL if path.stem == "model" else path.stem
        try:
            models[name] = joblib.load(path)
            logger.info("Modele '%s' charge depuis %s", name, path)
        except Exception:  # noqa: BLE001
            logger.exception("Echec du chargement du modele %s", path)
    return models


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge tous les modeles disponibles au demarrage et les libere a l'arret."""
    ml["models"] = _load_models()
    if not ml["models"]:
        logger.warning(
            "Aucun modele trouve dans %s : lancez 'make train' ou 'make train-models'",
            MODEL_DIR,
        )
    yield
    ml.clear()


app = FastAPI(title="Credit Card Approval API", version="0.2.0", lifespan=lifespan)


class Applicant(BaseModel):
    """Caracteristiques d'un demandeur (memes colonnes que le dataset d'entrainement)."""

    GENDER: str = Field(..., description="M / F / Unknown")
    Car_Owner: str = Field(..., description="Proprietaire d'une voiture (Y/N)")
    Propert_Owner: str = Field(..., description="Proprietaire d'un bien (Y/N)")
    CHILDREN: int = Field(..., ge=0, description="Nombre d'enfants")
    Annual_income: float = Field(..., ge=0, description="Revenu annuel")
    Type_Income: str = Field(..., description="Type de revenu")
    EDUCATION: str = Field(..., description="Niveau d'etudes")
    Marital_status: str = Field(..., description="Statut marital")
    Housing_type: str = Field(..., description="Type de logement")
    Birthday_count: float = Field(..., description="Jours depuis la naissance (negatif)")
    Employed_days: int = Field(..., description="Jours d'emploi (365243 = non employe)")
    Work_Phone: int = Field(..., ge=0, le=1, description="Telephone professionnel (0/1)")
    Phone: int = Field(..., ge=0, le=1, description="Telephone (0/1)")
    EMAIL_ID: int = Field(..., ge=0, le=1, description="Email renseigne (0/1)")
    Type_Occupation: str = Field(..., description="Profession (Unknown si inconnue)")
    Family_Members: int = Field(..., ge=0, description="Nombre de membres du foyer")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "GENDER": "M",
                    "Car_Owner": "Y",
                    "Propert_Owner": "Y",
                    "CHILDREN": 0,
                    "Annual_income": 180000.0,
                    "Type_Income": "Pensioner",
                    "EDUCATION": "Higher education",
                    "Marital_status": "Married",
                    "Housing_type": "House / apartment",
                    "Birthday_count": -18772.0,
                    "Employed_days": 365243,
                    "Work_Phone": 0,
                    "Phone": 0,
                    "EMAIL_ID": 0,
                    "Type_Occupation": "Unknown",
                    "Family_Members": 2,
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    """Reponse de l'endpoint /predict."""

    prediction: int = Field(..., description="Classe predite : 1 = acceptee, 0 = refusee")
    probability: float = Field(..., description="Probabilite de la classe 1")
    model: str = Field(..., description="Nom du modele utilise pour la prediction")


@app.get("/health")
def health() -> dict:
    """Verifie que l'API repond et que au moins un modele est charge."""
    return {"status": "ok", "model_loaded": bool(ml.get("models"))}


@app.get("/models")
def list_models() -> dict:
    """Liste les modeles disponibles pour l'inference."""
    return {
        "available": sorted(ml.get("models", {}).keys()),
        "default": DEFAULT_MODEL if DEFAULT_MODEL in ml.get("models", {}) else None,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(applicant: Applicant, model: str = DEFAULT_MODEL) -> PredictionOut:
    """Predit l'acceptation d'une demande. Choix du modele via ``?model=<nom>``."""
    models = ml.get("models", {})
    if not models:
        raise HTTPException(status_code=503, detail="Aucun modele charge")
    chosen = models.get(model)
    if chosen is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modele '{model}' introuvable (disponibles : {sorted(models.keys())})",
        )
    row = pd.DataFrame([applicant.model_dump()])
    proba = float(chosen.predict_proba(row)[0, 1])
    return PredictionOut(
        prediction=int(proba >= 0.5),
        probability=round(proba, 4),
        model=model,
    )


@app.get("/model-info")
def model_info() -> dict:
    """Retourne la version servie (variable d'environnement MODEL_VERSION)."""
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}


@app.get("/models/metrics")
def models_metrics() -> dict:
    """Retourne les metriques par modele issues du dernier entrainement."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return {"models": [], "best": None}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"metrics.json illisible : {exc}")  # noqa: B904


@app.get("/models/{name}/feature-importance")
def feature_importance(name: str, top: int = 15) -> dict:
    """Retourne les importances de features du modele si disponibles."""
    models = ml.get("models", {})
    pipe = models.get(name)
    if pipe is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modele '{name}' introuvable (disponibles : {sorted(models.keys())})",
        )
    clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else None
    if clf is None or not hasattr(clf, "feature_importances_"):
        raise HTTPException(
            status_code=400,
            detail=f"Modele '{name}' ne supporte pas feature_importances_",
        )
    pre = pipe.named_steps.get("preprocessor")
    try:
        names = list(pre.named_steps["columns"].get_feature_names_out())
    except Exception:  # noqa: BLE001
        names = [f"f{i}" for i in range(len(clf.feature_importances_))]
    importances = np.asarray(clf.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1][:top]
    return {
        "model": name,
        "features": [
            {"feature": names[i], "importance": float(importances[i])} for i in order
        ],
    }
