"""API d'inference FastAPI pour le modele de classification.

Sert le modele entraine (``models/model.joblib``) pour predire l'acceptation
(1) ou le refus (0) d'une demande de carte de credit. Le modele est charge une
seule fois au demarrage (lifespan), pas a chaque requete.

Lancement :
    PYTHONPATH=src uv run uvicorn api:app --reload   # make api
    -> documentation interactive sur http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modele au demarrage et le libere a l'arret."""
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists():
        ml["model"] = joblib.load(model_path)
        logger.info("Modele charge depuis %s", model_path)
    else:
        logger.warning("Modele introuvable (%s) : lancez 'make train' d'abord", model_path)
    yield
    ml.clear()


app = FastAPI(title="Credit Card Approval API", version="0.1.0", lifespan=lifespan)


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


@app.get("/health")
def health() -> dict:
    """Verifie que l'API repond et que le modele est charge."""
    return {"status": "ok", "model_loaded": "model" in ml}


@app.post("/predict", response_model=PredictionOut)
def predict(applicant: Applicant) -> PredictionOut:
    """Predit l'acceptation d'une demande a partir des caracteristiques fournies."""
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([applicant.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    """Retourne la version servie (variable d'environnement MODEL_VERSION)."""
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
