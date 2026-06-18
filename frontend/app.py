"""Frontend Streamlit : demonstrateur de l'API de classification.

Application qui appelle l'API FastAPI (src/api.py) pour predire l'acceptation
(1) ou le refus (0) d'une demande de carte de credit. L'utilisateur peut choisir
le modele a interroger parmi ceux exposes par ``GET /models``.

L'URL de l'API est lue depuis la variable d'environnement API_URL (en docker
compose, l'API est joignable via le nom de service `api`).

Lancement local : API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Valeurs possibles des variables categorielles (issues du dataset).
GENDER = ["M", "F", "Unknown"]
YES_NO = ["Y", "N"]
TYPE_INCOME = ["Working", "Commercial associate", "Pensioner", "State servant"]
EDUCATION = [
    "Higher education",
    "Secondary / secondary special",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
MARITAL_STATUS = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPE = [
    "House / apartment",
    "With parents",
    "Rented apartment",
    "Municipal apartment",
    "Co-op apartment",
    "Office apartment",
]
TYPE_OCCUPATION = [
    "Unknown",
    "Laborers",
    "Core staff",
    "Sales staff",
    "Managers",
    "Drivers",
    "High skill tech staff",
    "Accountants",
    "Medicine staff",
    "Cooking staff",
]

# Description des modeles servis par l'API (cf. src/train_models.py).
MODEL_DESCRIPTIONS = {
    "default": "Modele actif (model.joblib) : meilleur modele selectionne lors du dernier entrainement.",
    "random_forest": "Random Forest - ensemble d'arbres robuste, bonne baseline non lineaire.",
    "xgboost": "XGBoost - boosting de gradient performant et bien tolerant aux donnees mixtes.",
    "lightgbm": "LightGBM - boosting rapide, optimise pour les gros volumes.",
}


def fetch_models(url: str) -> list[str]:
    """Recupere la liste des modeles disponibles aupres de l'API."""
    try:
        data = httpx.get(f"{url}/models", timeout=5.0).json()
    except httpx.HTTPError:
        return []
    return list(data.get("available", []))


st.set_page_config(page_title="Acceptation carte de credit", layout="wide")

# --- Landing : problematique du projet ------------------------------------
st.title("Acceptation de carte de credit")
st.markdown(
    """
    ### Problematique

    Les etablissements bancaires recoivent un grand nombre de demandes de
    cartes de credit et doivent decider, pour chaque demandeur, d'**accepter**
    ou de **refuser** la demande. Cette decision doit etre :

    - **Rapide** : un demandeur attend une reponse quasi immediate.
    - **Coherente** : a profil equivalent, la decision doit etre la meme.
    - **Tracable** : la banque doit pouvoir expliquer le choix (regulation).

    Cette application met en demonstration un modele de **classification binaire**
    qui predit, a partir des caracteristiques du demandeur (revenus, situation
    familiale, anciennete d'emploi, etc.), la probabilite que sa demande soit
    acceptee (classe 1) ou refusee (classe 0).
    """
)

with st.expander("Comment ca marche"):
    st.markdown(
        """
        1. Le **service d'entrainement** compare plusieurs familles de modeles
           (Random Forest, XGBoost, LightGBM) avec une recherche
           d'hyperparametres et un suivi MLflow.
        2. Chaque modele est sauvegarde dans ``models/`` et expose par l'**API
           FastAPI** (cf. ``/docs``).
        3. Ce frontend Streamlit appelle l'API : choisissez un modele dans la
           barre laterale, saisissez les caracteristiques d'un demandeur,
           cliquez sur **Predire**.
        """
    )

st.divider()

# --- Barre laterale : API + selection du modele ---------------------------
with st.sidebar:
    st.header("API")
    api_url = st.text_input("URL de l'API", value=API_URL)
    if st.button("Verifier l'etat"):
        try:
            health = httpx.get(f"{api_url}/health", timeout=10.0).json()
            info = httpx.get(f"{api_url}/model-info", timeout=10.0).json()
        except httpx.HTTPError as exc:
            st.error(f"API injoignable : {exc}")
        else:
            if health.get("model_loaded"):
                st.success("API OK - modele charge")
            else:
                st.warning("API OK mais modele non charge")
            st.caption(f"Version servie : {info.get('version', 'unknown')}")

    st.header("Modele")
    available = fetch_models(api_url)
    if not available:
        st.warning("Aucun modele disponible (verifiez l'API).")
        chosen_model = "default"
    else:
        chosen_model = st.selectbox(
            "Choisir le modele",
            options=available,
            index=available.index("default") if "default" in available else 0,
        )
        st.caption(MODEL_DESCRIPTIONS.get(chosen_model, "Modele personnalise."))

# --- Formulaire de prediction ---------------------------------------------
st.subheader("Caracteristiques du demandeur")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Genre", GENDER)
        car_owner = st.selectbox("Proprietaire d'une voiture", YES_NO)
        propert_owner = st.selectbox("Proprietaire d'un bien", YES_NO)
        children = st.number_input("Nombre d'enfants", min_value=0, value=0, step=1)
        annual_income = st.number_input("Revenu annuel", min_value=0.0, value=180000.0, step=1000.0)
        type_income = st.selectbox("Type de revenu", TYPE_INCOME)
        education = st.selectbox("Niveau d'etudes", EDUCATION)
        marital_status = st.selectbox("Statut marital", MARITAL_STATUS)
    with col2:
        housing_type = st.selectbox("Type de logement", HOUSING_TYPE)
        birthday_count = st.number_input("Jours depuis la naissance (negatif)", value=-18772.0)
        employed_days = st.number_input("Jours d'emploi (365243 = non employe)", value=365243)
        type_occupation = st.selectbox("Profession", TYPE_OCCUPATION)
        family_members = st.number_input("Membres du foyer", min_value=0, value=2, step=1)
        work_phone = st.selectbox("Telephone professionnel", [0, 1])
        phone = st.selectbox("Telephone", [0, 1])
        email_id = st.selectbox("Email renseigne", [0, 1])

    submitted = st.form_submit_button("Predire")

if submitted:
    payload = {
        "GENDER": gender,
        "Car_Owner": car_owner,
        "Propert_Owner": propert_owner,
        "CHILDREN": int(children),
        "Annual_income": float(annual_income),
        "Type_Income": type_income,
        "EDUCATION": education,
        "Marital_status": marital_status,
        "Housing_type": housing_type,
        "Birthday_count": float(birthday_count),
        "Employed_days": int(employed_days),
        "Work_Phone": int(work_phone),
        "Phone": int(phone),
        "EMAIL_ID": int(email_id),
        "Type_Occupation": type_occupation,
        "Family_Members": int(family_members),
    }
    try:
        response = httpx.post(
            f"{api_url}/predict",
            params={"model": chosen_model},
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as exc:
        st.error(f"Appel a l'API impossible : {exc}")
    else:
        prediction = result["prediction"]
        probability = result["probability"]
        served_by = result.get("model", chosen_model)
        st.caption(f"Modele utilise : **{served_by}**")
        if prediction == 1:
            st.success("Demande ACCEPTEE (classe 1)")
        else:
            st.error("Demande REFUSEE (classe 0)")
        st.metric("Probabilite d'acceptation", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
