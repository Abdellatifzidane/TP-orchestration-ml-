"""Frontend Streamlit : demonstrateur de l'API de classification.

Application qui appelle l'API FastAPI (src/api.py) pour predire l'acceptation
(1) ou le refus (0) d'une demande de carte de credit. L'utilisateur peut
selectionner le modele (Random Forest, XGBoost, LightGBM, ou ``default``).

L'URL de l'API est lue depuis la variable d'environnement API_URL.
L'URL de MLflow est lue depuis MLFLOW_URL (defaut : http://localhost:5000).

Lancement local :
    API_URL=http://127.0.0.1:8000 MLFLOW_URL=http://127.0.0.1:5000 \\
        streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
# URL "publique" utilisee pour les liens cliquables du navigateur (Swagger,
# ReDoc, MLflow). En docker-compose, API_URL pointe sur l'hostname interne
# `http://api:8000`, alors que le navigateur a besoin de l'IP/host externe.
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", API_URL)
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://127.0.0.1:8080")
REPO_URL = os.environ.get(
    "REPO_URL", "https://github.com/Abdellatifzidane/TP-orchestration-ml-"
)

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

MODEL_DESCRIPTIONS = {
    "default": "Modele actif (model.joblib) : meilleur modele selectionne lors du dernier entrainement.",
    "random_forest": "Random Forest - ensemble d'arbres robuste, bonne baseline non lineaire.",
    "xgboost": "XGBoost - boosting de gradient performant et bien tolerant aux donnees mixtes.",
    "lightgbm": "LightGBM - boosting rapide, optimise pour les gros volumes.",
}


def fetch_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


st.set_page_config(page_title="Acceptation carte de credit", layout="wide")

# --- Sidebar : API + selection du modele + liens utiles -------------------
with st.sidebar:
    st.header("API")
    api_url = st.text_input("URL de l'API", value=API_URL)

    health = fetch_json(f"{api_url}/health")
    info = fetch_json(f"{api_url}/model-info")
    if health is None:
        st.error("API injoignable")
    elif health.get("model_loaded"):
        st.success("API OK - modeles charges")
    else:
        st.warning("API OK - aucun modele charge")
    if info:
        st.caption(f"Version servie : {info.get('version', 'unknown')}")

    st.header("Modele")
    models_payload = fetch_json(f"{api_url}/models") or {"available": []}
    available = list(models_payload.get("available", []))
    if not available:
        st.warning("Aucun modele disponible.")
        chosen_model = "default"
    else:
        default_idx = available.index("default") if "default" in available else 0
        chosen_model = st.selectbox("Choisir le modele", options=available, index=default_idx)
        st.caption(MODEL_DESCRIPTIONS.get(chosen_model, "Modele personnalise."))

    st.header("Navigation")
    st.markdown(
        f"""
- [API Swagger UI]({API_PUBLIC_URL}/docs)
- [API ReDoc]({API_PUBLIC_URL}/redoc)
- [MLflow Tracking]({MLFLOW_URL})
- [Airflow UI]({AIRFLOW_URL})
- [Code source]({REPO_URL})
"""
    )

# --- En-tete principal ----------------------------------------------------
st.title("Acceptation de carte de credit")
st.caption(
    "Demonstrateur MLOps : entrainement -> registry MLflow -> API FastAPI -> frontend Streamlit."
)

tab_landing, tab_predict, tab_perf = st.tabs(
    ["Problematique", "Prediction", "Performances des modeles"]
)

# --- Onglet 1 : landing / problematique -----------------------------------
with tab_landing:
    st.markdown(
        "<h1 style='text-align:center; font-size:3em; margin-top:0.5em;'>"
        "Abdellatif ZIDANE</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### Probleme metier

        Les etablissements bancaires recoivent un grand nombre de demandes de
        cartes de credit et doivent decider, pour chaque demandeur, d'**accepter**
        ou de **refuser** la demande. Cette decision doit etre :

        - **Rapide** : un demandeur attend une reponse quasi immediate.
        - **Coherente** : a profil equivalent, la decision doit etre la meme.
        - **Tracable** : la banque doit pouvoir expliquer le choix (regulation).

        Cette application met en demonstration un modele de **classification
        binaire** qui predit, a partir des caracteristiques du demandeur
        (revenus, situation familiale, anciennete d'emploi, etc.), la
        probabilite que sa demande soit acceptee (classe 1) ou refusee
        (classe 0).
        """
    )

    st.markdown("### Architecture")
    st.markdown(
        """
        1. **Entrainement** (``train_models.py``) : compare Random Forest,
           XGBoost et LightGBM via ``GridSearchCV``, suivi MLflow, persistance
           des modeles dans ``models/``.
        2. **API FastAPI** (``src/api.py``) : sert les modeles, expose
           ``/predict``, ``/models``, ``/models/metrics``,
           ``/models/{name}/feature-importance``.
        3. **Frontend Streamlit** (ce fichier) : interroge l'API, affiche les
           performances et les explications.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.link_button("Swagger UI", f"{API_PUBLIC_URL}/docs", use_container_width=True)
    col2.link_button("MLflow Tracking", MLFLOW_URL, use_container_width=True)
    col3.link_button("Airflow UI", AIRFLOW_URL, use_container_width=True)
    col4.link_button("Code source (GitHub)", REPO_URL, use_container_width=True)

# --- Onglet 2 : prediction ------------------------------------------------
with tab_predict:
    st.subheader("Caracteristiques du demandeur")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Genre", GENDER)
            car_owner = st.selectbox("Proprietaire d'une voiture", YES_NO)
            propert_owner = st.selectbox("Proprietaire d'un bien", YES_NO)
            children = st.number_input("Nombre d'enfants", min_value=0, value=0, step=1)
            annual_income = st.number_input(
                "Revenu annuel", min_value=0.0, value=180000.0, step=1000.0
            )
            type_income = st.selectbox("Type de revenu", TYPE_INCOME)
            education = st.selectbox("Niveau d'etudes", EDUCATION)
            marital_status = st.selectbox("Statut marital", MARITAL_STATUS)
        with col2:
            housing_type = st.selectbox("Type de logement", HOUSING_TYPE)
            birthday_count = st.number_input(
                "Jours depuis la naissance (negatif)", value=-18772.0
            )
            employed_days = st.number_input(
                "Jours d'emploi (365243 = non employe)", value=365243
            )
            type_occupation = st.selectbox("Profession", TYPE_OCCUPATION)
            family_members = st.number_input(
                "Membres du foyer", min_value=0, value=2, step=1
            )
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
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", "ACCEPTEE" if prediction == 1 else "REFUSEE")
            c2.metric("Probabilite d'acceptation", f"{probability:.1%}")
            c3.metric("Modele utilise", served_by)
            if prediction == 1:
                st.success("Demande ACCEPTEE (classe 1)")
            else:
                st.error("Demande REFUSEE (classe 0)")
            st.progress(min(max(probability, 0.0), 1.0))

# --- Onglet 3 : performances des modeles ----------------------------------
with tab_perf:
    st.subheader("Comparaison des modeles")
    metrics = fetch_json(f"{api_url}/models/metrics")
    if not metrics or not metrics.get("models"):
        st.info(
            "Aucune metrique disponible. Lance un entrainement complet : "
            "`make train-models` (ou `docker compose --profile train run --rm train`)."
        )
    else:
        rows = metrics["models"]
        st.caption(
            f"Meilleur modele : **{metrics.get('best', 'n/a')}** - "
            f"scoring CV : `{metrics.get('scoring', '?')}` - cv = {metrics.get('cv', '?')}"
        )
        df = pd.DataFrame(rows)
        st.dataframe(
            df[["name", "cv_score", "f1", "roc_auc"]].rename(
                columns={
                    "name": "Modele",
                    "cv_score": "Score CV",
                    "f1": "F1",
                    "roc_auc": "ROC AUC",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        chart_df = df.set_index("name")[["cv_score", "f1", "roc_auc"]]
        st.bar_chart(chart_df)

    st.subheader(f"Importance des variables - modele : {chosen_model}")
    fi = fetch_json(f"{api_url}/models/{chosen_model}/feature-importance?top=15")
    if not fi or not fi.get("features"):
        st.info(
            "Importance des variables indisponible pour ce modele "
            "(la regression logistique n'expose pas ``feature_importances_``)."
        )
    else:
        fi_df = pd.DataFrame(fi["features"]).set_index("feature")
        st.bar_chart(fi_df)
