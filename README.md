# MLOps IABD (ESGI) : supports de cours et projet etudiant

Ce depot regroupe les supports de cours (slides) et le squelette du projet fil
rouge a completer (`todo/`) du module d'orchestration Machine Learning.

## Contexte

Le cours s'appuie sur un cas d'usage de classification binaire tabulaire. En
cours, la demonstration se fait sur la prediction de churn client ; de votre
cote, vous adaptez le squelette `todo/` a **votre propre** probleme de
classification binaire (voir `todo/README.md`).

Chaque seance est un TP ou l'on complete des marqueurs `TODO (Sx-n)` dans un
code de depart deja fonctionnel.

## Plan des seances

Le cours est organise en 5 modules sur 20 seances (un fichier `Sx_*.pptx` par
seance, dans `slides/`).

- Module 1 : Fondations (S1 a S3) : intro MLOps, mise en place de
  l'environnement, baseline du fil rouge.
- Module 2 : Entrainement (S4 a S7) : reproductibilite/validation, MLflow
  Tracking, Optuna + Model Registry, AutoML & SHAP.
- Module 3 : Conteneurisation & qualite (S8 a S11) : Docker, conteneurisation
  de l'entrainement, pytest, tests donnees/modele (QCM intermediaire en S11).
- Module 4 : Deploiement API (S12 a S15) : FastAPI, API ML reliee au registry,
  conteneurisation + docker-compose, tests d'API.
- Module 5 : Orchestration & CI/CD (S16 a S19) : concepts + Airflow, DAG de
  re-entrainement, GitHub Actions CI puis CD.
- Cloture (S20) : bonus deploiement AWS (ECR + App Runner/Lambda) et prepa
  soutenance.

## Stack technique

- Python 3.13 (environnement gere par uv)
- MLflow (tracking + registry) via docker compose
- FastAPI + uvicorn pour servir le modele
- Airflow pour l'orchestration du re-entrainement
- Docker + docker-compose pour tous les services
- GitHub Actions pour l'integration continue

## Arborescence

```
mlops-iabd-esgi/
  README.md                 ce fichier
  .gitignore                exclusions (secrets, caches, artefacts ML)
  slides/                   supports de cours (un .pptx par seance) + syllabus
  todo/                     squelette du projet a completer (package mlproject)
    README.md               demarche pedagogique et feuille de route detaillee
    Makefile                cibles d'installation (uv), les autres a completer
    mlproject/              code source a completer (config, train, api...)
    frontend/               squelette frontend Streamlit
    docker/                 Dockerfiles (train a completer, api/frontend fournis)
    docker-compose.yml      squelette d'orchestration
    dags/                   squelette DAG Airflow de re-entrainement
```

## Mise en route

```bash
make -C todo install        # installe les dependances (uv)
export PYTHONPATH=todo       # rend le package mlproject importable
```

Adaptez ensuite `todo/mlproject/config.py` a votre dataset, puis verifiez :

```bash
PYTHONPATH=todo python -m mlproject.train   # doit afficher f1=... roc_auc=...
```

La demarche complete, la feuille de route des exercices et les consignes de
suivi GitHub sont detaillees dans `todo/README.md`.
