# MLOps - Acceptation de demande de carte de credit

Projet fil rouge du module d'orchestration Machine Learning (ESGI). On construit,
au fil des seances, un pipeline MLOps complet (entrainement, suivi d'experiences,
API, frontend, orchestration) autour d'un probleme de **classification binaire**.

## La problematique : un probleme de classification

On cherche a predire, pour un demandeur, si sa **demande de carte de credit sera
acceptee (`1`) ou refusee (`0`)** a partir de ses caracteristiques personnelles et
financieres (revenu, situation familiale, emploi, logement...).

> En une phrase : les `1` sont les demandes **acceptees**, les `0` les demandes
> **refusees** ; predire cette cible aide un etablissement a automatiser et
> fiabiliser sa decision d'octroi tout en limitant le risque.

Pourquoi c'est de la **classification** (et non de la regression) : la cible n'est
pas une quantite continue mais une **etiquette a deux valeurs**. Le modele apprend
une frontiere de decision a partir d'exemples passes, puis attribue a chaque
nouveau demandeur une **probabilite d'appartenir a la classe 1**, convertie en
decision via un seuil (0,5 par defaut).

### Un jeu de donnees desequilibre

La classe positive est minoritaire : **11,3 % de `1`** pour **88,7 % de `0`**
(175 acceptations sur 1 548). Consequence :

- l'**accuracy** est trompeuse (predire toujours `0` donne deja ~89 %) ;
- on suit en priorite le **ROC AUC** et le **F1** de la classe positive ;
- piste d'amelioration prevue : `class_weight="balanced"` / reechantillonnage.

## Le jeu de donnees

Source : [Kaggle - Credit Card Details](https://www.kaggle.com/datasets/rohitudageri/credit-card-details)

Deux fichiers bruts, joints sur `Ind_ID`, places dans `data/` :

| Fichier                       | Contenu                                            |
|-------------------------------|----------------------------------------------------|
| `data/Credit_card.csv`        | 1 548 demandeurs x 18 colonnes de caracteristiques |
| `data/Credit_card_label.csv`  | `Ind_ID` + `label` (0/1)                           |

- **Cible** : `target` (renommee depuis `label`), binaire 0/1.
- **Numeriques** (8) : `CHILDREN`, `Annual_income`, `Birthday_count`,
  `Employed_days`, `Work_Phone`, `Phone`, `EMAIL_ID`, `Family_Members`.
- **Categorielles** (8) : `GENDER`, `Car_Owner`, `Propert_Owner`, `Type_Income`,
  `EDUCATION`, `Marital_status`, `Housing_type`, `Type_Occupation`.

`data/dataset.csv` est issu de la fusion des deux CSV (jointure sur `Ind_ID`),
avec `label` renomme en `target`, suppression de `Ind_ID` et `Mobile_phone`
(constant) et imputation des manquants (mediane pour le numerique, `Unknown` pour
le categoriel).

## Structure du projet

```
.
├── pyproject.toml        dependances + outils (uv, ruff, mypy, pytest)
├── Makefile              commandes du projet (make help)
├── data/                 CSV bruts + dataset.csv prepare
├── src/                  modules baseline (lances via PYTHONPATH=src)
│   ├── config.py         configuration (dataset, cible, features)
│   ├── data.py           chargement + split train/test
│   ├── features.py       feature engineering + pre-processing (scaler + one-hot)
│   ├── train.py          entrainement de la baseline LogisticRegression + MLflow
│   ├── train_models.py   comparaison RF / XGBoost / LightGBM (GridSearchCV) + MLflow
│   ├── train_optuna.py   optimisation RF / XGBoost / LightGBM (Optuna TPE) + MLflow
│   ├── tracking.py       config MLflow partagee (experience, tags, dataset lineage)
│   ├── api.py            API FastAPI d'inference (/predict)
│   └── evaluation.py      summary plot SHAP loggue dans MLflow
├── scripts/              scripts utilitaires hors package
│   └── predict.py        prediction par lot (CSV/JSON) a partir du modele
├── docker/               Dockerfiles (train + api)
├── docker-compose.yml    stack mlflow + train + api
└── tests/                tests pytest
```

## Mise en route

L'environnement est gere par [`uv`](https://docs.astral.sh/uv/) (Python 3.13).

```bash
make install     # cree .venv + installe les dependances
make train       # entraine la baseline -> models/model.joblib
```

Sortie attendue (ordre de grandeur) :

```
f1=0.000  roc_auc=0.707
```

Le `roc_auc > 0.5` confirme que la baseline ordonne correctement les demandeurs.
Le `f1 = 0` vient du desequilibre des classes (11 % de `1`) : au seuil 0.5, la
regression logistique non ponderee predit surtout la classe majoritaire. La
gestion du desequilibre (`class_weight`, seuil, re-echantillonnage) et le suivi
MLflow seront ajoutes au fil des seances.

Les briques sont aussi utilisables directement (`PYTHONPATH=src`) :

```python
from data import load_data, split
from features import build_preprocessor

df = load_data()
x_train, x_test, y_train, y_test = split(df)
pre = build_preprocessor()
```

## Comparaison de modeles + suivi MLflow

`make train-models` optimise trois familles de modeles par `GridSearchCV` et
sauvegarde la meilleure dans `models/model.joblib` :

```bash
make train-models                  # CV=5, scoring=roc_auc
make train-models CV=3 SCORING=f1  # parametrable
```

- **Random Forest**, **XGBoost** et **LightGBM** (RF et LightGBM en
  `class_weight="balanced"` pour le desequilibre) ;
- chaque modele est trace dans un **run MLflow** imbrique sous `compare-models` :
  hyperparametres, metriques (`cv_*`, `f1`, `roc_auc`), matrice de confusion,
  rapport de classification et **summary plot SHAP** ;
- par defaut le suivi est **local** (dossier `./mlruns`). Pour pointer un serveur
  de tracking (et activer le Model Registry) :

  ```bash
  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
  mlflow ui                  # ou le serveur de la stack docker
  ```

Resultats indicatifs (CV=2) : Random Forest `roc_auc=0.85 / f1=0.60`, devant
XGBoost et LightGBM (`roc_auc=0.80`).

`make train-optuna` fait la meme comparaison avec une recherche **Optuna**
(sampler TPE) au lieu de la grille : chaque essai est trace dans MLflow.

```bash
make train-optuna                    # N_TRIALS=30, CV=5
make train-optuna N_TRIALS=50 CV=3
```

## Inference : API FastAPI et prediction par lot

Le modele sauvegarde (`models/model.joblib`) est servi par une **API FastAPI**
(`src/api.py`), qui le charge une seule fois au demarrage :

```bash
make api                       # http://127.0.0.1:8000/docs
```

- `GET /health` : etat de l'API + modele charge ;
- `POST /predict` : prediction (`prediction` 0/1 + `probability`) a partir des
  caracteristiques d'un demandeur ;
- `GET /model-info` : version servie (variable d'environnement `MODEL_VERSION`).

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"GENDER":"M","Car_Owner":"Y","Propert_Owner":"Y","CHILDREN":0,
       "Annual_income":180000.0,"Type_Income":"Pensioner","EDUCATION":"Higher education",
       "Marital_status":"Married","Housing_type":"House / apartment","Birthday_count":-18772.0,
       "Employed_days":365243,"Work_Phone":0,"Phone":0,"EMAIL_ID":0,
       "Type_Occupation":"Unknown","Family_Members":2}'
```

Pour une **prediction par lot** sans serveur, `scripts/predict.py` (hors `src/`)
applique le modele a un fichier CSV ou JSON :

```bash
make predict INPUT=data/dataset.csv                               # affiche les predictions
PYTHONPATH=src python scripts/predict.py --input demandeur.json --output predictions.csv
```

## Docker

Les images sont construites avec **uv** (modules plats dans `src/`) a partir de
`uv.lock` (build reproductible). `libgomp1` est installe pour LightGBM/XGBoost.

```bash
make docker-build              # images mlops-train + mlops-api
make docker-train              # entraine dans un conteneur -> ./models/model.joblib
make docker-api                # API conteneurisee sur http://localhost:8000/docs
```

Stack complete via `docker compose` (serveur **MLflow** + entrainement + **API**) :

```bash
docker compose up -d --build mlflow            # 1. serveur de suivi (:5000)
docker compose --profile train run --rm train  # 2. entrainement -> volume models_data
docker compose up -d --build api               # 3. API (:8000), lit le modele en lecture seule
docker compose down                            # arret de la stack
```

Le service `train` envoie son suivi vers `http://mlflow:5000` (resolution DNS
interne) et partage le modele avec l'`api` via le volume `models_data`.

## Qualite

```bash
make lint        # ruff
make type        # mypy
make test        # pytest
make check       # les trois
```
