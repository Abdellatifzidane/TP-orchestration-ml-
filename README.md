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
│   └── features.py       feature engineering + pre-processing (scaler + one-hot)
└── tests/                tests pytest
```

## Mise en route

L'environnement est gere par [`uv`](https://docs.astral.sh/uv/) (Python 3.13).

```bash
make install     # cree .venv + installe les dependances
make test        # verifie config + features (pytest)
```

Exemple d'utilisation des briques baseline (`PYTHONPATH=src`) :

```python
from data import load_data, split
from features import build_preprocessor

df = load_data()
x_train, x_test, y_train, y_test = split(df)
pre = build_preprocessor()        # a brancher sur un estimateur (TP suivants)
```

> Les briques `config`/`data`/`features` constituent la baseline. L'entrainement
> du modele, le suivi MLflow, l'API et l'orchestration seront ajoutes au fil des
> seances.

## Qualite

```bash
make lint        # ruff
make type        # mypy
make test        # pytest
make check       # les trois
```
