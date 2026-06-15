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

`python -m churn.prepare_data` fusionne les deux CSV, renomme `label` -> `target`,
supprime `Ind_ID` et `Mobile_phone` (constant), impute les manquants (mediane pour
le numerique, `Unknown` pour le categoriel) et ecrit `data/dataset.csv`.

## Structure du projet

```
.
├── pyproject.toml        dependances + outils (uv, ruff, mypy, pytest)
├── Makefile              commandes du projet (make help)
├── data/                 CSV bruts + dataset.csv prepare
├── churn/                package du projet
│   ├── config.py         configuration (dataset, cible, features)
│   ├── prepare_data.py   preparation des donnees -> data/dataset.csv
│   ├── data.py           chargement + split train/test
│   ├── features.py       pre-processing (StandardScaler + OneHotEncoder)
│   └── train.py          entrainement de la baseline LogisticRegression
└── tests/                tests pytest
```

## Mise en route

L'environnement est gere par [`uv`](https://docs.astral.sh/uv/) (Python 3.13).

```bash
make install     # cree .venv + installe le projet et les dependances
make data        # prepare data/dataset.csv (fusion + nettoyage des CSV Kaggle)
make train       # entraine / evalue la baseline -> models/model.joblib
```

Sortie attendue (ordre de grandeur) :

```
f1=0.056  roc_auc=0.649
```

Le `roc_auc > 0.5` confirme que la baseline fait mieux que le hasard ; le `f1`
faible reflete le desequilibre des classes. C'est le point de depart que les
seances suivantes (MLflow, Optuna, comparaison de modeles, API, orchestration...)
viendront enrichir.

## Qualite

```bash
make lint        # ruff
make type        # mypy
make test        # pytest
make check       # les trois
```
