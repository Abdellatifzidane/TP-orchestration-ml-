# ==============================================================================
# Projet fil rouge MLOps - classification (acceptation de carte de credit)
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := PYTHONPATH=src uv run python
RUN          := uv run
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
INPUT        ?= data/dataset.csv
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help check-uv install sync lock train train-models train-optuna api predict lint format type test check

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-14s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Environnement
# ------------------------------------------------------------------------------

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

install: check-uv ## Cree le venv et installe le projet + dependances dev
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

sync: install ## Alias de install

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	uv lock

# ------------------------------------------------------------------------------
# Entrainement
# ------------------------------------------------------------------------------

train: ## Entraine la baseline -> models/model.joblib (C=.. MAX_ITER=..)
	$(PYTHON) src/train.py --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RF / XGBoost / LightGBM (GridSearchCV) + suivi MLflow (CV=.. SCORING=..)
	$(PYTHON) src/train_models.py --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna (TPE) + suivi MLflow (N_TRIALS=.. CV=..)
	$(PYTHON) src/train_optuna.py --n-trials $(N_TRIALS) --cv $(CV)

# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

api: ## Lance l'API FastAPI (docs sur /docs) (API_HOST=.. API_PORT=..)
	PYTHONPATH=src $(RUN) uvicorn api:app --reload --host $(API_HOST) --port $(API_PORT)

predict: ## Prediction par lot sur un fichier (INPUT=data/dataset.csv)
	$(PYTHON) predict.py --input $(INPUT)

# ------------------------------------------------------------------------------
# Qualite
# ------------------------------------------------------------------------------

lint: ## Verifie le style (ruff)
	$(RUN) ruff check src tests predict.py

format: ## Formate le code (ruff)
	$(RUN) ruff format src tests predict.py

type: ## Verifie les types (mypy)
	$(RUN) mypy src predict.py

test: ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)
