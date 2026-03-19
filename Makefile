# ── 2D/3D AutoReg – Local Development (macOS Apple Silicon) ──────────
#
#   make install      Install all dependencies (backend + frontend)
#   make dev          Run backend and frontend concurrently
#   make backend      Run backend only
#   make frontend     Run frontend only
#   make docker       Run everything in Docker (CPU-only)
#   make clean        Remove generated artifacts
#
# Requirements: Python 3.11+, Node 20+, Test_Data/HN_P001.nii.gz
# ─────────────────────────────────────────────────────────────────────

# Paths
BACKEND_DIR   := backend
FRONTEND_DIR  := frontend
TEST_DATA_DIR := Test_Data
NIFTI_FILE    := $(TEST_DATA_DIR)/HN_P001.nii.gz
VENV_DIR      := $(BACKEND_DIR)/.venv

# Backend
PYTHON        := $(VENV_DIR)/bin/python
PIP           := $(VENV_DIR)/bin/pip
UVICORN       := $(VENV_DIR)/bin/uvicorn
BACKEND_PORT  := 8000

# Frontend
NPM           := npm
FRONTEND_PORT := 3000

# Environment
export NIFTI_PATH  := $(CURDIR)/$(NIFTI_FILE)
export VITE_API_BASE_URL := http://localhost:$(BACKEND_PORT)

# ── Phony targets ───────────────────────────────────────────────────

.PHONY: install install-backend install-frontend \
        dev backend frontend docker clean check-data help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Install ─────────────────────────────────────────────────────────

install: install-backend install-frontend ## Install all dependencies

install-backend: $(VENV_DIR)/bin/activate ## Install backend (Python venv)
$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt

install-frontend: $(FRONTEND_DIR)/node_modules ## Install frontend (npm)
$(FRONTEND_DIR)/node_modules: $(FRONTEND_DIR)/package.json
	cd $(FRONTEND_DIR) && $(NPM) install
	@touch $@

# ── Run ─────────────────────────────────────────────────────────────

check-data:
	@test -f $(NIFTI_FILE) || { echo "Error: $(NIFTI_FILE) not found. Place CT volume in Test_Data/."; exit 1; }

dev: install check-data ## Run backend + frontend concurrently
	@echo "Starting backend on :$(BACKEND_PORT) and frontend on :$(FRONTEND_PORT) …"
	@trap 'kill 0' INT TERM; \
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port $(BACKEND_PORT) \
		--app-dir $(BACKEND_DIR) & \
	cd $(FRONTEND_DIR) && $(NPM) run dev & \
	wait

backend: install-backend check-data ## Run backend only
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port $(BACKEND_PORT) \
		--app-dir $(BACKEND_DIR)

frontend: install-frontend ## Run frontend only
	cd $(FRONTEND_DIR) && $(NPM) run dev

# ── Docker ──────────────────────────────────────────────────────────

docker: ## Run via Docker Compose (CPU-only)
	docker compose up --build

# ── Clean ───────────────────────────────────────────────────────────

clean: ## Remove venv, node_modules, build artifacts
	rm -rf $(VENV_DIR)
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf $(FRONTEND_DIR)/dist
