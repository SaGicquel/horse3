# ============================================================================
# 🏇 Dockerfile Production - API Prédiction Courses Hippiques
# ============================================================================
# Multi-stage build pour optimiser la taille de l'image
# Image finale: ~500MB (vs ~2GB sans optimisation)
# ============================================================================

# ============================================================================
# STAGE 1: Builder - Installation dépendances
# ============================================================================
FROM python:3.11-slim as builder

LABEL maintainer="Phase 7 - Production & Monitoring"
LABEL description="API FastAPI pour prédictions courses hippiques (Stacking Ensemble)"

# Variables d'environnement pour optimisation Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer dépendances système pour compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Créer virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier requirements
COPY requirements-prod.txt .

# Installer dépendances Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-prod.txt


# ============================================================================
# STAGE 2: Runtime - Image finale légère
# ============================================================================
FROM python:3.11-slim

# Métadonnées
LABEL version="1.0.0"
LABEL model="Stacking Ensemble (RF + XGBoost + LightGBM)"
LABEL performance="ROC-AUC Test: 0.7009"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH="/app/models/ensemble_stacking.pkl"

# Installer uniquement les libs runtime nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root pour sécurité
RUN useradd -m -u 1000 -s /bin/bash apiuser && \
    mkdir -p /app/models /app/logs && \
    chown -R apiuser:apiuser /app

# Copier virtualenv depuis builder
COPY --from=builder /opt/venv /opt/venv

# Définir workdir
WORKDIR /app

# Copier code source
COPY --chown=apiuser:apiuser api_prediction.py .

# Copier modèle (sera monté en volume en prod)
COPY --chown=apiuser:apiuser data/models/ensemble_stacking.pkl models/

# Exposer port
EXPOSE 8000

# Passer à utilisateur non-root
USER apiuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Commande de démarrage
CMD ["uvicorn", "api_prediction:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]
