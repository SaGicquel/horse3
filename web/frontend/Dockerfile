# ============================================================================
# 🐴 Dockerfile - Frontend React (Multi-stage build)
# ============================================================================

# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copier les fichiers de dépendances
COPY package*.json ./

# Installer les dépendances
RUN npm ci

# Copier le code source
COPY . .

# Build de production
RUN npm run build

# Stage 2: Production avec Nginx
FROM nginx:alpine

# Installer curl pour healthcheck
RUN apk add --no-cache curl

# Copier la config nginx personnalisée
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copier le script de génération de config runtime
COPY docker-entrypoint.d/generate-config.sh /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/generate-config.sh

# Copier les fichiers buildés
COPY --from=builder /app/dist /usr/share/nginx/html

# Exposer le port
EXPOSE 80

# Variables d'environnement pour la configuration runtime
ENV VITE_API_BASE_URL=/api
ENV APP_ENVIRONMENT=production
ENV APP_VERSION=1.0.0

# Healthcheck sur /healthz (vérifie Nginx + connectivité)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:80/healthz || exit 1

# Entrypoint pour générer la config puis lancer nginx
ENTRYPOINT ["/docker-entrypoint.d/generate-config.sh"]
CMD ["nginx", "-g", "daemon off;"]
