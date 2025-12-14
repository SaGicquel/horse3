#!/bin/sh
# ============================================================================
# 🐴 generate-config.sh - Génère la configuration runtime pour le frontend
# ============================================================================
#
# Ce script génère /usr/share/nginx/html/app-config.js au démarrage du conteneur
# avec les variables d'environnement passées à Docker.
#
# Usage dans Dockerfile :
#   ENTRYPOINT ["/docker-entrypoint.d/generate-config.sh"]
#   CMD ["nginx", "-g", "daemon off;"]
#
# Variables d'environnement supportées :
#   - VITE_API_BASE_URL : URL de base de l'API (ex: https://api.example.com/api)
#   - APP_ENVIRONMENT   : Environnement (development, staging, production)
#   - APP_VERSION       : Version de l'application
#
# ============================================================================

set -e

CONFIG_FILE="/usr/share/nginx/html/app-config.js"

# Valeurs par défaut
API_BASE_URL="${VITE_API_BASE_URL:-/api}"
ENVIRONMENT="${APP_ENVIRONMENT:-production}"
VERSION="${APP_VERSION:-1.0.0}"

echo "🐴 Generating runtime config..."
echo "   API_BASE_URL: ${API_BASE_URL}"
echo "   ENVIRONMENT: ${ENVIRONMENT}"
echo "   VERSION: ${VERSION}"

# Générer le fichier de configuration
cat > "${CONFIG_FILE}" << EOF
/**
 * Configuration runtime injectée au démarrage du conteneur
 * Généré par generate-config.sh
 * Ne pas modifier manuellement - ce fichier est regénéré à chaque démarrage
 */
window.__APP_CONFIG__ = {
  apiBaseUrl: "${API_BASE_URL}",
  environment: "${ENVIRONMENT}",
  version: "${VERSION}",
  generatedAt: "$(date -Iseconds)"
};
console.log('[app-config] Runtime configuration loaded:', window.__APP_CONFIG__);
EOF

echo "✅ Config generated at ${CONFIG_FILE}"

# Exécuter la commande passée en argument (nginx)
exec "$@"
