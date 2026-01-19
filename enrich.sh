#!/bin/bash
# Script pour enrichir les données PMU avec tous les scrapers disponibles

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 ENRICHISSEMENT PMU - PIPELINE COMPLET${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Date par défaut = aujourd'hui
DATE=${1:-$(date +%Y-%m-%d)}

echo -e "${GREEN}📅 Date d'enrichissement: ${DATE}${NC}"
echo ""

# Lancer l'orchestrateur
echo -e "${YELLOW}▶️  Lancement des scrapers...${NC}"
python orchestrator_scrapers.py --date ${DATE}

# Vérifier le résultat
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Enrichissement terminé avec succès !${NC}"
    echo ""

    # Afficher les statistiques
    echo -e "${BLUE}📊 Statistiques de la base:${NC}"
    docker exec -i pmuBDD psql -U postgres -d pmubdd -c "
        SELECT
            COUNT(*) as total_participations,
            COUNT(course_id) as avec_metadata,
            COUNT(handicap_valeur) as avec_handicap,
            COUNT(entraineur_winrate_90j) as avec_connections
        FROM cheval_courses_seen;
    "

    echo ""
    docker exec -i pmuBDD psql -U postgres -d pmubdd -c "
        SELECT
            COUNT(*) as total_chevaux,
            COUNT(nb_places_12m) as avec_stats_perf,
            COUNT(score_forme_recent) as avec_forme
        FROM chevaux
        WHERE nombre_courses_total > 0;
    "
else
    echo ""
    echo -e "${YELLOW}⚠️  Enrichissement terminé avec des avertissements${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
