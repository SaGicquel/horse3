#!/bin/bash
# Script automatique : Scraping PMU + Migration + Recalcul
# Usage: ./scrape_and_sync.sh [YYYY-MM-DD]

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Date par défaut : aujourd'hui
DATE=${1:-$(date +%Y-%m-%d)}

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🏇 Horse3 - Scraping & Synchronisation Automatique${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}📅 Date cible : $DATE${NC}\n"

# Étape 1 : Scraping PMU
echo -e "${YELLOW}[1/3]${NC} 🌐 Scraping PMU..."
if python cli.py fetch --date "$DATE"; then
    echo -e "${GREEN}   ✅ Scraping terminé${NC}\n"
else
    echo -e "${RED}   ❌ Erreur lors du scraping${NC}"
    exit 1
fi

# Étape 2 : Migration des données
echo -e "${YELLOW}[2/3]${NC} 🔄 Migration des données..."
if python cli.py migrate; then
    echo -e "${GREEN}   ✅ Migration terminée${NC}\n"
else
    echo -e "${RED}   ❌ Erreur lors de la migration${NC}"
    exit 1
fi

# Étape 3 : Recalcul des statistiques
echo -e "${YELLOW}[3/3]${NC} 📊 Recalcul des statistiques..."
if python cli.py recompute; then
    echo -e "${GREEN}   ✅ Recalcul terminé${NC}\n"
else
    echo -e "${RED}   ❌ Erreur lors du recalcul${NC}"
    exit 1
fi

# Résumé
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Synchronisation terminée avec succès !${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Afficher quelques statistiques
echo -e "${BLUE}📊 Statistiques :${NC}"
sqlite3 data/database.db "SELECT '   • ' || COUNT(*) || ' chevaux PMU' FROM pmu_horses;"
sqlite3 data/database.db "SELECT '   • ' || COUNT(*) || ' performances' FROM performances;"
sqlite3 data/database.db "SELECT '   • ' || COUNT(*) || ' statistiques annuelles' FROM horse_year_stats;"

echo ""
echo -e "${BLUE}💡 Prochaines étapes :${NC}"
echo "   - Voir les performances : sqlite3 data/database.db 'SELECT * FROM performances ORDER BY race_date DESC LIMIT 10;'"
echo "   - Rapport matching : python cli.py match-report"
echo ""
