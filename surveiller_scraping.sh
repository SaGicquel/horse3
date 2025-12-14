#!/bin/bash
# Script de surveillance du scraping d'octobre

echo "🔍 Surveillance du scraping d'octobre 2024..."
echo "================================================"
echo ""

# Boucle de surveillance
while true; do
    # Vérifier si le processus tourne encore
    if pgrep -f "orchestrator_scrapers.py --start 2024-10-01 --end 2024-10-31" > /dev/null; then
        # Le processus tourne
        clear
        echo "🔄 SCRAPING EN COURS..."
        echo "================================================"
        date "+%Y-%m-%d %H:%M:%S"
        echo ""
        
        # Afficher l'état
        python3 watch_progress.py
        
        echo ""
        echo "⏳ Prochaine mise à jour dans 60 secondes..."
        echo "   (Ctrl+C pour arrêter la surveillance)"
        
        sleep 60
    else
        # Le processus est terminé
        clear
        echo "✅ SCRAPING TERMINÉ !"
        echo "================================================"
        date "+%Y-%m-%d %H:%M:%S"
        echo ""
        
        # État final
        python3 watch_progress.py
        
        echo ""
        echo "📊 Lancement de l'analyse finale..."
        echo ""
        
        # Analyse finale
        python3 analyse_finale_enrichissement.py
        
        echo ""
        echo "🎉 TERMINÉ ! Analyse complète disponible dans:"
        echo "   • ANALYSE_ENRICHISSEMENT_OCTOBRE_2024.txt"
        echo ""
        
        break
    fi
done
