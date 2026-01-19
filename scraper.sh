#!/bin/bash
# Script de lancement rapide pour les scrapers PMU

clear
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                         🏇 SCRAPER PMU - MENU RAPIDE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Que voulez-vous faire ?"
echo ""
echo "  1) Scraper AUJOURD'HUI"
echo "  2) Scraper HIER"
echo "  3) Scraper les 7 DERNIERS JOURS"
echo "  4) Scraper les 30 DERNIERS JOURS"
echo "  5) Mode INTERACTIF (menu complet)"
echo "  6) Mode AVANCÉ (ligne de commande)"
echo "  0) Quitter"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

read -p "👉 Votre choix (0-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Lancement du scraping d'AUJOURD'HUI..."
        python scraper_today.py
        ;;
    2)
        echo ""
        echo "🚀 Lancement du scraping d'HIER..."
        yesterday=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d)
        python scraper_dates.py "$yesterday"
        ;;
    3)
        echo ""
        echo "🚀 Lancement du scraping des 7 DERNIERS JOURS..."
        python scraper_dates.py --last-week
        ;;
    4)
        echo ""
        echo "🚀 Lancement du scraping des 30 DERNIERS JOURS..."
        echo "⚠️  Cela peut prendre plusieurs minutes..."
        read -p "Continuer ? (o/n): " confirm
        if [[ $confirm == "o" ]] || [[ $confirm == "O" ]]; then
            python scraper_dates.py --last-month
        else
            echo "❌ Opération annulée"
        fi
        ;;
    5)
        echo ""
        echo "🚀 Lancement du mode INTERACTIF..."
        python scraper_interactif.py
        ;;
    6)
        echo ""
        echo "📖 MODE AVANCÉ - Exemples de commandes:"
        echo ""
        echo "  python scraper_dates.py 2024-10-30"
        echo "  python scraper_dates.py 2024-10-15 2024-10-20"
        echo "  python scraper_dates.py 2024-10-15,2024-10-20,2024-10-25"
        echo ""
        read -p "Entrez votre commande (sans 'python '): " cmd
        if [ ! -z "$cmd" ]; then
            python $cmd
        fi
        ;;
    0)
        echo ""
        echo "👋 Au revoir !"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✨ Terminé !"
echo "════════════════════════════════════════════════════════════════════════════════"
