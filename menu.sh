#!/bin/bash
# Script de lancement rapide du menu CLI Horse3

cd "$(dirname "$0")"

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Lancer le menu
python3 cli.py menu
