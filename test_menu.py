#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du menu CLI - Simule l'affichage du menu principal
"""

import sys

sys.path.insert(0, ".")

from cli import print_main_menu, print_scraping_menu, print_audit_menu, print_cleaning_menu

print("=== TEST DES MENUS ===\n")

print("\n1. MENU PRINCIPAL:")
print_main_menu()

input("\nAppuyez sur Entrée pour voir le menu Scraping...")

print("\n2. MENU SCRAPING:")
print_scraping_menu()

input("\nAppuyez sur Entrée pour voir le menu Audit...")

print("\n3. MENU AUDIT:")
print_audit_menu()

input("\nAppuyez sur Entrée pour voir le menu Nettoyage...")

print("\n4. MENU NETTOYAGE:")
print_cleaning_menu()

print("\n✅ Test des menus terminé!")
