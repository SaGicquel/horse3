#!/usr/bin/env python3
"""
Script de validation finale de l'installation Horse3 Enrichissement.

Vérifie :
- Présence de tous les fichiers
- Structure du package enrichment/
- Fonctionnalité de base
- Tests unitaires
- Documentation

Usage : python validate_installation.py
"""

import os
import sys
from pathlib import Path


def print_section(title: str):
    """Affiche un titre de section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def check_file(filepath: str, description: str) -> bool:
    """Vérifie qu'un fichier existe."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description:40} {'OK' if exists else 'MANQUANT'}")
    return exists


def check_import(module_name: str, description: str) -> bool:
    """Vérifie qu'un module peut être importé."""
    try:
        __import__(module_name)
        print(f"✅ {description:40} OK")
        return True
    except ImportError as e:
        print(f"❌ {description:40} ERREUR: {e}")
        return False


def main():
    """Validation principale."""

    print("🔍 Validation de l'installation Horse3 Enrichissement")

    checks_passed = 0
    checks_total = 0

    # ========================================
    # 1. Vérification des fichiers principaux
    # ========================================

    print_section("1. Fichiers principaux")

    files_to_check = [
        ("enrichment/__init__.py", "Package enrichment"),
        ("enrichment/normalization.py", "Module normalisation"),
        ("enrichment/calculations.py", "Module calculs"),
        ("enrichment/matching.py", "Module matching"),
        ("enrichment/migrations.py", "Module migrations"),
        ("cli.py", "Interface CLI"),
        ("test_enrichment.py", "Tests unitaires"),
        ("demo_enrichment.py", "Script de démonstration"),
    ]

    for filepath, description in files_to_check:
        checks_total += 1
        if check_file(filepath, description):
            checks_passed += 1

    # ========================================
    # 2. Vérification de la documentation
    # ========================================

    print_section("2. Documentation")

    docs_to_check = [
        ("README_ENRICHISSEMENT.md", "README principal"),
        ("LIVRAISON_ENRICHISSEMENT.md", "Document de livraison"),
        ("STATISTIQUES_ENRICHISSEMENT.md", "Statistiques projet"),
        ("QUICK_START_ENRICHISSEMENT.md", "Guide express"),
        ("INDEX_ENRICHISSEMENT.md", "Index documentation"),
    ]

    for filepath, description in docs_to_check:
        checks_total += 1
        if check_file(filepath, description):
            checks_passed += 1

    # ========================================
    # 3. Vérification des imports Python
    # ========================================

    print_section("3. Imports Python")

    imports_to_check = [
        ("enrichment", "Package enrichment"),
        ("enrichment.normalization", "Module normalisation"),
        ("enrichment.calculations", "Module calculs"),
        ("enrichment.matching", "Module matching"),
        ("enrichment.migrations", "Module migrations"),
    ]

    for module_name, description in imports_to_check:
        checks_total += 1
        if check_import(module_name, description):
            checks_passed += 1

    # ========================================
    # 4. Test de fonctionnalité de base
    # ========================================

    print_section("4. Tests de fonctionnalité")

    try:
        from enrichment import normalize_name, parse_time_str, compute_reduction_km

        # Test normalisation
        checks_total += 1
        test_name = normalize_name("Élégant D'Avril (FR)")
        if test_name == "ELEGANT DAVRIL":
            print(f"✅ {'normalize_name()':40} OK")
            checks_passed += 1
        else:
            print(f"❌ {'normalize_name()':40} ERREUR: {test_name}")

        # Test parsing temps
        checks_total += 1
        test_time = parse_time_str("1'12\"8")
        if test_time == 72.8:
            print(f"✅ {'parse_time_str()':40} OK")
            checks_passed += 1
        else:
            print(f"❌ {'parse_time_str()':40} ERREUR: {test_time}")

        # Test réduction kilométrique
        checks_total += 1
        test_reduction = compute_reduction_km(72.8, 2400)
        if abs(test_reduction - 30.33) < 0.01:
            print(f"✅ {'compute_reduction_km()':40} OK")
            checks_passed += 1
        else:
            print(f"❌ {'compute_reduction_km()':40} ERREUR: {test_reduction}")

    except Exception as e:
        print(f"❌ Tests de fonctionnalité              ERREUR: {e}")

    # ========================================
    # 5. Vérification des tests unitaires
    # ========================================

    print_section("5. Tests unitaires")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "test_enrichment.py"], capture_output=True, text=True, timeout=30
        )

        checks_total += 1
        if result.returncode == 0 and "OK" in result.stderr:
            print(f"✅ {'Suite de tests complète':40} OK")
            checks_passed += 1

            # Extraire le nombre de tests
            if "Ran" in result.stderr:
                line = [l for l in result.stderr.split("\n") if "Ran" in l][0]
                print(f"   {line.strip()}")
        else:
            print(f"❌ {'Suite de tests complète':40} ERREUR")
            print(f"   Return code: {result.returncode}")
            if result.stderr:
                print(f"   Stderr: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        print(f"❌ {'Suite de tests complète':40} TIMEOUT")
    except Exception as e:
        print(f"❌ {'Suite de tests complète':40} ERREUR: {e}")

    # ========================================
    # 6. Résumé final
    # ========================================

    print_section("Résumé de validation")

    percentage = (checks_passed / checks_total * 100) if checks_total > 0 else 0

    print(f"\n✅ Tests réussis : {checks_passed}/{checks_total} ({percentage:.1f}%)")

    if checks_passed == checks_total:
        print("\n" + "=" * 60)
        print("  🎉 INSTALLATION VALIDÉE - Système prêt à l'emploi !")
        print("=" * 60)
        print("\n📖 Prochaines étapes :")
        print("   1. Lire QUICK_START_ENRICHISSEMENT.md")
        print("   2. Exécuter : python demo_enrichment.py")
        print("   3. Initialiser : python cli.py init-db")
        print("   4. Importer CSV : python cli.py import-ifce --path ./fichier-des-equides.csv")
        print("   5. Scraper : python cli.py fetch")
        print("\n📚 Documentation complète : README_ENRICHISSEMENT.md")
        return 0

    elif percentage >= 80:
        print("\n" + "=" * 60)
        print("  ⚠️  INSTALLATION PARTIELLE - Quelques problèmes détectés")
        print("=" * 60)
        print("\n🔧 Vérifiez les éléments marqués ❌ ci-dessus")
        print("📚 Consultez QUICK_START_ENRICHISSEMENT.md - Section 'Problèmes courants'")
        return 1

    else:
        print("\n" + "=" * 60)
        print("  ❌ INSTALLATION INCOMPLÈTE - Action requise")
        print("=" * 60)
        print("\n🆘 Actions recommandées :")
        print("   1. Vérifier que tous les fichiers sont présents")
        print("   2. Vérifier Python 3.8+ : python --version")
        print("   3. Consulter LIVRAISON_ENRICHISSEMENT.md")
        return 2


if __name__ == "__main__":
    sys.exit(main())
