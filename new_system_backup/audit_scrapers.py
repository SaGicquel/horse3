#!/usr/bin/env python3
"""
AUDIT COMPLET DES SCRAPERS
Analyse performance, qualité code, optimisations possibles
"""

import os
import re
import ast
from pathlib import Path
from collections import defaultdict

print("="*80)
print("🔍 AUDIT COMPLET DES SCRAPERS")
print("="*80)

# Lister tous les scrapers
scrapers_dir = Path("scrapers")
scrapers = [f for f in scrapers_dir.glob("*.py") if not f.name.startswith("__")]

print(f"\n📁 {len(scrapers)} scrapers détectés\n")

audit_results = {}

for scraper_path in sorted(scrapers):
    scraper_name = scraper_path.name
    
    with open(scraper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Métriques de base
    lines = content.split('\n')
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    blank_lines = len([l for l in lines if not l.strip()])
    
    # Analyse patterns
    issues = []
    optimizations = []
    
    # 1. Requêtes SQL inefficaces
    sql_patterns = [
        (r'SELECT \*', "SELECT * détecté (charger que colonnes nécessaires)", "performance"),
        (r'\.execute\([^)]*\n[^)]*\n', "Requête SQL multi-lignes (illisible)", "lisibilité"),
        (r'for .* in .*:\s+.*\.execute', "Boucle avec execute() (utiliser executemany)", "performance"),
        (r'commit\(\).*for', "Commit dans boucle (grouper commits)", "performance"),
    ]
    
    for pattern, desc, category in sql_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append({
                'type': category,
                'description': desc,
                'count': len(matches)
            })
    
    # 2. Gestion erreurs
    try_blocks = len(re.findall(r'\btry:', content))
    except_blocks = len(re.findall(r'\bexcept:', content))
    bare_except = len(re.findall(r'except:\s*$', content, re.MULTILINE))
    
    if bare_except > 0:
        issues.append({
            'type': 'erreurs',
            'description': f"Except bare détecté ({bare_except}) - masque erreurs",
            'count': bare_except
        })
    
    # 3. Imports
    import_lines = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]
    unused_imports = []
    
    # 4. Fonctions longues
    func_pattern = r'def (\w+)\([^)]*\):'
    functions = re.findall(func_pattern, content)
    
    # 5. Complexité
    if_count = len(re.findall(r'\bif\b', content))
    for_count = len(re.findall(r'\bfor\b', content))
    while_count = len(re.findall(r'\bwhile\b', content))
    complexity_score = if_count + for_count * 2 + while_count * 2
    
    # 6. Documentation
    docstrings = len(re.findall(r'""".*?"""', content, re.DOTALL))
    has_main_doc = '"""' in content[:500]
    
    # 7. Connexions DB
    db_connects = len(re.findall(r'psycopg2\.connect|get_connection', content))
    db_closes = len(re.findall(r'\.close\(\)', content))
    
    if db_connects > db_closes:
        issues.append({
            'type': 'ressources',
            'description': f"Connexions DB non fermées ({db_connects} connect, {db_closes} close)",
            'count': db_connects - db_closes
        })
    
    # 8. Performance patterns
    if 'sleep(' in content:
        sleep_count = len(re.findall(r'sleep\(', content))
        issues.append({
            'type': 'performance',
            'description': f"Utilise sleep() ({sleep_count}) - ralentit exécution",
            'count': sleep_count
        })
    
    # Calcul score qualité
    quality_score = 100
    quality_score -= len(issues) * 5
    quality_score -= (bare_except * 10)
    quality_score -= (complexity_score // 50) * 5
    quality_score += (docstrings * 2)
    quality_score = max(0, min(100, quality_score))
    
    # Suggestions optimisation
    if code_lines > 500:
        optimizations.append("Fichier volumineux (>500 lignes) - envisager découpage")
    
    if complexity_score > 100:
        optimizations.append("Complexité élevée - simplifier logique")
    
    if not has_main_doc:
        optimizations.append("Pas de docstring principale - ajouter documentation")
    
    if db_connects > 1:
        optimizations.append("Multiple connexions DB - réutiliser connexion")
    
    # Stocker résultats
    audit_results[scraper_name] = {
        'lines': {
            'total': total_lines,
            'code': code_lines,
            'comments': comment_lines,
            'blank': blank_lines
        },
        'complexity': complexity_score,
        'functions': len(functions),
        'issues': issues,
        'optimizations': optimizations,
        'quality_score': quality_score,
        'docstrings': docstrings,
        'try_blocks': try_blocks,
        'db_connects': db_connects
    }

# Affichage résultats
print("\n" + "="*80)
print("📊 RÉSULTATS PAR SCRAPER")
print("="*80)

# Trier par score qualité
sorted_scrapers = sorted(audit_results.items(), key=lambda x: x[1]['quality_score'])

for scraper_name, data in sorted_scrapers:
    score = data['quality_score']
    
    # Emoji selon score
    if score >= 80:
        emoji = "🟢"
    elif score >= 60:
        emoji = "🟡"
    else:
        emoji = "🔴"
    
    print(f"\n{emoji} {scraper_name} (Score: {score}/100)")
    print(f"   Lignes: {data['lines']['code']} code, {data['lines']['comments']} comments")
    print(f"   Complexité: {data['complexity']}, Fonctions: {data['functions']}")
    
    if data['issues']:
        print(f"   ⚠️  Issues ({len(data['issues'])}):")
        for issue in data['issues'][:3]:
            print(f"      - {issue['description']}")
    
    if data['optimizations']:
        print(f"   💡 Optimisations ({len(data['optimizations'])}):")
        for opt in data['optimizations'][:2]:
            print(f"      - {opt}")

# Statistiques globales
print("\n" + "="*80)
print("📈 STATISTIQUES GLOBALES")
print("="*80)

total_lines = sum(d['lines']['total'] for d in audit_results.values())
total_code = sum(d['lines']['code'] for d in audit_results.values())
total_issues = sum(len(d['issues']) for d in audit_results.values())
avg_score = sum(d['quality_score'] for d in audit_results.values()) / len(audit_results)

print(f"\n📁 Total: {len(audit_results)} scrapers")
print(f"📝 Lignes code: {total_code:,} lignes")
print(f"⚠️  Issues: {total_issues} problèmes détectés")
print(f"📊 Score moyen: {avg_score:.1f}/100")

# Top 5 problèmes
print("\n🔥 TOP PROBLÈMES:")
all_issues = []
for scraper, data in audit_results.items():
    for issue in data['issues']:
        all_issues.append((scraper, issue))

# Grouper par type
issues_by_type = defaultdict(list)
for scraper, issue in all_issues:
    issues_by_type[issue['type']].append((scraper, issue))

for issue_type, items in sorted(issues_by_type.items(), key=lambda x: -len(x[1]))[:5]:
    print(f"\n{issue_type.upper()}: {len(items)} occurrences")
    for scraper, issue in items[:3]:
        print(f"  - {scraper}: {issue['description']}")

# Top 5 à optimiser
print("\n🎯 TOP 5 SCRAPERS À OPTIMISER:")
for i, (scraper, data) in enumerate(sorted_scrapers[:5], 1):
    print(f"\n{i}. {scraper} (Score: {data['quality_score']}/100)")
    print(f"   Issues: {len(data['issues'])}")
    print(f"   Optimisations possibles: {len(data['optimizations'])}")

print("\n" + "="*80)
print("✅ AUDIT TERMINÉ")
print("="*80)

# Sauvegarder rapport détaillé
with open('audit_scrapers_detaille.txt', 'w') as f:
    f.write("AUDIT DÉTAILLÉ DES SCRAPERS\n")
    f.write("="*80 + "\n\n")
    
    for scraper_name, data in sorted(audit_results.items()):
        f.write(f"\n{scraper_name}\n")
        f.write("-"*80 + "\n")
        f.write(f"Score qualité: {data['quality_score']}/100\n")
        f.write(f"Lignes code: {data['lines']['code']}\n")
        f.write(f"Complexité: {data['complexity']}\n")
        f.write(f"Fonctions: {data['functions']}\n")
        f.write(f"Docstrings: {data['docstrings']}\n")
        f.write(f"Try blocks: {data['try_blocks']}\n")
        f.write(f"DB connections: {data['db_connects']}\n")
        
        if data['issues']:
            f.write(f"\nIssues ({len(data['issues'])}):\n")
            for issue in data['issues']:
                f.write(f"  - [{issue['type']}] {issue['description']} (x{issue['count']})\n")
        
        if data['optimizations']:
            f.write(f"\nOptimisations ({len(data['optimizations'])}):\n")
            for opt in data['optimizations']:
                f.write(f"  - {opt}\n")
        
        f.write("\n")

print("\n💾 Rapport détaillé sauvegardé: audit_scrapers_detaille.txt")
