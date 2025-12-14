#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Visualisation et Analyses Graphiques - Base de données PMU
Génère des graphiques avancés pour analyser les performances, statistiques et tendances
"""

import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration des styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

DB_PATH = "data/database.db"


def create_connection():
    """Crée une connexion à la base de données"""
    return sqlite3.connect(DB_PATH)


def print_section(title):
    """Affiche un séparateur de section"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


# ============================================================================
# GRAPHIQUE 1 : Évolution du nombre de courses par jour
# ============================================================================

def graph_evolution_courses():
    """Graphique d'évolution du nombre de courses par jour (30 derniers jours)"""
    print_section("📊 GRAPHIQUE 1 : Évolution du nombre de courses")
    
    conn = create_connection()
    
    query = """
    SELECT 
        substr(race_key, 1, 10) as date_course,
        COUNT(DISTINCT race_key) as nb_courses,
        COUNT(*) as nb_partants
    FROM cheval_courses_seen
    WHERE date(substr(race_key, 1, 10)) >= date('now', '-30 days')
    GROUP BY date_course
    ORDER BY date_course
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Graphique 1: Nombre de courses
    ax1.plot(df['date_course'], df['nb_courses'], marker='o', linewidth=2, color='#2E86AB', markersize=6)
    ax1.fill_between(df['date_course'], df['nb_courses'], alpha=0.3, color='#2E86AB')
    ax1.set_title('Évolution du nombre de courses par jour (30 derniers jours)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Nombre de courses', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Graphique 2: Nombre de partants
    ax2.bar(df['date_course'], df['nb_partants'], color='#A23B72', alpha=0.7)
    ax2.set_title('Nombre de partants par jour', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Nombre de partants', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('graphiques/evolution_courses.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/evolution_courses.png")
    plt.close()
    
    # Statistiques
    print(f"\n📈 Statistiques:")
    print(f"   • Moyenne courses/jour: {df['nb_courses'].mean():.1f}")
    print(f"   • Maximum: {df['nb_courses'].max()} courses ({df.loc[df['nb_courses'].idxmax(), 'date_course']})")
    print(f"   • Minimum: {df['nb_courses'].min()} courses ({df.loc[df['nb_courses'].idxmin(), 'date_course']})")
    print(f"   • Moyenne partants/jour: {df['nb_partants'].mean():.0f}")


# ============================================================================
# GRAPHIQUE 2 : Distribution des places
# ============================================================================

def graph_distribution_places():
    """Distribution des places obtenues par les chevaux"""
    print_section("📊 GRAPHIQUE 2 : Distribution des places")
    
    conn = create_connection()
    
    query = """
    SELECT place, COUNT(*) as nb_occurrences
    FROM chevaux
    WHERE place IS NOT NULL 
      AND place NOT IN ('D', 'A', 'T', 'NP', 'Ret')
      AND CAST(place AS INTEGER) BETWEEN 1 AND 15
    GROUP BY place
    ORDER BY CAST(place AS INTEGER)
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    # Convertir en numérique
    df['place'] = pd.to_numeric(df['place'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    bars = ax.bar(df['place'], df['nb_occurrences'], color=colors, edgecolor='black', linewidth=1.2)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title('Distribution des places obtenues', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Place', fontsize=12)
    ax.set_ylabel('Nombre d\'occurrences', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('graphiques/distribution_places.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/distribution_places.png")
    plt.close()
    
    # Statistiques
    total = df['nb_occurrences'].sum()
    top3 = df[df['place'] <= 3]['nb_occurrences'].sum()
    print(f"\n📈 Statistiques:")
    print(f"   • Total de courses: {total:,}")
    print(f"   • Top 3: {top3:,} ({top3/total*100:.1f}%)")
    print(f"   • Place la plus fréquente: {df.loc[df['nb_occurrences'].idxmax(), 'place']:.0f}")


# ============================================================================
# GRAPHIQUE 3 : Top 15 hippodromes
# ============================================================================

def graph_top_hippodromes():
    """Top 15 des hippodromes les plus actifs"""
    print_section("📊 GRAPHIQUE 3 : Top 15 hippodromes")
    
    conn = create_connection()
    
    query = """
    SELECT hippodrome, COUNT(*) as nb_courses
    FROM chevaux
    WHERE hippodrome IS NOT NULL
    GROUP BY hippodrome
    ORDER BY nb_courses DESC
    LIMIT 15
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df['hippodrome'], df['nb_courses'], color=colors, edgecolor='black', linewidth=1)
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width):,}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_title('Top 15 des hippodromes les plus actifs', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Nombre de courses', fontsize=12)
    ax.set_ylabel('Hippodrome', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('graphiques/top_hippodromes.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/top_hippodromes.png")
    plt.close()
    
    # Statistiques
    total = df['nb_courses'].sum()
    print(f"\n📈 Statistiques:")
    print(f"   • Total courses (top 15): {total:,}")
    print(f"   • Hippodrome #1: {df.iloc[0]['hippodrome']} ({df.iloc[0]['nb_courses']:,} courses)")


# ============================================================================
# GRAPHIQUE 4 : Analyse des gains
# ============================================================================

def graph_analyse_gains():
    """Analyse de la distribution des gains"""
    print_section("📊 GRAPHIQUE 4 : Analyse des gains")
    
    conn = create_connection()
    
    query = """
    SELECT 
        CASE 
            WHEN gains_carriere < 10000 THEN '< 10K'
            WHEN gains_carriere < 50000 THEN '10K-50K'
            WHEN gains_carriere < 100000 THEN '50K-100K'
            WHEN gains_carriere < 500000 THEN '100K-500K'
            WHEN gains_carriere < 1000000 THEN '500K-1M'
            ELSE '> 1M'
        END as tranche,
        COUNT(*) as nb_chevaux
    FROM chevaux
    WHERE gains_carriere IS NOT NULL AND gains_carriere > 0
    GROUP BY tranche
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Ordre des tranches
    ordre_tranches = ['< 10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '> 1M']
    df['tranche'] = pd.Categorical(df['tranche'], categories=ordre_tranches, ordered=True)
    df = df.sort_values('tranche')
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        conn.close()
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graphique 1: Diagramme circulaire
    colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(df)))
    wedges, texts, autotexts = ax1.pie(df['nb_chevaux'], labels=df['tranche'], autopct='%1.1f%%',
                                         colors=colors, startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Répartition des chevaux par tranche de gains', fontsize=14, fontweight='bold')
    
    # Graphique 2: Barres
    bars = ax2.bar(df['tranche'], df['nb_chevaux'], color=colors, edgecolor='black', linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_title('Nombre de chevaux par tranche', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tranche de gains (€)', fontsize=12)
    ax2.set_ylabel('Nombre de chevaux', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('graphiques/analyse_gains.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/analyse_gains.png")
    plt.close()
    
    # Statistiques détaillées
    query_stats = """
    SELECT 
        COUNT(*) as total_chevaux,
        AVG(gains_carriere) as gains_moyen,
        MAX(gains_carriere) as gains_max,
        MIN(gains_carriere) as gains_min
    FROM chevaux
    WHERE gains_carriere IS NOT NULL AND gains_carriere > 0
    """
    stats = pd.read_sql_query(query_stats, conn)
    conn.close()
    
    print(f"\n📈 Statistiques:")
    print(f"   • Total chevaux avec gains: {stats['total_chevaux'].iloc[0]:,}")
    print(f"   • Gains moyen: {stats['gains_moyen'].iloc[0]:,.0f} €")
    print(f"   • Gains maximum: {stats['gains_max'].iloc[0]:,.0f} €")
    print(f"   • Gains minimum: {stats['gains_min'].iloc[0]:,.0f} €")


# ============================================================================
# GRAPHIQUE 5 : Performances par âge
# ============================================================================

def graph_performances_par_age():
    """Analyse des performances par âge des chevaux"""
    print_section("📊 GRAPHIQUE 5 : Performances par âge")
    
    conn = create_connection()
    
    query = """
    SELECT 
        age,
        COUNT(*) as nb_courses,
        AVG(CASE WHEN place = '1' THEN 1 ELSE 0 END) * 100 as taux_victoire,
        AVG(CASE WHEN CAST(place AS INTEGER) <= 3 THEN 1 ELSE 0 END) * 100 as taux_place
    FROM chevaux
    WHERE age IS NOT NULL 
      AND age BETWEEN 2 AND 12
      AND place IS NOT NULL
      AND place NOT IN ('D', 'A', 'T', 'NP', 'Ret')
    GROUP BY age
    ORDER BY age
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Graphique 1: Nombre de courses par âge
    ax1.bar(df['age'], df['nb_courses'], color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_title('Nombre de courses par âge', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Âge', fontsize=12)
    ax1.set_ylabel('Nombre de courses', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: Taux de victoire et de place par âge
    x = df['age']
    width = 0.35
    ax2.bar(x - width/2, df['taux_victoire'], width, label='Taux de victoire', color='#4ECDC4', edgecolor='black')
    ax2.bar(x + width/2, df['taux_place'], width, label='Taux de place (top 3)', color='#FFE66D', edgecolor='black')
    ax2.set_title('Taux de réussite par âge', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Âge', fontsize=12)
    ax2.set_ylabel('Taux (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('graphiques/performances_par_age.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/performances_par_age.png")
    plt.close()
    
    # Statistiques
    age_optimal = df.loc[df['taux_victoire'].idxmax()]
    print(f"\n📈 Statistiques:")
    print(f"   • Âge optimal (victoires): {age_optimal['age']:.0f} ans ({age_optimal['taux_victoire']:.2f}%)")
    print(f"   • Âge le plus actif: {df.loc[df['nb_courses'].idxmax(), 'age']:.0f} ans")


# ============================================================================
# GRAPHIQUE 6 : Analyse des cotes
# ============================================================================

def graph_analyse_cotes():
    """Analyse de la relation entre cote et performance"""
    print_section("📊 GRAPHIQUE 6 : Analyse des cotes")
    
    conn = create_connection()
    
    query = """
    SELECT 
        CASE 
            WHEN cote_depart < 2 THEN '< 2'
            WHEN cote_depart < 5 THEN '2-5'
            WHEN cote_depart < 10 THEN '5-10'
            WHEN cote_depart < 20 THEN '10-20'
            WHEN cote_depart < 50 THEN '20-50'
            ELSE '> 50'
        END as tranche_cote,
        COUNT(*) as nb_courses,
        AVG(CASE WHEN place = '1' THEN 1 ELSE 0 END) * 100 as taux_victoire,
        AVG(CASE WHEN CAST(place AS INTEGER) <= 3 THEN 1 ELSE 0 END) * 100 as taux_place
    FROM chevaux
    WHERE cote_depart IS NOT NULL 
      AND cote_depart > 0
      AND place IS NOT NULL
      AND place NOT IN ('D', 'A', 'T', 'NP', 'Ret')
    GROUP BY tranche_cote
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    # Ordre des tranches
    ordre = ['< 2', '2-5', '5-10', '10-20', '20-50', '> 50']
    df['tranche_cote'] = pd.Categorical(df['tranche_cote'], categories=ordre, ordered=True)
    df = df.sort_values('tranche_cote')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['taux_victoire'], width, label='Taux de victoire', 
                   color='#00A878', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, df['taux_place'], width, label='Taux de place (top 3)', 
                   color='#FE5F55', edgecolor='black', linewidth=1)
    
    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title('Taux de réussite par tranche de cote', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tranche de cote', fontsize=12)
    ax.set_ylabel('Taux (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df['tranche_cote'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('graphiques/analyse_cotes.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/analyse_cotes.png")
    plt.close()
    
    # Statistiques
    print(f"\n📈 Statistiques:")
    print(f"   • Meilleur taux victoire: tranche '{df.loc[df['taux_victoire'].idxmax(), 'tranche_cote']}' ({df['taux_victoire'].max():.2f}%)")
    print(f"   • Favoris (< 2): {df[df['tranche_cote'] == '< 2']['taux_victoire'].values[0]:.2f}% de victoire")


# ============================================================================
# GRAPHIQUE 7 : Heatmap des performances par jour de la semaine et hippodrome
# ============================================================================

def graph_heatmap_performances():
    """Heatmap des performances par jour et hippodrome"""
    print_section("📊 GRAPHIQUE 7 : Heatmap performances (jour × hippodrome)")
    
    conn = create_connection()
    
    query = """
    SELECT 
        CASE CAST(strftime('%w', date_course) AS INTEGER)
            WHEN 0 THEN 'Dimanche'
            WHEN 1 THEN 'Lundi'
            WHEN 2 THEN 'Mardi'
            WHEN 3 THEN 'Mercredi'
            WHEN 4 THEN 'Jeudi'
            WHEN 5 THEN 'Vendredi'
            WHEN 6 THEN 'Samedi'
        END as jour_semaine,
        hippodrome,
        COUNT(*) as nb_courses
    FROM chevaux
    WHERE date_course IS NOT NULL 
      AND hippodrome IS NOT NULL
      AND date_course >= date('now', '-90 days')
    GROUP BY jour_semaine, hippodrome
    HAVING COUNT(*) >= 10
    ORDER BY nb_courses DESC
    LIMIT 70
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    # Créer une matrice pivot
    pivot_df = df.pivot_table(index='hippodrome', columns='jour_semaine', 
                               values='nb_courses', fill_value=0)
    
    # Ordre des jours
    ordre_jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    pivot_df = pivot_df[[col for col in ordre_jours if col in pivot_df.columns]]
    
    # Limiter aux top 15 hippodromes
    pivot_df = pivot_df.head(15)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, 
                cbar_kws={'label': 'Nombre de courses'}, ax=ax)
    
    ax.set_title('Activité par jour de la semaine et hippodrome (90 derniers jours)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Jour de la semaine', fontsize=12)
    ax.set_ylabel('Hippodrome', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('graphiques/heatmap_performances.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/heatmap_performances.png")
    plt.close()


# ============================================================================
# GRAPHIQUE 8 : Corrélation nombre de courses vs victoires
# ============================================================================

def graph_correlation_courses_victoires():
    """Analyse de corrélation entre nombre de courses et victoires"""
    print_section("📊 GRAPHIQUE 8 : Corrélation courses × victoires")
    
    conn = create_connection()
    
    query = """
    SELECT 
        nombre_courses_total,
        nombre_victoires_total,
        gains_carriere
    FROM chevaux
    WHERE nombre_courses_total > 0 
      AND nombre_courses_total < 200
      AND nombre_victoires_total IS NOT NULL
      AND gains_carriere > 0
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    # Calculer le taux de victoire
    df['taux_victoire'] = (df['nombre_victoires_total'] / df['nombre_courses_total']) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graphique 1: Scatter plot courses vs victoires
    scatter = ax1.scatter(df['nombre_courses_total'], df['nombre_victoires_total'], 
                          c=df['gains_carriere'], cmap='viridis', alpha=0.6, s=30)
    ax1.set_title('Nombre de victoires vs nombre de courses', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Nombre de courses', fontsize=12)
    ax1.set_ylabel('Nombre de victoires', fontsize=12)
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Gains carrière (€)', fontsize=10)
    
    # Ligne de tendance
    z = np.polyfit(df['nombre_courses_total'], df['nombre_victoires_total'], 1)
    p = np.poly1d(z)
    ax1.plot(df['nombre_courses_total'], p(df['nombre_courses_total']), 
             "r--", alpha=0.8, linewidth=2, label='Tendance')
    ax1.legend()
    
    # Graphique 2: Distribution du taux de victoire
    ax2.hist(df['taux_victoire'], bins=30, color='#E63946', alpha=0.7, edgecolor='black')
    ax2.axvline(df['taux_victoire'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Moyenne: {df["taux_victoire"].mean():.1f}%')
    ax2.axvline(df['taux_victoire'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Médiane: {df["taux_victoire"].median():.1f}%')
    ax2.set_title('Distribution du taux de victoire', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Taux de victoire (%)', fontsize=12)
    ax2.set_ylabel('Nombre de chevaux', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('graphiques/correlation_courses_victoires.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/correlation_courses_victoires.png")
    plt.close()
    
    # Calculer corrélation
    correlation = df['nombre_courses_total'].corr(df['nombre_victoires_total'])
    print(f"\n📈 Statistiques:")
    print(f"   • Corrélation: {correlation:.3f}")
    print(f"   • Taux victoire moyen: {df['taux_victoire'].mean():.2f}%")
    print(f"   • Taux victoire médian: {df['taux_victoire'].median():.2f}%")


# ============================================================================
# GRAPHIQUE 9 : Top 20 chevaux les plus performants
# ============================================================================

def graph_top_chevaux():
    """Top 20 des chevaux les plus performants"""
    print_section("📊 GRAPHIQUE 9 : Top 20 chevaux")
    
    conn = create_connection()
    
    query = """
    SELECT 
        nom_cheval,
        gains_carriere,
        nombre_victoires_total,
        nombre_courses_total
    FROM chevaux
    WHERE gains_carriere > 0
    ORDER BY gains_carriere DESC
    LIMIT 20
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ Aucune donnée disponible")
        return
    
    # Tronquer les noms trop longs
    df['nom_cheval_court'] = df['nom_cheval'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Graphique 1: Gains
    colors_gains = plt.cm.plasma(np.linspace(0.3, 0.9, len(df)))
    bars1 = ax1.barh(df['nom_cheval_court'], df['gains_carriere']/1000, 
                     color=colors_gains, edgecolor='black', linewidth=1)
    ax1.set_title('Top 20 - Gains carrière', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gains (K€)', fontsize=12)
    ax1.set_ylabel('Cheval', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f' {df.iloc[i]["gains_carriere"]/1000:.0f}K€',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Graphique 2: Victoires
    colors_victoires = plt.cm.cool(np.linspace(0.3, 0.9, len(df)))
    bars2 = ax2.barh(df['nom_cheval_court'], df['nombre_victoires_total'], 
                     color=colors_victoires, edgecolor='black', linewidth=1)
    ax2.set_title('Top 20 - Nombre de victoires', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Victoires', fontsize=12)
    ax2.set_ylabel('Cheval', fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(df.iloc[i]["nombre_victoires_total"])}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphiques/top_chevaux.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/top_chevaux.png")
    plt.close()
    
    # Statistiques
    print(f"\n📈 Statistiques:")
    print(f"   • Champion (gains): {df.iloc[0]['nom_cheval']} ({df.iloc[0]['gains_carriere']:,.0f} €)")
    print(f"   • Champion (victoires): {df.loc[df['nombre_victoires_total'].idxmax(), 'nom_cheval']} "
          f"({df['nombre_victoires_total'].max():.0f} victoires)")


# ============================================================================
# GRAPHIQUE 10 : Tableau de bord synthétique
# ============================================================================

def graph_dashboard():
    """Tableau de bord synthétique avec KPIs"""
    print_section("📊 GRAPHIQUE 10 : Dashboard synthétique")
    
    conn = create_connection()
    
    # Récupérer les KPIs
    query_kpi = """
    SELECT 
        COUNT(DISTINCT nom_cheval) as total_chevaux,
        COUNT(*) as total_courses,
        COUNT(DISTINCT hippodrome) as total_hippodromes,
        AVG(gains_carriere) as gains_moyen,
        SUM(CASE WHEN place = '1' THEN 1 ELSE 0 END) as total_victoires
    FROM chevaux
    WHERE gains_carriere > 0
    """
    
    kpi = pd.read_sql_query(query_kpi, conn)
    
    # Évolution derniers 30 jours
    query_evolution = """
    SELECT 
        substr(date_course, 1, 10) as date,
        COUNT(*) as nb_courses
    FROM chevaux
    WHERE date(date_course) >= date('now', '-30 days')
    GROUP BY date
    ORDER BY date
    """
    
    evolution = pd.read_sql_query(query_evolution, conn)
    
    # Top 5 hippodromes
    query_top_hippo = """
    SELECT hippodrome, COUNT(*) as nb
    FROM chevaux
    GROUP BY hippodrome
    ORDER BY nb DESC
    LIMIT 5
    """
    
    top_hippo = pd.read_sql_query(query_top_hippo, conn)
    
    conn.close()
    
    # Créer le dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # KPI Cards (ligne 1)
    kpi_data = [
        ('Chevaux', kpi['total_chevaux'].iloc[0], '#FF6B6B'),
        ('Courses', kpi['total_courses'].iloc[0], '#4ECDC4'),
        ('Hippodromes', kpi['total_hippodromes'].iloc[0], '#95E1D3')
    ]
    
    for idx, (titre, valeur, couleur) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, idx])
        ax.text(0.5, 0.6, f'{valeur:,.0f}', ha='center', va='center', 
                fontsize=36, fontweight='bold', color=couleur)
        ax.text(0.5, 0.3, titre, ha='center', va='center', 
                fontsize=16, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                    fill=False, edgecolor=couleur, linewidth=3))
    
    # Évolution (ligne 2, colonnes 1-2)
    if not evolution.empty:
        ax_evol = fig.add_subplot(gs[1, :2])
        ax_evol.plot(evolution['date'], evolution['nb_courses'], 
                     marker='o', color='#2E86AB', linewidth=2, markersize=4)
        ax_evol.fill_between(evolution['date'], evolution['nb_courses'], alpha=0.3, color='#2E86AB')
        ax_evol.set_title('Évolution sur 30 jours', fontsize=12, fontweight='bold')
        ax_evol.set_ylabel('Courses', fontsize=10)
        ax_evol.grid(True, alpha=0.3)
        ax_evol.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Top hippodromes (ligne 2, colonne 3)
    if not top_hippo.empty:
        ax_hippo = fig.add_subplot(gs[1, 2])
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_hippo)))
        ax_hippo.barh(top_hippo['hippodrome'], top_hippo['nb'], color=colors)
        ax_hippo.set_title('Top 5 Hippodromes', fontsize=12, fontweight='bold')
        ax_hippo.set_xlabel('Courses', fontsize=10)
        ax_hippo.invert_yaxis()
        ax_hippo.tick_params(labelsize=9)
    
    # Informations générales (ligne 3)
    ax_info = fig.add_subplot(gs[2, :])
    info_text = f"""
    📊 STATISTIQUES GLOBALES
    
    • Gains moyen par cheval: {kpi['gains_moyen'].iloc[0]:,.0f} €
    • Total victoires: {kpi['total_victoires'].iloc[0]:,.0f}
    • Taux de victoire global: {(kpi['total_victoires'].iloc[0] / kpi['total_courses'].iloc[0] * 100):.2f}%
    • Base de données mise à jour
    """
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                 fontsize=14, family='monospace', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.axis('off')
    
    plt.suptitle('🏇 TABLEAU DE BORD - SYSTÈME HORSE3', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('graphiques/dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: graphiques/dashboard.png")
    plt.close()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale pour générer tous les graphiques"""
    
    print("\n" + "="*80)
    print("  🎨 GÉNÉRATEUR DE GRAPHIQUES D'ANALYSE - HORSE3")
    print("="*80 + "\n")
    
    # Créer le dossier graphiques s'il n'existe pas
    import os
    os.makedirs('graphiques', exist_ok=True)
    print("📁 Dossier 'graphiques/' prêt\n")
    
    graphiques = [
        ("Evolution des courses", graph_evolution_courses),
        ("Distribution des places", graph_distribution_places),
        ("Top hippodromes", graph_top_hippodromes),
        ("Analyse des gains", graph_analyse_gains),
        ("Performances par âge", graph_performances_par_age),
        ("Analyse des cotes", graph_analyse_cotes),
        ("Heatmap performances", graph_heatmap_performances),
        ("Corrélation courses/victoires", graph_correlation_courses_victoires),
        ("Top 20 chevaux", graph_top_chevaux),
        ("Dashboard synthétique", graph_dashboard),
    ]
    
    total = len(graphiques)
    
    for idx, (nom, fonction) in enumerate(graphiques, 1):
        try:
            print(f"\n[{idx}/{total}] Génération: {nom}")
            fonction()
        except Exception as e:
            print(f"❌ Erreur lors de la génération de '{nom}': {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("  ✅ GÉNÉRATION TERMINÉE")
    print("="*80)
    print(f"\n📂 Tous les graphiques sont dans le dossier: graphiques/")
    print(f"   Total: {total} graphiques générés\n")


if __name__ == "__main__":
    main()
