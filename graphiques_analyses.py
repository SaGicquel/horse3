#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Visualisation Simplifié - Base de données PMU
Génère des graphiques à partir de la table chevaux
"""

import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Configuration des styles
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

DB_PATH = "data/database.db"


def create_connection():
    return sqlite3.connect(DB_PATH)


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


# GRAPHIQUE 1 : Distribution par race
def graph_distribution_races():
    print_section("📊 GRAPHIQUE 1 : Distribution par race")

    conn = create_connection()
    query = """
    SELECT race, COUNT(*) as nb_chevaux
    FROM chevaux
    WHERE race IS NOT NULL AND race != ''
    GROUP BY race
    ORDER BY nb_chevaux DESC
    LIMIT 15
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ Aucune donnée disponible")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df["race"], df["nb_chevaux"], color=colors, edgecolor="black", linewidth=1)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(width):,}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Top 15 des races de chevaux", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Nombre de chevaux", fontsize=12)
    ax.set_ylabel("Race", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("graphiques/distribution_races.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/distribution_races.png")
    plt.close()

    print("\n📈 Statistiques:")
    print(f"   • Race #1: {df.iloc[0]['race']} ({df.iloc[0]['nb_chevaux']:,} chevaux)")


# GRAPHIQUE 2 : Distribution par sexe
def graph_distribution_sexe():
    print_section("📊 GRAPHIQUE 2 : Distribution par sexe")

    conn = create_connection()
    query = """
    SELECT sexe, COUNT(*) as nb_chevaux
    FROM chevaux
    WHERE sexe IS NOT NULL AND sexe != ''
    GROUP BY sexe
    ORDER BY nb_chevaux DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ Aucune donnée disponible")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Camembert
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    wedges, texts, autotexts = ax1.pie(
        df["nb_chevaux"],
        labels=df["sexe"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax1.set_title("Répartition par sexe", fontsize=14, fontweight="bold")

    # Barres
    bars = ax2.bar(df["sexe"], df["nb_chevaux"], color=colors, edgecolor="black", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_title("Nombre par sexe", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Sexe", fontsize=12)
    ax2.set_ylabel("Nombre de chevaux", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("graphiques/distribution_sexe.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/distribution_sexe.png")
    plt.close()


# GRAPHIQUE 3 : Distribution par pays de naissance
def graph_distribution_pays():
    print_section("📊 GRAPHIQUE 3 : Distribution par pays")

    conn = create_connection()
    query = """
    SELECT pays_naissance, COUNT(*) as nb_chevaux
    FROM chevaux
    WHERE pays_naissance IS NOT NULL AND pays_naissance != ''
    GROUP BY pays_naissance
    ORDER BY nb_chevaux DESC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ Aucune donnée disponible")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(
        df["pays_naissance"], df["nb_chevaux"], color=colors, edgecolor="black", linewidth=1
    )

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(width):,}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Top 20 des pays de naissance", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Nombre de chevaux", fontsize=12)
    ax.set_ylabel("Pays", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("graphiques/distribution_pays.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/distribution_pays.png")
    plt.close()

    print("\n📈 Statistiques:")
    print(f"   • Pays #1: {df.iloc[0]['pays_naissance']} ({df.iloc[0]['nb_chevaux']:,} chevaux)")


# GRAPHIQUE 4 : Analyse des victoires
def graph_analyse_victoires():
    print_section("📊 GRAPHIQUE 4 : Analyse des victoires")

    conn = create_connection()
    query = """
    SELECT
        CASE
            WHEN nombre_victoires_total = 0 THEN '0'
            WHEN nombre_victoires_total BETWEEN 1 AND 5 THEN '1-5'
            WHEN nombre_victoires_total BETWEEN 6 AND 10 THEN '6-10'
            WHEN nombre_victoires_total BETWEEN 11 AND 20 THEN '11-20'
            WHEN nombre_victoires_total BETWEEN 21 AND 50 THEN '21-50'
            ELSE '> 50'
        END as tranche,
        COUNT(*) as nb_chevaux
    FROM chevaux
    WHERE nombre_victoires_total IS NOT NULL
    GROUP BY tranche
    """
    df = pd.read_sql_query(query, conn)

    ordre = ["0", "1-5", "6-10", "11-20", "21-50", "> 50"]
    df["tranche"] = pd.Categorical(df["tranche"], categories=ordre, ordered=True)
    df = df.sort_values("tranche")
    conn.close()

    if df.empty:
        print("❌ Aucune donnée disponible")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    bars = ax.bar(df["tranche"], df["nb_chevaux"], color=colors, edgecolor="black", linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Distribution par nombre de victoires", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Nombre de victoires", fontsize=12)
    ax.set_ylabel("Nombre de chevaux", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("graphiques/analyse_victoires.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/analyse_victoires.png")
    plt.close()


# GRAPHIQUE 5 : Top entraîneurs
def graph_top_entraineurs():
    print_section("📊 GRAPHIQUE 5 : Top 20 entraîneurs")

    conn = create_connection()
    query = """
    SELECT entraineur_courant, COUNT(*) as nb_chevaux
    FROM chevaux
    WHERE entraineur_courant IS NOT NULL AND entraineur_courant != ''
    GROUP BY entraineur_courant
    ORDER BY nb_chevaux DESC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ Aucune donnée disponible")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.cool(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(
        df["entraineur_courant"], df["nb_chevaux"], color=colors, edgecolor="black", linewidth=1
    )

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(width):,}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Top 20 des entraîneurs", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Nombre de chevaux", fontsize=12)
    ax.set_ylabel("Entraîneur", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("graphiques/top_entraineurs.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/top_entraineurs.png")
    plt.close()

    print("\n📈 Statistiques:")
    print(
        f"   • Entraîneur #1: {df.iloc[0]['entraineur_courant']} ({df.iloc[0]['nb_chevaux']:,} chevaux)"
    )


# GRAPHIQUE 6 : Top chevaux par victoires
def graph_top_chevaux_victoires():
    print_section("📊 GRAPHIQUE 6 : Top 20 chevaux (victoires)")

    conn = create_connection()
    query = """
    SELECT nom, nombre_victoires_total, nombre_courses_total
    FROM chevaux
    WHERE nombre_victoires_total > 0
    ORDER BY nombre_victoires_total DESC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ Aucune donnée disponible")
        return

    df["nom_court"] = df["nom"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(
        df["nom_court"], df["nombre_victoires_total"], color=colors, edgecolor="black", linewidth=1
    )

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {int(width)}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Top 20 des chevaux par victoires", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Nombre de victoires", fontsize=12)
    ax.set_ylabel("Cheval", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("graphiques/top_chevaux_victoires.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/top_chevaux_victoires.png")
    plt.close()

    print("\n📈 Statistiques:")
    print(f"   • Champion: {df.iloc[0]['nom']} ({df.iloc[0]['nombre_victoires_total']} victoires)")


# GRAPHIQUE 7 : Dashboard KPI
def graph_dashboard():
    print_section("📊 GRAPHIQUE 7 : Dashboard KPI")

    conn = create_connection()
    query = """
    SELECT
        COUNT(*) as total_chevaux,
        COUNT(DISTINCT race) as total_races,
        COUNT(DISTINCT pays_naissance) as total_pays,
        SUM(nombre_victoires_total) as total_victoires,
        SUM(nombre_courses_total) as total_courses,
        COUNT(DISTINCT entraineur_courant) as total_entraineurs
    FROM chevaux
    """
    kpi = pd.read_sql_query(query, conn)
    conn.close()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # KPI Cards
    kpis = [
        ("Chevaux", kpi["total_chevaux"].iloc[0], "#FF6B6B", (0, 0)),
        ("Races", kpi["total_races"].iloc[0], "#4ECDC4", (0, 1)),
        ("Pays", kpi["total_pays"].iloc[0], "#95E1D3", (0, 2)),
        ("Victoires", kpi["total_victoires"].iloc[0], "#FFE66D", (1, 0)),
        ("Courses", kpi["total_courses"].iloc[0], "#A8E6CF", (1, 1)),
        ("Entraîneurs", kpi["total_entraineurs"].iloc[0], "#FFDAB9", (1, 2)),
    ]

    for titre, valeur, couleur, (row, col) in kpis:
        ax = fig.add_subplot(gs[row, col])
        ax.text(
            0.5,
            0.6,
            f"{valeur:,.0f}",
            ha="center",
            va="center",
            fontsize=32,
            fontweight="bold",
            color=couleur,
        )
        ax.text(0.5, 0.3, titre, ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor=couleur, linewidth=3)
        )

    # Informations générales
    ax_info = fig.add_subplot(gs[2, :])
    taux_victoire = (
        (kpi["total_victoires"].iloc[0] / kpi["total_courses"].iloc[0] * 100)
        if kpi["total_courses"].iloc[0] > 0
        else 0
    )
    info_text = f"""
    📊 STATISTIQUES GLOBALES - BASE DE DONNÉES HORSE3

    • Total chevaux: {kpi['total_chevaux'].iloc[0]:,.0f}
    • Total victoires: {kpi['total_victoires'].iloc[0]:,.0f}
    • Total courses: {kpi['total_courses'].iloc[0]:,.0f}
    • Taux de victoire moyen: {taux_victoire:.2f}%
    • Nombre de races différentes: {kpi['total_races'].iloc[0]:,.0f}
    • Nombre de pays: {kpi['total_pays'].iloc[0]:,.0f}
    • Nombre d'entraîneurs: {kpi['total_entraineurs'].iloc[0]:,.0f}
    """
    ax_info.text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=13,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax_info.axis("off")

    plt.suptitle("🏇 TABLEAU DE BORD - SYSTÈME HORSE3", fontsize=18, fontweight="bold", y=0.98)

    plt.savefig("graphiques/dashboard.png", dpi=300, bbox_inches="tight")
    print("✅ Graphique sauvegardé: graphiques/dashboard.png")
    plt.close()


# FONCTION PRINCIPALE
def main():
    print("\n" + "=" * 80)
    print("  🎨 GÉNÉRATEUR DE GRAPHIQUES D'ANALYSE - HORSE3")
    print("=" * 80 + "\n")

    import os

    os.makedirs("graphiques", exist_ok=True)
    print("📁 Dossier 'graphiques/' prêt\n")

    graphiques = [
        ("Distribution par race", graph_distribution_races),
        ("Distribution par sexe", graph_distribution_sexe),
        ("Distribution par pays", graph_distribution_pays),
        ("Analyse des victoires", graph_analyse_victoires),
        ("Top 20 entraîneurs", graph_top_entraineurs),
        ("Top 20 chevaux (victoires)", graph_top_chevaux_victoires),
        ("Dashboard KPI", graph_dashboard),
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

    print("\n" + "=" * 80)
    print("  ✅ GÉNÉRATION TERMINÉE")
    print("=" * 80)
    print("\n📂 Tous les graphiques sont dans le dossier: graphiques/")
    print(f"   Total: {total} graphiques générés\n")


if __name__ == "__main__":
    main()
