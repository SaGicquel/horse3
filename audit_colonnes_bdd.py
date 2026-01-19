"""
AUDIT COMPLET DES COLONNES DE LA BASE DE DONNÉES
=================================================
Analyse exhaustive des tables chevaux et cheval_courses_seen pour:
1. Identifier les colonnes utilisées vs inutilisées
2. Détecter les doublons et redondances
3. Mesurer le taux de remplissage de chaque colonne
4. Tracer les origines des données (quel scraper alimente quelle colonne)
5. Proposer un plan de nettoyage professionnel
"""

import psycopg2
from psycopg2.extras import DictCursor
import os
import re
import glob
from collections import defaultdict
from datetime import datetime

# Configuration PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "port": "54624",
    "database": "pmubdd",
    "user": "postgres",
    "password": "okokok",
}


class AuditeurColonnes:
    def __init__(self):
        self.conn = None
        self.colonnes_chevaux = []
        self.colonnes_courses = []
        self.mapping_scrapers = defaultdict(list)
        self.stats_colonnes = {}

    def connecter(self):
        """Établir connexion PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("✅ Connexion PostgreSQL établie")
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            raise

    def deconnecter(self):
        """Fermer connexion"""
        if self.conn:
            self.conn.close()
            print("✅ Connexion fermée")

    def obtenir_structure_table(self, table_name):
        """Récupérer la structure complète d'une table"""
        query = """
        SELECT
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default,
            COALESCE(col_description(('"' || table_schema || '"."' || table_name || '"')::regclass::oid,
                     ordinal_position), '') as description
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = 'public'
        ORDER BY ordinal_position;
        """

        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (table_name,))
            return cur.fetchall()

    def analyser_taux_remplissage(self, table_name, column_name):
        """Calculer le taux de remplissage d'une colonne"""
        query = f"""
        SELECT
            COUNT(*) as total,
            COUNT({column_name}) as non_null,
            COUNT(DISTINCT {column_name}) as distinct_values,
            ROUND(100.0 * COUNT({column_name}) / NULLIF(COUNT(*), 0), 2) as taux_remplissage
        FROM {table_name};
        """

        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query)
                result = cur.fetchone()
                return {
                    "total": result["total"],
                    "non_null": result["non_null"],
                    "distinct": result["distinct_values"],
                    "taux_remplissage": float(result["taux_remplissage"])
                    if result["taux_remplissage"]
                    else 0.0,
                }
        except Exception as e:
            print(f"   ⚠️  Erreur analyse {column_name}: {e}")
            return {"total": 0, "non_null": 0, "distinct": 0, "taux_remplissage": 0.0}

    def echantillon_valeurs(self, table_name, column_name, limit=5):
        """Obtenir un échantillon de valeurs non nulles"""
        query = f"""
        SELECT DISTINCT {column_name}
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        LIMIT %s;
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (limit,))
                values = [str(row[0]) for row in cur.fetchall()]
                return values
        except:
            return []

    def scanner_scrapers(self):
        """Scanner tous les scrapers pour trouver quelles colonnes ils alimentent"""
        print("\n" + "=" * 80)
        print("🔍 SCAN DES SCRAPERS")
        print("=" * 80)

        scrapers_dir = "scrapers"
        if not os.path.exists(scrapers_dir):
            print("❌ Dossier scrapers/ introuvable")
            return

        # Patterns de détection
        patterns = {
            "UPDATE": re.compile(
                r"UPDATE\s+(?:chevaux|cheval_courses_seen)\s+SET\s+([^=]+)\s*=", re.IGNORECASE
            ),
            "INSERT_COLS": re.compile(
                r"INSERT\s+INTO\s+(?:chevaux|cheval_courses_seen)\s*\(([^)]+)\)", re.IGNORECASE
            ),
            "DICT_KEY": re.compile(r'["\']([a-z_]+)["\']\s*[:=]'),  # Clés de dictionnaire
            "SQL_COL": re.compile(
                r"\b([a-z_]+)\s*=\s*%s", re.IGNORECASE
            ),  # Colonnes SQL avec placeholder
        }

        for filepath in glob.glob(f"{scrapers_dir}/*.py"):
            filename = os.path.basename(filepath)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Chercher toutes les colonnes mentionnées
                    colonnes_trouvees = set()

                    for pattern_name, pattern in patterns.items():
                        matches = pattern.findall(content)
                        for match in matches:
                            # Nettoyer et extraire les noms de colonnes
                            cols = [c.strip().strip('"').strip("'") for c in match.split(",")]
                            colonnes_trouvees.update(cols)

                    # Filtrer les vrais noms de colonnes (snake_case, pas de mots-clés SQL)
                    colonnes_valides = set()
                    for col in colonnes_trouvees:
                        if re.match(r"^[a-z][a-z0-9_]{2,}$", col) and col not in [
                            "select",
                            "from",
                            "where",
                            "and",
                            "or",
                            "values",
                        ]:
                            colonnes_valides.add(col)

                    if colonnes_valides:
                        print(f"\n📄 {filename}")
                        for col in sorted(colonnes_valides):
                            self.mapping_scrapers[col].append(filename)
                            print(f"   └─ {col}")

            except Exception as e:
                print(f"⚠️  Erreur lecture {filename}: {e}")

        print(
            f"\n✅ Scan terminé: {len(self.mapping_scrapers)} colonnes identifiées dans les scrapers"
        )

    def detecter_doublons_semantiques(self):
        """Détecter les colonnes qui semblent dupliquer la même information"""
        print("\n" + "=" * 80)
        print("🔎 DÉTECTION DES DOUBLONS SÉMANTIQUES")
        print("=" * 80)

        doublons_potentiels = []

        # Groupes de colonnes similaires
        groupes = {
            "nombre_courses": ["nombre_courses_total", "nb_courses_total", "nombre_courses_2025"],
            "nombre_victoires": ["nombre_victoires_total", "nb_victoires", "nombre_victoires_2025"],
            "nb_places": ["nb_places", "nb_places_12m"],
            "forme_recente": [
                "forme_recente",
                "forme_recente_30j",
                "forme_recente_60j",
                "forme_recente_90j",
                "score_forme_recent",
            ],
            "derniere_course": ["derniere_course", "derniere_course_date", "date_derniere_course"],
            "gains": [
                "gains_total",
                "gains_carriere",
                "gains_trot",
                "gains_plat",
                "gains_obstacle",
            ],
            "elo": ["elo_courant", "elo_avant_course"],
            "cotes": [
                "cote_matin",
                "cote_finale",
                "cote_finale_dec",
                "cote_ouverture_dec",
                "cote_live_dec",
            ],
            "temps": ["temps_str", "temps_sec", "temps_total_s"],
            "reduction_km": [
                "reduction_km_sec",
                "reduction_km",
                "reduction_km_record_trot",
                "reduction_km_record_plat",
            ],
            "speed_figure": ["sf_brut", "sf_adj", "sf_pace", "sf_best", "sf_mediane_90j"],
            "identifiants": ["num_pmu", "id_cheval_pmu", "pmu_course_id", "pmu_reunion_id"],
        }

        for groupe_nom, colonnes in groupes.items():
            print(f"\n📊 Groupe: {groupe_nom}")
            for col in colonnes:
                # Vérifier dans quelle table
                table = None
                if col in [c[0] for c in self.colonnes_chevaux]:
                    table = "chevaux"
                elif col in [c[0] for c in self.colonnes_courses]:
                    table = "cheval_courses_seen"

                if table:
                    stats = self.stats_colonnes.get(f"{table}.{col}", {})
                    taux = stats.get("taux_remplissage", 0)
                    print(f"   • {col:40s} [{table:20s}] {taux:6.2f}%")

            # Recommandation
            colonnes_existantes = [
                c
                for c in colonnes
                if f"chevaux.{c}" in self.stats_colonnes
                or f"cheval_courses_seen.{c}" in self.stats_colonnes
            ]
            if len(colonnes_existantes) > 1:
                doublons_potentiels.append((groupe_nom, colonnes_existantes))

        return doublons_potentiels

    def generer_rapport_complet(self):
        """Générer le rapport d'audit complet"""
        print("\n" + "=" * 80)
        print("📊 AUDIT COMPLET DES COLONNES")
        print("=" * 80)

        self.connecter()

        # 1. Structure des tables
        print("\n" + "=" * 80)
        print("🗂️  STRUCTURE TABLE CHEVAUX")
        print("=" * 80)
        self.colonnes_chevaux = self.obtenir_structure_table("chevaux")
        print(f"Total colonnes: {len(self.colonnes_chevaux)}")

        print("\n" + "=" * 80)
        print("🗂️  STRUCTURE TABLE CHEVAL_COURSES_SEEN")
        print("=" * 80)
        self.colonnes_courses = self.obtenir_structure_table("cheval_courses_seen")
        print(f"Total colonnes: {len(self.colonnes_courses)}")

        # 2. Analyse taux remplissage
        print("\n" + "=" * 80)
        print("📈 ANALYSE DU TAUX DE REMPLISSAGE")
        print("=" * 80)

        for table, colonnes in [
            ("chevaux", self.colonnes_chevaux),
            ("cheval_courses_seen", self.colonnes_courses),
        ]:
            print(f"\n{'='*80}")
            print(f"TABLE: {table}")
            print(f"{'='*80}")

            for col_info in colonnes:
                col_name = col_info["column_name"]

                # Skip colonnes système
                if col_name in ["id_cheval", "created_at"]:
                    continue

                stats = self.analyser_taux_remplissage(table, col_name)
                echantillon = self.echantillon_valeurs(table, col_name, 3)

                self.stats_colonnes[f"{table}.{col_name}"] = stats

                # Déterminer le statut
                taux = stats["taux_remplissage"]
                if taux == 0:
                    statut = "❌ VIDE"
                elif taux < 1:
                    statut = "🔴 CRITIQUE"
                elif taux < 10:
                    statut = "🟠 FAIBLE"
                elif taux < 50:
                    statut = "🟡 MOYEN"
                elif taux < 90:
                    statut = "🟢 BON"
                else:
                    statut = "✅ EXCELLENT"

                # Source (scraper)
                scrapers = self.mapping_scrapers.get(col_name, [])
                source = ", ".join(scrapers[:2]) if scrapers else "❓ Inconnue"

                print(f"\n{col_name:35s} {statut}")
                print(
                    f"   Taux: {taux:6.2f}% | {stats['non_null']:>8,} / {stats['total']:>8,} | Distinct: {stats['distinct']:>8,}"
                )
                print(f"   Type: {col_info['data_type']:15s} | Source: {source}")
                if echantillon:
                    print(f"   Exemple: {', '.join(echantillon[:2])}")

        # 3. Scanner scrapers
        self.scanner_scrapers()

        # 4. Doublons
        doublons = self.detecter_doublons_semantiques()

        # 5. Générer rapport markdown
        self.generer_rapport_markdown(doublons)

        self.deconnecter()

    def generer_rapport_markdown(self, doublons):
        """Générer un rapport markdown détaillé"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RAPPORT_AUDIT_COLONNES_{timestamp}.md"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("# 🔍 AUDIT COMPLET DES COLONNES - BASE DE DONNÉES PMU\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Résumé exécutif
            f.write("## 📊 RÉSUMÉ EXÉCUTIF\n\n")

            total_cols_chevaux = len(self.colonnes_chevaux)
            total_cols_courses = len(self.colonnes_courses)
            total_cols = total_cols_chevaux + total_cols_courses

            # Compter colonnes vides
            vides_chevaux = sum(
                1
                for col in self.colonnes_chevaux
                if self.stats_colonnes.get(f"chevaux.{col['column_name']}", {}).get(
                    "taux_remplissage", 100
                )
                == 0
            )
            vides_courses = sum(
                1
                for col in self.colonnes_courses
                if self.stats_colonnes.get(f"cheval_courses_seen.{col['column_name']}", {}).get(
                    "taux_remplissage", 100
                )
                == 0
            )

            # Compter colonnes critiques (<1%)
            critiques_chevaux = sum(
                1
                for col in self.colonnes_chevaux
                if 0
                < self.stats_colonnes.get(f"chevaux.{col['column_name']}", {}).get(
                    "taux_remplissage", 100
                )
                < 1
            )
            critiques_courses = sum(
                1
                for col in self.colonnes_courses
                if 0
                < self.stats_colonnes.get(f"cheval_courses_seen.{col['column_name']}", {}).get(
                    "taux_remplissage", 100
                )
                < 1
            )

            f.write("| Métrique | Valeur |\n")
            f.write("|----------|--------|\n")
            f.write(
                f"| **Total colonnes** | {total_cols} ({total_cols_chevaux} chevaux + {total_cols_courses} courses) |\n"
            )
            f.write(
                f"| **Colonnes 100% vides** | {vides_chevaux + vides_courses} ({vides_chevaux} chevaux + {vides_courses} courses) |\n"
            )
            f.write(
                f"| **Colonnes critiques (<1%)** | {critiques_chevaux + critiques_courses} ({critiques_chevaux} chevaux + {critiques_courses} courses) |\n"
            )
            f.write(f"| **Groupes de doublons** | {len(doublons)} |\n\n")

            # Détails par table
            for table, colonnes in [
                ("chevaux", self.colonnes_chevaux),
                ("cheval_courses_seen", self.colonnes_courses),
            ]:
                f.write(f"\n## 📋 TABLE: `{table}`\n\n")
                f.write(f"**Total colonnes**: {len(colonnes)}\n\n")

                # Trier par taux de remplissage
                colonnes_triees = []
                for col in colonnes:
                    col_name = col["column_name"]
                    stats = self.stats_colonnes.get(f"{table}.{col_name}", {})
                    colonnes_triees.append((col_name, col, stats))

                colonnes_triees.sort(key=lambda x: x[2].get("taux_remplissage", 0))

                # Tableau détaillé
                f.write("| Colonne | Taux | Remplies | Total | Distinct | Type | Scrapers |\n")
                f.write("|---------|------|----------|-------|----------|------|----------|\n")

                for col_name, col_info, stats in colonnes_triees:
                    taux = stats.get("taux_remplissage", 0)
                    non_null = stats.get("non_null", 0)
                    total = stats.get("total", 0)
                    distinct = stats.get("distinct", 0)
                    dtype = col_info["data_type"]

                    # Emoji statut
                    if taux == 0:
                        emoji = "❌"
                    elif taux < 1:
                        emoji = "🔴"
                    elif taux < 10:
                        emoji = "🟠"
                    elif taux < 50:
                        emoji = "🟡"
                    elif taux < 90:
                        emoji = "🟢"
                    else:
                        emoji = "✅"

                    scrapers = self.mapping_scrapers.get(col_name, [])
                    source = ", ".join(scrapers[:2]) if scrapers else "❓"

                    f.write(
                        f"| {emoji} `{col_name}` | {taux:.1f}% | {non_null:,} | {total:,} | {distinct:,} | {dtype} | {source} |\n"
                    )

            # Doublons sémantiques
            f.write("\n## 🔄 DOUBLONS SÉMANTIQUES DÉTECTÉS\n\n")
            if doublons:
                for groupe_nom, colonnes in doublons:
                    f.write(f"\n### Groupe: `{groupe_nom}`\n\n")
                    for col in colonnes:
                        for table in ["chevaux", "cheval_courses_seen"]:
                            key = f"{table}.{col}"
                            if key in self.stats_colonnes:
                                stats = self.stats_colonnes[key]
                                f.write(
                                    f"- `{col}` ({table}): {stats['taux_remplissage']:.1f}% rempli\n"
                                )
            else:
                f.write("Aucun doublon détecté.\n")

            # Recommandations
            f.write("\n## 💡 RECOMMANDATIONS\n\n")
            f.write("### 🗑️ Colonnes à supprimer (100% vides)\n\n")
            for table, colonnes in [
                ("chevaux", self.colonnes_chevaux),
                ("cheval_courses_seen", self.colonnes_courses),
            ]:
                cols_vides = [
                    col["column_name"]
                    for col in colonnes
                    if self.stats_colonnes.get(f"{table}.{col['column_name']}", {}).get(
                        "taux_remplissage", 100
                    )
                    == 0
                ]
                if cols_vides:
                    f.write(f"\n**Table `{table}`**:\n")
                    for col in cols_vides:
                        f.write(f"- `{col}`\n")

            f.write("\n### ⚠️ Colonnes critiques à investiguer (<1% remplissage)\n\n")
            for table, colonnes in [
                ("chevaux", self.colonnes_chevaux),
                ("cheval_courses_seen", self.colonnes_courses),
            ]:
                cols_critiques = [
                    (
                        col["column_name"],
                        self.stats_colonnes.get(f"{table}.{col['column_name']}", {}),
                    )
                    for col in colonnes
                    if 0
                    < self.stats_colonnes.get(f"{table}.{col['column_name']}", {}).get(
                        "taux_remplissage", 100
                    )
                    < 1
                ]
                if cols_critiques:
                    f.write(f"\n**Table `{table}`**:\n")
                    for col, stats in cols_critiques:
                        f.write(f"- `{col}`: {stats.get('taux_remplissage', 0):.2f}%\n")

        print(f"\n✅ Rapport généré: {filename}")
        return filename


def main():
    """Fonction principale"""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "🔍 AUDIT COLONNES BASE DE DONNÉES" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")

    auditeur = AuditeurColonnes()
    auditeur.generer_rapport_complet()

    print("\n" + "=" * 80)
    print("✅ AUDIT TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()
