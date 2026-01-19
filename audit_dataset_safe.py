#!/usr/bin/env python3
"""
AUDIT DU DATASET SAFE - VÉRIFICATION ANTI-LEAKAGE
=====================================================

Audit complet pour s'assurer qu'aucune donnée du futur ne fuite
dans les features du dataset SAFE.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def audit_dataset_safe():
    """Audit complet du dataset SAFE pour détecter les leakages"""

    logging.info("🔍 DÉMARRAGE AUDIT DATASET SAFE")
    logging.info("=" * 70)

    try:
        # Chargement des datasets
        logging.info("📂 Chargement dataset SAFE...")
        df_safe = pd.read_csv("data/ml_features_SAFE.csv")

        logging.info("📂 Chargement ancien dataset (potentiellement leaké)...")
        df_old = pd.read_csv("data/ml_features_complete.csv")

        logging.info(
            f"✅ Dataset SAFE chargé: {df_safe.shape[0]:,} lignes × {df_safe.shape[1]} colonnes"
        )
        logging.info(
            f"✅ Ancien dataset chargé: {df_old.shape[0]:,} lignes × {df_old.shape[1]} colonnes"
        )

        # 1. ANALYSE DES COLONNES
        logging.info("\n🏷️  AUDIT 1: ANALYSE DES COLONNES")
        print("-" * 50)

        colonnes_safe = set(df_safe.columns)
        colonnes_old = set(df_old.columns)

        colonnes_supprimees = colonnes_old - colonnes_safe
        colonnes_ajoutees = colonnes_safe - colonnes_old
        colonnes_communes = colonnes_safe & colonnes_old

        print(f"📊 Colonnes SAFE: {len(colonnes_safe)}")
        print(f"📊 Colonnes ANCIENNES: {len(colonnes_old)}")
        print(f"📊 Colonnes COMMUNES: {len(colonnes_communes)}")

        if colonnes_supprimees:
            print(f"\n🗑️  COLONNES SUPPRIMÉES ({len(colonnes_supprimees)}):")
            for col in sorted(colonnes_supprimees):
                print(f"   ❌ {col}")

        if colonnes_ajoutees:
            print(f"\n➕ COLONNES AJOUTÉES ({len(colonnes_ajoutees)}):")
            for col in sorted(colonnes_ajoutees):
                print(f"   ✅ {col}")

        # 2. DÉTECTION DES COLONNES SUSPECTES
        logging.info("\n🚨 AUDIT 2: DÉTECTION COLONNES SUSPECTES")
        print("-" * 50)

        # Mots-clés suspects dans les noms de colonnes
        mots_suspects = [
            "victoire",
            "place",
            "position",
            "arrivee",
            "resultat",
            "classement",
            "rang",
            "gagnant",
            "performance",
            "temps_final",
            "ecart_vainqueur",
        ]

        colonnes_suspectes = []
        for col in df_safe.columns:
            col_lower = col.lower()
            for mot in mots_suspects:
                if mot in col_lower and col != "position_arrivee":  # position_arrivee est le target
                    colonnes_suspectes.append(col)
                    break

        if colonnes_suspectes:
            print(f"⚠️  COLONNES POTENTIELLEMENT SUSPECTES ({len(colonnes_suspectes)}):")
            for col in colonnes_suspectes:
                print(f"   🔍 {col}")
        else:
            print("✅ Aucune colonne suspecte détectée!")

        # 3. VÉRIFICATION DES TARGETS
        logging.info("\n🎯 AUDIT 3: VÉRIFICATION TARGETS")
        print("-" * 50)

        targets_detectes = []
        for col in df_safe.columns:
            if col in ["position_arrivee", "victoire", "place"]:
                targets_detectes.append(col)

                # Distribution des valeurs
                print(f"\n🎯 TARGET: {col}")
                if df_safe[col].dtype in ["int64", "float64"]:
                    print(f"   📈 Min: {df_safe[col].min()}")
                    print(f"   📈 Max: {df_safe[col].max()}")
                    print(f"   📈 Moyenne: {df_safe[col].mean():.3f}")

                value_counts = df_safe[col].value_counts().head(10)
                print("   📊 Distribution (top 10):")
                for val, count in value_counts.items():
                    print(f"      {val}: {count:,} ({count/len(df_safe)*100:.1f}%)")

        # 4. VÉRIFICATION ABSENCE FUITES HISTORIQUES
        logging.info("\n🔒 AUDIT 4: VÉRIFICATION ANTI-LEAKAGE")
        print("-" * 50)

        # Colonnes qui contenaient des fuites dans l'ancien système
        anciennes_fuites = [
            "nb_victoires_carriere",
            "taux_victoires_cheval",
            "nb_places_carriere",
            "taux_places_cheval",
            "nb_victoires_jockey",
            "taux_victoires_jockey",
            "nb_places_jockey",
            "taux_places_jockey",
            "moyenne_position_cheval",
            "moyenne_position_jockey",
        ]

        fuites_trouvees = []
        for col in anciennes_fuites:
            if col in df_safe.columns:
                fuites_trouvees.append(col)

        if fuites_trouvees:
            print(f"🚨 ATTENTION: {len(fuites_trouvees)} anciennes colonnes à fuites détectées!")
            for col in fuites_trouvees:
                print(f"   ⚠️  {col}")
        else:
            print("✅ Aucune ancienne colonne à fuite détectée!")

        # 5. ÉCHANTILLON DES DONNÉES
        logging.info("\n📋 AUDIT 5: ÉCHANTILLON DES DONNÉES")
        print("-" * 50)

        # Afficher les premières lignes
        print("🔍 ÉCHANTILLON DES DONNÉES (5 premières lignes):")
        print(df_safe.head().to_string(max_cols=10))

        # Types des colonnes
        print("\n📊 TYPES DES COLONNES:")
        type_counts = df_safe.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"   {dtype}: {count} colonnes")

        # Valeurs manquantes
        print("\n🕳️  VALEURS MANQUANTES:")
        missing = df_safe.isnull().sum()
        missing_cols = missing[missing > 0].sort_values(ascending=False)

        if len(missing_cols) > 0:
            for col, count in missing_cols.head(10).items():
                pct = (count / len(df_safe)) * 100
                print(f"   {col}: {count:,} ({pct:.1f}%)")
        else:
            print("   ✅ Aucune valeur manquante!")

        # 6. COMPARAISON AVEC ANCIEN DATASET
        if len(colonnes_communes) > 0:
            logging.info("\n🔄 AUDIT 6: COMPARAISON ANCIENS/NOUVEAUX DONNÉES")
            print("-" * 50)

            # Prendre un échantillon commun pour comparer
            common_cols = (
                ["id_course", "num_cheval"]
                if all(c in colonnes_communes for c in ["id_course", "num_cheval"])
                else list(colonnes_communes)[:5]
            )

            print(f"📊 Comparaison sur {len(common_cols)} colonnes communes")

            for col in common_cols[:3]:  # Limiter pour lisibilité
                if col in df_safe.columns and col in df_old.columns:
                    # Statistiques de base
                    if df_safe[col].dtype in ["int64", "float64"] and df_old[col].dtype in [
                        "int64",
                        "float64",
                    ]:
                        safe_mean = df_safe[col].mean()
                        old_mean = df_old[col].mean()
                        diff_pct = ((safe_mean - old_mean) / old_mean * 100) if old_mean != 0 else 0

                        print(f"   {col}:")
                        print(f"      SAFE moyenne: {safe_mean:.3f}")
                        print(f"      OLD moyenne: {old_mean:.3f}")
                        print(f"      Différence: {diff_pct:+.1f}%")

        # 7. RÉSUMÉ FINAL
        logging.info("\n📝 RÉSUMÉ FINAL DE L'AUDIT")
        print("=" * 70)

        score_securite = 0
        max_score = 5

        # Critère 1: Pas de colonnes à fuite historique
        if not fuites_trouvees:
            score_securite += 1
            print("✅ [1/5] Aucune colonne à fuite historique")
        else:
            print(f"❌ [0/5] {len(fuites_trouvees)} colonnes à fuite détectées")

        # Critère 2: Colonnes supprimées (bon signe)
        if colonnes_supprimees:
            score_securite += 1
            print(f"✅ [1/5] {len(colonnes_supprimees)} colonnes supprimées (nettoyage)")
        else:
            print("⚠️  [0/5] Aucune colonne supprimée")

        # Critère 3: Nombre raisonnable de features
        if 40 <= df_safe.shape[1] <= 60:
            score_securite += 1
            print(f"✅ [1/5] Nombre de features raisonnable ({df_safe.shape[1]})")
        else:
            print(f"⚠️  [0/5] Nombre de features suspect ({df_safe.shape[1]})")

        # Critère 4: Pas trop de valeurs manquantes
        missing_pct = (df_safe.isnull().sum().sum() / (df_safe.shape[0] * df_safe.shape[1])) * 100
        if missing_pct < 10:
            score_securite += 1
            print(f"✅ [1/5] Peu de valeurs manquantes ({missing_pct:.1f}%)")
        else:
            print(f"⚠️  [0/5] Beaucoup de valeurs manquantes ({missing_pct:.1f}%)")

        # Critère 5: Targets présents
        if len(targets_detectes) >= 2:
            score_securite += 1
            print(f"✅ [1/5] Targets détectés ({len(targets_detectes)})")
        else:
            print("❌ [0/5] Targets manquants")

        print(f"\n🏆 SCORE DE SÉCURITÉ: {score_securite}/{max_score}")

        if score_securite >= 4:
            print("🟢 DATASET SÉCURISÉ - Prêt pour l'entraînement!")
        elif score_securite >= 3:
            print("🟡 DATASET PARTIELLEMENT SÉCURISÉ - Vérifications supplémentaires recommandées")
        else:
            print("🔴 DATASET NON SÉCURISÉ - Corrections nécessaires!")

        return {
            "score_securite": score_securite,
            "max_score": max_score,
            "colonnes_supprimees": list(colonnes_supprimees),
            "fuites_detectees": fuites_trouvees,
            "nb_lignes": len(df_safe),
            "nb_colonnes": df_safe.shape[1],
        }

    except Exception as e:
        logging.error(f"❌ Erreur lors de l'audit: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    audit_dataset_safe()
