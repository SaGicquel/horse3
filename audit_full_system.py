#!/usr/bin/env python3
"""
AUDIT SYSTÈME COMPLET - Validation du pipeline de conseils complet
Compare algo brut vs système complet (value + Kelly + filtres + profils)
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
import warnings

warnings.filterwarnings("ignore")

print("=" * 120)
print("AUDIT SYSTÈME COMPLET - PIPELINE CONSEILS vs ALGO BRUT")
print("=" * 120)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Config algo brut (validé à +70-93% ROI)
ALGO_CONFIG = {
    "features": ["cote_reference", "cote_log", "distance_m", "age", "poids_kg"],
    "xgb_params": {
        "max_depth": 7,
        "learning_rate": 0.04,
        "n_estimators": 350,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
    },
    "cote_min": 7,
    "cote_max": 15,
    "threshold": 0.50,
}

# Config système complet (comme dans user_app_api.py)
SYSTEM_CONFIG = {
    "value_min": -5,  # Value minimale acceptée (%)
    "proba_min": 1,  # Probabilité minimale (%)
    "proba_max": 95,  # Probabilité maximale (%)
    "kelly_fraction": 0.25,  # Fraction de Kelly (25% = prudent)
    "max_stake_pct": 5,  # Mise max (% bankroll)
    "bankroll": 1000,  # Bankroll de référence
}

# Périodes de test
PERIODS = [
    ("2025-08-01", "2025-08-31", "Aout 2025"),
    ("2025-09-01", "2025-09-30", "Sep 2025"),
    ("2025-10-01", "2025-10-31", "Oct 2025"),
    ("2025-11-01", "2025-11-30", "Nov 2025"),
    ("2025-12-01", "2025-12-31", "Dec 2025"),
]

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def calculate_value(proba, odds):
    """Calcule la value d'un pari (%)"""
    expected_odds = 100 / proba if proba > 0 else 999
    return ((odds / expected_odds) - 1) * 100


def calculate_kelly(proba, odds):
    """Formule de Kelly : (p * odds - 1) / (odds - 1)"""
    p = proba / 100  # Convertir en probabilité [0-1]
    if odds <= 1:
        return 0
    kelly = (p * odds - 1) / (odds - 1)
    return max(0, kelly)  # Jamais négatif


def calculate_kelly_stake(proba, odds, fraction=0.25, max_pct=5):
    """Calcule la mise Kelly fractionnée (% bankroll)"""
    kelly = calculate_kelly(proba, odds)
    stake_pct = kelly * 100 * fraction  # Fraction de Kelly
    return min(stake_pct, max_pct)  # Plafonner à max_pct%


# ============================================================================
# CHARGEMENT DONNÉES
# ============================================================================

conn = get_connection()

query = """
SELECT
    nom_norm,
    race_key,
    annee,
    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
    place_finale,
    cote_reference,
    discipline,
    distance_m,
    age,
    sexe,
    hippodrome_code,
    numero_dossard,
    poids_kg,
    handicap_distance,
    etat_piste,
    meteo_code,
    temperature_c,
    vent_kmh
FROM cheval_courses_seen
WHERE cote_reference IS NOT NULL
  AND cote_reference > 0
  AND place_finale IS NOT NULL
  AND annee >= 2023
ORDER BY race_key ASC
"""

print("\n[STEP 1/5] Chargement des données...")
df = pd.read_sql(query, conn)
conn.close()
print(f"✓ {len(df):,} courses chargées")

# Préparation
df["date"] = pd.to_datetime(df["race_key"].str.split("|").str[0])
df["cote_place"] = 1 + (df["cote_reference"] - 1) / 3.5
df["cote_log"] = np.log1p(df["cote_reference"])

# Encoder catégories
categorical_cols = ["discipline", "sexe", "hippodrome_code", "etat_piste", "meteo_code"]
df_encoded = df.copy()
for col in categorical_cols:
    if col in df_encoded.columns:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes

# ============================================================================
# FONCTION DE TEST
# ============================================================================


def test_period(train_df, test_df, period_name, config_type="algo"):
    """Teste une période avec algo brut ou système complet"""

    # Calculer stats hippodrome sur TRAIN
    hippo_stats = (
        train_df.groupby("hippodrome_code")
        .agg({"target_place": "mean", "cote_reference": "mean"})
        .reset_index()
    )
    hippo_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

    train_df = train_df.merge(hippo_stats, on="hippodrome_code", how="left", suffixes=("", "_drop"))
    test_df = test_df.merge(hippo_stats, on="hippodrome_code", how="left", suffixes=("", "_drop"))

    # Supprimer colonnes dupliquées
    train_df = train_df[[c for c in train_df.columns if not c.endswith("_drop")]]
    test_df = test_df[[c for c in test_df.columns if not c.endswith("_drop")]]

    test_df["hippodrome_place_rate"] = test_df["hippodrome_place_rate"].fillna(0.313)
    test_df["hippodrome_avg_cote"] = test_df["hippodrome_avg_cote"].fillna(
        df["cote_reference"].mean()
    )

    # Features avec stats hippodrome
    features_with_hippo = ALGO_CONFIG["features"] + ["hippodrome_place_rate", "hippodrome_avg_cote"]

    # Entraîner modèle
    X_train = train_df[features_with_hippo].values
    y_train = train_df["target_place"].values
    X_test = test_df[features_with_hippo].values

    model = xgb.XGBClassifier(**ALGO_CONFIG["xgb_params"])
    model.fit(X_train, y_train, verbose=False)

    # Prédictions
    pred_proba = model.predict_proba(X_test)[:, 1] * 100  # En %
    test_df = test_df.copy()
    test_df["proba"] = pred_proba

    # Calculer value pour chaque cheval
    test_df["value_pct"] = test_df.apply(
        lambda row: calculate_value(row["proba"], row["cote_reference"]), axis=1
    )

    if config_type == "algo":
        # ============ ALGO BRUT ============
        # Filtre: cote 7-15 + seuil 0.50 + mise uniforme 10€
        mask = (
            (test_df["proba"] >= ALGO_CONFIG["threshold"] * 100)
            & (test_df["cote_reference"] >= ALGO_CONFIG["cote_min"])
            & (test_df["cote_reference"] <= ALGO_CONFIG["cote_max"])
        )
        selected = test_df[mask].copy()
        selected["mise"] = 10  # Uniforme
        selected["type"] = "Algo brut"

    else:
        # ============ SYSTÈME COMPLET ============
        # Filtres qualité
        mask = (
            (test_df["value_pct"] >= SYSTEM_CONFIG["value_min"])
            & (test_df["proba"] >= SYSTEM_CONFIG["proba_min"])
            & (test_df["proba"] <= SYSTEM_CONFIG["proba_max"])
        )
        selected = test_df[mask].copy()

        # Calculer mise Kelly pour chaque pari
        selected["kelly_pct"] = selected.apply(
            lambda row: calculate_kelly_stake(
                row["proba"],
                row["cote_reference"],
                fraction=SYSTEM_CONFIG["kelly_fraction"],
                max_pct=SYSTEM_CONFIG["max_stake_pct"],
            ),
            axis=1,
        )

        # Mise en euros (% de bankroll)
        selected["mise"] = (selected["kelly_pct"] / 100 * SYSTEM_CONFIG["bankroll"]).round(2)

        # Filtrer mises trop faibles (< 1€)
        selected = selected[selected["mise"] >= 1].copy()
        selected["type"] = "Système complet"

    if len(selected) == 0:
        return {
            "period": period_name,
            "type": config_type,
            "n_paris": 0,
            "wins": 0,
            "win_rate": 0,
            "mise_totale": 0,
            "retour": 0,
            "roi": 0,
            "roi_baseline": 0,
            "p_value": 1.0,
            "significant": False,
        }

    # Calculs performance
    n_paris = len(selected)
    wins = selected["target_place"].sum()
    win_rate = wins / n_paris * 100

    mise_totale = selected["mise"].sum()
    selected["gain"] = selected.apply(
        lambda row: row["mise"] * row["cote_place"] if row["target_place"] == 1 else 0, axis=1
    )
    retour = selected["gain"].sum()
    roi = (retour - mise_totale) / mise_totale * 100

    # Baseline: tous les semi-outsiders 7-15
    baseline = test_df[(test_df["cote_reference"] >= 7) & (test_df["cote_reference"] <= 15)]
    baseline_rate = baseline["target_place"].mean() * 100 if len(baseline) > 0 else 0
    roi_baseline = (baseline_rate * baseline["cote_place"].mean() - 100) if len(baseline) > 0 else 0

    # Test statistique
    p_value = 1.0
    if n_paris >= 10 and len(baseline) >= 10:
        result = binomtest(
            int(wins), n_paris, baseline["target_place"].mean(), alternative="greater"
        )
        p_value = result.pvalue

    return {
        "period": period_name,
        "type": config_type,
        "n_paris": n_paris,
        "wins": int(wins),
        "win_rate": win_rate,
        "mise_totale": mise_totale,
        "retour": retour,
        "roi": roi,
        "roi_baseline": roi_baseline,
        "baseline_rate": baseline_rate,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cote_moy": selected["cote_reference"].mean(),
        "value_moy": selected["value_pct"].mean() if "value_pct" in selected.columns else 0,
        "mise_moy": selected["mise"].mean(),
    }


# ============================================================================
# TESTS SUR TOUTES LES PÉRIODES
# ============================================================================

print("\n[STEP 2/5] Préparation des périodes de test...")
results_algo = []
results_system = []

for start_date, end_date, period_name in PERIODS:
    print(f"\n{'='*120}")
    print(f"PÉRIODE: {period_name} ({start_date} → {end_date})")
    print(f"{'='*120}")

    train = df_encoded[df_encoded["date"] < pd.to_datetime(start_date)].copy()
    test = df_encoded[
        (df_encoded["date"] >= pd.to_datetime(start_date))
        & (df_encoded["date"] <= pd.to_datetime(end_date))
    ].copy()

    print(f"Train: {len(train):,} courses | Test: {len(test):,} courses")

    # Test ALGO BRUT
    print("\n[TEST 1/2] Algo brut (XGBoost + cote 7-15 + seuil 0.50)...")
    result_algo = test_period(train, test, period_name, config_type="algo")
    results_algo.append(result_algo)

    print(
        f"  → {result_algo['n_paris']} paris | {result_algo['win_rate']:.1f}% wins | ROI {result_algo['roi']:+.2f}%"
    )

    # Test SYSTÈME COMPLET
    print("\n[TEST 2/2] Système complet (value + Kelly + filtres)...")
    result_system = test_period(train, test, period_name, config_type="system")
    results_system.append(result_system)

    print(
        f"  → {result_system['n_paris']} paris | {result_system['win_rate']:.1f}% wins | ROI {result_system['roi']:+.2f}%"
    )
    print(
        f"  → Value moyenne: {result_system['value_moy']:+.1f}% | Mise moyenne: {result_system['mise_moy']:.2f}€"
    )

# ============================================================================
# COMPARAISON FINALE
# ============================================================================

print("\n" + "=" * 120)
print("COMPARAISON FINALE - ALGO BRUT vs SYSTÈME COMPLET")
print("=" * 120)

df_algo = pd.DataFrame(results_algo)
df_system = pd.DataFrame(results_system)

print("\n📊 RÉSULTATS PAR PÉRIODE:\n")
print(
    f"{'Période':<15} | {'Type':<15} | {'Paris':<6} | {'Wins':<6} | {'Win%':<7} | {'Mise tot':<10} | {'ROI':<10} | p-value  | Signif"
)
print("-" * 120)

for i, period_name in enumerate([p[2] for p in PERIODS]):
    algo_row = df_algo[df_algo["period"] == period_name].iloc[0]
    sys_row = df_system[df_system["period"] == period_name].iloc[0]

    # Algo brut
    signif_algo = "✅" if algo_row["significant"] else "⚠️"
    print(
        f"{period_name:<15} | {'Algo brut':<15} | {algo_row['n_paris']:<6} | {algo_row['wins']:<6} | "
        f"{algo_row['win_rate']:>6.1f}% | {algo_row['mise_totale']:>9.0f}€ | {algo_row['roi']:>+9.2f}% | "
        f"{algo_row['p_value']:>7.4f} | {signif_algo}"
    )

    # Système complet
    signif_sys = "✅" if sys_row["significant"] else "⚠️"
    print(
        f"{period_name:<15} | {'Système complet':<15} | {sys_row['n_paris']:<6} | {sys_row['wins']:<6} | "
        f"{sys_row['win_rate']:>6.1f}% | {sys_row['mise_totale']:>9.0f}€ | {sys_row['roi']:>+9.2f}% | "
        f"{sys_row['p_value']:>7.4f} | {signif_sys}"
    )
    print("-" * 120)

# ============================================================================
# SYNTHÈSE GLOBALE
# ============================================================================

print("\n" + "=" * 120)
print("SYNTHÈSE GLOBALE")
print("=" * 120)

# Totaux
total_algo_paris = df_algo["n_paris"].sum()
total_algo_mise = df_algo["mise_totale"].sum()
total_algo_retour = df_algo["retour"].sum()
total_algo_roi = (
    (total_algo_retour - total_algo_mise) / total_algo_mise * 100 if total_algo_mise > 0 else 0
)

total_sys_paris = df_system["n_paris"].sum()
total_sys_mise = df_system["mise_totale"].sum()
total_sys_retour = df_system["retour"].sum()
total_sys_roi = (
    (total_sys_retour - total_sys_mise) / total_sys_mise * 100 if total_sys_mise > 0 else 0
)

print(f"\n{'Métrique':<25} | {'Algo brut':<20} | {'Système complet':<20} | {'Différence':<15}")
print("-" * 90)
print(
    f"{'Paris totaux':<25} | {total_algo_paris:<20} | {total_sys_paris:<20} | {total_sys_paris - total_algo_paris:+}"
)
print(
    f"{'Mise totale':<25} | {total_algo_mise:<20.2f}€ | {total_sys_mise:<20.2f}€ | {total_sys_mise - total_algo_mise:+.2f}€"
)
print(
    f"{'Retour total':<25} | {total_algo_retour:<20.2f}€ | {total_sys_retour:<20.2f}€ | {total_sys_retour - total_algo_retour:+.2f}€"
)
print(
    f"{'ROI moyen':<25} | {total_algo_roi:<+20.2f}% | {total_sys_roi:<+20.2f}% | {total_sys_roi - total_algo_roi:+.2f}pp"
)

# Win rate global
total_algo_wins = df_algo["wins"].sum()
total_sys_wins = df_system["wins"].sum()
algo_win_rate = total_algo_wins / total_algo_paris * 100 if total_algo_paris > 0 else 0
sys_win_rate = total_sys_wins / total_sys_paris * 100 if total_sys_paris > 0 else 0

print(
    f"{'Win rate global':<25} | {algo_win_rate:<20.1f}% | {sys_win_rate:<20.1f}% | {sys_win_rate - algo_win_rate:+.1f}pp"
)

# Périodes significatives
algo_signif = df_algo["significant"].sum()
sys_signif = df_system["significant"].sum()
print(
    f"{'Périodes significatives':<25} | {algo_signif}/{len(PERIODS):<20} | {sys_signif}/{len(PERIODS):<20} | {sys_signif - algo_signif:+}"
)

# ============================================================================
# ANALYSE DÉTAILLÉE
# ============================================================================

print("\n" + "=" * 120)
print("ANALYSE DÉTAILLÉE")
print("=" * 120)

# Mise moyenne
mise_moy_algo = df_algo["mise_moy"].mean()
mise_moy_sys = df_system["mise_moy"].mean()

print("\n📈 GESTION DE MISE:")
print("  Algo brut      : Uniforme 10€")
print(
    f"  Système complet: Kelly {SYSTEM_CONFIG['kelly_fraction']*100:.0f}% (moyenne {mise_moy_sys:.2f}€, max {SYSTEM_CONFIG['max_stake_pct']}% bankroll)"
)
print(f"  → Différence moyenne: {mise_moy_sys - 10:+.2f}€ par pari")

# Value
value_moy = df_system["value_moy"].mean()
print("\n💎 VALUE BETTING:")
print("  Algo brut      : Pas de filtre value")
print(f"  Système complet: Value >= {SYSTEM_CONFIG['value_min']}% (moyenne {value_moy:+.1f}%)")

# Filtrage
print("\n🔍 FILTRAGE:")
print(
    f"  Algo brut      : Cote {ALGO_CONFIG['cote_min']}-{ALGO_CONFIG['cote_max']} + Proba >= {ALGO_CONFIG['threshold']*100}%"
)
print(
    f"  Système complet: Value >= {SYSTEM_CONFIG['value_min']}% + Proba {SYSTEM_CONFIG['proba_min']}-{SYSTEM_CONFIG['proba_max']}%"
)

# ============================================================================
# RECOMMANDATION FINALE
# ============================================================================

print("\n" + "=" * 120)
print("RECOMMANDATION FINALE")
print("=" * 120)

if total_sys_roi > total_algo_roi and sys_signif >= algo_signif:
    print("\n✅ SYSTÈME COMPLET GAGNANT")
    print("   Le système complet surperforme l'algo brut:")
    print(
        f"   - ROI: {total_sys_roi:+.2f}% vs {total_algo_roi:+.2f}% ({total_sys_roi - total_algo_roi:+.2f}pp)"
    )
    print(
        f"   - Périodes significatives: {sys_signif}/{len(PERIODS)} vs {algo_signif}/{len(PERIODS)}"
    )
    print("\n   → Utiliser le système complet (value + Kelly) pour les conseils")

elif total_algo_roi > total_sys_roi and algo_signif >= sys_signif:
    print("\n⚠️  ALGO BRUT GAGNANT")
    print("   L'algo brut surperforme le système complet:")
    print(
        f"   - ROI: {total_algo_roi:+.2f}% vs {total_sys_roi:+.2f}% ({total_algo_roi - total_sys_roi:+.2f}pp)"
    )
    print(
        f"   - Périodes significatives: {algo_signif}/{len(PERIODS)} vs {sys_signif}/{len(PERIODS)}"
    )
    print("\n   → Simplifier le système de conseils (retirer filtres complexes)")

else:
    print("\n🤔 RÉSULTATS MITIGÉS")
    print("   Les deux approches ont des forces différentes:")
    print(
        f"   - Algo brut: ROI {total_algo_roi:+.2f}%, {algo_signif} périodes significatives, {total_algo_paris} paris"
    )
    print(
        f"   - Système complet: ROI {total_sys_roi:+.2f}%, {sys_signif} périodes significatives, {total_sys_paris} paris"
    )
    print("\n   → Analyser les différences période par période pour optimiser")

print("\n" + "=" * 120)
print("FIN DE L'AUDIT")
print("=" * 120)
