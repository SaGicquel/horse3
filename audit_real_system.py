#!/usr/bin/env python3
"""
AUDIT SYSTÈME RÉEL - Pipeline complet Algo + AiSupervisor
Teste le vrai système de conseils avec Agent IA
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

# Import du système réel
from ai_supervisor import AiSupervisor, RaceContext, HorseAnalysis, SupervisorResult

print("=" * 120)
print("AUDIT SYSTÈME RÉEL - ALGO + AI SUPERVISOR")
print("=" * 120)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Config algo (identique à celle validée)
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
    "threshold_ml": 0.50,  # Seuil ML avant supervision
}

# Config AiSupervisor
SUPERVISOR_CONFIG = {
    "confidence_threshold": 0.4,  # Seuil abaissé pour rule-based (0.6 trop strict)
    "enable_llm": True,  # Activer OpenAI/Gemini si disponible
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

print("\n[STEP 1/4] Chargement des données...")
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
# INITIALISATION AI SUPERVISOR
# ============================================================================

print("\n[STEP 2/4] Initialisation AiSupervisor...")
try:
    supervisor = AiSupervisor()
    if supervisor.provider and supervisor.provider.is_available():
        print(f"✓ AiSupervisor initialisé avec {type(supervisor.provider).__name__}")
        llm_available = True
        use_supervisor = True
    else:
        print("⚠️  AiSupervisor en mode rule-based (LLM non disponible)")
        print("   → Pour ce test, utilisation algo brut uniquement (sans supervision)")
        llm_available = False
        use_supervisor = False  # Désactiver supervision si pas de LLM
        supervisor = None  # Forcer à None pour mode algo brut
except Exception as e:
    print(f"❌ Erreur initialisation AiSupervisor: {e}")
    print("   → Utilisation algo brut uniquement")
    supervisor = None
    llm_available = False
    use_supervisor = False

# ============================================================================
# FONCTION DE TEST SYSTÈME RÉEL
# ============================================================================


def test_period_with_supervisor(train_df, test_df, period_name, supervisor):
    """Teste une période avec le pipeline complet Algo + AiSupervisor"""

    # Calculer stats hippodrome sur TRAIN
    hippo_stats = (
        train_df.groupby("hippodrome_code")
        .agg({"target_place": "mean", "cote_reference": "mean"})
        .reset_index()
    )
    hippo_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

    train_df = train_df.merge(hippo_stats, on="hippodrome_code", how="left", suffixes=("", "_drop"))
    test_df = test_df.merge(hippo_stats, on="hippodrome_code", how="left", suffixes=("", "_drop"))

    train_df = train_df[[c for c in train_df.columns if not c.endswith("_drop")]]
    test_df = test_df[[c for c in test_df.columns if not c.endswith("_drop")]]

    test_df["hippodrome_place_rate"] = test_df["hippodrome_place_rate"].fillna(0.313)
    test_df["hippodrome_avg_cote"] = test_df["hippodrome_avg_cote"].fillna(
        df["cote_reference"].mean()
    )

    # Features
    features_with_hippo = ALGO_CONFIG["features"] + ["hippodrome_place_rate", "hippodrome_avg_cote"]

    # Entraîner modèle
    X_train = train_df[features_with_hippo].values
    y_train = train_df["target_place"].values
    X_test = test_df[features_with_hippo].values

    model = xgb.XGBClassifier(**ALGO_CONFIG["xgb_params"])
    model.fit(X_train, y_train, verbose=False)

    # Prédictions ML brutes
    pred_proba = model.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df["prob_ml"] = pred_proba

    # Feature importances
    feature_imp = dict(zip(features_with_hippo, model.feature_importances_))

    # Grouper par course
    selected_bets = []
    races = test_df.groupby("race_key")

    total_candidates_found = 0
    total_races_processed = 0

    for race_key, race_df in races:
        total_races_processed += 1

        # Filtrer semi-outsiders avec prob ML >= seuil
        candidates = race_df[
            (race_df["cote_reference"] >= ALGO_CONFIG["cote_min"])
            & (race_df["cote_reference"] <= ALGO_CONFIG["cote_max"])
            & (race_df["prob_ml"] >= ALGO_CONFIG["threshold_ml"])
        ].copy()

        total_candidates_found += len(candidates)

        if len(candidates) == 0:
            continue

        # Si pas de supervisor, prendre tous les candidats ML
        if supervisor is None:
            for _, bet in candidates.iterrows():
                selected_bets.append(
                    {
                        "race_key": race_key,
                        "nom": bet["nom_norm"],
                        "numero": bet["numero_dossard"],
                        "cote": bet["cote_reference"],
                        "prob_ml": bet["prob_ml"],
                        "target": bet["target_place"],
                        "confidence": bet["prob_ml"],  # Pas de supervision
                        "supervisor_decision": "NO_SUPERVISOR",
                        "anomalies": 0,
                    }
                )
            continue

        # ============ SUPERVISION IA ============
        try:
            # Préparer contexte course
            race_context = RaceContext(
                course_id=race_key,
                date=race_df["date"].iloc[0].strftime("%Y-%m-%d"),
                hippodrome=race_df["hippodrome_code"].iloc[0],
                distance=int(race_df["distance_m"].iloc[0]),
                discipline=race_df["discipline"].iloc[0],
                nombre_partants=len(race_df),
            )

            # Préparer analyse chevaux
            horses_analysis = []
            for _, horse in race_df.iterrows():
                horses_analysis.append(
                    HorseAnalysis(
                        cheval_id=str(horse.name),  # Index pandas
                        nom=horse["nom_norm"],
                        numero=int(horse["numero_dossard"]),
                        cote_sp=float(horse["cote_reference"]),
                        prob_model=float(horse["prob_ml"]),
                        rang_model=int(race_df["prob_ml"].rank(ascending=False).loc[horse.name]),
                        forme_5c=0.0,  # Non disponible dans ce dataset
                        nb_courses_12m=0,
                        nb_victoires_12m=0,
                    )
                )

            # Analyser avec AiSupervisor
            result = supervisor.analyze(race_context, horses_analysis, feature_imp)

            # DEBUG: Afficher première course analysée
            if total_races_processed == 1:
                print(
                    f"  → DEBUG Premier supervisor result: confidence={result.confidence_score:.3f}, anomalies={len(result.anomalies)}"
                )
                for anom in result.anomalies[:3]:
                    print(f"      - {anom['type']}: {anom['detail']}")

            # Décision GLOBALE pour la course:
            # Si confidence >= seuil, GARDER tous les candidats ML de cette course
            if result.confidence_score >= SUPERVISOR_CONFIG["confidence_threshold"]:
                for _, bet in candidates.iterrows():
                    selected_bets.append(
                        {
                            "race_key": race_key,
                            "nom": bet["nom_norm"],
                            "numero": bet["numero_dossard"],
                            "cote": bet["cote_reference"],
                            "prob_ml": bet["prob_ml"],
                            "target": bet["target_place"],
                            "confidence": result.confidence_score,
                            "supervisor_decision": "APPROVED",
                            "anomalies": len(result.anomalies),
                        }
                    )
            # Sinon, REJETER toute la course (anomalies détectées)

        except Exception as e:
            # Fallback: si erreur supervision, prendre candidats ML
            for _, bet in candidates.iterrows():
                selected_bets.append(
                    {
                        "race_key": race_key,
                        "nom": bet["nom_norm"],
                        "numero": bet["numero_dossard"],
                        "cote": bet["cote_reference"],
                        "prob_ml": bet["prob_ml"],
                        "target": bet["target_place"],
                        "confidence": bet["prob_ml"],
                        "supervisor_decision": "ERROR",
                        "anomalies": 0,
                    }
                )

    print(
        f"  → DEBUG: {total_races_processed} courses, {total_candidates_found} candidats ML trouvés"
    )

    # Calculs performance
    if len(selected_bets) == 0:
        return {
            "period": period_name,
            "n_paris": 0,
            "wins": 0,
            "win_rate": 0,
            "roi": 0,
            "baseline_rate": 0,
            "p_value": 1.0,
            "significant": False,
            "confidence_avg": 0,
            "anomalies_avg": 0,
            "approved": 0,
            "rejected": 0,
            "cote_moy": 0,
        }

    df_bets = pd.DataFrame(selected_bets)
    n_paris = len(df_bets)
    wins = df_bets["target"].sum()
    win_rate = wins / n_paris * 100

    # Calcul ROI (mise 10€)
    df_bets["cote_place"] = 1 + (df_bets["cote"] - 1) / 3.5
    df_bets["gain"] = df_bets.apply(
        lambda row: 10 * row["cote_place"] if row["target"] == 1 else 0, axis=1
    )
    mise_totale = n_paris * 10
    retour = df_bets["gain"].sum()
    roi = (retour - mise_totale) / mise_totale * 100

    # Stats supervision
    confidence_avg = df_bets["confidence"].mean()
    anomalies_avg = df_bets["anomalies"].mean()
    approved = len(df_bets[df_bets["supervisor_decision"] == "APPROVED"])
    rejected_count = len(candidates) - approved if supervisor else 0

    # Baseline
    baseline = test_df[(test_df["cote_reference"] >= 7) & (test_df["cote_reference"] <= 15)]
    baseline_rate = baseline["target_place"].mean() * 100 if len(baseline) > 0 else 0

    # Test statistique
    p_value = 1.0
    if n_paris >= 10 and len(baseline) >= 10:
        result = binomtest(
            int(wins), n_paris, baseline["target_place"].mean(), alternative="greater"
        )
        p_value = result.pvalue

    return {
        "period": period_name,
        "n_paris": n_paris,
        "wins": int(wins),
        "win_rate": win_rate,
        "roi": roi,
        "baseline_rate": baseline_rate,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "confidence_avg": confidence_avg,
        "anomalies_avg": anomalies_avg,
        "approved": approved,
        "rejected": rejected_count,
        "cote_moy": df_bets["cote"].mean(),
    }


# ============================================================================
# TESTS SUR TOUTES LES PÉRIODES
# ============================================================================

print("\n[STEP 3/4] Tests sur périodes...")
results = []

for start_date, end_date, period_name in PERIODS:
    print(f"\n{'='*120}")
    print(f"PÉRIODE: {period_name}")
    print(f"{'='*120}")

    train = df_encoded[df_encoded["date"] < pd.to_datetime(start_date)].copy()
    test = df_encoded[
        (df_encoded["date"] >= pd.to_datetime(start_date))
        & (df_encoded["date"] <= pd.to_datetime(end_date))
    ].copy()

    print(f"Train: {len(train):,} | Test: {len(test):,}")

    result = test_period_with_supervisor(train, test, period_name, supervisor)
    results.append(result)

    print(f"  → Paris: {result['n_paris']}")
    print(f"  → Win rate: {result['win_rate']:.1f}% (baseline {result['baseline_rate']:.1f}%)")
    print(f"  → ROI: {result['roi']:+.2f}%")
    if supervisor:
        print(f"  → Confidence moyenne: {result['confidence_avg']:.3f}")
        print(f"  → Anomalies moyenne: {result['anomalies_avg']:.1f}")

# ============================================================================
# SYNTHÈSE
# ============================================================================

print("\n" + "=" * 120)
print("SYNTHÈSE FINALE - SYSTÈME RÉEL AVEC AI SUPERVISOR")
print("=" * 120)

df_results = pd.DataFrame(results)

print(
    f"\n{'Période':<15} | {'Paris':<6} | {'Wins':<6} | {'Win%':<7} | {'ROI':<10} | {'Conf':<6} | p-value  | Signif"
)
print("-" * 90)

for _, row in df_results.iterrows():
    signif = "✅" if row["significant"] else "⚠️"
    conf_str = f"{row['confidence_avg']:.3f}" if row["confidence_avg"] > 0 else "N/A"
    print(
        f"{row['period']:<15} | {row['n_paris']:<6} | {row['wins']:<6} | "
        f"{row['win_rate']:>6.1f}% | {row['roi']:>+9.2f}% | {conf_str:<6} | "
        f"{row['p_value']:>7.4f} | {signif}"
    )

# Totaux
total_paris = df_results["n_paris"].sum()
total_wins = df_results["wins"].sum()
total_mise = total_paris * 10
total_retour = (df_results["wins"] * df_results["cote_moy"] * 10).sum()
total_roi = (total_retour - total_mise) / total_mise * 100 if total_mise > 0 else 0
avg_confidence = df_results["confidence_avg"].mean()
total_signif = df_results["significant"].sum()

print("=" * 90)
print(
    f"{'TOTAL':<15} | {total_paris:<6} | {total_wins:<6} | "
    f"{total_wins/total_paris*100:>6.1f}% | {total_roi:>+9.2f}% | {avg_confidence:.3f} | "
    f"       - | {total_signif}/5"
)

print("\n" + "=" * 120)
print("CONCLUSION")
print("=" * 120)

if supervisor:
    if llm_available:
        print("\n✅ SYSTÈME TESTÉ: Algo ML + AiSupervisor (avec LLM)")
        print(f"   - Provider: {type(supervisor.provider).__name__}")
    else:
        print("\n⚠️  SYSTÈME TESTÉ: Algo ML + AiSupervisor (rule-based)")
        print("   - LLM non disponible, utilisation règles seulement")

    print("\n📊 PERFORMANCE:")
    print(f"   - Paris totaux: {total_paris}")
    print(f"   - ROI moyen: {total_roi:+.2f}%")
    print(f"   - Confidence moyenne: {avg_confidence:.3f}")
    print(f"   - Périodes significatives: {total_signif}/5")
else:
    print("\n❌ SYSTÈME NON TESTÉ")
    print("   AiSupervisor non disponible, résultats = algo brut seulement")

print("\n" + "=" * 120)
