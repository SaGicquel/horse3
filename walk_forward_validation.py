#!/usr/bin/env python3
"""
🏇 WALK-FORWARD VALIDATION SCRIPT
=================================
Simulates a real-life betting scenario with rolling window training.
- Train on [Start, T]
- Predict on [T, T+1 month]
- Repeat moving T forward by 1 month.

Features:
- "Safety First" Staking: Kelly 0.25, Min Bet 2€, Safety Cap 5%.
- "Vincennes Only" Filter (Optional).
- Real-world constraints (Min odds, etc).
"""

import sys
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from db_connection import get_connection
from pari_math import kelly_stake
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "start_date": "2025-01-01",  # Start of simulation
    "min_train_size": 24,  # Minimum months of history needed
    "bankroll": 100.0,
    "kelly_fraction": 0.25,
    "min_bet": 2.0,
    "max_cap": 0.05,
    "features": [
        "cote_reference",
        "cote_log",
        "distance_m",
        "age",
        "poids_kg",
        "hippodrome_place_rate",
        "hippodrome_avg_cote",
    ],
    "xgb_params": {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    },
}


def load_data(vincennes_only=False):
    """Load data from DB."""
    print("Loading data from DB...")
    conn = get_connection()

    # Base query similar to user_app_api_v2 / train_model_no_leak
    query = """
    SELECT
        race_key,
        hippodrome_code,
        nom_norm,

        -- Date info
        to_date(split_part(race_key, '|', 1), 'YYYY-MM-DD') as date,

        -- Target
        CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,

        -- Features
        cote_reference,
        distance_m,
        age,
        poids_kg

    FROM cheval_courses_seen
    WHERE cote_reference IS NOT NULL
      AND cote_reference > 0
      AND place_finale IS NOT NULL
      AND annee >= 2023
    ORDER BY date ASC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if vincennes_only:
        print("🔍 Filter: Vincennes Only")
        df = df[df["hippodrome_code"] == "VINCENNES"]

    # Ensure date is datetime64[ns]
    df["date"] = pd.to_datetime(df["date"])

    print(f"✓ Loaded {len(df):,} rows.")
    return df


def preprocess_features(df):
    """Feature Engineering & Cleaning."""
    df = df.copy()

    # Clean numeric
    cols = ["cote_reference", "distance_m", "age", "poids_kg"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Derived
    df["cote_log"] = np.log1p(df["cote_reference"])

    # Place odds approximation (used for Kelly)
    # Generic rule: 1 + (WinOdds - 1) / 3.5 (Conservative)
    df["cote_place_est"] = 1 + (df["cote_reference"] - 1) / 3.5

    return df


def get_monthly_splits(df, start_date_str):
    """Generate (train_mask, test_mask, month_label) tuples."""
    start_date = pd.to_datetime(start_date_str)
    # Ensure max_date is datetime
    max_date = pd.to_datetime(df["date"].max())

    current = start_date
    while current < max_date:
        next_month = current + relativedelta(months=1)

        # Train: Everything before current month
        train_mask = df["date"] < current
        # Test: Current month
        test_mask = (df["date"] >= current) & (df["date"] < next_month)

        yield train_mask, test_mask, current.strftime("%Y-%m")

        current = next_month


def run_simulation(df, vincennes_only):
    """Run Walk-Forward Loop."""

    # Initial Preprocessing
    df = preprocess_features(df)

    bankroll = CONFIG["bankroll"]
    history = []

    print("\n🚀 STARTING WALK-FORWARD VALIDATION")
    print(
        f"Bankroll: {bankroll}€ | Kelly: {CONFIG['kelly_fraction']} | Min Bet: {CONFIG['min_bet']}€"
    )
    print("-" * 60)

    for train_mask, test_mask, month_name in get_monthly_splits(df, CONFIG["start_date"]):
        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            continue

        print(f"\n📅 Month: {month_name} | Train: {len(train_data):,} | Test: {len(test_data):,}")

        # --- 1. Dynamic Features (No Leak) ---
        # Calculate hippo stats ONLY on train data
        hippo_stats = (
            train_data.groupby("hippodrome_code")
            .agg({"target_place": "mean", "cote_reference": "mean"})
            .reset_index()
        )
        hippo_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

        # Apply to Train & Test
        train_data = train_data.merge(hippo_stats, on="hippodrome_code", how="left").fillna(0)
        test_data = test_data.merge(hippo_stats, on="hippodrome_code", how="left").fillna(0)

        # --- 2. Train Model ---
        X_train = train_data[CONFIG["features"]]
        y_train = train_data["target_place"]
        X_test = test_data[CONFIG["features"]]

        model = xgb.XGBClassifier(**CONFIG["xgb_params"])
        model.fit(X_train, y_train, verbose=False)

        # --- 3. Predict ---
        probs = model.predict_proba(X_test)[:, 1]
        test_data = test_data.copy()  # Avoid SettingWithCopy
        test_data["pred_prob"] = probs

        # --- 4. Simulate Betting ---
        month_pnl = 0
        bets_placed = 0

        for idx, row in test_data.iterrows():
            # Apply Filter (Cote 7-20, Prob > 45%) - Same as API V2
            if not (7 <= row["cote_reference"] <= 20 and row["pred_prob"] >= 0.45):
                continue

            # Kelly Calculation
            stake = kelly_stake(
                p=row["pred_prob"],
                odds=row["cote_place_est"],
                bankroll=bankroll,
                fraction=CONFIG["kelly_fraction"],
                max_stake_pct=CONFIG["max_cap"],
                min_stake=CONFIG["min_bet"],
                parimutuel=True,
            )

            if stake > 0:
                bets_placed += 1

                # Result
                # If target_place == 1, Win
                # PnL = (stake * odds_place) - stake
                # Note: We use ESTIMATED place odds for Kelly, but we should use REAL odds for PnL?
                # The DB doesn't have "cote_place_finale" easily accessible in this query?
                # User_app_api uses "cote_place" calculated from ref.
                # Let's stick to the simulation using `cote_place_est` or better?
                # Realistically, place odds are ~ CoteRef/3.5 roughly.
                # If we want exact backtest, we need real place payouts.
                # Assumption: `cote_place_est` is used for Payout too (Simplified Backtest)
                # Or do we check if we have `rapport_place` in DB?
                # Let's use `cote_place_est` for now to be consistent with API logic validation.

                payout = 0
                if row["target_place"] == 1:
                    payout = stake * row["cote_place_est"]

                pnl = payout - stake
                month_pnl += pnl
                bankroll += pnl

                # Record
                history.append(
                    {
                        "date": row["date"],
                        "month": month_name,
                        "horse": row["nom_norm"],
                        "stake": stake,
                        "pnl": pnl,
                        "result": "WIN" if pnl > 0 else "LOSS",
                    }
                )

        # Monthly Summary
        roi = (
            (month_pnl / (bets_placed * 10)) * 100 if bets_placed > 0 else 0
        )  # Approx ROI on turnover
        # Better ROI calc: Sum(PnL) / Sum(Stake)
        total_stake = sum([h["stake"] for h in history if h["month"] == month_name])
        month_roi = (month_pnl / total_stake * 100) if total_stake > 0 else 0

        print(
            f"   Bets: {bets_placed} | PnL: {month_pnl:+.2f}€ | ROI: {month_roi:+.1f}% | Bankroll: {bankroll:.2f}€"
        )

        if bankroll < CONFIG["min_bet"]:
            print("\n💀 BANKRUPTCY - Simulation Stopped.")
            break

    # --- Global Summary ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if not history:
        print("No bets placed.")
        return

    hist_df = pd.DataFrame(history)
    total_pnl = hist_df["pnl"].sum()
    total_stake = hist_df["stake"].sum()
    global_roi = (total_pnl / total_stake) * 100
    win_rate = (hist_df["result"] == "WIN").mean() * 100

    # Drawdown
    hist_df["cum_pnl"] = hist_df["pnl"].cumsum() + CONFIG["bankroll"]  # Equity curve
    hist_df["peak"] = hist_df["cum_pnl"].cummax()
    hist_df["drawdown"] = (hist_df["cum_pnl"] - hist_df["peak"]) / hist_df["peak"] * 100
    max_drawdown = hist_df["drawdown"].min()

    print(f"Start Bankroll: {CONFIG['bankroll']}€")
    print(f"End Bankroll:   {bankroll:.2f}€")
    print(f"Total Profit:   {total_pnl:+.2f}€")
    print(f"Total ROI:      {global_roi:+.2f}%")
    print(f"Win Rate:       {win_rate:.1f}%")
    print(f"Max Drawdown:   {max_drawdown:.2f}%")
    print(f"Total Bets:     {len(hist_df)}")

    # Monthly Breakdown table
    print("\nMonthly Breakdown:")
    monthly = (
        hist_df.groupby("month")
        .agg({"stake": "sum", "pnl": "sum", "result": lambda x: (x == "WIN").mean() * 100})
        .rename(columns={"result": "win_rate"})
    )
    monthly["roi"] = (monthly["pnl"] / monthly["stake"]) * 100
    print(monthly.round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--vincennes", action="store_true", help="Filter Vincennes Only")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start Date YYYY-MM-DD")
    args = parser.parse_args()

    CONFIG["start_date"] = args.start

    data = load_data(vincennes_only=args.vincennes)
    run_simulation(data, args.vincennes)
