"""
Monitoring Service - Drift Detection & Alerting
===============================================
Analyzes model performance daily and alerts on drift.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("monitoring.log"), logging.StreamHandler()],
)
logger = logging.getLogger("MonitoringService")

DATA_DIR = Path("data")
PREDICTIONS_FILE = DATA_DIR / "predictions_store.json"
FEEDBACK_FILE = DATA_DIR / "feedback_store.json"
METRICS_FILE = DATA_DIR / "metrics_store.json"


class MonitoringService:
    def __init__(self):
        self.predictions = self._load_json(PREDICTIONS_FILE)
        self.feedback = self._load_json(FEEDBACK_FILE)
        self.metrics_history = self._load_json(METRICS_FILE)

    def _load_json(self, path: Path) -> List[Dict]:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return []

    def _save_metrics(self):
        try:
            with open(METRICS_FILE, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def run_daily_check(self):
        """Run daily performance check and drift detection."""
        logger.info("Running daily monitoring check...")

        # 1. Match Predictions with Feedback
        # Map course_id -> prediction
        pred_map = {p["course_id"]: p for p in self.predictions}

        # Filter feedback for last 7 days (rolling window)
        now = datetime.now()
        start_date = now - timedelta(days=7)

        recent_feedback = [
            f
            for f in self.feedback
            if datetime.fromisoformat(f["timestamp_feedback"]) >= start_date
        ]

        if not recent_feedback:
            logger.warning("No feedback data for last 7 days. Skipping check.")
            return

        total_races = 0
        correct_top1 = 0
        correct_top3 = 0

        for fb in recent_feedback:
            course_id = fb["course_id"]
            if course_id not in pred_map:
                continue

            pred = pred_map[course_id]
            total_races += 1

            # Get winner (position_arrivee = 1)
            winner_id = next(
                (r["cheval_id"] for r in fb["resultats"] if r["position_arrivee"] == 1), None
            )

            if not winner_id:
                continue

            # Check model rank for winner
            # predictions list: [{"cheval_id": ..., "rang": 1}, ...]
            model_winner_rank = next(
                (p["rang"] for p in pred["predictions"] if p["cheval_id"] == winner_id), 999
            )

            if model_winner_rank == 1:
                correct_top1 += 1
            if model_winner_rank <= 3:
                correct_top3 += 1

        if total_races == 0:
            logger.warning("No matching prediction/feedback pairs found.")
            return

        # 2. Calculate Metrics
        acc_top1 = correct_top1 / total_races
        acc_top3 = correct_top3 / total_races

        logger.info(
            f"Metrics (Last 7d): Top1={acc_top1:.2%}, Top3={acc_top3:.2%} ({total_races} races)"
        )

        # 3. Store Metrics
        metric_record = {
            "date": now.isoformat(),
            "window_days": 7,
            "nb_races": total_races,
            "accuracy_top1": acc_top1,
            "accuracy_top3": acc_top3,
        }
        self.metrics_history.append(metric_record)
        self._save_metrics()

        # 4. Check Drift
        DRIFT_THRESHOLD = 0.25  # 25%

        if acc_top3 < DRIFT_THRESHOLD:
            self._trigger_alert(acc_top3, DRIFT_THRESHOLD)
        else:
            logger.info("✅ Model performance healthy.")

    def _trigger_alert(self, current_acc, threshold):
        """Send alert (Log/Email/Slack)."""
        msg = f"🚨 DRIFT DETECTED! Rolling Top-3 Accuracy: {current_acc:.2%} (Threshold: {threshold:.2%})"
        logger.error(msg)
        # Placeholder for Slack/Email integration
        # send_slack_alert(msg)


if __name__ == "__main__":
    service = MonitoringService()
    service.run_daily_check()
