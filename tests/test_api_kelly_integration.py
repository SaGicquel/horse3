import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_app_api_v2 import app, ALGO_CONFIG


class TestAPIKellyIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Mock dataframe for races with valid timestamp string for heure_depart
        # Current time is ~1.7e12, so 1.8e12 is definitely future
        future_ts_str = "1800000000000"

        self.mock_races = pd.DataFrame(
            {
                "nom_norm": ["HORSE A", "HORSE B"],
                "race_key": ["2026-01-23|R1|C1", "2026-01-23|R1|C1"],
                "cote_reference": [8.0, 10.0],
                "distance_m": [2700, 2700],
                "age": [5, 6],
                "poids_kg": [60, 60],
                "hippodrome_code": ["VINCENNES", "VINCENNES"],
                "numero_dossard": [1, 2],
                "place_finale": [None, None],
                "heure_depart": [future_ts_str, future_ts_str],  # Needs to be digits string
            }
        )

        # Mock hippo stats
        self.mock_hippo_stats = pd.DataFrame(
            {
                "hippodrome_code": ["VINCENNES"],
                "hippodrome_place_rate": [0.4],
                "hippodrome_avg_cote": [8.0],
            }
        )

    @patch("user_app_api_v2.train_model_for_date")
    @patch("user_app_api_v2.get_races_for_date")
    def test_default_fixed_strategy(self, mock_get_races, mock_train):
        """Test that default strategy is fixed stake (10€)."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])

        mock_train.return_value = (mock_model, self.mock_hippo_stats.copy())
        mock_get_races.return_value = self.mock_races.copy()

        response = self.client.get("/daily-advice-v2?date_str=2026-01-23")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(len(data) > 0)

        # Verify default is fixed 10€
        for item in data:
            self.assertEqual(item["mise"], 10.0)
            self.assertEqual(item["action"], "PLACE")
            self.assertEqual(item["risk_level"], "NORMAL")

    @patch("user_app_api_v2.train_model_for_date")
    @patch("user_app_api_v2.get_races_for_date")
    def test_kelly_strategy_high_bankroll(self, mock_get_races, mock_train):
        """Test Kelly strategy with sufficient bankroll."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
        mock_train.return_value = (mock_model, self.mock_hippo_stats.copy())
        mock_get_races.return_value = self.mock_races.copy()

        # Bankroll 1000€
        response = self.client.get(
            "/daily-advice-v2?date_str=2026-01-23&strategy=kelly&current_bankroll=1000"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        for item in data:
            # Kelly should calculate a stake
            self.assertTrue(item["mise"] > 0)
            self.assertEqual(item["action"], "PLACE")
            self.assertLessEqual(item["mise"], 50.0)

    @patch("user_app_api_v2.train_model_for_date")
    @patch("user_app_api_v2.get_races_for_date")
    def test_kelly_strategy_low_bankroll_skip(self, mock_get_races, mock_train):
        """Test Kelly strategy with tiny bankroll triggers SKIP."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
        mock_train.return_value = (mock_model, self.mock_hippo_stats.copy())
        mock_get_races.return_value = self.mock_races.copy()

        # Bankroll 10€ -> Kelly stake likely < 2€
        response = self.client.get(
            "/daily-advice-v2?date_str=2026-01-23&strategy=kelly&current_bankroll=10"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        for item in data:
            # Should be skipped
            self.assertEqual(item["mise"], 0.0)
            self.assertEqual(item["action"], "SKIP")
            self.assertEqual(item["risk_level"], "HIGH_RISK_LOW_REWARD")


if __name__ == "__main__":
    unittest.main()
