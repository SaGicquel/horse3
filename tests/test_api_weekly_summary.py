import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_app_api_v2 import app


class TestAPIWeeklySummary(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("user_app_api_v2.get_connection")
    def test_weekly_summary_valid(self, mock_get_conn):
        """Test summary with data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock row: total_bets, wins, total_stake, total_pnl, avg_prob, avg_odds
        # 10 bets, 6 wins, 100 stake, 20 pnl, 0.55 avg prob, 2.5 avg odds
        mock_cursor.fetchone.return_value = [10, 6, 100.0, 20.0, 0.55, 2.5]

        response = self.client.get("/api/weekly-summary?week=2026-W03")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_bets"], 10)
        self.assertEqual(data["wins"], 6)
        self.assertEqual(data["win_rate"], 60.0)  # 6/10
        self.assertEqual(data["roi_percent"], 20.0)  # 20/100
        self.assertEqual(data["pnl"], 20.0)
        self.assertEqual(data["status"], "ON_TRACK")

    @patch("user_app_api_v2.get_connection")
    def test_weekly_summary_no_data(self, mock_get_conn):
        """Test summary with no data found."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = None  # No rows

        response = self.client.get("/api/weekly-summary?week=2026-W99")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "NO_DATA")
        self.assertEqual(data["total_bets"], 0)


if __name__ == "__main__":
    unittest.main()
