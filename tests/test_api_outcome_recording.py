import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_app_api_v2 import app


class TestAPIOutcomeRecording(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("user_app_api_v2.get_connection")
    @patch("user_app_api_v2.BankrollManager")
    def test_record_win(self, MockBankrollManager, mock_get_conn):
        """Test recording a WIN outcome."""
        # Setup DB mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        # Returning ID 123
        mock_cursor.fetchone.return_value = [123]

        # Setup Bankroll Mock
        mock_bm_instance = MockBankrollManager.return_value

        payload = {
            "race_date": "2026-01-23",
            "hippodrome": "VINCENNES",
            "horse_name": "Winner Horse",
            "predicted_prob": 0.65,
            "odds_obtained": 3.0,
            "stake": 10.0,
            "result": "WIN",
        }

        response = self.client.post("/api/record-bet-outcome", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify PnL calculation
        # Profit = (10 * 3) - 10 = 20
        self.assertEqual(data["profit_loss"], 20.0)
        self.assertEqual(data["id"], 123)

        # Verify Bankroll Update called
        mock_bm_instance.update_pnl.assert_called_with(20.0)

        # Verify DB Insert called
        self.assertTrue(mock_cursor.execute.called)
        args, _ = mock_cursor.execute.call_args
        self.assertIn("INSERT INTO bet_tracking", args[0])
        # Check params tuple
        params = args[1]
        self.assertEqual(params[2], "Winner Horse")  # horse_name
        self.assertEqual(params[10], 20.0)  # profit_loss stored in DB

    @patch("user_app_api_v2.get_connection")
    @patch("user_app_api_v2.BankrollManager")
    def test_record_loss(self, MockBankrollManager, mock_get_conn):
        """Test recording a LOSS outcome."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = [124]

        mock_bm_instance = MockBankrollManager.return_value

        payload = {
            "race_date": "2026-01-23",
            "hippodrome": "VINCENNES",
            "horse_name": "Loser Horse",
            "predicted_prob": 0.4,
            "odds_obtained": 5.0,
            "stake": 10.0,
            "result": "LOSS",
        }

        response = self.client.post("/api/record-bet-outcome", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify PnL calculation
        # Loss = -Stake = -10
        self.assertEqual(data["profit_loss"], -10.0)

        # Verify Bankroll Update called
        mock_bm_instance.update_pnl.assert_called_with(-10.0)


if __name__ == "__main__":
    unittest.main()
