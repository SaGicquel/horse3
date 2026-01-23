import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os
import json
from pathlib import Path

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_app_api_v2 import app
from bankroll_manager import BankrollManager

TEST_STATE_FILE = Path("data/test_api_bankroll_state.json")


class TestAPIStopLossIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        if TEST_STATE_FILE.exists():
            os.remove(TEST_STATE_FILE)

        # Patch the STATE_FILE in user_app_api_v2's BankrollManager usage?
        # BankrollManager is imported in user_app_api_v2.
        # We need to mock BankrollManager inside user_app_api_v2 or patch its state_file.
        # Easier: Patch BankrollManager class in user_app_api_v2

    def tearDown(self):
        if TEST_STATE_FILE.exists():
            os.remove(TEST_STATE_FILE)

    @patch("user_app_api_v2.BankrollManager")
    @patch("user_app_api_v2.train_model_for_date")
    @patch("user_app_api_v2.get_races_for_date")
    def test_stop_loss_not_triggered(self, mock_get_races, mock_train, MockBankrollManager):
        """Test API returns 200 when Stop-Loss is NOT active."""
        # Setup mocks
        mock_bm_instance = MockBankrollManager.return_value
        mock_bm_instance.is_stop_loss_active.return_value = False

        # Mock ML returns empty list to simplify (we just want to pass the check)
        # Or non-empty
        mock_train.return_value = (MagicMock(), None)
        mock_get_races.return_value = pd.DataFrame()  # No races found, but passing check

        response = self.client.get("/daily-advice-v2")
        self.assertEqual(response.status_code, 200)

    @patch("user_app_api_v2.BankrollManager")
    def test_stop_loss_triggered_blocks_api(self, MockBankrollManager):
        """Test API returns 403 when Stop-Loss IS active."""
        # Setup mocks
        mock_bm_instance = MockBankrollManager.return_value
        mock_bm_instance.is_stop_loss_active.return_value = True

        response = self.client.get("/daily-advice-v2")

        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertEqual(data["detail"]["code"], "STOP_LOSS_TRIGGERED")

    @patch("user_app_api_v2.BankrollManager")
    def test_update_pnl_endpoint(self, MockBankrollManager):
        """Test /update-pnl endpoint calls BankrollManager."""
        mock_bm_instance = MockBankrollManager.return_value
        mock_bm_instance.get_status.return_value = {
            "current_daily_pnl": -50.0,
            "stop_loss_triggered": True,
        }

        payload = {"amount": -50.0}
        response = self.client.post("/update-pnl", json=payload)

        self.assertEqual(response.status_code, 200)

        # Verify call
        mock_bm_instance.update_pnl.assert_called_with(-50.0)

        data = response.json()
        self.assertEqual(data["stop_loss_triggered"], True)


if __name__ == "__main__":
    import pandas as pd  # Import here to mock pd in patches

    unittest.main()
