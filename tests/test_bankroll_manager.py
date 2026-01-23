import unittest
import shutil
import os
from pathlib import Path
from bankroll_manager import BankrollManager

TEST_STATE_FILE = Path("data/test_bankroll_state.json")


class TestBankrollManager(unittest.TestCase):
    def setUp(self):
        # Ensure clean state
        if TEST_STATE_FILE.exists():
            os.remove(TEST_STATE_FILE)
        self.bm = BankrollManager(state_file=TEST_STATE_FILE)

    def tearDown(self):
        if TEST_STATE_FILE.exists():
            os.remove(TEST_STATE_FILE)

    def test_initialization(self):
        """Test fresh state initialization."""
        status = self.bm.get_status()
        self.assertEqual(status["current_daily_pnl"], 0.0)
        self.assertFalse(status["stop_loss_triggered"])
        # Default bankroll is 100
        self.assertEqual(status["daily_starting_bankroll"], 100.0)

    def test_pnl_update_no_trigger(self):
        """Test normal PnL updates."""
        self.bm.set_starting_bankroll(100.0)

        # Win 5€
        self.bm.update_pnl(5.0)
        self.assertEqual(self.bm.get_status()["current_daily_pnl"], 5.0)
        self.assertFalse(self.bm.is_stop_loss_active())

        # Lose 5€ (back to 0)
        self.bm.update_pnl(-5.0)
        self.assertEqual(self.bm.get_status()["current_daily_pnl"], 0.0)
        self.assertFalse(self.bm.is_stop_loss_active())

    def test_stop_loss_trigger(self):
        """Test triggering the stop-loss."""
        self.bm.set_starting_bankroll(100.0)

        # Lose 10€ (10% of 100) -> Trigger
        self.bm.update_pnl(-10.0)

        self.assertTrue(self.bm.is_stop_loss_active())
        self.assertEqual(self.bm.get_status()["current_daily_pnl"], -10.0)

    def test_stop_loss_latch(self):
        """Test that stop-loss stays active even if PnL recovers (latch behavior)."""
        self.bm.set_starting_bankroll(100.0)

        # Trigger
        self.bm.update_pnl(-15.0)
        self.assertTrue(self.bm.is_stop_loss_active())

        # Recover slightly (maybe a late win came in)
        self.bm.update_pnl(10.0)
        # PnL is now -5 (which is > -10 limit), but latch should hold?
        # Re-reading code: "We'll treat it as a latch... pass" in else block.
        # So yes, it should stay True.

        self.assertTrue(self.bm.is_stop_loss_active())


if __name__ == "__main__":
    unittest.main()
