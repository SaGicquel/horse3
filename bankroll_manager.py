import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Path to persistent state file
STATE_FILE = Path("data/bankroll_state.json")


class BankrollManager:
    """
    Manages daily bankroll state and enforces Stop-Loss protection.
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or initialize defaults."""
        if not self.state_file.exists():
            return self._initialize_state()

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Check for daily reset
            today = datetime.now().strftime("%Y-%m-%d")
            if state.get("date") != today:
                return self._initialize_state()

            return state
        except Exception:
            return self._initialize_state()

    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize a fresh daily state."""
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "date": today,
            "daily_starting_bankroll": 100.0,  # Default, can be updated
            "current_daily_pnl": 0.0,
            "stop_loss_triggered": False,
        }

    def _save_state(self):
        """Persist state to file."""
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def set_starting_bankroll(self, amount: float):
        """Set the starting bankroll for the day (if not already set/modified)."""
        # Reload to check date
        self._reload_if_needed()
        self.state["daily_starting_bankroll"] = float(amount)
        self._save_state()

    def update_pnl(self, amount: float):
        """Update PnL with a win (positive) or loss (negative)."""
        self._reload_if_needed()

        self.state["current_daily_pnl"] += float(amount)
        self._check_stop_loss()
        self._save_state()

    def _check_stop_loss(self):
        """Check if PnL has breached the -10% threshold."""
        start_bank = self.state["daily_starting_bankroll"]
        pnl = self.state["current_daily_pnl"]

        # Threshold: -10% of starting bankroll
        limit = -0.10 * start_bank

        if pnl <= limit:
            self.state["stop_loss_triggered"] = True
        else:
            # Can reset if PnL recovers (optional, but safer to keep triggered?
            # PRD says "stop if I lose 10%". Usually stop-loss is a latch.)
            # We'll treat it as a latch for the day. Once triggered, stays triggered.
            pass

    def is_stop_loss_active(self) -> bool:
        """Return True if betting should be stopped."""
        self._reload_if_needed()
        return self.state["stop_loss_triggered"]

    def _reload_if_needed(self):
        """Check date and reset if needed before operations."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.get("date") != today:
            self.state = self._initialize_state()
            self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Return full status."""
        self._reload_if_needed()
        return self.state
