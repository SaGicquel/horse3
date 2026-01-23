import unittest
import sys
import os

# Add root directory to path to import pari_math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pari_math import kelly_stake


class TestKellyImplementation(unittest.TestCase):
    def test_basic_kelly(self):
        """Test basic Kelly calculation with positive EV."""
        bankroll = 100.0
        p = 0.6  # 60% chance
        odds = 2.0  # Even money
        # Kelly full = (1*0.6 - 0.4) / 1 = 0.2 (20%)
        # Kelly 0.25 = 5%
        # Stake = 5% of 100 = 5€

        # Note: pari_math reduces odds slightly for parimutuel safety if parimutuel=True
        # Let's test with parimutuel=False first to verify raw math
        stake = kelly_stake(p, odds, bankroll, fraction=0.25, max_stake_pct=1.0, parimutuel=False)
        self.assertAlmostEqual(stake, 5.0, delta=0.1)

    def test_safety_cap(self):
        """Test that stake never exceeds max_stake_pct (5%)."""
        bankroll = 1000.0
        p = 0.9  # 90% chance
        odds = 2.0
        # Kelly full is high (~80%)
        # Kelly 0.25 is 20% -> 200€
        # Cap is 5% -> 50€

        stake = kelly_stake(p, odds, bankroll, fraction=0.25, max_stake_pct=0.05, parimutuel=False)
        self.assertEqual(stake, 50.0)

    def test_min_bet_rule_skip(self):
        """Test that stake is 0 if calculated stake < min_stake (2€)."""
        bankroll = 10.0  # Small bankroll
        p = 0.6
        odds = 2.0
        # Kelly 0.25 ~ 5%
        # Stake = 0.5€
        # Min bet = 2€
        # Should return 0

        stake = kelly_stake(p, odds, bankroll, fraction=0.25, min_stake=2.0, parimutuel=False)
        self.assertEqual(stake, 0.0)

    def test_min_bet_rule_pass(self):
        """Test that stake is returned if calculated stake >= min_stake (2€)."""
        bankroll = 100.0
        p = 0.6
        odds = 2.0
        # Kelly 0.25 ~ 5%
        # Stake = 5€
        # Min bet = 2€
        # Should return 5.0

        stake = kelly_stake(p, odds, bankroll, fraction=0.25, min_stake=2.0, parimutuel=False)
        self.assertAlmostEqual(stake, 5.0, delta=0.1)

    def test_parimutuel_adjustment(self):
        """Test that parimutuel=True reduces stake slightly (safety)."""
        bankroll = 100.0
        p = 0.6
        odds = 2.0

        stake_raw = kelly_stake(p, odds, bankroll, fraction=0.25, parimutuel=False)
        stake_safe = kelly_stake(p, odds, bankroll, fraction=0.25, parimutuel=True)

        self.assertLess(stake_safe, stake_raw)
        self.assertGreater(stake_safe, 0)  # Should still be positive for this good bet


if __name__ == "__main__":
    unittest.main()
