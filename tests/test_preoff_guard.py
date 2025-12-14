import pytest

from web.backend.odds_guard import select_preoff_market_odds


def test_preoff_guard_rejects_postoff():
    with pytest.raises(ValueError):
        select_preoff_market_odds(cote_finale=5.0, cote_reference=None, place_finale=1)


def test_preoff_guard_prefers_reference():
    odds = select_preoff_market_odds(cote_finale=8.0, cote_reference=6.0, place_finale=None)
    assert odds == 6.0


def test_preoff_guard_allows_final_if_no_result():
    odds = select_preoff_market_odds(cote_finale=10.0, cote_reference=None, place_finale=None)
    assert odds == 10.0
