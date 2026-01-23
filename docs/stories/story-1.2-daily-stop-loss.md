# Story 1.2: Implement Daily Stop-Loss System

**Status**: Ready for Review
**Priority**: High
**Epic**: Epic 1: Safety First & Validation Architecture

## Description
As a **User**,
I want **the system to stop giving advice if I lose 10% of my bankroll today**,
so that **I don't tilt and lose everything in one bad day.**

## Acceptance Criteria
- [ ] AC1: System tracks `daily_starting_bankroll` and `current_daily_pnl`.
- [ ] AC2: If `current_daily_pnl <= -0.10 * daily_starting_bankroll`, API returns `STOP_LOSS_TRIGGERED`.
- [ ] AC3: Stop-loss resets automatically at midnight.
- [ ] AC4: API persists daily PnL state (simple file or DB).

## Integration Verification
- [ ] IV1: Simulate a series of losses and verify API blocking advice.
- [ ] IV2: Verify reset at next day.

## Tasks
- [x] Task 1: Create a PnL Tracking Manager (class/module) to handle daily PnL state.
- [x] Task 2: Implement persistent state storage (simple JSON file in `data/` is sufficient for Phase 1).
- [x] Task 3: Update `user_app_api_v2.py` to check Stop-Loss status BEFORE generating advice.
- [x] Task 4: Add `report_outcome` logic (or manual PnL adjustment endpoint) to update the daily PnL (Prerequisite for triggering stop-loss).
- [x] Task 5: Implement unit tests for `BankrollManager` (reset logic, threshold logic).
- [x] Task 6: Verify Integration via API tests.

## Dev Agent Record

### Agent Model Used
- Model: Google DeepMind / Gemini 2.0 Flash

### Debug Log References
- Log File: .ai/debug-log.md

### Completion Notes
- Implemented `BankrollManager` in `bankroll_manager.py` with persistent JSON state.
- Integrated Stop-Loss check into `user_app_api_v2.py` (raises 403 STOP_LOSS_TRIGGERED).
- Added `POST /update-pnl` endpoint to feed PnL data.
- Verified with unit tests (`tests/test_bankroll_manager.py`) and API integration tests (`tests/test_api_stop_loss_integration.py`).

### File List
- bankroll_manager.py
- user_app_api_v2.py
- tests/test_bankroll_manager.py
- tests/test_api_stop_loss_integration.py

### Change Log
- 2026-01-23: Story created.
- 2026-01-23: Implemented BankrollManager and API integration. Tests passed.
