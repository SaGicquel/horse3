# Story 1.3: Create Bet Outcome Recording API

**Status**: Ready for Review
**Priority**: High
**Epic**: Epic 1: Safety First & Validation Architecture

## Description
As a **User**,
I want **to record the result of every bet I place**,
so that **I have trusted data to calculate my true live ROI.**

## Acceptance Criteria
- [ ] AC1: `POST /api/record-bet-outcome` accepts bet details and result.
- [ ] AC2: Data is stored in `bet_tracking` table in the database.
- [ ] AC3: Fields include: `date`, `hippodrome`, `horse`, `predicted_prob`, `odds`, `stake`, `result` (WIN/LOSS), `profit_loss`.
- [ ] AC4: Data is immutable (cannot be deleted or modified via API).
- [ ] AC5: The endpoint automatically updates the `BankrollManager` PnL when a bet outcome is recorded.

## Integration Verification
- [ ] IV1: Verify data persistence in DB after API call.
- [ ] IV2: Verify this data is separate from historical training data (no leakage).
- [ ] IV3: Verify that recording a outcome updates the daily PnL (linking Story 1.2 and 1.3).

## Tasks
- [x] Task 1: Design and create SQL table `bet_tracking`.
- [x] Task 2: Implement `POST /api/record-bet-outcome` endpoint in `user_app_api_v2.py`.
- [x] Task 3: Integrate with `BankrollManager` to update PnL automatically.
- [x] Task 4: Create integration tests for the new endpoint.
- [x] Task 5: Verify data integrity and permissions.

## Dev Agent Record

### Agent Model Used
- Model: Google DeepMind / Gemini 2.0 Flash

### Debug Log References
- Log File: .ai/debug-log.md

### Completion Notes
- Created `bet_tracking` table via SQL migration.
- Implemented `POST /api/record-bet-outcome` in `user_app_api_v2.py`.
- Integrated `BankrollManager.update_pnl()` into the recording flow.
- Verified logic with integration tests `tests/test_api_outcome_recording.py`.

### File List
- sql/create_bet_tracking_table.sql
- apply_migration_story_1_3.py
- user_app_api_v2.py
- tests/test_api_outcome_recording.py

### Change Log
- 2026-01-23: Story created.
- 2026-01-23: Implemented full recording loop. Tests passed.
