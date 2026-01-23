# Story 1.4: Create Weekly ROI Dashboard

**Status**: Ready for Review
**Priority**: High
**Epic**: Epic 1: Safety First & Validation Architecture

## Description
As a **User**,
I want **a weekly summary of my performance**,
so that **I can decide whether to continue or stop Phase 1.**

## Acceptance Criteria
- [ ] AC1: `GET /api/weekly-summary` endpoint created.
- [ ] AC2: Returns stats from `bet_tracking` table: Total Bets, Win Rate, ROI %, PnL.
- [ ] AC3: Can filter by specific week (e.g. `2026-W03`) or defaults to current week.
- [ ] AC4: Returns comparison vs "Expected ROI" (from model prediction).

## Integration Verification
- [ ] IV1: Verify math accuracy (manual check vs JSON output).
- [ ] IV2: Verify performance with SQL aggregation.

## Tasks
- [x] Task 1: Design SQL query for aggregations (weekly grouping).
- [x] Task 2: Implement `GET /api/weekly-summary` endpoint in `user_app_api_v2.py`.
- [x] Task 3: Create integration tests with mock DB data.
- [x] Task 4: Verify correct ROI calculation formulas.

## Dev Agent Record

### Agent Model Used
- Model: Google DeepMind / Gemini 2.0 Flash

### Debug Log References
- Log File: .ai/debug-log.md

### Completion Notes
- Implemented `GET /api/weekly-summary` in `user_app_api_v2.py`.
- Calculates stats directly from `bet_tracking` table using SQL Aggregation.
- Filters by ISO Week (e.g. `2026-W03`).
- Verified calculations via `tests/test_api_weekly_summary.py`.

### File List
- user_app_api_v2.py
- tests/test_api_weekly_summary.py

### Change Log
- 2026-01-23: Story created.
- 2026-01-23: Dashboard endpoint implemented and tested.
