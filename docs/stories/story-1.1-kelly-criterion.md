# Story 1.1: Implement Kelly Criterion with Min-Bet Logic

**Status**: Ready for Review
**Priority**: High
**Epic**: Epic 1: Safety First & Validation Architecture

## Description
As a **User**,
I want **the system to calculate the optimal bet size using Kelly Criterion**,
so that **I maximize growth while minimizing ruin risk.**

## Acceptance Criteria
- [ ] AC1: API accepts `current_bankroll` as input.
- [ ] AC2: System calculates Kelly fraction (0.25) stake.
- [ ] AC3: **Critical**: If calculated stake < 2.00€, API returns `stake: 0` and `action: SKIP`.
- [ ] AC4: **Critical**: Stake never exceeds 5% of bankroll (Safety Cap).
- [ ] AC5: Existing `fixed_stake` logic remains available via config flag (legacy support).

## Integration Verification
- [ ] IV1: Verify `/daily-advice` returns standard response when `strategy=fixed`.
- [ ] IV2: Verify `stake=0` when bankroll is set to very low value (e.g., 10€).

## Tasks
- [x] Task 1: Explore existing codebase to identify API endpoint and betting logic location.
- [x] Task 2: Verify `pari_math.py` meets Kelly Criterion requirements (create verification test).
- [x] Task 3: Update API endpoint (`/daily-advice-v2`) to accept `current_bankroll` and `strategy` parameters.
- [x] Task 4: Integrate Kelly utility into the API logic to return `suggested_stake`, `action`, and `risk_level`.
- [x] Task 5: Implement tests for Kelly calculation (unit tests) and API endpoint (integration tests).
- [x] Task 6: Verify all Acceptance Criteria and Integration Verifications.

## Dev Agent Record

### Agent Model Used
- Model: Google DeepMind / Gemini 2.0 Flash

### Debug Log References
- Log File: .ai/debug-log.md

### Completion Notes
- Implemented Kelly Criterion logic using `pari_math.py` utility.
- Updated `user_app_api_v2.py` to support `strategy="kelly"` and `current_bankroll` parameters.
- Added strict Min Bet Rule (< 2€ -> SKIP) and Safety Cap (5%).
- Verified via new test suites: `tests/test_kelly_implementation.py` and `tests/test_api_kelly_integration.py`.

### File List
- user_app_api_v2.py
- tests/test_kelly_implementation.py
- tests/test_api_kelly_integration.py

### Change Log
- 2026-01-23: Story created.
- 2026-01-23: Implemented Kelly logic and API updates. Tests passed.
