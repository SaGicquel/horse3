# Horse3 Brownfield Enhancement PRD
**Status**: DRAFT
**Author**: Product Manager (John)
**Date**: 2026-01-23

## 1. Intro Project Analysis and Context

### 1.1 Existing Project Overview
**Analysis Source**: User-provided information (Brainstorming Session Results - `docs/brainstorming-session-results.md`)

**Current Project State**:
The project is an algorithmic horse betting system currently operating on a 5-year historical database.
- **Core Logic**: Machine Learning model with a 56.8% win rate.
- **Current Staking**: Fixed stake of 10€ per bet.
- **Performance**: Backtested ROI of +71%, but suspected of overfitting.
- **Critical Flaw**: High risk of ruin (~50%) for a 100€ bankroll due to rigid staking.

### 1.2 Available Documentation Analysis
**Available Documentation**:
- [x] Brainstorming Session Results (`docs/brainstorming-session-results.md`)
- [ ] Tech Stack Documentation (Inferred Python from context)
- [ ] Source Tree/Architecture (Not explicitly analyzed, assumes standard structure)
- [ ] API Documentation (Mention of `/daily-advice` endpoint)

### 1.3 Enhancement Scope Definition
**Enhancement Type**:
- [x] Major Feature Modification (Staking Logic)
- [x] Risk Management Implementation
- [x] Stability & Validation Improvements

**Enhancement Description**:
Transformation of the betting engine from a fixed-stake model to a dynamic "Safety First" architecture. This includes implementing the Kelly Criterion for optimal staking, a Daily Stop-Loss for capital protection, and a Walk-Forward Validation pipeline to audit the model's true performance and eliminate overfitting biases.

**Impact Assessment**:
- [x] **Moderate Impact**: Requires changes to the core betting advice logic (API) and creation of new validation workflows.

### 1.4 Goals and Background Context
**Goals**:
- **Eliminate Ruin Risk**: Reduce risk of ruin from ~50% to sustainable levels (<5%) via Kelly Criterion (fraction 0.25).
- **Capital Protection**: Enforce a hard Daily Stop-Loss (10%) to prevent emotional trading.
- **True Performance Validation**: Confirm or debunk the 71% ROI using Walk-Forward Validation on "out-of-sample" data.

**Background Context**:
The current system operates on a naive fixed-staking approach which, despite high theoretical backtest numbers, is mathematically unsound for a small bankroll (100€). The recent brainstorming session identified "Risk Management" as the #1 priority. Without these changes, the project faces a near-certain probability of bankruptcy despite having a predictive edge.

## 2. Constraints & Risk Factors

### 2.1 Technical Constraints
- **Bankroll**: 100€ (very small, requires conservative approach).
- **Minimum Bet**: 2.00€ (PMU Minimum).
- **Data Quality**: 5-year BDD may include obsolete data (retired jockeys, track renovations).

### 2.2 Key Assumptions
- Model generalizes from backtest to live data.
- Win rate 56.8% holds in live environment.
- User will strict follow Kelly sizing + daily stop-loss (no overrides).

### 2.3 Open Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Live ROI << backtest | Walk-forward validation pipeline |
| Emotional betting | Automated Stop-Loss |
| Data obsolescence | Audit BDD + Filter retired jockeys |

## 3. Requirements

### 3.1 Functional Requirements (FR)
**FR1: Kelly Criterion Betting Sizing**
The system MUST calculate the optimal stake using Kelly Criterion (fraction 0.25).
- **Input**: Current Bankroll, Estimated Odds, Estimated Win Probability.
- **Output**: Suggested Stake.
- **Minimum Bet Rule (CRITICAL)**: If `calculated_stake < 2.00€` (PMU min), return `0` (NO BET). Do NOT round up.
  - *Rationale*: Maintains mathematical integrity and validation purity. Avoids forcing marginal bets.

**FR2: Daily Stop-Loss Protection**
The system MUST block new recommendations if daily cumulative loss >= 10% of starting daily bankroll.
- **Action**: API returns `STOP_LOSS_TRIGGERED`.
- **Reset**: Resets automatically at 00:00 local time.

**FR3: Walk-Forward Validation Pipeline**
The system MUST provide a script to run Walk-Forward Validation (Train on T, Test on T+1).
- **Success Criteria**: Live ROI > 20% (on out-of-sample data).

**FR4: Vincennes-Only Filter (Phase 1)**
The system MUST allow filtering recommendations to strictly "Hippodrome de Vincennes" via config.

**FR5: Bet Outcome Recording (New)**
The system MUST provide an endpoint (`POST /api/record-bet-outcome`) to log actual bet results.
- **Fields**: date, horse, hippodrome, predicted_prob, suggested_stake, actual_stake, odds, result (WIN/LOSS).
- *Purpose*: Enable true ROI tracking.

**FR6: Weekly ROI Summary Dashboard (New)**
The system MUST provide an endpoint (`GET /api/weekly-summary`) returning:
- Bets placed, Win Rate, ROI %, Comparison vs Backtest.

### 3.2 Non-Functional Requirements (NFR)
**NFR1: Bankroll Integrity**
Maximum stake per bet MUST NEVER exceed 5% of total bankroll (Hardcoded Safety Cap).

**NFR2: Latency**
API response time for advice + risk check MUST be < 200ms.

**NFR3: Data Integrity**
Bet outcomes MUST be stored in an immutable ledger (append-only table) for auditability.

### 3.3 Compatibility Requirements (CR)
**CR1: Existing API Compatibility**
Current `/daily-advice` response structure MUST be preserved. New fields (`suggested_stake`, `risk_level`, `action`) added as optional extensions.

**CR2: Database Schema**
Historical data schema MUST remain unchanged. New tables created for `bet_tracking` and `validation_runs`.

## 4. Technical Constraints and Integration Requirements

### 4.1 Existing Technology Stack
**Languages**: Python (assumed based on context)
**Frameworks**: Likely Flask or FastAPI (inferred from "API")
**Database**: SQL (implied by "BDD 5 ans")
**Infrastructure**: Local execution (assumed for Phase 1)

### 4.2 Integration Approach
**Database Integration Strategy**: Add new tables `bet_tracking` and `daily_bankroll_snapshots`. Do not alter `historical_races`.
**API Integration Strategy**: Extend `/daily-advice` with non-breaking changes. Add new endpoints `/record-bet-outcome` and `/weekly-summary`.
**Frontend Integration Strategy**: N/A (Headless API for now, Dashboard is JSON response).

## 5. Epic and Story Structure

**Epic Structure Decision**: Single "Validation & Safety" Epic.
*Rationale*: All requirements are tightly coupled around the "Phase 1 Validation" goal. Splitting them would create artificial dependencies.

### Epic 1: Safety First & Validation Architecture
**Epic Goal**: Secure the 100€ bankroll against ruin and validate the 71% ROI hypothesis with real-world tracking.
**Integration Requirements**: Must run alongside existing model without breaking legacy backtest scripts.

#### Story 1.1: Implement Kelly Criterion with Min-Bet Logic
As a **User**,
I want **the system to calculate the optimal bet size using Kelly Criterion**,
so that **I maximize growth while minimizing ruin risk.**

**Acceptance Criteria:**
1.  API accepts `current_bankroll` as input.
2.  System calculates Kelly fraction (0.25) stake.
3.  **Critical**: If calculated stake < 2.00€, API returns `stake: 0` and `action: SKIP`.
4.  **Critical**: Stake never exceeds 5% of bankroll (Safety Cap).
5.  Existing `fixed_stake` logic remains available via config flag (legacy support).

**Integration Verification:**
-   IV1: Verify `/daily-advice` returns standard response when `strategy=fixed`.
-   IV2: Verify `stake=0` when bankroll is set to very low value (e.g., 10€).

#### Story 1.2: Implement Daily Stop-Loss System
As a **User**,
I want **the system to stop giving advice if I lose 10% of my bankroll today**,
so that **I don't tilt and lose everything in one bad day.**

**Acceptance Criteria:**
1.  System tracks `daily_starting_bankroll` and `current_daily_pnl`.
2.  If `current_daily_pnl <= -0.10 * daily_starting_bankroll`, API returns `STOP_LOSS_TRIGGERED`.
3.  Stop-loss resets automatically at midnight.

**Integration Verification:**
-   IV1: Simulate a series of losses and verify API blocking advice.
-   IV2: Verify reset at next day.

#### Story 1.3: Create Bet Outcome Recording API
As a **User**,
I want **to record the result of every bet I place**,
so that **I have trusted data to calculate my true live ROI.**

**Acceptance Criteria:**
1.  `POST /api/record-bet-outcome` accepts bet details and result.
2.  Data is stored in `bet_tracking` table.
3.  Fields include: `date`, `hippodrome`, `predicted_prob`, `odds`, `stake`, `result`.
4.  Data is immutable (cannot be deleted via API).

**Integration Verification:**
-   IV1: Verify data persistence in DB after API call.
-   IV2: Verify this data is separate from historical training data (no leakage).

#### Story 1.4: Create Weekly ROI Dashboard
As a **User**,
I want **a weekly summary of my performance**,
so that **I can decide whether to continue or stop Phase 1.**

**Acceptance Criteria:**
1.  `GET /api/weekly-summary` calculates stats from `bet_tracking` table.
2.  Returns: Total Bets, Win Rate, ROI %, PnL.
3.  Returns comparison vs "Expected ROI" (from model prediction).

**Integration Verification:**
-   IV1: Verify math accuracy (manual check vs JSON output).

#### Story 1.5: Build Walk-Forward Validation Script
As a **Data Scientist**,
I want **to run a Walk-Forward backtest**,
so that **I can measure the realistic performance of the model without look-ahead bias.**

**Acceptance Criteria:**
1.  Script simulates "Train on Month M, Test on Month M+1".
2.  Iterates over the last 12 months of data.
3.  Outputs monthly ROI and Max Drawdown.
4.  Specifically handles "Vincennes Only" filter option.

**Integration Verification:**
-   IV1: Verify results differ from standard "Cross-Validation" (should be lower/more realistic).
-   IV2: Verify no data leakage (future data used for training).
