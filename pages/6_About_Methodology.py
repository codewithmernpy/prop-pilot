"""Page 6: About & Methodology."""

from __future__ import annotations

import streamlit as st

from proppilot.ui import init_page_and_sidebar, render_suite_footer

init_page_and_sidebar("About & Methodology")

st.title("6. About & Methodology")
st.caption("Mathematical design, simulation assumptions, and honest validation protocol.")

st.markdown(
    """
### Why Profitable Strategies Still Fail Prop Challenges
A positive-expectancy crypto strategy can still fail prop evaluations regularly if position sizing ignores
**asymmetric absorbing barriers** (daily loss limits and trailing/static max loss limits):
1. **Gambler's Ruin Near the Boundary**: Volatility that is harmless in an unconstrained compounding account
   terminates a prop account permanently on a single -5% day or -10% drawdown.
2. **Same-Day Crypto Correlation**: BTC, ETH, SOL, and altcoins frequently flush together during liquidation cascades.
   Five simultaneous `1%` trades behave like one `5%` bet when correlation spikes.
3. **Asymmetric Target Protection**: Once you are +6% toward an +8% Phase 1 target, full risk buys little extra speed
   while risking a giveback toward drawdown — scaling down near the target materially boosts pass rate.

### Simulation Engine Architecture
- **Event-Ordered Risk Reservation**: Each trade entry reserves `risk_pct` of reference balance; PnL (`R * reserved_risk`)
  is realised at `exit_time`. Overlapping concurrent trades are tracked as a live open-risk portfolio.
- **Daily Loss Guard & Firm Breach**: Resets at `daily_reset` UTC hour. Firm breach triggers if
  `day_start_ref - balance > daily_loss_pct%`. The policy's own entry guard blocks new entries whenever
  `realised_day_loss + open_risk + new_risk > own_daily_stop_pct%`.
- **Rolling Starts**: Launches an independent challenge at the first trade of every UTC calendar day in the history.
  Challenges still open when history ends are tracked as `unfinished` and excluded from pass/fail denominators.
- **Weekly Block Bootstrap (Monte Carlo)**: Resamples full 7-day calendar blocks (idle weeks included) with replacement
  to construct 1-year synthetic sequences preserving intra-week trade clustering, day-of-week cadence, and trade overlap.
- **Funded Phase & CPPI**: Pays out profit above `keep_buffer_pct` every `payout_frequency_days` (multiplied by `profit_split`,
  gated by optional `consistency_pct`; withdrawals lower the trailing high-water mark, they are not losses). Optional CPPI
  scales funded risk by `min(cap, cushion / c0)`, where cushion is the % of initial balance above the max-loss floor, so
  risk shrinks toward zero near the floor.
- **Trailing floors**: `trailing_lock_at_initial` in the rules decides whether a trailing max-loss floor stops at the
  initial balance; the challenge and funded simulators apply the same field.
- **Funded-risk constraint**: every optimisation picks the funded risk / CPPI variant with the highest expected payout
  among those with P(lose funded account within 12 months) <= `max_lose_funded` (default 15%); the plan card shows that
  probability (optimisation and validation bootstraps) next to the payout. `max_value` maximises
  E[12m payouts] - E[fees] x (1 + P(lose)).

### Honest 60/40 Out-of-Sample Protocol
PropPilot strictly splits trade logs chronologically:
- **First 60%**: Grid search and objective selection (`fastest`, `cheapest`, `max_value`).
- **Last 40%**: Completely unseen out-of-sample validation rolling starts. Overfit warnings fire automatically if
  validation pass rate drops sharply or median days-to-pass degrades > 1.5x.

### Explicit Limitations
- **Trade-level resolution**: Intraday mark-to-market equity between `entry_time` and `exit_time` is approximated at exits.
- **Execution friction**: Slippage, spread widening, exchange fees, and funding changes must already be reflected in net `R`.
- **Non-stationarity**: Past historical trade sequences never guarantee future edge persistence or regime stability.
"""
)

render_suite_footer()
