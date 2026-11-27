# PropPilot Methodology

This document details the quantitative models, event-simulation guarantees, and statistical validation architecture behind **PropPilot**.

---

## 1. Gambler's-Ruin Intuition Under Asymmetric Prop Rules

Suppose a trading strategy has positive expectancy $\mathbb{E}[R] > 0$ per unit of initial risk. On an unconstrained personal account with geometric compounding, modest variance is merely drawdowns along a rising equity curve. Under prop-firm rules, however, the account sits inside **absorbing barriers**:

1. **Daily Loss Barrier ($L_d$)**: Reaching $D_t - B_t > L_d$ (where $D_t$ is the day-start reference balance and $B_t$ is current balance) terminates the account immediately.
2. **Maximum Loss Barrier ($L_m$)**: Either a static floor $B_0 (1 - L_m)$ or a trailing high-water mark floor $\text{HWM}_t (1 - L_m)$ terminates the account immediately. Whether a trailing floor stops rising at the initial balance is an explicit rules field, `trailing_lock_at_initial` (floor $= \min(B_0, \text{HWM}_t (1 - L_m))$ when true); it is applied identically in the challenge and funded simulators.
3. **Asymmetric Upside Cap**: Reaching the profit target ($+8\%$ in Phase 1, $+5\%$ in Phase 2) closes the phase—extra upside beyond the target earns nothing during evaluation, while any downside past the barrier forfeits the entire fee and accumulated progress.

Because upside is capped while downside is absorbing, increasing risk per trade $f$ exhibits a sharp phase transition:
- Too small $f$: expected time to target grows large (increasing chance of abandonment or time-limit breach).
- Too large $f$: probability of hitting the daily or overall barrier spikes rapidly, especially when correlated crypto assets (e.g., BTC, ETH, SOL) trigger simultaneous stop-outs on the same UTC day.
- **Near-target size reduction**: Once balance reaches a fraction `near_frac` (e.g., $0.75$) of the target gain, reducing trade risk by `near_mult` (e.g., $0.5\times$) shrinks variance near the finish line without sacrificing early compounding.

---

## 2. Event-Driven Trade Simulation & Approximations

PropPilot processes trade logs chronologically as two distinct event classes per trade $i$:

1. **Entry (`entry_time`)**:
   - Computes effective risk percentage:
     - Base `risk_pct`, optionally scaled by `near_mult` when phase progress $\ge$ `near_frac`, and optionally scaled by `behind_mult` when drawdown from high-water mark $\ge$ `behind_dd_pct`.
     - Clamped by optional firm rule `max_risk_per_trade_pct`.
   - Reserves dollar risk $r_i = \text{ref\_balance} \times \text{eff\_risk\_pct} / 100$.
   - Evaluates the policy's **entry guards** prior to admitting the trade:
     - **Own Daily Stop**: refuses entry if $\max(0, D_t - B_t) + \text{open\_risk} + r_i > D_t \times \text{own\_daily\_stop\_pct} / 100$.
     - **Per-Symbol Daily Cap**: refuses entry if symbol $s$ reserved risk today $+ r_i$ exceeds `per_symbol_daily_cap_pct`.
     - **Max Open Risk**: refuses entry if $\text{open\_risk} + r_i$ exceeds `max_open_risk_pct`.
   - Marks the current reset day as an active trading day (`trading_days += 1` if first admitted entry on that reset day).

2. **Exit (`exit_time`)**:
   - Releases reserved risk $r_i$ and realises $\text{PnL}_i = R_i \cdot r_i$ into `balance`.
   - Checks in strict sequence:
     1. Daily loss breach: $D_t - B_t > D_t \times \text{daily\_loss\_pct} / 100$.
     2. High-water mark update & max loss breach: $B_t < \text{floor}_t$.
     3. Phase target pass: $B_t \ge \text{target}$, $\text{trading\_days} \ge \text{min\_days}$, and (if configured) $\text{best\_day} \le \text{total\_profit} \times \text{consistency\_pct} / 100$, where `best_day` is the largest *final* PnL of a completed reset-day (or the current day's PnL so far), never an intraday peak that was later given back.
   - When advancing from Phase 1 to Phase 2, balance resets to `account_size`, Phase 1 open trades are cleared, and Phase 2 starts fresh on subsequent entries.

**Tie-breaking**: At identical timestamps, exits of previously opened trades (`exit_time > entry_time`) execute before new entries (`entry_time`), and zero-duration trades (`exit_time == entry_time`) exit only after every entry at that timestamp has been processed.

**Daily reset**: Reset-day indices follow $\lfloor (t - h_{\text{reset}} \cdot 3600) / 86400 \rfloor$ in UTC.

**Trade-level approximation**: Because PropPilot ingests trade-level `(entry_time, exit_time, symbol, R)` rather than tick-by-tick order-book or price bars, intraday mark-to-market drawdown between entry and exit is approximated at realised exits (`day_start_max_balance_equity` and `trailing_equity` evaluate identically to closed-balance variants and are explicitly flagged in outputs).

---

## 3. Rolling Starts

To eliminate start-date luck, PropPilot launches a fresh challenge starting at the **first trade of every UTC calendar day** in the history. Only trades entering at or after each start timestamp are eligible.

- Runs reaching pass, rule breach, or `restart_after_days` are counted in `n_finished`.
- Runs still open when the dataset ends are tracked separately as `unfinished` and excluded from pass/fail rate denominators.
- Expected attempts per funded account:
  $$\mathbb{E}[\text{attempts}] = \frac{1}{\text{pass\_rate}}$$
- Expected evaluation fee cost per funded account:
  $$\mathbb{E}[\text{fees}] = \mathbb{E}[\text{attempts}] \times \text{fee}$$

---

## 4. Weekly Block-Bootstrap Monte Carlo

Standard i.i.d. trade shuffling destroys two crucial features of crypto trading:
1. **Intraday cross-symbol correlation** (e.g., multiple correlated stop-outs during a single market flush).
2. **Multi-day volatility clustering and weekend/session cadence**.

PropPilot partitions historical trades into whole 7-day Monday-to-Sunday UTC blocks (trimming partial edge weeks when the history spans at least 4 weeks). Weeks without trades are kept as empty blocks: idle weeks are part of the strategy's real behaviour, and dropping them would inflate trade density and speed. For each synthetic 1-year path ($N = 2000$ default), it samples 53 week blocks with replacement using a deterministic seed (`numpy.random.default_rng`), shifts timestamps forward week-by-week while preserving intra-week entry order and cross-week open trade overlap, and executes the exact same challenge or funded state machine.

---

## 5. Funded Phase & CPPI Sizing

Once funded, profit targets disappear while daily loss, max loss, and optional consistency constraints remain. The historical run and the bootstrap execute the same per-path state machine (`proppilot/sim/funded.py::_run_funded_path`):

- **Max-loss floor**: static $B_0(1 - L_m)$, or trailing $\text{HWM}_t(1 - L_m)$; with `trailing_lock_at_initial: true` the trailing floor locks permanently at $B_0$ the first time it reaches it.
- **Payouts & kept buffer**: every `payout_frequency_days`, the balance above $B_0(1 + \text{keep\_buffer\_pct}/100)$ is withdrawn and the trader receives `profit_split` % of it. A withdrawal is not a trading loss: it lowers the trailing high-water mark by the same amount and is excluded from the daily-loss reference.
- **Consistency** (optional): a payout is deferred while the best day of the payout cycle (completed days' final PnL, plus the current day) exceeds `consistency_pct` % of the cycle profit.
- **CPPI multiplier** (optional, `cppi_c0` set): constant-proportion sizing on the cushion above the floor,
  $$m_t = \min\!\left(\text{cap},\; \frac{\text{cushion}_t}{c_0}\right), \qquad \text{cushion}_t = \max(0, B_t - \text{floor}_t) / B_0 \times 100,$$
  so $m_t = 1$ when the cushion equals $c_0$ (the optimiser uses $c_0 = L_m$, i.e. 1x at the starting cushion), risk shrinks toward zero as the account approaches the floor, and grows up to `cap` as profits build. Because risk shrinks in proportion to the remaining cushion, losses cannot reach the max-loss floor unless trades lose more than the R values in the history allow (gaps are only modelled through the supplied R values), so CPPI variants often show P(lose) near 0 from max loss; daily-loss breaches are still simulated.
- **Reported figures**: mean/median monthly payout averages all paths *including* the zero months after an account is lost; `P(lose account)` is the share of paths that breached daily or max loss within 12 months.

---

## 6. Honest 60/40 Optimisation & Overfit Detection

To guard against curve-fitting sizing parameters to historical quirks:
1. Trades are split **strictly chronologically** into the **first 60%** (optimisation set) and **last 40%** (validation set).
2. `optimise_on_train()` receives *only* the first 60% split; validation data is structurally unreachable during grid search and `max_value` Monte Carlo scoring.
3. The winning policy is evaluated out-of-sample on the unseen last 40% rolling starts and reported side by side with optimisation and bootstrap metrics.
4. Overfit warnings automatically trigger when validation pass rate drops materially below optimisation or median time-to-funded degrades by $> 1.5\times$.

### Objectives and the funded-risk constraint

- `fastest`: lowest median days to pass among policies with $P(\text{non-pass}) \le$ `max_fail` (if none qualifies, the lowest non-pass policy is used and a warning is emitted).
- `cheapest`: lowest expected fees $= \text{fee} / \text{pass\_rate}$ among the same feasible set.
- `max_value`: highest expected 12-month value
  $$V = \mathbb{E}[\text{12m payouts}] - \mathbb{E}[\text{fees}] \times (1 + P(\text{lose funded}))$$
  (a lost funded account must be bought and passed again), ranked after the funded-risk constraint.
- For every objective the funded phase is chosen separately on the optimisation slice: funded risk $\in \{0.25, 0.5, 0.75, 1.0\}\%$ (capped by the firm's per-trade cap and the policy's own daily stop) $\times$ CPPI off/on. Among variants with $P(\text{lose funded within 12m}) \le$ `max_lose_funded` (default 15 %, CLI `--max-lose-funded`), the one with the highest expected payouts wins; if none qualifies, the lowest-risk variant is shown and flagged HIGH RISK.
- The reported funded projection is re-simulated with a fresh seed on the optimisation slice and, as an out-of-sample check, on a bootstrap of the unseen validation weeks; the plan card states both P(lose) figures next to the payout and flags any breach of the limit.
