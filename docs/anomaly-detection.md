# Anomaly Detection (Phase 3)

> **An anomaly is a statistically unusual month, judged only against the months before it.**
> Detection is deterministic and interpretable. Every result states the observed value, the
> expected value, the deviation, the method, the score, the threshold and the historical window.
> It never states a cause and never labels a movement good or bad. Explaining *why* a movement
> happened is left to later, evidence-based agent reasoning (Phase 4).

- Forecasting, series preparation and the forecasting models: [forecasting.md](forecasting.md)
- KPI definitions: [kpi-catalog.md](kpi-catalog.md)

## 1. Modules

| Module | Responsibility |
|---|---|
| `app/anomalies/config.py` | `AnomalyConfig`, `StandardizedThresholds`, `IQRFences` (every tunable number) |
| `app/anomalies/detectors.py` | The three detectors and the transforms (pure functions of a numeric series) |
| `app/anomalies/thresholds.py` | The severity policy and flag thresholds |
| `app/anomalies/models.py` | `AnomalyResult`, `AnomalyReport`, `AnomalyScan` and detector-specific details |
| `app/anomalies/service.py` | `AnomalyService`, `detect_anomalies`, and the pure core `detect_in_series` |

Series come from the shared preparation layer (`app/timeseries`, built on Phase 2 KPIs, see
[forecasting.md §2–3](forecasting.md#2-supported-metrics)). The anomaly package never reads the
database directly, never reads the generator or the injected-event ground truth, and contains no
hard-coded event months, countries or segments (`tests/unit/test_phase3_isolation.py`).

## 2. Interface

```python
from datetime import date
from app.anomalies import detect_anomalies, AnomalyService

report = detect_anomalies(
    db,
    metric="support_ticket_volume",      # revenue | mrr | customer_count | support_ticket_volume | product_adoption
    start_date=date(2025, 9, 1),         # first month scored (default: 11 months before the end month)
    end_date=date(2026, 8, 31),          # last month scored; nothing later is read (default: as-of date)
    detector="rolling_zscore",           # rolling_zscore | iqr | forecast_residual
    filters={"ticket_category": "Bug"},  # optional, one explicit group, validated by the Phase 2 filters
    window=12,                           # prior months forming the baseline
)
report.anomalies        # flagged months, ranked (see §7)
report.results          # every scored month, chronological, including normal ones
report.skipped          # months that could not be scored, with the reason

AnomalyService(db).scan(detector="forecast_residual")   # one detector over an explicit metric list
```

Scope is always explicit: one metric, one detector, optionally one filter set (total, a segment, a
region, a country, a plan, ...). Dimension combinations are **never scanned automatically**.
`scan` runs one detector over a named list of metrics (default: the four metrics that need no
filter) and ranks their anomalies together.

## 3. Point-in-time rules

For the month being judged, month **t**:

- The baseline is built from the **window** `[t − window, t − 1]` (default 12 months). Month t is
  **never** in its own baseline, dispersion, quartiles, forecast or threshold.
- Months after t are never used. The series query itself ends at `end_date`.
- A month needs at least `min_history` (default 6) prior observations (or prior residuals) to be
  scored. Otherwise it is listed in `skipped` with the reason.

Tests: `tests/unit/test_anomaly_leakage.py` changes x_t by +5,000 and asserts that the baseline,
dispersion, quartiles, forecast, bounds and threshold are unchanged. It also changes every later
month and asserts that earlier assessments are identical. `tests/integration/test_phase3_leakage_database.py`
alters a copy of the real database after 2026-05-31 and asserts that detections ending at
2026-05-31 are identical.

## 4. Transforms

The business series trend upwards. A level-based baseline would flag ordinary growth: for a straight
line, the current month is about 1.8 standard deviations above the mean of the previous 12. The
rolling z-score and IQR detectors therefore compare **movements**:

| Metric | Default transform |
|---|---|
| revenue, mrr, customer_count, support_ticket_volume | `pct_change`: x_t / x_{t−1} − 1 |
| product_adoption | `difference`: x_t − x_{t−1} (percentage-point change of a rate) |
| any (explicit) | `level`: x_t itself |

The expected value is always reported in the metric's unit. For `pct_change` it is
`x_{t−1} · (1 + baseline change)`, i.e. "what this month would have been at the usual growth". A
`pct_change` is undefined after a zero or missing month. That month is then skipped, never scored
as infinite. The forecast-residual detector works on levels, because its expectation model
handles the trend itself.

## 5. Detectors

### 5.1 Rolling z-score

    baseline mean  μ = mean(z_{t−w} .. z_{t−1})
    dispersion     s = sample std (ddof = 1) of the same window
    score          (z_t − μ) / s

Returned details: `transform`, `transformed_value`, `baseline_mean`, `baseline_std` and a
two-sided **tail probability**. Under i.i.d. normal data, (z_t − μ)/(s·√(1 + 1/n)) follows
Student's t with n − 1 degrees of freedom. This accounts for small windows and answers "how
unusual is this month compared with its baseline?". `lower_bound` and `upper_bound` are the metric
values at which the score would reach the flag threshold.

### 5.2 Robust IQR (Tukey fences)

    Q1, median, Q3 of the prior window (linear interpolation, Hyndman–Fan type 7)
    IQR = Q3 − Q1
    inner fences: Q1 − 1.5·IQR, Q3 + 1.5·IQR      outer fences: Q1 − 3·IQR, Q3 + 3·IQR
    score = (z_t − Q3)/IQR above the box, (z_t − Q1)/IQR below it, 0 inside it

Returned details: `q1, median, q3, iqr`, inner and outer fences (transformed units),
`lower_bound`/`upper_bound` (the inner fences in metric units) and the expected value (the median
movement applied to last month). Quartiles are robust to earlier outliers, which suits
distributions that are not normal.

### 5.3 Forecast residual

    for every month j: fit the expectation model (default: drift) on the `window` months before j,
                       forecast j one step ahead,  r_j = x_j − forecast_j
    for month t:       r̄, s_r = mean and std of the prior residuals r_{t−w} .. r_{t−1}
                       expected = forecast_t + r̄   (bias-corrected one-step forecast)
                       score    = (x_t − expected) / s_r

Returned details: `expectation_model`, the training window, `one_step_forecast`, `residual`,
`residual_mean`, `residual_std`, `standardized_residual` and the tail probability. The dispersion
comes from **historical out-of-sample** one-step errors, none of which uses x_t. The expectation
model can be any forecasting model from the allow-list (`naive`, `seasonal_naive`,
`moving_average`, `drift`, `ets_damped_trend`). Drift is the default because it follows the trend,
works from few months and can be checked by hand.

## 6. Thresholds and severity

Severity is deterministic and depends only on the score:

| Detector family | normal | watch | significant | extreme |
|---|---|---|---|---|
| Rolling z-score, forecast residual (standardised score) | \|s\| < 2 | 2 ≤ \|s\| < 3 | 3 ≤ \|s\| < 4 | \|s\| ≥ 4 |
| IQR (Tukey) | inside the inner fences | — | beyond the inner fence (1.5·IQR) | beyond the outer fence (3·IQR) |

- For a normal distribution the standardised cut-offs have two-sided tail probabilities of about
  4.6%, 0.27% and 0.006%. Small windows have heavier tails, which is why the t tail probability is
  also reported.
- Tukey defines no band below the inner fence, so the IQR detector never reports `watch`. For a
  normal distribution the inner fence lies about 2.7σ and the outer fence about 4.7σ from the
  median, broadly in line with the standardised cut-offs.
- A month is flagged (`is_anomaly`) from `significant` upwards (`flag_severity`, configurable).
  `threshold` is the score at which a month is flagged: 3.0 for the standardised detectors and
  1.5 for IQR.
- **Zero dispersion.** When every value in the window is identical, a month equal to the window has
  score 0. A month that differs has an unbounded score (`score=None`) and is `extreme`, because it
  lies outside everything previously observed.

## 7. Direction, ranking and anomaly start

- `direction` is **statistical only**: `positive` = observed above expected, `negative` = below,
  `none` = equal. It carries no business meaning: more support tickets is `positive`. Every report
  says so in its limitations.
- Flagged months are **ranked by the absolute standardised score, descending**. Unbounded scores
  come first, and ties go in chronological order (`AnomalyResult.rank_key`). There is no subjective
  "business importance" ranking.
- `anomaly_start` answers "when did it begin?". It is the first month of the consecutive,
  same-direction run of flagged months that the anomaly belongs to.

## 8. Result contract

`AnomalyResult` (one per scored month):

| Group | Fields |
|---|---|
| What | `metric`, `metric_name`, `unit`, `period`, `period_start`, `period_end`, `filters` |
| Numbers | `observed_value`, `expected_value`, `deviation`, `deviation_percentage`, `lower_bound`, `upper_bound` |
| Judgement | `detector`, `score`, `threshold`, `severity`, `direction`, `is_anomaly`, `anomaly_start` |
| Why | `historical_window` (start, end, observations, window size), `details` (detector-specific statistics), `explanation` |
| Provenance | `source_tables`, `query_id`, `operation_id`, `execution_timestamp`, `calculation` |
| Caveats | `limitations` |

`AnomalyReport` wraps the results with `status` (`ok`, `no_data`, `insufficient_history`),
`method` (detector, description, transform, window, minimum history, threshold, severity policy,
full configuration), `skipped`, the prepared `series` and the full `provenance` (every SQL
statement with its bound parameters and lineage record).

Example `explanation` (real output, total MRR, rolling z-score):

> 2026-08: observed 5,645,109 vs expected 5,807,770 (-162,662, -2.8%). This month's
> month-over-month percentage change -0.76% vs a mean of +2.10% (std 0.83%) over 12 prior months
> (2025-08 to 2026-07). Score -3.46; flag threshold 3: significant, flagged, direction negative.
> Values outside 5,666,801 to 5,948,740 are flagged.

## 9. Measured results on the generated dataset

The following was produced by running the detectors with default settings (last 12 months to
2026-08-31, total level) on the seed-42 dataset. It is measured output, not a target.

| Detector | Flagged months (severity, score) |
|---|---|
| rolling z-score | revenue 2026-08 (significant, −3.69); MRR 2026-08 (significant, −3.46); tickets 2026-06 (extreme, +11.3) |
| IQR | revenue 2026-08 (−2.45) and 2026-03 (+1.81); MRR 2026-08 (−2.39); customers 2026-08 (−2.90) and 2026-07 (−2.01); tickets 2026-06 (extreme, +13.8) and 2026-08 (extreme, −5.0) |
| forecast residual | revenue 2026-08 (−3.64); MRR 2025-11 (+3.75) and 2026-08 (−3.55); tickets 2026-06 (extreme, +12.5) |

No month of Dashboards adoption was flagged. For the explicit group `country=Singapore,
segment=Enterprise`, every detector flags August 2026 MRR, revenue and customer count as
`extreme` and negative.

**Comparison with the injected events.** This comparison is done only in the separate evaluation
test `tests/integration/test_phase3_event_detection.py`. It is the only Phase 3 test that reads
the ground truth, and only to learn the event months:

| Event | Result |
|---|---|
| E1 August 2026 Singapore Enterprise churn wave | Detected by all three detectors in total revenue and MRR (negative), and as `extreme` in the Singapore Enterprise group |
| E2 June–July 2026 support-ticket spike | Detected by all three detectors: 2026-06 is `extreme` and positive, with `anomaly_start` 2026-06. July is not flagged: its change from June is small, and every July baseline already contains June |
| E3 inefficient campaign | Not in scope: a cross-sectional campaign comparison, not a monthly series (Phase 2 campaign analytics shows it) |
| E4, E6, E7 | Not in scope: rep-level, segment-level and customer-level patterns (Phase 2 analytics) |
| E5 AI Insights launch | Not detectable here: 6 monthly points are below the minimum history (`insufficient_history`) |

Other flags also occur: revenue 2026-03 (IQR), MRR 2025-11 (forecast residual), customers
2026-07 (IQR) and tickets 2026-08 (IQR). The tickets 2026-08 flag is the fall back from the spike,
the known effect of change-based transforms. The evaluation treats none of them as a confirmation
or an error: each is a statistically unusual month relative to its window, reported as it is.

Performance (median warm latency, full dataset): one detection 8–150 ms; a four-metric scan
about 0.25 s (`tests/integration/test_phase3_performance.py` enforces budgets).

## 10. Limitations

- The window is at most 12 months and often shorter at the start of the data, so scores are noisy.
  The t tail probability accounts for window size; a normal assumption is still an approximation.
- Change-based transforms compare movements. The month after an unusual month can itself score as
  unusual when the metric moves back.
- A baseline that already contains an anomaly absorbs part of it: the second month of a sustained
  shift is often not flagged.
- Seasonality is not modelled. With two years of data, a regular seasonal movement can appear
  unusual against the previous 12 months.
- An anomaly is a movement that is unusual relative to recent history. It does not establish a
  cause, and it is not by itself evidence of a business failure.

## 11. How Phase 4 will consume this

The planned `detect_anomalies` tool maps onto the service. Phase 4 must:

- use `report.anomalies` (ranked) and each result's `explanation`, `score`, `threshold` and
  `historical_window` as the evidence for "was this month unusual?";
- register `operation_id` / `query_id` as the evidence reference;
- keep `direction` statistical and add any business interpretation or cause only as separately
  evidenced reasoning (for example, with a Phase 2 revenue decomposition);
- treat `insufficient_history` / `no_data` as an explicit lack of evidence.

## 12. Configuration reference (`AnomalyConfig`)

| Field | Default | Meaning |
|---|---|---|
| `detector` | `rolling_zscore` | `rolling_zscore`, `iqr` or `forecast_residual` |
| `window` | 12 | Prior months forming the baseline (3–36) |
| `min_history` | 6 | Minimum prior observations / residuals to score a month (≤ window) |
| `transform` | metric default | `level`, `difference` or `pct_change` (ignored by `forecast_residual`) |
| `thresholds` | 2 / 3 / 4 | Standardised watch / significant / extreme cut-offs |
| `iqr_fences` | 1.5 / 3.0 | Tukey inner / outer fence multipliers |
| `expectation_model` | `drift` | Forecasting model behind `forecast_residual` |
| `flag_severity` | `significant` | Lowest severity counted as an anomaly |
