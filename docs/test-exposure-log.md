# Test Exposure Log for the Historical Backtest Window

Window: purchases from 2018-05-26 18:18:03 to 2018-08-29 15:00:37 (`historical_backtest`; 19,661 orders, 19,294 in Cohort A, 674 late).

## 1. Timeline (from run logs and file modification times)

| When (2026, UTC+8) | Event | Exposure of the window |
| :--- | :--- | :--- |
| 19 Oct, 22:54–23:04 | Legacy v1 diagnostic run `review-20261019-backtest-v1` started and was interrupted during validation fold 4 (`artifacts/review-2026-10-19/run-status.json`) | **None from models.** The run never reached the backtest and produced no backtest metrics. |
| 19 Oct (audit) | Project audit computed descriptive statistics of the window: order counts and the 3.49% late rate | Descriptive only |
| 20 Oct, 00:37 | `configs/research-v2.json`, `configs/experiment-v2.json` and `configs/smoke-v2.json` frozen (candidate grid, selection rule, policy, cost scenarios) | — |
| 20 Oct, 02:30 and 02:37 | Smoke runs `smoke-v2-003` and `smoke-v2-004` (reduced grid, 100 bootstrap replicates) scored the window | **Model scores seen** for the reduced grid |
| 20 Oct, 02:35–02:47 | `src/olist_delay/evaluation.py`, `pipeline_runner.py` and `validation.py` edited | Possible influence on implementation, not on configuration |
| 20 Oct, 02:47 | Formal run `experiment-v2-001` started; selection used only the four validation folds | — |
| 20 Oct, 03:33 | `smoke-v2-005`, run on the final code, reproduced `smoke-v2-004` predictions and daily policy bit-for-bit | The edits after the smoke runs did not change any smoke prediction |

## 2. Classification

The window is a **diagnostic holdout**, not a blind test:
- The configuration, candidate grid and selection rule were fixed before any model scored the window.
- Smoke runs did score it before the formal run, and code was edited in between.
- Reports must describe it this way.

## 3. Retraction

An earlier version of this log listed "legacy v1 backtest metrics" (PR-AUC 0.1539, ROC-AUC 0.7303 and recall 0.3436), and the v2.0.0 report cited a legacy PR-AUC of 0.0566. No artifact contains these values: the v1 run never scored the window. Both statements are withdrawn. <!-- check:ignore: retracted values quoted on purpose -->

## 4. Second evaluation period (late 2017) and the v3 candidates

| When (2026, UTC+8) | Event | Exposure |
| :--- | :--- | :--- |
| 20 Oct, morning | Report layer built on `experiment-v2-001`. Descriptive monthly late rates for 2017 were plotted (Figure 2), but no model was scored on any 2017 window | Descriptive only |
| 20 Oct, up to 12:40:00 | Last edit to `src/olist_delay/`: code fixes, censoring-aware label rule, v3 route features and rules | — |
| 20 Oct, 12:40:26 | `configs/research-v3.json`, `experiment-v3.json`, `experiment-v3-late2017.json` and `smoke-v3.json` written and frozen; the script refuses to overwrite them | — |
| 20 Oct, 12:43:55 | `smoke-v3-001` (reduced grid) scored the **2018** window to check the v3 code path | 2018 window only |
| 20 Oct, 12:46:47 | `experiment-v3-late2017-001` started: the first time any model was scored on 26 Oct–31 Dec 2017 | — |
| 20 Oct, about 12:47 | `experiment-v2-002` (v2 configuration, fixed code), `experiment-v3-2018-001` and `smoke-v2-006` started | — |

Classification:
- **Late-2017 backtest (`experiment-v3-late2017-001`):** a pre-registered, out-of-sample test of the v2 method, the three v3 rules, censoring-aware labels and the shipping-limit signal (which was itself found post hoc in 2018).
- **`experiment-v3-2018-001`:** post hoc, because the 2018 window had been inspected.
- **`experiment-v2-002`:** re-runs the frozen v2 configuration with the fixed code, and replaces `experiment-v2-001` as the formal run.

## 5. Post hoc analyses on the window

Two analyses were designed after the formal backtest results were known, and they are labelled post hoc in every document:
- the shipping-limit (Milestone 2) signal;
- the paired bootstrap comparison between `rule_short_promised_lead` and the selected rule.

They need confirmation on a different period.
