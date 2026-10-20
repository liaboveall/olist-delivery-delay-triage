# Research Contract: Olist Purchase-Time Delivery Delay Warning (v2)

> Erratum (2026-10-20): this contract was drafted one minute before the frozen configuration `configs/experiment-v2.json`, which is the executed specification. Two statements were aligned with it: the tree grid uses `min_samples_leaf in [150, 300]` (the draft said `[50, 200]`), and the selection metric averages fold-level recall (Section 4.3). The configuration also asks for 2,000 bootstrap replicates; the formal run used 400 (see `docs/decision-log.md`, D11). <!-- check:ignore: draft values quoted on purpose -->

## 1. Operational Decision Scenario & Timing Contract

- **Business Problem**: In a multi-merchant e-commerce marketplace (Olist Brazil), customer dissatisfaction is strongly driven by orders delivered after the promised estimated delivery date (`order_estimated_delivery_date`). However, proactive operational interventions—such as expediting seller dispatch, prioritizing carrier linehaul handovers, or sending proactive customer expectation updates—are capacity-constrained.
- **Decision Timing (`decision_at`)**: Daily end-of-day batch execution at **`23:59:59.999999` local source time** on each purchase calendar day `d` (`order_purchase_timestamp.dt.floor("D") == d`).
- **Operational Action**: Flag a bounded daily queue of high-risk orders placed on day `d` for next-morning operations review (`1 day before` morning fulfillment triage).
- **Daily Capacity Constraint**:
  - Let $N_{\text{eligible}, d}$ be the number of eligible orders placed on calendar day $d$.
  - Daily alert quota:
    $$q_d = \lfloor k \cdot N_{\text{eligible}, d} \rfloor$$
  - Primary budget: $k = 0.10$ (10% daily capacity). Sensitivity budgets: $k \in \{0.05, 0.10, 0.15, 0.20\}$.
  - On low-volume days where $k \cdot N_{\text{eligible}, d} < 1$, $q_d = 0$ is explicitly permitted.
  - **No cross-day quota carryover**: Unused alert capacity on day $d$ cannot be transferred to day $d' \neq d$, and multi-month global top-$K$ pooling is strictly forbidden as the primary operational policy.
- **Operational Eligibility Filter at `decision_at`**:
  - Orders placed on day $d$ are eligible by default (`ELIGIBLE_NEW_ORDER`), except:
    1. `DELIVERED_BEFORE_DECISION`: Orders with a known `order_delivered_customer_date <= decision_at` (already fulfilled prior to the batch run; excluded from alert allocation).
    2. `ALREADY_OVERDUE_BEFORE_DECISION`: Orders whose promised date is already strictly prior to day $d$ (`order_estimated_delivery_date.dt.floor("D") < d`; routed to a separate overdue-exception workflow rather than predictive early-warning capacity).
- **Deterministic Tie-Breaking Contract**:
  - Within each calendar day $d$ (and within each bootstrap draw instance `sample_draw_id`), eligible orders are ranked by:
    1. `-risk_score` (descending predicted delay probability or rule score)
    2. `+order_purchase_timestamp` (ascending purchase timestamp)
    3. `+order_id` (ascending lexicographical order identifier)
    4. `+sample_draw_id` (ascending bootstrap draw identifier, default `0` in point evaluation)

---

## 2. Target Variable & Cohort Accounting Contract

### 2.1 Primary and Audit Labels
- **Primary Label (`late_calendar_day`, aliased as `late` under default config)**:
  - Defined for orders in **Cohort A** (`order_status == "delivered"`, `has_items == True`, non-null `order_delivered_customer_date`, non-null `order_estimated_delivery_date`):
    $$\text{late\_calendar\_day} = \mathbb{I}\!\left(\lfloor \text{order\_delivered\_customer\_date} \rfloor_{\text{day}} > \lfloor \text{order\_estimated\_delivery\_date} \rfloor_{\text{day}}\right)$$
  - Stored as nullable integer `Int8` (`1`, `0`, or `pd.NA`).
- **Audit Label (`late_timestamp`)**:
  - Defined on the same Cohort A orders using exact timestamp inequality:
    $$\text{late\_timestamp} = \mathbb{I}\!\left(\text{order\_delivered\_customer\_date} > \text{order\_estimated\_delivery\_date}\right)$$
  - Because `order_estimated_delivery_date` is stored at `00:00:00` midnight in Olist, orders delivered on the promised calendar date after `00:00:00` (`same_day_after_midnight == True`) have `late_calendar_day = 0` and `late_timestamp = 1`.
  - Across all `96,470` Cohort A orders: `7,826` (`late_timestamp`) $-$ `6,534` (`late_calendar_day`) $=$ `1,292` (`same_day_after_midnight`).
  - Across the `2018-05-26 18:18:03` to `2018-08-30 00:00:00` Cohort A historical backtest window (`19,294` delivered orders, `19,661` Path B orders): `1,021` (`late_timestamp`) $-$ `674` (`late_calendar_day`) $=$ `347` (`same_day_after_midnight`).

### 2.2 Unknown Outcome Accounting (Path A vs. Path B)
- **Strict Prohibition**: Orders without an observed customer delivery timestamp (`2,971` orders out of `99,441` total orders) **must never be coerced to `0` (on-time)**. Their `late`, `late_calendar_day`, and `late_timestamp` values must remain `pd.NA`.
- **Dual Evaluation Denominators**:
  - **Path A (Cohort A Delivered Evaluable, $N = 96,470$)**: Supervised training, validation model selection, and primary classification/capture evaluation on orders with observed delivery outcomes.
  - **Path B (Full Operational Stream, $N = 99,441$)**: Operational capacity accounting across all incoming purchase-time orders, explicitly tracking how many daily alert slots are consumed by:
    - Cohort A delivered orders (`in_cohort_a == True`)
    - Non-delivered / unknown-outcome orders (`2,196` orders with items in `shipped`, `canceled`, `invoiced`, `processing`, or missing delivery timestamp)
    - Itemless orders (`775` orders without `olist_order_items_dataset` rows, scored via explicit `fallback_no_items` prior)

---

## 3. Feature Availability & Dual-Clock Entity History Contract

### 3.1 Feature Version Registry
1. **`base_v2` (Primary Strict Purchase-Time Feature Set)**:
   - Order-time calendar & promise features: `purchase_hour`, `purchase_day_of_week`, `purchase_month`, `purchase_is_weekend`, `estimated_lead_calendar_days`, `estimated_lead_Exact_days`, `promised_Slack_days`, `min_shipping_limit_days`, `max_shipping_limit_days`.
   - Item & commercial structure (`olist_order_items_dataset` + `olist_products_dataset`): `item_count`, `distinct_seller_count`, `distinct_product_count`, `distinct_category_count`, `is_multi_item`, `is_multi_seller`, `total_price`, `mean_item_price`, `max_item_price`, `total_freight`, `max_freight`, `freight_ratio`, `order_total_value`, `total_weight_kg`, `max_weight_kg`, `total_volume_cm3`, `max_volume_cm3`, `weight_missing`, `volume_missing`, `primary_category`.
   - Geography & route features (`olist_customers_dataset`, `olist_sellers_dataset`, cleaned `olist_geolocation_dataset`): `customer_state`, `customer_region`, `primary_seller_state`, `primary_seller_region`, `route_state_pair`, `route_region_pair`, `same_state`, `same_region`, `any_cross_state_item`, `haversine_distance_km`, `distance_km_max`, `primary_distance_km`, `customer_geo_fallback`, `seller_geo_fallback`, `geo_missing`.
   - Dual-clock entity histories: seller (multi-seller bridge aggregated via `mean`), route (`route_state_pair`), customer state (`customer_state`), product category (`primary_category`), and repeat customer (`customer_unique_id`).
2. **`extended_purchase_assumptions_v2` (Ablation Feature Set)**:
   - Extends `base_v2` with `payment_value_sum`, `payment_installments_max`, `payment_sequential_count`, `has_boleto`, `has_credit_card`, `has_voucher`, and `primary_payment_type`.
   - Isolated as an explicit ablation because `olist_order_payments_dataset` lacks independent payment-completion timestamps.

### 3.2 Forbidden Columns (Zero-Leakage Guard)
- The following post-purchase, fulfillment, review, and target columns are unconditionally blocked from feature matrices:
  - `order_status`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `label_available_at`
  - `review_id`, `review_score`, `review_score_mean`, `review_score_min`, `review_count`, `review_comment_title`, `review_comment_message`, `review_creation_date`, `review_answer_timestamp`
  - `late`, `late_calendar_day`, `late_timestamp`, `same_day_after_midnight`, `delay_calendar_days`, `delay_days_continuous`, `in_cohort_a`, `is_label_observable`, `label_reason_code`

### 3.3 Dual-Clock History & Multi-Seller Bridge Rules
- **Prior Purchase Load Clock**:
  $$\text{past.order\_purchase\_timestamp} < \text{current.order\_purchase\_timestamp} \quad \land \quad \text{past.order\_id} \neq \text{current.order\_id}$$
- **Matured Outcome Clock**:
  $$\text{past.order\_purchase\_timestamp} < \text{current.order\_purchase\_timestamp} \quad \land \quad \text{past.label\_available\_at} < \text{current.order\_purchase\_timestamp} \quad \land \quad \text{past.order\_id} \neq \text{current.order\_id}$$
- **Timestamp Inversion Guard**: Any historical event with `label_available_at < order_purchase_timestamp` is flagged in `data_quality_exceptions.csv` (`DELIVERY_BEFORE_PURCHASE_TIMESTAMP`) and excluded from matured outcome history.
- **Multi-Seller Order Aggregation**:
  - Deduplicate `olist_order_items_dataset` to unique `(order_id, seller_id)` pairs.
  - Compute seller history per `(order_id, seller_id)` at `current.order_purchase_timestamp`, then aggregate across sellers within each `order_id` using `mean` (`multi_seller_aggregation = "mean"`).
- **Bayesian Laplace Smoothing**:
  - Historical late rate smoothed with $\alpha = 5.0$ (`smoothing_alpha` in config) toward the expanding-window matured prior (`default_prior` fallback on empty history).

---

## 4. Chronological Splits, Label Maturity, Calibration & Selection Contract

### 4.1 Rolling Validation Folds & Locked Historical Backtest
| Split ID | Split Type | Purchase Window (`[start, end)`) | Train Cutoff (`train_at`) | Maturity Cutoff (`selection_at` / `freeze_at`) |
| :--- | :--- | :--- | :--- | :--- |
| `val_2018_01` | Rolling Validation | `2018-01-01` to `2018-02-01` | `2018-01-01 00:00:00` | `2018-05-26 18:18:03` |
| `val_2018_02` | Rolling Validation | `2018-02-01` to `2018-03-01` | `2018-02-01 00:00:00` | `2018-05-26 18:18:03` |
| `val_2018_03` | Rolling Validation | `2018-03-01` to `2018-04-01` | `2018-03-01 00:00:00` | `2018-05-26 18:18:03` |
| `val_2018_04` | Rolling Validation | `2018-04-01` to `2018-05-01` | `2018-04-01 00:00:00` | `2018-05-26 18:18:03` |
| `historical_backtest` | Locked Backtest | `2018-05-26 18:18:03` to `2018-08-30 00:00:00` | `2018-05-26 18:18:03` | `2018-05-26 18:18:03` |

- **Training Maturity Rule**: An order is eligible for model fitting at cutoff $T_{\text{fit}}$ if and only if:
  $$\text{order\_purchase\_timestamp} < T_{\text{fit}} \quad \land \quad \text{label\_available\_at} \le T_{\text{fit}}$$
  Orders purchased before $T_{\text{fit}}$ but delivered after $T_{\text{fit}}$ are dropped as in-transit (`train_in_transit_dropped_mask`).
- **Validation Maturity Selection Bias Disclosure**:
  - At `selection_at = 2018-05-26 18:18:03`, `val_2018_04` has `104` Cohort A orders that were delivered after `2018-05-26 18:18:03`, of which `89` (`85.6%`) were late. Both the strict `selection_at`-matured validation metrics and the full post-maturity validation sensitivity metrics must be reported side by side.

### 4.2 Time-Consistent Chronological Calibration
- Within each training window ending at $T_{\text{fit}}$, matured training orders are ordered chronologically and split into:
  - **Core Fit Window** (earliest 80%, cutoff $T_{\text{core}}$): Only orders with `order_purchase_timestamp < T_core` AND `label_available_at <= T_core` are used to fit the base estimator (`core_model`). Orders purchased before $T_{\text{core}}$ that mature in $(T_{\text{core}}, T_{\text{fit}}]$ are excluded from the core fit to prevent cross-boundary maturity leakage.
  - **Chronological Calibration Tail** (latest 20%, purchase in $[T_{\text{core}}, T_{\text{fit}})$ and `label_available_at <= T_fit`): Scored out-of-time by `core_model` to fit `platt` (sigmoid) and `isotonic` calibrators.
  - **Model-Calibrator Alignment**: The `CalibratedModelBundle` retains the exact `core_model` against which the calibrator was fitted, ensuring calibrated probabilities correspond strictly to the calibrated score distribution. Calibration methods that fail or face degenerate single-class tails are explicitly marked `UNAVAILABLE` (never silently aliased to `uncalibrated`).

### 4.3 Baseline Rules, Finite Model Grid & Selection Protocol
- **Mandatory Baseline Rules** (evaluated under the exact same daily capacity policy):
  1. `rule_random`: Seeded uniform random score (`seed = 5126`).
  2. `rule_short_promised_lead`: Higher risk for shorter `promised_lead_days` ($-\text{promised\_lead\_days}$).
  3. `rule_seller_history`: Smoothed seller historical late rate (`seller_hist_late_rate_smoothed`), falling back to route/global prior on cold start.
  4. `rule_region_lead`: Regional promised-lead tightness relative to historical route delivery duration (`route_hist_late_rate_smoothed` combined with relative lead gap).
  5. `global_prior_baseline`: Expanding-window global matured delay rate reference.
- **Finite Interpretable Model Grid ($\le 24$ total candidates)**:
  - `logistic_regression`: `C in [0.1, 1.0]`, `class_weight in [None, "balanced"]`, `feature_version in ["base_v2", "extended_purchase_assumptions_v2"]` ($2 \times 2 \times 2 = 8$ candidates).
  - `decision_tree`: `max_depth in [3, 5]`, `min_samples_leaf in [150, 300]`, `class_weight in [None, "balanced"]`, `feature_version in ["base_v2", "extended_purchase_assumptions_v2"]` ($2 \times 2 \times 2 \times 2 = 16$ candidates).
- **Selection Decision Rule**:
  1. Compute the daily 10%-budget delay capture rate on each validation fold and average it across the four folds (the run records this as `val_mean_daily_top10_recall`; the configuration names the metric `pooled_daily_10pct_capture_on_val_2018_01_to_04`, but the implementation averages fold-level recall rather than pooling orders).
  2. Identify the best baseline rule ($R^*$) and the best ML candidate ($M^*$).
  3. If $\text{Capture}(M^*) - \text{Capture}(R^*) < 0.005$, record that ML does not exceed the minimum operational gain threshold (`0.5` percentage points) and evaluate both the selected ML model and the winning heuristic rule transparently in the backtest and business recommendation.
  4. Among ML candidates within `0.005` of $M^*$ on primary capture, prefer simpler models (`base_v2` over `extended_purchase_assumptions_v2`, `logistic_regression` over `decision_tree`, smaller complexity/depth) and better probability calibration (`brier_score` and `calibration_slope_distance_from_1`).

---

## 5. Addendum: v3 contract (frozen 2026-10-20 12:40)

- **Configurations:** `configs/research-v3.json`, `configs/experiment-v3.json` (2018 splits; post hoc), `configs/experiment-v3-late2017.json` (pre-registered second period) and `configs/smoke-v3.json`.
- **Label maturity:** `label_known_rule = delivered_or_overdue`. For a Cohort A order whose primary label is late, `label_known_at = min(delivery, promised day + 1)`; otherwise it is the delivery timestamp. Training and selection maturity use `label_known_at`.
- **Features (`base_v3`):** `base_v2` plus the following, all strictly causal at purchase time:
  - `route_recent_60d_late_rate_ca`: the censoring-aware 60-day route late rate;
  - `route_recent_60d_mean_delivery_days`: the delivery-clock 60-day route mean delivery time;
  - `promise_slack_days_60d`: promised lead days minus `route_recent_60d_mean_delivery_days`.
- **Rules:** `rule_route_recent_lead`, `rule_route_recent_ca_lead` and `rule_promise_slack`, added to the five v2 rules.
- **Model grid:** logistic regression and decision tree as in v2, over `base_v2` and `base_v3`, giving 24 fitted models.
- **Selection:** mean daily top-10% recall over the validation folds. A model needs an absolute gain of at least 0.005 over the best rule.
- **Second period:** validation folds `val_2017_06` to `val_2017_09`; selection and training cutoff 2017-10-26 00:00:00; backtest purchases from 2017-10-26 to 2018-01-01 (exclusive).
