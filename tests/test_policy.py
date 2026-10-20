"""Unit tests for policy allocation (A08, T10, T11), calibration & models (A11, A12), metrics & stage breakdown (A13, A14, T12), and block bootstrap (A15, T13)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olist_delay.config import expand_model_candidates, load_config
from olist_delay.evaluation import (
    compute_bootstrap_ci,
    compute_classification_metrics,
    compute_ex_post_stage_breakdown,
)
from olist_delay.features import BASE_V2_FEATURES
from olist_delay.models import fit_model_with_chronological_calibration
from olist_delay.policy import (
    allocate_daily_alerts,
    assign_risk_tier,
    evaluate_cost_ratio_sensitivity,
    evaluate_topk_frontier_cohort_a,
    evaluate_topk_frontier_path_b,
    rank_orders_deterministically,
    summarize_daily_policy,
)

pytestmark = pytest.mark.unit


def test_t10_deterministic_tie_breaking_and_shuffle_invariance() -> None:
    """T10 & A08: Identical scores break ties deterministically by (+order_purchase_timestamp, +order_id, +sample_draw_id) and are shuffle-invariant."""
    df = pd.DataFrame(
        [
            {"order_id": "ord_C", "ts": "2018-06-01 10:00:00", "draw": 0, "score": 0.50, "late": 1},
            {"order_id": "ord_A", "ts": "2018-06-01 10:00:00", "draw": 1, "score": 0.50, "late": 0},
            {"order_id": "ord_A", "ts": "2018-06-01 10:00:00", "draw": 0, "score": 0.50, "late": 0},
            {"order_id": "ord_B", "ts": "2018-06-01 09:00:00", "draw": 0, "score": 0.50, "late": 1},
            {"order_id": "ord_D", "ts": "2018-06-01 12:00:00", "draw": 0, "score": 0.90, "late": 1},
            {"order_id": "ord_E", "ts": "2018-06-01 08:00:00", "draw": 0, "score": 0.10, "late": 0},
        ]
    )
    ranked_1 = rank_orders_deterministically(
        df["order_id"],
        df["score"],
        purchase_timestamps=df["ts"],
        sample_draw_ids=df["draw"],
    )
    # Expected order:
    # 1: ord_D (score 0.90)
    # 2: ord_B (score 0.50, ts 09:00)
    # 3: ord_A (score 0.50, ts 10:00, draw 0)
    # 4: ord_A (score 0.50, ts 10:00, draw 1)
    # 5: ord_C (score 0.50, ts 10:00, draw 0)
    # 6: ord_E (score 0.10)
    assert list(ranked_1["risk_rank"]) == [5, 4, 3, 2, 1, 6]

    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    ranked_2 = rank_orders_deterministically(
        df_shuffled["order_id"],
        df_shuffled["score"],
        purchase_timestamps=df_shuffled["ts"],
        sample_draw_ids=df_shuffled["draw"],
    )
    merged = pd.concat([df_shuffled[["order_id", "draw"]], ranked_2[["risk_rank"]]], axis=1).sort_values("risk_rank")
    assert list(zip(merged["order_id"], merged["draw"])) == [
        ("ord_D", 0),
        ("ord_B", 0),
        ("ord_A", 0),
        ("ord_A", 1),
        ("ord_C", 0),
        ("ord_E", 0),
    ]


def test_a08_six_order_10pct_quota_and_daily_eligibility() -> None:
    """A08: 6 orders at 10% quota must yield floor(0.6) = 0 alerts; same-day delivered and already-overdue orders are excluded from eligible pool."""
    order_ids = pd.Series([f"o{i}" for i in range(1, 7)])
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y_true = pd.Series([1, 0, 1, 0, 0, 0])

    # 1. Batch frontier floor check on 6 orders at k=0.10 -> 0 alerts
    pol_6 = evaluate_topk_frontier_cohort_a(
        order_ids=order_ids,
        y_true=y_true,
        scores=scores,
        budgets=[0.10],
        model_name="m_test",
        split_id="s_test",
    )
    assert int(pol_6.iloc[0]["alert_budget_k"]) == 0
    assert int(pol_6.iloc[0]["tp"]) == 0

    ranked_6 = rank_orders_deterministically(order_ids, scores)
    tiers_6 = assign_risk_tier(ranked_6["risk_percentile"], primary_budget=0.10, secondary_budget=0.20)
    assert int(np.sum(tiers_6 == "HIGH_PRIORITY_ALERT_TOP10")) == 0
    assert int(np.sum(tiers_6 == "MEDIUM_WATCHLIST_TOP20")) == 1  # floor(0.20 * 6) = 1

    # 2. Daily operational allocation:
    # Day 1 (2018-06-01): 6 eligible orders -> q_d = floor(0.6) = 0 alerts (no carryover to Day 2!)
    # Day 2 (2018-06-02): 12 orders placed, of which 1 is delivered same day (before 23:59:59.999999)
    # and 1 has estimated delivery on 2018-06-01 (already overdue before 2018-06-02).
    # Remaining eligible on Day 2 = 10 -> q_d = floor(0.10 * 10) = 1 alert.
    rows = []
    for i in range(6):
        rows.append(
            {
                "order_id": f"d1_{i}",
                "order_purchase_timestamp": f"2018-06-01 10:0{i}:00",
                "order_estimated_delivery_date": "2018-06-10 00:00:00",
                "order_delivered_customer_date": "2018-06-05 12:00:00",
                "in_cohort_a": True,
                "late": 1 if i == 0 else 0,
                "score": 0.90 - i * 0.05,
            }
        )
    for i in range(12):
        deliv = "2018-06-02 18:00:00" if i == 0 else "2018-06-12 12:00:00"
        est = "2018-06-01 00:00:00" if i == 1 else "2018-06-10 00:00:00"
        rows.append(
            {
                "order_id": f"d2_{i:02d}",
                "order_purchase_timestamp": f"2018-06-02 11:{i:02d}:00",
                "order_estimated_delivery_date": est,
                "order_delivered_customer_date": deliv,
                "in_cohort_a": True,
                "late": 1 if i in (0, 1, 2) else 0,
                "score": 0.99 - i * 0.02,
            }
        )
    df_daily = pd.DataFrame(rows)
    alloc = allocate_daily_alerts(df_daily, df_daily["score"], budget_k=0.10, label_col="late")
    summary = summarize_daily_policy(alloc, model_name="m_test", split_id="bt", budget_k=0.10).set_index("decision_date")

    assert int(summary.loc["2018-06-01", "eligible_orders_n"]) == 6
    assert int(summary.loc["2018-06-01", "daily_quota_q_d"]) == 0
    assert int(summary.loc["2018-06-01", "alerts_selected"]) == 0

    assert int(summary.loc["2018-06-02", "total_orders_placed"]) == 12
    assert int(summary.loc["2018-06-02", "excluded_already_delivered"]) == 1
    assert int(summary.loc["2018-06-02", "separated_already_overdue"]) == 1
    assert int(summary.loc["2018-06-02", "eligible_orders_n"]) == 10
    assert int(summary.loc["2018-06-02", "daily_quota_q_d"]) == 1
    assert int(summary.loc["2018-06-02", "alerts_selected"]) == 1
    # Highest-scoring eligible order on Day 2 is d2_02 (since d2_00 was delivered same day and d2_01 was already overdue)
    alerted_ids = alloc.loc[alloc["alert_selected"], "order_id"].tolist()
    assert alerted_ids == ["d2_02"]


def test_t11_path_a_vs_path_b_denominator_and_unlabeled_capacity_consumption() -> None:
    """T11: Path B top-K alerts include unlabeled/non-delivered orders consuming alert capacity without coercing them to 0."""
    df_b = pd.DataFrame(
        [
            {"order_id": "o1", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 1, "score": 0.95},
            {"order_id": "o2", "order_status": "shipped", "has_items": True, "in_cohort_a": False, "late": pd.NA, "score": 0.90},
            {"order_id": "o3", "order_status": "unavailable", "has_items": False, "in_cohort_a": False, "late": pd.NA, "score": 0.85},
            {"order_id": "o4", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 0, "score": 0.80},
            {"order_id": "o5", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 1, "score": 0.20},
            {"order_id": "o6", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 0, "score": 0.10},
            {"order_id": "o7", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 0, "score": 0.08},
            {"order_id": "o8", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 0, "score": 0.06},
            {"order_id": "o9", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 0, "score": 0.04},
            {"order_id": "o10", "order_status": "delivered", "has_items": True, "in_cohort_a": True, "late": 0, "score": 0.02},
        ]
    )
    res_b = evaluate_topk_frontier_path_b(
        df_path_b=df_b,
        scores=df_b["score"],
        budgets=[0.40],
        model_name="test_model",
        split_id="test_split",
        label_col="late",
    )
    row = res_b.iloc[0]
    assert row["path_b_total_orders"] == 10
    assert row["alert_total_path_b"] == 4
    assert row["alert_in_cohort_a_count"] == 2
    assert row["alert_unlabeled_non_delivered_count"] == 2
    assert row["alert_without_items_count"] == 1
    assert row["alert_shipped_undelivered_count"] == 1
    assert row["alert_canceled_or_unavailable_count"] == 1
    assert row["observed_tp_in_cohort_a"] == 1
    assert row["observed_fp_in_cohort_a"] == 1
    assert np.isclose(row["conditional_precision_among_observed"], 0.50)
    assert np.isclose(row["conditional_recall_on_cohort_a"], 0.50)
    assert np.isclose(row["lower_bound_precision_path_b"], 0.25)
    assert np.isclose(row["upper_bound_exception_rate_path_b"], 0.75)


def test_a11_chronological_calibration_maturity_and_degenerate_tail_fallback() -> None:
    """A11: fit_core must drop orders unmatured at T_core, and single-class tails must set UNAVAILABLE_DEGENERATE_TAIL and fallback to uncalibrated."""
    rng = np.random.default_rng(5126)
    n = 120
    X = pd.DataFrame({col: rng.normal(size=n) for col in BASE_V2_FEATURES})
    X["primary_payment_type"] = "credit_card"
    X["primary_category"] = "bed_bath_table"
    X["customer_state"] = "SP"
    X["customer_region"] = "Southeast"
    X["primary_seller_state"] = "SP"
    X["seller_region"] = "Southeast"
    X["route_state_pair"] = "SP->SP"

    purchase_ts = pd.date_range("2018-01-01", periods=n, freq="D")
    # Make 10 core rows mature AFTER T_core (purchase_ts[96]) so they must be dropped from fit_core
    label_avail = purchase_ts + pd.Timedelta(hours=12)
    label_avail_list = list(label_avail)
    for idx in range(70, 80):
        label_avail_list[idx] = purchase_ts[-1] + pd.Timedelta(days=5)
    label_avail_s = pd.Series(label_avail_list)

    # Core has both classes, but chronological tail (last 20% = 24 rows) has ONLY class 0 (single-class degenerate tail!)
    y = pd.Series([1 if i % 4 == 0 and i < 90 else 0 for i in range(n)])

    bundle = fit_model_with_chronological_calibration(
        model_name="logistic_regression",
        X_train=X,
        y_train=y,
        purchase_timestamps=pd.Series(purchase_ts),
        label_available_at=label_avail_s,
        params={"C": 1.0, "penalty": "l2"},
        random_seed=5126,
        calibration_tail_fraction=0.20,
        active_calibration="platt",
        feature_version="base_v2",
        min_tail_samples=10,
        min_tail_positives=5,
        refit_full_after_calibration=False,
    )
    assert bundle.calibration_status == "UNAVAILABLE_DEGENERATE_TAIL"
    assert bundle.active_calibration == "uncalibrated"
    assert bundle.core_dropped_unmatured_count == 10
    p_uncal = bundle.predict_proba_variant(X.head(10), method="uncalibrated")
    p_platt = bundle.predict_proba_variant(X.head(10), method="platt")
    np.testing.assert_allclose(p_uncal, p_platt)


def test_a12_candidate_grid_and_five_baselines(project_root: Path) -> None:
    """A12: experiment-v2.json expands to <= 24 candidates and all 5 baselines fit and predict cleanly."""
    cfg = load_config(project_root / "configs" / "experiment-v2.json", root=project_root)
    cands = expand_model_candidates(cfg)
    assert len(cands) == 24
    families = {c["model_family"] for c in cands}
    assert families == {"logistic_regression", "decision_tree"}

    rng = np.random.default_rng(5126)
    n = 150
    X = pd.DataFrame({col: rng.uniform(1, 20, size=n) for col in BASE_V2_FEATURES})
    X["primary_payment_type"] = "credit_card"
    X["primary_category"] = "bed_bath_table"
    X["customer_state"] = rng.choice(["SP", "RJ", "BA"], size=n)
    X["customer_region"] = rng.choice(["Southeast", "Northeast"], size=n)
    X["primary_seller_state"] = "SP"
    X["seller_region"] = "Southeast"
    X["route_state_pair"] = "SP->RJ"
    y = pd.Series((rng.uniform(0, 1, size=n) < 0.12).astype(int))
    ts = pd.Series(pd.date_range("2018-01-01", periods=n, freq="h"))

    for b_name in [
        "rule_random",
        "rule_short_promised_lead",
        "rule_seller_history",
        "rule_region_lead",
        "global_prior_baseline",
        "decision_tree",
    ]:
        b_bundle = fit_model_with_chronological_calibration(
            model_name=b_name,
            X_train=X,
            y_train=y,
            purchase_timestamps=ts,
            params={"max_depth": 3, "min_samples_leaf": 20} if b_name == "decision_tree" else {},
            random_seed=5126,
        )
        preds = b_bundle.predict_proba(X.head(20))
        assert len(preds) == 20
        assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_a13_a14_degenerate_metrics_and_ex_post_stage_alignment() -> None:
    """A13 & A14: Single-class metrics return DEGENERATE_SINGLE_CLASS without crashing; ex-post stage breakdown uses identical cohort origin."""
    order_ids = pd.Series(["o1", "o2", "o3", "o4", "o5"])
    y_single = np.zeros(5, dtype=int)
    p_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    m_deg = compute_classification_metrics(
        order_ids=order_ids,
        y_true=y_single,
        y_prob=p_scores,
        model_name="m_deg",
        calibration_method="uncalibrated",
        split_id="s_deg",
        label_version="calendar_day",
    )
    assert m_deg["metrics_status"] == "DEGENERATE_SINGLE_CLASS"
    assert m_deg["pr_auc"] is None
    assert m_deg["roc_auc"] is None
    assert m_deg["brier_score"] is not None

    # A14: Ex-post stage breakdown origin for seller_exceeded_shipping_limit
    df_stage = pd.DataFrame(
        [
            {
                "order_id": "s1",
                "in_cohort_a": True,
                "late": 1,
                "order_purchase_timestamp": "2018-01-01 10:00:00",
                "order_approved_at": "2018-01-01 12:00:00",
                "order_delivered_carrier_date": "2018-01-06 12:00:00",
                "order_delivered_customer_date": "2018-01-15 12:00:00",
                "order_estimated_delivery_date": "2018-01-10 00:00:00",
                "earliest_shipping_limit_date": "2018-01-05 00:00:00",
            },
            {
                "order_id": "s2",
                "in_cohort_a": True,
                "late": 0,
                "order_purchase_timestamp": "2018-01-01 10:00:00",
                "order_approved_at": "2018-01-01 11:00:00",
                "order_delivered_carrier_date": pd.NA,  # Missing carrier date -> must not be counted in valid shipping limit denominator
                "order_delivered_customer_date": "2018-01-08 12:00:00",
                "order_estimated_delivery_date": "2018-01-10 00:00:00",
                "earliest_shipping_limit_date": "2018-01-05 00:00:00",
            },
            {
                "order_id": "s3",
                "in_cohort_a": True,
                "late": 0,
                "order_purchase_timestamp": "2018-01-01 10:00:00",
                "order_approved_at": "2018-01-01 11:00:00",
                "order_delivered_carrier_date": "2018-01-03 12:00:00",
                "order_delivered_customer_date": "2018-01-08 12:00:00",
                "order_estimated_delivery_date": "2018-01-10 00:00:00",
                "earliest_shipping_limit_date": "2018-01-05 00:00:00",
            },
        ]
    )
    stage_res = compute_ex_post_stage_breakdown(df_stage, label_col="late").set_index("label_value")
    assert int(stage_res.loc[0, "shipping_limit_valid_orders"]) == 1
    assert int(stage_res.loc[0, "seller_exceeded_shipping_limit_count"]) == 0
    assert int(stage_res.loc[1, "shipping_limit_valid_orders"]) == 1
    assert int(stage_res.loc[1, "seller_exceeded_shipping_limit_count"]) == 1


def test_t12_t13_a15_metrics_and_weekly_block_bootstrap() -> None:
    """T12, T13 & A15: Classification metrics, cost sensitivity, and paired weekly block bootstrap with sample_draw_id tie-breaking."""
    rng = np.random.default_rng(5126)
    n = 240
    order_ids = pd.Series([f"ord_{i:04d}" for i in range(n)])
    purchase_ts = pd.Series(pd.date_range("2018-06-01 08:00:00", periods=n, freq="6h"))
    y_true = (rng.uniform(0, 1, size=n) < 0.15).astype(int)
    p_model = np.clip(y_true * 0.4 + rng.uniform(0.01, 0.3, size=n), 0.001, 0.999)
    p_base = np.clip(rng.uniform(0.05, 0.25, size=n), 0.001, 0.999)

    m = compute_classification_metrics(
        order_ids=order_ids,
        y_true=y_true,
        y_prob=p_model,
        model_name="m1",
        calibration_method="platt",
        split_id="s1",
        label_version="calendar_day",
        purchase_timestamps=purchase_ts,
    )
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["brier_score"] <= 1.0
    assert "daily_top10_precision" in m

    cost_df = evaluate_cost_ratio_sensitivity(
        y_true=y_true,
        scores_calibrated=p_model,
        scores_uncalibrated=np.clip(p_model * 1.5, 0.001, 0.999),
        cost_ratios=[0.10, 0.20],
        model_name="m1",
        split_id="s1",
        scores_random_baseline=p_base,
    )
    assert set(cost_df["probability_variant"]) == {"calibrated", "uncalibrated", "treat_all", "random_top10"}

    ci_block = compute_bootstrap_ci(
        y_true=y_true,
        y_prob_model=p_model,
        y_prob_baseline=p_base,
        n_bootstrap=50,
        random_seed=5126,
        order_ids=order_ids,
        purchase_timestamps=purchase_ts,
        method="weekly_block_bootstrap",
    )
    assert ci_block["method"] == "weekly_block_bootstrap"
    assert ci_block["n_blocks"] >= 4
    assert (
        ci_block["metrics"]["pr_auc"]["ci95_low"]
        <= ci_block["metrics"]["pr_auc"]["mean"]
        <= ci_block["metrics"]["pr_auc"]["ci95_high"]
    )
    assert "daily_top10_recall_diff" in ci_block["paired_difference_vs_baseline"]
