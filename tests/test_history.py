"""Unit tests for T05 (dual-clock historical feature correctness) and T06 (future event mutation invariance)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist_delay.history import attach_event_clock_history_features
from olist_delay.validation import _run_causal_perturbation_probe

pytestmark = pytest.mark.unit



def test_t05_dual_clock_entity_history_hand_calculated() -> None:
    """T05: Verify current-order exclusion, exact-second exclusion, in-flight vs matured counts, and Bayesian smoothing."""
    # All orders share seller 'S1' except Q_cold_start
    # Default prior = 0.10, smoothing_m = 10.0 -> m * prior = 1.0
    orders = pd.DataFrame(
        [
            # Event A: Purchased Jan 1, delivered Jan 5 12:00:00, late=1 (delay=3.0 days)
            {
                "order_id": "A",
                "customer_unique_id": "C1",
                "primary_seller_id": "S1",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "order_purchase_timestamp": "2018-01-01 10:00:00",
                "label_available_at": "2018-01-05 12:00:00",
                "late": 1,
                "delay_calendar_days": 3.0,
            },
            # Event B: Purchased Jan 2, delivered Jan 10 12:00:00, late=0 (delay=-2.0 days)
            {
                "order_id": "B",
                "customer_unique_id": "C2",
                "primary_seller_id": "S1",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "order_purchase_timestamp": "2018-01-02 10:00:00",
                "label_available_at": "2018-01-10 12:00:00",
                "late": 0,
                "delay_calendar_days": -2.0,
            },
            # Query Q_exact_sec: Purchased at exact second of Event A delivery (2018-01-05 12:00:00)
            # Strict inequality `label_available_at < prediction_at` means Event A is NOT yet matured!
            # Prior purchases strictly before 12:00:00 = 2 (A and B), completed strictly before 12:00:00 = 0
            {
                "order_id": "Q_exact_sec",
                "customer_unique_id": "C1",
                "primary_seller_id": "S1",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "order_purchase_timestamp": "2018-01-05 12:00:00",
                "label_available_at": "2018-01-12 00:00:00",
                "late": 0,
                "delay_calendar_days": -1.0,
            },
            # Query Q_after_A: Purchased 1 second AFTER Event A delivery (2018-01-05 12:00:01)
            # Prior purchases strictly before 12:00:01 = 3 (A, B, Q_exact_sec), completed = 1 (A, late=1)
            # Smoothed rate = (1 + 10 * 0.10) / (1 + 10) = 2.0 / 11.0
            {
                "order_id": "Q_after_A",
                "customer_unique_id": "C3",
                "primary_seller_id": "S1",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "order_purchase_timestamp": "2018-01-05 12:00:01",
                "label_available_at": "2018-01-15 00:00:00",
                "late": 0,
                "delay_calendar_days": -1.0,
            },
            # Query Q_cold_start: New seller 'S_NEW' -> prior_purchase=0, completed=0, rate=prior (0.10)
            {
                "order_id": "Q_cold_start",
                "customer_unique_id": "C4",
                "primary_seller_id": "S_NEW",
                "route_state_pair": "SP->MG",
                "customer_state": "MG",
                "primary_category": "cat2",
                "order_purchase_timestamp": "2018-01-06 08:00:00",
                "label_available_at": "2018-01-16 00:00:00",
                "late": 0,
                "delay_calendar_days": 0.0,
            },
        ]
    )

    res = attach_event_clock_history_features(
        target_orders=orders,
        history_events=orders,
        label_col="late",
        smoothing_m=10.0,
        default_prior=0.10,
    ).set_index("order_id")

    # Verify Q_exact_sec (exact timestamp boundary excluded from completed, counted in prior purchases)
    assert res.loc["Q_exact_sec", "seller_prior_purchase_count"] == 2.0
    assert res.loc["Q_exact_sec", "seller_completed_count"] == 0.0
    assert res.loc["Q_exact_sec", "seller_is_cold_start"] == 1
    assert np.isclose(res.loc["Q_exact_sec", "seller_hist_late_rate_smoothed"], 0.10)
    assert res.loc["Q_exact_sec", "customer_prior_order_count"] == 1.0

    # Verify Q_after_A (1 matured late order A, 3 prior purchases A, B, Q_exact_sec)
    assert res.loc["Q_after_A", "seller_prior_purchase_count"] == 3.0
    assert res.loc["Q_after_A", "seller_completed_count"] == 1.0
    assert np.isclose(res.loc["Q_after_A", "seller_hist_late_rate_smoothed"], 2.0 / 11.0)
    assert np.isclose(res.loc["Q_after_A", "seller_hist_mean_delay_days"], 3.0)

    # Verify Q_cold_start
    assert res.loc["Q_cold_start", "seller_prior_purchase_count"] == 0.0
    assert res.loc["Q_cold_start", "seller_completed_count"] == 0.0
    assert res.loc["Q_cold_start", "seller_is_cold_start"] == 1
    assert np.isclose(res.loc["Q_cold_start", "seller_hist_late_rate_smoothed"], 0.10)


def test_t06_future_data_mutation_invariance() -> None:
    """T06: Mutating future delivery outcomes or in-transit outcomes must not alter earlier purchase-time features."""
    probe = _run_causal_perturbation_probe()
    assert probe["passed"] is True
    assert probe["max_abs_feature_diff"] == 0.0
    assert probe["checked_history_columns"] > 10


def test_a05_self_event_and_inverted_completion_timestamp_excluded() -> None:
    """A05: An order with corrupted/inverted label_available_at < order_purchase_timestamp must never count itself."""
    invalid_self_event = pd.DataFrame(
        {
            "order_id": ["o1"],
            "primary_seller_id": ["s1"],
            "route_state_pair": ["SP->RJ"],
            "customer_state": ["RJ"],
            "primary_category": ["cat"],
            "customer_unique_id": ["u1"],
            "order_purchase_timestamp": pd.to_datetime(["2018-01-10 00:00:00"]),
            "label_available_at": pd.to_datetime(["2018-01-05 00:00:00"]),
            "late": [1],
            "delay_calendar_days": [3.0],
        }
    )
    hist_df = attach_event_clock_history_features(invalid_self_event)
    assert float(hist_df.loc[0, "seller_completed_count"]) == 0.0
    assert float(hist_df.loc[0, "seller_prior_purchase_count"]) == 0.0
    assert float(hist_df.loc[0, "route_completed_count"]) == 0.0


def test_a05_multi_seller_bridge_mean_aggregation() -> None:
    """A05: Multi-seller orders aggregate deduplicated (order_id, seller_id) history via mean."""
    orders = pd.DataFrame(
        [
            # Seller S1 completed 1 late order before Jan 10
            {
                "order_id": "O_s1",
                "primary_seller_id": "S1",
                "seller_ids_pipe": "S1",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "customer_unique_id": "C1",
                "order_purchase_timestamp": "2018-01-01 10:00:00",
                "label_available_at": "2018-01-05 10:00:00",
                "late": 1,
                "delay_calendar_days": 4.0,
            },
            # Seller S2 completed 3 on-time orders before Jan 10
            {
                "order_id": "O_s2_a",
                "primary_seller_id": "S2",
                "seller_ids_pipe": "S2",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "customer_unique_id": "C2",
                "order_purchase_timestamp": "2018-01-02 10:00:00",
                "label_available_at": "2018-01-06 10:00:00",
                "late": 0,
                "delay_calendar_days": -2.0,
            },
            {
                "order_id": "O_s2_b",
                "primary_seller_id": "S2",
                "seller_ids_pipe": "S2",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "customer_unique_id": "C3",
                "order_purchase_timestamp": "2018-01-03 10:00:00",
                "label_available_at": "2018-01-07 10:00:00",
                "late": 0,
                "delay_calendar_days": -1.0,
            },
            # Multi-seller query order with both S1 (1 completed, 1 late) and S2 (2 completed, 0 late)
            # Notice duplicate S1|S1|S2 in pipe is deduplicated to S1 and S2 -> mean completed_count = (1 + 2) / 2 = 1.5
            {
                "order_id": "O_multi",
                "primary_seller_id": "S1",
                "seller_ids_pipe": "S1|S1|S2",
                "route_state_pair": "SP->RJ",
                "customer_state": "RJ",
                "primary_category": "cat1",
                "customer_unique_id": "C4",
                "order_purchase_timestamp": "2018-01-10 10:00:00",
                "label_available_at": "2018-01-20 10:00:00",
                "late": 0,
                "delay_calendar_days": 0.0,
            },
        ]
    )
    res = attach_event_clock_history_features(
        target_orders=orders,
        history_events=orders,
        smoothing_alpha=5.0,
        default_prior=0.10,
    ).set_index("order_id")

    assert np.isclose(res.loc["O_multi", "seller_completed_count"], 1.5)
    assert np.isclose(res.loc["O_multi", "seller_prior_purchase_count"], 1.5)
    rate_s1 = (1.0 + 5.0 * 0.10) / (1.0 + 5.0)
    rate_s2 = (0.0 + 5.0 * 0.10) / (2.0 + 5.0)
    assert np.isclose(res.loc["O_multi", "seller_hist_late_rate_smoothed"], 0.5 * (rate_s1 + rate_s2))


def test_a06_feature_versions_and_minimal_input_validation() -> None:
    """A06: get_feature_columns distinguishes base_v2 vs extended_purchase_assumptions_v2, and build_purchase_time_features validates schema."""
    import pytest
    from olist_delay.features import build_purchase_time_features, get_feature_columns

    num_base, cat_base, all_base = get_feature_columns("base_v2")
    num_ext, cat_ext, all_ext = get_feature_columns("extended_purchase_assumptions_v2")

    assert "payment_value_sum" not in all_base
    assert "primary_payment_type" not in all_base
    assert "payment_value_sum" in all_ext
    assert "primary_payment_type" in all_ext
    assert len(all_ext) == len(all_base) + 7

    minimal_df = pd.DataFrame(
        {
            "order_id": ["o1"],
            "order_purchase_timestamp": ["2018-06-01 10:00:00"],
            "order_estimated_delivery_date": ["2018-06-10 00:00:00"],
        }
    )
    with pytest.raises(ValueError, match="missing required master/scoring columns"):
        build_purchase_time_features(minimal_df)

