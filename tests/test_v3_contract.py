"""Unit tests for the v3 contract: censoring-aware label maturity, v3 route features, new rules and code fixes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olist_delay.artifacts import sha256_file, verify_model_artifact_hash
from olist_delay.config import load_config
from olist_delay.features import BASE_V2_ALL_FEATURES, BASE_V3_ALL_FEATURES, MODEL_INPUT_COLUMNS, get_feature_columns
from olist_delay.history import attach_v3_route_features
from olist_delay.labels import compute_label_known_at
from olist_delay.models import build_ml_pipeline
from olist_delay.pipeline_runner import (
    bootstrap_replicates_from_config,
    compute_cost_scenario_table,
    select_candidate_from_validation,
)
from olist_delay.splits import build_split_masks


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["on_time", "late_long", "late_quick", "not_a"],
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-01-01", "2018-01-01"]),
            "order_estimated_delivery_date": pd.to_datetime(["2018-01-10", "2018-01-10", "2018-01-10", "2018-01-10"]),
            "label_available_at": pd.to_datetime(
                ["2018-01-08 00:00:00", "2018-01-25 00:00:00", "2018-01-11 08:00:00", None], format="%Y-%m-%d %H:%M:%S"
            ),
            "late": pd.array([0, 1, 1, pd.NA], dtype="Int8"),
            "in_cohort_a": [True, True, True, False],
        }
    )


@pytest.mark.unit
def test_label_known_at_rules() -> None:
    df = _orders()
    v2 = compute_label_known_at(df, "delivered_at")
    assert v2.equals(df["label_available_at"])
    ca = compute_label_known_at(df, "delivered_or_overdue")
    assert ca.iloc[0] == pd.Timestamp("2018-01-08")  # on time: known at delivery
    assert ca.iloc[1] == pd.Timestamp("2018-01-11")  # late, delivered later: known the day after the promise
    assert ca.iloc[2] == pd.Timestamp("2018-01-11")  # late, delivered on the overdue day: the earlier of the two
    assert pd.isna(ca.iloc[3])  # outside Cohort A: never known
    with pytest.raises(ValueError):
        compute_label_known_at(df, "whenever")


@pytest.mark.unit
def test_censoring_aware_maturity_admits_known_late_orders_only() -> None:
    df = _orders()
    df["label_known_at"] = compute_label_known_at(df, "delivered_or_overdue")
    kwargs = dict(
        split_id="t", split_type="validation", train_start="2017-12-01", train_end="2018-01-15",
        eval_start="2018-02-01", eval_end="2018-03-01", selection_at="2018-03-15",
    )
    v2 = build_split_masks(df, **kwargs)
    v3 = build_split_masks(df, known_col="label_known_at", **kwargs)
    assert v2.train_matured_mask.tolist() == [True, False, True, False]
    # The long-delayed order is still in transit on 15 Jan, but it is known to be late since 11 Jan.
    assert v3.train_matured_mask.tolist() == [True, True, True, False]


def _route_events() -> pd.DataFrame:
    base = pd.Timestamp("2018-03-01")
    rows = []
    for i in range(30):
        rows.append(
            {
                "order_id": f"e{i}",
                "order_purchase_timestamp": base + pd.Timedelta(days=i % 10),
                "order_estimated_delivery_date": base + pd.Timedelta(days=i % 10 + 10),
                "label_available_at": base + pd.Timedelta(days=i % 10 + (25 if i % 3 == 0 else 8)),
                "late": 1 if i % 3 == 0 else 0,
                "route_state_pair": "SP->SP",
                "estimated_lead_calendar_days": 10.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_v3_route_features_are_causal_and_count_overdue_orders_early() -> None:
    ev = _route_events()
    query = pd.DataFrame(
        {
            "order_id": ["q"],
            "order_purchase_timestamp": [pd.Timestamp("2018-03-22")],
            "route_state_pair": ["SP->SP"],
            "estimated_lead_calendar_days": [7.0],
        }
    )
    out = attach_v3_route_features(query, history_events=ev, smoothing_m=0.0)
    # On 22 Mar the late orders (delivered 26 Mar or later) are not delivered yet, but those whose promise passed
    # before 22 Mar already count as late; on-time orders delivered before 22 Mar count as on time.
    known = ev.assign(
        known=np.where(ev["late"] == 1, ev["order_estimated_delivery_date"] + pd.Timedelta(days=1), ev["label_available_at"])
    )
    seen = known[known["known"] < pd.Timestamp("2018-03-22")]
    assert out["route_recent_60d_late_rate_ca"].iloc[0] == pytest.approx(seen["late"].mean())
    # Durations only use orders already delivered (delivery clock).
    delivered = ev[ev["label_available_at"] < pd.Timestamp("2018-03-22")]
    expected_days = ((delivered["label_available_at"] - delivered["order_purchase_timestamp"]).dt.days).mean()
    assert out["route_recent_60d_mean_delivery_days"].iloc[0] == pytest.approx(expected_days)
    assert out["promise_slack_days_60d"].iloc[0] == pytest.approx(7.0 - expected_days)

    # Perturbing an event's outcome that is not yet knowable must not change the features.
    ev2 = ev.copy()
    future = ev2["label_available_at"] >= pd.Timestamp("2018-03-22")
    ev2.loc[future & (ev2["late"] == 0), "label_available_at"] += pd.Timedelta(days=30)
    out2 = attach_v3_route_features(query, history_events=ev2, smoothing_m=0.0)
    assert np.allclose(out[["route_recent_60d_late_rate_ca", "route_recent_60d_mean_delivery_days"]].to_numpy(),
                       out2[["route_recent_60d_late_rate_ca", "route_recent_60d_mean_delivery_days"]].to_numpy())


@pytest.mark.unit
def test_base_v3_extends_base_v2_without_touching_v2_versions() -> None:
    num3, cat3, all3 = get_feature_columns("base_v3")
    assert set(BASE_V2_ALL_FEATURES) < set(BASE_V3_ALL_FEATURES) == set(all3)
    assert {"route_recent_60d_late_rate_ca", "promise_slack_days_60d"} <= set(num3)
    assert "promise_slack_days_60d" not in get_feature_columns("extended_purchase_assumptions_v2")[2]
    assert set(all3) <= set(MODEL_INPUT_COLUMNS)


@pytest.mark.unit
def test_v3_rules_rank_as_designed() -> None:
    X = pd.DataFrame(
        {
            "route_recent_60d_late_rate_smoothed": [0.02, 0.10, 0.10],
            "route_recent_60d_late_rate_ca": [0.02, 0.10, 0.20],
            "estimated_lead_calendar_days": [20.0, 20.0, 5.0],
            "promise_slack_days_60d": [10.0, 0.0, -5.0],
        }
    )
    y = np.array([0, 1, 1])
    for name in ["rule_route_recent_lead", "rule_route_recent_ca_lead", "rule_promise_slack"]:
        p = build_ml_pipeline(name).fit(X, y).predict_proba(X)[:, 1]
        assert p[0] < p[1] < p[2], name
        assert np.all((p > 0) & (p < 1))


@pytest.mark.unit
def test_bootstrap_replicates_come_from_config() -> None:
    assert bootstrap_replicates_from_config({"uncertainty_contract": {"row_bootstrap_replicates": 2000, "block_bootstrap_replicates": 1500}}) == (2000, 1500)
    assert bootstrap_replicates_from_config({}) == (2000, 2000)
    cfg = load_config("configs/experiment-v2.json")
    assert bootstrap_replicates_from_config(cfg) == (2000, 2000)
    with pytest.raises(ValueError):
        bootstrap_replicates_from_config({"uncertainty_contract": {"row_bootstrap_replicates": 0}})


@pytest.mark.unit
def test_cost_scenario_table_daily_floor_and_breakeven() -> None:
    days = pd.date_range("2018-06-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(20)],
            "order_purchase_timestamp": [days[i // 10] + pd.Timedelta(minutes=i) for i in range(20)],
            "late_calendar_day": [1] + [0] * 9 + [0] * 10,
        }
    )
    scores = np.linspace(1.0, 0.0, 20)
    sc = [{"scenario_id": "s", "benefit_per_tp": 5.0, "cost_per_contact": 0.25, "cost_per_fp": 0.15, "cost_per_fn": 2.0}]
    t = compute_cost_scenario_table(df, scores, "late_calendar_day", [0.10], sc, "m", "bt")
    row = t.iloc[0]
    assert (row["alerts"], row["tp"], row["fp"]) == (2, 1, 1)  # floor(0.1 * 10) per day
    assert row["net_value_per_1000_orders"] == pytest.approx(1000 * (5.0 - 2 * 0.25 - 0.15) / 20)
    assert row["breakeven_precision"] == pytest.approx(0.4 / 5.15, abs=1e-6)


@pytest.mark.unit
def test_model_hash_is_checked_before_unpickling(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    model = run / "models" / "selected_model.pkl"
    model.write_bytes(b"not really a pickle")
    (run / "manifest.json").write_text(
        json.dumps({"artifacts": {"models/selected_model.pkl": {"sha256": sha256_file(model)}}}), encoding="utf-8"
    )
    assert verify_model_artifact_hash(run, model) == sha256_file(model)
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_model_artifact_hash(run, model)


def _val_metrics(rule_recall: float, ml_recall: float) -> pd.DataFrame:
    rows = []
    for split in ["f1", "f2"]:
        for cid, rec, cal in [("rule_region_lead", rule_recall, "uncalibrated"), ("lr_x", ml_recall, "platt")]:
            rows.append(
                {
                    "candidate_id": cid, "model_name": cid if cid.startswith("rule") else "logistic_regression",
                    "feature_version": "base_v2", "hyperparams_json": "{}", "calibration_method": cal,
                    "label_version": "calendar_day", "metrics_status": "OK", "split_id": split,
                    "daily_top10_recall": rec, "daily_top10_precision": rec, "pr_auc": 0.1, "roc_auc": 0.6,
                    "brier_score": 0.05, "ece_quantile_10": 0.01, "precision_top10": rec, "recall_top10": rec,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_selection_requires_configured_minimum_gain() -> None:
    cfg = {"selection_contract": {"primary_metric": "mean_daily_10pct_capture", "min_relative_gain_over_best_rule": 0.005}}
    small = select_candidate_from_validation(_val_metrics(0.200, 0.203), cfg)
    assert small["selected_candidate_id"] == "rule_region_lead"
    assert small["primary_metric_aggregation"] == "mean_over_validation_folds"
    big = select_candidate_from_validation(_val_metrics(0.200, 0.210), cfg)
    assert big["selected_candidate_id"] == "lr_x"
    legacy = select_candidate_from_validation(_val_metrics(0.200, 0.203), {"selection_contract": {"primary_metric": "pooled_daily_10pct_capture"}})
    assert legacy["selected_candidate_id"] == "lr_x"  # no configured minimum: any gain counts
    assert "mean of fold-level" in legacy["primary_metric_note"]
