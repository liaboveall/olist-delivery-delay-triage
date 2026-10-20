"""Unit and dataset tests for T01 (dual-label boundary semantics) and T02 (nullable unknown outcomes)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from olist_delay.labels import compute_order_labels, summarize_label_reconciliation

pytestmark = pytest.mark.unit



def test_t01_dual_label_boundaries(synthetic_label_orders: pd.DataFrame) -> None:
    """T01: Same-day 23:59:59, exact midnight, next-day 00:00:00, and dual-label semantics."""
    df_cal = compute_order_labels(synthetic_label_orders, primary_label="calendar_day")
    by_id = df_cal.set_index("order_id")

    # Early delivery -> both 0
    assert by_id.loc["ord_early", "late_calendar_day"] == 0
    assert by_id.loc["ord_early", "late_timestamp"] == 0
    assert bool(by_id.loc["ord_early", "same_day_after_midnight"]) is False
    assert by_id.loc["ord_early", "late"] == 0

    # Same-day 23:59:59 -> calendar_day=0, timestamp=1, same_day_after_midnight=True
    assert by_id.loc["ord_same_day_late_ts", "late_calendar_day"] == 0
    assert by_id.loc["ord_same_day_late_ts", "late_timestamp"] == 1
    assert bool(by_id.loc["ord_same_day_late_ts", "same_day_after_midnight"]) is True
    assert by_id.loc["ord_same_day_late_ts", "late"] == 0

    # Exact midnight -> both 0
    assert by_id.loc["ord_exact_midnight", "late_calendar_day"] == 0
    assert by_id.loc["ord_exact_midnight", "late_timestamp"] == 0
    assert bool(by_id.loc["ord_exact_midnight", "same_day_after_midnight"]) is False

    # Next-day 00:00:00 -> both 1
    assert by_id.loc["ord_next_day_midnight", "late_calendar_day"] == 1
    assert by_id.loc["ord_next_day_midnight", "late_timestamp"] == 1
    assert bool(by_id.loc["ord_next_day_midnight", "same_day_after_midnight"]) is False
    assert by_id.loc["ord_next_day_midnight", "late"] == 1

    # Primary label switch to 'timestamp'
    df_ts = compute_order_labels(synthetic_label_orders, primary_label="timestamp")
    assert df_ts.set_index("order_id").loc["ord_same_day_late_ts", "late"] == 1


def test_t02_nullable_unknown_outcomes_never_coerced_to_zero(synthetic_label_orders: pd.DataFrame) -> None:
    """T02: Canceled, shipped, missing delivery date, and unavailable orders must retain pd.NA labels."""
    df = compute_order_labels(synthetic_label_orders, primary_label="calendar_day")
    by_id = df.set_index("order_id")

    for oid, expected_reason in [
        ("ord_canceled", "UNDELIVERED_CANCELED"),
        ("ord_shipped", "UNDELIVERED_SHIPPED"),
        ("ord_delivered_missing_ts", "DELIVERED_MISSING_DELIVERY_TS"),
        ("ord_unavailable", "UNDELIVERED_UNAVAILABLE"),
    ]:
        assert bool(by_id.loc[oid, "in_cohort_a"]) is False
        assert bool(by_id.loc[oid, "is_label_observable"]) is False
        assert pd.isna(by_id.loc[oid, "late_calendar_day"])
        assert pd.isna(by_id.loc[oid, "late_timestamp"])
        assert pd.isna(by_id.loc[oid, "late"])
        assert by_id.loc[oid, "label_reason_code"] == expected_reason

    rec = summarize_label_reconciliation(df)
    assert rec["cohort_a_delivered_orders"] == 4
    assert rec["unlabeled_excluded_orders"] == 4
    assert rec["full_cohort_a"]["calendar_day_late_positives"] == 1
    assert rec["full_cohort_a"]["timestamp_late_positives"] == 2
    assert rec["full_cohort_a"]["same_day_after_midnight_count"] == 1
    assert rec["full_cohort_a"]["exact_difference_matches_same_day"] is True


def test_t01_t02_real_cohort_reconciliation(project_root: Path) -> None:
    """Verify full Olist dataset reconciliation matches exact audited counts."""
    data_dir = project_root / "data"
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv", low_memory=False)
    items_order_ids = set(
        pd.read_csv(data_dir / "olist_order_items_dataset.csv", usecols=["order_id"], low_memory=False)["order_id"]
    )
    orders["has_items"] = orders["order_id"].isin(items_order_ids)
    df = compute_order_labels(orders, primary_label="calendar_day")

    assert len(df) == 99441
    assert int(df["has_items"].sum()) == 98666
    assert int((~df["has_items"]).sum()) == 775
    assert int(df["in_cohort_a"].sum()) == 96470
    assert int((~df["in_cohort_a"]).sum()) == 2971

    # Verify non-Cohort A rows have strictly missing labels (never 0)
    non_a = df[~df["in_cohort_a"]]
    assert non_a["late"].isna().all()
    assert non_a["late_calendar_day"].isna().all()
    assert non_a["late_timestamp"].isna().all()

    # Verify exact dual-label difference on Cohort A (7,826 - 6,534 = 1,292)
    coh_a = df[df["in_cohort_a"]]
    assert int(coh_a["late_calendar_day"].sum()) == 6534
    assert int(coh_a["late_timestamp"].sum()) == 7826
    assert int(coh_a["same_day_after_midnight"].sum()) == 1292


def test_a04_delivered_without_items_excluded_from_cohort_a(synthetic_label_orders: pd.DataFrame) -> None:
    """A04: A delivered order with has_items=False must be excluded from Cohort A and retain NA labels."""
    df_in = synthetic_label_orders.copy()
    df_in["has_items"] = True
    df_in.loc[df_in["order_id"] == "ord_early", "has_items"] = False
    df_out = compute_order_labels(df_in, primary_label="calendar_day").set_index("order_id")

    assert bool(df_out.loc["ord_early", "in_cohort_a"]) is False
    assert pd.isna(df_out.loc["ord_early", "late_calendar_day"])
    assert pd.isna(df_out.loc["ord_early", "late_timestamp"])
    assert df_out.loc["ord_early", "label_reason_code"] == "DELIVERED_NO_ITEMS"

