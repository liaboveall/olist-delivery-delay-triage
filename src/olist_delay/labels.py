"""Dual-label computation (calendar_day vs timestamp), label_available_at, and reason codes."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


REQUIRED_ORDER_COLS = {
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
}


def compute_order_labels(orders: pd.DataFrame, primary_label: str = "calendar_day") -> pd.DataFrame:
    """Compute dual labels, label availability timestamps, and mutually exclusive reason codes.

    Rules (frozen in research-contract.md):
    - Conditionally evaluable research cohort (Path A):
      `order_status == 'delivered'` AND `has_items == True` (when `has_items` column present)
      AND valid non-null `order_purchase_timestamp`, `order_delivered_customer_date`,
      and `order_estimated_delivery_date`.
    - `late_calendar_day`: 1 if `delivered_date > estimated_date` (normalized calendar day), 0 if `<=`, NA if not in Path A.
    - `late_timestamp`: 1 if `delivered_ts > estimated_ts` (full timestamp), 0 if `<=`, NA if not in Path A.
    - `same_day_after_midnight`: True when `delivered_date == estimated_date` AND `delivered_ts > estimated_ts`.
    - `label_available_at`: `order_delivered_customer_date` for Path A orders, else `NaT`.
    - Non-Path-A orders retain `pd.NA` targets (never coerced to 0!).
    """
    missing_cols = REQUIRED_ORDER_COLS - set(orders.columns)
    if missing_cols:
        raise ValueError(f"Orders table missing required columns: {sorted(missing_cols)}")

    df = orders.copy()
    time_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in time_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    has_purchase = df["order_purchase_timestamp"].notna()
    has_estimated = df["order_estimated_delivery_date"].notna()
    has_delivered_ts = df["order_delivered_customer_date"].notna()
    is_delivered_status = df["order_status"].eq("delivered")
    has_items_mask = (
        df["has_items"].fillna(False).astype(bool)
        if "has_items" in df.columns
        else pd.Series(True, index=df.index)
    )

    in_cohort_a = is_delivered_status & has_purchase & has_estimated & has_delivered_ts & has_items_mask

    # Normalized calendar dates
    deliv_day = df["order_delivered_customer_date"].dt.normalize()
    est_day = df["order_estimated_delivery_date"].dt.normalize()

    cal_gt = deliv_day > est_day
    ts_gt = df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    same_day = (deliv_day == est_day) & ts_gt & in_cohort_a

    df["in_cohort_a"] = in_cohort_a
    df["is_label_observable"] = in_cohort_a
    df["label_available_at"] = df["order_delivered_customer_date"].where(in_cohort_a, pd.NaT)

    # Nullable integer targets so unknown outcomes are NEVER turned into 0
    late_cal = pd.Series(pd.NA, index=df.index, dtype="Int8")
    late_ts = pd.Series(pd.NA, index=df.index, dtype="Int8")
    late_cal.loc[in_cohort_a] = cal_gt.loc[in_cohort_a].astype("int8").to_numpy()
    late_ts.loc[in_cohort_a] = ts_gt.loc[in_cohort_a].astype("int8").to_numpy()

    df["late_calendar_day"] = late_cal
    df["late_timestamp"] = late_ts
    df["same_day_after_midnight"] = same_day

    if primary_label == "calendar_day":
        df["late"] = df["late_calendar_day"]
    elif primary_label == "timestamp":
        df["late"] = df["late_timestamp"]
    else:
        raise ValueError(f"Unknown primary_label: {primary_label!r}")

    # Continuous delay in days and calendar delay in days (only for observable outcomes)
    df["delay_days_continuous"] = np.where(
        in_cohort_a,
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0,
        np.nan,
    )
    df["delay_calendar_days"] = np.where(
        in_cohort_a,
        (deliv_day - est_day).dt.total_seconds() / 86400.0,
        np.nan,
    )

    # Mutually exclusive primary reason code for cohort reconciliation
    reason = np.full(len(df), "OTHER_UNCLASSIFIED", dtype=object)
    reason[in_cohort_a.to_numpy()] = "OBSERVED_DELIVERED"
    reason[(is_delivered_status & has_purchase & has_estimated & has_delivered_ts & ~has_items_mask).to_numpy()] = (
        "DELIVERED_NO_ITEMS"
    )
    reason[(~has_purchase | ~has_estimated).to_numpy()] = "INVALID_KEY_TIMESTAMPS"
    reason[(is_delivered_status & has_purchase & has_estimated & ~has_delivered_ts).to_numpy()] = (
        "DELIVERED_MISSING_DELIVERY_TS"
    )
    reason[(~is_delivered_status & has_purchase & has_estimated & has_delivered_ts).to_numpy()] = (
        "NON_DELIVERED_WITH_DELIVERY_TS"
    )
    for st in ["shipped", "canceled", "unavailable", "invoiced", "processing", "created", "approved"]:
        mask = (~is_delivered_status) & has_purchase & has_estimated & (~has_delivered_ts) & df["order_status"].eq(st)
        reason[mask.to_numpy()] = f"UNDELIVERED_{st.upper()}"

    df["label_reason_code"] = reason
    return df


LABEL_KNOWN_RULES = ("delivered_at", "delivered_or_overdue")


def compute_label_known_at(df: pd.DataFrame, rule: str = "delivered_at") -> pd.Series:
    """Timestamp from which an order's primary label could be known by an observer.

    - `delivered_at` (v2 contract): the customer delivery timestamp (`label_available_at`).
    - `delivered_or_overdue` (censoring-aware): for a late order, the start of the day after the promised date if
      that comes before delivery, because an undelivered order is known to be late once its promised day has
      passed. On-time orders are still known only at delivery. Only Cohort A orders get a timestamp, so the
      eventual-delivery conditioning is the same as under `delivered_at`.
    """
    if rule not in LABEL_KNOWN_RULES:
        raise ValueError(f"Unsupported label_known_rule: {rule!r}; expected one of {LABEL_KNOWN_RULES}")
    avail = pd.to_datetime(df["label_available_at"])
    if rule == "delivered_at":
        return avail.copy()
    overdue_at = pd.to_datetime(df["order_estimated_delivery_date"]).dt.normalize() + pd.Timedelta(days=1)
    is_late = pd.to_numeric(df["late"], errors="coerce").eq(1)
    earlier = is_late & avail.notna() & overdue_at.notna() & (overdue_at < avail)
    return avail.where(~earlier, overdue_at)


def summarize_label_reconciliation(
    labeled_orders: pd.DataFrame,
    backtest_start: str = "2018-05-26 18:18:03",
    backtest_end: str = "2018-08-30 00:00:00",
) -> dict[str, Any]:
    """Reconcile calendar-day vs timestamp labels across full Cohort A and backtest window."""
    cohort_a = labeled_orders[labeled_orders["in_cohort_a"]].copy()
    bt_start = pd.Timestamp(backtest_start)
    bt_end = pd.Timestamp(backtest_end)
    bt = cohort_a[
        (cohort_a["order_purchase_timestamp"] >= bt_start)
        & (cohort_a["order_purchase_timestamp"] < bt_end)
    ]

    est_non_null = labeled_orders["order_estimated_delivery_date"].dropna()
    all_est_midnight = bool((est_non_null == est_non_null.dt.normalize()).all())

    ts_pos_all = int(cohort_a["late_timestamp"].sum())
    cal_pos_all = int(cohort_a["late_calendar_day"].sum())
    same_day_all = int(cohort_a["same_day_after_midnight"].sum())

    ts_pos_bt = int(bt["late_timestamp"].sum())
    cal_pos_bt = int(bt["late_calendar_day"].sum())
    same_day_bt = int(bt["same_day_after_midnight"].sum())

    return {
        "total_source_orders": int(len(labeled_orders)),
        "cohort_a_delivered_orders": int(len(cohort_a)),
        "unlabeled_excluded_orders": int(len(labeled_orders) - len(cohort_a)),
        "all_estimated_dates_at_midnight": all_est_midnight,
        "full_cohort_a": {
            "n": int(len(cohort_a)),
            "timestamp_late_positives": ts_pos_all,
            "late_timestamp_count": ts_pos_all,
            "timestamp_late_rate": float(ts_pos_all / max(1, len(cohort_a))),
            "calendar_day_late_positives": cal_pos_all,
            "late_calendar_day_count": cal_pos_all,
            "calendar_day_late_rate": float(cal_pos_all / max(1, len(cohort_a))),
            "same_day_after_midnight_count": same_day_all,
            "exact_difference_matches_same_day": (ts_pos_all - cal_pos_all) == same_day_all,
        },
        "backtest_window": {
            "n": int(len(bt)),
            "timestamp_late_positives": ts_pos_bt,
            "late_timestamp_count": ts_pos_bt,
            "timestamp_late_rate": float(ts_pos_bt / max(1, len(bt))),
            "calendar_day_late_positives": cal_pos_bt,
            "late_calendar_day_count": cal_pos_bt,
            "calendar_day_late_rate": float(cal_pos_bt / max(1, len(bt))),
            "same_day_after_midnight_count": same_day_bt,
            "exact_difference_matches_same_day": (ts_pos_bt - cal_pos_bt) == same_day_bt,
        },
        "reason_code_counts": labeled_orders["label_reason_code"].value_counts().to_dict(),
    }
