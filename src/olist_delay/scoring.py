"""Offline purchase-time batch scoring (`scoring-input-v2`), raw-table adapter, event snapshot filtering, and daily alert allocation."""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any
import numpy as np
import pandas as pd

from .artifacts import now_iso, sha256_file
from .data import build_master_order_table, load_raw_tables
from .features import (
    ALL_MODEL_FEATURES,
    FORBIDDEN_POST_PURCHASE_COLUMNS,
    assert_zero_leakage_features,
    build_purchase_time_features,
    get_feature_columns,
    get_features_for_version,
)
from .geography import BRAZIL_STATE_TO_REGION, compute_haversine_km
from .policy import (
    allocate_daily_alerts,
    assign_risk_tier,
    rank_orders_deterministically,
    summarize_daily_policy,
)


SCORING_INPUT_V2_VERSION = "scoring-input-v2"

SCORING_INPUT_V2_REQUIRED_COLUMNS: list[str] = [
    "order_id",
    "customer_unique_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_state",
    "customer_zip_code_prefix",
    "has_items",
    "item_count",
    "distinct_product_count",
    "primary_seller_id",
    "primary_seller_state",
    "primary_category",
    "total_price",
    "max_item_price",
    "mean_item_price",
    "freight_ratio",
    "total_weight_g",
    "max_weight_g",
    "total_volume_cm3",
    "max_volume_cm3",
    "mean_product_photos_qty",
]

SCORING_INPUT_V2_OPTIONAL_COLUMNS: list[str] = [
    "customer_id",
    "customer_city",
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
    "distinct_seller_count",
    "seller_count",
    "distinct_category_count",
    "is_multi_item",
    "is_multi_seller",
    "primary_seller_zip_prefix",
    "primary_seller_region",
    "seller_ids_pipe",
    "all_seller_ids",
    "total_freight",
    "total_freight_value",
    "max_freight",
    "order_total_value",
    "total_weight_kg",
    "max_weight_kg",
    "weight_missing",
    "volume_missing",
    "mean_description_length",
    "mean_description_lenght",
    "max_shipping_limit_days",
    "min_shipping_limit_days",
    "haversine_distance_km",
    "distance_km_max",
    "primary_distance_km",
    "same_state",
    "same_region",
    "any_cross_state_item",
    "geo_missing",
    "same_zip_prefix",
    "route_state_pair",
    "route_region_pair",
    "customer_region",
    "seller_region",
    "payment_value_sum",
    "payment_installments_max",
    "max_payment_installments",
    "payment_sequential_count",
    "payment_type_count",
    "primary_payment_type",
    "has_boleto",
    "boleto_flag",
    "used_boleto",
    "has_credit_card",
    "used_credit_card",
    "has_voucher",
    "used_voucher",
    "payment_total_value",
    "estimated_lead_calendar_days",
    "estimated_lead_Exact_days",
    "promised_Slack_days",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "purchase_is_weekend",
]

SCORING_HISTORY_EVENTS_REQUIRED_COLUMNS: list[str] = [
    "order_id",
    "primary_seller_id",
    "route_state_pair",
    "customer_state",
    "primary_category",
    "order_purchase_timestamp",
    "label_available_at",
    "late",
]

HISTORICAL_FEATURE_COLUMNS: set[str] = {
    "seller_prior_purchase_count",
    "seller_purchase_count_7d",
    "seller_purchase_count_30d",
    "seller_load_ratio_7d_vs_30d",
    "seller_purchase_count_7d_30d_ratio",
    "customer_prior_order_count",
    "customer_prior_purchase_count",
    "global_hist_completed_count",
    "global_completed_count",
    "global_hist_late_rate",
    "global_recent_30d_late_rate",
    "seller_completed_count",
    "seller_completed_count_30d",
    "seller_completed_count_90d",
    "seller_is_cold_start",
    "seller_hist_late_rate_smoothed",
    "seller_recent_30d_late_rate_smoothed",
    "seller_recent_90d_late_rate_smoothed",
    "seller_hist_mean_delay_days",
    "seller_excess_late_vs_global",
    "seller_any_completed_count",
    "seller_max_late_rate_smoothed",
    "seller_mean_late_rate_smoothed",
    "seller_max_recent_30d_late_rate_smoothed",
    "route_completed_count",
    "route_hist_late_rate_smoothed",
    "route_recent_60d_late_rate_smoothed",
    "route_hist_mean_delay_days",
    "route_excess_late_vs_global",
    "cust_state_completed_count",
    "cust_state_hist_late_rate_smoothed",
    "cust_state_recent_30d_late_rate_smoothed",
    "category_completed_count",
    "category_hist_late_rate_smoothed",
}


def _normalize_scoring_order_columns(orders: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema alias columns and compute derived geographical/route/temporal fields when missing."""
    work = orders.copy()
    alias_pairs = [
        ("distinct_seller_count", "seller_count"),
        ("seller_ids_pipe", "all_seller_ids"),
        ("total_freight", "total_freight_value"),
        ("mean_description_length", "mean_description_lenght"),
        ("payment_installments_max", "max_payment_installments"),
        ("payment_sequential_count", "payment_type_count"),
        ("payment_value_sum", "payment_total_value"),
        ("has_boleto", "used_boleto"),
        ("boleto_flag", "has_boleto"),
        ("has_credit_card", "used_credit_card"),
        ("has_voucher", "used_voucher"),
    ]
    for canonical, alias in alias_pairs:
        if canonical not in work.columns and alias in work.columns:
            work[canonical] = work[alias]
        elif alias not in work.columns and canonical in work.columns:
            work[alias] = work[canonical]

    # Weight unit conversions (kg <-> g)
    if "total_weight_g" not in work.columns and "total_weight_kg" in work.columns:
        work["total_weight_g"] = pd.to_numeric(work["total_weight_kg"], errors="coerce") * 1000.0
    elif "total_weight_kg" not in work.columns and "total_weight_g" in work.columns:
        work["total_weight_kg"] = pd.to_numeric(work["total_weight_g"], errors="coerce") / 1000.0

    if "max_weight_g" not in work.columns and "max_weight_kg" in work.columns:
        work["max_weight_g"] = pd.to_numeric(work["max_weight_kg"], errors="coerce") * 1000.0
    elif "max_weight_kg" not in work.columns and "max_weight_g" in work.columns:
        work["max_weight_kg"] = pd.to_numeric(work["max_weight_g"], errors="coerce") / 1000.0

    if "mean_product_photos_qty" not in work.columns:
        work["mean_product_photos_qty"] = 1.0

    for bcol, default_val in [
        ("has_boleto", 0),
        ("boleto_flag", 0),
        ("has_credit_card", 1),
        ("has_voucher", 0),
    ]:
        if bcol in work.columns:
            work[bcol] = pd.to_numeric(work[bcol], errors="coerce").fillna(default_val).astype(int)
        else:
            work[bcol] = default_val

    if "distinct_seller_count" not in work.columns:
        work["distinct_seller_count"] = 1
    if "distinct_category_count" not in work.columns:
        work["distinct_category_count"] = 1
    if "is_multi_item" not in work.columns:
        work["is_multi_item"] = (pd.to_numeric(work.get("item_count", 1), errors="coerce").fillna(1) > 1).astype(int)
    if "is_multi_seller" not in work.columns:
        work["is_multi_seller"] = (pd.to_numeric(work["distinct_seller_count"], errors="coerce").fillna(1) > 1).astype(int)
    if "total_freight" not in work.columns:
        work["total_freight"] = 0.0
    if "max_freight" not in work.columns:
        work["max_freight"] = work["total_freight"]
    if "order_total_value" not in work.columns:
        work["order_total_value"] = (
            pd.to_numeric(work.get("total_price", 0.0), errors="coerce").fillna(0.0)
            + pd.to_numeric(work["total_freight"], errors="coerce").fillna(0.0)
        )
    if "payment_value_sum" not in work.columns:
        work["payment_value_sum"] = work["order_total_value"]
    if "payment_installments_max" not in work.columns:
        work["payment_installments_max"] = 1.0
    if "payment_sequential_count" not in work.columns:
        work["payment_sequential_count"] = 1.0
    if "primary_payment_type" not in work.columns:
        work["primary_payment_type"] = "credit_card"

    if "weight_missing" not in work.columns:
        work["weight_missing"] = (
            work["total_weight_kg"].isna().astype(int)
            if "total_weight_kg" in work.columns
            else 1
        )
    if "volume_missing" not in work.columns:
        work["volume_missing"] = (
            work["total_volume_cm3"].isna().astype(int)
            if "total_volume_cm3" in work.columns
            else 1
        )

    if "mean_description_length" not in work.columns:
        work["mean_description_length"] = 500.0
    if "max_shipping_limit_days" not in work.columns:
        work["max_shipping_limit_days"] = 5.0
    if "min_shipping_limit_days" not in work.columns:
        work["min_shipping_limit_days"] = work["max_shipping_limit_days"]

    c_state = work["customer_state"].fillna("UNKNOWN").astype(str) if "customer_state" in work.columns else pd.Series("UNKNOWN", index=work.index)
    s_state = work["primary_seller_state"].fillna("UNKNOWN").astype(str) if "primary_seller_state" in work.columns else pd.Series("UNKNOWN", index=work.index)
    if "same_state" not in work.columns:
        work["same_state"] = ((c_state == s_state) & (s_state != "UNKNOWN")).astype(int)
    if "any_cross_state_item" not in work.columns:
        work["any_cross_state_item"] = (work["same_state"] == 0).astype(int)
    if "same_zip_prefix" not in work.columns:
        c_zip = work["customer_zip_code_prefix"].fillna("").astype(str) if "customer_zip_code_prefix" in work.columns else pd.Series("", index=work.index)
        s_zip = (
            work["primary_seller_zip_prefix"].fillna("UNKNOWN").astype(str)
            if "primary_seller_zip_prefix" in work.columns
            else pd.Series("UNKNOWN", index=work.index)
        )
        work["same_zip_prefix"] = ((c_zip == s_zip) & (s_zip != "UNKNOWN") & (c_zip != "")).astype(int)
    if "route_state_pair" not in work.columns:
        work["route_state_pair"] = s_state + "->" + c_state
    if "customer_region" not in work.columns:
        work["customer_region"] = c_state.map(lambda s: BRAZIL_STATE_TO_REGION.get(s, "Unknown"))
    if "seller_region" not in work.columns:
        work["seller_region"] = s_state.map(lambda s: BRAZIL_STATE_TO_REGION.get(s, "Unknown"))
    if "primary_seller_region" not in work.columns:
        work["primary_seller_region"] = work["seller_region"]
    if "same_region" not in work.columns:
        work["same_region"] = ((work["customer_region"] == work["primary_seller_region"]) & (work["primary_seller_region"] != "Unknown")).astype(int)
    if "route_region_pair" not in work.columns:
        work["route_region_pair"] = work["primary_seller_region"].astype(str) + "->" + work["customer_region"].astype(str)

    for gcol in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]:
        if gcol not in work.columns:
            work[gcol] = np.nan
    if "haversine_distance_km" not in work.columns:
        if {"customer_lat", "customer_lng", "seller_lat", "seller_lng"}.issubset(work.columns):
            work["haversine_distance_km"] = compute_haversine_km(
                pd.to_numeric(work["customer_lat"], errors="coerce"),
                pd.to_numeric(work["customer_lng"], errors="coerce"),
                pd.to_numeric(work["seller_lat"], errors="coerce"),
                pd.to_numeric(work["seller_lng"], errors="coerce"),
            )
        else:
            work["haversine_distance_km"] = np.nan
    if "distance_km_max" not in work.columns:
        work["distance_km_max"] = work["haversine_distance_km"]
    if "primary_distance_km" not in work.columns:
        work["primary_distance_km"] = work["haversine_distance_km"]
    if "geo_missing" not in work.columns:
        work["geo_missing"] = (work["customer_lat"].isna() | work["seller_lat"].isna()).astype(int)

    if {"order_purchase_timestamp", "order_estimated_delivery_date"}.issubset(work.columns):
        p_ts = pd.to_datetime(work["order_purchase_timestamp"], errors="coerce")
        e_ts = pd.to_datetime(work["order_estimated_delivery_date"], errors="coerce")
        if "estimated_lead_calendar_days" not in work.columns:
            work["estimated_lead_calendar_days"] = (e_ts.dt.normalize() - p_ts.dt.normalize()).dt.total_seconds() / 86400.0
        if "estimated_lead_Exact_days" not in work.columns:
            work["estimated_lead_Exact_days"] = (e_ts - p_ts).dt.total_seconds() / 86400.0
        if "promised_Slack_days" not in work.columns:
            work["promised_Slack_days"] = work["estimated_lead_Exact_days"] + 1.0
        if "purchase_month" not in work.columns:
            work["purchase_month"] = p_ts.dt.month
        if "purchase_day_of_week" not in work.columns:
            work["purchase_day_of_week"] = p_ts.dt.dayofweek
        if "purchase_hour" not in work.columns:
            work["purchase_hour"] = p_ts.dt.hour
        if "purchase_is_weekend" not in work.columns:
            work["purchase_is_weekend"] = (work["purchase_day_of_week"] >= 5).astype(int)

    return work


def extract_scoring_input_v2_from_master(
    master_orders: pd.DataFrame,
    filter_purchase_date: str | None = None,
    order_ids: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Extract clean `scoring-input-v2` wide table from `master_orders` (before feature engineering).

    Never includes post-purchase outcome columns or pre-computed historical features.
    """
    work = _normalize_scoring_order_columns(master_orders)
    if filter_purchase_date is not None:
        p_dates = pd.to_datetime(work["order_purchase_timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
        work = work.loc[p_dates == str(filter_purchase_date)].copy()
    if order_ids is not None:
        id_set = {str(x) for x in order_ids}
        work = work.loc[work["order_id"].astype(str).isin(id_set)].copy()

    keep_cols = []
    seen = set()
    for c in SCORING_INPUT_V2_REQUIRED_COLUMNS + SCORING_INPUT_V2_OPTIONAL_COLUMNS:
        if (
            c not in seen
            and c in work.columns
            and c not in FORBIDDEN_POST_PURCHASE_COLUMNS
            and c not in HISTORICAL_FEATURE_COLUMNS
        ):
            keep_cols.append(c)
            seen.add(c)
    return work[keep_cols].reset_index(drop=True)


def build_scoring_input_v2_from_raw(
    raw_data_dir: str | Path | dict[str, pd.DataFrame],
    filter_purchase_date: str | None = None,
    order_ids: list[str] | set[str] | None = None,
    geo_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapter that constructs `scoring_orders_df` (if raw_tables dict passed) or `(scoring_orders_df, history_events_df, decision_events_df)` (if path passed)."""
    is_dict_input = isinstance(raw_data_dir, dict)
    tables = raw_data_dir if is_dict_input else load_raw_tables(Path(raw_data_dir))
    master, _, _ = build_master_order_table(tables, primary_label="calendar_day", geo_cfg=geo_cfg)
    scoring_orders = extract_scoring_input_v2_from_master(
        master,
        filter_purchase_date=filter_purchase_date,
        order_ids=order_ids,
    )
    if is_dict_input:
        return scoring_orders

    if len(scoring_orders) > 0:
        max_purchase_day = (
            pd.to_datetime(scoring_orders["order_purchase_timestamp"], errors="coerce")
            .max()
            .floor("D")
        )
        as_of_cutoff = max_purchase_day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    else:
        as_of_cutoff = None

    hist_events, decision_events = build_history_and_decision_events_from_master(
        master,
        as_of=as_of_cutoff,
        target_order_ids=scoring_orders["order_id"].astype(str).tolist(),
    )
    return scoring_orders, hist_events, decision_events


def build_history_and_decision_events_from_master(
    master_orders: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    target_order_ids: list[str] | set[str] | None = None,
    cutoff_ts: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal outcome history events and status decision events strictly visible at `as_of`."""
    effective_as_of = as_of if as_of is not None else cutoff_ts
    work = master_orders.copy()
    p_ts = pd.to_datetime(work["order_purchase_timestamp"], errors="coerce")

    if effective_as_of is not None:
        as_of_ts = pd.Timestamp(effective_as_of)
        work = work.loc[p_ts <= as_of_ts].copy()
        l_ts_sub = pd.to_datetime(work["label_available_at"], errors="coerce")
        unmatured_at_as_of = l_ts_sub.isna() | (l_ts_sub > as_of_ts)
        work.loc[unmatured_at_as_of, "label_available_at"] = pd.NaT
        work.loc[unmatured_at_as_of, "in_cohort_a"] = False
        for col in ["late_calendar_day", "late_timestamp", "late", "delay_calendar_days"]:
            if col in work.columns:
                work.loc[unmatured_at_as_of, col] = pd.NA

    hist_cols = [
        c
        for c in [
            "order_id",
            "customer_unique_id",
            "primary_seller_id",
            "seller_ids_pipe",
            "route_state_pair",
            "customer_state",
            "primary_category",
            "order_purchase_timestamp",
            "label_available_at",
            "in_cohort_a",
            "late_calendar_day",
            "late_timestamp",
            "late",
            "delay_calendar_days",
        ]
        if c in work.columns
    ]
    hist_events = work[hist_cols].reset_index(drop=True)

    # Build decision status events (e.g., orders already delivered on or before as_of)
    dec_work = master_orders.copy()
    if target_order_ids is not None:
        id_set = {str(x) for x in target_order_ids}
        dec_work = dec_work.loc[dec_work["order_id"].astype(str).isin(id_set)].copy()

    deliv_ts = pd.to_datetime(dec_work.get("order_delivered_customer_date"), errors="coerce")
    p_ts_dec = pd.to_datetime(dec_work["order_purchase_timestamp"], errors="coerce")
    valid_deliv = deliv_ts.notna() & (deliv_ts >= p_ts_dec)
    if effective_as_of is not None:
        valid_deliv = valid_deliv & (deliv_ts <= pd.Timestamp(effective_as_of))

    deliv_rows = dec_work.loc[valid_deliv]
    if len(deliv_rows) > 0:
        decision_events = pd.DataFrame(
            {
                "order_id": deliv_rows["order_id"].astype(str).to_numpy(),
                "event_type": "delivered",
                "event_timestamp": pd.to_datetime(deliv_rows["order_delivered_customer_date"]).to_numpy(),
            }
        )
    else:
        decision_events = pd.DataFrame(columns=["order_id", "event_type", "event_timestamp"])

    return hist_events, decision_events


def validate_and_prepare_scoring_inputs(
    orders_in: pd.DataFrame,
    events_in: pd.DataFrame,
    decision_events_in: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """Validate `scoring-input-v2` schema, strip any pre-computed history or forbidden columns, and enforce `as_of`."""
    if orders_in.empty:
        raise ValueError("Scoring input CSV is empty (0 rows).")

    missing_in = sorted(set(SCORING_INPUT_V2_REQUIRED_COLUMNS) - set(orders_in.columns))
    if missing_in:
        raise ValueError(
            f"Scoring input CSV does not conform to '{SCORING_INPUT_V2_VERSION}' (scoring_input_v2) — "
            f"missing {len(missing_in)} required columns: {missing_in}. "
            f"(Pass --raw-data-dir <dir> if adapting minimal raw order rows.)"
        )

    orders = _normalize_scoring_order_columns(orders_in)

    missing_ev = sorted(set(["order_id", "primary_seller_id", "order_purchase_timestamp", "label_available_at", "late"]) - set(events_in.columns))
    if missing_ev:
        raise ValueError(
            f"History events CSV missing required columns: {missing_ev}"
        )

    if orders["order_id"].isna().any() or (orders["order_id"].astype(str).str.strip() == "").any():
        raise ValueError("Input orders contain null or empty order_id values.")
    if orders["order_id"].astype(str).duplicated().any():
        dups = orders.loc[orders["order_id"].astype(str).duplicated(), "order_id"].astype(str).unique().tolist()
        raise ValueError(f"Input orders contain duplicate order_id values: {dups[:10]}")

    q_ts = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    if q_ts.isna().any():
        raise ValueError("Input orders contain unparseable or null order_purchase_timestamp!")
    orders["order_purchase_timestamp"] = q_ts

    est_ts = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
    if est_ts.isna().any():
        raise ValueError("Input orders contain unparseable or null order_estimated_delivery_date!")
    orders["order_estimated_delivery_date"] = est_ts

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        if (q_ts > as_of_ts).any():
            raise ValueError(
                f"Input orders contain order_purchase_timestamp after `--as-of` ({as_of_ts.isoformat()})!"
            )

    # Check numeric columns for +/-Inf
    num_cols = [
        "item_count",
        "distinct_product_count",
        "distinct_seller_count",
        "total_price",
        "total_freight",
        "max_item_price",
        "mean_item_price",
        "freight_ratio",
        "total_weight_g",
        "max_weight_g",
        "total_volume_cm3",
        "max_volume_cm3",
        "mean_product_photos_qty",
        "mean_description_length",
        "max_shipping_limit_days",
        "min_shipping_limit_days",
        "haversine_distance_km",
    ]
    for c in num_cols:
        if c in orders.columns:
            vals = pd.to_numeric(orders[c], errors="coerce").to_numpy(dtype=np.float64)
            if np.isinf(vals).any():
                raise ValueError(f"Input orders column {c!r} contains Inf / infinite values!")

    # Strip any pre-existing historical columns or post-purchase outcome columns so scoring never trusts them
    cols_to_strip = [
        c
        for c in orders.columns
        if c in HISTORICAL_FEATURE_COLUMNS or c in FORBIDDEN_POST_PURCHASE_COLUMNS
    ]
    if cols_to_strip:
        orders = orders.drop(columns=cols_to_strip)

    # Filter history events by as_of
    events = events_in.copy()
    if "route_state_pair" not in events.columns:
        s_st = events.get("primary_seller_state", pd.Series("UNKNOWN", index=events.index)).fillna("UNKNOWN").astype(str)
        c_st = events.get("customer_state", pd.Series("UNKNOWN", index=events.index)).fillna("UNKNOWN").astype(str)
        events["route_state_pair"] = s_st + "->" + c_st
    if "customer_state" not in events.columns:
        events["customer_state"] = "UNKNOWN"
    if "primary_category" not in events.columns:
        events["primary_category"] = "unknown"

    ev_p_ts = pd.to_datetime(events["order_purchase_timestamp"], errors="coerce")
    ev_l_ts = pd.to_datetime(events["label_available_at"], errors="coerce")
    events["order_purchase_timestamp"] = ev_p_ts
    events["label_available_at"] = ev_l_ts

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        events = events.loc[ev_p_ts.notna() & (ev_p_ts <= as_of_ts)].copy()
        ev_l_sub = pd.to_datetime(events["label_available_at"], errors="coerce")
        unmatured_mask = ev_l_sub.isna() | (ev_l_sub > as_of_ts)
        events.loc[unmatured_mask, "label_available_at"] = pd.NaT
        if "in_cohort_a" in events.columns:
            events.loc[unmatured_mask, "in_cohort_a"] = False
        for col in ["late_calendar_day", "late_timestamp", "late", "delay_calendar_days"]:
            if col in events.columns:
                events.loc[unmatured_mask, col] = pd.NA

    dec_events: pd.DataFrame | None = None
    if decision_events_in is not None:
        dec_events = decision_events_in.copy()
        if not dec_events.empty:
            if not {"order_id", "event_type", "event_timestamp"}.issubset(dec_events.columns):
                raise ValueError(
                    "Decision events CSV must contain columns: ['order_id', 'event_type', 'event_timestamp']"
                )
            dec_ts = pd.to_datetime(dec_events["event_timestamp"], errors="coerce")
            dec_events["event_timestamp"] = dec_ts
            if as_of is not None:
                dec_events = dec_events.loc[dec_ts.notna() & (dec_ts <= pd.Timestamp(as_of))].copy()

    input_audit = {
        "schema_version": SCORING_INPUT_V2_VERSION,
        "orders_count": int(len(orders)),
        "events_count": int(len(events)),
        "stripped_columns": cols_to_strip,
        "as_of": pd.Timestamp(as_of).isoformat() if as_of is not None else None,
    }
    return orders.reset_index(drop=True), events.reset_index(drop=True), dec_events, input_audit


def score_orders_batch(
    orders_in: pd.DataFrame,
    events_in: pd.DataFrame | None = None,
    saved_model_payload: dict[str, Any] | None = None,
    decision_events_in: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp | None = None,
    primary_budget: float = 0.10,
    secondary_budget: float = 0.20,
    smoothing_alpha: float = 5.0,
    *,
    history_events: pd.DataFrame | None = None,
    decision_events: pd.DataFrame | None = None,
    bundle: Any = None,
    model_name: str | None = None,
    calibration_method: str | None = None,
    fallback_prior: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | tuple[pd.DataFrame, dict[str, Any]]:
    """Run full purchase-time feature engineering, probability prediction, and daily capacity alert allocation."""
    ev_df = events_in if events_in is not None else history_events
    if ev_df is None:
        raise ValueError("Either `events_in` or `history_events` must be provided to score_orders_batch.")
    dec_df = decision_events_in if decision_events_in is not None else decision_events

    orders_clean, events_clean, dec_clean, _ = validate_and_prepare_scoring_inputs(
        orders_in=orders_in,
        events_in=ev_df,
        decision_events_in=dec_df,
        as_of=as_of,
    )

    return_two_tuple = saved_model_payload is None and bundle is not None
    if saved_model_payload is not None:
        bundle = saved_model_payload["selected_bundle"]
        cal_method = str(saved_model_payload["selected_calibration_method"])
        mod_name = str(saved_model_payload["selected_model_name"])
        feature_version = str(
            saved_model_payload.get("selected_feature_version")
            or getattr(bundle, "feature_version", "extended_purchase_assumptions_v2")
        )
        feature_cols = list(
            saved_model_payload.get("selected_feature_columns")
            or getattr(bundle, "feature_cols", None)
            or get_features_for_version(feature_version)
        )
        fb_prior = float(saved_model_payload.get("fallback_prior", 0.068))
    else:
        cal_method = str(calibration_method or "isotonic")
        mod_name = str(model_name or getattr(bundle, "model_name", "selected_model"))
        feature_version = str(getattr(bundle, "feature_version", "extended_purchase_assumptions_v2"))
        feature_cols = list(getattr(bundle, "feature_cols", None) or get_features_for_version(feature_version))
        fb_prior = float(fallback_prior if fallback_prior is not None else 0.068)

    assert_zero_leakage_features(feature_cols)

    featured_in = build_purchase_time_features(
        master_orders=orders_clean,
        history_events=events_clean,
        label_col="late",
        smoothing_alpha=smoothing_alpha,
    )

    probs_raw = np.asarray(bundle.predict_proba_variant(featured_in[feature_cols], method="uncalibrated"), dtype=np.float64)
    probs_cal = np.asarray(bundle.predict_proba_variant(featured_in[feature_cols], method=cal_method), dtype=np.float64)

    has_items = (
        featured_in["has_items"].fillna(False).to_numpy(dtype=bool)
        if "has_items" in featured_in.columns
        else np.ones(len(featured_in), dtype=bool)
    )
    probs_cal = np.where(has_items, probs_cal, fb_prior)
    probs_raw = np.where(has_items, probs_raw, fb_prior)

    alloc_df = allocate_daily_alerts(
        df=featured_in,
        scores=probs_cal,
        primary_budget=primary_budget,
        secondary_budget=secondary_budget,
        decision_events=dec_clean,
        as_of=as_of,
    )
    daily_summary_df = summarize_daily_policy(
        df_orders=featured_in,
        alloc_df=alloc_df,
        primary_budget=primary_budget,
    )

    # Also compute global batch rank for backwards compatibility alongside daily policy rank
    global_ranked = rank_orders_deterministically(
        order_ids=featured_in["order_id"],
        scores=probs_cal,
        purchase_timestamps=featured_in["order_purchase_timestamp"],
    )

    out_df = pd.DataFrame(
        {
            "order_id": featured_in["order_id"].astype(str),
            "order_purchase_timestamp": featured_in["order_purchase_timestamp"],
            "order_estimated_delivery_date": featured_in["order_estimated_delivery_date"],
            "decision_date": alloc_df["decision_date"],
            "decision_cutoff_at": alloc_df["decision_cutoff_at"],
            "has_items": has_items,
            "is_eligible_for_warning": alloc_df["is_eligible_for_warning"],
            "eligibility_reason": alloc_df["eligibility_reason"],
            "scoring_mode": np.where(has_items, "full_purchase_time_model", "fallback_no_items"),
            "model_name": mod_name,
            "feature_version": feature_version,
            "calibration_method": cal_method,
            "predicted_delay_prob_uncalibrated_full": probs_raw,
            "predicted_delay_probability_full": probs_cal,
            "predicted_delay_prob_uncalibrated": np.round(probs_raw, 6),
            "predicted_delay_probability": np.round(probs_cal, 6),
            "daily_eligible_n": alloc_df["daily_eligible_n"],
            "daily_quota_q": alloc_df["daily_quota_q"],
            "daily_rank": alloc_df["daily_rank"],
            "daily_percentile": np.round(alloc_df["daily_percentile"], 6),
            "selected_top10": alloc_df["selected_top10"],
            "selected_top20": alloc_df["selected_top20"],
            "risk_rank": np.where(
                alloc_df["daily_rank"] > 0,
                alloc_df["daily_rank"],
                global_ranked["risk_rank"],
            ),
            "risk_percentile": np.round(
                np.where(
                    ~np.isnan(alloc_df["daily_percentile"]),
                    alloc_df["daily_percentile"],
                    global_ranked["risk_percentile"],
                ),
                6,
            ),
            "risk_tier": alloc_df["risk_tier"],
            "recommended_action": alloc_df["recommended_action"],
            "policy_version": alloc_df["policy_version"],
        }
    )

    meta = {
        "scored_at": now_iso(),
        "schema_version": SCORING_INPUT_V2_VERSION,
        "as_of": pd.Timestamp(as_of).isoformat() if as_of is not None else None,
        "model_name": mod_name,
        "feature_version": feature_version,
        "calibration_method": cal_method,
        "policy_version": str(alloc_df["policy_version"].iloc[0]) if len(alloc_df) > 0 else "daily_end_of_day_top10_floor_v2",
        "primary_budget_k": float(primary_budget),
        "input_orders_count": int(len(out_df)),
        "eligible_orders_count": int(alloc_df["is_eligible_for_warning"].sum()),
        "orders_with_items_count": int(np.sum(has_items)),
        "fallback_no_items_count": int(np.sum(~has_items)),
        "high_priority_top10_count": int(alloc_df["selected_top10"].sum()),
        "medium_watchlist_top20_count": int(
            np.sum(alloc_df["risk_tier"] == "MEDIUM_WATCHLIST_TOP20")
        ),
        "decision_days_count": int(len(daily_summary_df)),
    }
    if return_two_tuple:
        return out_df, meta
    return out_df, daily_summary_df, meta
