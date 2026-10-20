"""Purchase-time feature engineering, feature contract registry, and feature availability audit."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .history import attach_event_clock_history_features, attach_v3_route_features


PAYMENT_NUMERIC_FEATURES: list[str] = [
    "payment_value_sum",
    "payment_installments_max",
    "payment_sequential_count",
    "has_boleto",
    "has_credit_card",
    "has_voucher",
]

PAYMENT_CATEGORICAL_FEATURES: list[str] = [
    "primary_payment_type",
]

NUMERIC_FEATURES: list[str] = [
    # Order & basket structure
    "item_count",
    "distinct_product_count",
    "distinct_seller_count",
    "distinct_category_count",
    "is_multi_item",
    "is_multi_seller",
    # Value, freight, & payment structure
    "total_price",
    "max_item_price",
    "mean_item_price",
    "total_freight",
    "max_freight",
    "freight_ratio",
    "order_total_value",
    "payment_value_sum",
    "payment_installments_max",
    "payment_sequential_count",
    "has_boleto",
    "has_credit_card",
    "has_voucher",
    # Product physical dimensions
    "total_weight_kg",
    "max_weight_kg",
    "total_volume_cm3",
    "max_volume_cm3",
    "weight_missing",
    "volume_missing",
    "freight_per_kg",
    # Geography & route geometry
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
    "haversine_distance_km",
    "distance_km_max",
    "primary_distance_km",
    "same_state",
    "same_region",
    "any_cross_state_item",
    "geo_missing",
    # Promised delivery window & tightness
    "estimated_lead_calendar_days",
    "estimated_lead_Exact_days",
    "promised_Slack_days",
    "distance_per_promised_day",
    # Temporal & seasonal attributes at purchase
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "purchase_is_weekend",
    "purchase_month_sin",
    "purchase_month_cos",
    # Purchase-clock volume & congestion (strictly order_purchase_timestamp < query_ts)
    "seller_prior_purchase_count",
    "seller_purchase_count_7d",
    "seller_purchase_count_30d",
    "seller_load_ratio_7d_vs_30d",
    "customer_prior_order_count",
    # Completion-clock outcome history (strictly label_available_at < query_ts)
    "global_hist_completed_count",
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
]

CATEGORICAL_FEATURES: list[str] = [
    "customer_state",
    "customer_region",
    "primary_seller_state",
    "primary_seller_region",
    "route_region_pair",
    "primary_category",
    "primary_payment_type",
]

ALL_MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BASE_V2_NUMERIC_FEATURES: list[str] = [
    col for col in NUMERIC_FEATURES if col not in set(PAYMENT_NUMERIC_FEATURES)
]
BASE_V2_CATEGORICAL_FEATURES: list[str] = [
    col for col in CATEGORICAL_FEATURES if col not in set(PAYMENT_CATEGORICAL_FEATURES)
]
BASE_V2_ALL_FEATURES: list[str] = BASE_V2_NUMERIC_FEATURES + BASE_V2_CATEGORICAL_FEATURES

REQUIRED_FEATURE_BUILDER_INPUT_COLS: set[str] = {
    "order_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "primary_seller_id",
    "route_state_pair",
    "customer_state",
    "primary_category",
    "estimated_lead_calendar_days",
    "haversine_distance_km",
    "total_weight_kg",
    "total_freight",
    "purchase_month",
}

FORBIDDEN_POST_PURCHASE_COLUMNS: set[str] = {
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "label_available_at",
    "order_status",
    "in_cohort_a",
    "is_label_observable",
    "label_reason_code",
    "late",
    "late_calendar_day",
    "late_timestamp",
    "same_day_after_midnight",
    "delay_days_continuous",
    "delay_calendar_days",
    "review_score",
    "review_score_mean",
    "review_score_min",
    "review_count",
    "review_creation_date",
    "review_answer_timestamp",
    "shipping_limit_date",
    "min_shipping_limit_days",
    "max_shipping_limit_days",
    "approval_hours",
    "seller_to_carrier_days",
    "carrier_to_customer_days",
}


FEATURE_METADATA: dict[str, tuple[str, str, str, str]] = {
    # (feature_group, source_tables, event_clock_type, availability_rule)
    "item_count": ("basket_structure", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "distinct_product_count": ("basket_structure", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "distinct_seller_count": ("basket_structure", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "distinct_category_count": ("basket_structure", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "is_multi_item": ("basket_structure", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "is_multi_seller": ("basket_structure", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "total_price": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "max_item_price": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "mean_item_price": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "total_freight": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "max_freight": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "freight_ratio": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "order_total_value": ("value_and_payment", "items", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "payment_value_sum": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
    "payment_installments_max": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
    "payment_sequential_count": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
    "has_boleto": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
    "has_credit_card": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
    "has_voucher": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
    "total_weight_kg": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "max_weight_kg": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "total_volume_cm3": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "max_volume_cm3": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "weight_missing": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "volume_missing": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "freight_per_kg": ("physical_dimensions", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "customer_lat": ("geography_and_route", "customers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "customer_lng": ("geography_and_route", "customers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "seller_lat": ("geography_and_route", "items,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "seller_lng": ("geography_and_route", "items,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "haversine_distance_km": ("geography_and_route", "customers,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "distance_km_max": ("geography_and_route", "customers,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "primary_distance_km": ("geography_and_route", "customers,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "same_state": ("geography_and_route", "customers,sellers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "same_region": ("geography_and_route", "customers,sellers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "any_cross_state_item": ("geography_and_route", "customers,sellers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "geo_missing": ("geography_and_route", "customers,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "estimated_lead_calendar_days": ("promised_schedule", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "estimated_lead_Exact_days": ("promised_schedule", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "promised_Slack_days": ("promised_schedule", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "distance_per_promised_day": ("promised_schedule", "orders,customers,sellers,geolocation", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "purchase_month": ("temporal_seasonality", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "purchase_day_of_week": ("temporal_seasonality", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "purchase_hour": ("temporal_seasonality", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "purchase_is_weekend": ("temporal_seasonality", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "purchase_month_sin": ("temporal_seasonality", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "purchase_month_cos": ("temporal_seasonality", "orders", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "seller_prior_purchase_count": ("purchase_clock_congestion", "orders,items", "purchase_clock_strictly_prior", "prior.order_purchase_timestamp < current.order_purchase_timestamp (past.order_id != current.order_id)"),
    "seller_purchase_count_7d": ("purchase_clock_congestion", "orders,items", "purchase_clock_strictly_prior", "current.order_purchase_timestamp - 7d <= prior.order_purchase_timestamp < current.order_purchase_timestamp"),
    "seller_purchase_count_30d": ("purchase_clock_congestion", "orders,items", "purchase_clock_strictly_prior", "current.order_purchase_timestamp - 30d <= prior.order_purchase_timestamp < current.order_purchase_timestamp"),
    "seller_load_ratio_7d_vs_30d": ("purchase_clock_congestion", "orders,items", "purchase_clock_strictly_prior", "prior.order_purchase_timestamp < current.order_purchase_timestamp"),
    "customer_prior_order_count": ("purchase_clock_congestion", "orders,customers", "purchase_clock_strictly_prior", "prior.order_purchase_timestamp < current.order_purchase_timestamp"),
    "global_hist_completed_count": ("completion_clock_history", "orders", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "global_hist_late_rate": ("completion_clock_history", "orders", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "global_recent_30d_late_rate": ("completion_clock_history", "orders", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 30d <= prior.label_available_at < current.order_purchase_timestamp"),
    "seller_completed_count": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "seller_completed_count_30d": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 30d <= prior.label_available_at < current.order_purchase_timestamp"),
    "seller_completed_count_90d": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 90d <= prior.label_available_at < current.order_purchase_timestamp"),
    "seller_is_cold_start": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "seller_hist_late_rate_smoothed": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "seller_recent_30d_late_rate_smoothed": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 30d <= prior.label_available_at < current.order_purchase_timestamp"),
    "seller_recent_90d_late_rate_smoothed": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 90d <= prior.label_available_at < current.order_purchase_timestamp"),
    "seller_hist_mean_delay_days": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "seller_excess_late_vs_global": ("completion_clock_history", "orders,items", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "route_completed_count": ("completion_clock_history", "orders,items,customers,sellers", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "route_hist_late_rate_smoothed": ("completion_clock_history", "orders,items,customers,sellers", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "route_recent_60d_late_rate_smoothed": ("completion_clock_history", "orders,items,customers,sellers", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 60d <= prior.label_available_at < current.order_purchase_timestamp"),
    "route_hist_mean_delay_days": ("completion_clock_history", "orders,items,customers,sellers", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "route_excess_late_vs_global": ("completion_clock_history", "orders,items,customers,sellers", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "cust_state_completed_count": ("completion_clock_history", "orders,customers", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "cust_state_hist_late_rate_smoothed": ("completion_clock_history", "orders,customers", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "cust_state_recent_30d_late_rate_smoothed": ("completion_clock_history", "orders,customers", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 30d <= prior.label_available_at < current.order_purchase_timestamp"),
    "category_completed_count": ("completion_clock_history", "orders,items,products", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "category_hist_late_rate_smoothed": ("completion_clock_history", "orders,items,products", "completion_clock_strictly_prior", "prior.label_available_at < current.order_purchase_timestamp"),
    "route_recent_60d_late_rate_ca": ("completion_clock_history", "orders,items,customers,sellers", "known_clock_strictly_prior", "current.order_purchase_timestamp - 60d <= prior.label_known_at < current.order_purchase_timestamp, label_known_at = min(delivery, promised day + 1) for late orders"),
    "route_recent_60d_mean_delivery_days": ("completion_clock_history", "orders,items,customers,sellers", "completion_clock_strictly_prior", "current.order_purchase_timestamp - 60d <= prior.label_available_at < current.order_purchase_timestamp"),
    "promise_slack_days_60d": ("promised_schedule", "orders,items,customers,sellers", "completion_clock_strictly_prior", "estimated_lead_calendar_days - route_recent_60d_mean_delivery_days"),
    "customer_state": ("geography_and_route", "customers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "customer_region": ("geography_and_route", "customers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "primary_seller_state": ("geography_and_route", "items,sellers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "primary_seller_region": ("geography_and_route", "items,sellers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "route_region_pair": ("geography_and_route", "customers,sellers", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "primary_category": ("basket_structure", "items,products", "static_at_purchase", "available_at == order_purchase_timestamp"),
    "primary_payment_type": ("value_and_payment", "payments", "static_at_purchase", "available_at == order_purchase_timestamp (extended_purchase_assumptions_v2)"),
}


BASE_V2_FEATURES: list[str] = BASE_V2_ALL_FEATURES

# v3 additions (pre-registered in configs/experiment-v3*.json): route timeliness and promise slack. They are kept
# out of ALL_MODEL_FEATURES so the v2 feature versions, and therefore v2 runs, are unchanged.
V3_EXTRA_NUMERIC_FEATURES: list[str] = [
    "route_recent_60d_late_rate_ca",
    "route_recent_60d_mean_delivery_days",
    "promise_slack_days_60d",
]
BASE_V3_NUMERIC_FEATURES: list[str] = BASE_V2_NUMERIC_FEATURES + V3_EXTRA_NUMERIC_FEATURES
BASE_V3_CATEGORICAL_FEATURES: list[str] = list(BASE_V2_CATEGORICAL_FEATURES)
BASE_V3_ALL_FEATURES: list[str] = BASE_V3_NUMERIC_FEATURES + BASE_V3_CATEGORICAL_FEATURES

# Every column a model or rule may read; feature versions select subsets of it.
MODEL_INPUT_COLUMNS: list[str] = ALL_MODEL_FEATURES + V3_EXTRA_NUMERIC_FEATURES


def get_feature_columns(
    feature_version: str = "extended_purchase_assumptions_v2",
) -> tuple[list[str], list[str], list[str]]:
    """Return `(numeric_features, categorical_features, all_features)` for the requested feature version."""
    if feature_version == "base_v2":
        assert_zero_leakage_features(BASE_V2_ALL_FEATURES)
        return list(BASE_V2_NUMERIC_FEATURES), list(BASE_V2_CATEGORICAL_FEATURES), list(BASE_V2_ALL_FEATURES)
    if feature_version == "base_v3":
        assert_zero_leakage_features(BASE_V3_ALL_FEATURES)
        return list(BASE_V3_NUMERIC_FEATURES), list(BASE_V3_CATEGORICAL_FEATURES), list(BASE_V3_ALL_FEATURES)
    if feature_version in ("extended_purchase_assumptions_v2", "v1"):
        assert_zero_leakage_features(ALL_MODEL_FEATURES)
        return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES), list(ALL_MODEL_FEATURES)
    raise ValueError(
        f"Unsupported feature_version: {feature_version!r}. "
        "Expected 'base_v2', 'extended_purchase_assumptions_v2' or 'base_v3'."
    )


def get_features_for_version(feature_version: str = "extended_purchase_assumptions_v2") -> list[str]:
    """Return the combined feature column list for `feature_version`."""
    return get_feature_columns(feature_version)[2]


def get_feature_columns_by_type(
    feature_version: str = "extended_purchase_assumptions_v2",
) -> tuple[list[str], list[str]]:
    """Return `(numeric_features, categorical_features)` for `feature_version`."""
    num_cols, cat_cols, _ = get_feature_columns(feature_version)
    return num_cols, cat_cols


def assert_zero_leakage_features(feature_cols: list[str] | set[str]) -> None:
    """Fail immediately if any post-purchase or target column appears in model feature list."""
    leaked = set(feature_cols).intersection(FORBIDDEN_POST_PURCHASE_COLUMNS)
    if leaked:
        raise ValueError(
            f"Forbidden post-purchase or target columns detected in model features: {sorted(leaked)}"
        )


def validate_feature_builder_inputs(master_orders: pd.DataFrame) -> None:
    """Validate that `master_orders` contains all required master/scoring columns before building features."""
    missing = REQUIRED_FEATURE_BUILDER_INPUT_COLS - set(master_orders.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required master/scoring columns for feature engineering: {sorted(missing)}"
        )


def build_purchase_time_features(
    master_orders: pd.DataFrame,
    history_events: pd.DataFrame | None = None,
    label_col: str = "late",
    smoothing_m: float = 5.0,
    default_prior: float = 0.068,
    smoothing_alpha: float | None = None,
    seller_bridge_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach both static purchase-time engineered ratios and causal dual-clock historical features."""
    assert_zero_leakage_features(ALL_MODEL_FEATURES)
    validate_feature_builder_inputs(master_orders)

    if smoothing_alpha is not None:
        smoothing_m = float(smoothing_alpha)

    df = attach_event_clock_history_features(
        target_orders=master_orders,
        history_events=history_events,
        label_col=label_col,
        smoothing_m=smoothing_m,
        default_prior=default_prior,
        seller_bridge_events=seller_bridge_events,
    )

    df = attach_v3_route_features(
        target_orders=df,
        history_events=history_events,
        label_col=label_col,
        smoothing_m=smoothing_m,
        default_prior=default_prior,
    )

    # Derived static ratios at purchase time
    lead_safe = np.maximum(df["estimated_lead_calendar_days"].fillna(1.0), 1.0)
    df["distance_per_promised_day"] = df["haversine_distance_km"] / lead_safe

    weight_safe = np.maximum(df["total_weight_kg"].fillna(0.5), 0.05)
    df["freight_per_kg"] = np.where(
        df["total_weight_kg"].notna() & (df["total_weight_kg"] > 0),
        df["total_freight"] / weight_safe,
        np.nan,
    )

    month_rad = 2.0 * np.pi * (df["purchase_month"].astype(float) - 1.0) / 12.0
    df["purchase_month_sin"] = np.sin(month_rad)
    df["purchase_month_cos"] = np.cos(month_rad)

    df["seller_load_ratio_7d_vs_30d"] = (
        df["seller_purchase_count_7d"] * (30.0 / 7.0)
    ) / np.maximum(df["seller_purchase_count_30d"], 1.0)

    df["seller_excess_late_vs_global"] = (
        df["seller_hist_late_rate_smoothed"] - df["global_hist_late_rate"]
    )
    df["route_excess_late_vs_global"] = (
        df["route_hist_late_rate_smoothed"] - df["global_hist_late_rate"]
    )

    # Ensure all categorical features present are clean strings without NaNs
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
        else:
            df[col] = "unknown"

    return df


def generate_feature_availability_audit(featured_orders: pd.DataFrame) -> pd.DataFrame:
    """Generate the formal feature availability & leakage audit table (`feature_availability_audit.csv`)."""
    assert_zero_leakage_features(ALL_MODEL_FEATURES)

    cohort_a = featured_orders[featured_orders["in_cohort_a"]]
    base_v2_set = set(BASE_V2_ALL_FEATURES)
    base_v3_set = set(BASE_V3_ALL_FEATURES)
    extended_set = set(ALL_MODEL_FEATURES)
    rows: list[dict[str, Any]] = []
    for col in MODEL_INPUT_COLUMNS:
        if col not in featured_orders.columns:
            raise ValueError(f"Model feature '{col}' missing from featured_orders DataFrame")
        group, sources, clock_type, rule = FEATURE_METADATA[col]
        is_cat = col in CATEGORICAL_FEATURES
        miss_a = float(cohort_a[col].isna().mean()) if len(cohort_a) > 0 else 0.0
        miss_b = float(featured_orders[col].isna().mean()) if len(featured_orders) > 0 else 0.0
        in_forbidden = col in FORBIDDEN_POST_PURCHASE_COLUMNS
        rows.append(
            {
                "feature_name": col,
                "feature_type": "categorical" if is_cat else "numeric",
                "feature_group": group,
                "source_tables": sources,
                "event_clock_type": clock_type,
                "availability_rule": rule,
                "in_base_v2": col in base_v2_set,
                "in_extended_purchase_assumptions_v2": col in extended_set,
                "in_base_v3": col in base_v3_set,
                "post_purchase_leakage_risk": "HIGH" if in_forbidden else "NONE",
                "missing_rate_cohort_a": round(miss_a, 6),
                "missing_rate_path_b": round(miss_b, 6),
                "imputation_strategy": "constant_unknown_category" if is_cat else "fold_train_median_with_missing_flag",
                "audit_passed": not in_forbidden,
            }
        )
    return pd.DataFrame(rows)
