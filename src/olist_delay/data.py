"""Source CSV loading, schema validation, order-grain aggregation, and cohort reconciliation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geography import (
    DEFAULT_BRAZIL_LAT_BOUNDS,
    DEFAULT_BRAZIL_LNG_BOUNDS,
    build_geolocation_lookup,
    haversine_km,
    map_state_to_region,
    resolve_coordinates_with_fallback,
)
from .labels import compute_order_labels, summarize_label_reconciliation


REQUIRED_SCHEMAS: dict[str, set[str]] = {
    "orders": {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "items": {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    },
    "customers": {
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    },
    "sellers": {
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    },
    "products": {
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    },
    "payments": {
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    },
    "reviews": {
        "review_id",
        "order_id",
        "review_score",
    },
    "geolocation": {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    },
    "category_translation": {
        "product_category_name",
        "product_category_name_english",
    },
}

DEFAULT_FILENAMES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def validate_table_schema(df: pd.DataFrame, table_key: str) -> None:
    expected = REQUIRED_SCHEMAS[table_key]
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Table '{table_key}' is missing required columns: {sorted(missing)}")


def load_raw_tables(data_dir: Path, filenames: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    fmap = dict(DEFAULT_FILENAMES)
    if filenames:
        fmap.update(filenames)

    tables: dict[str, pd.DataFrame] = {}
    for key, fname in fmap.items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required dataset file for '{key}' not found: {fpath}")
        df = pd.read_csv(fpath, low_memory=False)
        validate_table_schema(df, key)
        tables[key] = df

    # Parse timestamps strictly on orders and items
    order_time_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in order_time_cols:
        raw_non_null = int(tables["orders"][col].notna().sum())
        parsed = pd.to_datetime(tables["orders"][col], errors="coerce")
        parsed_non_null = int(parsed.notna().sum())
        if parsed_non_null < raw_non_null:
            raise ValueError(
                f"Failed to parse {raw_non_null - parsed_non_null} non-null timestamps in orders.{col}"
            )
        tables["orders"][col] = parsed

    raw_ship = int(tables["items"]["shipping_limit_date"].notna().sum())
    parsed_ship = pd.to_datetime(tables["items"]["shipping_limit_date"], errors="coerce")
    if int(parsed_ship.notna().sum()) < raw_ship:
        raise ValueError("Failed to parse non-null timestamps in items.shipping_limit_date")
    tables["items"]["shipping_limit_date"] = parsed_ship

    return tables


def build_data_quality_exceptions(master_orders: pd.DataFrame) -> pd.DataFrame:
    """Export all excluded/unlabeled orders and any timestamp/price/weight/volume anomaly rows."""
    anom_cols = [
        "anomaly_negative_estimated_lead",
        "anomaly_delivered_before_purchase",
        "anomaly_carrier_before_purchase",
        "anomaly_delivered_before_carrier",
        "anomaly_zero_or_negative_price",
    ]
    has_any_anom = pd.Series(False, index=master_orders.index)
    for col in anom_cols:
        if col in master_orders.columns:
            has_any_anom = has_any_anom | master_orders[col].fillna(False).astype(bool)

    mask = (~master_orders["in_cohort_a"]) | has_any_anom | (master_orders["weight_missing"].fillna(0) == 1)
    cols = [
        "order_id",
        "order_status",
        "has_items",
        "in_cohort_a",
        "label_reason_code",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
        "weight_missing",
        "volume_missing",
        *anom_cols,
    ]
    present_cols = [c for c in cols if c in master_orders.columns]
    return master_orders.loc[mask, present_cols].reset_index(drop=True)


def build_cohort_observation_analysis(
    master_orders: pd.DataFrame,
    selection_at: str = "2018-05-26 18:18:03",
) -> pd.DataFrame:
    """Summarize monthly Cohort A vs Path B vs matured-at-selection-time counts and late rates."""
    df = master_orders.copy()
    sel_ts = pd.Timestamp(selection_at)
    ym = df["order_purchase_timestamp"].dt.strftime("%Y-%m").fillna("unknown")
    df["_ym"] = ym

    rows: list[dict[str, Any]] = []
    for month_str, sub in df.groupby("_ym", sort=True):
        path_b_n = int(len(sub))
        with_items_n = int(sub["has_items"].sum())
        coh_a = sub[sub["in_cohort_a"]]
        coh_a_n = int(len(coh_a))
        excluded_n = path_b_n - coh_a_n

        matured = coh_a[coh_a["label_available_at"].notna() & (coh_a["label_available_at"] <= sel_ts)]
        unmatured = coh_a[coh_a["label_available_at"].isna() | (coh_a["label_available_at"] > sel_ts)]

        coh_a_late_cal = int(coh_a["late_calendar_day"].sum()) if coh_a_n > 0 else 0
        coh_a_late_ts = int(coh_a["late_timestamp"].sum()) if coh_a_n > 0 else 0
        mat_n = int(len(matured))
        mat_late_cal = int(matured["late_calendar_day"].sum()) if mat_n > 0 else 0
        unmat_n = int(len(unmatured))
        unmat_late_cal = int(unmatured["late_calendar_day"].sum()) if unmat_n > 0 else 0

        rows.append(
            {
                "purchase_year_month": str(month_str),
                "path_b_total_orders": path_b_n,
                "orders_with_items": with_items_n,
                "cohort_a_delivered_orders": coh_a_n,
                "excluded_unlabeled_orders": excluded_n,
                "cohort_a_coverage_rate": round(coh_a_n / max(1, path_b_n), 6),
                "cohort_a_late_calendar_day_positives": coh_a_late_cal,
                "cohort_a_late_calendar_day_rate": round(coh_a_late_cal / max(1, coh_a_n), 6) if coh_a_n > 0 else np.nan,
                "cohort_a_late_timestamp_positives": coh_a_late_ts,
                "cohort_a_late_timestamp_rate": round(coh_a_late_ts / max(1, coh_a_n), 6) if coh_a_n > 0 else np.nan,
                "matured_by_selection_at_count": mat_n,
                "matured_by_selection_at_late_positives": mat_late_cal,
                "matured_by_selection_at_late_rate": round(mat_late_cal / max(1, mat_n), 6) if mat_n > 0 else np.nan,
                "unmatured_by_selection_at_count": unmat_n,
                "unmatured_by_selection_at_late_positives": unmat_late_cal,
                "unmatured_by_selection_at_late_rate": round(unmat_late_cal / max(1, unmat_n), 6) if unmat_n > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_master_order_table(
    tables: dict[str, pd.DataFrame],
    primary_label: str = "calendar_day",
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the 1:1 order-level master table, item-seller event table, and cohort audit summary.

    Returns:
      - master_orders: 99,441 rows (1 row per order_id), covering both Path B (all orders) and Path A (`in_cohort_a`).
      - order_items_enriched: item-level table with seller/product/route info and order outcomes for history builders.
      - audit_summary: structured dictionary reconciling row counts, missingness, anomalies, and labels.
    """
    orders_raw = tables["orders"].copy()
    if orders_raw["order_id"].duplicated().any():
        raise ValueError("Duplicate order_id detected in orders dataset!")

    items = tables["items"].copy()
    orders_with_items_set = set(items["order_id"].dropna())
    orders_raw["has_items"] = orders_raw["order_id"].isin(orders_with_items_set)

    orders = compute_order_labels(orders_raw, primary_label=primary_label)

    customers = tables["customers"].copy()
    if customers["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id detected in customers dataset!")

    sellers = tables["sellers"].copy()
    products = tables["products"].copy()
    translations = tables["category_translation"].copy()
    payments = tables["payments"].copy()
    reviews = tables["reviews"].copy()

    geo_cfg = (config or {}).get("geolocation", {})
    lat_bounds = tuple(geo_cfg.get("lat_bounds", DEFAULT_BRAZIL_LAT_BOUNDS))
    lng_bounds = tuple(geo_cfg.get("lng_bounds", DEFAULT_BRAZIL_LNG_BOUNDS))
    dedup_geo = bool(geo_cfg.get("deduplicate_exact_coords", True))

    geo_lookup = build_geolocation_lookup(
        tables["geolocation"],
        lat_bounds=(float(lat_bounds[0]), float(lat_bounds[1])),
        lng_bounds=(float(lng_bounds[0]), float(lng_bounds[1])),
        deduplicate_exact_coords=dedup_geo,
    )

    # 1. Enrich products with English categories and physical volume
    prod_cat = products.merge(translations, on="product_category_name", how="left")
    prod_cat["category_en"] = (
        prod_cat["product_category_name_english"]
        .fillna(prod_cat["product_category_name"])
        .fillna("unknown")
    )
    prod_cat["weight_kg"] = prod_cat["product_weight_g"] / 1000.0
    prod_cat["volume_cm3"] = (
        prod_cat["product_length_cm"]
        * prod_cat["product_height_cm"]
        * prod_cat["product_width_cm"]
    )
    prod_cat["weight_missing_flag"] = prod_cat["weight_kg"].isna().astype(int)
    prod_cat["volume_missing_flag"] = prod_cat["volume_cm3"].isna().astype(int)

    # 2. Enrich sellers with coordinates and region
    sellers["seller_region"] = map_state_to_region(sellers["seller_state"])
    seller_coords = resolve_coordinates_with_fallback(
        sellers,
        zip_col="seller_zip_code_prefix",
        state_col="seller_state",
        prefix="seller",
        geo_lookup=geo_lookup,
    )
    sellers = pd.concat([sellers, seller_coords], axis=1)

    # 3. Enrich customers with coordinates and region
    customers["customer_region"] = map_state_to_region(customers["customer_state"])
    cust_coords = resolve_coordinates_with_fallback(
        customers,
        zip_col="customer_zip_code_prefix",
        state_col="customer_state",
        prefix="customer",
        geo_lookup=geo_lookup,
    )
    customers = pd.concat([customers, cust_coords], axis=1)

    # 4. Merge orders + customers (1:1 grain check)
    master = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
                "customer_region",
                "customer_lat",
                "customer_lng",
                "customer_geo_fallback",
                "customer_geo_missing_zip",
            ]
        ],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # 5. Enrich items with product + seller + order customer coords
    items_full = (
        items.merge(
            prod_cat[
                [
                    "product_id",
                    "category_en",
                    "weight_kg",
                    "volume_cm3",
                    "weight_missing_flag",
                    "volume_missing_flag",
                ]
            ],
            on="product_id",
            how="left",
        )
        .merge(
            sellers[
                [
                    "seller_id",
                    "seller_zip_code_prefix",
                    "seller_city",
                    "seller_state",
                    "seller_region",
                    "seller_lat",
                    "seller_lng",
                    "seller_geo_fallback",
                    "seller_geo_missing_zip",
                ]
            ],
            on="seller_id",
            how="left",
        )
        .merge(
            master[
                [
                    "order_id",
                    "order_purchase_timestamp",
                    "order_estimated_delivery_date",
                    "order_delivered_customer_date",
                    "label_available_at",
                    "in_cohort_a",
                    "late_calendar_day",
                    "late_timestamp",
                    "late",
                    "delay_calendar_days",
                    "customer_state",
                    "customer_region",
                    "customer_lat",
                    "customer_lng",
                ]
            ],
            on="order_id",
            how="left",
        )
    )
    items_full["item_distance_km"] = haversine_km(
        items_full["seller_lat"],
        items_full["seller_lng"],
        items_full["customer_lat"],
        items_full["customer_lng"],
    )
    items_full["item_same_state"] = (items_full["seller_state"] == items_full["customer_state"]).astype(int)
    items_full["item_same_region"] = (items_full["seller_region"] == items_full["customer_region"]).astype(int)
    items_full["shipping_limit_days"] = (
        (items_full["shipping_limit_date"] - items_full["order_purchase_timestamp"]).dt.total_seconds()
        / 86400.0
    )

    # 6. Deterministic primary item selection per order:
    # highest price -> highest freight_value -> smallest order_item_id
    items_sorted = items_full.sort_values(
        by=["order_id", "price", "freight_value", "order_item_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    primary_items = items_sorted.drop_duplicates(subset=["order_id"], keep="first")[
        [
            "order_id",
            "seller_id",
            "seller_state",
            "seller_region",
            "seller_zip_code_prefix",
            "seller_lat",
            "seller_lng",
            "seller_geo_fallback",
            "seller_geo_missing_zip",
            "category_en",
            "item_distance_km",
        ]
    ].rename(
        columns={
            "seller_id": "primary_seller_id",
            "seller_state": "primary_seller_state",
            "seller_region": "primary_seller_region",
            "seller_zip_code_prefix": "primary_seller_zip_prefix",
            "category_en": "primary_category",
            "item_distance_km": "primary_distance_km",
        }
    )

    # Deduplicated (order_id, seller_id) bridge represented as pipe-delimited string per order
    order_seller_dedup = (
        items_full[["order_id", "seller_id"]]
        .dropna(subset=["order_id", "seller_id"])
        .drop_duplicates(subset=["order_id", "seller_id"])
        .sort_values(["order_id", "seller_id"], kind="mergesort")
    )
    dup_order_mask = order_seller_dedup["order_id"].duplicated(keep=False)
    single_seller_df = order_seller_dedup.loc[~dup_order_mask, ["order_id", "seller_id"]].rename(
        columns={"seller_id": "seller_ids_pipe"}
    )
    if bool(dup_order_mask.any()):
        multi_seller_df = (
            order_seller_dedup.loc[dup_order_mask]
            .groupby("order_id", as_index=False)
            .agg(seller_ids_pipe=("seller_id", "|".join))
        )
        seller_pipe_df = pd.concat([single_seller_df, multi_seller_df], ignore_index=True)
    else:
        seller_pipe_df = single_seller_df.reset_index(drop=True)

    # 7. Order-level aggregation of items (fully Cythonized without Python lambdas)
    items_full["_cross_state"] = (items_full["item_same_state"] == 0).astype(int)
    item_agg = items_full.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        distinct_product_count=("product_id", "nunique"),
        distinct_seller_count=("seller_id", "nunique"),
        distinct_category_count=("category_en", "nunique"),
        total_price=("price", "sum"),
        max_item_price=("price", "max"),
        mean_item_price=("price", "mean"),
        total_freight=("freight_value", "sum"),
        max_freight=("freight_value", "max"),
        total_weight_kg=("weight_kg", "sum"),
        _weight_valid_cnt=("weight_kg", "count"),
        max_weight_kg=("weight_kg", "max"),
        total_volume_cm3=("volume_cm3", "sum"),
        _vol_valid_cnt=("volume_cm3", "count"),
        max_volume_cm3=("volume_cm3", "max"),
        weight_missing=("weight_missing_flag", "max"),
        volume_missing=("volume_missing_flag", "max"),
        distance_km_mean=("item_distance_km", "mean"),
        distance_km_max=("item_distance_km", "max"),
        min_shipping_limit_days=("shipping_limit_days", "min"),
        max_shipping_limit_days=("shipping_limit_days", "max"),
        any_cross_state_item=("_cross_state", "max"),
    )
    item_agg.loc[item_agg["_weight_valid_cnt"] == 0, "total_weight_kg"] = np.nan
    item_agg.loc[item_agg["_vol_valid_cnt"] == 0, "total_volume_cm3"] = np.nan
    item_agg.drop(columns=["_weight_valid_cnt", "_vol_valid_cnt"], inplace=True)

    # 8. Order-level aggregation of payments (fully Cythonized without Python lambdas)
    payments_sorted = payments.sort_values(
        by=["order_id", "payment_value", "payment_sequential"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    primary_pay = payments_sorted.drop_duplicates(subset=["order_id"], keep="first")[
        ["order_id", "payment_type"]
    ].rename(columns={"payment_type": "primary_payment_type"})

    payments["_is_boleto"] = (payments["payment_type"] == "boleto").astype(int)
    payments["_is_cc"] = (payments["payment_type"] == "credit_card").astype(int)
    payments["_is_voucher"] = (payments["payment_type"] == "voucher").astype(int)
    pay_agg = payments.groupby("order_id", as_index=False).agg(
        payment_value_sum=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
        payment_sequential_count=("payment_sequential", "count"),
        has_boleto=("_is_boleto", "max"),
        has_credit_card=("_is_cc", "max"),
        has_voucher=("_is_voucher", "max"),
    )

    # 9. Post-delivery review summary (diagnostic only — never used in purchase-time features)
    rev_agg = reviews.groupby("order_id", as_index=False).agg(
        review_score_mean=("review_score", "mean"),
        review_score_min=("review_score", "min"),
        review_count=("review_id", "count"),
    )

    # 10. Combine onto 1:1 master table
    master = (
        master.merge(item_agg, on="order_id", how="left", validate="one_to_one")
        .merge(primary_items, on="order_id", how="left", validate="one_to_one")
        .merge(seller_pipe_df, on="order_id", how="left", validate="one_to_one")
        .merge(pay_agg, on="order_id", how="left", validate="one_to_one")
        .merge(primary_pay, on="order_id", how="left", validate="one_to_one")
        .merge(rev_agg, on="order_id", how="left", validate="one_to_one")
    )

    master["has_items"] = master["item_count"].notna() & (master["item_count"] > 0)
    master["item_count"] = master["item_count"].fillna(0).astype(int)
    master["distinct_product_count"] = master["distinct_product_count"].fillna(0).astype(int)
    master["distinct_seller_count"] = master["distinct_seller_count"].fillna(0).astype(int)
    master["distinct_category_count"] = master["distinct_category_count"].fillna(0).astype(int)
    master["is_multi_item"] = (master["item_count"] > 1).astype(int)
    master["is_multi_seller"] = (master["distinct_seller_count"] > 1).astype(int)

    master["primary_seller_id"] = master["primary_seller_id"].fillna("NO_SELLER")
    master["seller_ids_pipe"] = master["seller_ids_pipe"].fillna("NO_SELLER")
    master["primary_seller_state"] = master["primary_seller_state"].fillna("UNK")
    master["primary_seller_region"] = master["primary_seller_region"].fillna("Unknown")
    master["primary_category"] = master["primary_category"].fillna("unknown")
    master["primary_payment_type"] = master["primary_payment_type"].fillna("unknown")
    master["seller_geo_fallback"] = master["seller_geo_fallback"].fillna("missing_seller")
    master["seller_geo_missing_zip"] = master["seller_geo_missing_zip"].fillna(1).astype(int)
    master["geo_missing"] = (
        (master["customer_geo_missing_zip"] == 1) | (master["seller_geo_missing_zip"] == 1)
    ).astype(int)

    master["freight_ratio"] = np.where(
        master["total_price"] > 0,
        master["total_freight"] / master["total_price"],
        np.nan,
    )
    master["order_total_value"] = master["total_price"].fillna(0.0) + master["total_freight"].fillna(0.0)
    master["haversine_distance_km"] = master["distance_km_mean"]
    master["same_state"] = np.where(
        master["has_items"],
        (master["customer_state"] == master["primary_seller_state"]).astype(int),
        0,
    )
    master["same_region"] = np.where(
        master["has_items"],
        (master["customer_region"] == master["primary_seller_region"]).astype(int),
        0,
    )
    master["route_state_pair"] = master["primary_seller_state"] + "->" + master["customer_state"]
    master["route_region_pair"] = master["primary_seller_region"] + "->" + master["customer_region"]

    # Promised delivery window at purchase time
    # Note: order_estimated_delivery_date is stored at midnight (00:00:00) of the promised calendar date.
    est_day = master["order_estimated_delivery_date"].dt.normalize()
    purch_day = master["order_purchase_timestamp"].dt.normalize()
    master["estimated_lead_calendar_days"] = (est_day - purch_day).dt.total_seconds() / 86400.0
    master["estimated_lead_Exact_days"] = (
        master["order_estimated_delivery_date"] - master["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    # Effective promised horizon to end-of-estimated-calendar-day (23:59:59)
    master["promised_Slack_days"] = master["estimated_lead_Exact_days"] + 1.0

    # Temporal purchase attributes
    master["purchase_year"] = master["order_purchase_timestamp"].dt.year
    master["purchase_month"] = master["order_purchase_timestamp"].dt.month
    master["purchase_day_of_week"] = master["order_purchase_timestamp"].dt.dayofweek
    master["purchase_hour"] = master["order_purchase_timestamp"].dt.hour
    master["purchase_is_weekend"] = (master["purchase_day_of_week"] >= 5).astype(int)
    master["purchase_year_month"] = master["order_purchase_timestamp"].dt.strftime("%Y-%m")

    # Anomaly detection flags for data quality audit
    master["anomaly_negative_estimated_lead"] = (master["estimated_lead_calendar_days"] < 0).fillna(False)
    master["anomaly_delivered_before_purchase"] = (
        master["order_delivered_customer_date"] < master["order_purchase_timestamp"]
    ).fillna(False)
    master["anomaly_carrier_before_purchase"] = (
        master["order_delivered_carrier_date"] < master["order_purchase_timestamp"]
    ).fillna(False)
    master["anomaly_delivered_before_carrier"] = (
        master["order_delivered_customer_date"] < master["order_delivered_carrier_date"]
    ).fillna(False)
    master["anomaly_zero_or_negative_price"] = (master["total_price"] <= 0).fillna(False)

    if len(master) != len(orders):
        raise AssertionError(f"Master table row count {len(master)} != source orders {len(orders)}")

    label_reconciliation = summarize_label_reconciliation(master)
    audit_summary = {
        "row_reconciliation": {
            "source_orders_total": int(len(orders)),
            "master_orders_total": int(len(master)),
            "unexplained_row_delta": int(len(orders) - len(master)),
            "path_b_all_orders": int(len(master)),
            "path_b_orders_with_items": int(master["has_items"].sum()),
            "path_b_orders_without_items": int((~master["has_items"]).sum()),
            "path_a_delivered_evaluable_orders": int(master["in_cohort_a"].sum()),
            "path_a_excluded_orders": int((~master["in_cohort_a"]).sum()),
            "all_path_a_orders_have_items": bool(master.loc[master["in_cohort_a"], "has_items"].all()),
        },
        "label_reconciliation": label_reconciliation,
        "status_breakdown": (
            master.groupby(["order_status", "has_items", "in_cohort_a"], dropna=False)
            .size()
            .reset_index(name="order_count")
            .to_dict(orient="records")
        ),
        "geo_fallback_counts": {
            "customer": master["customer_geo_fallback"].value_counts().to_dict(),
            "seller": master["seller_geo_fallback"].value_counts().to_dict(),
            "out_of_bounds_geolocation_rows_dropped": int(geo_lookup.get("out_of_bounds_dropped_count", 0)),  # type: ignore[arg-type]
        },
        "anomaly_counts": {
            "negative_estimated_lead": int(master["anomaly_negative_estimated_lead"].sum()),
            "delivered_before_purchase": int(master["anomaly_delivered_before_purchase"].sum()),
            "carrier_before_purchase": int(master["anomaly_carrier_before_purchase"].sum()),
            "delivered_before_carrier": int(master["anomaly_delivered_before_carrier"].sum()),
            "zero_or_negative_price": int(master["anomaly_zero_or_negative_price"].sum()),
            "missing_weight_in_cohort_a": int(master.loc[master["in_cohort_a"], "weight_missing"].fillna(1).sum()),
            "missing_volume_in_cohort_a": int(master.loc[master["in_cohort_a"], "volume_missing"].fillna(1).sum()),
        },
    }

    return master, items_full, audit_summary
