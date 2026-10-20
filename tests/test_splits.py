"""Unit and dataset tests for T03 (training maturity), T04 (selection maturity), T07 (joins), T08 (geography), and T09 (in-fold preprocessing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist_delay.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from olist_delay.geography import build_geolocation_lookup, haversine_km, resolve_coordinates_with_fallback
from olist_delay.models import fit_model_with_chronological_calibration
from olist_delay.splits import build_split_masks, summarize_splits

pytestmark = pytest.mark.unit



def test_t03_t04_training_and_selection_label_maturity() -> None:
    """T03 & T04: Orders purchased before train_end whose delivery is > train_end must be dropped as in-transit."""
    df = pd.DataFrame(
        [
            # Matured before train_end (2018-01-01) -> included in train_matured_mask
            {
                "order_id": "tr_matured",
                "in_cohort_a": True,
                "order_purchase_timestamp": "2017-12-10 10:00:00",
                "label_available_at": "2017-12-28 15:00:00",
                "late": 0,
            },
            # Purchased before train_end, delivered in January -> dropped from train as in-transit
            {
                "order_id": "tr_in_transit",
                "in_cohort_a": True,
                "order_purchase_timestamp": "2017-12-20 10:00:00",
                "label_available_at": "2018-01-12 09:00:00",
                "late": 1,
            },
            # Validation order matured before selection_at (2018-05-26) -> included in eval_matured_at_selection_mask
            {
                "order_id": "val_matured",
                "in_cohort_a": True,
                "order_purchase_timestamp": "2018-01-10 10:00:00",
                "label_available_at": "2018-01-25 14:00:00",
                "late": 0,
            },
            # Validation order maturing AFTER selection_at -> excluded from selection evaluation!
            {
                "order_id": "val_unmatured_at_selection",
                "in_cohort_a": True,
                "order_purchase_timestamp": "2018-01-20 10:00:00",
                "label_available_at": "2018-06-05 14:00:00",
                "late": 1,
            },
        ]
    )

    sp = build_split_masks(
        df=df,
        split_id="val_2018_01",
        split_type="validation",
        train_start="2016-09-01 00:00:00",
        train_end="2018-01-01 00:00:00",
        eval_start="2018-01-01 00:00:00",
        eval_end="2018-02-01 00:00:00",
        selection_at="2018-05-26 18:18:03",
    )

    by_id_tr = pd.Series(sp.train_matured_mask.to_numpy(), index=df["order_id"])
    by_id_drop = pd.Series(sp.train_in_transit_dropped_mask.to_numpy(), index=df["order_id"])
    by_id_eval = pd.Series(sp.eval_matured_at_selection_mask.to_numpy(), index=df["order_id"])
    by_id_unmat = pd.Series(sp.eval_unmatured_at_selection_mask.to_numpy(), index=df["order_id"])

    assert bool(by_id_tr["tr_matured"]) is True
    assert bool(by_id_tr["tr_in_transit"]) is False
    assert bool(by_id_drop["tr_in_transit"]) is True

    assert bool(by_id_eval["val_matured"]) is True
    assert bool(by_id_eval["val_unmatured_at_selection"]) is False
    assert bool(by_id_unmat["val_unmatured_at_selection"]) is True

    summary_df = summarize_splits(df, [sp], label_col="late")
    assert bool(summary_df.iloc[0]["chronological_strict_pass"]) is True
    assert bool(summary_df.iloc[0]["maturity_strict_pass"]) is True


def test_t08_geography_bounding_box_and_centroid_fallback() -> None:
    """T08: Out-of-bounds coordinates are excluded and fallback hierarchy (zip -> state -> national) works."""
    geo_raw = pd.DataFrame(
        [
            {"geolocation_zip_code_prefix": 1001, "geolocation_lat": -23.55, "geolocation_lng": -46.63, "geolocation_state": "SP"},
            {"geolocation_zip_code_prefix": 1001, "geolocation_lat": -23.55, "geolocation_lng": -46.63, "geolocation_state": "SP"},
            {"geolocation_zip_code_prefix": 1001, "geolocation_lat": -23.56, "geolocation_lng": -46.64, "geolocation_state": "SP"},
            # Corrupt out-of-Brazil coordinate (e.g. lat=42.0, lng=12.0)
            {"geolocation_zip_code_prefix": 99999, "geolocation_lat": 42.0, "geolocation_lng": 12.0, "geolocation_state": "SP"},
        ]
    )
    lookup = build_geolocation_lookup(geo_raw)
    zip_df = lookup["zip_centroids"]
    assert 1001 in set(zip_df["zip_prefix"])
    assert 99999 not in set(zip_df["zip_prefix"])

    query_df = pd.DataFrame(
        [
            {"zip": 1001, "state": "SP"},       # Matches zip_prefix
            {"zip": 99999, "state": "SP"},      # Missing zip -> falls back to state_centroid
            {"zip": 88888, "state": "ZZ"},      # Missing zip & state -> falls back to national_centroid
        ]
    )
    resolved = resolve_coordinates_with_fallback(query_df, "zip", "state", "cust", lookup)
    assert list(resolved["cust_geo_fallback"]) == ["zip_prefix", "state_centroid", "national_centroid"]

    d_zero = haversine_km(np.array([-23.55]), np.array([-46.63]), np.array([-23.55]), np.array([-46.63]))
    assert np.isclose(d_zero[0], 0.0)


def test_a07_all_invalid_coordinates_never_fallback_to_corrupt_rows() -> None:
    """A07: When all geolocation rows are outside Brazil bounds, lookup returns NaN national centroid and 'missing' fallback."""
    bad_geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1000, 2000],
            "geolocation_lat": [80.0, 82.0],
            "geolocation_lng": [10.0, 12.0],
            "geolocation_state": ["SP", "RJ"],
        }
    )
    lookup = build_geolocation_lookup(bad_geo)
    assert len(lookup["zip_centroids"]) == 0
    assert len(lookup["state_centroids"]) == 0
    nat_lat, nat_lng = lookup["national_centroid"]
    assert np.isnan(nat_lat) and np.isnan(nat_lng)

    query_df = pd.DataFrame([{"zip": 1000, "state": "SP"}])
    resolved = resolve_coordinates_with_fallback(query_df, "zip", "state", "cust", lookup)
    assert np.isnan(resolved.loc[0, "cust_lat"])
    assert np.isnan(resolved.loc[0, "cust_lng"])
    assert resolved.loc[0, "cust_geo_fallback"] == "missing"
    assert int(resolved.loc[0, "cust_geo_missing_zip"]) == 1


def test_t09_in_fold_preprocessing_isolation() -> None:
    """T09: Validation extreme values or unseen categories must not alter training imputer/encoder/scaler parameters."""
    rng = np.random.default_rng(5126)
    n_tr = 300
    data = {col: rng.uniform(0.1, 100.0, size=n_tr) for col in NUMERIC_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = ["SP_CAT"] * n_tr
    X_tr = pd.DataFrame(data)
    y_tr = pd.Series((rng.uniform(0, 1, size=n_tr) < 0.15).astype(int))
    ts_tr = pd.Series(pd.date_range("2017-01-01", periods=n_tr, freq="D"))

    bundle = fit_model_with_chronological_calibration(
        model_name="logistic_regression",
        X_train=X_tr,
        y_train=y_tr,
        purchase_timestamps=ts_tr,
        random_seed=5126,
    )
    pre_medians = (
        bundle.base_estimator.named_steps["preprocess"]
        .transformers_[0][1]
        .named_steps["imputer"]
        .statistics_.copy()
    )

    # Construct validation input with extreme numeric outliers and unseen categorical values
    X_val = X_tr.iloc[:10].copy()
    X_val["total_price"] = 9999999.0
    X_val["customer_state"] = "UNSEEN_STATE_ZZ"
    probs = bundle.predict_proba_variant(X_val, method="platt")
    post_medians = (
        bundle.base_estimator.named_steps["preprocess"]
        .transformers_[0][1]
        .named_steps["imputer"]
        .statistics_.copy()
    )

    assert np.allclose(pre_medians, post_medians)
    assert len(probs) == 10
    assert np.all((probs >= 0.0) & (probs <= 1.0))
