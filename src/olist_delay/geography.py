"""Brazilian state-to-region mapping, ZIP centroid lookup with fallback hierarchy, and Haversine distance."""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_BRAZIL_LAT_BOUNDS: tuple[float, float] = (-33.8, 5.4)
DEFAULT_BRAZIL_LNG_BOUNDS: tuple[float, float] = (-74.1, -28.8)

STATE_TO_REGION: dict[str, str] = {
    "AC": "North",
    "AP": "North",
    "AM": "North",
    "PA": "North",
    "RO": "North",
    "RR": "North",
    "TO": "North",
    "AL": "Northeast",
    "BA": "Northeast",
    "CE": "Northeast",
    "MA": "Northeast",
    "PB": "Northeast",
    "PE": "Northeast",
    "PI": "Northeast",
    "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Central-West",
    "GO": "Central-West",
    "MT": "Central-West",
    "MS": "Central-West",
    "ES": "Southeast",
    "MG": "Southeast",
    "RJ": "Southeast",
    "SP": "Southeast",
    "PR": "South",
    "RS": "South",
    "SC": "South",
}


BRAZIL_STATE_TO_REGION: dict[str, str] = STATE_TO_REGION


def map_state_to_region(state_series: pd.Series) -> pd.Series:
    return state_series.map(STATE_TO_REGION).fillna("Unknown")


def haversine_km(
    lat1: np.ndarray | pd.Series,
    lon1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lon2: np.ndarray | pd.Series,
) -> np.ndarray:
    """Vectorized great-circle distance in kilometers on Earth (R = 6371.0088 km)."""
    r = 6371.0088
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dlambda = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return r * c


compute_haversine_km = haversine_km


def build_geolocation_lookup(
    geo_df: pd.DataFrame,
    lat_bounds: tuple[float, float] = DEFAULT_BRAZIL_LAT_BOUNDS,
    lng_bounds: tuple[float, float] = DEFAULT_BRAZIL_LNG_BOUNDS,
    deduplicate_exact_coords: bool = True,
) -> dict[str, pd.DataFrame | tuple[float, float] | int]:
    """Aggregate raw geolocation records into ZIP-prefix centroids, state centroids, and national centroid.

    Strictly filters out coordinates outside `lat_bounds` and `lng_bounds`.
    Never falls back to out-of-bounds coordinates when all rows are invalid; instead returns
    empty centroid tables and `(np.nan, np.nan)` for `national_centroid`.
    """
    required = {"geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_state"}
    missing = required - set(geo_df.columns)
    if missing:
        raise ValueError(f"Geolocation table missing columns: {sorted(missing)}")

    lat_num = pd.to_numeric(geo_df["geolocation_lat"], errors="coerce")
    lng_num = pd.to_numeric(geo_df["geolocation_lng"], errors="coerce")
    in_bounds = lat_num.between(float(lat_bounds[0]), float(lat_bounds[1])) & lng_num.between(
        float(lng_bounds[0]), float(lng_bounds[1])
    )
    out_of_bounds_count = int((~in_bounds).sum())
    valid = geo_df.loc[in_bounds].copy()

    if valid.empty:
        return {
            "zip_centroids": pd.DataFrame(columns=["zip_prefix", "lat", "lng"]),
            "state_centroids": pd.DataFrame(columns=["state", "state_lat", "state_lng"]),
            "national_centroid": (float("nan"), float("nan")),
            "out_of_bounds_dropped_count": out_of_bounds_count,
            "valid_row_count": 0,
        }

    valid["geolocation_lat"] = lat_num.loc[in_bounds]
    valid["geolocation_lng"] = lng_num.loc[in_bounds]

    if deduplicate_exact_coords:
        valid = valid.drop_duplicates(
            subset=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
        )

    zip_centroids = (
        valid.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            lat=("geolocation_lat", "median"),
            lng=("geolocation_lng", "median"),
        )
        .rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
    )
    state_centroids = (
        valid.groupby("geolocation_state", as_index=False)
        .agg(
            state_lat=("geolocation_lat", "median"),
            state_lng=("geolocation_lng", "median"),
        )
        .rename(columns={"geolocation_state": "state"})
    )
    nat_lat = float(valid["geolocation_lat"].median())
    nat_lng = float(valid["geolocation_lng"].median())

    return {
        "zip_centroids": zip_centroids,
        "state_centroids": state_centroids,
        "national_centroid": (nat_lat, nat_lng),
        "out_of_bounds_dropped_count": out_of_bounds_count,
        "valid_row_count": int(len(valid)),
    }


def resolve_coordinates_with_fallback(
    df: pd.DataFrame,
    zip_col: str,
    state_col: str,
    prefix: str,
    geo_lookup: dict[str, pd.DataFrame | tuple[float, float] | int],
) -> pd.DataFrame:
    """Resolve (lat, lng) with explicit fallback hierarchy: zip_prefix -> state_centroid -> national_centroid -> missing."""
    out = df[[zip_col, state_col]].copy()
    zip_centroids: pd.DataFrame = geo_lookup["zip_centroids"]  # type: ignore[assignment]
    state_centroids: pd.DataFrame = geo_lookup["state_centroids"]  # type: ignore[assignment]
    nat_lat, nat_lng = geo_lookup["national_centroid"]  # type: ignore[misc]
    has_nat_coord = bool(np.isfinite(nat_lat) and np.isfinite(nat_lng))

    merged = out.merge(
        zip_centroids,
        left_on=zip_col,
        right_on="zip_prefix",
        how="left",
    ).merge(
        state_centroids,
        left_on=state_col,
        right_on="state",
        how="left",
    )

    has_zip = merged[zip_col].notna()
    has_zip_coord = merged["lat"].notna() & merged["lng"].notna()
    has_state_coord = merged["state_lat"].notna() & merged["state_lng"].notna()

    default_fb = "national_centroid" if has_nat_coord else "missing"
    fallback = np.full(len(merged), default_fb, dtype=object)
    fallback[~has_zip.to_numpy()] = "missing_zip" if has_nat_coord else "missing"
    fallback[(has_zip & ~has_zip_coord & has_state_coord).to_numpy()] = "state_centroid"
    fallback[has_zip_coord.to_numpy()] = "zip_prefix"

    final_lat = np.where(
        has_zip_coord,
        merged["lat"],
        np.where(has_state_coord, merged["state_lat"], nat_lat if has_nat_coord else np.nan),
    )
    final_lng = np.where(
        has_zip_coord,
        merged["lng"],
        np.where(has_state_coord, merged["state_lng"], nat_lng if has_nat_coord else np.nan),
    )

    res = pd.DataFrame(index=df.index)
    res[f"{prefix}_lat"] = final_lat
    res[f"{prefix}_lng"] = final_lng
    res[f"{prefix}_geo_fallback"] = fallback
    res[f"{prefix}_geo_missing_zip"] = (~has_zip_coord).astype(int)
    return res
