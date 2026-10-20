"""Causal event-clock historical feature builder (purchase clock vs completion clock) with multi-seller bridge."""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_HISTORY_TARGET_COLS: set[str] = {
    "order_id",
    "order_purchase_timestamp",
    "primary_seller_id",
    "route_state_pair",
    "customer_state",
    "primary_category",
}


def _group_window_counts_and_sums(
    event_groups: pd.Series,
    event_times: pd.Series,
    event_values: pd.Series | None,
    query_groups: pd.Series,
    query_times: pd.Series,
    window_days: float | None = None,
    event_order_ids: pd.Series | None = None,
    query_order_ids: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact causal counts and value sums for events with `event_time < query_time`
    (and `event_time >= query_time - window_days` if `window_days` is given) within each group,
    strictly excluding any event where `event_order_id == query_order_id`.

    Uses per-group sorted timestamps and `np.searchsorted(..., side='left')` in O((E + Q) log E).
    """
    n_queries = len(query_times)
    counts = np.zeros(n_queries, dtype=np.float64)
    sums = np.zeros(n_queries, dtype=np.float64)

    valid_ev = event_times.notna() & event_groups.notna()
    if event_values is not None:
        valid_ev = valid_ev & event_values.notna()

    if not valid_ev.any() or n_queries == 0:
        return counts, sums

    ev_data: dict[str, np.ndarray] = {
        "g": event_groups[valid_ev].to_numpy(),
        "t": pd.to_datetime(event_times[valid_ev]).to_numpy(dtype="datetime64[ns]"),
        "v": (
            event_values[valid_ev].to_numpy(dtype=np.float64)
            if event_values is not None
            else np.ones(int(valid_ev.sum()), dtype=np.float64)
        ),
    }
    if event_order_ids is not None:
        ev_data["oid"] = event_order_ids[valid_ev].astype(str).to_numpy()

    ev_df = pd.DataFrame(ev_data)
    ev_df.sort_values(["g", "t"], kind="mergesort", inplace=True)

    q_data: dict[str, np.ndarray] = {
        "q_idx": np.arange(n_queries, dtype=np.int64),
        "g": query_groups.to_numpy(),
        "t": pd.to_datetime(query_times).to_numpy(dtype="datetime64[ns]"),
    }
    if query_order_ids is not None:
        q_data["oid"] = query_order_ids.astype(str).to_numpy()
    q_df = pd.DataFrame(q_data)

    ev_g = ev_df["g"].to_numpy()
    ev_t = ev_df["t"].to_numpy()
    ev_v = ev_df["v"].to_numpy()
    ev_oid = ev_df["oid"].to_numpy() if "oid" in ev_df.columns else None

    unique_ev_g, ev_starts, ev_counts = np.unique(ev_g, return_index=True, return_counts=True)
    ev_lookup = {
        g_val: (int(start), int(start + cnt))
        for g_val, start, cnt in zip(unique_ev_g, ev_starts, ev_counts)
    }

    delta_ns = (
        np.timedelta64(int(round(window_days * 86400)), "s").astype("timedelta64[ns]")
        if window_days is not None
        else None
    )

    q_valid = q_df[q_df["g"].notna() & pd.notna(q_df["t"])].copy()
    if len(q_valid) == 0:
        return counts, sums

    q_valid.sort_values(["g", "t"], kind="mergesort", inplace=True)
    qv_g = q_valid["g"].to_numpy()
    qv_t = q_valid["t"].to_numpy()
    qv_idx = q_valid["q_idx"].to_numpy()

    unique_q_g, q_starts, q_counts = np.unique(qv_g, return_index=True, return_counts=True)
    for g_val, q_start, q_cnt in zip(unique_q_g, q_starts, q_counts):
        bounds = ev_lookup.get(g_val)
        if bounds is None:
            continue
        s_idx, e_idx = bounds
        g_times = ev_t[s_idx:e_idx]
        g_vals = ev_v[s_idx:e_idx]
        prefix_vals = np.concatenate(([0.0], np.cumsum(g_vals)))

        q_end = int(q_start + q_cnt)
        qt = qv_t[int(q_start):q_end]
        q_indices = qv_idx[int(q_start):q_end]

        right_idx = np.searchsorted(g_times, qt, side="left")
        if delta_ns is not None:
            left_times = qt - delta_ns
            left_idx = np.searchsorted(g_times, left_times, side="left")
            c = (right_idx - left_idx).astype(np.float64)
            s = prefix_vals[right_idx] - prefix_vals[left_idx]
        else:
            c = right_idx.astype(np.float64)
            s = prefix_vals[right_idx]

        counts[q_indices] = c
        sums[q_indices] = s

    # Explicit vectorized self-order exclusion: an order must never count itself
    if ev_oid is not None and "oid" in q_valid.columns and len(ev_df) > 0:
        self_matches = q_valid[["q_idx", "g", "oid", "t"]].merge(
            ev_df[["g", "oid", "t", "v"]],
            on=["g", "oid"],
            how="inner",
            suffixes=("_q", "_ev"),
        )
        if len(self_matches) > 0:
            cond = self_matches["t_ev"] < self_matches["t_q"]
            if delta_ns is not None:
                cond = cond & (self_matches["t_ev"] >= (self_matches["t_q"] - delta_ns))
            hit = self_matches.loc[cond]
            if len(hit) > 0:
                sub_agg = hit.groupby("q_idx", as_index=False).agg(
                    self_c=("v", "count"),
                    self_s=("v", "sum"),
                )
                q_idx_arr = sub_agg["q_idx"].to_numpy(dtype=np.int64)
                counts[q_idx_arr] = np.maximum(
                    0.0, counts[q_idx_arr] - sub_agg["self_c"].to_numpy(dtype=np.float64)
                )
                sums[q_idx_arr] -= sub_agg["self_s"].to_numpy(dtype=np.float64)

    return counts, sums


def compute_global_causal_prior(
    event_completion_times: pd.Series,
    event_outcomes: pd.Series,
    query_times: pd.Series,
    default_prior: float = 0.068,
    min_completed_orders: int = 20,
    window_days: float | None = None,
    event_order_ids: pd.Series | None = None,
    query_order_ids: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global completed count and late rate strictly before each query timestamp."""
    dummy_ev_g = pd.Series("GLOBAL", index=event_completion_times.index, dtype=object)
    dummy_q_g = pd.Series("GLOBAL", index=query_times.index, dtype=object)
    counts, sums = _group_window_counts_and_sums(
        event_groups=dummy_ev_g,
        event_times=event_completion_times,
        event_values=event_outcomes,
        query_groups=dummy_q_g,
        query_times=query_times,
        window_days=window_days,
        event_order_ids=event_order_ids,
        query_order_ids=query_order_ids,
    )
    rates = np.where(
        counts >= min_completed_orders,
        sums / np.maximum(counts, 1.0),
        default_prior,
    )
    return counts, rates


def _expand_order_seller_bridge(
    df: pd.DataFrame,
    include_outcomes: bool,
    label_col: str = "late",
) -> pd.DataFrame:
    """Build deduplicated `(order_id, seller_id)` bridge rows from an orders DataFrame.

    If `seller_ids_pipe` is present (e.g. `'s1|s2'`), explodes multi-seller orders into one row per
    distinct `(order_id, seller_id)` pair; otherwise uses `(order_id, primary_seller_id)`.
    """
    base_cols = ["order_id", "order_purchase_timestamp"]
    if include_outcomes:
        for col in ["label_available_at", label_col, "delay_calendar_days"]:
            if col in df.columns and col not in base_cols:
                base_cols.append(col)

    out = df[base_cols].copy()
    out["_row_pos"] = np.arange(len(df), dtype=np.int64)

    if "seller_ids_pipe" in df.columns:
        pipe_s = df["seller_ids_pipe"].fillna(df["primary_seller_id"]).astype(str)
        has_multi = pipe_s.str.contains("|", regex=False)
        if bool(has_multi.any()):
            out["seller_id"] = pipe_s.str.split("|")
            out = out.explode("seller_id", ignore_index=True)
        else:
            out["seller_id"] = pipe_s.to_numpy()
    elif "seller_id" in df.columns and "primary_seller_id" not in df.columns:
        out["seller_id"] = df["seller_id"].astype(str).to_numpy()
    else:
        out["seller_id"] = df["primary_seller_id"].astype(str).to_numpy()

    out = out.drop_duplicates(subset=["order_id", "seller_id"], keep="first").reset_index(drop=True)
    return out


def attach_event_clock_history_features(
    target_orders: pd.DataFrame,
    history_events: pd.DataFrame | None = None,
    label_col: str = "late",
    smoothing_m: float = 5.0,
    default_prior: float = 0.068,
    smoothing_alpha: float | None = None,
    seller_bridge_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute purchase-time historical features using strict dual-clock event logic.

    - Purchase clock (`event_time = order_purchase_timestamp`):
      Only counts prior order volume/congestion (`< query_purchase_ts`, `event.order_id != query.order_id`).
      Never accesses outcomes.
    - Completion clock (`event_time = label_available_at`):
      Only aggregates outcomes (`late`, `delay_calendar_days`) for orders completed strictly before
      `query_purchase_ts` (`order_purchase_timestamp <= label_available_at < query_purchase_ts` and
      `event.order_id != query.order_id`).
      Any event with inverted timestamps (`label_available_at < order_purchase_timestamp`) is rejected.
    - Multi-seller orders:
      Uses deduplicated `(order_id, seller_id)` bridge rows and aggregates seller features to `order_id`
      via `mean`.
    """
    missing_target = REQUIRED_HISTORY_TARGET_COLS - set(target_orders.columns)
    if missing_target:
        raise ValueError(
            f"target_orders missing required columns for historical feature engineering: {sorted(missing_target)}"
        )

    if smoothing_alpha is not None:
        smoothing_m = float(smoothing_alpha)

    df = target_orders.copy()
    events = history_events.copy() if history_events is not None else df.copy()

    q_times = pd.to_datetime(df["order_purchase_timestamp"])
    q_oids = df["order_id"].astype(str)

    ev_purch_times = pd.to_datetime(events["order_purchase_timestamp"])
    ev_comp_raw = pd.to_datetime(events["label_available_at"])
    # Reject physically inverted completion timestamps where label_available_at < order_purchase_timestamp
    valid_comp_clock = ev_comp_raw.notna() & ev_purch_times.notna() & (ev_comp_raw >= ev_purch_times)
    ev_comp_times = ev_comp_raw.where(valid_comp_clock, pd.NaT)

    ev_outcomes = pd.to_numeric(events[label_col], errors="coerce")
    ev_delays = (
        pd.to_numeric(events["delay_calendar_days"], errors="coerce")
        if "delay_calendar_days" in events.columns
        else pd.Series(np.nan, index=events.index)
    )
    ev_oids = events["order_id"].astype(str)

    # 1. Global causal completion-clock prior at each query timestamp
    global_comp_cnt, global_prior_rate = compute_global_causal_prior(
        event_completion_times=ev_comp_times,
        event_outcomes=ev_outcomes,
        query_times=q_times,
        default_prior=default_prior,
        min_completed_orders=20,
        window_days=None,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    _, global_recent_30d_rate = compute_global_causal_prior(
        event_completion_times=ev_comp_times,
        event_outcomes=ev_outcomes,
        query_times=q_times,
        default_prior=default_prior,
        min_completed_orders=20,
        window_days=30.0,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    df["global_hist_completed_count"] = global_comp_cnt
    df["global_hist_late_rate"] = global_prior_rate
    df["global_recent_30d_late_rate"] = global_recent_30d_rate

    # 2 & 3. Seller purchase-clock & completion-clock history via deduplicated (order_id, seller_id) bridge
    if seller_bridge_events is not None:
        ev_bridge = seller_bridge_events.drop_duplicates(subset=["order_id", "seller_id"]).reset_index(drop=True)
    else:
        ev_bridge = _expand_order_seller_bridge(events, include_outcomes=True, label_col=label_col)

    q_bridge = _expand_order_seller_bridge(df, include_outcomes=False, label_col=label_col)
    q_bridge_row_pos = q_bridge["_row_pos"].to_numpy(dtype=np.int64)
    q_bridge_times = pd.to_datetime(q_bridge["order_purchase_timestamp"])
    q_bridge_oids = q_bridge["order_id"].astype(str)
    q_bridge_sellers = q_bridge["seller_id"]

    ev_b_purch = pd.to_datetime(ev_bridge["order_purchase_timestamp"])
    ev_b_comp_raw = pd.to_datetime(ev_bridge["label_available_at"])
    ev_b_comp = ev_b_comp_raw.where(
        ev_b_comp_raw.notna() & ev_b_purch.notna() & (ev_b_comp_raw >= ev_b_purch),
        pd.NaT,
    )
    ev_b_outcomes = pd.to_numeric(ev_bridge[label_col], errors="coerce")
    ev_b_delays = (
        pd.to_numeric(ev_bridge["delay_calendar_days"], errors="coerce")
        if "delay_calendar_days" in ev_bridge.columns
        else pd.Series(np.nan, index=ev_bridge.index)
    )
    ev_b_oids = ev_bridge["order_id"].astype(str)
    ev_b_sellers = ev_bridge["seller_id"]

    sb_purch_all, _ = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_purch,
        event_values=None,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=None,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )
    sb_purch_7d, _ = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_purch,
        event_values=None,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=7.0,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )
    sb_purch_30d, _ = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_purch,
        event_values=None,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=30.0,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )

    sb_comp_all, sb_late_all = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_comp,
        event_values=ev_b_outcomes,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=None,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )
    sb_comp_30d, sb_late_30d = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_comp,
        event_values=ev_b_outcomes,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=30.0,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )
    sb_comp_90d, sb_late_90d = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_comp,
        event_values=ev_b_outcomes,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=90.0,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )
    _, sb_delay_sum_all = _group_window_counts_and_sums(
        event_groups=ev_b_sellers,
        event_times=ev_b_comp,
        event_values=ev_b_delays,
        query_groups=q_bridge_sellers,
        query_times=q_bridge_times,
        window_days=None,
        event_order_ids=ev_b_oids,
        query_order_ids=q_bridge_oids,
    )

    gp_rate_b = global_prior_rate[q_bridge_row_pos]
    gp_30d_b = global_recent_30d_rate[q_bridge_row_pos]

    sb_hist_rate = (sb_late_all + smoothing_m * gp_rate_b) / (sb_comp_all + smoothing_m)
    sb_30d_rate = (sb_late_30d + (smoothing_m / 2.0) * gp_30d_b) / (sb_comp_30d + (smoothing_m / 2.0))
    sb_90d_rate = (sb_late_90d + smoothing_m * gp_30d_b) / (sb_comp_90d + smoothing_m)
    sb_mean_delay = np.where(sb_comp_all > 0, sb_delay_sum_all / np.maximum(sb_comp_all, 1.0), 0.0)
    sb_cold_start = (sb_comp_all < 5).astype(np.float64)

    if len(q_bridge) == len(df) and np.array_equal(q_bridge_row_pos, np.arange(len(df), dtype=np.int64)):
        df["seller_prior_purchase_count"] = sb_purch_all
        df["seller_purchase_count_7d"] = sb_purch_7d
        df["seller_purchase_count_30d"] = sb_purch_30d
        df["seller_completed_count"] = sb_comp_all
        df["seller_completed_count_30d"] = sb_comp_30d
        df["seller_completed_count_90d"] = sb_comp_90d
        df["seller_hist_late_rate_smoothed"] = sb_hist_rate
        df["seller_recent_30d_late_rate_smoothed"] = sb_30d_rate
        df["seller_recent_90d_late_rate_smoothed"] = sb_90d_rate
        df["seller_hist_mean_delay_days"] = sb_mean_delay
        df["seller_is_cold_start"] = (sb_cold_start >= 0.5).astype(int)
    else:
        sb_df = pd.DataFrame(
            {
                "_row_pos": q_bridge_row_pos,
                "seller_prior_purchase_count": sb_purch_all,
                "seller_purchase_count_7d": sb_purch_7d,
                "seller_purchase_count_30d": sb_purch_30d,
                "seller_completed_count": sb_comp_all,
                "seller_completed_count_30d": sb_comp_30d,
                "seller_completed_count_90d": sb_comp_90d,
                "seller_hist_late_rate_smoothed": sb_hist_rate,
                "seller_recent_30d_late_rate_smoothed": sb_30d_rate,
                "seller_recent_90d_late_rate_smoothed": sb_90d_rate,
                "seller_hist_mean_delay_days": sb_mean_delay,
                "seller_is_cold_start": sb_cold_start,
            }
        )
        sb_agg = sb_df.groupby("_row_pos", sort=True).mean()
        for col in [
            "seller_prior_purchase_count",
            "seller_purchase_count_7d",
            "seller_purchase_count_30d",
            "seller_completed_count",
            "seller_completed_count_30d",
            "seller_completed_count_90d",
            "seller_hist_late_rate_smoothed",
            "seller_recent_30d_late_rate_smoothed",
            "seller_recent_90d_late_rate_smoothed",
            "seller_hist_mean_delay_days",
        ]:
            df[col] = sb_agg[col].to_numpy()
        df["seller_is_cold_start"] = (sb_agg["seller_is_cold_start"].to_numpy() >= 0.5).astype(int)

    # 4. Route (state->state) completion-clock outcomes ( strictly label_available_at < t )
    route_comp_all, route_late_all = _group_window_counts_and_sums(
        event_groups=events["route_state_pair"],
        event_times=ev_comp_times,
        event_values=ev_outcomes,
        query_groups=df["route_state_pair"],
        query_times=q_times,
        window_days=None,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    route_comp_60d, route_late_60d = _group_window_counts_and_sums(
        event_groups=events["route_state_pair"],
        event_times=ev_comp_times,
        event_values=ev_outcomes,
        query_groups=df["route_state_pair"],
        query_times=q_times,
        window_days=60.0,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    _, route_delay_sum_all = _group_window_counts_and_sums(
        event_groups=events["route_state_pair"],
        event_times=ev_comp_times,
        event_values=ev_delays,
        query_groups=df["route_state_pair"],
        query_times=q_times,
        window_days=None,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    df["route_completed_count"] = route_comp_all
    df["route_hist_late_rate_smoothed"] = (
        route_late_all + smoothing_m * global_prior_rate
    ) / (route_comp_all + smoothing_m)
    df["route_recent_60d_late_rate_smoothed"] = (
        route_late_60d + smoothing_m * global_recent_30d_rate
    ) / (route_comp_60d + smoothing_m)
    df["route_hist_mean_delay_days"] = np.where(
        route_comp_all > 0,
        route_delay_sum_all / np.maximum(route_comp_all, 1.0),
        0.0,
    )

    # 5. Customer destination state completion-clock outcomes ( strictly label_available_at < t )
    state_comp_all, state_late_all = _group_window_counts_and_sums(
        event_groups=events["customer_state"],
        event_times=ev_comp_times,
        event_values=ev_outcomes,
        query_groups=df["customer_state"],
        query_times=q_times,
        window_days=None,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    state_comp_30d, state_late_30d = _group_window_counts_and_sums(
        event_groups=events["customer_state"],
        event_times=ev_comp_times,
        event_values=ev_outcomes,
        query_groups=df["customer_state"],
        query_times=q_times,
        window_days=30.0,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    df["cust_state_completed_count"] = state_comp_all
    df["cust_state_hist_late_rate_smoothed"] = (
        state_late_all + smoothing_m * global_prior_rate
    ) / (state_comp_all + smoothing_m)
    df["cust_state_recent_30d_late_rate_smoothed"] = (
        state_late_30d + smoothing_m * global_recent_30d_rate
    ) / (state_comp_30d + smoothing_m)

    # 6. Product category completion-clock outcomes ( strictly label_available_at < t )
    cat_comp_all, cat_late_all = _group_window_counts_and_sums(
        event_groups=events["primary_category"],
        event_times=ev_comp_times,
        event_values=ev_outcomes,
        query_groups=df["primary_category"],
        query_times=q_times,
        window_days=None,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    df["category_completed_count"] = cat_comp_all
    df["category_hist_late_rate_smoothed"] = (
        cat_late_all + smoothing_m * global_prior_rate
    ) / (cat_comp_all + smoothing_m)

    # 7. Customer repeat purchase count (purchase clock: strictly order_purchase_timestamp < t)
    if "customer_unique_id" in df.columns and "customer_unique_id" in events.columns:
        cust_prior_orders, _ = _group_window_counts_and_sums(
            event_groups=events["customer_unique_id"],
            event_times=ev_purch_times,
            event_values=None,
            query_groups=df["customer_unique_id"],
            query_times=q_times,
            window_days=None,
            event_order_ids=ev_oids,
            query_order_ids=q_oids,
        )
        df["customer_prior_order_count"] = cust_prior_orders
    else:
        df["customer_prior_order_count"] = 0.0

    return df


V3_ROUTE_WINDOW_DAYS = 60.0
V3_FALLBACK_DELIVERY_DAYS = 12.0


def attach_v3_route_features(
    target_orders: pd.DataFrame,
    history_events: pd.DataFrame | None = None,
    label_col: str = "late",
    smoothing_m: float = 5.0,
    default_prior: float = 0.068,
    window_days: float = V3_ROUTE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Add the v3 route features (`base_v3`), all strictly causal at purchase time.

    - `route_recent_60d_late_rate_ca`: smoothed late rate of the seller-state -> customer-state route over the last
      `window_days`, counting a late order from the moment its lateness was knowable: the start of the day after its
      promised date, or delivery if that came first. Only the binary outcome is used, so nothing about the eventual
      delay length leaks.
    - `route_recent_60d_mean_delivery_days`: smoothed mean purchase-to-delivery days of orders on the route delivered
      in the last `window_days` (delivery clock).
    - `promise_slack_days_60d`: promised lead days minus that recent route delivery time.

    Without `order_estimated_delivery_date` in the events (e.g. reduced scoring event files) the late rate falls back
    to the delivery clock.
    """
    df = target_orders.copy()
    events = history_events.copy() if history_events is not None else df.copy()
    q_times = pd.to_datetime(df["order_purchase_timestamp"])
    q_oids = df["order_id"].astype(str)

    ev_purch = pd.to_datetime(events["order_purchase_timestamp"])
    ev_comp_raw = pd.to_datetime(events["label_available_at"])
    valid = ev_comp_raw.notna() & ev_purch.notna() & (ev_comp_raw >= ev_purch)
    ev_comp = ev_comp_raw.where(valid, pd.NaT)
    ev_out = pd.to_numeric(events[label_col], errors="coerce")
    ev_oids = events["order_id"].astype(str)

    if "order_estimated_delivery_date" in events.columns:
        overdue_at = pd.to_datetime(events["order_estimated_delivery_date"]).dt.normalize() + pd.Timedelta(days=1)
        earlier = ev_out.eq(1) & ev_comp.notna() & overdue_at.notna() & (overdue_at < ev_comp)
        ev_known = ev_comp.where(~earlier, overdue_at)
    else:
        ev_known = ev_comp

    _, global_ca_rate = compute_global_causal_prior(
        event_completion_times=ev_known,
        event_outcomes=ev_out,
        query_times=q_times,
        default_prior=default_prior,
        min_completed_orders=20,
        window_days=window_days,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    r_cnt, r_late = _group_window_counts_and_sums(
        event_groups=events["route_state_pair"],
        event_times=ev_known,
        event_values=ev_out,
        query_groups=df["route_state_pair"],
        query_times=q_times,
        window_days=window_days,
        event_order_ids=ev_oids,
        query_order_ids=q_oids,
    )
    df["route_recent_60d_late_rate_ca"] = (r_late + smoothing_m * global_ca_rate) / (r_cnt + smoothing_m)

    # Delivery durations are only known at delivery, so they stay on the delivery clock.
    ev_days = ((ev_comp - ev_purch).dt.total_seconds() / 86400.0).where(ev_comp.notna() & ev_out.notna())
    dummy_ev = pd.Series("GLOBAL", index=events.index, dtype=object)
    dummy_q = pd.Series("GLOBAL", index=df.index, dtype=object)
    g_cnt, g_sum = _group_window_counts_and_sums(
        dummy_ev, ev_comp, ev_days, dummy_q, q_times, window_days, ev_oids, q_oids
    )
    g_cnt_all, g_sum_all = _group_window_counts_and_sums(
        dummy_ev, ev_comp, ev_days, dummy_q, q_times, None, ev_oids, q_oids
    )
    global_mean_days = np.where(
        g_cnt >= 20,
        g_sum / np.maximum(g_cnt, 1.0),
        np.where(g_cnt_all >= 20, g_sum_all / np.maximum(g_cnt_all, 1.0), V3_FALLBACK_DELIVERY_DAYS),
    )
    rd_cnt, rd_sum = _group_window_counts_and_sums(
        events["route_state_pair"], ev_comp, ev_days, df["route_state_pair"], q_times, window_days, ev_oids, q_oids
    )
    df["route_recent_60d_mean_delivery_days"] = (rd_sum + smoothing_m * global_mean_days) / (rd_cnt + smoothing_m)
    df["promise_slack_days_60d"] = (
        pd.to_numeric(df["estimated_lead_calendar_days"], errors="coerce") - df["route_recent_60d_mean_delivery_days"]
    )
    return df
