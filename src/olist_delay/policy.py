"""Capacity-constrained daily Top-k% alert policies, Path A / Path B dual-population accounting, and cost-ratio sensitivity."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def rank_orders_deterministically(
    order_ids: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    purchase_timestamps: pd.Series | np.ndarray | None = None,
    sample_draw_ids: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Rank orders deterministically by `(-risk_score, +order_purchase_timestamp, +order_id, +sample_draw_id)`."""
    n_rows = len(order_ids)
    if purchase_timestamps is not None:
        ts_parsed = pd.to_datetime(pd.Series(purchase_timestamps), errors="coerce")
        ts_ns = ts_parsed.astype("int64").to_numpy()
    else:
        ts_ns = np.zeros(n_rows, dtype=np.int64)

    if sample_draw_ids is not None:
        draw_ids = np.asarray(sample_draw_ids, dtype=np.int64)
    else:
        draw_ids = np.zeros(n_rows, dtype=np.int64)

    df = pd.DataFrame(
        {
            "order_id": np.asarray(order_ids, dtype=str),
            "risk_score": np.asarray(scores, dtype=np.float64),
            "_purchase_ts_ns": ts_ns,
            "_sample_draw_id": draw_ids,
        }
    )
    df["orig_pos"] = np.arange(len(df), dtype=np.int64)
    df.sort_values(
        by=["risk_score", "_purchase_ts_ns", "order_id", "_sample_draw_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
        inplace=True,
    )
    n = max(1, len(df))
    df["risk_rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    df["risk_percentile"] = df["risk_rank"] / float(n)
    df.sort_values("orig_pos", kind="mergesort", inplace=True)
    return df[["order_id", "risk_score", "risk_rank", "risk_percentile"]].reset_index(drop=True)


def assign_risk_tier(
    risk_percentile: pd.Series | np.ndarray,
    primary_budget: float = 0.10,
    secondary_budget: float = 0.20,
) -> np.ndarray:
    """Assign human-readable operational tier based on deterministic risk percentile (`floor(k * n)`)."""
    pct = np.asarray(risk_percentile, dtype=np.float64)
    tiers = np.full(len(pct), "LOW_NORMAL", dtype=object)
    valid = ~np.isnan(pct)
    tiers[valid & (pct <= secondary_budget + 1e-12)] = "MEDIUM_WATCHLIST_TOP20"
    tiers[valid & (pct <= primary_budget + 1e-12)] = "HIGH_PRIORITY_ALERT_TOP10"
    tiers[~valid] = "INELIGIBLE"
    return tiers


def allocate_daily_alerts(
    df: pd.DataFrame | None = None,
    scores: pd.Series | np.ndarray | None = None,
    primary_budget: float = 0.10,
    secondary_budget: float = 0.20,
    policy_version: str = "daily_end_of_day_top10_floor_v2",
    exclude_known_delivered_before_decision: bool = True,
    separate_already_overdue_before_decision: bool = True,
    decision_events: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp | None = None,
    df_orders: pd.DataFrame | None = None,
    budget_k: float | None = None,
    label_col: str = "late",
    seed: int = 5126,
) -> pd.DataFrame:
    """Allocate daily capacity-constrained alerts at end-of-day cutoff `T_decide(d) = d 23:59:59.999999`.

    Enforces:
    - Daily quota `q_d = floor(primary_budget * N_eligible,d)` (0 alerts allowed when `N_eligible,d < 1/primary_budget`).
    - No cross-day quota carryover; future days' orders never alter past days' selections.
    - Orders already known delivered on or before `T_decide(d)` are excluded (`KNOWN_DELIVERED_BEFORE_DECISION`).
    - Orders whose promised delivery date is strictly earlier than `d` are separated (`ALREADY_OVERDUE_BEFORE_DECISION`).
    - Deterministic tie-breaking: `(-risk_score, +order_purchase_timestamp, +order_id, +sample_draw_id)`.
    """
    eff_df = df if df is not None else df_orders
    if eff_df is None:
        raise ValueError("Either 'df' or 'df_orders' must be provided to allocate_daily_alerts.")
    eff_budget = float(budget_k) if budget_k is not None else float(primary_budget)

    n = len(eff_df)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "order_id",
                "order_purchase_timestamp",
                "decision_date",
                "decision_cutoff_at",
                "risk_score",
                "is_eligible_for_warning",
                "eligibility_reason",
                "daily_eligible_n",
                "daily_quota_q",
                "daily_secondary_quota_q20",
                "daily_rank",
                "daily_percentile",
                "selected_top10",
                "alert_selected",
                "selected_top20",
                "risk_tier",
                "recommended_action",
                "policy_version",
            ]
        )

    order_ids = eff_df["order_id"].astype(str).to_numpy()
    s_arr = np.asarray(scores, dtype=np.float64)
    purchase_ts = pd.to_datetime(eff_df["order_purchase_timestamp"], errors="coerce")
    purchase_day = purchase_ts.dt.floor("D")
    decision_date = purchase_ts.dt.strftime("%Y-%m-%d").fillna("INVALID_DATE")
    decision_cutoff_at = purchase_day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        effective_cutoff = decision_cutoff_at.where(decision_cutoff_at <= as_of_ts, as_of_ts)
    else:
        effective_cutoff = decision_cutoff_at

    # Eligibility determination
    is_eligible = np.ones(n, dtype=bool)
    reason = np.full(n, "ELIGIBLE_END_OF_DAY_BATCH", dtype=object)

    invalid_mask = (
        purchase_ts.isna().to_numpy()
        | np.isnan(s_arr)
        | (eff_df["order_id"].isna().to_numpy())
        | (order_ids == "")
    )
    if as_of is not None:
        after_as_of = (purchase_ts > pd.Timestamp(as_of)).fillna(False).to_numpy()
        is_eligible[after_as_of] = False
        reason[after_as_of] = "PURCHASE_AFTER_AS_OF_CUTOFF"

    is_eligible[invalid_mask] = False
    reason[invalid_mask] = "INVALID_TIMESTAMP_OR_SCORE"

    if exclude_known_delivered_before_decision:
        known_delivered = np.zeros(n, dtype=bool)
        if decision_events is not None and not decision_events.empty:
            ev = decision_events.copy()
            if {"order_id", "event_type", "event_timestamp"}.issubset(ev.columns):
                ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"], errors="coerce")
                deliv_ev = (
                    ev[ev["event_type"].astype(str).str.lower() == "delivered"]
                    .dropna(subset=["order_id", "event_timestamp"])
                    .groupby("order_id", as_index=False)["event_timestamp"]
                    .min()
                )
                deliv_map = dict(zip(deliv_ev["order_id"].astype(str), deliv_ev["event_timestamp"]))
                mapped_deliv = pd.Series(order_ids).map(deliv_map)
                known_delivered = (
                    mapped_deliv.notna()
                    & (mapped_deliv >= purchase_ts.reset_index(drop=True))
                    & (mapped_deliv <= effective_cutoff.reset_index(drop=True))
                ).to_numpy(dtype=bool)
        elif "order_delivered_customer_date" in eff_df.columns:
            deliv_ts = pd.to_datetime(eff_df["order_delivered_customer_date"], errors="coerce")
            known_delivered = (
                deliv_ts.notna()
                & (deliv_ts >= purchase_ts)
                & (deliv_ts <= effective_cutoff)
            ).to_numpy(dtype=bool)

        deliv_excl = is_eligible & known_delivered
        is_eligible[deliv_excl] = False
        reason[deliv_excl] = "KNOWN_DELIVERED_BEFORE_DECISION"

    if separate_already_overdue_before_decision and "order_estimated_delivery_date" in eff_df.columns:
        est_ts = pd.to_datetime(eff_df["order_estimated_delivery_date"], errors="coerce")
        already_overdue = (est_ts.notna() & (est_ts.dt.floor("D") < purchase_day)).to_numpy(dtype=bool)
        overdue_excl = is_eligible & already_overdue
        is_eligible[overdue_excl] = False
        reason[overdue_excl] = "ALREADY_OVERDUE_BEFORE_DECISION"

    sample_draws = (
        eff_df["sample_draw_id"].to_numpy(dtype=np.int64)
        if "sample_draw_id" in eff_df.columns
        else np.zeros(n, dtype=np.int64)
    )

    daily_eligible_n = np.zeros(n, dtype=np.int64)
    daily_quota_q = np.zeros(n, dtype=np.int64)
    daily_secondary_q20 = np.zeros(n, dtype=np.int64)
    daily_rank = np.full(n, -1, dtype=np.int64)
    daily_percentile = np.full(n, np.nan, dtype=np.float64)
    selected_top10 = np.zeros(n, dtype=bool)
    selected_top20 = np.zeros(n, dtype=bool)

    work = pd.DataFrame(
        {
            "orig_pos": np.arange(n, dtype=np.int64),
            "decision_date": decision_date.to_numpy(dtype=str),
            "is_eligible": is_eligible,
            "order_id": order_ids,
            "risk_score": s_arr,
            "purchase_ts": purchase_ts.to_numpy(),
            "sample_draw_id": sample_draws,
        }
    )

    for d_str, grp in work.groupby("decision_date", sort=False):
        idx_all = grp["orig_pos"].to_numpy()
        elig_grp = grp[grp["is_eligible"]]
        n_elig = len(elig_grp)
        q_d = int(np.floor(eff_budget * n_elig + 1e-12)) if (n_elig > 0 and eff_budget > 0) else 0
        q20_d = int(np.floor(secondary_budget * n_elig + 1e-12)) if (n_elig > 0 and secondary_budget > 0) else 0

        daily_eligible_n[idx_all] = n_elig
        daily_quota_q[idx_all] = q_d
        daily_secondary_q20[idx_all] = q20_d

        if n_elig > 0:
            ranked_d = rank_orders_deterministically(
                order_ids=elig_grp["order_id"],
                scores=elig_grp["risk_score"],
                purchase_timestamps=elig_grp["purchase_ts"],
                sample_draw_ids=elig_grp["sample_draw_id"],
            )
            idx_elig = elig_grp["orig_pos"].to_numpy()
            ranks_d = ranked_d["risk_rank"].to_numpy(dtype=np.int64)
            pcts_d = ranked_d["risk_percentile"].to_numpy(dtype=np.float64)

            daily_rank[idx_elig] = ranks_d
            daily_percentile[idx_elig] = pcts_d
            selected_top10[idx_elig] = ranks_d <= q_d
            selected_top20[idx_elig] = ranks_d <= q20_d

    tiers = np.full(n, "INELIGIBLE", dtype=object)
    tiers[is_eligible] = "LOW_NORMAL"
    tiers[selected_top20] = "MEDIUM_WATCHLIST_TOP20"
    tiers[selected_top10] = "HIGH_PRIORITY_ALERT_TOP10"

    actions = np.where(
        selected_top10,
        "PROACTIVE_REVIEW_AND_EXPEDITE_CHECK",
        np.where(
            selected_top20,
            "MONITOR_CARRIER_HANDOVER_SLA",
            np.where(is_eligible, "STANDARD_FULFILLMENT", "EXCLUDED_NOT_ELIGIBLE"),
        ),
    )

    out = pd.DataFrame(
        {
            "order_id": order_ids,
            "order_purchase_timestamp": purchase_ts.to_numpy(),
            "decision_date": decision_date.to_numpy(dtype=str),
            "decision_cutoff_at": effective_cutoff.to_numpy(),
            "risk_score": s_arr,
            "is_eligible_for_warning": is_eligible,
            "eligibility_reason": reason,
            "daily_eligible_n": daily_eligible_n,
            "daily_quota_q": daily_quota_q,
            "daily_secondary_quota_q20": daily_secondary_q20,
            "daily_rank": daily_rank,
            "daily_percentile": daily_percentile,
            "selected_top10": selected_top10,
            "alert_selected": selected_top10,
            "selected_top20": selected_top20,
            "risk_tier": tiers,
            "recommended_action": actions,
            "policy_version": policy_version,
        }
    )
    if "in_cohort_a" in eff_df.columns:
        out["in_cohort_a"] = eff_df["in_cohort_a"].to_numpy()
    if label_col in eff_df.columns:
        out[label_col] = eff_df[label_col].to_numpy()
    if "late" in eff_df.columns and "late" not in out.columns:
        out["late"] = eff_df["late"].to_numpy()
    return out


def summarize_daily_policy(
    df_orders: pd.DataFrame | None = None,
    alloc_df: pd.DataFrame | None = None,
    primary_budget: float = 0.10,
    label_col: str = "late",
    df_allocated: pd.DataFrame | None = None,
    model_name: str = "selected_model",
    split_id: str = "historical_backtest",
    budget_k: float | None = None,
) -> pd.DataFrame:
    """Summarize daily capacity allocation, eligibility exclusions, and Path A / Path B outcomes per day `d`."""
    if df_allocated is not None:
        eff_alloc = df_allocated
        eff_orders = df_orders if df_orders is not None else df_allocated
    elif alloc_df is None and df_orders is not None:
        eff_alloc = df_orders
        eff_orders = df_orders
    else:
        eff_alloc = alloc_df
        eff_orders = df_orders if df_orders is not None else alloc_df

    if eff_alloc is None or eff_orders is None:
        raise ValueError("Allocation DataFrame must be provided to summarize_daily_policy.")

    eff_budget = float(budget_k) if budget_k is not None else float(primary_budget)
    merged = eff_alloc.copy()
    if "in_cohort_a" in eff_orders.columns:
        merged["in_cohort_a"] = eff_orders["in_cohort_a"].fillna(False).to_numpy(dtype=bool)
    else:
        merged["in_cohort_a"] = (
            eff_orders[label_col].notna().to_numpy(dtype=bool) if label_col in eff_orders.columns else False
        )

    if label_col in eff_orders.columns:
        merged["_y"] = pd.to_numeric(eff_orders[label_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        merged["_y"] = np.nan

    rows: list[dict[str, Any]] = []
    for d_str, grp in merged.groupby("decision_date", sort=True):
        in_a = grp["in_cohort_a"].to_numpy(dtype=bool)
        sel_col = "selected_top10" if "selected_top10" in grp.columns else "alert_selected"
        sel = grp[sel_col].to_numpy(dtype=bool)
        y_val = grp["_y"].to_numpy(dtype=np.float64)
        reasons = grp["eligibility_reason"].astype(str).to_numpy()
        scores_sel = grp.loc[sel, "risk_score"].to_numpy(dtype=np.float64)

        n_total_d = int(len(grp))
        n_a_d = int(np.sum(in_a))
        n_unlabeled_d = n_total_d - n_a_d
        n_elig_d = int(grp["is_eligible_for_warning"].sum())
        q_d = int(grp["daily_quota_q"].iloc[0]) if len(grp) > 0 else 0
        sel_cnt = int(np.sum(sel))
        sel_in_a = int(np.sum(sel & in_a))
        sel_unlabeled = int(np.sum(sel & (~in_a)))
        tp_d = int(np.sum(sel & in_a & (y_val == 1.0)))
        fp_d = int(np.sum(sel & in_a & (y_val == 0.0)))
        pos_d = int(np.sum(in_a & (y_val == 1.0)))
        excl_deliv = int(np.sum(reasons == "KNOWN_DELIVERED_BEFORE_DECISION"))
        excl_overdue = int(np.sum(reasons == "ALREADY_OVERDUE_BEFORE_DECISION"))

        rows.append(
            {
                "split_id": str(split_id),
                "model_name": str(model_name),
                "decision_date": str(d_str),
                "policy_version": str(grp["policy_version"].iloc[0]),
                "primary_budget_k": float(eff_budget),
                "budget_k": float(eff_budget),
                "total_orders_d": n_total_d,
                "total_orders_placed": n_total_d,
                "cohort_a_orders_d": n_a_d,
                "unlabeled_orders_d": n_unlabeled_d,
                "excluded_delivered_before_decision_d": excl_deliv,
                "excluded_already_delivered": excl_deliv,
                "excluded_already_overdue_d": excl_overdue,
                "separated_already_overdue": excl_overdue,
                "eligible_orders_d": n_elig_d,
                "eligible_orders_n": n_elig_d,
                "daily_quota_q": q_d,
                "daily_quota_q_d": q_d,
                "zero_quota_day": bool(q_d == 0),
                "selected_alerts_d": sel_cnt,
                "alerts_selected": sel_cnt,
                "selected_in_cohort_a_d": sel_in_a,
                "selected_unlabeled_d": sel_unlabeled,
                "total_positives_in_cohort_a_d": pos_d,
                "selected_tp_d": tp_d,
                "selected_fp_d": fp_d,
                "implicit_score_threshold_d": round(float(np.min(scores_sel)), 6) if sel_cnt > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_topk_frontier_cohort_a(
    order_ids: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    budgets: list[float],
    model_name: str = "selected_model",
    split_id: str = "historical_backtest",
    label_name: str = "calendar_day",
    purchase_timestamps: pd.Series | np.ndarray | None = None,
    sample_draw_ids: pd.Series | np.ndarray | None = None,
    daily_allocation: bool = False,
) -> pd.DataFrame:
    """Evaluate capacity-constrained Top-k% alert policies on conditional delivered Cohort A.

    Uses `floor(k * n)` quota rounding (`0` alerts allowed when `k * n < 1`).
    When `daily_allocation=True` and `purchase_timestamps` is provided, allocates `floor(k * N_d)` per day `d`.
    """
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=np.float64)
    ranked = rank_orders_deterministically(
        order_ids=order_ids,
        scores=s_arr,
        purchase_timestamps=purchase_timestamps,
        sample_draw_ids=sample_draw_ids,
    )
    ranks = ranked["risk_rank"].to_numpy()

    n_total = len(y_arr)
    total_pos = int(np.sum(y_arr == 1))
    total_neg = int(np.sum(y_arr == 0))
    base_rate = float(total_pos / max(1, n_total))

    rows: list[dict[str, Any]] = []
    for k in sorted(budgets):
        if daily_allocation and purchase_timestamps is not None and n_total > 0:
            tmp_df = pd.DataFrame(
                {
                    "order_id": np.asarray(order_ids, dtype=str),
                    "order_purchase_timestamp": purchase_timestamps,
                }
            )
            if sample_draw_ids is not None:
                tmp_df["sample_draw_id"] = sample_draw_ids
            alloc = allocate_daily_alerts(
                df=tmp_df,
                scores=s_arr,
                primary_budget=float(k),
                exclude_known_delivered_before_decision=False,
                separate_already_overdue_before_decision=False,
            )
            alert_mask = alloc["selected_top10"].to_numpy(dtype=bool)
            k_count = int(np.sum(alert_mask))
        else:
            k_count = int(np.floor(k * n_total + 1e-12)) if (n_total > 0 and k > 0) else 0
            alert_mask = ranks <= k_count

        tp = int(np.sum((y_arr == 1) & alert_mask))
        fp = int(np.sum((y_arr == 0) & alert_mask))
        fn = total_pos - tp
        tn = total_neg - fp

        prec = float(tp / max(1, k_count)) if k_count > 0 else 0.0
        rec = float(tp / max(1, total_pos)) if total_pos > 0 else 0.0
        lift = float(prec / base_rate) if (base_rate > 0 and k_count > 0) else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        thresh = float(np.min(s_arr[alert_mask])) if k_count > 0 else 1.0
        mean_score_selected = float(np.mean(s_arr[alert_mask])) if k_count > 0 else 0.0

        rows.append(
            {
                "split_id": split_id,
                "population_path": "Path_A_conditional_delivered",
                "model_name": model_name,
                "label_version": label_name,
                "budget_k": float(k),
                "budget_pct_label": f"Top-{int(round(k * 100))}%",
                "population_n": n_total,
                "total_positives": total_pos,
                "base_late_rate": round(base_rate, 6),
                "alert_count": k_count,
                "alert_budget_k": k_count,
                "actual_alert_rate": round(float(k_count / max(1, n_total)), 6),
                "implicit_score_threshold": round(thresh, 6),
                "mean_predicted_prob_in_alerts": round(mean_score_selected, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "lift": round(lift, 6),
                "f1": round(f1, 6),
                "false_alarms_per_true_alert": round(float(fp / max(1, tp)), 4) if tp > 0 else 0.0,
                "orders_screened_per_late_caught": round(float(k_count / max(1, tp)), 4) if tp > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_topk_frontier_path_b(
    df_path_b: pd.DataFrame,
    scores: pd.Series | np.ndarray,
    budgets: list[float],
    model_name: str = "selected_model",
    split_id: str = "historical_backtest",
    label_col: str = "late",
    daily_allocation: bool = False,
) -> pd.DataFrame:
    """Evaluate capacity-constrained Top-k% alert policies on Path B (all scorable orders at purchase time).

    Explicitly separates:
    - `alert_in_cohort_a_count` (orders whose delivery outcome is observed ex-post)
    - `alert_unlabeled_non_delivered_count` (orders that remained shipped, canceled, unavailable, invoiced, etc.)
    Never coerces unlabeled outcomes to 0 and never mixes Path A and Path B denominators.
    Uses `floor(k * n)` rounding (`0` alerts allowed when `k * n < 1`).
    """
    s_arr = np.asarray(scores, dtype=np.float64)
    purchase_ts = (
        df_path_b["order_purchase_timestamp"]
        if "order_purchase_timestamp" in df_path_b.columns
        else None
    )
    ranked = rank_orders_deterministically(
        order_ids=df_path_b["order_id"],
        scores=s_arr,
        purchase_timestamps=purchase_ts,
    )
    ranks = ranked["risk_rank"].to_numpy()

    in_a = df_path_b["in_cohort_a"].fillna(False).to_numpy(dtype=bool)
    has_items = df_path_b["has_items"].fillna(False).to_numpy(dtype=bool)
    y_raw = df_path_b[label_col]
    y_obs = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=np.float64)
    status_arr = df_path_b["order_status"].astype(str).to_numpy()

    n_total_b = len(df_path_b)
    n_total_a = int(np.sum(in_a))
    n_total_unlabeled = n_total_b - n_total_a
    total_obs_pos = int(np.nansum(y_obs[in_a] == 1.0))

    rows: list[dict[str, Any]] = []
    for k in sorted(budgets):
        if daily_allocation and purchase_ts is not None and n_total_b > 0:
            alloc = allocate_daily_alerts(
                df=df_path_b,
                scores=s_arr,
                primary_budget=float(k),
            )
            alert_mask = alloc["selected_top10"].to_numpy(dtype=bool)
            k_count = int(np.sum(alert_mask))
        else:
            k_count = int(np.floor(k * n_total_b + 1e-12)) if (n_total_b > 0 and k > 0) else 0
            alert_mask = ranks <= k_count

        alert_in_a = int(np.sum(alert_mask & in_a))
        alert_unlabeled = int(np.sum(alert_mask & (~in_a)))
        alert_without_items = int(np.sum(alert_mask & (~has_items)))

        tp_obs = int(np.sum(alert_mask & in_a & (y_obs == 1.0)))
        fp_obs = int(np.sum(alert_mask & in_a & (y_obs == 0.0)))

        alert_shipped_undelivered = int(np.sum(alert_mask & (~in_a) & (status_arr == "shipped")))
        alert_canceled_or_unavail = int(
            np.sum(alert_mask & (~in_a) & np.isin(status_arr, ["canceled", "unavailable"]))
        )

        cond_prec_on_a = float(tp_obs / max(1, alert_in_a)) if alert_in_a > 0 else 0.0
        lb_prec_path_b = float(tp_obs / max(1, k_count)) if k_count > 0 else 0.0
        ub_exception_rate_path_b = float((tp_obs + alert_unlabeled) / max(1, k_count)) if k_count > 0 else 0.0
        cond_recall_on_a = float(tp_obs / max(1, total_obs_pos)) if total_obs_pos > 0 else 0.0
        thresh = float(np.min(s_arr[alert_mask])) if k_count > 0 else 1.0

        rows.append(
            {
                "split_id": split_id,
                "population_path": "Path_B_all_scorable_orders",
                "model_name": model_name,
                "budget_k": float(k),
                "budget_pct_label": f"Top-{int(round(k * 100))}%",
                "path_b_total_orders": n_total_b,
                "path_a_observed_orders_in_window": n_total_a,
                "unlabeled_non_delivered_orders_in_window": n_total_unlabeled,
                "alert_total_path_b": k_count,
                "actual_alert_rate_path_b": round(float(k_count / max(1, n_total_b)), 6),
                "implicit_score_threshold": round(thresh, 6),
                "alert_in_cohort_a_count": alert_in_a,
                "alert_unlabeled_non_delivered_count": alert_unlabeled,
                "alert_shipped_undelivered_count": alert_shipped_undelivered,
                "alert_canceled_or_unavailable_count": alert_canceled_or_unavail,
                "alert_without_items_count": alert_without_items,
                "observed_tp_in_cohort_a": tp_obs,
                "observed_fp_in_cohort_a": fp_obs,
                "conditional_precision_among_observed": round(cond_prec_on_a, 6),
                "conditional_recall_on_cohort_a": round(cond_recall_on_a, 6),
                "lower_bound_precision_path_b": round(lb_prec_path_b, 6),
                "upper_bound_exception_rate_path_b": round(ub_exception_rate_path_b, 6),
            }
        )
    return pd.DataFrame(rows)


def evaluate_cost_ratio_sensitivity(
    y_true: pd.Series | np.ndarray,
    scores_calibrated: pd.Series | np.ndarray,
    scores_uncalibrated: pd.Series | np.ndarray | None = None,
    cost_ratios: list[float] | None = None,
    model_name: str = "selected_model",
    split_id: str = "historical_backtest",
    scores_random_baseline: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate parametric cost-ratio sensitivity (`p* = c_alert / (p_mitigate * L_late)`).

    Note: All utilities are normalized in units of `p_mitigate * L_late = 1.0` per 1,000 orders and
    explicitly marked as a parametric sensitivity analysis rather than observed Olist financial returns.
    """
    ratios = cost_ratios or [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    y_arr = np.asarray(y_true, dtype=int)
    p_cal = np.asarray(scores_calibrated, dtype=np.float64)
    p_uncal = (
        np.asarray(scores_uncalibrated, dtype=np.float64)
        if scores_uncalibrated is not None
        else p_cal
    )
    n_total = max(1, len(y_arr))
    total_pos = max(1, int(np.sum(y_arr == 1)))
    base_rate = float(np.sum(y_arr == 1) / n_total)
    k_rand10 = int(np.floor(0.10 * len(y_arr) + 1e-12))

    rows: list[dict[str, Any]] = []
    for r in sorted(ratios):
        # Reference utilities per 1,000 orders:
        # treat_none = 0.0
        # treat_all = 1000 * (base_rate - r)
        # random_top10 = 1000 * 0.10 * (base_rate - r)
        treat_all_net = 1000.0 * (base_rate - float(r))
        random_top10_net = 1000.0 * 0.10 * (base_rate - float(r))

        for variant_label, prob_arr in [("calibrated", p_cal), ("uncalibrated", p_uncal)]:
            mask = prob_arr >= r
            alert_cnt = int(np.sum(mask))
            tp = int(np.sum((y_arr == 1) & mask))
            fp = int(np.sum((y_arr == 0) & mask))
            prec = float(tp / max(1, alert_cnt)) if alert_cnt > 0 else 0.0
            rec = float(tp / total_pos)
            net_utility_per_1k = 1000.0 * (tp - r * alert_cnt) / float(n_total)

            rows.append(
                {
                    "split_id": split_id,
                    "model_name": model_name,
                    "probability_variant": variant_label,
                    "cost_ratio_p_star": float(r),
                    "alert_count": alert_cnt,
                    "alert_rate": round(float(alert_cnt / n_total), 6),
                    "tp": tp,
                    "fp": fp,
                    "precision": round(prec, 6),
                    "recall": round(rec, 6),
                    "parametric_net_utility_per_1000_orders": round(net_utility_per_1k, 4),
                    "treat_all_net_utility_per_1000_orders": round(treat_all_net, 4),
                    "random_top10_net_utility_per_1000_orders": round(random_top10_net, 4),
                    "beats_no_action": bool(net_utility_per_1k > 0.0),
                    "beats_random_top10": bool(net_utility_per_1k > random_top10_net),
                }
            )

        rows.append(
            {
                "split_id": split_id,
                "model_name": model_name,
                "probability_variant": "treat_all",
                "cost_ratio_p_star": float(r),
                "alert_count": int(len(y_arr)),
                "alert_rate": 1.0,
                "tp": int(np.sum(y_arr == 1)),
                "fp": int(np.sum(y_arr == 0)),
                "precision": round(base_rate, 6),
                "recall": 1.0,
                "parametric_net_utility_per_1000_orders": round(treat_all_net, 4),
                "treat_all_net_utility_per_1000_orders": round(treat_all_net, 4),
                "random_top10_net_utility_per_1000_orders": round(random_top10_net, 4),
                "beats_no_action": bool(treat_all_net > 0.0),
                "beats_random_top10": bool(treat_all_net > random_top10_net),
            }
        )
        rows.append(
            {
                "split_id": split_id,
                "model_name": model_name,
                "probability_variant": "random_top10",
                "cost_ratio_p_star": float(r),
                "alert_count": k_rand10,
                "alert_rate": round(float(k_rand10 / n_total), 6),
                "tp": int(round(0.10 * np.sum(y_arr == 1))),
                "fp": max(0, k_rand10 - int(round(0.10 * np.sum(y_arr == 1)))),
                "precision": round(base_rate, 6),
                "recall": 0.10,
                "parametric_net_utility_per_1000_orders": round(random_top10_net, 4),
                "treat_all_net_utility_per_1000_orders": round(treat_all_net, 4),
                "random_top10_net_utility_per_1000_orders": round(random_top10_net, 4),
                "beats_no_action": bool(random_top10_net > 0.0),
                "beats_random_top10": False,
            }
        )
    return pd.DataFrame(rows)
