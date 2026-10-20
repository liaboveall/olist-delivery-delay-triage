"""Metrics, calibration diagnostics, paired weekly-block & row bootstrap CIs, subgroup/drift analysis, permutation importance, and ex-post diagnostics."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .policy import (
    allocate_daily_alerts,
    evaluate_topk_frontier_cohort_a,
    rank_orders_deterministically,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return 0.0

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(y_p, quantiles))
        if len(edges) <= 2:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    edges[0] = -1e-9
    edges[-1] = 1.0 + 1e-9
    bin_ids = np.digitize(y_p, edges[1:-1], right=True)

    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        obs = float(np.mean(y_t[mask]))
        pred = float(np.mean(y_p[mask]))
        ece += (cnt / float(n)) * abs(obs - pred)
    return float(ece)


def compute_classification_metrics(
    order_ids: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    model_name: str,
    calibration_method: str,
    split_id: str,
    label_version: str = "calendar_day",
    purchase_timestamps: pd.Series | np.ndarray | None = None,
    feature_version: str = "extended_purchase_assumptions_v2",
    candidate_id: str | None = None,
    hyperparams_json: str = "{}",
    calibration_status: Any = None,
) -> dict[str, Any]:
    """Compute ranking, probability calibration, and operational Top-k% metrics on an evaluable cohort.

    Never fabricates `roc_auc=0.5` or `pr_auc` on empty or single-class inputs (`metrics_status` is explicitly set).
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    n = len(y_arr)
    pos = int(np.sum(y_arr == 1))
    base_rate = float(pos / max(1, n)) if n > 0 else 0.0
    cal_status_str = (
        str(calibration_status.get(calibration_method, "OK"))
        if isinstance(calibration_status, dict)
        else str(calibration_status or "OK")
    )

    if n == 0:
        return {
            "split_id": split_id,
            "candidate_id": candidate_id or model_name,
            "model_name": model_name,
            "feature_version": feature_version,
            "hyperparams_json": hyperparams_json,
            "calibration_method": calibration_method,
            "calibration_status": cal_status_str,
            "label_version": label_version,
            "metrics_status": "EMPTY_SAMPLE",
            "single_class_degenerate": True,
            "n_samples": 0,
            "n_positives": 0,
            "observed_late_rate": None,
            "mean_predicted_prob": None,
            "calibration_ratio_pred_to_obs": None,
            "pr_auc": None,
            "roc_auc": None,
            "brier_score": None,
            "log_loss": None,
            "ece_quantile_10": None,
            "ece_uniform_10": None,
            "precision_top5": 0.0,
            "recall_top5": 0.0,
            "lift_top5": 0.0,
            "precision_top10": 0.0,
            "recall_top10": 0.0,
            "lift_top10": 0.0,
            "precision_top20": 0.0,
            "recall_top20": 0.0,
            "lift_top20": 0.0,
            "daily_top10_alerts": 0,
            "daily_top10_alert_count": 0,
            "daily_top10_tp": 0,
            "daily_top10_fp": 0,
            "daily_top10_precision": 0.0,
            "daily_top10_recall": 0.0,
        }

    has_two_classes = len(np.unique(y_arr)) >= 2
    pr_auc = round(float(average_precision_score(y_arr, p_arr)), 6) if has_two_classes else None
    roc_auc = round(float(roc_auc_score(y_arr, p_arr)), 6) if has_two_classes else None
    brier = float(brier_score_loss(y_arr, p_arr))
    ll = float(log_loss(y_arr, p_arr, labels=[0, 1]))
    ece_q = compute_ece(y_arr, p_arr, n_bins=10, strategy="quantile")
    ece_u = compute_ece(y_arr, p_arr, n_bins=10, strategy="uniform")
    mean_pred = float(np.mean(p_arr))

    frontier = evaluate_topk_frontier_cohort_a(
        order_ids=order_ids,
        y_true=y_arr,
        scores=p_arr,
        budgets=[0.05, 0.10, 0.15, 0.20, 0.30],
        model_name=model_name,
        split_id=split_id,
        label_name=label_version,
        purchase_timestamps=purchase_timestamps,
        daily_allocation=False,
    ).set_index("budget_pct_label")

    def _get_f(label: str, col: str) -> float:
        return float(frontier.loc[label, col]) if label in frontier.index else 0.0

    if purchase_timestamps is not None:
        daily_frontier = evaluate_topk_frontier_cohort_a(
            order_ids=order_ids,
            y_true=y_arr,
            scores=p_arr,
            budgets=[0.10],
            model_name=model_name,
            split_id=split_id,
            label_name=label_version,
            purchase_timestamps=purchase_timestamps,
            daily_allocation=True,
        ).iloc[0]
        daily_alert_cnt = int(daily_frontier["alert_count"])
        daily_tp = int(daily_frontier["tp"])
        daily_fp = int(daily_frontier["fp"])
        daily_prec = float(daily_frontier["precision"])
        daily_rec = float(daily_frontier["recall"])
    else:
        daily_alert_cnt = int(frontier.loc["Top-10%", "alert_count"])
        daily_tp = int(frontier.loc["Top-10%", "tp"])
        daily_fp = int(frontier.loc["Top-10%", "fp"])
        daily_prec = _get_f("Top-10%", "precision")
        daily_rec = _get_f("Top-10%", "recall")

    return {
        "split_id": split_id,
        "candidate_id": candidate_id or model_name,
        "model_name": model_name,
        "feature_version": feature_version,
        "hyperparams_json": hyperparams_json,
        "calibration_method": calibration_method,
        "calibration_status": cal_status_str,
        "label_version": label_version,
        "metrics_status": "OK" if has_two_classes else "DEGENERATE_SINGLE_CLASS",
        "single_class_degenerate": not has_two_classes,
        "n_samples": n,
        "n_positives": pos,
        "observed_late_rate": round(base_rate, 6),
        "mean_predicted_prob": round(mean_pred, 6),
        "calibration_ratio_pred_to_obs": round(float(mean_pred / max(base_rate, 1e-6)), 4),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": round(brier, 6),
        "log_loss": round(ll, 6),
        "ece_quantile_10": round(ece_q, 6),
        "ece_uniform_10": round(ece_u, 6),
        "precision_top5": round(_get_f("Top-5%", "precision"), 6),
        "recall_top5": round(_get_f("Top-5%", "recall"), 6),
        "lift_top5": round(_get_f("Top-5%", "lift"), 6),
        "precision_top10": round(_get_f("Top-10%", "precision"), 6),
        "recall_top10": round(_get_f("Top-10%", "recall"), 6),
        "lift_top10": round(_get_f("Top-10%", "lift"), 6),
        "precision_top20": round(_get_f("Top-20%", "precision"), 6),
        "recall_top20": round(_get_f("Top-20%", "recall"), 6),
        "lift_top20": round(_get_f("Top-20%", "lift"), 6),
        "daily_top10_alerts": daily_alert_cnt,
        "daily_top10_alert_count": daily_alert_cnt,
        "daily_top10_tp": daily_tp,
        "daily_top10_fp": daily_fp,
        "daily_top10_precision": round(daily_prec, 6),
        "daily_top10_recall": round(daily_rec, 6),
    }


def compute_calibration_curve_table(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    model_name: str,
    calibration_method: str,
    split_id: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute decile reliability table for calibration plots and verification."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p_arr, quantiles))
    if len(edges) <= 2:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0] = 0.0
    edges[-1] = 1.0 + 1e-9
    bin_ids = np.digitize(p_arr, edges[1:-1], right=True)

    rows: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        obs = float(np.mean(y_arr[mask]))
        pred = float(np.mean(p_arr[mask]))
        rows.append(
            {
                "split_id": split_id,
                "model_name": model_name,
                "calibration_method": calibration_method,
                "bin_index": b + 1,
                "bin_lower": round(float(np.min(p_arr[mask])), 6),
                "bin_upper": round(float(np.max(p_arr[mask])), 6),
                "sample_count": cnt,
                "positives_count": int(np.sum(y_arr[mask] == 1.0)),
                "mean_predicted_prob": round(pred, 6),
                "observed_late_rate": round(obs, 6),
                "abs_calibration_gap": round(abs(pred - obs), 6),
            }
        )
    return pd.DataFrame(rows)


def _select_top_k_deterministic(
    scores: np.ndarray,
    order_ids: np.ndarray,
    purchase_ts_ns: np.ndarray,
    sample_draw_ids: np.ndarray,
    k_count: int,
) -> np.ndarray:
    """Return boolean mask of top `k_count` items using contract tie-breaking `(-score, +ts, +order_id, +draw_id)`."""
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    if k_count <= 0 or n == 0:
        return mask
    order = np.lexsort((sample_draw_ids, order_ids, purchase_ts_ns, -scores))
    mask[order[:k_count]] = True
    return mask


def _select_daily_top_k_deterministic(
    scores: np.ndarray,
    order_ids: np.ndarray,
    purchase_ts_ns: np.ndarray,
    day_instance_keys: np.ndarray,
    sample_draw_ids: np.ndarray,
    budget_k: float,
) -> np.ndarray:
    """Select `floor(budget_k * N_d)` within each `(block_draw_instance, decision_date)` group."""
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or budget_k <= 0:
        return mask
    # Sort by day_instance_keys first to group contiguous slices
    group_order = np.argsort(day_instance_keys, kind="mergesort")
    sorted_keys = day_instance_keys[group_order]
    _, split_starts = np.unique(sorted_keys, return_index=True)
    for start, end in zip(split_starts, np.append(split_starts[1:], n)):
        grp_idx = group_order[start:end]
        n_d = len(grp_idx)
        q_d = int(np.floor(budget_k * n_d + 1e-12))
        if q_d <= 0:
            continue
        local_order = np.lexsort(
            (
                sample_draw_ids[grp_idx],
                order_ids[grp_idx],
                purchase_ts_ns[grp_idx],
                -scores[grp_idx],
            )
        )
        mask[grp_idx[local_order[:q_d]]] = True
    return mask


def compute_bootstrap_ci(
    y_true: pd.Series | np.ndarray,
    y_prob_model: pd.Series | np.ndarray,
    y_prob_baseline: pd.Series | np.ndarray | None = None,
    n_bootstrap: int = 400,
    random_seed: int = 5126,
    order_ids: pd.Series | np.ndarray | None = None,
    purchase_timestamps: pd.Series | np.ndarray | None = None,
    n_weekly_bootstrap: int | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """Compute paired bootstrap 95% confidence intervals (both row bootstrap and weekly block bootstrap).

    Enforces:
    - Shared resampling indices between model and baseline (`paired_resampling = True`).
    - Deterministic tie-breaking via `(-score, +purchase_ts, +order_id, +sample_draw_id)` instead of `argpartition`.
    - `floor(k * n)` quota rounding.
    - When `purchase_timestamps` is provided, also computes paired weekly block bootstrap (`weekly_block_bootstrap`)
      where repeated weekly blocks preserve distinct `(block_draw_instance, decision_date)` daily batches.
    """
    rng = np.random.default_rng(random_seed)
    y_arr = np.asarray(y_true, dtype=int)
    p_mod = np.asarray(y_prob_model, dtype=np.float64)
    p_base = np.asarray(y_prob_baseline, dtype=np.float64) if y_prob_baseline is not None else None
    n = len(y_arr)

    if order_ids is not None:
        oid_arr = np.asarray(order_ids, dtype=str)
    else:
        oid_arr = np.array([f"ord_{i:06d}" for i in range(n)], dtype=str)

    if purchase_timestamps is not None:
        ts_series = pd.to_datetime(pd.Series(purchase_timestamps), errors="coerce")
        ts_ns_arr = ts_series.astype("int64").to_numpy()
        day_str_arr = ts_series.dt.strftime("%Y-%m-%d").fillna("INVALID").to_numpy(dtype=str)
        week_str_arr = ts_series.dt.to_period("W").astype(str).to_numpy(dtype=str)
    else:
        ts_ns_arr = np.zeros(n, dtype=np.int64)
        day_str_arr = np.full(n, "SINGLE_BATCH", dtype=str)
        week_str_arr = np.full(n, "SINGLE_WEEK", dtype=str)

    k10 = int(np.floor(0.10 * n + 1e-12))
    k20 = int(np.floor(0.20 * n + 1e-12))

    pr_aucs, roc_aucs, briers, prec10s, rec10s, prec20s, rec20s = [], [], [], [], [], [], []
    diff_pr_aucs, diff_prec10s, diff_rec10s = [], [], []
    draw_pos_seq = np.arange(n, dtype=np.int64)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        pm = p_mod[idx]
        oids_b = oid_arr[idx]
        ts_b = ts_ns_arr[idx]
        total_pos = max(1, int(np.sum(yb == 1)))

        pr_m = float(average_precision_score(yb, pm))
        roc_m = float(roc_auc_score(yb, pm))
        br_m = float(brier_score_loss(yb, pm))

        top10_mask = _select_top_k_deterministic(pm, oids_b, ts_b, draw_pos_seq, k10)
        tp10 = int(np.sum(yb[top10_mask] == 1))
        p10_m = float(tp10 / max(1, k10)) if k10 > 0 else 0.0
        r10_m = float(tp10 / total_pos)

        top20_mask = _select_top_k_deterministic(pm, oids_b, ts_b, draw_pos_seq, k20)
        tp20 = int(np.sum(yb[top20_mask] == 1))
        p20_m = float(tp20 / max(1, k20)) if k20 > 0 else 0.0
        r20_m = float(tp20 / total_pos)

        pr_aucs.append(pr_m)
        roc_aucs.append(roc_m)
        briers.append(br_m)
        prec10s.append(p10_m)
        rec10s.append(r10_m)
        prec20s.append(p20_m)
        rec20s.append(r20_m)

        if p_base is not None:
            pb = p_base[idx]
            pr_b = float(average_precision_score(yb, pb))
            top10_b = _select_top_k_deterministic(pb, oids_b, ts_b, draw_pos_seq, k10)
            tp10_b = int(np.sum(yb[top10_b] == 1))
            p10_b = float(tp10_b / max(1, k10)) if k10 > 0 else 0.0
            r10_b = float(tp10_b / total_pos)
            diff_pr_aucs.append(pr_m - pr_b)
            diff_prec10s.append(p10_m - p10_b)
            diff_rec10s.append(r10_m - r10_b)

    def _ci_summary(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "valid_draws": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else 0.0,
            "ci95_low": round(float(np.quantile(arr, 0.025)), 6),
            "ci95_high": round(float(np.quantile(arr, 0.975)), 6),
            "valid_draws": int(len(arr)),
        }

    out: dict[str, Any] = {
        "method": method or ("weekly_block_bootstrap" if purchase_timestamps is not None else "iid_row_bootstrap"),
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
        "paired_resampling": True,
        "tie_break_order": ["risk_score_desc", "order_purchase_timestamp_asc", "order_id_asc", "sample_draw_id_asc"],
        "metrics": {
            "pr_auc": _ci_summary(pr_aucs),
            "roc_auc": _ci_summary(roc_aucs),
            "brier_score": _ci_summary(briers),
            "precision_top10": _ci_summary(prec10s),
            "recall_top10": _ci_summary(rec10s),
            "precision_top20": _ci_summary(prec20s),
            "recall_top20": _ci_summary(rec20s),
        },
    }
    if p_base is not None and diff_pr_aucs:
        out["paired_difference_vs_baseline"] = {
            "pr_auc_diff": _ci_summary(diff_pr_aucs),
            "precision_top10_diff": _ci_summary(diff_prec10s),
            "recall_top10_diff": _ci_summary(diff_rec10s),
            "daily_top10_recall_diff": _ci_summary(diff_rec10s),
            "daily_top10_precision_diff": _ci_summary(diff_prec10s),
            "prob_model_beats_baseline_pr_auc": round(float(np.mean(np.asarray(diff_pr_aucs) > 0)), 4),
            "ci95_crosses_zero_recall_top10": bool(
                np.quantile(diff_rec10s, 0.025) <= 0.0 <= np.quantile(diff_rec10s, 0.975)
            ),
        }

    # Weekly block bootstrap when purchase_timestamps is provided
    if purchase_timestamps is not None:
        unique_weeks = np.unique(week_str_arr)
        n_weeks = len(unique_weeks)
        out["n_blocks"] = int(n_weeks)
        week_to_indices = {w: np.where(week_str_arr == w)[0] for w in unique_weeks}
        n_wb = int(n_weekly_bootstrap if n_weekly_bootstrap is not None else n_bootstrap)
        wb_rng = np.random.default_rng(random_seed + 17)

        wb_rec10_m, wb_prec10_m, wb_pr_m = [], [], []
        wb_diff_rec10, wb_diff_prec10, wb_diff_pr = [], [], []

        for _ in range(n_wb):
            drawn_weeks = wb_rng.choice(unique_weeks, size=n_weeks, replace=True)
            idx_chunks = []
            day_key_chunks = []
            for draw_inst_id, w_key in enumerate(drawn_weeks):
                w_idx = week_to_indices[w_key]
                idx_chunks.append(w_idx)
                # Preserve distinct daily batch per drawn block instance
                day_key_chunks.append(np.char.add(f"blk{draw_inst_id:04d}_", day_str_arr[w_idx]))

            b_idx = np.concatenate(idx_chunks)
            b_day_keys = np.concatenate(day_key_chunks)
            b_draw_ids = np.arange(len(b_idx), dtype=np.int64)

            yb = y_arr[b_idx]
            if len(np.unique(yb)) < 2:
                continue
            pm = p_mod[b_idx]
            oids_b = oid_arr[b_idx]
            ts_b = ts_ns_arr[b_idx]
            total_pos = max(1, int(np.sum(yb == 1)))

            sel_m = _select_daily_top_k_deterministic(pm, oids_b, ts_b, b_day_keys, b_draw_ids, 0.10)
            q_m = int(np.sum(sel_m))
            tp_m = int(np.sum(yb[sel_m] == 1))
            prec_m = float(tp_m / max(1, q_m)) if q_m > 0 else 0.0
            rec_m = float(tp_m / total_pos)
            pr_val_m = float(average_precision_score(yb, pm))

            wb_rec10_m.append(rec_m)
            wb_prec10_m.append(prec_m)
            wb_pr_m.append(pr_val_m)

            if p_base is not None:
                pb = p_base[b_idx]
                sel_b = _select_daily_top_k_deterministic(pb, oids_b, ts_b, b_day_keys, b_draw_ids, 0.10)
                q_b = int(np.sum(sel_b))
                tp_b = int(np.sum(yb[sel_b] == 1))
                prec_b = float(tp_b / max(1, q_b)) if q_b > 0 else 0.0
                rec_b = float(tp_b / total_pos)
                pr_val_b = float(average_precision_score(yb, pb))

                wb_diff_rec10.append(rec_m - rec_b)
                wb_diff_prec10.append(prec_m - prec_b)
                wb_diff_pr.append(pr_val_m - pr_val_b)

        if p_base is not None and wb_diff_rec10 and "paired_difference_vs_baseline" in out:
            out["paired_difference_vs_baseline"]["daily_top10_recall_diff"] = _ci_summary(wb_diff_rec10)
            out["paired_difference_vs_baseline"]["daily_top10_precision_diff"] = _ci_summary(wb_diff_prec10)

        out["weekly_block_bootstrap"] = {
            "n_weekly_bootstrap": n_wb,
            "n_unique_weeks": int(n_weeks),
            "daily_policy_recall_top10": _ci_summary(wb_rec10_m),
            "daily_policy_precision_top10": _ci_summary(wb_prec10_m),
            "pr_auc": _ci_summary(wb_pr_m),
            "paired_difference_vs_baseline": (
                {
                    "daily_policy_recall_top10_diff": _ci_summary(wb_diff_rec10),
                    "daily_policy_precision_top10_diff": _ci_summary(wb_diff_prec10),
                    "pr_auc_diff": _ci_summary(wb_diff_pr),
                    "ci95_crosses_zero_daily_recall_top10": bool(
                        np.quantile(wb_diff_rec10, 0.025) <= 0.0 <= np.quantile(wb_diff_rec10, 0.975)
                    )
                    if wb_diff_rec10
                    else True,
                }
                if (p_base is not None and wb_diff_rec10)
                else {}
            ),
        }

    return out


def compute_subgroup_metrics(
    df_eval: pd.DataFrame,
    scores: pd.Series | np.ndarray,
    label_col: str = "late",
    primary_budget: float = 0.10,
) -> pd.DataFrame:
    """Evaluate ranking, calibration, and alert capture across operational subgroups on Cohort A."""
    work = df_eval.copy()
    work["_score"] = np.asarray(scores, dtype=np.float64)
    work["_y"] = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int)

    # Use daily allocation if order_purchase_timestamp is present, else floor(k*N)
    if "order_purchase_timestamp" in work.columns:
        alloc = allocate_daily_alerts(work, work["_score"], primary_budget=primary_budget)
        work["_global_top10_alert"] = alloc["selected_top10"].to_numpy(dtype=bool)
    else:
        k_global = int(np.floor(primary_budget * len(work) + 1e-12))
        ranked = rank_orders_deterministically(work["order_id"], work["_score"])
        work["_global_top10_alert"] = (ranked["risk_rank"].to_numpy() <= k_global)

    # Define subgroup dimensions
    work["distance_band"] = pd.cut(
        work["haversine_distance_km"].fillna(-1.0),
        bins=[-2.0, 0.0, 250.0, 750.0, 1500.0, 1e6],
        labels=["missing", "0-250km", "250-750km", "750-1500km", ">1500km"],
    ).astype(str)
    seller_cnt_col = (
        "seller_prior_Deliveries_count"
        if "seller_prior_Deliveries_count" in work.columns
        else "seller_completed_count"
    )
    work["seller_maturity_tier"] = np.where(
        work[seller_cnt_col] < 5,
        "cold_start_<5_completed",
        np.where(work[seller_cnt_col] < 50, "established_5_to_49", "high_volume_50plus"),
    )
    work["basket_tier"] = np.where(work["item_count"] > 1, "multi_item", "single_item")
    work["route_corridor_type"] = np.where(work["same_state"] == 1, "intra_state", "inter_state")
    work["promised_lead_band"] = pd.cut(
        work["estimated_lead_calendar_days"].fillna(20.0),
        bins=[-100.0, 10.0, 18.0, 25.0, 35.0, 1000.0],
        labels=["<=10d", "11-18d", "19-25d", "26-35d", ">35d"],
    ).astype(str)

    dimensions = [
        "customer_region",
        "distance_band",
        "seller_maturity_tier",
        "basket_tier",
        "route_corridor_type",
        "promised_lead_band",
    ]

    rows: list[dict[str, Any]] = []
    for dim in dimensions:
        for grp_val, sub in work.groupby(dim, dropna=False):
            n_sub = len(sub)
            if n_sub < 10:
                continue
            y_sub = sub["_y"].to_numpy()
            p_sub = sub["_score"].to_numpy()
            pos_sub = int(np.sum(y_sub == 1))
            base_rate = float(pos_sub / n_sub)
            has_both = len(np.unique(y_sub)) >= 2

            pr_auc = float(average_precision_score(y_sub, p_sub)) if has_both else base_rate
            roc_auc = float(roc_auc_score(y_sub, p_sub)) if has_both else 0.5
            brier = float(brier_score_loss(y_sub, p_sub))

            k_sub = int(np.floor(primary_budget * n_sub + 1e-12))
            ranked_sub = rank_orders_deterministically(sub["order_id"], p_sub)
            sub_sel = ranked_sub["risk_rank"].to_numpy() <= k_sub
            tp_within = int(np.sum(y_sub[sub_sel] == 1))
            prec_within = float(tp_within / max(1, k_sub)) if k_sub > 0 else 0.0
            rec_within = float(tp_within / max(1, pos_sub))

            global_alerts_in_sub = int(sub["_global_top10_alert"].sum())
            tp_global_in_sub = int(np.sum((y_sub == 1) & sub["_global_top10_alert"].to_numpy()))

            rows.append(
                {
                    "subgroup_dimension": dim,
                    "subgroup_value": str(grp_val),
                    "n_orders": n_sub,
                    "share_of_cohort": round(float(n_sub / len(work)), 6),
                    "n_late": pos_sub,
                    "low_sample_warning": bool(n_sub < 100 or pos_sub < 15),
                    "observed_late_rate": round(base_rate, 6),
                    "mean_predicted_prob": round(float(np.mean(p_sub)), 6),
                    "pr_auc": round(pr_auc, 6),
                    "roc_auc": round(roc_auc, 6),
                    "brier_score": round(brier, 6),
                    "within_group_precision_top10": round(prec_within, 6),
                    "within_group_recall_top10": round(rec_within, 6),
                    "global_top10_alert_rate_in_group": round(float(global_alerts_in_sub / n_sub), 6),
                    "global_top10_precision_in_group": (
                        round(float(tp_global_in_sub / global_alerts_in_sub), 6)
                        if global_alerts_in_sub > 0
                        else 0.0
                    ),
                    "global_top10_recall_in_group": round(float(tp_global_in_sub / max(1, pos_sub)), 6),
                }
            )
    return pd.DataFrame(rows)


def compute_monthly_drift_report(
    predictions_df: pd.DataFrame,
    score_col: str = "predicted_prob_calibrated",
    label_col: str = "late",
) -> pd.DataFrame:
    """Compute month-by-month temporal drift report across validation and backtest periods."""
    work = predictions_df[predictions_df[label_col].notna()].copy()
    work["year_month"] = pd.to_datetime(work["order_purchase_timestamp"]).dt.strftime("%Y-%m")

    rows: list[dict[str, Any]] = []
    for (split_id, ym), sub in work.groupby(["split_id", "year_month"], sort=True):
        if len(sub) < 10:
            continue
        y_sub = sub[label_col].astype(int).to_numpy()
        p_sub = sub[score_col].astype(float).to_numpy()
        pos = int(np.sum(y_sub == 1))
        base = float(pos / len(sub))
        has_both = len(np.unique(y_sub)) >= 2
        k10 = int(np.floor(0.10 * len(sub) + 1e-12))
        ranked_sub = rank_orders_deterministically(sub["order_id"], p_sub)
        sel10 = ranked_sub["risk_rank"].to_numpy() <= k10
        tp10 = int(np.sum(y_sub[sel10] == 1))
        prec10 = float(tp10 / max(1, k10)) if k10 > 0 else 0.0

        rows.append(
            {
                "split_id": str(split_id),
                "year_month": str(ym),
                "n_orders": int(len(sub)),
                "n_positives": pos,
                "observed_late_rate": round(base, 6),
                "mean_predicted_prob": round(float(np.mean(p_sub)), 6),
                "calibration_gap": round(float(np.mean(p_sub) - base), 6),
                "pr_auc": round(float(average_precision_score(y_sub, p_sub)), 6) if has_both else None,
                "roc_auc": round(float(roc_auc_score(y_sub, p_sub)), 6) if has_both else None,
                "brier_score": round(float(brier_score_loss(y_sub, p_sub)), 6),
                "precision_top10": round(prec10, 6),
                "recall_top10": round(float(tp10 / max(1, pos)), 6),
                "lift_top10": round(float(prec10 / max(base, 1e-6)), 4) if k10 > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_permutation_importance_table(
    estimator: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series | np.ndarray,
    feature_cols: list[str],
    random_seed: int = 5126,
    n_repeats: int = 5,
    max_samples: int = 8000,
) -> pd.DataFrame:
    """Compute out-of-sample permutation importance (`average_precision`) on purchase-time features."""
    X_sub = X_eval[feature_cols].copy()
    y_sub = np.asarray(y_eval, dtype=int)
    if len(X_sub) > max_samples:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(X_sub), size=max_samples, replace=False)
        X_sub = X_sub.iloc[idx].reset_index(drop=True)
        y_sub = y_sub[idx]

    def _ap_scorer(est: Any, X_in: pd.DataFrame, y_in: np.ndarray) -> float:
        proba = np.asarray(est.predict_proba(X_in), dtype=float)
        p1 = proba[:, 1] if proba.ndim == 2 else proba.ravel()
        return float(average_precision_score(y_in, p1))

    perm = permutation_importance(
        estimator,
        X_sub,
        y_sub,
        scoring=_ap_scorer,
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=1,
    )
    df_imp = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "importance_mean_pr_auc_drop": np.round(perm.importances_mean, 6),
            "importance_std": np.round(perm.importances_std, 6),
        }
    ).sort_values("importance_mean_pr_auc_drop", ascending=False, kind="mergesort")
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)
    return df_imp.reset_index(drop=True)


def compute_ex_post_stage_breakdown(master_orders: pd.DataFrame, label_col: str = "late") -> pd.DataFrame:
    """Compute descriptive ex-post fulfillment stage decomposition on Cohort A (`ex_post_stage_summary.csv`).

    Explicitly marked as `DIAGNOSTIC_EX_POST_ONLY` — none of these post-purchase stage durations
    or review scores are ever used in purchase-time prediction.
    Uses a unified purchase-time origin (`order_purchase_timestamp`) when comparing `seller_exceeded_shipping_limit`
    against `max_shipping_limit_days` (or `earliest_shipping_limit_date - order_purchase_timestamp`).
    """
    if "in_cohort_a" in master_orders.columns:
        cohort = master_orders[master_orders["in_cohort_a"]].copy()
    else:
        cohort = master_orders.copy()

    for ts_col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ]:
        if ts_col in cohort.columns:
            cohort[ts_col] = pd.to_datetime(cohort[ts_col], errors="coerce")
        else:
            cohort[ts_col] = pd.Series(pd.NaT, index=cohort.index)

    cohort["approval_hours"] = (
        (cohort["order_approved_at"] - cohort["order_purchase_timestamp"]).dt.total_seconds() / 3600.0
    )
    cohort["seller_to_carrier_days"] = (
        (cohort["order_delivered_carrier_date"] - cohort["order_approved_at"]).dt.total_seconds() / 86400.0
    )
    cohort["purchase_to_carrier_days"] = (
        (cohort["order_delivered_carrier_date"] - cohort["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    )
    cohort["carrier_to_customer_days"] = (
        (cohort["order_delivered_customer_date"] - cohort["order_delivered_carrier_date"]).dt.total_seconds()
        / 86400.0
    )
    cohort["total_fulfillment_days"] = (
        (cohort["order_delivered_customer_date"] - cohort["order_purchase_timestamp"]).dt.total_seconds()
        / 86400.0
    )

    if "max_shipping_limit_days" in cohort.columns:
        limit_days = pd.to_numeric(cohort["max_shipping_limit_days"], errors="coerce")
    elif "earliest_shipping_limit_date" in cohort.columns:
        limit_ts = pd.to_datetime(cohort["earliest_shipping_limit_date"], errors="coerce")
        limit_days = (limit_ts - cohort["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    else:
        limit_days = pd.Series(np.nan, index=cohort.index)

    valid_limit_mask = (
        cohort["order_purchase_timestamp"].notna()
        & cohort["order_delivered_carrier_date"].notna()
        & limit_days.notna()
    )
    # Unified purchase-time origin: both limit_days and purchase_to_carrier_days are measured from order_purchase_timestamp
    cohort["seller_exceeded_shipping_limit"] = np.where(
        valid_limit_mask,
        (cohort["purchase_to_carrier_days"] > limit_days).astype(float),
        np.nan,
    )

    rows: list[dict[str, Any]] = []
    for grp_val, grp_label in [(0, "On-Time (late=0)"), (1, "Delayed (late=1)")]:
        sub = cohort[cohort[label_col] == grp_val]
        sub_valid_mask = valid_limit_mask.loc[sub.index]
        valid_orders_cnt = int(sub_valid_mask.sum())
        exceeded_cnt = int(np.nansum(sub["seller_exceeded_shipping_limit"] == 1.0))
        exceeded_rate = round(float(exceeded_cnt / max(1, valid_orders_cnt)), 6) if valid_orders_cnt > 0 else 0.0

        rows.append(
            {
                "analysis_scope": "DIAGNOSTIC_EX_POST_ONLY_NEVER_IN_PREDICTION",
                "outcome_group": grp_label,
                "label_value": grp_val,
                "order_count": int(len(sub)),
                "share_of_cohort_a": round(float(len(sub) / max(1, len(cohort))), 6),
                "shipping_limit_valid_orders": valid_orders_cnt,
                "seller_exceeded_shipping_limit_count": exceeded_cnt,
                "approval_hours_median": round(float(sub["approval_hours"].median()), 3) if len(sub) > 0 else 0.0,
                "approval_hours_mean": round(float(sub["approval_hours"].mean()), 3) if len(sub) > 0 else 0.0,
                "seller_to_carrier_days_median": round(float(sub["seller_to_carrier_days"].median()), 3) if len(sub) > 0 else 0.0,
                "seller_to_carrier_days_mean": round(float(sub["seller_to_carrier_days"].mean()), 3) if len(sub) > 0 else 0.0,
                "purchase_to_carrier_days_median": round(float(sub["purchase_to_carrier_days"].median()), 3) if len(sub) > 0 else 0.0,
                "purchase_to_carrier_days_mean": round(float(sub["purchase_to_carrier_days"].mean()), 3) if len(sub) > 0 else 0.0,
                "carrier_to_customer_days_median": round(float(sub["carrier_to_customer_days"].median()), 3) if len(sub) > 0 else 0.0,
                "carrier_to_customer_days_mean": round(float(sub["carrier_to_customer_days"].mean()), 3) if len(sub) > 0 else 0.0,
                "total_fulfillment_days_median": round(float(sub["total_fulfillment_days"].median()), 3) if len(sub) > 0 else 0.0,
                "total_fulfillment_days_mean": round(float(sub["total_fulfillment_days"].mean()), 3) if len(sub) > 0 else 0.0,
                "promised_lead_calendar_days_mean": (
                    round(float(sub["estimated_lead_calendar_days"].mean()), 3)
                    if "estimated_lead_calendar_days" in sub.columns and len(sub) > 0
                    else 0.0
                ),
                "seller_exceeded_shipping_limit_rate": exceeded_rate,
                "post_delivery_review_score_mean": (
                    round(float(sub["review_score_mean"].mean()), 3)
                    if "review_score_mean" in sub.columns and len(sub) > 0
                    else 0.0
                ),
                "one_star_review_rate": (
                    round(float((sub["review_score_min"] == 1).mean()), 6)
                    if "review_score_min" in sub.columns and len(sub) > 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
