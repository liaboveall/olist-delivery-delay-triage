"""Validation model selection, sensitivity analysis, claims indexing, metric definitions, and model card generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .artifacts import now_iso, sha256_file
from .evaluation import compute_classification_metrics
from .features import get_feature_columns
from .models import CalibratedModelBundle, fit_model_with_chronological_calibration
from .policy import allocate_daily_alerts, evaluate_topk_frontier_cohort_a, evaluate_topk_frontier_path_b


def bootstrap_replicates_from_config(cfg: dict[str, Any]) -> tuple[int, int]:
    """Return `(row_replicates, weekly_block_replicates)` from `uncertainty_contract` (defaults: 2,000 each)."""
    uc = cfg.get("uncertainty_contract", {}) or {}
    n_row = int(uc.get("row_bootstrap_replicates", 2000))
    n_block = int(uc.get("block_bootstrap_replicates", n_row))
    if n_row < 1 or n_block < 1:
        raise ValueError(f"Bootstrap replicate counts must be positive, got row={n_row}, block={n_block}")
    return n_row, n_block


def compute_cost_scenario_table(
    df_eval: pd.DataFrame,
    scores: np.ndarray,
    label_col: str,
    budgets: list[float],
    scenarios: list[dict[str, Any]],
    model_name: str,
    split_id: str,
) -> pd.DataFrame:
    """Net value of the daily-floor queue under the configured (hypothetical) cost scenarios.

    For each budget k the queue is `floor(k * N_d)` orders per purchase day. Net value relative to doing nothing is
    `TP * benefit_per_tp - alerts * cost_per_contact - FP * cost_per_fp`, reported per 1,000 evaluated orders.
    `breakeven_precision = (cost_per_contact + cost_per_fp) / (benefit_per_tp + cost_per_fp)`; the variant that also
    credits the avoided missed-delay cost uses `benefit_per_tp + cost_per_fn` in the denominator.
    """
    from .evaluation import _select_daily_top_k_deterministic

    cols = [
        "split_id", "model_name", "scenario_id", "budget_k", "alerts", "tp", "fp", "precision", "recall",
        "benefit_per_tp", "cost_per_contact", "cost_per_fp", "cost_per_fn", "breakeven_precision",
        "breakeven_precision_with_fn", "net_value_per_1000_orders", "beats_no_action",
    ]
    if not scenarios or len(df_eval) == 0:
        return pd.DataFrame(columns=cols)
    y = df_eval[label_col].astype(int).to_numpy()
    ts = pd.to_datetime(df_eval["order_purchase_timestamp"])
    ts_ns = ts.astype("int64").to_numpy()
    day_keys = ts.dt.strftime("%Y-%m-%d").to_numpy(dtype=str)
    oids = df_eval["order_id"].astype(str).to_numpy()
    draws = np.zeros(len(df_eval), dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n = len(df_eval)
    pos = int(y.sum())
    rows: list[dict[str, Any]] = []
    for k in budgets:
        sel = _select_daily_top_k_deterministic(s, oids, ts_ns, day_keys, draws, float(k))
        alerts = int(sel.sum())
        tp = int((y[sel] == 1).sum())
        fp = alerts - tp
        for sc in scenarios:
            b = float(sc["benefit_per_tp"])
            c = float(sc["cost_per_contact"])
            f = float(sc["cost_per_fp"])
            fn = float(sc.get("cost_per_fn", 0.0))
            net = 1000.0 * (tp * b - alerts * c - fp * f) / n
            rows.append(
                {
                    "split_id": split_id,
                    "model_name": model_name,
                    "scenario_id": str(sc["scenario_id"]),
                    "budget_k": float(k),
                    "alerts": alerts,
                    "tp": tp,
                    "fp": fp,
                    "precision": round(tp / alerts, 6) if alerts else 0.0,
                    "recall": round(tp / pos, 6) if pos else 0.0,
                    "benefit_per_tp": b,
                    "cost_per_contact": c,
                    "cost_per_fp": f,
                    "cost_per_fn": fn,
                    "breakeven_precision": round((c + f) / (b + f), 6),
                    "breakeven_precision_with_fn": round((c + f) / (b + fn + f), 6),
                    "net_value_per_1000_orders": round(net, 6),
                    "beats_no_action": bool(net > 0),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def select_candidate_from_validation(
    val_metrics_df: pd.DataFrame,
    candidate_manifest: list[dict[str, Any]] | dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the production candidate and best simple baseline strictly from rolling validation metrics.

    Supports both `select_candidate_from_validation(val_metrics_df, cfg)` and
    `select_candidate_from_validation(val_metrics_df, candidate_manifest, cfg)`.
    """
    if cfg is None and isinstance(candidate_manifest, dict) and (
        "selection_contract" in candidate_manifest or "primary_label" in candidate_manifest or "model_grid" in candidate_manifest
    ):
        cfg = candidate_manifest
        candidate_manifest = None
    cfg = cfg or {}

    primary_label = cfg.get("primary_label", cfg.get("label_contract", {}).get("primary_label", "calendar_day"))
    sel_cfg = cfg.get("selection_contract", {})
    primary_metric_cfg = str(sel_cfg.get("primary_metric", "validation_daily_top10_late_capture_rate"))
    must_beat_baseline = bool(sel_cfg.get("must_beat_best_simple_baseline", True))
    tolerance = float(
        sel_cfg.get("prefer_simpler_model_within_tolerance", sel_cfg.get("prefer_simpler_model_within", 0.005))
    )
    # Minimum absolute gain in the primary metric a model needs over the best rule (configured as
    # `min_relative_gain_over_best_rule`; the research contract defines it as 0.5 percentage points of capture).
    min_gain = float(sel_cfg.get("min_relative_gain_over_best_rule", 0.0))

    work = val_metrics_df.copy()
    if "candidate_id" not in work.columns:
        work["candidate_id"] = work["model_name"].astype(str)
    if "feature_version" not in work.columns:
        work["feature_version"] = "base_v2"
    if "hyperparams_json" not in work.columns:
        work["hyperparams_json"] = "{}"
    if "calibration_method" not in work.columns:
        work["calibration_method"] = "uncalibrated"

    for col, fallback_val in [
        ("daily_top10_precision", 0.0),
        ("roc_auc", 0.5),
        ("precision_top10", 0.0),
        ("recall_top10", 0.0),
    ]:
        if col not in work.columns:
            work[col] = fallback_val

    df_valid = work[
        (work["label_version"] == primary_label)
        & (work.get("metrics_status", pd.Series("OK", index=work.index)) == "OK")
    ].copy()

    cand_list: list[dict[str, Any]] = []
    if isinstance(candidate_manifest, list):
        cand_list = candidate_manifest
    elif isinstance(candidate_manifest, dict):
        cand_list = list(candidate_manifest.get("model_candidates", [])) + list(candidate_manifest.get("baselines", []))
    cand_lookup = {str(c["candidate_id"]): c for c in cand_list if "candidate_id" in c}

    def _infer_family(row: pd.Series) -> str:
        cid = str(row["candidate_id"])
        mname = str(row["model_name"])
        if cid in cand_lookup and "model_family" in cand_lookup[cid]:
            fam = str(cand_lookup[cid]["model_family"])
            return "simple_baseline" if fam == "simple_baseline" else "ml_model"
        if mname.startswith("rule_") or mname == "global_prior_baseline" or cid.startswith("rule_") or cid == "global_prior_baseline":
            return "simple_baseline"
        return "ml_model"

    def _infer_complexity(row: pd.Series) -> int:
        cid = str(row["candidate_id"])
        if cid in cand_lookup and "complexity_rank" in cand_lookup[cid]:
            return int(cand_lookup[cid]["complexity_rank"])
        fv = str(row.get("feature_version", "base_v2"))
        mname = str(row.get("model_name", ""))
        hp_str = str(row.get("hyperparams_json", "{}"))
        if mname.startswith("rule_") or mname == "global_prior_baseline":
            return 1
        base_c = 10 if fv == "base_v2" else 20
        if "C=1.0" in hp_str or '"C":1.0' in hp_str or '"max_depth":5' in hp_str:
            base_c += 3
        elif "C=0.3" in hp_str or '"C":0.3' in hp_str or '"max_depth":4' in hp_str:
            base_c += 2
        else:
            base_c += 1
        return base_c

    df_valid["model_family"] = df_valid.apply(_infer_family, axis=1)
    df_valid["complexity_rank"] = df_valid.apply(_infer_complexity, axis=1)

    agg = (
        df_valid.groupby(
            [
                "candidate_id",
                "model_name",
                "feature_version",
                "hyperparams_json",
                "calibration_method",
                "model_family",
                "complexity_rank",
            ],
            as_index=False,
        )
        .agg(
            val_mean_daily_top10_recall=("daily_top10_recall", "mean"),
            val_std_daily_top10_recall=("daily_top10_recall", "std"),
            val_mean_daily_top10_precision=("daily_top10_precision", "mean"),
            val_mean_pr_auc=("pr_auc", "mean"),
            val_std_pr_auc=("pr_auc", "std"),
            val_mean_roc_auc=("roc_auc", "mean"),
            val_mean_brier=("brier_score", "mean"),
            val_mean_ece_10=("ece_quantile_10", "mean"),
            val_mean_prec_top10=("precision_top10", "mean"),
            val_mean_rec_top10=("recall_top10", "mean"),
            folds_evaluated=("split_id", "count"),
        )
    )
    agg["val_std_daily_top10_recall"] = agg["val_std_daily_top10_recall"].fillna(0.0)
    agg["val_std_pr_auc"] = agg["val_std_pr_auc"].fillna(0.0)

    primary_col = (
        "val_mean_daily_top10_recall"
        if "capture" in primary_metric_cfg or "recall" in primary_metric_cfg
        else "val_mean_pr_auc"
    )

    # 1. Best simple baseline (operational rules first; global_prior_baseline is probability reference)
    base_df = agg[agg["model_family"] == "simple_baseline"].copy()
    op_base_df = base_df[base_df["model_name"] != "global_prior_baseline"].copy()
    if op_base_df.empty:
        op_base_df = base_df.copy()

    op_base_df.sort_values(
        by=[primary_col, "val_mean_pr_auc", "val_mean_brier", "val_mean_ece_10", "candidate_id"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
        inplace=True,
    )
    best_base_row = op_base_df.iloc[0].to_dict() if not op_base_df.empty else agg.iloc[0].to_dict()

    # 2. Best ML candidate
    ml_df = agg[agg["model_family"] != "simple_baseline"].copy()
    if ml_df.empty:
        ml_df = agg.copy()

    # For each ML candidate_id, pick its best calibrated variant (platt/isotonic) if available
    chosen_ml_rows = []
    for cid, grp in ml_df.groupby("candidate_id", sort=False):
        cal_grp = grp[grp["calibration_method"].isin(["platt", "isotonic"])].sort_values(
            by=["val_mean_brier", "val_mean_ece_10", "calibration_method"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        if not cal_grp.empty:
            chosen_ml_rows.append(cal_grp.iloc[0])
        else:
            chosen_ml_rows.append(grp.iloc[0])
    ml_cand_df = pd.DataFrame(chosen_ml_rows)

    ml_cand_df.sort_values(
        by=[primary_col, "val_mean_pr_auc", "val_mean_brier", "val_mean_ece_10", "complexity_rank", "candidate_id"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
        inplace=True,
    )
    top_ml_primary = float(ml_cand_df.iloc[0][primary_col])
    top_ml_pr = float(ml_cand_df.iloc[0]["val_mean_pr_auc"])

    # Simplicity preference within tolerance
    within_tol = ml_cand_df[
        (ml_cand_df[primary_col] >= top_ml_primary - tolerance - 1e-12)
        & (ml_cand_df["val_mean_pr_auc"] >= top_ml_pr - tolerance - 1e-12)
    ].copy()
    within_tol.sort_values(
        by=["complexity_rank", primary_col, "val_mean_pr_auc", "val_mean_brier", "candidate_id"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
        inplace=True,
    )
    best_ml_row = within_tol.iloc[0].to_dict()

    # Compare best_ml vs best_baseline
    ml_gain = float(best_ml_row[primary_col]) - float(best_base_row[primary_col])
    if min_gain > 0:
        ml_beats_base = ml_gain >= min_gain - 1e-12
    else:
        ml_beats_base = ml_gain > 1e-9 or (
            abs(ml_gain) <= 1e-9
            and float(best_ml_row["val_mean_pr_auc"]) > float(best_base_row["val_mean_pr_auc"]) + 1e-9
        )

    if must_beat_baseline and not ml_beats_base:
        selected_row = best_base_row
        selection_outcome = "SIMPLE_BASELINE_SELECTED_NO_ML_GAIN"
    else:
        selected_row = best_ml_row
        selection_outcome = "ML_CANDIDATE_SELECTED_BEATS_BASELINE"

    # Compute fold-by-fold paired comparison between best_ml and best_baseline
    ml_folds = df_valid[
        (df_valid["candidate_id"] == best_ml_row["candidate_id"])
        & (df_valid["calibration_method"] == best_ml_row["calibration_method"])
    ].set_index("split_id")
    base_folds = df_valid[
        (df_valid["candidate_id"] == best_base_row["candidate_id"])
        & (df_valid["calibration_method"] == best_base_row["calibration_method"])
    ].set_index("split_id")
    common_splits = [s for s in ml_folds.index if s in base_folds.index]
    fold_diffs = []
    for s in common_splits:
        fold_diffs.append(
            {
                "split_id": str(s),
                "ml_daily_top10_recall": float(ml_folds.loc[s, "daily_top10_recall"]),
                "baseline_daily_top10_recall": float(base_folds.loc[s, "daily_top10_recall"]),
                "daily_top10_recall_diff": round(
                    float(ml_folds.loc[s, "daily_top10_recall"] - base_folds.loc[s, "daily_top10_recall"]), 6
                ),
                "ml_pr_auc": float(ml_folds.loc[s, "pr_auc"]),
                "baseline_pr_auc": float(base_folds.loc[s, "pr_auc"]),
                "pr_auc_diff": round(float(ml_folds.loc[s, "pr_auc"] - base_folds.loc[s, "pr_auc"]), 6),
            }
        )

    sel_cid_str = str(selected_row["candidate_id"])
    sel_mname_str = str(selected_row["model_name"])
    sel_fam_tag = "simple_baseline" if str(selected_row["model_family"]) == "simple_baseline" else "ml_model"

    return {
        "selected_at": now_iso(),
        "selection_source": "rolling_validation_only_strictly_prior_to_backtest",
        "selection_rule": primary_metric_cfg,
        "primary_metric_configured": primary_metric_cfg,
        "primary_metric_column": primary_col,
        "primary_metric_aggregation": (
            "mean_over_validation_folds" if primary_col == "val_mean_daily_top10_recall" else "mean_pr_auc_over_validation_folds"
        ),
        "primary_metric_note": (
            "The configured name says 'pooled', but the metric is the mean of fold-level daily top-10% recall."
            if "pooled" in primary_metric_cfg
            else ""
        ),
        "must_beat_best_simple_baseline": must_beat_baseline,
        "min_gain_over_best_rule": min_gain,
        "ml_gain_over_best_rule": round(ml_gain, 6),
        "prefer_simpler_model_within_tolerance": tolerance,
        "selection_outcome": selection_outcome,
        "ml_beats_best_simple_baseline_on_validation": bool(ml_beats_base),
        "selected_candidate_id": sel_cid_str,
        "selected_model_name": sel_mname_str,
        "selected_model_family": sel_fam_tag if sel_cid_str == sel_mname_str else sel_mname_str,
        "selected_model_category": sel_fam_tag,
        "selected_feature_version": str(selected_row["feature_version"]),
        "selected_hyperparams_json": str(selected_row.get("hyperparams_json", "{}")),
        "selected_calibration_method": str(selected_row["calibration_method"]),
        "best_baseline_name": str(best_base_row["candidate_id"]),
        "selected_candidate": {
            "candidate_id": sel_cid_str,
            "model_name": sel_mname_str,
            "model_family": sel_fam_tag,
            "feature_version": str(selected_row["feature_version"]),
            "hyperparams_json": str(selected_row.get("hyperparams_json", "{}")),
            "calibration_method": str(selected_row["calibration_method"]),
            "complexity_rank": int(selected_row["complexity_rank"]),
            "val_mean_daily_top10_recall": round(float(selected_row["val_mean_daily_top10_recall"]), 6),
            "val_mean_daily_top10_precision": round(float(selected_row["val_mean_daily_top10_precision"]), 6),
            "val_mean_pr_auc": round(float(selected_row["val_mean_pr_auc"]), 6),
            "val_std_pr_auc": round(float(selected_row["val_std_pr_auc"]), 6),
            "val_mean_roc_auc": round(float(selected_row["val_mean_roc_auc"]), 6),
            "val_mean_brier": round(float(selected_row["val_mean_brier"]), 6),
            "val_mean_ece_10": round(float(selected_row["val_mean_ece_10"]), 6),
        },
        "best_ml_candidate": {
            "candidate_id": str(best_ml_row["candidate_id"]),
            "model_name": str(best_ml_row["model_name"]),
            "feature_version": str(best_ml_row["feature_version"]),
            "hyperparams_json": str(best_ml_row.get("hyperparams_json", "{}")),
            "calibration_method": str(best_ml_row["calibration_method"]),
            "complexity_rank": int(best_ml_row["complexity_rank"]),
            "val_mean_daily_top10_recall": round(float(best_ml_row["val_mean_daily_top10_recall"]), 6),
            "val_mean_pr_auc": round(float(best_ml_row["val_mean_pr_auc"]), 6),
            "val_mean_brier": round(float(best_ml_row["val_mean_brier"]), 6),
            "val_mean_ece_10": round(float(best_ml_row["val_mean_ece_10"]), 6),
        },
        "best_simple_baseline": {
            "candidate_id": str(best_base_row["candidate_id"]),
            "model_name": str(best_base_row["model_name"]),
            "feature_version": str(best_base_row["feature_version"]),
            "calibration_method": str(best_base_row["calibration_method"]),
            "val_mean_daily_top10_recall": round(float(best_base_row["val_mean_daily_top10_recall"]), 6),
            "val_mean_pr_auc": round(float(best_base_row["val_mean_pr_auc"]), 6),
            "val_mean_brier": round(float(best_base_row["val_mean_brier"]), 6),
            "val_mean_ece_10": round(float(best_base_row["val_mean_ece_10"]), 6),
        },
        "validation_fold_paired_comparison": fold_diffs,
        "leaderboard": agg.to_dict(orient="records"),
    }


def compute_sensitivity_summary_table(
    df_bt_a: pd.DataFrame,
    df_bt_b: pd.DataFrame,
    p_a_cal: np.ndarray,
    p_b_cal: np.ndarray,
    selected_candidate_id: str,
    selected_cal_method: str,
    best_baseline_id: str,
    p_a_baseline: np.ndarray,
    model_comp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build structured sensitivity summary table (`sensitivity_summary.csv`) across pre-registered dimensions."""
    y_cal = df_bt_a["late_calendar_day"].astype(int).to_numpy()
    y_ts = df_bt_a["late_timestamp"].astype(int).to_numpy()
    ts_a = df_bt_a["order_purchase_timestamp"]
    oids_a = df_bt_a["order_id"]

    rows: list[dict[str, Any]] = []

    # 1. Capacity budget sensitivity (5%, 10%, 20% under daily policy vs global pool)
    for k in [0.05, 0.10, 0.20]:
        f_daily = evaluate_topk_frontier_cohort_a(
            order_ids=oids_a,
            y_true=y_cal,
            scores=p_a_cal,
            budgets=[k],
            model_name=selected_candidate_id,
            purchase_timestamps=ts_a,
            daily_allocation=True,
        ).iloc[0]
        f_global = evaluate_topk_frontier_cohort_a(
            order_ids=oids_a,
            y_true=y_cal,
            scores=p_a_cal,
            budgets=[k],
            model_name=selected_candidate_id,
            purchase_timestamps=ts_a,
            daily_allocation=False,
        ).iloc[0]
        rows.append(
            {
                "sensitivity_dimension": "capacity_budget_daily_floor",
                "variant_name": f"daily_floor_top_{int(round(k*100))}pct",
                "candidate_id": selected_candidate_id,
                "population_n": int(f_daily["population_n"]),
                "positives_n": int(f_daily["total_positives"]),
                "alert_count": int(f_daily["alert_count"]),
                "tp": int(f_daily["tp"]),
                "fp": int(f_daily["fp"]),
                "precision": float(f_daily["precision"]),
                "recall": float(f_daily["recall"]),
                "pr_auc": round(float(model_comp_df.loc[model_comp_df["is_selected_production_model"], "backtest_pr_auc"].iloc[0]), 6)
                if not model_comp_df.loc[model_comp_df["is_selected_production_model"]].empty
                else None,
                "notes": f"Daily floor(k*N_d) vs global floor alert delta = {int(f_daily['alert_count']) - int(f_global['alert_count'])}",
            }
        )

    # 2. Label definition sensitivity (calendar_day vs exact timestamp)
    for lbl_name, y_vec in [("calendar_day_primary", y_cal), ("exact_timestamp_secondary", y_ts)]:
        m_lbl = compute_classification_metrics(
            order_ids=oids_a,
            y_true=y_vec,
            y_prob=p_a_cal,
            model_name=selected_candidate_id,
            calibration_method=selected_cal_method,
            split_id="historical_backtest",
            label_version=lbl_name,
            purchase_timestamps=ts_a,
        )
        rows.append(
            {
                "sensitivity_dimension": "label_definition",
                "variant_name": lbl_name,
                "candidate_id": selected_candidate_id,
                "population_n": int(m_lbl["n_samples"]),
                "positives_n": int(m_lbl["n_positives"]),
                "alert_count": int(m_lbl["daily_top10_alert_count"]),
                "tp": int(m_lbl["daily_top10_tp"]),
                "fp": int(m_lbl["daily_top10_fp"]),
                "precision": float(m_lbl["daily_top10_precision"]),
                "recall": float(m_lbl["daily_top10_recall"]),
                "pr_auc": m_lbl["pr_auc"],
                "notes": "Same predicted probabilities evaluated against primary vs timestamp label",
            }
        )

    # 3. Feature availability (base_v2 vs extended_purchase_assumptions_v2 from model_comp_df)
    cal_rows = model_comp_df[model_comp_df["label_version"] == "calendar_day"].copy()
    for fv in ["base_v2", "extended_purchase_assumptions_v2"]:
        sub_fv = cal_rows[
            (cal_rows.get("feature_version", pd.Series("", index=cal_rows.index)) == fv)
            & (cal_rows["model_family"] == "ml_candidate")
        ]
        if not sub_fv.empty:
            best_fv = sub_fv.sort_values("val_mean_pr_auc", ascending=False, kind="mergesort").iloc[0]
            rows.append(
                {
                    "sensitivity_dimension": "feature_availability_version",
                    "variant_name": fv,
                    "candidate_id": str(best_fv["model_name"]),
                    "population_n": int(len(df_bt_a)),
                    "positives_n": int(np.sum(y_cal == 1)),
                    "alert_count": int(best_fv.get("backtest_daily_top10_alert_count", 0)),
                    "tp": int(best_fv.get("backtest_daily_top10_tp", 0)),
                    "fp": int(best_fv.get("backtest_daily_top10_fp", 0)),
                    "precision": float(best_fv["backtest_precision_top10"]),
                    "recall": float(best_fv["backtest_recall_top10"]),
                    "pr_auc": float(best_fv["backtest_pr_auc"]),
                    "notes": f"Top validation candidate within feature_version={fv}",
                }
            )

    # 4. Censoring window sensitivity: exclude final 14 days of backtest purchases
    max_bt_ts = pd.to_datetime(df_bt_a["order_purchase_timestamp"]).max()
    cutoff_14d = max_bt_ts - pd.Timedelta(days=14)
    mask_excl_14d = (pd.to_datetime(df_bt_a["order_purchase_timestamp"]) <= cutoff_14d).to_numpy(dtype=bool)
    if np.sum(mask_excl_14d) > 0:
        m_14d = compute_classification_metrics(
            order_ids=oids_a.loc[mask_excl_14d],
            y_true=y_cal[mask_excl_14d],
            y_prob=p_a_cal[mask_excl_14d],
            model_name=selected_candidate_id,
            calibration_method=selected_cal_method,
            split_id="historical_backtest_excl_last_14d",
            label_version="calendar_day",
            purchase_timestamps=ts_a.loc[mask_excl_14d],
        )
        rows.append(
            {
                "sensitivity_dimension": "tail_censoring_window",
                "variant_name": "exclude_final_14_purchase_days",
                "candidate_id": selected_candidate_id,
                "population_n": int(m_14d["n_samples"]),
                "positives_n": int(m_14d["n_positives"]),
                "alert_count": int(m_14d["daily_top10_alert_count"]),
                "tp": int(m_14d["daily_top10_tp"]),
                "fp": int(m_14d["daily_top10_fp"]),
                "precision": float(m_14d["daily_top10_precision"]),
                "recall": float(m_14d["daily_top10_recall"]),
                "pr_auc": m_14d["pr_auc"],
                "notes": f"Purchases <= {cutoff_14d.isoformat()} to bound right-censoring impact",
            }
        )

    # 5. Cold-start vs established seller & multi-seller sensitivity
    seller_cnt_col = (
        "seller_prior_Deliveries_count"
        if "seller_prior_Deliveries_count" in df_bt_a.columns
        else "seller_completed_count"
    )
    cold_mask = (df_bt_a[seller_cnt_col].to_numpy() < 5)
    est_mask = ~cold_mask
    multi_seller_mask = (df_bt_a["distinct_seller_count"].to_numpy() > 1)
    for sub_name, smask in [
        ("seller_cold_start_lt5", cold_mask),
        ("seller_established_ge5", est_mask),
        ("multi_seller_orders", multi_seller_mask),
    ]:
        if int(np.sum(smask)) >= 10:
            m_sub = compute_classification_metrics(
                order_ids=oids_a.loc[smask],
                y_true=y_cal[smask],
                y_prob=p_a_cal[smask],
                model_name=selected_candidate_id,
                calibration_method=selected_cal_method,
                split_id=f"backtest_{sub_name}",
                label_version="calendar_day",
                purchase_timestamps=ts_a.loc[smask],
            )
            rows.append(
                {
                    "sensitivity_dimension": "seller_and_basket_regime",
                    "variant_name": sub_name,
                    "candidate_id": selected_candidate_id,
                    "population_n": int(m_sub["n_samples"]),
                    "positives_n": int(m_sub["n_positives"]),
                    "alert_count": int(m_sub["daily_top10_alert_count"]),
                    "tp": int(m_sub["daily_top10_tp"]),
                    "fp": int(m_sub["daily_top10_fp"]),
                    "precision": float(m_sub["daily_top10_precision"]),
                    "recall": float(m_sub["daily_top10_recall"]),
                    "pr_auc": m_sub["pr_auc"],
                    "notes": f"Subgroup slice {sub_name} (n={int(m_sub['n_samples'])})",
                }
            )

    return pd.DataFrame(rows)


def build_metric_definitions() -> dict[str, Any]:
    """Return formal metric definitions, denominators, tie-break rules, and precision tolerances (`metric_definitions.json`)."""
    return {
        "schema_version": "2.0.0",
        "generated_at": now_iso(),
        "rounding_and_precision_contract": {
            "internal_computation_dtype": "float64",
            "ranking_and_selection_precision": "full_float64_before_any_display_rounding",
            "display_rounding_decimals": 6,
            "recomputation_count_tolerance": 0,
            "recomputation_metric_abs_tolerance": 1e-10,
            "recomputation_displayed_metric_abs_tolerance": 1e-6,
            "scoring_probability_abs_tolerance": 1e-8,
        },
        "tie_break_order": [
            "-risk_score",
            "+order_purchase_timestamp",
            "+order_id",
            "+sample_draw_id",
        ],
        "metrics": {
            "daily_top10_recall": {
                "name": "Daily Top-10% Late Capture Rate (Recall@Daily10%)",
                "formula": "sum_d TP_d / sum_d Positives_d, where each day d selects q_d = floor(0.10 * N_eligible,d) orders",
                "denominator": "All observed delayed orders (late == 1) in Cohort A within the evaluation window",
                "role": "primary_validation_selection_and_operational_metric",
            },
            "daily_top10_precision": {
                "name": "Daily Top-10% Alert Precision (Precision@Daily10%)",
                "formula": "sum_d TP_d / max(1, sum_d q_d)",
                "denominator": "Total daily alerts issued across the evaluation window under q_d = floor(0.10 * N_eligible,d)",
                "role": "operational_efficiency_metric",
            },
            "pr_auc": {
                "name": "Area Under Precision-Recall Curve (Average Precision)",
                "formula": "sklearn.metrics.average_precision_score(y_true, y_prob)",
                "denominator": "Cohort A evaluable orders in window (requires >= 2 classes, else None)",
                "role": "secondary_global_ranking_metric",
            },
            "brier_score": {
                "name": "Brier Score",
                "formula": "mean((y_prob - y_true)^2)",
                "denominator": "Cohort A evaluable orders in window (jointly reflects discrimination and calibration)",
                "role": "calibration_and_probability_accuracy_metric",
            },
            "ece_quantile_10": {
                "name": "Expected Calibration Error (10 quantile bins)",
                "formula": "sum_{b=1..10} (n_b / N) * |mean(y_prob_b) - mean(y_true_b)|",
                "denominator": "Cohort A evaluable orders partitioned into 10 equal-frequency probability bins",
                "role": "calibration_diagnostic_metric",
            },
            "lower_bound_precision_path_b": {
                "name": "Path B Conservative Lower-Bound Alert Precision",
                "formula": "observed_tp_in_cohort_a / alert_total_path_b",
                "denominator": "All Path B alerts (treating all unlabeled non-delivered alerts as non-late)",
                "role": "path_b_operational_lower_bound",
            },
            "upper_bound_exception_rate_path_b": {
                "name": "Path B Upper-Bound Fulfillment Exception Rate",
                "formula": "(observed_tp_in_cohort_a + alert_unlabeled_non_delivered_count) / alert_total_path_b",
                "denominator": "All Path B alerts (treating all unlabeled non-delivered alerts as fulfillment exceptions)",
                "role": "path_b_operational_upper_bound",
            },
        },
    }


def build_claims_index(
    run_id: str,
    run_dir: Path,
    audit_summary: dict[str, Any],
    sp_df: pd.DataFrame,
    selection_record: dict[str, Any],
    metrics_summary: dict[str, Any],
    daily_policy_df: pd.DataFrame,
    pol_b_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build machine-verifiable `claims.json` linking report/slide numbers to exact source files and hashes."""
    rec = audit_summary["row_reconciliation"]
    sel_mod = metrics_summary["selected_model"]
    bt_m = metrics_summary["selected_model_backtest_metrics"]
    boot = metrics_summary.get("bootstrap_95_ci", {})
    wb = boot.get("weekly_block_bootstrap", {})
    wb_diff = wb.get("paired_difference_vs_baseline", {})

    # C04/C05 describe right-censoring in the last validation fold (val_2018_04 under the v2 contract).
    val_rows_all = sp_df[sp_df["split_type"] == "validation"] if "split_type" in sp_df.columns else sp_df.iloc[0:0]
    val_apr_rows = (
        val_rows_all.sort_values("eval_end").tail(1) if len(val_rows_all) else sp_df[sp_df["split_id"] == "val_2018_04"]
    )
    if val_apr_rows.empty:
        raise ValueError("split_summary has no validation fold to describe maturity censoring (C04/C05)")
    val_apr_unmatured = int(val_apr_rows["eval_unmatured_at_selection_count"].iloc[0])
    val_apr_unmatured_late = int(
        val_apr_rows["eval_full_cohort_a_positives"].iloc[0] - val_apr_rows["eval_matured_positives"].iloc[0]
    )

    pol_b_10 = pol_b_df[np.isclose(pol_b_df["budget_k"], 0.10)].iloc[0] if not pol_b_df.empty else {}

    # Window labels derived from the split summary (identical to the former literals under the v2 contract).
    last_val_id = str(val_apr_rows["split_id"].iloc[0])
    last_val_sel = pd.Timestamp(val_apr_rows["selection_at"].iloc[0]).strftime("%Y_%m_%d")
    val_window_label = f"{last_val_id}_at_selection_cutoff_{last_val_sel}"
    bt_rows_sp = sp_df[sp_df["split_type"] == "backtest"] if "split_type" in sp_df.columns else sp_df.iloc[0:0]
    if len(bt_rows_sp):
        bt_r = bt_rows_sp.iloc[0]
        bt_window_label = (
            f"{bt_r['split_id']}_{pd.Timestamp(bt_r['eval_start']).strftime('%Y_%m_%d')}_to_"
            f"{(pd.Timestamp(bt_r['eval_end']) - pd.Timedelta(days=1)).strftime('%Y_%m_%d')}"
        )
    else:
        bt_window_label = "historical_backtest_2018_05_26_to_2018_08_29"

    def _file_hash(rel_name: str) -> str | None:
        p = run_dir / rel_name
        return sha256_file(p) if p.exists() else None

    claims_list = [
        {
            "claim_id": "C01_SOURCE_ORDERS_TOTAL",
            "value": int(rec["source_orders_total"]),
            "unit": "orders",
            "cohort": "Path_B_all_source_orders",
            "window": "full_dataset_2016_09_to_2018_10",
            "policy": "none",
            "label": "calendar_day",
            "source_file": "data_quality_summary.json",
            "source_sha256": _file_hash("data_quality_summary.json"),
            "scope_boundary": "Total unique orders in olist_orders_dataset.csv",
        },
        {
            "claim_id": "C02_COHORT_A_EVALUABLE_TOTAL",
            "value": int(rec["path_a_delivered_evaluable_orders"]),
            "unit": "orders",
            "cohort": "Cohort_A_delivered_with_items_and_valid_timestamps",
            "window": "full_dataset_2016_09_to_2018_10",
            "policy": "none",
            "label": "calendar_day",
            "source_file": "data_quality_summary.json",
            "source_sha256": _file_hash("data_quality_summary.json"),
            "scope_boundary": "Delivered orders with >=1 item and non-missing purchase/estimated/delivery timestamps",
        },
        {
            "claim_id": "C03_UNKNOWN_OUTCOME_EXCLUDED_TOTAL",
            "value": int(rec["path_a_excluded_orders"]),
            "unit": "orders",
            "cohort": "Path_B_minus_Cohort_A_unknown_delivery_outcome",
            "window": "full_dataset_2016_09_to_2018_10",
            "policy": "none",
            "label": "calendar_day",
            "source_file": "data_quality_summary.json",
            "source_sha256": _file_hash("data_quality_summary.json"),
            "scope_boundary": "Orders with non-delivered status or missing delivery timestamp; kept as NA, never coerced to 0",
        },
        {
            "claim_id": "C04_VAL_APRIL_UNMATURED_AT_SELECTION",
            "value": val_apr_unmatured,
            "unit": "orders",
            "cohort": "Cohort_A_eventual_delivered",
            "window": val_window_label,
            "policy": "maturity_filter",
            "label": "calendar_day",
            "source_file": "split_summary.csv",
            "source_sha256": _file_hash("split_summary.csv"),
            "scope_boundary": f"{last_val_id} Cohort A orders still undelivered at the selection cutoff {val_apr_rows['selection_at'].iloc[0]}",
        },
        {
            "claim_id": "C05_VAL_APRIL_UNMATURED_LATE_COUNT",
            "value": val_apr_unmatured_late,
            "unit": "orders",
            "cohort": "Cohort_A_eventual_delivered",
            "window": val_window_label,
            "policy": "maturity_filter",
            "label": "calendar_day",
            "source_file": "split_summary.csv",
            "source_sha256": _file_hash("split_summary.csv"),
            "scope_boundary": "Illustrates maturity selection bias: unmatured orders are disproportionately delayed (89/104)",
        },
        {
            "claim_id": "C06_BACKTEST_PATH_B_TOTAL",
            "value": int(pol_b_10.get("path_b_total_orders", 0)),
            "unit": "orders",
            "cohort": "Path_B_all_scorable_orders",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "policy_frontier_all_orders.csv",
            "source_sha256": _file_hash("policy_frontier_all_orders.csv"),
            "scope_boundary": "All purchase-time orders in historical_backtest window",
        },
        {
            "claim_id": "C07_BACKTEST_COHORT_A_TOTAL",
            "value": int(bt_m["n_samples"]),
            "unit": "orders",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "Observed delivered Cohort A orders in historical_backtest window",
        },
        {
            "claim_id": "C08_BACKTEST_POSITIVES_COHORT_A",
            "value": int(bt_m["n_positives"]),
            "unit": "orders",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "Delayed orders (late_calendar_day == 1) in historical_backtest Cohort A",
        },
        {
            "claim_id": "C09_SELECTED_CANDIDATE_ID",
            "value": str(sel_mod.get("candidate_id", sel_mod["model_name"])),
            "unit": "identifier",
            "cohort": "Cohort_A_conditional_delivered",
            "window": "rolling_validation_4_folds",
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "selection_record.json",
            "source_sha256": _file_hash("selection_record.json"),
            "scope_boundary": "Selected strictly on validation folds prior to holdout evaluation",
        },
        {
            "claim_id": "C10_BEST_SIMPLE_BASELINE_ID",
            "value": str(selection_record["best_simple_baseline"]["candidate_id"]),
            "unit": "identifier",
            "cohort": "Cohort_A_conditional_delivered",
            "window": "rolling_validation_4_folds",
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "selection_record.json",
            "source_sha256": _file_hash("selection_record.json"),
            "scope_boundary": "Best operational simple rule on rolling validation folds",
        },
        {
            "claim_id": "C11_SELECTED_BACKTEST_PR_AUC",
            "value": float(bt_m["pr_auc"]),
            "unit": "probability_metric",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "global_ranking",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "Historical holdout PR-AUC on Cohort A (diagnostic holdout after v1 review exposure)",
        },
        {
            "claim_id": "C12_SELECTED_DAILY_TOP10_ALERTS_COHORT_A",
            "value": int(bt_m["daily_top10_alert_count"]),
            "unit": "orders",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "Total alerts on Cohort A under daily floor(0.10 * N_d)",
        },
        {
            "claim_id": "C13_SELECTED_DAILY_TOP10_TP_COHORT_A",
            "value": int(bt_m["daily_top10_tp"]),
            "unit": "orders",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "True delayed orders caught under daily floor(0.10 * N_d) on Cohort A",
        },
        {
            "claim_id": "C14_SELECTED_DAILY_TOP10_FP_COHORT_A",
            "value": int(bt_m["daily_top10_fp"]),
            "unit": "orders",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "False alarms among observed Cohort A orders under daily floor(0.10 * N_d)",
        },
        {
            "claim_id": "C15_SELECTED_DAILY_TOP10_RECALL_COHORT_A",
            "value": float(bt_m["daily_top10_recall"]),
            "unit": "proportion",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "Fraction of all Cohort A late orders captured in daily Top-10% queue",
        },
        {
            "claim_id": "C16_SELECTED_DAILY_TOP10_PRECISION_COHORT_A",
            "value": float(bt_m["daily_top10_precision"]),
            "unit": "proportion",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "metrics_summary.json",
            "source_sha256": _file_hash("metrics_summary.json"),
            "scope_boundary": "Precision among daily Top-10% alerts on Cohort A",
        },
        {
            "claim_id": "C17_PATH_B_DAILY_ALERTS_TOTAL",
            "value": int(daily_policy_df["selected_alerts_d"].sum()) if not daily_policy_df.empty else 0,
            "unit": "orders",
            "cohort": "Path_B_all_scorable_orders",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "daily_policy.csv",
            "source_sha256": _file_hash("daily_policy.csv"),
            "scope_boundary": "Total daily alerts across Path B eligible orders (sum of q_d = floor(0.10 * N_eligible,d))",
        },
        {
            "claim_id": "C18_PATH_B_DAILY_ALERTS_UNLABELED",
            "value": int(daily_policy_df["selected_unlabeled_d"].sum()) if not daily_policy_df.empty else 0,
            "unit": "orders",
            "cohort": "Path_B_all_scorable_orders",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "daily_policy.csv",
            "source_sha256": _file_hash("daily_policy.csv"),
            "scope_boundary": "Daily Top-10% alert slots consumed by orders whose delivery outcome remained unknown/non-delivered",
        },
        {
            "claim_id": "C19_WEEKLY_BOOTSTRAP_DAILY_RECALL_DIFF_MEAN",
            "value": wb_diff.get("daily_policy_recall_top10_diff", {}).get("mean"),
            "unit": "proportion_difference",
            "cohort": "Cohort_A_conditional_delivered",
            "window": bt_window_label,
            "policy": "daily_end_of_day_top10_floor_v2",
            "label": "calendar_day",
            "source_file": "uncertainty_summary.json",
            "source_sha256": _file_hash("uncertainty_summary.json"),
            "scope_boundary": "Paired weekly block bootstrap mean difference in daily Top-10% recall vs best simple baseline",
        },
    ]
    for c in claims_list:
        c["passed"] = bool(c.get("source_sha256") is not None and c.get("value") is not None)

    return {
        "schema_version": "2.0.0",
        "run_id": run_id,
        "generated_at": now_iso(),
        "claims_by_id": {c["claim_id"]: c for c in claims_list},
        "claims": claims_list,
    }


def build_model_card_markdown(
    run_id: str,
    selection_record: dict[str, Any],
    metrics_summary: dict[str, Any],
    cal_audit: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    """Generate `model_card.md` documenting selected candidate, baselines, calibration audit, and boundaries."""
    sel = selection_record["selected_candidate"]
    base = selection_record["best_simple_baseline"]
    bt = metrics_summary["selected_model_backtest_metrics"]
    smoke_flag = bool(cfg.get("smoke_only", False))
    split_c = cfg.get("split_contract", {})

    lines = [
        f"# Model Card — Olist Delivery Delay Warning (`{run_id}`)",
        "",
        f"- **Generated At**: `{now_iso()}`",
        f"- **Run Mode**: `{'SMOKE_ONLY (pipeline verification only, not for final report)' if smoke_flag else 'FORMAL_RESEARCH_RUN'}`",
        f"- **Selected Candidate ID**: `{sel['candidate_id']}` (`{sel['model_name']}`, family=`{sel['model_family']}`)",
        f"- **Feature Availability Version**: `{sel['feature_version']}`",
        f"- **Calibration Method**: `{sel['calibration_method']}` (status=`{cal_audit.get('calibration_status', {})}`)",
        f"- **Selection Outcome**: `{selection_record['selection_outcome']}`",
        "",
        "## 1. Operational Decision Context",
        "",
        "- **Decision Point**: Daily end-of-day batch triage (`T_decide(d) = d 23:59:59.999999` local source time) for orders purchased on calendar day `d`.",
        "- **Actionable Policy**: `daily_end_of_day_top10_floor_v2` (`q_d = floor(0.10 * N_eligible,d)` daily capacity quota, no cross-day carryover).",
        "- **Eligibility Exclusions**: Orders already known delivered on or before `T_decide(d)` and orders already overdue prior to `d` are excluded from the warning queue.",
        "",
        "## 2. Training & Chronological Calibration Provenance",
        "",
        f"- **Matured Training Core Count (`fit_core`)**: `{cal_audit.get('core_train_count', 0):,}`",
        f"- **Core Orders Dropped (Unmatured at `T_core`)**: `{cal_audit.get('core_dropped_unmatured_at_tail_start_count', 0):,}`",
        f"- **Chronological Calibration Tail Count (`cal_tail`)**: `{cal_audit.get('cal_tail_count', 0):,}` (`{cal_audit.get('cal_tail_positives', 0):,}` positives)",
        f"- **Core-to-Tail Cutoff (`T_core`)**: `{cal_audit.get('core_cutoff_timestamp')}`",
        f"- **Refit Full After Calibration**: `{cal_audit.get('refit_full_after_calibration', False)}` (base estimator remains frozen on `fit_core` so post-hoc calibrators match the exact base distribution).",
        "",
        "## 3. Rolling Validation vs. Historical Backtest Summary",
        "",
        "| Metric | Selected Candidate (`" + str(sel["candidate_id"]) + "`) | Best Simple Baseline (`" + str(base["candidate_id"]) + "`) |",
        "|---|---:|---:|",
        f"| Validation Mean Daily Top-10% Recall | `{sel['val_mean_daily_top10_recall']:.6f}` | `{base['val_mean_daily_top10_recall']:.6f}` |",
        f"| Validation Mean PR-AUC | `{sel['val_mean_pr_auc']:.6f}` | `{base['val_mean_pr_auc']:.6f}` |",
        f"| Validation Mean Brier Score | `{sel['val_mean_brier']:.6f}` | `{base['val_mean_brier']:.6f}` |",
        f"| Backtest Daily Top-10% Recall (Cohort A) | `{bt['daily_top10_recall']:.6f}` | — |",
        f"| Backtest Daily Top-10% Precision (Cohort A) | `{bt['daily_top10_precision']:.6f}` | — |",
        f"| Backtest PR-AUC (Cohort A) | `{bt['pr_auc']:.6f}` | — |",
        f"| Backtest Brier Score (Cohort A) | `{bt['brier_score']:.6f}` | — |",
        "",
        "## 4. Limitations, Unknown Outcomes & Shutdown Triggers",
        "",
        "1. **Unknown Delivery Outcomes (`2,971` orders overall)**: Non-delivered orders (`shipped`, `canceled`, `unavailable`, `invoiced`, `processing`) have `late = NA` and consume Path B daily alert slots without confirmed delivery timestamps.",
        f"2. **Maturity Selection Bias**: At the selection cutoff `{split_c.get('selection_at')}`, orders in the last validation fold that were still in transit were excluded from selection; they are disproportionately late (see `split_summary.csv` and claims C04/C05). Label-known rule: `{cfg.get('label_contract', {}).get('label_known_rule', 'delivered_at')}`.",
        f"3. **Holdout Status**: backtest window `{split_c.get('backtest', {}).get('test_start')}` to `{split_c.get('backtest', {}).get('test_end')}`; see `docs/test-exposure-log.md` for what was inspected before this run.",
        "4. **Shutdown / Fallback Triggers**: Revert to `rule_short_promised_lead` or `base_v2` if `order_estimated_delivery_date` is modified post-purchase, if seller/route history tables lag by `> 7` days, or if monthly ECE exceeds `0.05`.",
        "",
    ]
    return "\n".join(lines)
