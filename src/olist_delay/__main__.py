"""CLI entrypoint for `python -m olist_delay {inspect-config,audit,prepare,backtest,evaluate,verify,score}`."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    RunContext,
    assert_writable_path,
    atomic_write_json,
    now_iso,
    verify_baseline_protection,
    verify_model_artifact_hash,
)
from .config import (
    PROJECT_ROOT,
    expand_model_candidates,
    inspect_config_summary,
    load_config,
    resolve_project_path,
)
from .data import build_master_order_table, load_raw_tables
from .labels import compute_label_known_at
from .evaluation import (
    compute_bootstrap_ci,
    compute_calibration_curve_table,
    compute_classification_metrics,
    compute_ex_post_stage_breakdown,
    compute_monthly_drift_report,
    compute_permutation_importance_table,
    compute_subgroup_metrics,
)
from .features import (
    ALL_MODEL_FEATURES,
    MODEL_INPUT_COLUMNS,
    BASE_V2_FEATURES,
    FORBIDDEN_POST_PURCHASE_COLUMNS,
    assert_zero_leakage_features,
    build_purchase_time_features,
    generate_feature_availability_audit,
    get_features_for_version,
)
from .models import (
    CalibratedModelBundle,
    fit_model_with_chronological_calibration,
)
from .pipeline_runner import (
    bootstrap_replicates_from_config,
    build_claims_index,
    build_metric_definitions,
    build_model_card_markdown,
    compute_cost_scenario_table,
    compute_sensitivity_summary_table,
    select_candidate_from_validation,
)
from .policy import (
    allocate_daily_alerts,
    assign_risk_tier,
    evaluate_cost_ratio_sensitivity,
    evaluate_topk_frontier_cohort_a,
    evaluate_topk_frontier_path_b,
    rank_orders_deterministically,
    summarize_daily_policy,
)
from .reporting import generate_all_run_figures
from .scoring import (
    build_history_and_decision_events_from_master,
    build_scoring_input_v2_from_raw,
    extract_scoring_input_v2_from_master,
    score_orders_batch,
)
from .splits import (
    build_cohort_observation_analysis,
    build_split_membership_table,
    generate_all_splits,
    summarize_splits,
)
from .validation import verify_run_directory


V2_BASELINE_NAMES = [
    "rule_random",
    "rule_short_promised_lead",
    "rule_seller_history",
    "rule_region_lead",
    "global_prior_baseline",
]


def _build_candidate_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic manifest of all model candidates and baselines for a run."""
    model_candidates = expand_model_candidates(cfg)
    baseline_names = list(cfg.get("baseline_rules", V2_BASELINE_NAMES))
    baseline_entries: list[dict[str, Any]] = []
    for b_name in baseline_names:
        baseline_entries.append(
            {
                "candidate_id": b_name,
                "model_family": b_name,
                "model_name": b_name,
                "feature_version": "base_v2",
                "calibration_method": "uncalibrated",
                "hyperparams": {},
                "hyperparams_json": "{}",
                "is_baseline": True,
            }
        )
    ml_entries: list[dict[str, Any]] = []
    for c in model_candidates:
        ml_entries.append(
            {
                "candidate_id": str(c["candidate_id"]),
                "model_family": str(c["model_family"]),
                "model_name": str(c["model_family"]),
                "feature_version": str(c["feature_version"]),
                "calibration_method": str(c["calibration_method"]),
                "hyperparams": dict(c["hyperparams"]),
                "hyperparams_json": str(c["hyperparams_json"]),
                "is_baseline": False,
            }
        )
    return {
        "schema_version": str(cfg.get("schema_version", "2.0.0")),
        "generated_at": now_iso(),
        "smoke_only": bool(cfg.get("smoke_only", False)),
        "model_candidate_count": len(ml_entries),
        "baseline_count": len(baseline_entries),
        "total_evaluated_entries": len(ml_entries) + len(baseline_entries),
        "model_candidates": ml_entries,
        "baselines": baseline_entries,
    }


def cmd_inspect_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=PROJECT_ROOT)
    summary = inspect_config_summary(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=PROJECT_ROOT)
    base_check = verify_baseline_protection(root=PROJECT_ROOT)
    if not base_check["all_intact"]:
        print(json.dumps(base_check, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    data_dir = resolve_project_path(cfg.get("paths", {}).get("data_dir", "data"), root=PROJECT_ROOT)
    tables = load_raw_tables(data_dir)
    primary_label = cfg.get("primary_label", cfg.get("label_contract", {}).get("primary_label", "calendar_day"))
    master, exceptions_df, audit_summary = build_master_order_table(tables, primary_label=primary_label)
    featured = build_purchase_time_features(master, label_col="late")
    fa_df = generate_feature_availability_audit(featured)
    splits = generate_all_splits(featured, cfg["split_contract"])
    sp_df = summarize_splits(featured, splits, label_col="late")
    cohort_obs_df = build_cohort_observation_analysis(featured, splits)

    out_dir = resolve_project_path(args.output_dir, root=PROJECT_ROOT) if args.output_dir else None
    if out_dir is not None:
        assert_writable_path(out_dir, root=PROJECT_ROOT)
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "baseline_protection.json", base_check)
        atomic_write_json(out_dir / "data_quality_summary.json", audit_summary)
        exceptions_df.to_csv(out_dir / "data_quality_exceptions.csv", index=False)
        cohort_obs_df.to_csv(out_dir / "cohort_observation_analysis.csv", index=False)
        fa_df.to_csv(out_dir / "feature_availability_audit.csv", index=False)
        sp_df.to_csv(out_dir / "split_summary.csv", index=False)

    summary_out = {
        "audited_at": now_iso(),
        "baseline_intact": base_check["all_intact"],
        "row_reconciliation": audit_summary["row_reconciliation"],
        "label_reconciliation": {
            "full_cohort_a": audit_summary["label_reconciliation"]["full_cohort_a"],
            "backtest_window": audit_summary["label_reconciliation"]["backtest_window"],
        },
        "feature_count": len(fa_df),
        "all_features_zero_leakage": bool(fa_df["audit_passed"].all()),
        "splits_verified": len(sp_df),
        "exceptions_count": len(exceptions_df),
    }
    print(json.dumps(summary_out, indent=2, ensure_ascii=False))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=PROJECT_ROOT)
    base_check = verify_baseline_protection(root=PROJECT_ROOT)
    if not base_check["all_intact"]:
        raise RuntimeError(f"Baseline integrity check failed: {base_check['mismatches']}")

    with RunContext(
        run_id=args.run_id,
        config=cfg,
        command="prepare",
        root=PROJECT_ROOT,
        parent_run_id=args.parent_run_id,
        rerun_reason=args.rerun_reason,
    ) as ctx:
        data_dir = resolve_project_path(cfg.get("paths", {}).get("data_dir", "data"), root=PROJECT_ROOT)
        ctx.log("Loading and validating 9 raw tables...")
        tables = load_raw_tables(data_dir)
        primary_label = cfg.get("primary_label", cfg.get("label_contract", {}).get("primary_label", "calendar_day"))

        ctx.log(f"Building 1:1 master order table with primary_label={primary_label!r}...")
        master, exceptions_df, audit_summary = build_master_order_table(tables, primary_label=primary_label)
        ctx.log("Attaching purchase-time and dual-clock historical features...")
        featured = build_purchase_time_features(master, label_col="late")

        fa_df = generate_feature_availability_audit(featured)
        splits = generate_all_splits(featured, cfg["split_contract"])
        sp_df = summarize_splits(featured, splits, label_col="late")
        cohort_obs_df = build_cohort_observation_analysis(featured, splits)
        split_mem_df = build_split_membership_table(featured, splits)

        cohort_cols = [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
            "label_available_at",
            "has_items",
            "in_cohort_a",
            "is_label_observable",
            "label_reason_code",
            "late_calendar_day",
            "late_timestamp",
            "same_day_after_midnight",
            "late",
        ]
        featured[cohort_cols].to_csv(ctx.run_dir / "cohort_audit.csv", index=False)
        atomic_write_json(ctx.run_dir / "data_quality_summary.json", audit_summary)
        exceptions_df.to_csv(ctx.run_dir / "data_quality_exceptions.csv", index=False)
        cohort_obs_df.to_csv(ctx.run_dir / "cohort_observation_analysis.csv", index=False)
        fa_df.to_csv(ctx.run_dir / "feature_availability_audit.csv", index=False)
        sp_df.to_csv(ctx.run_dir / "split_summary.csv", index=False)
        split_mem_df.to_csv(ctx.run_dir / "split_membership.csv", index=False)

        # Save sample inputs for offline `score` CLI demonstration & testing (orders on 2018-06-01 <= 2018-06-01T23:59:59)
        sample_as_of = pd.Timestamp("2018-06-01 23:59:59")
        sample_start = pd.Timestamp("2018-06-01 00:00:00")
        p_ts = pd.to_datetime(featured["order_purchase_timestamp"])
        sample_mask = (p_ts >= sample_start) & (p_ts <= sample_as_of)
        if not sample_mask.any():
            bt_split = [s for s in splits if s.split_type == "backtest"][0]
            sample_slice = featured.loc[bt_split.eval_path_b_all_mask].sort_values(
                by=["order_purchase_timestamp", "order_id"], kind="mergesort"
            ).head(250)
            sample_as_of = pd.to_datetime(sample_slice["order_purchase_timestamp"]).max()
        else:
            sample_slice = featured.loc[sample_mask].sort_values(
                by=["order_purchase_timestamp", "order_id"], kind="mergesort"
            )

        sample_orders_v2 = extract_scoring_input_v2_from_master(sample_slice)
        sample_orders_v2.to_csv(ctx.run_dir / "sample_orders.csv", index=False)

        hist_events, dec_events = build_history_and_decision_events_from_master(
            featured, cutoff_ts=sample_as_of
        )
        hist_events.to_csv(ctx.run_dir / "sample_events.csv", index=False)
        dec_events.to_csv(ctx.run_dir / "sample_decision_events.csv", index=False)
        ctx.log(
            f"Prepared {len(featured):,} orders, {len(fa_df)} verified features, "
            f"and {len(sample_orders_v2)} sample scoring orders (as_of={sample_as_of})."
        )

    print(f"Successfully completed prepare run: {args.run_id}")
    return 0


def _compute_and_write_evaluation_from_predictions(
    out_dir: Path,
    figures_dir: Path,
    run_id: str,
    cfg: dict[str, Any],
    audit_summary: dict[str, Any],
    sp_df: pd.DataFrame,
    cohort_obs_df: pd.DataFrame,
    candidate_manifest: dict[str, Any],
    selection_record: dict[str, Any],
    val_metrics_df: pd.DataFrame,
    val_preds_df: pd.DataFrame,
    backtest_preds_df: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    ex_post_df: pd.DataFrame,
) -> None:
    """Compute all model comparison, calibration, policy, sensitivity, uncertainty, claims, and figure outputs purely from frozen prediction tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("random_seed", 5126))
    budgets = cfg.get("policy_contract", {}).get("capacity_budgets_k", [0.05, 0.10, 0.15, 0.20, 0.30])
    cost_ratios = cfg.get("policy_contract", {}).get("cost_ratio_grid", [0.05, 0.08, 0.10, 0.15, 0.20, 0.25])

    selected_cand_id = str(selection_record["selected_candidate_id"])
    selected_model_name = str(selection_record.get("selected_model_name", selection_record["selected_model_family"]))
    selected_cal_method = str(selection_record["selected_calibration_method"])
    selected_feat_ver = str(selection_record.get("selected_feature_version", "base_v2"))
    selected_hp_json = str(selection_record.get("selected_hyperparams_json", "{}"))

    # Filter Cohort A matured orders in historical_backtest
    if "is_matured_at_selection" in backtest_preds_df.columns:
        df_bt_a = backtest_preds_df[
            backtest_preds_df["in_cohort_a"].astype(bool)
            & backtest_preds_df["is_matured_at_selection"].astype(bool)
        ].copy().reset_index(drop=True)
    else:
        df_bt_a = backtest_preds_df[backtest_preds_df["in_cohort_a"].astype(bool)].copy().reset_index(drop=True)

    y_bt_cal = df_bt_a["late_calendar_day"].astype(int).to_numpy()
    y_bt_ts = df_bt_a["late_timestamp"].astype(int).to_numpy()

    # Aggregate validation metrics per candidate_id
    val_agg = (
        val_metrics_df.groupby("candidate_id", as_index=False)
        .agg(
            val_mean_pr_auc=("pr_auc", "mean"),
            val_std_pr_auc=("pr_auc", "std"),
            val_mean_roc_auc=("roc_auc", "mean"),
            val_mean_brier=("brier_score", "mean"),
            val_mean_ece_10=("ece_quantile_10", "mean"),
            val_mean_prec_top10=("precision_top10", "mean"),
            val_mean_rec_top10=("recall_top10", "mean"),
        )
        .set_index("candidate_id")
    )

    all_entries = list(candidate_manifest.get("model_candidates", [])) + list(candidate_manifest.get("baselines", []))
    # Put selected candidate first so .iloc[0] queries on model_name match the selected candidate
    all_entries.sort(key=lambda e: (0 if str(e["candidate_id"]) == selected_cand_id else 1, str(e["candidate_id"])))

    model_comp_rows: list[dict[str, Any]] = []
    cal_curve_dfs: list[pd.DataFrame] = []
    pol_a_dfs: list[pd.DataFrame] = []

    for entry in all_entries:
        cand_id = str(entry["candidate_id"])
        m_name = str(entry["model_name"])
        var = str(entry["calibration_method"])
        feat_ver = str(entry.get("feature_version", "base_v2"))
        hp_json = str(entry.get("hyperparams_json", "{}"))
        is_base = bool(entry.get("is_baseline", False))
        score_col = f"score_{cand_id}"
        if score_col not in df_bt_a.columns:
            continue
        p_bt = df_bt_a[score_col].astype(float).to_numpy()

        is_sel = cand_id == selected_cand_id
        is_sel_base_family = (
            (not is_base)
            and (m_name == selected_model_name)
            and (feat_ver == selected_feat_ver)
            and (hp_json == selected_hp_json)
        )
        is_default_cal = is_base or (var == selected_cal_method)

        if is_sel_base_family or is_base:
            cal_df = compute_calibration_curve_table(
                y_true=y_bt_cal,
                y_prob=p_bt,
                model_name=m_name,
                calibration_method=var,
                split_id="historical_backtest",
            )
            cal_df["candidate_id"] = cand_id
            cal_curve_dfs.append(cal_df)

        v_row = val_agg.loc[cand_id] if cand_id in val_agg.index else None
        for lbl_ver, y_vec in [("calendar_day", y_bt_cal), ("timestamp", y_bt_ts)]:
            m_bt = compute_classification_metrics(
                order_ids=df_bt_a["order_id"],
                y_true=y_vec,
                y_prob=p_bt,
                model_name=m_name,
                calibration_method=var,
                split_id="historical_backtest",
                label_version=lbl_ver,
                purchase_timestamps=df_bt_a["order_purchase_timestamp"],
                candidate_id=cand_id,
                feature_version=feat_ver,
                hyperparams_json=hp_json,
            )
            model_comp_rows.append(
                {
                    "candidate_id": cand_id,
                    "model_name": m_name,
                    "model_family": "simple_baseline" if is_base else "ml_candidate",
                    "feature_version": feat_ver,
                    "hyperparams_json": hp_json,
                    "calibration_method": var,
                    "is_model_default_calibration": is_default_cal,
                    "is_selected_base_family": is_sel_base_family,
                    "is_selected_production_model": is_sel,
                    "label_version": lbl_ver,
                    "val_mean_pr_auc": round(float(v_row["val_mean_pr_auc"]), 6) if v_row is not None else np.nan,
                    "val_std_pr_auc": round(float(v_row["val_std_pr_auc"]), 6) if v_row is not None else np.nan,
                    "val_mean_roc_auc": round(float(v_row["val_mean_roc_auc"]), 6) if v_row is not None else np.nan,
                    "val_mean_brier": round(float(v_row["val_mean_brier"]), 6) if v_row is not None else np.nan,
                    "val_mean_ece_10": round(float(v_row["val_mean_ece_10"]), 6) if v_row is not None else np.nan,
                    "backtest_pr_auc": m_bt["pr_auc"],
                    "backtest_roc_auc": m_bt["roc_auc"],
                    "backtest_brier": m_bt["brier_score"],
                    "backtest_log_loss": m_bt["log_loss"],
                    "backtest_ece_10": m_bt["ece_quantile_10"],
                    "backtest_precision_top5": m_bt["precision_top5"],
                    "backtest_recall_top5": m_bt["recall_top5"],
                    "backtest_precision_top10": m_bt["precision_top10"],
                    "backtest_recall_top10": m_bt["recall_top10"],
                    "backtest_lift_top10": m_bt["lift_top10"],
                    "backtest_daily_precision_top10": m_bt["daily_top10_precision"],
                    "backtest_daily_recall_top10": m_bt["daily_top10_recall"],
                    "backtest_daily_alerts_top10": m_bt["daily_top10_alerts"],
                    "backtest_daily_top10_alert_count": m_bt["daily_top10_alert_count"],
                    "backtest_daily_top10_tp": m_bt["daily_top10_tp"],
                    "backtest_daily_top10_fp": m_bt["daily_top10_fp"],
                    "backtest_precision_top20": m_bt["precision_top20"],
                    "backtest_recall_top20": m_bt["recall_top20"],
                    "backtest_lift_top20": m_bt["lift_top20"],
                }
            )
            if is_sel or is_base:
                pol_df = evaluate_topk_frontier_cohort_a(
                    order_ids=df_bt_a["order_id"],
                    y_true=y_vec,
                    scores=p_bt,
                    budgets=budgets,
                    model_name=m_name,
                    split_id="historical_backtest",
                    label_name=lbl_ver,
                )
                pol_df["candidate_id"] = cand_id
                pol_df["is_selected_production_model"] = is_sel
                pol_a_dfs.append(pol_df)

        # Also add the uncalibrated diagnostic variant for the selected candidate so Figure 5 shows all 3 variants
        if is_sel and var != "uncalibrated" and "predicted_prob_uncalibrated" in df_bt_a.columns:
            p_uncal_bt = df_bt_a["predicted_prob_uncalibrated"].astype(float).to_numpy()
            uncal_id = f"{cand_id}__uncalibrated"
            cal_df_u = compute_calibration_curve_table(
                y_true=y_bt_cal,
                y_prob=p_uncal_bt,
                model_name=m_name,
                calibration_method="uncalibrated",
                split_id="historical_backtest",
            )
            cal_df_u["candidate_id"] = uncal_id
            cal_curve_dfs.append(cal_df_u)
            for lbl_ver, y_vec in [("calendar_day", y_bt_cal), ("timestamp", y_bt_ts)]:
                m_bt_u = compute_classification_metrics(
                    order_ids=df_bt_a["order_id"],
                    y_true=y_vec,
                    y_prob=p_uncal_bt,
                    model_name=m_name,
                    calibration_method="uncalibrated",
                    split_id="historical_backtest",
                    label_version=lbl_ver,
                    purchase_timestamps=df_bt_a["order_purchase_timestamp"],
                    candidate_id=uncal_id,
                    feature_version=feat_ver,
                    hyperparams_json=hp_json,
                )
                model_comp_rows.append(
                    {
                        "candidate_id": uncal_id,
                        "model_name": m_name,
                        "model_family": "ml_candidate",
                        "feature_version": feat_ver,
                        "hyperparams_json": hp_json,
                        "calibration_method": "uncalibrated",
                        "is_model_default_calibration": False,
                        "is_selected_base_family": True,
                        "is_selected_production_model": False,
                        "label_version": lbl_ver,
                        "val_mean_pr_auc": round(float(v_row["val_mean_pr_auc"]), 6) if v_row is not None else np.nan,
                        "val_std_pr_auc": round(float(v_row["val_std_pr_auc"]), 6) if v_row is not None else np.nan,
                        "val_mean_roc_auc": round(float(v_row["val_mean_roc_auc"]), 6) if v_row is not None else np.nan,
                        "val_mean_brier": np.nan,
                        "val_mean_ece_10": np.nan,
                        "backtest_pr_auc": m_bt_u["pr_auc"],
                        "backtest_roc_auc": m_bt_u["roc_auc"],
                        "backtest_brier": m_bt_u["brier_score"],
                        "backtest_log_loss": m_bt_u["log_loss"],
                        "backtest_ece_10": m_bt_u["ece_quantile_10"],
                        "backtest_precision_top5": m_bt_u["precision_top5"],
                        "backtest_recall_top5": m_bt_u["recall_top5"],
                        "backtest_precision_top10": m_bt_u["precision_top10"],
                        "backtest_recall_top10": m_bt_u["recall_top10"],
                        "backtest_lift_top10": m_bt_u["lift_top10"],
                        "backtest_daily_precision_top10": m_bt_u["daily_top10_precision"],
                        "backtest_daily_recall_top10": m_bt_u["daily_top10_recall"],
                        "backtest_daily_alerts_top10": m_bt_u["daily_top10_alerts"],
                        "backtest_daily_top10_alert_count": m_bt_u["daily_top10_alert_count"],
                        "backtest_daily_top10_tp": m_bt_u["daily_top10_tp"],
                        "backtest_daily_top10_fp": m_bt_u["daily_top10_fp"],
                        "backtest_precision_top20": m_bt_u["precision_top20"],
                        "backtest_recall_top20": m_bt_u["recall_top20"],
                        "backtest_lift_top20": m_bt_u["lift_top20"],
                    }
                )

    model_comp_df = pd.DataFrame(model_comp_rows)
    cal_summary_df = pd.concat(cal_curve_dfs, ignore_index=True) if cal_curve_dfs else pd.DataFrame()
    pol_a_all_df = pd.concat(pol_a_dfs, ignore_index=True) if pol_a_dfs else pd.DataFrame()

    p_b_cal = backtest_preds_df["predicted_prob_calibrated"].astype(float).to_numpy()
    pol_b_df = evaluate_topk_frontier_path_b(
        df_path_b=backtest_preds_df,
        scores=p_b_cal,
        budgets=budgets,
        model_name=selected_model_name,
        split_id="historical_backtest",
        label_col="late",
    )

    # Daily operational alert allocation (Path B all orders + Cohort A matured)
    daily_alloc_df = allocate_daily_alerts(
        df_orders=backtest_preds_df,
        scores=p_b_cal,
        budget_k=0.10,
        label_col="late",
        seed=seed,
    )
    daily_policy_df = summarize_daily_policy(
        df_allocated=daily_alloc_df,
        model_name=selected_model_name,
        split_id="historical_backtest",
        budget_k=0.10,
        label_col="late",
    )

    p_a_cal = df_bt_a["predicted_prob_calibrated"].astype(float).to_numpy()
    p_a_uncal = df_bt_a["predicted_prob_uncalibrated"].astype(float).to_numpy()
    p_a_rand = (
        df_bt_a["score_rule_random"].astype(float).to_numpy()
        if "score_rule_random" in df_bt_a.columns
        else None
    )

    cost_sens_df = evaluate_cost_ratio_sensitivity(
        y_true=y_bt_cal,
        scores_calibrated=p_a_cal,
        scores_uncalibrated=p_a_uncal,
        cost_ratios=cost_ratios,
        model_name=selected_model_name,
        split_id="historical_backtest",
        scores_random_baseline=p_a_rand,
    )

    subgroup_df = compute_subgroup_metrics(
        df_eval=df_bt_a,
        scores=p_a_cal,
        label_col="late",
        primary_budget=0.10,
    )

    # Monthly drift strictly from frozen OOF validation_predictions.csv + backtest_predictions.csv
    val_coh_a = val_preds_df[val_preds_df["in_cohort_a"].astype(bool)][
        ["order_id", "split_id", "order_purchase_timestamp", "late", "predicted_prob_calibrated"]
    ].copy()
    bt_coh_a_slice = df_bt_a[
        ["order_id", "split_id", "order_purchase_timestamp", "late", "predicted_prob_calibrated"]
    ].copy()
    all_eval_preds_df = pd.concat([val_coh_a, bt_coh_a_slice], ignore_index=True)
    monthly_drift_df = compute_monthly_drift_report(
        all_eval_preds_df, score_col="predicted_prob_calibrated", label_col="late"
    )

    # Best operational baseline on validation folds for paired bootstrap comparison
    best_base_id = str(selection_record.get("best_baseline_name", "rule_region_lead"))
    base_col = f"score_{best_base_id}" if f"score_{best_base_id}" in df_bt_a.columns else "score_global_prior_baseline"
    p_a_base = df_bt_a[base_col].astype(float).to_numpy()

    # Replicate counts come from the configuration's uncertainty contract (previously hard-coded to 400).
    # One call computes both the row bootstrap (`metrics`) and the weekly block bootstrap (nested block); the two
    # summary sections below were previously produced by two identical calls with the same seed.
    n_row_boot, n_block_boot = bootstrap_replicates_from_config(cfg)
    boot_all = compute_bootstrap_ci(
        y_true=y_bt_cal,
        y_prob_model=p_a_cal,
        y_prob_baseline=p_a_base,
        n_bootstrap=n_row_boot,
        random_seed=seed,
        order_ids=df_bt_a["order_id"],
        purchase_timestamps=df_bt_a["order_purchase_timestamp"],
        n_weekly_bootstrap=n_block_boot,
    )
    boot_iid = {**boot_all, "method": "iid_bootstrap"}
    boot_block = {**boot_all, "method": "weekly_block_bootstrap"}

    cost_scenarios_df = compute_cost_scenario_table(
        df_eval=df_bt_a,
        scores=p_a_cal,
        label_col="late_calendar_day",
        budgets=budgets,
        scenarios=cfg.get("cost_scenarios", []),
        model_name=selected_model_name,
        split_id="historical_backtest",
    )
    uncertainty_summary = {
        "run_id": run_id,
        "generated_at": now_iso(),
        "selected_candidate_id": selected_cand_id,
        "compared_baseline_id": best_base_id,
        "iid_bootstrap": boot_iid,
        "weekly_block_bootstrap": boot_block,
    }

    sel_bt_metrics_cal = compute_classification_metrics(
        order_ids=df_bt_a["order_id"],
        y_true=y_bt_cal,
        y_prob=p_a_cal,
        model_name=selected_model_name,
        calibration_method=selected_cal_method,
        split_id="historical_backtest",
        label_version="calendar_day",
        purchase_timestamps=df_bt_a["order_purchase_timestamp"],
        candidate_id=selected_cand_id,
        feature_version=selected_feat_ver,
        hyperparams_json=selected_hp_json,
    )
    sel_bt_metrics_ts = compute_classification_metrics(
        order_ids=df_bt_a["order_id"],
        y_true=y_bt_ts,
        y_prob=p_a_cal,
        model_name=selected_model_name,
        calibration_method=selected_cal_method,
        split_id="historical_backtest",
        label_version="timestamp",
        purchase_timestamps=df_bt_a["order_purchase_timestamp"],
        candidate_id=selected_cand_id,
        feature_version=selected_feat_ver,
        hyperparams_json=selected_hp_json,
    )

    sens_df = compute_sensitivity_summary_table(
        df_bt_a=df_bt_a,
        df_bt_b=backtest_preds_df,
        p_a_cal=p_a_cal,
        p_b_cal=p_b_cal,
        selected_candidate_id=selected_cand_id,
        selected_cal_method=selected_cal_method,
        best_baseline_id=best_base_id,
        p_a_baseline=p_a_base,
        model_comp_df=model_comp_df,
    )

    sel_val_row = val_agg.loc[selected_cand_id]
    metrics_summary = {
        "run_id": run_id,
        "generated_at": now_iso(),
        "selected_model": {
            "candidate_id": selected_cand_id,
            "model_name": selected_model_name,
            "feature_version": selected_feat_ver,
            "calibration_method": selected_cal_method,
            "selection_rule": selection_record.get("selection_rule", "highest_mean_validation_pr_auc_across_4_rolling_folds"),
            "validation_mean_pr_auc": round(float(sel_val_row["val_mean_pr_auc"]), 6),
            "validation_mean_roc_auc": round(float(sel_val_row["val_mean_roc_auc"]), 6),
            "validation_mean_brier": round(float(sel_val_row["val_mean_brier"]), 6),
        },
        "selected_model_backtest_metrics": sel_bt_metrics_cal,
        "selected_model_backtest_metrics_timestamp_label": sel_bt_metrics_ts,
        "bootstrap_95_ci": boot_block,
        "iid_bootstrap_95_ci": boot_iid,
        "top_10_features_by_permutation_importance": feat_imp_df.head(10).to_dict(orient="records"),
    }

    # Write all evaluation tables and JSONs to out_dir BEFORE computing claims.json file hashes
    val_metrics_df.to_csv(out_dir / "validation_folds_metrics.csv", index=False)
    model_comp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    cal_summary_df.to_csv(out_dir / "calibration_summary.csv", index=False)
    pol_a_all_df.to_csv(out_dir / "policy_frontier_cohort_a.csv", index=False)
    pol_b_df.to_csv(out_dir / "policy_frontier_all_orders.csv", index=False)
    daily_policy_df.to_csv(out_dir / "daily_policy.csv", index=False)
    cost_sens_df.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    cost_scenarios_df.to_csv(out_dir / "cost_scenarios.csv", index=False)
    subgroup_df.to_csv(out_dir / "subgroup_metrics.csv", index=False)
    monthly_drift_df.to_csv(out_dir / "monthly_drift_report.csv", index=False)
    feat_imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    ex_post_df.to_csv(out_dir / "ex_post_stage_summary.csv", index=False)
    sens_df.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    atomic_write_json(out_dir / "metrics_summary.json", metrics_summary)
    atomic_write_json(out_dir / "uncertainty_summary.json", uncertainty_summary)

    claims_index = build_claims_index(
        run_id=run_id,
        run_dir=out_dir,
        audit_summary=audit_summary,
        sp_df=sp_df,
        selection_record=selection_record,
        metrics_summary=metrics_summary,
        daily_policy_df=daily_policy_df,
        pol_b_df=pol_b_df,
    )
    metric_defs = build_metric_definitions()
    cal_audit_dict = audit_summary.get("calibration_audit", selection_record.get("calibration_audit", {}))
    model_card_md = build_model_card_markdown(
        run_id=run_id,
        selection_record=selection_record,
        metrics_summary=metrics_summary,
        cal_audit=cal_audit_dict,
        cfg=cfg,
    )

    atomic_write_json(out_dir / "claims.json", claims_index)
    atomic_write_json(out_dir / "metric_definitions.json", metric_defs)
    (out_dir / "model_card.md").write_text(model_card_md, encoding="utf-8")

    generate_all_run_figures(
        figures_dir=figures_dir,
        audit_summary=audit_summary,
        split_summary_df=sp_df,
        model_comparison_df=model_comp_df,
        calibration_curve_df=cal_summary_df,
        policy_cohort_a_df=pol_a_all_df,
        policy_path_b_df=pol_b_df,
        cost_sensitivity_df=cost_sens_df,
        subgroup_df=subgroup_df,
        monthly_drift_df=monthly_drift_df,
        feature_importance_df=feat_imp_df,
        ex_post_stage_df=ex_post_df,
        backtest_predictions_df=backtest_preds_df,
        selected_model_name=selected_model_name,
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=PROJECT_ROOT)
    base_check = verify_baseline_protection(root=PROJECT_ROOT)
    if not base_check["all_intact"]:
        raise RuntimeError(f"Baseline integrity check failed: {base_check['mismatches']}")

    seed = int(cfg.get("random_seed", 5126))
    with RunContext(
        run_id=args.run_id,
        config=cfg,
        command="backtest",
        root=PROJECT_ROOT,
        parent_run_id=args.parent_run_id,
        rerun_reason=args.rerun_reason,
    ) as ctx:
        data_dir = resolve_project_path(cfg.get("paths", {}).get("data_dir", "data"), root=PROJECT_ROOT)
        ctx.log("Loading and validating 9 raw tables...")
        tables = load_raw_tables(data_dir)
        primary_label = cfg.get("primary_label", cfg.get("label_contract", {}).get("primary_label", "calendar_day"))

        ctx.log("Building master order table and causal dual-clock features...")
        master, exceptions_df, audit_summary = build_master_order_table(tables, primary_label=primary_label)
        featured = build_purchase_time_features(master, label_col="late")
        fa_df = generate_feature_availability_audit(featured)
        # When an order's label counts as known: delivery (v2) or, censoring-aware (v3), the day after a missed promise.
        known_rule = str(cfg.get("label_contract", {}).get("label_known_rule", "delivered_at"))
        featured["label_known_at"] = compute_label_known_at(featured, known_rule)
        ctx.log(f"Label-known rule: {known_rule}")
        splits = generate_all_splits(featured, cfg["split_contract"], known_col="label_known_at")
        sp_df = summarize_splits(featured, splits, label_col="late", known_col="label_known_at")
        cohort_obs_df = build_cohort_observation_analysis(featured, splits)
        split_mem_df = build_split_membership_table(featured, splits)

        cohort_cols = [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
            "label_available_at",
            "has_items",
            "in_cohort_a",
            "is_label_observable",
            "label_reason_code",
            "late_calendar_day",
            "late_timestamp",
            "same_day_after_midnight",
            "late",
        ]
        featured[cohort_cols].to_csv(ctx.run_dir / "cohort_audit.csv", index=False)
        atomic_write_json(ctx.run_dir / "data_quality_summary.json", audit_summary)
        exceptions_df.to_csv(ctx.run_dir / "data_quality_exceptions.csv", index=False)
        cohort_obs_df.to_csv(ctx.run_dir / "cohort_observation_analysis.csv", index=False)
        fa_df.to_csv(ctx.run_dir / "feature_availability_audit.csv", index=False)
        sp_df.to_csv(ctx.run_dir / "split_summary.csv", index=False)
        split_mem_df.to_csv(ctx.run_dir / "split_membership.csv", index=False)

        candidate_manifest = _build_candidate_manifest(cfg)
        atomic_write_json(ctx.run_dir / "candidate_manifest.json", candidate_manifest)

        cal_cfg = cfg.get("calibration_contract", cfg.get("calibration", {}))
        cal_tail_frac = float(cal_cfg.get("tail_fraction", cal_cfg.get("calibration_tail_fraction", 0.20)))
        min_tail_samples = int(cal_cfg.get("min_tail_samples", 100))
        min_tail_positives = int(cal_cfg.get("min_tail_positives", 10))
        refit_full = bool(cal_cfg.get("refit_full_after_calibration", False))

        ml_candidates = candidate_manifest["model_candidates"]
        baseline_entries = candidate_manifest["baselines"]

        # Group ML candidates by (model_family, feature_version, hyperparams_json) so we fit each base model once per fold
        base_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for c in ml_candidates:
            key = (str(c["model_family"]), str(c["feature_version"]), str(c["hyperparams_json"]))
            base_groups.setdefault(key, []).append(c)

        # 1. Rolling validation across 4 expanding-window folds
        val_rows: list[dict[str, Any]] = []
        val_pred_slices: list[pd.DataFrame] = []
        val_splits = [s for s in splits if s.split_type == "validation"]

        for sp in val_splits:
            ctx.log(
                f"Running validation fold {sp.split_id}: matured_train={int(sp.train_matured_mask.sum()):,}, "
                f"dropped_in_transit={int(sp.train_in_transit_dropped_mask.sum()):,}, "
                f"val_matured={int(sp.eval_matured_at_selection_mask.sum()):,}"
            )
            df_tr = featured.loc[sp.train_matured_mask]
            df_va = featured.loc[sp.eval_matured_at_selection_mask].copy().reset_index(drop=True)
            X_tr = df_tr[MODEL_INPUT_COLUMNS]
            y_tr = df_tr["late"].astype(int)
            ts_tr = df_tr["order_purchase_timestamp"]
            la_tr = df_tr["label_known_at"]
            X_va = df_va[MODEL_INPUT_COLUMNS]
            y_va = df_va["late"].astype(int)

            fold_pred_df = pd.DataFrame(
                {
                    "split_id": sp.split_id,
                    "order_id": df_va["order_id"],
                    "order_purchase_timestamp": df_va["order_purchase_timestamp"],
                    "order_estimated_delivery_date": df_va["order_estimated_delivery_date"],
                    "in_cohort_a": df_va["in_cohort_a"],
                    "late": df_va["late"],
                    "late_calendar_day": df_va["late_calendar_day"],
                    "late_timestamp": df_va["late_timestamp"],
                    "customer_state": df_va["customer_state"],
                    "customer_region": df_va["customer_region"],
                }
            )

            # Evaluate all baselines on this validation fold
            for b_entry in baseline_entries:
                b_name = str(b_entry["model_name"])
                b_bundle = fit_model_with_chronological_calibration(
                    model_name=b_name,
                    X_train=X_tr,
                    y_train=y_tr,
                    purchase_timestamps=ts_tr,
                    label_available_at=la_tr,
                    params=b_entry["hyperparams"],
                    random_seed=seed,
                    calibration_tail_fraction=cal_tail_frac,
                    active_calibration="uncalibrated",
                    feature_version="base_v2",
                    min_tail_samples=min_tail_samples,
                    min_tail_positives=min_tail_positives,
                    refit_full_after_calibration=refit_full,
                )
                p_va = b_bundle.predict_proba_variant(X_va, method="uncalibrated")
                fold_pred_df[f"score_{b_name}"] = p_va
                fold_pred_df[f"uncal_{b_name}"] = p_va
                m_res = compute_classification_metrics(
                    order_ids=df_va["order_id"],
                    y_true=y_va,
                    y_prob=p_va,
                    model_name=b_name,
                    calibration_method="uncalibrated",
                    split_id=sp.split_id,
                    label_version=primary_label,
                    purchase_timestamps=df_va["order_purchase_timestamp"],
                    candidate_id=b_name,
                    feature_version="base_v2",
                    hyperparams_json="{}",
                    calibration_status=b_bundle.calibration_status,
                )
                val_rows.append(m_res)

            # Evaluate all ML candidate groups on this validation fold
            for (m_fam, f_ver, hp_json), group_cands in base_groups.items():
                hp_dict = group_cands[0]["hyperparams"]
                bundle = fit_model_with_chronological_calibration(
                    model_name=m_fam,
                    X_train=X_tr,
                    y_train=y_tr,
                    purchase_timestamps=ts_tr,
                    label_available_at=la_tr,
                    params=hp_dict,
                    random_seed=seed,
                    calibration_tail_fraction=cal_tail_frac,
                    active_calibration="platt",
                    feature_version=f_ver,
                    min_tail_samples=min_tail_samples,
                    min_tail_positives=min_tail_positives,
                    refit_full_after_calibration=refit_full,
                )
                p_va_uncal = bundle.predict_proba_variant(X_va, method="uncalibrated")
                for c in group_cands:
                    cand_id = str(c["candidate_id"])
                    cal_m = str(c["calibration_method"])
                    p_va = bundle.predict_proba_variant(X_va, method=cal_m)
                    fold_pred_df[f"score_{cand_id}"] = p_va
                    fold_pred_df[f"uncal_{cand_id}"] = p_va_uncal
                    m_res = compute_classification_metrics(
                        order_ids=df_va["order_id"],
                        y_true=y_va,
                        y_prob=p_va,
                        model_name=m_fam,
                        calibration_method=cal_m,
                        split_id=sp.split_id,
                        label_version=primary_label,
                        purchase_timestamps=df_va["order_purchase_timestamp"],
                        candidate_id=cand_id,
                        feature_version=f_ver,
                        hyperparams_json=hp_json,
                        calibration_status=bundle.calibration_status,
                    )
                    val_rows.append(m_res)

            val_pred_slices.append(fold_pred_df)

        val_metrics_df = pd.DataFrame(val_rows)
        val_preds_df = pd.concat(val_pred_slices, ignore_index=True)

        # Select winning candidate strictly on rolling validation folds BEFORE touching historical_backtest
        selection_record = select_candidate_from_validation(val_metrics_df, cfg)
        selected_cand_id = str(selection_record["selected_candidate_id"])
        selected_model_name = str(selection_record.get("selected_model_name", selection_record["selected_model_family"]))
        selected_cal_method = str(selection_record["selected_calibration_method"])
        selected_feat_ver = str(selection_record["selected_feature_version"])
        selected_hp_json = str(selection_record["selected_hyperparams_json"])

        val_preds_df["selected_candidate_id"] = selected_cand_id
        val_preds_df["predicted_prob_calibrated"] = val_preds_df[f"score_{selected_cand_id}"]
        val_preds_df["predicted_prob_uncalibrated"] = val_preds_df.get(
            f"uncal_{selected_cand_id}", val_preds_df[f"score_{selected_cand_id}"]
        )
        uncal_cols = [c for c in val_preds_df.columns if c.startswith("uncal_")]
        val_preds_df.drop(columns=uncal_cols, inplace=True)

        # Persist validation artifacts immediately before holdout backtest
        val_metrics_df.to_csv(ctx.run_dir / "validation_folds_metrics.csv", index=False)
        val_preds_df.to_csv(ctx.run_dir / "validation_predictions.csv", index=False)
        atomic_write_json(ctx.run_dir / "selection_record.json", selection_record)
        ctx.log(
            f"Selected candidate on validation folds: {selected_cand_id} "
            f"({selected_model_name}, feature_version={selected_feat_ver}, calibration={selected_cal_method})"
        )

        # 2. Fit models on historical_backtest matured training set
        bt_split = [s for s in splits if s.split_type == "backtest"][0]
        df_bt_tr = featured.loc[bt_split.train_matured_mask]
        X_bt_tr = df_bt_tr[MODEL_INPUT_COLUMNS]
        y_bt_tr = df_bt_tr["late"].astype(int)
        ts_bt_tr = df_bt_tr["order_purchase_timestamp"]
        la_bt_tr = df_bt_tr["label_known_at"]

        ctx.log(
            f"Fitting models on historical_backtest matured train ({len(df_bt_tr):,} orders; "
            f"{int(bt_split.train_in_transit_dropped_mask.sum()):,} in-transit dropped)..."
        )
        bt_bundles_by_cand: dict[str, CalibratedModelBundle] = {}

        for b_entry in baseline_entries:
            b_name = str(b_entry["model_name"])
            bt_bundles_by_cand[b_name] = fit_model_with_chronological_calibration(
                model_name=b_name,
                X_train=X_bt_tr,
                y_train=y_bt_tr,
                purchase_timestamps=ts_bt_tr,
                label_available_at=la_bt_tr,
                params=b_entry["hyperparams"],
                random_seed=seed,
                calibration_tail_fraction=cal_tail_frac,
                active_calibration="uncalibrated",
                feature_version="base_v2",
                min_tail_samples=min_tail_samples,
                min_tail_positives=min_tail_positives,
                refit_full_after_calibration=refit_full,
            )

        for (m_fam, f_ver, hp_json), group_cands in base_groups.items():
            hp_dict = group_cands[0]["hyperparams"]
            shared_bundle = fit_model_with_chronological_calibration(
                model_name=m_fam,
                X_train=X_bt_tr,
                y_train=y_bt_tr,
                purchase_timestamps=ts_bt_tr,
                label_available_at=la_bt_tr,
                params=hp_dict,
                random_seed=seed,
                calibration_tail_fraction=cal_tail_frac,
                active_calibration=selected_cal_method if (m_fam == selected_model_name and f_ver == selected_feat_ver and hp_json == selected_hp_json) else "platt",
                feature_version=f_ver,
                min_tail_samples=min_tail_samples,
                min_tail_positives=min_tail_positives,
                refit_full_after_calibration=refit_full,
            )
            for c in group_cands:
                bt_bundles_by_cand[str(c["candidate_id"])] = shared_bundle

        sel_bundle = bt_bundles_by_cand[selected_cand_id]
        fallback_prior = float(bt_bundles_by_cand["global_prior_baseline"].base_estimator.prior_)
        audit_summary["calibration_audit"] = sel_bundle.audit_summary()
        atomic_write_json(ctx.run_dir / "data_quality_summary.json", audit_summary)

        # Persist selected model bundle for `score` CLI
        ctx.models_dir.mkdir(parents=True, exist_ok=True)
        with (ctx.models_dir / "selected_model.pkl").open("wb") as f_mod:
            pickle.dump(
                {
                    "selected_candidate_id": selected_cand_id,
                    "selected_model_name": selected_model_name,
                    "selected_feature_version": selected_feat_ver,
                    "selected_calibration_method": selected_cal_method,
                    "selected_bundle": sel_bundle,
                    "all_bundles": bt_bundles_by_cand,
                    "fallback_prior": fallback_prior,
                },
                f_mod,
            )

        # 3. Score full Path B backtest window (all 19,661 orders placed in [test_start, test_end))
        df_bt_b = featured.loc[bt_split.eval_path_b_all_mask].copy().reset_index(drop=True)
        is_matured_bt = bt_split.eval_matured_at_selection_mask.loc[bt_split.eval_path_b_all_mask].to_numpy(dtype=bool)
        X_bt_b = df_bt_b[MODEL_INPUT_COLUMNS]
        has_items_mask = df_bt_b["has_items"].fillna(False).to_numpy(dtype=bool)

        p_b_uncal = sel_bundle.predict_proba_variant(X_bt_b, method="uncalibrated")
        p_b_cal = sel_bundle.predict_proba_variant(X_bt_b, method=selected_cal_method)
        p_b_cal = np.where(has_items_mask, p_b_cal, fallback_prior)
        p_b_uncal = np.where(has_items_mask, p_b_uncal, fallback_prior)

        ranked_b = rank_orders_deterministically(
            df_bt_b["order_id"],
            p_b_cal,
            purchase_timestamps=df_bt_b["order_purchase_timestamp"],
        )
        tiers_b = assign_risk_tier(ranked_b["risk_percentile"], primary_budget=0.10, secondary_budget=0.20)

        backtest_preds = pd.DataFrame(
            {
                "order_id": df_bt_b["order_id"],
                "split_id": "historical_backtest",
                "order_purchase_timestamp": df_bt_b["order_purchase_timestamp"],
                "order_estimated_delivery_date": df_bt_b["order_estimated_delivery_date"],
                "order_delivered_customer_date": df_bt_b["order_delivered_customer_date"],
                "label_available_at": df_bt_b["label_available_at"],
                "order_status": df_bt_b["order_status"],
                "has_items": df_bt_b["has_items"],
                "in_cohort_a": df_bt_b["in_cohort_a"],
                "is_matured_at_selection": is_matured_bt,
                "label_reason_code": df_bt_b["label_reason_code"],
                "late": df_bt_b["late"],
                "late_calendar_day": df_bt_b["late_calendar_day"],
                "late_timestamp": df_bt_b["late_timestamp"],
                "customer_state": df_bt_b["customer_state"],
                "customer_region": df_bt_b["customer_region"],
                "primary_seller_state": df_bt_b["primary_seller_state"],
                "same_state": df_bt_b["same_state"],
                "same_region": df_bt_b["same_region"],
                "seller_completed_count": df_bt_b["seller_completed_count"],
                "seller_is_cold_start": df_bt_b["seller_is_cold_start"],
                "haversine_distance_km": df_bt_b["haversine_distance_km"],
                "estimated_lead_calendar_days": df_bt_b["estimated_lead_calendar_days"],
                "freight_ratio": df_bt_b["freight_ratio"],
                "item_count": df_bt_b["item_count"],
                "distinct_seller_count": df_bt_b["distinct_seller_count"],
                "seller_count": df_bt_b["distinct_seller_count"],
                "selected_candidate_id": selected_cand_id,
                "predicted_prob_uncalibrated": p_b_uncal,
                "predicted_prob_calibrated": p_b_cal,
                "risk_rank_path_b": ranked_b["risk_rank"],
                "risk_percentile_path_b": np.round(ranked_b["risk_percentile"], 6),
                "risk_tier": tiers_b,
            }
        )

        # Store per-candidate and per-baseline scores on Path B
        for b_entry in baseline_entries:
            b_name = str(b_entry["model_name"])
            s_b = bt_bundles_by_cand[b_name].predict_proba_variant(X_bt_b, method="uncalibrated")
            backtest_preds[f"score_{b_name}"] = np.where(has_items_mask, s_b, fallback_prior)

        best_by_family: dict[str, tuple[float, np.ndarray]] = {}
        val_mean_by_cand = val_metrics_df.groupby("candidate_id")["pr_auc"].mean().to_dict()
        for c in ml_candidates:
            cand_id = str(c["candidate_id"])
            cal_m = str(c["calibration_method"])
            m_fam = str(c["model_family"])
            s_c = bt_bundles_by_cand[cand_id].predict_proba_variant(X_bt_b, method=cal_m)
            s_c = np.where(has_items_mask, s_c, fallback_prior)
            backtest_preds[f"score_{cand_id}"] = s_c
            v_score = float(val_mean_by_cand.get(cand_id, -1.0))
            if m_fam not in best_by_family or v_score > best_by_family[m_fam][0]:
                best_by_family[m_fam] = (v_score, s_c)

        for m_fam, (_, s_arr) in best_by_family.items():
            backtest_preds[f"score_{m_fam}"] = s_arr

        backtest_preds.to_csv(ctx.run_dir / "backtest_predictions.csv", index=False)

        # Compute permutation importance and ex-post stage breakdown once during backtest
        df_bt_a = featured.loc[bt_split.eval_matured_at_selection_mask].copy().reset_index(drop=True)
        feat_cols_sel = get_features_for_version(selected_feat_ver)
        ctx.log("Computing out-of-sample permutation feature importance on historical_backtest...")
        feat_imp_df = compute_permutation_importance_table(
            estimator=sel_bundle.base_estimator,
            X_eval=df_bt_a[feat_cols_sel],
            y_eval=df_bt_a["late_calendar_day"].astype(int).to_numpy(),
            feature_cols=feat_cols_sel,
            random_seed=seed,
            n_repeats=5,
            max_samples=8000,
        )
        ex_post_df = compute_ex_post_stage_breakdown(featured, label_col="late")

        ctx.log("Computing evaluation tables, uncertainty CIs, claims index, model card, and 8 figures...")
        _compute_and_write_evaluation_from_predictions(
            out_dir=ctx.run_dir,
            figures_dir=ctx.figures_dir,
            run_id=ctx.run_id,
            cfg=cfg,
            audit_summary=audit_summary,
            sp_df=sp_df,
            cohort_obs_df=cohort_obs_df,
            candidate_manifest=candidate_manifest,
            selection_record=selection_record,
            val_metrics_df=val_metrics_df,
            val_preds_df=val_preds_df,
            backtest_preds_df=backtest_preds,
            feat_imp_df=feat_imp_df,
            ex_post_df=ex_post_df,
        )

    print(f"Successfully completed backtest run: {args.run_id}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Recompute evaluation tables and figures from frozen validation_predictions.csv & backtest_predictions.csv into a separate --output-dir."""
    if not args.output_dir:
        raise ValueError(
            "The `evaluate` command requires an explicit `--output-dir` outside `artifacts/runs/<run-id>/` "
            "to prevent mutating frozen run artifacts."
        )
    run_dir = (PROJECT_ROOT / "artifacts" / "runs" / args.run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = resolve_project_path(args.output_dir, root=PROJECT_ROOT).resolve()
    assert_writable_path(out_dir, root=PROJECT_ROOT)
    if out_dir == run_dir or out_dir.is_relative_to(run_dir):
        raise PermissionError(
            f"Refusing to write evaluation output inside frozen run directory: {out_dir} (run_dir={run_dir})"
        )

    val_preds_path = run_dir / "validation_predictions.csv"
    bt_preds_path = run_dir / "backtest_predictions.csv"
    if not val_preds_path.exists():
        raise FileNotFoundError(
            f"Cannot evaluate run '{args.run_id}': missing required frozen OOF predictions at {val_preds_path}. "
            "Re-scoring validation folds with the backtest model is forbidden by the research contract."
        )
    if not bt_preds_path.exists():
        raise FileNotFoundError(
            f"Cannot evaluate run '{args.run_id}': missing required backtest predictions at {bt_preds_path}."
        )

    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    audit_summary = json.loads((run_dir / "data_quality_summary.json").read_text(encoding="utf-8"))
    sp_df = pd.read_csv(run_dir / "split_summary.csv")
    cohort_obs_df = (
        pd.read_csv(run_dir / "cohort_observation_analysis.csv")
        if (run_dir / "cohort_observation_analysis.csv").exists()
        else pd.DataFrame()
    )
    candidate_manifest = json.loads((run_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    selection_record = json.loads((run_dir / "selection_record.json").read_text(encoding="utf-8"))
    val_preds_df = pd.read_csv(val_preds_path, low_memory=False)
    backtest_preds_df = pd.read_csv(bt_preds_path, low_memory=False)
    feat_imp_df = pd.read_csv(run_dir / "feature_importance.csv")
    ex_post_df = pd.read_csv(run_dir / "ex_post_stage_summary.csv")

    # Recompute validation_folds_metrics.csv directly from frozen validation_predictions.csv
    primary_label = cfg.get("primary_label", cfg.get("label_contract", {}).get("primary_label", "calendar_day"))
    all_entries = list(candidate_manifest.get("model_candidates", [])) + list(candidate_manifest.get("baselines", []))
    recomputed_val_rows: list[dict[str, Any]] = []
    for sp_id, df_fold in val_preds_df.groupby("split_id", sort=False):
        y_va = df_fold["late"].astype(int).to_numpy()
        for entry in all_entries:
            cand_id = str(entry["candidate_id"])
            score_col = f"score_{cand_id}"
            if score_col not in df_fold.columns:
                continue
            p_va = df_fold[score_col].astype(float).to_numpy()
            recomputed_val_rows.append(
                compute_classification_metrics(
                    order_ids=df_fold["order_id"],
                    y_true=y_va,
                    y_prob=p_va,
                    model_name=str(entry["model_name"]),
                    calibration_method=str(entry["calibration_method"]),
                    split_id=str(sp_id),
                    label_version=primary_label,
                    purchase_timestamps=df_fold["order_purchase_timestamp"],
                    candidate_id=cand_id,
                    feature_version=str(entry.get("feature_version", "base_v2")),
                    hyperparams_json=str(entry.get("hyperparams_json", "{}")),
                )
            )
    val_metrics_df = pd.DataFrame(recomputed_val_rows)

    _compute_and_write_evaluation_from_predictions(
        out_dir=out_dir,
        figures_dir=out_dir / "figures",
        run_id=args.run_id,
        cfg=cfg,
        audit_summary=audit_summary,
        sp_df=sp_df,
        cohort_obs_df=cohort_obs_df,
        candidate_manifest=candidate_manifest,
        selection_record=selection_record,
        val_metrics_df=val_metrics_df,
        val_preds_df=val_preds_df,
        backtest_preds_df=backtest_preds_df,
        feat_imp_df=feat_imp_df,
        ex_post_df=ex_post_df,
    )
    print(f"Successfully evaluated run '{args.run_id}' into {out_dir}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / args.run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1
    report = verify_run_directory(
        run_dir=run_dir,
        root=PROJECT_ROOT,
        update_manifest=not args.no_update_manifest,
        check_docs_sync=args.check_docs,
        allow_smoke=getattr(args, "allow_smoke", False),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall_passed"] else 1


def cmd_score(args: argparse.Namespace) -> int:
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / args.run_id
    model_path = run_dir / "models" / "selected_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model bundle not found at {model_path}")

    out_path = assert_writable_path(args.output, root=PROJECT_ROOT)
    ev_path = resolve_project_path(args.history_events, root=PROJECT_ROOT)
    events_in = pd.read_csv(ev_path, low_memory=False)

    dec_in: pd.DataFrame | None = None
    if getattr(args, "decision_events", None):
        dec_path = resolve_project_path(args.decision_events, root=PROJECT_ROOT)
        dec_in = pd.read_csv(dec_path, low_memory=False)

    if getattr(args, "raw_data_dir", None):
        raw_dir = resolve_project_path(args.raw_data_dir, root=PROJECT_ROOT)
        raw_tables = load_raw_tables(raw_dir)
        order_ids_filter: list[str] | None = None
        if getattr(args, "input", None):
            in_df = pd.read_csv(resolve_project_path(args.input, root=PROJECT_ROOT), low_memory=False)
            if "order_id" in in_df.columns:
                order_ids_filter = in_df["order_id"].astype(str).tolist()
        orders_in = build_scoring_input_v2_from_raw(raw_tables, order_ids=order_ids_filter)
    else:
        if not getattr(args, "input", None):
            raise ValueError("Either `--input` (scoring_input_v2 CSV) or `--raw-data-dir` must be provided.")
        in_path = resolve_project_path(args.input, root=PROJECT_ROOT)
        orders_in = pd.read_csv(in_path, low_memory=False)

    verify_model_artifact_hash(run_dir, model_path)
    with model_path.open("rb") as f_mod:
        saved = pickle.load(f_mod)

    bundle: CalibratedModelBundle = saved["selected_bundle"]
    cal_method: str = saved["selected_calibration_method"]
    model_name: str = saved["selected_model_name"]
    fallback_prior: float = float(saved.get("fallback_prior", 0.068))

    out_df, meta = score_orders_batch(
        orders_in=orders_in,
        history_events=events_in,
        bundle=bundle,
        model_name=model_name,
        calibration_method=cal_method,
        fallback_prior=fallback_prior,
        as_of=getattr(args, "as_of", None),
        decision_events=dec_in,
        primary_budget=0.10,
        secondary_budget=0.20,
    )
    assert_zero_leakage_features(ALL_MODEL_FEATURES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    meta_out = {
        "scored_at": now_iso(),
        "run_id": args.run_id,
        **meta,
        "output_path": str(out_path),
    }
    atomic_write_json(out_path.with_suffix(".meta.json"), meta_out)
    print(json.dumps(meta_out, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="olist_delay",
        description="Olist Delivery Delay Warning & Risk Decision Pipeline",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_insp = sub.add_parser("inspect-config", help="Validate and inspect configuration, baselines, and expanded model grid")
    p_insp.add_argument("--config", required=True, help="Path to research or experiment config JSON")
    p_insp.set_defaults(func=cmd_inspect_config)

    p_audit = sub.add_parser("audit", help="Run baseline, schema, cohort, and feature availability audit")
    p_audit.add_argument("--config", required=True, help="Path to research or experiment config JSON")
    p_audit.add_argument("--output-dir", default=None, help="Optional directory to write audit artifacts")
    p_audit.set_defaults(func=cmd_audit)

    p_prep = sub.add_parser("prepare", help="Prepare canonical order table, causal features, and splits")
    p_prep.add_argument("--config", required=True, help="Path to config JSON")
    p_prep.add_argument("--run-id", required=True, help="Unique non-overwriting run identifier")
    p_prep.add_argument("--parent-run-id", default=None, help="Optional parent run ID")
    p_prep.add_argument("--rerun-reason", default=None, help="Optional reason for new run")
    p_prep.set_defaults(func=cmd_prepare)

    p_bt = sub.add_parser("backtest", help="Run rolling validation, model selection, and historical backtest")
    p_bt.add_argument("--config", required=True, help="Path to experiment config JSON")
    p_bt.add_argument("--run-id", required=True, help="Unique non-overwriting run identifier")
    p_bt.add_argument("--parent-run-id", default=None, help="Optional parent run ID")
    p_bt.add_argument("--rerun-reason", default=None, help="Optional reason for new run")
    p_bt.set_defaults(func=cmd_backtest)

    p_eval = sub.add_parser("evaluate", help="Re-evaluate metrics, policies, and figures from frozen predictions into --output-dir")
    p_eval.add_argument("--run-id", required=True, help="Existing run identifier")
    p_eval.add_argument("--output-dir", default=None, help="Required output directory outside artifacts/runs/<run-id>/")
    p_eval.set_defaults(func=cmd_evaluate)

    p_ver = sub.add_parser("verify", help="Verify run artifacts, hashes, leakage guards, and metric consistency")
    p_ver.add_argument("--run-id", required=True, help="Run identifier to verify")
    p_ver.add_argument("--no-update-manifest", action="store_true", help="Do not update manifest.json")
    p_ver.add_argument("--check-docs", action="store_true", help="Also verify docs narrative sync")
    p_ver.add_argument("--allow-smoke", action="store_true", help="Allow verifying a smoke-only run")
    p_ver.set_defaults(func=cmd_verify)

    p_score = sub.add_parser("score", help="Score a batch of purchase-time orders using a frozen run model")
    p_score.add_argument("--run-id", required=True, help="Run identifier with trained model")
    p_score.add_argument("--input", default=None, help="Path to scoring_input_v2 orders CSV")
    p_score.add_argument("--raw-data-dir", default=None, help="Optional raw Olist dataset directory for raw mode")
    p_score.add_argument("--history-events", required=True, help="Path to historical outcomes events CSV")
    p_score.add_argument("--decision-events", default=None, help="Optional path to historical decision events CSV")
    p_score.add_argument("--as-of", default=None, help="Optional cutoff timestamp (ISO format)")
    p_score.add_argument("--output", required=True, help="Path to output predictions CSV")
    p_score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
