"""Publication-grade figure generation and markdown summary builders for experiment runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def generate_all_run_figures(
    figures_dir: Path,
    audit_summary: dict[str, Any],
    split_summary_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame,
    calibration_curve_df: pd.DataFrame,
    policy_cohort_a_df: pd.DataFrame,
    policy_path_b_df: pd.DataFrame,
    cost_sensitivity_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    monthly_drift_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    ex_post_stage_df: pd.DataFrame,
    backtest_predictions_df: pd.DataFrame,
    selected_model_name: str,
) -> list[str]:
    """Generate the 8 standardized diagnostic and operational figures into `figures_dir`."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # 1. Cohort & Dual-Label Reconciliation
    f1 = figures_dir / "01_cohort_and_label_reconciliation.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    rc = audit_summary["row_reconciliation"]
    lc = audit_summary["label_reconciliation"]

    cohort_labels = [
        "Source Orders\n(Path B Total)",
        "Orders w/ Items\n(Path B Scorable)",
        "Delivered Valid\n(Path A Cohort)",
        "Excluded / Unlabeled\n(Non-Delivered/Missing)",
    ]
    cohort_vals = [
        rc["source_orders_total"],
        rc["path_b_orders_with_items"],
        rc["path_a_delivered_evaluable_orders"],
        rc["path_a_excluded_orders"],
    ]
    colors_1 = ["#2b5c8f", "#388087", "#2ca02c", "#d95f02"]
    bars1 = axes[0].bar(cohort_labels, cohort_vals, color=colors_1, width=0.58)
    axes[0].set_title("A. Dual-Population Cohort Reconciliation (N = 99,441)")
    axes[0].set_ylabel("Number of Orders")
    for b in bars1:
        h = int(b.get_height())
        axes[0].text(b.get_x() + b.get_width() / 2, h + 1200, f"{h:,}", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylim(0, max(cohort_vals) * 1.16)

    lbl_cats = [
        "Full Cohort A:\nTimestamp Late",
        "Full Cohort A:\nCalendar-Day Late",
        "Full Cohort A:\nSame-Day >00:00",
        "Backtest:\nTimestamp Late",
        "Backtest:\nCalendar-Day Late",
        "Backtest:\nSame-Day >00:00",
    ]
    lbl_vals = [
        lc["full_cohort_a"]["timestamp_late_positives"],
        lc["full_cohort_a"]["calendar_day_late_positives"],
        lc["full_cohort_a"]["same_day_after_midnight_count"],
        lc["backtest_window"]["timestamp_late_positives"],
        lc["backtest_window"]["calendar_day_late_positives"],
        lc["backtest_window"]["same_day_after_midnight_count"],
    ]
    colors_2 = ["#7570b3", "#1b9e77", "#e7298a", "#7570b3", "#1b9e77", "#e7298a"]
    bars2 = axes[1].bar(lbl_cats, lbl_vals, color=colors_2, width=0.58)
    axes[1].set_title("B. Timestamp vs Calendar-Day Label Reconciliation")
    axes[1].set_ylabel("Positive Order Count")
    axes[1].tick_params(axis="x", rotation=20)
    for b in bars2:
        h = int(b.get_height())
        axes[1].text(b.get_x() + b.get_width() / 2, h + 120, f"{h:,}", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_ylim(0, max(lbl_vals) * 1.18)
    fig.tight_layout()
    fig.savefig(f1)
    plt.close(fig)
    generated.append(f1.name)

    # 2. Rolling Splits & Monthly Drift
    f2 = figures_dir / "02_rolling_splits_and_drift.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x_pos = np.arange(len(split_summary_df))
    axes[0].bar(
        x_pos - 0.18,
        split_summary_df["train_matured_count"],
        width=0.35,
        label="Matured Train (label_available_at <= train_end)",
        color="#2b5c8f",
    )
    axes[0].bar(
        x_pos + 0.18,
        split_summary_df["train_dropped_in_transit_count"],
        width=0.35,
        label="In-Transit Dropped at Cutoff",
        color="#e66101",
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(split_summary_df["split_id"], rotation=15)
    axes[0].set_title("A. Expanding-Window Train Maturity Filtering")
    axes[0].set_ylabel("Order Count")
    axes[0].legend(loc="upper left")

    if not monthly_drift_df.empty:
        m_sub = monthly_drift_df.sort_values("year_month")
        ax2 = axes[1]
        ax2.plot(
            m_sub["year_month"],
            m_sub["observed_late_rate"] * 100.0,
            marker="o",
            linewidth=2.0,
            color="#d95f02",
            label="Observed Late Rate (%)",
        )
        ax2.plot(
            m_sub["year_month"],
            m_sub["mean_predicted_prob"] * 100.0,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color="#1b9e77",
            label="Mean Calibrated Pred Prob (%)",
        )
        ax2.set_title("B. Monthly Late-Rate Regime Shift & Tracking (2018)")
        ax2.set_ylabel("Rate (%)")
        ax2.tick_params(axis="x", rotation=25)
        ax2_r = ax2.twinx()
        ax2_r.plot(
            m_sub["year_month"],
            m_sub["pr_auc"],
            marker="^",
            linestyle=":",
            color="#2b5c8f",
            label="Monthly PR-AUC (Right Axis)",
        )
        ax2_r.set_ylabel("PR-AUC")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f2)
    plt.close(fig)
    generated.append(f2.name)

    # 3. Model Comparison across Rolling Validation and Historical Backtest
    f3 = figures_dir / "03_model_comparison_validation_and_backtest.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    mc_cal = model_comparison_df[
        (model_comparison_df["label_version"] == "calendar_day")
        & (model_comparison_df["is_model_default_calibration"])
    ].copy()
    if "candidate_id" in mc_cal.columns and len(mc_cal) > 12:
        # Keep all baselines + selected candidate + top representative per (model_name, feature_version, calibration_method)
        is_base = mc_cal["model_family"] == "simple_baseline"
        is_sel = mc_cal["is_selected_production_model"].astype(bool)
        ml_part = (
            mc_cal[~is_base]
            .sort_values("val_mean_pr_auc", ascending=False)
            .groupby(["model_name", "feature_version"], as_index=False)
            .first()
        )
        mc_cal = pd.concat([mc_cal[is_base | is_sel], ml_part], ignore_index=True).drop_duplicates(
            subset=["candidate_id"]
        )
    mc_cal.sort_values("val_mean_pr_auc", ascending=True, inplace=True)
    y_labels = (
        mc_cal["candidate_id"].astype(str) + " (" + mc_cal["model_name"].astype(str) + ")"
        if "candidate_id" in mc_cal.columns
        else mc_cal["model_name"].astype(str)
    )

    y_idx = np.arange(len(mc_cal))
    axes[0].barh(
        y_idx - 0.18,
        mc_cal["val_mean_pr_auc"],
        height=0.35,
        label="Rolling Val Mean PR-AUC (4 Folds)",
        color="#388087",
    )
    axes[0].barh(
        y_idx + 0.18,
        mc_cal["backtest_pr_auc"],
        height=0.35,
        label="Historical Backtest PR-AUC",
        color="#2b5c8f",
    )
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels(y_labels, fontsize=8)
    axes[0].set_xlabel("PR-AUC (Average Precision)")
    axes[0].set_title("A. Primary Label (Calendar-Day) PR-AUC Comparison")
    axes[0].legend(loc="lower right")

    axes[1].barh(
        y_idx - 0.18,
        mc_cal["val_mean_roc_auc"],
        height=0.35,
        label="Rolling Val Mean ROC-AUC",
        color="#7570b3",
    )
    axes[1].barh(
        y_idx + 0.18,
        mc_cal["backtest_roc_auc"],
        height=0.35,
        label="Historical Backtest ROC-AUC",
        color="#1b9e77",
    )
    axes[1].set_yticks(y_idx)
    axes[1].set_yticklabels(y_labels, fontsize=8)
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_xlim(0.45, max(0.82, float(mc_cal["backtest_roc_auc"].max()) + 0.05))
    axes[1].set_title("B. Primary Label (Calendar-Day) ROC-AUC Comparison")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f3)
    plt.close(fig)
    generated.append(f3.name)

    # 4. PR and ROC Curves on Historical Backtest
    f4 = figures_dir / "04_precision_recall_and_roc_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bt_coh_a = backtest_predictions_df[backtest_predictions_df["in_cohort_a"]].copy()
    if "is_matured_at_selection" in bt_coh_a.columns:
        bt_coh_a = bt_coh_a[bt_coh_a["is_matured_at_selection"]].copy()
    y_bt = bt_coh_a["late"].astype(int).to_numpy()
    base_bt = float(np.mean(y_bt))

    curve_specs = [
        ("predicted_prob_calibrated", f"Selected ({selected_model_name})", "#1b9e77", "-"),
        ("score_logistic_regression", "Best Logistic Regression", "#2b5c8f", "--"),
        ("score_decision_tree", "Best Shallow Decision Tree", "#7570b3", "-."),
        ("score_rule_region_lead", "Rule: route history x promise tightness", "#d95f02", ":"),
        ("score_rule_short_promised_lead", "Rule: Short Promised Lead", "#e7298a", ":"),
        ("score_random_forest", "Random Forest", "#7570b3", "-."),
        ("score_route_smoothed_lookup", "Route Smoothed Baseline", "#d95f02", ":"),
    ]
    plotted_cols: set[str] = set()
    for col, lbl, clr, ls in curve_specs:
        if col not in bt_coh_a.columns or col in plotted_cols:
            continue
        plotted_cols.add(col)
        s = bt_coh_a[col].astype(float).to_numpy()
        prec, rec, _ = precision_recall_curve(y_bt, s)
        fpr, tpr, _ = roc_curve(y_bt, s)
        axes[0].plot(rec, prec, label=lbl, color=clr, linestyle=ls, linewidth=1.9)
        axes[1].plot(fpr, tpr, label=lbl, color=clr, linestyle=ls, linewidth=1.9)

    axes[0].axhline(base_bt, color="gray", linestyle=":", label=f"No-Skill Base Rate ({base_bt:.2%})")
    axes[0].set_title(f"A. Historical Backtest Precision-Recall Curve (N={len(bt_coh_a):,})")
    axes[0].set_xlabel("Recall (Share of Delayed Orders Caught)")
    axes[0].set_ylabel("Precision (Delayed Share in Alerts)")
    axes[0].legend(loc="upper right")

    axes[1].plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Ranking (AUC=0.50)")
    axes[1].set_title(f"B. Historical Backtest ROC Curve (N={len(bt_coh_a):,})")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f4)
    plt.close(fig)
    generated.append(f4.name)

    # 5. Calibration Reliability Curves
    f5 = figures_dir / "05_calibration_reliability_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cal_sel = calibration_curve_df[
        (calibration_curve_df["split_id"] == "historical_backtest")
        & (calibration_curve_df["model_name"] == selected_model_name)
    ]
    max_val = 0.25
    for variant, clr, mkr in [
        ("uncalibrated", "#d95f02", "o"),
        ("platt", "#1b9e77", "s"),
        ("isotonic", "#2b5c8f", "^"),
    ]:
        sub_c = cal_sel[cal_sel["calibration_method"] == variant]
        if sub_c.empty:
            continue
        max_val = max(
            max_val,
            float(sub_c["mean_predicted_prob"].max()),
            float(sub_c["observed_late_rate"].max()),
        )
        axes[0].plot(
            sub_c["mean_predicted_prob"],
            sub_c["observed_late_rate"],
            marker=mkr,
            color=clr,
            linewidth=1.8,
            label=f"{variant.capitalize()}",
        )
    axes[0].plot([0, max_val], [0, max_val], color="black", linestyle=":", label="Perfect Calibration (y=x)")
    axes[0].set_title(f"A. Reliability Curve on Backtest ({selected_model_name})")
    axes[0].set_xlabel("Mean Predicted Probability (Decile Bin)")
    axes[0].set_ylabel("Observed Late Frequency")
    axes[0].legend(loc="upper left")

    mc_variants = model_comparison_df[
        (model_comparison_df["model_name"] == selected_model_name)
        & (model_comparison_df["label_version"] == "calendar_day")
    ].copy()
    if "is_selected_base_family" in mc_variants.columns and mc_variants["is_selected_base_family"].any():
        mc_variants = mc_variants[mc_variants["is_selected_base_family"]].copy()
    else:
        mc_variants = mc_variants.drop_duplicates(subset=["calibration_method"])
    if not mc_variants.empty:
        vx = np.arange(len(mc_variants))
        axes[1].bar(
            vx - 0.18,
            mc_variants["backtest_brier"],
            width=0.35,
            label="Backtest Brier Score",
            color="#2b5c8f",
        )
        axes[1].bar(
            vx + 0.18,
            mc_variants["backtest_ece_10"],
            width=0.35,
            label="Backtest 10-Bin ECE",
            color="#e7298a",
        )
        axes[1].set_xticks(vx)
        axes[1].set_xticklabels(mc_variants["calibration_method"])
        axes[1].set_title("B. Calibration Error Reduction Across Variants")
        axes[1].set_ylabel("Error (Lower is Better)")
        axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f5)
    plt.close(fig)
    generated.append(f5.name)

    # 6. Top-k% Policy Frontier (Path A vs Path B)
    f6 = figures_dir / "06_policy_frontier_topk_and_path_b.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    pol_a = policy_cohort_a_df[
        (policy_cohort_a_df["model_name"] == selected_model_name)
        & (policy_cohort_a_df["label_version"] == "calendar_day")
    ].copy()
    if "is_selected_production_model" in pol_a.columns and pol_a["is_selected_production_model"].any():
        pol_a = pol_a[pol_a["is_selected_production_model"]].copy()
    pol_a = pol_a.drop_duplicates(subset=["budget_k"]).sort_values("budget_k")

    axes[0].plot(
        pol_a["budget_k"] * 100.0,
        pol_a["recall"] * 100.0,
        marker="o",
        linewidth=2.0,
        color="#1b9e77",
        label="Recall (% of Delayed Caught)",
    )
    axes[0].plot(
        pol_a["budget_k"] * 100.0,
        pol_a["precision"] * 100.0,
        marker="s",
        linewidth=2.0,
        color="#2b5c8f",
        label="Precision (% Delayed in Alerts)",
    )
    axes[0].set_title("A. Path A Operational Alert Frontier (Cohort A, N=19,294)")
    axes[0].set_xlabel("Alert Capacity Budget k (% of Orders Screened)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend(loc="center right")

    pol_b = policy_path_b_df.sort_values("budget_k")
    bx = np.arange(len(pol_b))
    axes[1].bar(
        bx,
        pol_b["observed_tp_in_cohort_a"],
        label="Observed Delayed (Path A TP)",
        color="#d95f02",
    )
    axes[1].bar(
        bx,
        pol_b["alert_unlabeled_non_delivered_count"],
        bottom=pol_b["observed_tp_in_cohort_a"],
        label="Unlabeled Non-Delivered Exceptions (Shipped/Canceled/etc.)",
        color="#7570b3",
    )
    axes[1].set_xticks(bx)
    axes[1].set_xticklabels(pol_b["budget_pct_label"])
    axes[1].set_title("B. Path B All-Order Alert Exceptions Captured (N=19,661)")
    axes[1].set_ylabel("Orders in Alert Queue")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f6)
    plt.close(fig)
    generated.append(f6.name)

    # 7. Parametric Cost Sensitivity & Regional Subgroups
    f7 = figures_dir / "07_cost_sensitivity_and_subgroups.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for var_lbl, clr, mkr in [
        ("calibrated", "#1b9e77", "o"),
        ("uncalibrated", "#d95f02", "s"),
        ("random_top10", "#7570b3", "^"),
    ]:
        c_sub = cost_sensitivity_df[cost_sensitivity_df["probability_variant"] == var_lbl].sort_values(
            "cost_ratio_p_star"
        )
        if c_sub.empty:
            continue
        axes[0].plot(
            c_sub["cost_ratio_p_star"],
            c_sub["parametric_net_utility_per_1000_orders"],
            marker=mkr,
            linewidth=2.0,
            color=clr,
            label=f"Rule ({var_lbl})",
        )
    axes[0].axhline(0.0, color="gray", linestyle=":", label="No-Intervention Baseline (Utility = 0)")
    axes[0].set_title("A. Parametric Cost-Ratio Sensitivity Frontier (Simulated)")
    axes[0].set_xlabel("Cost Ratio p* = C_alert / (p_mitigate * L_late)")
    axes[0].set_ylabel("Net Parametric Utility per 1,000 Orders")
    axes[0].legend(loc="upper right")

    reg_sub = subgroup_df[subgroup_df["subgroup_dimension"] == "customer_region"].sort_values(
        "observed_late_rate", ascending=True
    )
    if not reg_sub.empty:
        rx = np.arange(len(reg_sub))
        axes[1].barh(
            rx - 0.18,
            reg_sub["observed_late_rate"] * 100.0,
            height=0.35,
            label="Observed Late Rate (%)",
            color="#d95f02",
        )
        axes[1].barh(
            rx + 0.18,
            reg_sub["within_group_precision_top10"] * 100.0,
            height=0.35,
            label="Top-10% Alert Precision (%)",
            color="#2b5c8f",
        )
        axes[1].set_yticks(rx)
        axes[1].set_yticklabels(reg_sub["subgroup_value"])
        axes[1].set_xlabel("Percentage (%)")
        axes[1].set_title("B. Regional Late Rate vs Top-10% Precision")
        axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f7)
    plt.close(fig)
    generated.append(f7.name)

    # 8. Out-of-Sample Permutation Importance vs Ex-Post Fulfillment Stage Breakdown
    f8 = figures_dir / "08_feature_importance_and_ex_post_stages.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    top_imp = feature_importance_df.head(12).iloc[::-1]
    axes[0].barh(
        top_imp["feature_name"],
        top_imp["importance_mean_pr_auc_drop"],
        xerr=top_imp["importance_std"],
        color="#2b5c8f",
        height=0.65,
    )
    axes[0].set_title("A. Purchase-Time Permutation Importance (Backtest PR-AUC Drop)")
    axes[0].set_xlabel("Mean Decrease in PR-AUC upon Permutation")

    if not ex_post_stage_df.empty:
        stages = ["Approval\n(Days)", "Seller->Carrier\n(Days)", "Carrier->Customer\n(Days)"]
        ontime = ex_post_stage_df[ex_post_stage_df["label_value"] == 0].iloc[0]
        delayed = ex_post_stage_df[ex_post_stage_df["label_value"] == 1].iloc[0]
        v_on = [
            float(ontime["approval_hours_mean"]) / 24.0,
            float(ontime["seller_to_carrier_days_mean"]),
            float(ontime["carrier_to_customer_days_mean"]),
        ]
        v_del = [
            float(delayed["approval_hours_mean"]) / 24.0,
            float(delayed["seller_to_carrier_days_mean"]),
            float(delayed["carrier_to_customer_days_mean"]),
        ]
        sx = np.arange(len(stages))
        axes[1].bar(sx - 0.18, v_on, width=0.35, label="On-Time Orders (late=0)", color="#1b9e77")
        axes[1].bar(sx + 0.18, v_del, width=0.35, label="Delayed Orders (late=1)", color="#d95f02")
        axes[1].set_xticks(sx)
        axes[1].set_xticklabels(stages)
        axes[1].set_ylabel("Mean Duration (Days)")
        axes[1].set_title("B. Ex-Post Stage Breakdown [Diagnostic Only - Not in Model]")
        axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f8)
    plt.close(fig)
    generated.append(f8.name)

    return generated
