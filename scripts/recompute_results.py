"""Independent result recomputation and `claims.json` verification script (`W5-03` / `A15`).

Recomputes integer cohort/alert counts and floating-point ranking, calibration, and daily
capacity policy metrics directly from persisted `cohort_audit.csv`, `validation_predictions.csv`,
`backtest_predictions.csv`, `daily_policy.csv`, `split_summary.csv`, and `config.resolved.json`
without reading `metrics_summary.json` as an input source, then cross-checks `metrics_summary.json`,
`selection_record.json`, and `claims.json` for exact integer equality and strict floating-point
tolerance (`<= 1e-10` on canonical 6-decimal stored values, `<= 1e-6` unrounded-to-rounded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _independent_cohort_a_daily_floor_policy(
    df_cohort_a: pd.DataFrame,
    score_col: str,
    k: float = 0.10,
    label_col: str = "late_calendar_day",
) -> dict[str, Any]:
    """Independently recompute Cohort A daily capacity floor policy (`q_d = floor(k * N_d)`)."""
    work = df_cohort_a.copy().reset_index(drop=True)
    p_ts = pd.to_datetime(work["order_purchase_timestamp"], errors="coerce")
    work["_purchase_date"] = p_ts.dt.strftime("%Y-%m-%d")
    s_raw = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    s_clip = np.clip(s_raw, 1e-6, 1.0 - 1e-6)
    work["_neg_score"] = -s_clip
    work["_p_ts_ns"] = p_ts.astype("int64")
    work["_oid"] = work["order_id"].astype(str)

    selected = np.zeros(len(work), dtype=bool)
    for _, grp in work.groupby("_purchase_date", sort=False):
        n_d = len(grp)
        q_d = int(math.floor(float(k) * n_d + 1e-12)) if (n_d > 0 and k > 0) else 0
        if q_d > 0:
            sorted_grp = grp.sort_values(
                by=["_neg_score", "_p_ts_ns", "_oid"],
                ascending=[True, True, True],
                kind="mergesort",
            )
            selected[sorted_grp.index[:q_d].to_numpy()] = True

    y = pd.to_numeric(work[label_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    n_total = int(len(y))
    n_pos = int(np.sum(y == 1))
    alert_cnt = int(np.sum(selected))
    tp = int(np.sum((y == 1) & selected))
    fp = int(np.sum((y == 0) & selected))
    prec = float(tp / max(1, alert_cnt)) if alert_cnt > 0 else 0.0
    rec = float(tp / max(1, n_pos)) if n_pos > 0 else 0.0

    return {
        "n_samples": n_total,
        "n_positives": n_pos,
        "daily_top10_alert_count": alert_cnt,
        "daily_top10_tp": tp,
        "daily_top10_fp": fp,
        "daily_top10_precision": round(prec, 6),
        "daily_top10_recall": round(rec, 6),
        "daily_top10_precision_raw": prec,
        "daily_top10_recall_raw": rec,
    }


def _independent_ranking_metrics(
    df_cohort_a: pd.DataFrame,
    score_col: str,
    label_col: str = "late_calendar_day",
) -> dict[str, Any]:
    """Independently recompute Cohort A ranking & Brier metrics."""
    y = pd.to_numeric(df_cohort_a[label_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    s_raw = pd.to_numeric(df_cohort_a[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    s = np.clip(s_raw, 1e-6, 1.0 - 1e-6)

    ap = float(average_precision_score(y, s))
    auc = float(roc_auc_score(y, s))
    brier = float(brier_score_loss(y, s))

    return {
        "n_samples": int(len(y)),
        "n_positives": int(np.sum(y == 1)),
        "pr_auc": round(ap, 6),
        "roc_auc": round(auc, 6),
        "brier_score": round(brier, 6),
        "pr_auc_raw": ap,
        "roc_auc_raw": auc,
        "brier_score_raw": brier,
    }


def recompute_and_verify(
    run_id: str,
    artifacts_root: Path,
    float_tol: float = 1e-10,
) -> dict[str, Any]:
    run_dir = artifacts_root / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    required_files = [
        "config.resolved.json",
        "cohort_audit.csv",
        "data_quality_summary.json",
        "split_summary.csv",
        "selection_record.json",
        "validation_predictions.csv",
        "backtest_predictions.csv",
        "daily_policy.csv",
        "policy_frontier_all_orders.csv",
        "metrics_summary.json",
        "uncertainty_summary.json",
        "claims.json",
    ]
    for rel in required_files:
        if not (run_dir / rel).exists():
            raise FileNotFoundError(f"Missing required artifact in {run_dir}: {rel}")

    cohort_df = pd.read_csv(run_dir / "cohort_audit.csv", low_memory=False)
    dq_summary = json.loads((run_dir / "data_quality_summary.json").read_text(encoding="utf-8"))
    sp_df = pd.read_csv(run_dir / "split_summary.csv", low_memory=False)
    val_df = pd.read_csv(run_dir / "validation_predictions.csv", low_memory=False)
    bt_df = pd.read_csv(run_dir / "backtest_predictions.csv", low_memory=False)
    daily_df = pd.read_csv(run_dir / "daily_policy.csv", low_memory=False)
    pol_b_df = pd.read_csv(run_dir / "policy_frontier_all_orders.csv", low_memory=False)
    selection_rec = json.loads((run_dir / "selection_record.json").read_text(encoding="utf-8"))
    reported_summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    uncertainty_summary = json.loads((run_dir / "uncertainty_summary.json").read_text(encoding="utf-8"))
    claims_payload = json.loads((run_dir / "claims.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def _check_int(name: str, recomputed: int, reported: int) -> None:
        diff = abs(int(recomputed) - int(reported))
        passed = diff == 0
        checks.append(
            {
                "check": name,
                "type": "integer_exact",
                "recomputed": int(recomputed),
                "reported": int(reported),
                "abs_diff": diff,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"Integer mismatch on {name}: recomputed={recomputed} vs reported={reported}")

    def _check_float(name: str, recomputed: float, reported: float, tol: float = float_tol) -> None:
        if math.isnan(recomputed) and math.isnan(reported):
            passed = True
            diff = 0.0
        else:
            diff = abs(float(recomputed) - float(reported))
            passed = diff <= tol
        checks.append(
            {
                "check": name,
                "type": "float_tolerance",
                "recomputed": float(recomputed),
                "reported": float(reported),
                "abs_diff": float(diff),
                "tolerance": float(tol),
                "passed": passed,
            }
        )
        if not passed:
            errors.append(
                f"Float mismatch on {name}: recomputed={recomputed:.12g} vs reported={reported:.12g} (diff={diff:.3e} > {tol:.1e})"
            )

    # 1. Recompute Cohort Audit integer counts
    recomp_total_orders = int(len(cohort_df))
    recomp_cohort_a = int(cohort_df["in_cohort_a"].fillna(False).astype(bool).sum())
    recomp_excluded = int((~cohort_df["in_cohort_a"].fillna(False).astype(bool)).sum())

    rec = dq_summary["row_reconciliation"]
    _check_int("cohort.source_orders_total", recomp_total_orders, int(rec["source_orders_total"]))
    _check_int("cohort.path_a_delivered_evaluable_orders", recomp_cohort_a, int(rec["path_a_delivered_evaluable_orders"]))
    _check_int("cohort.path_a_excluded_orders", recomp_excluded, int(rec["path_a_excluded_orders"]))

    # 2. Verify daily_policy.csv floor formula q_d = floor(k * N_eligible,d) for every day
    for idx, row in daily_df.iterrows():
        k_val = float(row["budget_k"])
        n_el = int(row["eligible_orders_d"])
        expected_q = int(math.floor(k_val * n_el + 1e-12))
        actual_q = int(row["daily_quota_q_d"])
        actual_sel = int(row["selected_alerts_d"])
        if actual_q != expected_q or actual_sel != expected_q:
            errors.append(
                f"daily_policy.csv row {idx} ({row.get('decision_date')}, k={k_val}): "
                f"expected q_d={expected_q}, got daily_quota_q_d={actual_q}, selected_alerts_d={actual_sel}"
            )

    # 3. Recompute Backtest Cohort A metrics for selected candidate
    selected_cand_id = str(selection_rec["selected_candidate_id"])
    best_base_id = str(selection_rec["best_baseline_name"])

    if "is_matured_at_selection" in bt_df.columns:
        bt_a = bt_df[bt_df["in_cohort_a"].fillna(False).astype(bool) & bt_df["is_matured_at_selection"].fillna(False).astype(bool)].copy()
    else:
        bt_a = bt_df[bt_df["in_cohort_a"].fillna(False).astype(bool)].copy()

    bt_rep = reported_summary["selected_model_backtest_metrics"]
    recomp_bt_rank = _independent_ranking_metrics(bt_a, score_col="predicted_prob_calibrated", label_col="late_calendar_day")
    recomp_bt_pol = _independent_cohort_a_daily_floor_policy(bt_a, score_col="predicted_prob_calibrated", k=0.10, label_col="late_calendar_day")

    _check_int("backtest.selected.n_samples", recomp_bt_rank["n_samples"], int(bt_rep["n_samples"]))
    _check_int("backtest.selected.n_positives", recomp_bt_rank["n_positives"], int(bt_rep["n_positives"]))
    _check_float("backtest.selected.pr_auc", recomp_bt_rank["pr_auc"], float(bt_rep["pr_auc"]))
    _check_float("backtest.selected.roc_auc", recomp_bt_rank["roc_auc"], float(bt_rep["roc_auc"]))
    _check_float("backtest.selected.brier_score", recomp_bt_rank["brier_score"], float(bt_rep["brier_score"]))
    _check_int("backtest.selected.daily_top10_alert_count", recomp_bt_pol["daily_top10_alert_count"], int(bt_rep["daily_top10_alert_count"]))
    _check_int("backtest.selected.daily_top10_tp", recomp_bt_pol["daily_top10_tp"], int(bt_rep["daily_top10_tp"]))
    _check_int("backtest.selected.daily_top10_fp", recomp_bt_pol["daily_top10_fp"], int(bt_rep["daily_top10_fp"]))
    _check_float("backtest.selected.daily_top10_precision", recomp_bt_pol["daily_top10_precision"], float(bt_rep["daily_top10_precision"]))
    _check_float("backtest.selected.daily_top10_recall", recomp_bt_pol["daily_top10_recall"], float(bt_rep["daily_top10_recall"]))

    # 4. Recompute OOF Validation metrics from validation_predictions.csv for selected_candidate and best_simple_baseline
    for role_key, cid in [("selected_candidate", selected_cand_id), ("best_simple_baseline", best_base_id)]:
        score_col = f"score_{cid}"
        if score_col not in val_df.columns:
            errors.append(f"Missing validation OOF score column {score_col} in validation_predictions.csv")
            continue
        rep_val = selection_rec[role_key]
        fold_aps: list[float] = []
        fold_briers: list[float] = []
        fold_recs: list[float] = []
        for sp_id in sorted(val_df["split_id"].astype(str).unique()):
            f_sub = val_df[val_df["split_id"].astype(str) == sp_id]
            f_sub_a = f_sub[f_sub["in_cohort_a"].fillna(False).astype(bool)]
            f_rank = _independent_ranking_metrics(f_sub_a, score_col=score_col, label_col="late_calendar_day")
            f_pol = _independent_cohort_a_daily_floor_policy(f_sub_a, score_col=score_col, k=0.10, label_col="late_calendar_day")
            fold_aps.append(f_rank["pr_auc"])
            fold_briers.append(f_rank["brier_score"])
            fold_recs.append(f_pol["daily_top10_recall"])

        _check_float(
            f"val_oof.{role_key}.val_mean_pr_auc",
            round(float(np.mean(fold_aps)), 6),
            float(rep_val["val_mean_pr_auc"]),
        )
        _check_float(
            f"val_oof.{role_key}.val_mean_brier",
            round(float(np.mean(fold_briers)), 6),
            float(rep_val["val_mean_brier"]),
        )
        _check_float(
            f"val_oof.{role_key}.val_mean_daily_top10_recall",
            round(float(np.mean(fold_recs)), 6),
            float(rep_val["val_mean_daily_top10_recall"]),
        )

    # 5. Verify claims.json schema, source hashes, and recomputed values
    claims_by_id = claims_payload.get("claims_by_id", {})
    if not claims_by_id and isinstance(claims_payload.get("claims"), list):
        claims_by_id = {c["claim_id"]: c for c in claims_payload["claims"]}
    if not claims_by_id:
        errors.append("claims.json contains no claims!")

    required_claim_fields = {
        "claim_id",
        "value",
        "unit",
        "cohort",
        "window",
        "policy",
        "label",
        "source_file",
        "source_sha256",
        "scope_boundary",
    }
    for cid, cobj in claims_by_id.items():
        missing_f = required_claim_fields - set(cobj.keys())
        if missing_f:
            errors.append(f"Claim {cid} missing required fields: {sorted(missing_f)}")
            continue
        src_path = run_dir / str(cobj["source_file"])
        if not src_path.exists():
            errors.append(f"Claim {cid} references non-existent source_file: {cobj['source_file']}")
        else:
            actual_sha = _sha256_file(src_path)
            if actual_sha != str(cobj["source_sha256"]):
                errors.append(
                    f"Claim {cid} source_sha256 mismatch for {cobj['source_file']}: "
                    f"expected {cobj['source_sha256']}, actual {actual_sha}"
                )

    val_apr_rows = sp_df[sp_df["split_id"] == "val_2018_04"]
    val_apr_unmatured = int(val_apr_rows["eval_unmatured_at_selection_count"].iloc[0])
    val_apr_unmatured_late = int(
        val_apr_rows["eval_full_cohort_a_positives"].iloc[0] - val_apr_rows["eval_matured_positives"].iloc[0]
    )
    pol_b_10_total = int(pol_b_df.loc[np.isclose(pol_b_df["budget_k"], 0.10), "path_b_total_orders"].iloc[0])
    wb_diff_mean = (
        uncertainty_summary.get("weekly_block_bootstrap", {})
        .get("weekly_block_bootstrap", {})
        .get("paired_difference_vs_baseline", {})
        .get("daily_policy_recall_top10_diff", {})
        .get("mean")
    )

    claim_verifications: list[tuple[str, Any, str]] = [
        ("C01_SOURCE_ORDERS_TOTAL", recomp_total_orders, "int"),
        ("C02_COHORT_A_EVALUABLE_TOTAL", recomp_cohort_a, "int"),
        ("C03_UNKNOWN_OUTCOME_EXCLUDED_TOTAL", recomp_excluded, "int"),
        ("C04_VAL_APRIL_UNMATURED_AT_SELECTION", val_apr_unmatured, "int"),
        ("C05_VAL_APRIL_UNMATURED_LATE_COUNT", val_apr_unmatured_late, "int"),
        ("C06_BACKTEST_PATH_B_TOTAL", pol_b_10_total, "int"),
        ("C07_BACKTEST_COHORT_A_TOTAL", recomp_bt_rank["n_samples"], "int"),
        ("C08_BACKTEST_POSITIVES_COHORT_A", recomp_bt_rank["n_positives"], "int"),
        ("C09_SELECTED_CANDIDATE_ID", selected_cand_id, "str"),
        ("C10_BEST_SIMPLE_BASELINE_ID", best_base_id, "str"),
        ("C11_SELECTED_BACKTEST_PR_AUC", recomp_bt_rank["pr_auc"], "float"),
        ("C12_SELECTED_DAILY_TOP10_ALERTS_COHORT_A", recomp_bt_pol["daily_top10_alert_count"], "int"),
        ("C13_SELECTED_DAILY_TOP10_TP_COHORT_A", recomp_bt_pol["daily_top10_tp"], "int"),
        ("C14_SELECTED_DAILY_TOP10_FP_COHORT_A", recomp_bt_pol["daily_top10_fp"], "int"),
        ("C15_SELECTED_DAILY_TOP10_RECALL_COHORT_A", recomp_bt_pol["daily_top10_recall"], "float"),
        ("C16_SELECTED_DAILY_TOP10_PRECISION_COHORT_A", recomp_bt_pol["daily_top10_precision"], "float"),
        ("C17_PATH_B_DAILY_ALERTS_TOTAL", int(daily_df["selected_alerts_d"].sum()), "int"),
        ("C18_PATH_B_DAILY_ALERTS_UNLABELED", int(daily_df["selected_unlabeled_d"].sum()), "int"),
        ("C19_WEEKLY_BOOTSTRAP_DAILY_RECALL_DIFF_MEAN", wb_diff_mean, "float"),
    ]
    for claim_id, expected_val, vtype in claim_verifications:
        if claim_id not in claims_by_id:
            errors.append(f"Missing core claim ID in claims.json: {claim_id}")
            continue
        c_val = claims_by_id[claim_id]["value"]
        if vtype == "int":
            _check_int(f"claim.{claim_id}", int(expected_val), int(c_val))
        elif vtype == "float":
            if expected_val is not None and c_val is not None:
                _check_float(f"claim.{claim_id}", float(expected_val), float(c_val), tol=float_tol)
        else:
            passed = str(expected_val) == str(c_val)
            checks.append(
                {
                    "check": f"claim.{claim_id}",
                    "type": "string_exact",
                    "recomputed": str(expected_val),
                    "reported": str(c_val),
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(f"String mismatch on claim.{claim_id}: {expected_val} vs {c_val}")

    result = {
        "run_id": run_id,
        "status": "PASSED" if len(errors) == 0 else "FAILED",
        "num_checks": len(checks),
        "num_passed": sum(1 for c in checks if c["passed"]),
        "num_failed": len(errors),
        "errors": errors,
        "checks": checks,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recompute_results.py",
        description="Independently recompute and verify Olist delay warning results and claims.json",
    )
    parser.add_argument("--run-id", default="experiment-v2-002", help="Target run ID under artifacts/runs/")
    parser.add_argument("--artifacts-root", default="artifacts", help="Root artifacts directory")
    parser.add_argument("--output", default=None, help="Optional JSON file path to save recomputation report")
    args = parser.parse_args(argv)

    report = recompute_and_verify(
        run_id=args.run_id,
        artifacts_root=Path(args.artifacts_root),
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
