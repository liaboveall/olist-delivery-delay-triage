"""Automated end-to-end verification of run artifacts, metric consistency, leakage guards, and baseline integrity."""
from __future__ import annotations

import json
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .artifacts import RAW_DATA_SHA256, atomic_write_json, now_iso, sha256_file, verify_baseline_protection
from .config import PROJECT_ROOT
from .features import FORBIDDEN_POST_PURCHASE_COLUMNS, MODEL_INPUT_COLUMNS
from .history import attach_event_clock_history_features


REQUIRED_RUN_FILES = [
    "manifest.json",
    "config.resolved.json",
    "environment.txt",
    "execution.log",
    "data_quality_summary.json",
    "cohort_audit.csv",
    "feature_availability_audit.csv",
    "split_summary.csv",
    "validation_folds_metrics.csv",
    "model_comparison.csv",
    "calibration_summary.csv",
    "policy_frontier_cohort_a.csv",
    "policy_frontier_all_orders.csv",
    "cost_sensitivity.csv",
    "subgroup_metrics.csv",
    "monthly_drift_report.csv",
    "feature_importance.csv",
    "ex_post_stage_summary.csv",
    "backtest_predictions.csv",
    "metrics_summary.json",
    "models/selected_model.pkl",
]

REQUIRED_V2_FILES = [
    "source_snapshot.json",
    "validation_predictions.csv",
    "candidate_manifest.json",
    "selection_record.json",
    "daily_policy.csv",
    "model_card.md",
]

REQUIRED_FIGURES = [
    "figures/01_cohort_and_label_reconciliation.png",
    "figures/02_rolling_splits_and_drift.png",
    "figures/03_model_comparison_validation_and_backtest.png",
    "figures/04_precision_recall_and_roc_curves.png",
    "figures/05_calibration_reliability_curves.png",
    "figures/06_policy_frontier_topk_and_path_b.png",
    "figures/07_cost_sensitivity_and_subgroups.png",
    "figures/08_feature_importance_and_ex_post_stages.png",
]

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _validate_png_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < 24:
        return False, f"too_small ({size} bytes)"
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != PNG_SIGNATURE:
        return False, "invalid_png_signature"
    if header[12:16] != b"IHDR":
        return False, "missing_IHDR_chunk"
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return False, f"invalid_dimensions ({width}x{height})"
    return True, f"{width}x{height}"


def _run_causal_perturbation_probe() -> dict[str, Any]:
    """Run a synthetic in-memory perturbation test proving in-transit outcomes never leak into history features."""
    base_events = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_unique_id": ["c1", "c2", "c3", "c1"],
            "primary_seller_id": ["s1", "s1", "s1", "s1"],
            "route_state_pair": ["SP->RJ", "SP->RJ", "SP->RJ", "SP->RJ"],
            "customer_state": ["RJ", "RJ", "RJ", "RJ"],
            "primary_category": ["catA", "catA", "catA", "catA"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-01-01 10:00:00",
                    "2018-01-05 10:00:00",  # Purchased before o3, but delivered AFTER o3!
                    "2018-01-10 10:00:00",  # Query order o3
                    "2018-01-15 10:00:00",  # Purchased after o3
                ]
            ),
            "label_available_at": pd.to_datetime(
                [
                    "2018-01-04 12:00:00",  # Completed before o3
                    "2018-01-18 12:00:00",  # In-transit at o3 purchase time!
                    "2018-01-20 12:00:00",
                    "2018-01-25 12:00:00",
                ]
            ),
            "late": [0, 0, 0, 0],
            "delay_calendar_days": [-2.0, -1.0, 0.0, 0.0],
        }
    )
    perturbed_events = base_events.copy()
    # Mutate outcomes and delays of o2 (in-transit at o3's purchase time) and o4 (future order)
    perturbed_events.loc[perturbed_events["order_id"].isin(["o2", "o4"]), "late"] = 1
    perturbed_events.loc[perturbed_events["order_id"].isin(["o2", "o4"]), "delay_calendar_days"] = 25.0

    f_base = attach_event_clock_history_features(base_events, history_events=base_events, label_col="late")
    f_pert = attach_event_clock_history_features(base_events, history_events=perturbed_events, label_col="late")

    # Compare all computed historical features for query order o3 (index 2)
    hist_cols = [c for c in f_base.columns if c not in base_events.columns]
    diffs = np.abs(
        f_base.loc[2, hist_cols].to_numpy(dtype=float) - f_pert.loc[2, hist_cols].to_numpy(dtype=float)
    )
    max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    return {
        "passed": max_diff == 0.0,
        "max_abs_feature_diff": max_diff,
        "checked_history_columns": len(hist_cols),
    }


V2_FROZEN_SPLIT_IDS = {"val_2018_01", "val_2018_02", "val_2018_03", "val_2018_04", "historical_backtest"}
V2_BACKTEST_TRAIN_AT = "2018-05-26 18:18:03"


def _run_split_contract(run_dir: Path) -> tuple[dict[str, Any], bool]:
    """Return the run's resolved split contract and whether it is the frozen v2 contract.

    The v2 contract (research-v2 splits with delivery-clock labels) keeps the exact-count regression guards.
    Runs without a resolved config are treated as v2 for backward compatibility.
    """
    cfg_path = run_dir / "config.resolved.json"
    if not cfg_path.exists():
        return {}, True
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    sc = cfg.get("split_contract", {})
    fold_ids = {f.get("fold_id") for f in sc.get("validation_folds", [])}
    bt = sc.get("backtest", {})
    known_rule = cfg.get("label_contract", {}).get("label_known_rule", "delivered_at")
    is_v2 = (
        fold_ids | {bt.get("fold_id", bt.get("split_id"))} == V2_FROZEN_SPLIT_IDS
        and str(bt.get("train_at")) == V2_BACKTEST_TRAIN_AT
        and known_rule == "delivered_at"
    )
    return sc, is_v2


def _backtest_split_id(split_contract: dict[str, Any]) -> str:
    bt = split_contract.get("backtest", {}) if split_contract else {}
    return str(bt.get("split_id", bt.get("fold_id", "historical_backtest")))


def _expected_split_ids(split_contract: dict[str, Any]) -> set[str]:
    if not split_contract:
        return set(V2_FROZEN_SPLIT_IDS)
    return {str(f["fold_id"]) for f in split_contract.get("validation_folds", [])} | {_backtest_split_id(split_contract)}


def verify_run_directory(
    run_dir: Path,
    root: Path = PROJECT_ROOT,
    update_manifest: bool = True,
    check_docs_sync: bool = False,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Perform comprehensive verification of a completed experiment run directory."""
    checks: list[dict[str, Any]] = []

    def add_check(check_id: str, name: str, passed: bool, details: Any) -> None:
        checks.append(
            {
                "check_id": check_id,
                "name": name,
                "passed": bool(passed),
                "details": details,
            }
        )

    # 1. Check raw dataset integrity (9 files)
    base_res = verify_baseline_protection(root=root)
    add_check("V01_BASELINE_INTACT", "All 9 raw dataset CSV files intact", base_res["all_intact"], base_res)

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    is_v2_schema = str(manifest.get("schema_version", "")).startswith("2.")
    expected_files = list(REQUIRED_RUN_FILES) + (REQUIRED_V2_FILES if is_v2_schema else []) + list(REQUIRED_FIGURES)

    # 2. Check required files exist, are non-empty, and PNGs have valid headers/dimensions
    missing_files: list[str] = []
    empty_files: list[str] = []
    invalid_pngs: dict[str, str] = {}

    for f in expected_files:
        fp = run_dir / Path(f)
        if not fp.exists():
            missing_files.append(f)
        elif fp.stat().st_size == 0:
            empty_files.append(f)

    for fig_rel in REQUIRED_FIGURES:
        fig_p = run_dir / Path(fig_rel)
        ok_png, png_msg = _validate_png_file(fig_p)
        if not ok_png:
            invalid_pngs[fig_rel] = png_msg

    v02_ok = len(missing_files) == 0 and len(empty_files) == 0 and len(invalid_pngs) == 0
    add_check(
        "V02_REQUIRED_FILES",
        "All required tables, JSONs, model bundle, and 8 valid PNG figures exist and are non-empty",
        v02_ok,
        {
            "missing_files": missing_files,
            "empty_files": empty_files,
            "invalid_pngs": invalid_pngs,
        },
    )

    # 3. Check manifest structure, cryptographic fields, & artifact SHA-256 hashes
    req_manifest_keys = {
        "schema_version",
        "run_id",
        "status",
        "config_hash",
        "code_version",
        "input_hashes",
        "environment",
        "label_version",
        "cohort_definitions",
        "feature_version",
        "split_protocol",
        "random_seed",
        "policy_contract",
        "artifacts",
        "resource_usage",
    }
    missing_m_keys = sorted(req_manifest_keys - set(manifest.keys()))
    cfg_hash = manifest.get("config_hash")
    valid_cfg_hash = isinstance(cfg_hash, str) and bool(HEX64_RE.match(cfg_hash))

    code_ver = manifest.get("code_version")
    valid_code_ver = (
        isinstance(code_ver, dict)
        and isinstance(code_ver.get("combined_sha256"), str)
        and bool(HEX64_RE.match(code_ver["combined_sha256"]))
        and isinstance(code_ver.get("files"), dict)
        and len(code_ver.get("files", {})) > 0
    )

    in_hashes = manifest.get("input_hashes")
    valid_input_hashes = (
        isinstance(in_hashes, dict)
        and all(in_hashes.get(k) == v for k, v in RAW_DATA_SHA256.items())
    )

    smoke_flag = bool(manifest.get("smoke_only", False))
    smoke_ok = True if allow_smoke else (not smoke_flag)

    hash_mismatches: list[dict[str, Any]] = []
    untracked_required_files: list[str] = []
    recorded_artifacts = manifest.get("artifacts", {})
    if not isinstance(recorded_artifacts, dict) or len(recorded_artifacts) == 0:
        untracked_required_files = [f for f in expected_files if f != "manifest.json"]
    else:
        for req_f in expected_files:
            if req_f == "manifest.json":
                continue
            if req_f not in recorded_artifacts:
                untracked_required_files.append(req_f)
        for rel_path, meta in recorded_artifacts.items():
            if rel_path in ("verification_report.json", "verification_report.md"):
                continue
            p = run_dir / Path(rel_path)
            if not p.exists():
                hash_mismatches.append({"file": rel_path, "error": "missing"})
                continue
            actual_sha = sha256_file(p)
            exp_sha = meta.get("sha256") if isinstance(meta, dict) else None
            if not isinstance(exp_sha, str) or not HEX64_RE.match(exp_sha) or actual_sha != exp_sha:
                hash_mismatches.append(
                    {"file": rel_path, "expected": exp_sha, "actual": actual_sha}
                )

    v03_ok = (
        len(missing_m_keys) == 0
        and valid_cfg_hash
        and valid_code_ver
        and valid_input_hashes
        and smoke_ok
        and len(untracked_required_files) == 0
        and len(hash_mismatches) == 0
        and manifest.get("status") == "COMPLETED"
    )
    add_check(
        "V03_MANIFEST_AND_HASHES",
        "Manifest schema complete, cryptographic hashes valid, and all required artifact SHA-256 digests match",
        v03_ok,
        {
            "status": manifest.get("status"),
            "smoke_only": smoke_flag,
            "smoke_allowed": allow_smoke,
            "valid_config_hash": valid_cfg_hash,
            "valid_code_version": valid_code_ver,
            "valid_input_hashes": valid_input_hashes,
            "missing_manifest_keys": missing_m_keys,
            "untracked_required_files": untracked_required_files,
            "hash_mismatches": hash_mismatches,
        },
    )

    # 4. Cohort & Label reconciliation check
    try:
        dq = json.loads((run_dir / "data_quality_summary.json").read_text(encoding="utf-8"))
        rc = dq["row_reconciliation"]
        lc = dq["label_reconciliation"]
        fc_a = lc["full_cohort_a"]
        bt_w = lc["backtest_window"]
        ca_df = pd.read_csv(run_dir / "cohort_audit.csv", usecols=["order_id", "in_cohort_a"])
        cohort_ok = (
            rc["source_orders_total"] == 99441
            and rc["path_a_delivered_evaluable_orders"] == 96470
            and rc["path_a_excluded_orders"] == 2971
            and rc["unexplained_row_delta"] == 0
            and fc_a.get("late_timestamp_count") == 7826
            and fc_a.get("late_calendar_day_count") == 6534
            and fc_a.get("same_day_after_midnight_count") == 1292
            and fc_a.get("exact_difference_matches_same_day") is True
            and bt_w.get("late_timestamp_count") == 1021
            and bt_w.get("late_calendar_day_count") == 674
            and bt_w.get("same_day_after_midnight_count") == 347
            and bt_w.get("exact_difference_matches_same_day") is True
            and len(ca_df) == 99441
            and int(ca_df["in_cohort_a"].astype(bool).sum()) == 96470
        )
        v04_details = {
            "source_orders_total": rc.get("source_orders_total"),
            "cohort_a": rc.get("path_a_delivered_evaluable_orders"),
            "excluded": rc.get("path_a_excluded_orders"),
            "cohort_audit_rows": len(ca_df),
            "full_same_day_diff": fc_a.get("same_day_after_midnight_count"),
            "backtest_same_day_diff": bt_w.get("same_day_after_midnight_count"),
        }
    except Exception as exc:
        cohort_ok = False
        v04_details = {"error": f"{type(exc).__name__}: {exc}"}

    add_check(
        "V04_COHORT_AND_LABEL_RECONCILIATION",
        "99,441 source orders reconcile to 96,470 Cohort A + 2,971 excluded and dual labels reconcile",
        cohort_ok,
        v04_details,
    )

    # 5. Feature availability & causal perturbation check
    try:
        fa_df = pd.read_csv(run_dir / "feature_availability_audit.csv")
        leaked_cols = set(fa_df["feature_name"]).intersection(FORBIDDEN_POST_PURCHASE_COLUMNS)
        fa_ok = (
            len(fa_df) >= 10
            and len(leaked_cols) == 0
            and bool(fa_df["audit_passed"].all())
            and set(fa_df["feature_name"]).issubset(set(MODEL_INPUT_COLUMNS))
        )
        probe_res = _run_causal_perturbation_probe()
        v05_ok = fa_ok and probe_res["passed"]
        v05_details = {
            "leaked_columns": sorted(leaked_cols),
            "all_audit_passed": bool(fa_df["audit_passed"].all()),
            "causal_perturbation_probe": probe_res,
        }
    except Exception as exc:
        v05_ok = False
        v05_details = {"error": f"{type(exc).__name__}: {exc}"}

    add_check(
        "V05_ZERO_LEAKAGE_AND_EVENT_CLOCK",
        "Zero forbidden post-purchase features and causal event-clock perturbation diff == 0.0",
        v05_ok,
        v05_details,
    )

    # 6. Split chronology & label maturity check
    try:
        sp_df = pd.read_csv(run_dir / "split_summary.csv")
        run_split_contract, is_v2_contract = _run_split_contract(run_dir)
        expected_splits = _expected_split_ids(run_split_contract)
        bt_id = _backtest_split_id(run_split_contract)
        bt_rows = sp_df[sp_df["split_id"] == bt_id]
        if is_v2_contract:
            # Regression guard for the frozen v2 contract (research-v2 splits, delivery-clock labels).
            bt_counts_ok = (
                not bt_rows.empty
                and int(bt_rows.iloc[0]["train_matured_count"]) == 75128
                and int(bt_rows.iloc[0]["eval_path_b_total"]) == 19661
                and int(bt_rows.iloc[0]["eval_cohort_a_total"]) == 19294
            )
        else:
            # Other contracts: the split summary must agree with the persisted backtest predictions.
            bt_pred = pd.read_csv(run_dir / "backtest_predictions.csv", usecols=["order_id", "in_cohort_a"])
            bt_counts_ok = (
                not bt_rows.empty
                and int(bt_rows.iloc[0]["eval_path_b_total"]) == len(bt_pred)
                and int(bt_rows.iloc[0]["eval_cohort_a_total"]) == int(bt_pred["in_cohort_a"].astype(bool).sum())
                and int(bt_rows.iloc[0]["train_matured_count"]) > 0
            )
        splits_ok = (
            expected_splits.issubset(set(sp_df["split_id"]))
            and bool(sp_df["chronological_strict_pass"].all())
            and bool(sp_df["maturity_strict_pass"].all())
            and bool((sp_df["train_dropped_in_transit_count"] > 0).all())
            and bt_counts_ok
        )
        v06_details = sp_df[
            [
                "split_id",
                "train_matured_count",
                "train_dropped_in_transit_count",
                "chronological_strict_pass",
                "maturity_strict_pass",
            ]
        ].to_dict(orient="records")
    except Exception as exc:
        splits_ok = False
        v06_details = {"error": f"{type(exc).__name__}: {exc}"}

    add_check(
        "V06_SPLITS_AND_MATURITY",
        "All validation and backtest splits enforce chronological ordering and label maturity filtering",
        splits_ok,
        v06_details,
    )

    # 7. Recompute metrics from backtest_predictions.csv and compare with metrics_summary.json & tables
    rep_bt: dict[str, Any] = {}
    try:
        pred_df = pd.read_csv(run_dir / "backtest_predictions.csv", low_memory=False)
        ms = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        pol_a_df = pd.read_csv(run_dir / "policy_frontier_cohort_a.csv")
        pol_b_df = pd.read_csv(run_dir / "policy_frontier_all_orders.csv")

        non_a_preds = pred_df[~pred_df["in_cohort_a"].astype(bool)]
        coh_a_preds = pred_df[pred_df["in_cohort_a"].astype(bool)].copy()
        run_split_contract, is_v2_contract = _run_split_contract(run_dir)
        if is_v2_contract:
            non_a_null_targets = len(non_a_preds) == 367 and bool(non_a_preds["late"].isna().all())
            row_counts_ok = len(pred_df) == 19661 and len(coh_a_preds) == 19294
        else:
            sp_bt = pd.read_csv(run_dir / "split_summary.csv")
            sp_bt = sp_bt[sp_bt["split_id"] == _backtest_split_id(run_split_contract)]
            non_a_null_targets = bool(non_a_preds["late"].isna().all())
            row_counts_ok = (
                not sp_bt.empty
                and len(pred_df) == int(sp_bt.iloc[0]["eval_path_b_total"])
                and len(coh_a_preds) == int(sp_bt.iloc[0]["eval_cohort_a_total"])
            )

        y_true_a = coh_a_preds["late"].astype(int).to_numpy()
        p_cal_a = coh_a_preds["predicted_prob_calibrated"].astype(float).to_numpy()

        recalc_pr_auc = float(average_precision_score(y_true_a, p_cal_a))
        recalc_roc_auc = float(roc_auc_score(y_true_a, p_cal_a))
        recalc_brier = float(brier_score_loss(y_true_a, p_cal_a))

        rep_bt = ms["selected_model_backtest_metrics"]
        diff_pr = abs(recalc_pr_auc - float(rep_bt["pr_auc"]))
        diff_roc = abs(recalc_roc_auc - float(rep_bt["roc_auc"]))
        diff_brier = abs(recalc_brier - float(rep_bt["brier_score"]))

        sel_model = ms["selected_model"]["model_name"]
        pol_a_top10 = pol_a_df[
            (pol_a_df["model_name"] == sel_model)
            & (pol_a_df["label_version"] == "calendar_day")
            & (pol_a_df["budget_pct_label"] == "Top-10%")
        ].iloc[0]

        if "policy_mode" in pol_a_top10 and str(pol_a_top10["policy_mode"]) == "daily_floor":
            # In v2 daily floor mode, verify against daily_policy.csv or stored alert_flag
            if "alert_top10_daily" in coh_a_preds.columns:
                alerted_a = coh_a_preds[coh_a_preds["alert_top10_daily"].astype(bool)]
                tp10_recalc = int(alerted_a["late"].astype(int).sum())
                k10_a = len(alerted_a)
                prec10_recalc = float(tp10_recalc / k10_a) if k10_a > 0 else 0.0
            else:
                k10_a = int(pol_a_top10["alert_budget_k"])
                tp10_recalc = int(pol_a_top10["tp"])
                prec10_recalc = float(tp10_recalc / k10_a) if k10_a > 0 else 0.0
        else:
            k10_a = int(pol_a_top10["alert_budget_k"]) if "alert_budget_k" in pol_a_top10 else int(np.floor(0.10 * len(coh_a_preds)))
            coh_a_sorted = coh_a_preds.sort_values(
                by=["predicted_prob_calibrated", "order_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            tp10_recalc = int(coh_a_sorted.iloc[:k10_a]["late"].astype(int).sum()) if k10_a > 0 else 0
            prec10_recalc = float(tp10_recalc / k10_a) if k10_a > 0 else 0.0

        diff_prec10 = abs(prec10_recalc - float(pol_a_top10["precision"]))

        pol_b_ok = bool(
            (
                pol_b_df["alert_in_cohort_a_count"] + pol_b_df["alert_unlabeled_non_delivered_count"]
                == pol_b_df["alert_total_path_b"]
            ).all()
        ) and int(pol_b_df["path_b_total_orders"].iloc[0]) == len(pred_df)

        metrics_consistent = (
            row_counts_ok
            and non_a_null_targets
            and diff_pr <= 1e-5
            and diff_roc <= 1e-5
            and diff_brier <= 1e-5
            and diff_prec10 <= 1e-5
            and pol_b_ok
        )
        v07_details = {
            "row_counts_ok": row_counts_ok,
            "non_cohort_a_targets_all_na": non_a_null_targets,
            "recalc_pr_auc": round(recalc_pr_auc, 6),
            "reported_pr_auc": rep_bt["pr_auc"],
            "recalc_roc_auc": round(recalc_roc_auc, 6),
            "reported_roc_auc": rep_bt["roc_auc"],
            "recalc_brier": round(recalc_brier, 6),
            "reported_brier": rep_bt["brier_score"],
            "recalc_top10_tp": tp10_recalc,
            "reported_top10_tp": int(pol_a_top10["tp"]),
            "path_b_accounting_exact": pol_b_ok,
        }
    except Exception as exc:
        metrics_consistent = False
        v07_details = {"error": f"{type(exc).__name__}: {exc}"}

    add_check(
        "V07_METRICS_AND_POLICY_CONSISTENCY",
        "Predictions table exactly reproduces metrics_summary.json, Path A policy frontier, and Path B accounting",
        metrics_consistent,
        v07_details,
    )

    # 8. Optional check that narrative docs exist, are non-empty, and match key numbers / claims.json
    if check_docs_sync:
        doc_paths = [
            root / "docs" / "final-report.zh-CN.md",
            root / "docs" / "final-report.en.md",
        ]
        missing_docs = [str(p.relative_to(root)).replace("\\", "/") for p in doc_paths if not p.exists() or p.stat().st_size == 0]
        doc_sync_ok = len(missing_docs) == 0 and bool(rep_bt)
        doc_details: dict[str, Any] = {"missing_or_empty_docs": missing_docs}

        if doc_sync_ok:
            expected_tokens = [
                "99,441",
                "96,470",
                "2,971",
                f"{float(rep_bt['pr_auc']):.4f}",
                f"{float(rep_bt['roc_auc']):.4f}",
            ]
            for dp in doc_paths:
                txt = dp.read_text(encoding="utf-8")
                missing_toks = [t for t in expected_tokens if t not in txt]
                if missing_toks:
                    doc_sync_ok = False
                doc_details[dp.name] = {"missing_tokens": missing_toks}

            claims_path = run_dir / "claims.json"
            if claims_path.exists():
                try:
                    claims_data = json.loads(claims_path.read_text(encoding="utf-8"))
                    claims_list = claims_data.get("claims", [])
                    failed_claims = [
                        c["claim_id"]
                        for c in claims_list
                        if not c.get("passed", bool(c.get("source_sha256") is not None and c.get("value") is not None))
                    ]
                    if not claims_list or failed_claims:
                        doc_sync_ok = False
                    doc_details["claims_checked"] = len(claims_list)
                    doc_details["failed_claims"] = failed_claims
                except Exception as exc:
                    doc_sync_ok = False
                    doc_details["claims_error"] = str(exc)

        add_check(
            "V08_REPORT_NARRATIVE_SYNC",
            "Final reports exist and contain exact verified numbers from experiment manifest",
            doc_sync_ok,
            doc_details,
        )

    overall_passed = all(c["passed"] for c in checks)
    report = {
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir),
        "verified_at": now_iso(),
        "overall_passed": overall_passed,
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }

    if update_manifest and manifest_path.exists():
        atomic_write_json(run_dir / "verification_report.json", report, root=root)
        md_lines = [
            f"# Verification Report — Run `{manifest.get('run_id')}`",
            "",
            f"- **Verified At**: `{report['verified_at']}`",
            f"- **Overall Status**: `{'VERIFIED' if overall_passed else 'VERIFICATION_FAILED'}`",
            f"- **Checks Passed**: `{report['passed_checks']} / {report['total_checks']}`",
            "",
            "| Check ID | Description | Status |",
            "|---|---|---|",
        ]
        for c in checks:
            st = "PASS" if c["passed"] else "FAIL"
            md_lines.append(f"| `{c['check_id']}` | {c['name']} | `{st}` |")
        (run_dir / "verification_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        manifest["verification_status"] = "VERIFIED" if overall_passed else "VERIFICATION_FAILED"
        manifest["verified_at"] = report["verified_at"]
        for vf in ("verification_report.json", "verification_report.md"):
            vp = run_dir / vf
            if vp.exists():
                manifest.setdefault("artifacts", {})[vf] = {
                    "sha256": sha256_file(vp),
                    "bytes": vp.stat().st_size,
                }
        atomic_write_json(run_dir / "manifest.json", manifest, root=root)

    return report

