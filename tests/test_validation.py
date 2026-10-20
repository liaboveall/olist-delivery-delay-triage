"""Unit and tamper tests for T14 (negative tamper detection) and T17 (failure & overwrite protection)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from olist_delay.artifacts import RunContext, verify_baseline_protection
from olist_delay.config import load_config
from olist_delay.features import assert_zero_leakage_features
from olist_delay.validation import verify_run_directory

pytestmark = pytest.mark.unit



def test_t17_raw_data_protection_all_9_files_intact(project_root: Path) -> None:
    """T17: All 9 raw Olist dataset CSV files in data/ must match their frozen SHA-256 digests."""
    check = verify_baseline_protection(root=project_root)
    assert check["all_intact"] is True
    assert check["total_checked"] == 9
    assert check["mismatches"] == []


def test_t17_duplicate_run_id_overwrite_protection(project_root: Path, tmp_path: Path) -> None:
    """T17: Attempting to start a new run with an existing run_id must raise FileExistsError."""
    cfg = load_config("configs/research-v1.json", root=project_root)
    cfg.setdefault("paths", {})["data_dir"] = str(project_root / "data")
    with RunContext(run_id="run_test_1", config=cfg, command="prepare", root=tmp_path) as ctx:
        (ctx.run_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    with pytest.raises(FileExistsError):
        with RunContext(run_id="run_test_1", config=cfg, command="prepare", root=tmp_path):
            pass


def test_t17_interrupted_run_marked_failed(project_root: Path, tmp_path: Path) -> None:
    """T17: An exception inside RunContext must mark manifest status=FAILED and verification_status=NOT_RUN."""
    cfg = load_config("configs/research-v1.json", root=project_root)
    cfg.setdefault("paths", {})["data_dir"] = str(project_root / "data")
    with pytest.raises(ValueError, match="simulated crash"):
        with RunContext(run_id="run_crash", config=cfg, command="backtest", root=tmp_path):
            raise ValueError("simulated crash")

    manifest = json.loads((tmp_path / "artifacts" / "runs" / "run_crash" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["verification_status"] == "NOT_RUN"
    assert "simulated crash" in manifest["failure_reason"]


def test_t14_leakage_guard_rejects_forbidden_post_purchase_columns() -> None:
    """T14: Passing any post-purchase or target column into assert_zero_leakage_features must raise ValueError."""
    for forbidden_col in [
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "review_score",
        "late_calendar_day",
        "late",
    ]:
        with pytest.raises(ValueError, match="Forbidden post-purchase or target columns"):
            assert_zero_leakage_features(["total_price", forbidden_col])


def test_t14_negative_tamper_detection_on_run_directory(project_root: Path, tmp_path: Path) -> None:
    """T14: Cloning a valid run directory and tampering with hashes, cohort counts, or leakage flags must fail verification."""
    src_run = project_root / "artifacts" / "runs" / "experiment-v2-001"
    if not src_run.exists():
        src_run = project_root / "artifacts" / "runs" / "experiment-v1"
    if not src_run.exists():
        pytest.skip("experiment-v2-001 not yet generated")

    # 1. Clean copy of experiment-v1 passes verification
    clean_copy = tmp_path / "exp-v1-clean"
    shutil.copytree(src_run, clean_copy)
    res_clean = verify_run_directory(clean_copy, root=project_root, update_manifest=False)
    assert res_clean["overall_passed"] is True

    # 2. Tamper with a CSV byte -> V03_MANIFEST_AND_HASHES must fail
    tampered_hash_dir = tmp_path / "exp-v1-tampered-hash"
    shutil.copytree(src_run, tampered_hash_dir)
    sp_csv = tampered_hash_dir / "split_summary.csv"
    sp_csv.write_text(sp_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    res_hash = verify_run_directory(tampered_hash_dir, root=project_root, update_manifest=False)
    assert res_hash["overall_passed"] is False
    failed_ids_hash = {c["check_id"] for c in res_hash["checks"] if not c["passed"]}
    assert "V03_MANIFEST_AND_HASHES" in failed_ids_hash

    # 3. Tamper with data_quality_summary.json row reconciliation -> V04_COHORT_AND_LABEL_RECONCILIATION must fail
    tampered_row_dir = tmp_path / "exp-v1-tampered-rows"
    shutil.copytree(src_run, tampered_row_dir)
    dq_path = tampered_row_dir / "data_quality_summary.json"
    dq = json.loads(dq_path.read_text(encoding="utf-8"))
    dq["row_reconciliation"]["unexplained_row_delta"] = 5
    dq_path.write_text(json.dumps(dq, indent=2), encoding="utf-8")
    res_row = verify_run_directory(tampered_row_dir, root=project_root, update_manifest=False)
    assert res_row["overall_passed"] is False
    failed_ids_row = {c["check_id"] for c in res_row["checks"] if not c["passed"]}
    assert "V04_COHORT_AND_LABEL_RECONCILIATION" in failed_ids_row

    # 4. Tamper with feature_availability_audit.csv to inject a failed leakage row -> V05_ZERO_LEAKAGE_AND_EVENT_CLOCK must fail
    tampered_feat_dir = tmp_path / "exp-v1-tampered-feat"
    shutil.copytree(src_run, tampered_feat_dir)
    fa_path = tampered_feat_dir / "feature_availability_audit.csv"
    fa_df = pd.read_csv(fa_path)
    fa_df.loc[0, "audit_passed"] = False
    fa_df.to_csv(fa_path, index=False)
    res_feat = verify_run_directory(tampered_feat_dir, root=project_root, update_manifest=False)
    assert res_feat["overall_passed"] is False
    failed_ids_feat = {c["check_id"] for c in res_feat["checks"] if not c["passed"]}
    assert "V05_ZERO_LEAKAGE_AND_EVENT_CLOCK" in failed_ids_feat


def test_a01_config_v2_schema_and_grid_cap(project_root: Path) -> None:
    """A01: research-v2, experiment-v2, and smoke-v2 pass validation; unknown fields and >24 candidate grids are rejected."""
    from olist_delay.__main__ import main as cli_main
    from olist_delay.config import expand_model_grid, inspect_config_summary, validate_config

    res_cfg = load_config("configs/research-v2.json", root=project_root)
    exp_cfg = load_config("configs/experiment-v2.json", root=project_root)
    smk_cfg = load_config("configs/smoke-v2.json", root=project_root)

    assert res_cfg["config_version"] == "research-v2.0"
    assert exp_cfg["config_version"] == "experiment-v2.0"
    assert smk_cfg["smoke_only"] is True

    cands = expand_model_grid(exp_cfg)
    assert len(cands) == 24
    smk_cands = expand_model_grid(smk_cfg)
    assert len(smk_cands) == 4

    summary = inspect_config_summary(exp_cfg)
    assert summary["expanded_model_candidates_count"] == 24
    assert summary["within_model_budget"] is True
    assert summary["baseline_rules_count"] == 5

    rc = cli_main(["inspect-config", "--config", "configs/experiment-v2.json"])
    assert rc == 0

    # Unknown top-level field must fail
    bad_unknown = dict(exp_cfg)
    bad_unknown["undocumented_magic_key"] = 123
    with pytest.raises(ValueError, match="Unknown top-level configuration keys"):
        validate_config(bad_unknown)

    # Grid exceeding max_fitted_model_configs=24 must fail
    bad_grid = json.loads(json.dumps({k: v for k, v in exp_cfg.items() if not k.startswith("_")}))
    bad_grid["model_grid"]["logistic_regression"]["C"] = [0.01, 0.1, 1.0, 10.0]
    with pytest.raises(ValueError, match="exceeding selection_contract.max_fitted_model_configs=24"):
        validate_config(bad_grid)


def test_a02_write_protection_and_verified_immutability(project_root: Path, tmp_path: Path) -> None:
    """A02: Protected paths cannot be written, non-empty directories cannot be reused, and VERIFIED runs are immutable."""
    from olist_delay.artifacts import assert_writable_path

    for prot in [
        project_root / "data" / "olist_orders_dataset.csv",
        project_root / "artifacts" / "review-2026-10-19" / "probe-results.json",
        project_root / "artifacts" / "runs" / "review-20261019-backtest-v1" / "manifest.json",
        project_root / "configs" / "research-v1.json",
        project_root / "configs" / "experiment-v1.json",
    ]:
        with pytest.raises(PermissionError, match="Refusing to write to protected path"):
            assert_writable_path(prot, root=project_root)

    cfg = load_config("configs/research-v2.json", root=project_root)
    cfg.setdefault("paths", {})["data_dir"] = str(project_root / "data")

    # Non-empty directory without manifest.json must also raise FileExistsError
    non_empty_dir = tmp_path / "artifacts" / "runs" / "pre_existing_dir"
    non_empty_dir.mkdir(parents=True, exist_ok=True)
    (non_empty_dir / "stray_file.txt").write_text("stray", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        with RunContext(run_id="pre_existing_dir", config=cfg, command="prepare", root=tmp_path):
            pass

    # VERIFIED run must refuse allow_existing_for_update=True
    with RunContext(run_id="verified_run_test", config=cfg, command="prepare", root=tmp_path) as ctx:
        assert (ctx.run_dir / "source_snapshot.json").exists()
    m_path = tmp_path / "artifacts" / "runs" / "verified_run_test" / "manifest.json"
    m_data = json.loads(m_path.read_text(encoding="utf-8"))
    m_data["verification_status"] = "VERIFIED"
    m_path.write_text(json.dumps(m_data, indent=2), encoding="utf-8")

    with pytest.raises(PermissionError, match="already VERIFIED and frozen"):
        with RunContext(
            run_id="verified_run_test",
            config=cfg,
            command="evaluate",
            root=tmp_path,
            allow_existing_for_update=True,
        ):
            pass


def test_a03_incomplete_synthetic_run_rejected_by_verifier(project_root: Path, tmp_path: Path) -> None:
    """A03: A synthetic incomplete run (0-byte PNGs, empty artifacts dict, missing model, fake counts) must fail V02, V03, V04, V06, V07, V08."""
    from olist_delay.features import ALL_MODEL_FEATURES
    from olist_delay.validation import REQUIRED_FIGURES, REQUIRED_RUN_FILES

    fake_run = tmp_path / "fake_run"
    (fake_run / "figures").mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_RUN_FILES + REQUIRED_FIGURES:
        if rel == "models/selected_model.pkl":
            continue
        p = fake_run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    manifest = {
        "schema_version": "1.0",
        "run_id": "fake",
        "status": "COMPLETED",
        "config_hash": {},
        "code_version": {},
        "input_hashes": {},
        "environment": {},
        "label_version": "calendar_day",
        "cohort_definitions": {},
        "feature_version": "v1",
        "split_protocol": {},
        "random_seed": 0,
        "policy_contract": {},
        "artifacts": {},
        "resource_usage": {},
    }
    (fake_run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dq = {
        "row_reconciliation": {
            "source_orders_total": 99441,
            "path_a_delivered_evaluable_orders": 96470,
            "path_a_excluded_orders": 2971,
            "unexplained_row_delta": 0,
        },
        "label_reconciliation": {
            "full_cohort_a": {"same_day_after_midnight_count": 0, "exact_difference_matches_same_day": True},
            "backtest_window": {"same_day_after_midnight_count": 0, "exact_difference_matches_same_day": True},
        },
    }
    (fake_run / "data_quality_summary.json").write_text(json.dumps(dq), encoding="utf-8")
    pd.DataFrame({"feature_name": ALL_MODEL_FEATURES, "audit_passed": [True] * len(ALL_MODEL_FEATURES)}).to_csv(
        fake_run / "feature_availability_audit.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "split_id": "val_2018_01",
                "train_matured_count": 10,
                "train_dropped_in_transit_count": 1,
                "chronological_strict_pass": True,
                "maturity_strict_pass": True,
            }
        ]
    ).to_csv(fake_run / "split_summary.csv", index=False)
    pd.DataFrame(
        [
            {"order_id": "a", "in_cohort_a": True, "late": 0, "predicted_prob_calibrated": 0.1},
            {"order_id": "b", "in_cohort_a": True, "late": 1, "predicted_prob_calibrated": 0.9},
        ]
    ).to_csv(fake_run / "backtest_predictions.csv", index=False)
    (fake_run / "metrics_summary.json").write_text(
        json.dumps(
            {
                "selected_model": {"model_name": "m"},
                "selected_model_backtest_metrics": {"pr_auc": 1.0, "roc_auc": 1.0, "brier_score": 0.01},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"model_name": "m", "label_version": "calendar_day", "budget_pct_label": "Top-10%", "precision": 1.0, "tp": 1}]
    ).to_csv(fake_run / "policy_frontier_cohort_a.csv", index=False)
    pd.DataFrame(
        [
            {
                "alert_in_cohort_a_count": 1,
                "alert_unlabeled_non_delivered_count": 0,
                "alert_total_path_b": 1,
                "path_b_total_orders": 2,
            }
        ]
    ).to_csv(fake_run / "policy_frontier_all_orders.csv", index=False)

    res = verify_run_directory(fake_run, root=project_root, update_manifest=False, check_docs_sync=True)
    assert res["overall_passed"] is False
    failed_ids = {c["check_id"] for c in res["checks"] if not c["passed"]}
    for expected_fail in (
        "V02_REQUIRED_FILES",
        "V03_MANIFEST_AND_HASHES",
        "V04_COHORT_AND_LABEL_RECONCILIATION",
        "V06_SPLITS_AND_MATURITY",
        "V07_METRICS_AND_POLICY_CONSISTENCY",
        "V08_REPORT_NARRATIVE_SYNC",
    ):
        assert expected_fail in failed_ids, f"Expected {expected_fail} to fail on synthetic incomplete run"

