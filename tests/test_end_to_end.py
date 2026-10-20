"""End-to-end and contract tests for A09 (scoring_input_v2 validation), A10 (non-mutating evaluate from frozen predictions), T15 (batch scoring), and T16/T18 (run & doc verification)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olist_delay.__main__ import main as cli_main
from olist_delay.features import FORBIDDEN_POST_PURCHASE_COLUMNS
from olist_delay.scoring import (
    SCORING_INPUT_V2_REQUIRED_COLUMNS,
    validate_and_prepare_scoring_inputs,
)
from olist_delay.validation import verify_run_directory


@pytest.mark.unit
def test_a09_scoring_input_v2_contract_rejects_4col_and_strips_preexisting_history() -> None:
    """A09: 4-column minimal input must raise ValueError; duplicates, Inf, future orders (> as_of) must fail; pre-existing history columns must be stripped."""
    minimal_4col = pd.DataFrame(
        [
            {
                "order_id": "ord_1",
                "order_purchase_timestamp": "2018-06-01 10:00:00",
                "order_estimated_delivery_date": "2018-06-15 00:00:00",
                "customer_state": "SP",
            }
        ]
    )
    valid_hist = pd.DataFrame(
        [
            {
                "order_id": "hist_1",
                "primary_seller_id": "s1",
                "route_state_pair": "SP->SP",
                "customer_state": "SP",
                "primary_category": "bed_bath_table",
                "order_purchase_timestamp": "2018-05-01 10:00:00",
                "label_available_at": "2018-05-10 12:00:00",
                "late": 0,
                "delay_calendar_days": -2.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="scoring_input_v2"):
        validate_and_prepare_scoring_inputs(minimal_4col, valid_hist)

    # Build a valid scoring_input_v2 row
    valid_order_row = {
        "order_id": "ord_valid_1",
        "customer_unique_id": "cust_u1",
        "order_purchase_timestamp": "2018-06-01 12:00:00",
        "order_estimated_delivery_date": "2018-06-15 00:00:00",
        "customer_zip_code_prefix": "01001",
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "customer_lat": -23.55,
        "customer_lng": -46.63,
        "has_items": True,
        "item_count": 1.0,
        "distinct_product_count": 1.0,
        "seller_count": 1.0,
        "primary_seller_id": "s1",
        "all_seller_ids": "s1",
        "primary_seller_state": "SP",
        "primary_seller_zip_prefix": "01001",
        "seller_lat": -23.55,
        "seller_lng": -46.63,
        "total_price": 100.0,
        "max_item_price": 100.0,
        "mean_item_price": 100.0,
        "total_freight_value": 15.0,
        "freight_ratio": 0.15,
        "total_weight_g": 500.0,
        "max_weight_g": 500.0,
        "total_volume_cm3": 2000.0,
        "max_volume_cm3": 2000.0,
        "mean_product_photos_qty": 2.0,
        "mean_description_lenght": 400.0,
        "primary_category": "bed_bath_table",
        "payment_total_value": 115.0,
        "max_payment_installments": 2.0,
        "payment_type_count": 1.0,
        "primary_payment_type": "credit_card",
        "used_boleto": 0,
        "used_credit_card": 1,
        "used_voucher": 0,
    }
    assert set(SCORING_INPUT_V2_REQUIRED_COLUMNS).issubset(set(valid_order_row.keys()))

    # Inject a poisoned pre-existing history column and forbidden post-purchase column
    poisoned_df = pd.DataFrame([{**valid_order_row, "seller_hist_late_rate_smoothed": 0.999, "late": 1}])
    clean_orders, _, _, _ = validate_and_prepare_scoring_inputs(
        poisoned_df, valid_hist, as_of="2018-06-01T23:59:59"
    )
    assert "seller_hist_late_rate_smoothed" not in clean_orders.columns
    assert "late" not in clean_orders.columns

    # Duplicate order_id must fail
    dup_df = pd.DataFrame([valid_order_row, valid_order_row])
    with pytest.raises(ValueError, match="duplicate"):
        validate_and_prepare_scoring_inputs(dup_df, valid_hist)

    # Order after as_of must fail
    with pytest.raises(ValueError, match="after `--as-of`"):
        validate_and_prepare_scoring_inputs(
            pd.DataFrame([valid_order_row]), valid_hist, as_of="2018-06-01T10:00:00"
        )

    # Inf numeric value must fail
    inf_row = dict(valid_order_row)
    inf_row["total_price"] = float("inf")
    with pytest.raises(ValueError, match="Inf"):
        validate_and_prepare_scoring_inputs(pd.DataFrame([inf_row]), valid_hist)


@pytest.mark.unit
def test_a10_evaluate_requires_explicit_output_dir_and_validation_predictions(
    project_root: Path, tmp_path: Path
) -> None:
    """A10: `evaluate` must refuse missing --output-dir, refuse writing inside run_dir, and fail if validation_predictions.csv is missing."""
    # 1. Missing --output-dir raises ValueError
    with pytest.raises(ValueError, match="--output-dir"):
        cli_main(["evaluate", "--run-id", "review-20261019-backtest-v1"])

    # 2. Writing inside run_dir raises PermissionError
    v1_run_dir = project_root / "artifacts" / "runs" / "review-20261019-backtest-v1"
    if v1_run_dir.exists():
        with pytest.raises(PermissionError):
            cli_main(
                [
                    "evaluate",
                    "--run-id",
                    "review-20261019-backtest-v1",
                    "--output-dir",
                    str(v1_run_dir / "sub_eval"),
                ]
            )
        # 3. Legacy v1 run lacks validation_predictions.csv -> must raise FileNotFoundError instead of re-scoring with backtest model
        with pytest.raises(FileNotFoundError, match="validation_predictions.csv"):
            cli_main(
                [
                    "evaluate",
                    "--run-id",
                    "review-20261019-backtest-v1",
                    "--output-dir",
                    str(tmp_path / "eval_out"),
                ]
            )


@pytest.mark.integration
def test_t15_unlabeled_purchase_time_batch_scoring(project_root: Path, tmp_path: Path) -> None:
    """T15: `olist_delay score` scores unlabeled purchase-time orders under scoring_input_v2 contract."""
    run_id = "experiment-v2-001"
    exp_dir = project_root / "artifacts" / "runs" / run_id
    if not exp_dir.exists():
        run_id = "smoke-v2-001"
        exp_dir = project_root / "artifacts" / "runs" / run_id
    if not exp_dir.exists():
        pytest.skip("Neither experiment-v2-001 nor smoke-v2-001 has been generated yet")

    data_dir = project_root / "artifacts" / "runs" / "data-v2-001"
    assert data_dir.exists(), "Run `prepare --run-id data-v2-001` first."

    sample_in = data_dir / "sample_orders.csv"
    sample_ev = data_dir / "sample_events.csv"
    sample_dec = data_dir / "sample_decision_events.csv"
    df_in = pd.read_csv(sample_in, low_memory=False)

    # Verify sample_orders.csv has zero forbidden post-purchase or label columns and has all 38 scoring_input_v2 columns
    leaked_in_sample = set(df_in.columns) & FORBIDDEN_POST_PURCHASE_COLUMNS
    assert leaked_in_sample == set()
    assert set(SCORING_INPUT_V2_REQUIRED_COLUMNS).issubset(set(df_in.columns))

    out_csv = tmp_path / "scored_batch.csv"
    cmd_args = [
        "score",
        "--run-id",
        run_id,
        "--input",
        str(sample_in),
        "--history-events",
        str(sample_ev),
        "--as-of",
        "2018-06-01T23:59:59",
        "--output",
        str(out_csv),
    ]
    if sample_dec.exists():
        cmd_args.extend(["--decision-events", str(sample_dec)])

    rc = cli_main(cmd_args)
    assert rc == 0
    assert out_csv.exists()
    scored_df = pd.read_csv(out_csv)
    assert len(scored_df) == len(df_in)
    assert "predicted_delay_probability" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert np.all(
        (scored_df["predicted_delay_probability"] >= 0.0)
        & (scored_df["predicted_delay_probability"] <= 1.0)
    )

    # Verify Determinism: scoring a second time yields identical probabilities and ranks
    out_csv_2 = tmp_path / "scored_batch_2.csv"
    cmd_args_2 = list(cmd_args)
    cmd_args_2[cmd_args_2.index(str(out_csv))] = str(out_csv_2)
    rc2 = cli_main(cmd_args_2)
    assert rc2 == 0
    scored_df_2 = pd.read_csv(out_csv_2)
    pd.testing.assert_frame_equal(scored_df, scored_df_2)

    # Verify refusal to write into protected `data/` directory
    with pytest.raises(PermissionError):
        cli_main(
            [
                "score",
                "--run-id",
                run_id,
                "--input",
                str(sample_in),
                "--history-events",
                str(sample_ev),
                "--output",
                str(project_root / "data" / "forbidden_write.csv"),
            ]
        )


@pytest.mark.integration
def test_t16_t18_experiment_v2_verification_and_evaluate(project_root: Path, tmp_path: Path) -> None:
    """T16, T18 & A10: `experiment-v2-001` passes verification and `evaluate` reproduces metrics into an external directory without mutating run_dir."""
    exp_dir = project_root / "artifacts" / "runs" / "experiment-v2-001"
    if not exp_dir.exists():
        pytest.skip("experiment-v2-001 not yet generated")

    eval_out = tmp_path / "recomputed_eval"
    rc = cli_main(["evaluate", "--run-id", "experiment-v2-001", "--output-dir", str(eval_out)])
    assert rc == 0
    assert (eval_out / "metrics_summary.json").exists()
    assert (eval_out / "figures" / "01_cohort_and_label_reconciliation.png").exists()

    has_final_docs = (
        (project_root / "docs" / "final-report.en.md").exists()
        and (project_root / "docs" / "final-report.zh-CN.md").exists()
    )
    report = verify_run_directory(
        run_dir=exp_dir,
        root=project_root,
        update_manifest=False,
        check_docs_sync=has_final_docs,
    )
    failed = [c for c in report["checks"] if not c["passed"]]
    assert report["overall_passed"] is True, f"Verification checks failed: {failed}"


@pytest.mark.integration
def test_a15_independent_recompute_results_and_claims(project_root: Path, tmp_path: Path) -> None:
    """A15: Independent recompute_results.py verifies exact integer equality and float tolerance on experiment-v2-001."""
    exp_dir = project_root / "artifacts" / "runs" / "experiment-v2-001"
    if not exp_dir.exists():
        pytest.skip("experiment-v2-001 not yet generated")

    import importlib.util

    script_path = project_root / "scripts" / "recompute_results.py"
    spec = importlib.util.spec_from_file_location("recompute_results", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    res = mod.recompute_and_verify(
        run_id="experiment-v2-001",
        artifacts_root=project_root / "artifacts",
    )
    assert res["status"] == "PASSED", f"Independent recomputation failed: {res['errors']}"
    assert res["num_failed"] == 0
    assert res["num_passed"] == res["num_checks"]

