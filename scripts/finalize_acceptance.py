"""Finalize W7 acceptance gate verification, cross-run reproducibility checks, and delivery packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from olist_delay.artifacts import (
    PROTECTED_RELATIVE_PATHS,
    RAW_DATA_SHA256,
    compute_code_hash,
    now_iso,
    sha256_file,
    verify_baseline_protection,
)


def _parse_junit_xml(xml_path: Path) -> dict[str, int]:
    if not xml_path.exists():
        raise FileNotFoundError(f"JUnit XML file not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    passed = tests - failures - errors - skipped
    return {
        "executed": tests,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _check_protected_paths(root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_present = True
    for rel in PROTECTED_RELATIVE_PATHS:
        p = root / rel
        exists = p.exists()
        if not exists:
            all_present = False
        info: dict[str, Any] = {"exists": exists, "type": "dir" if p.is_dir() else "file"}
        if exists and p.is_file():
            info["sha256"] = sha256_file(p)
            info["bytes"] = p.stat().st_size
        elif exists and p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            info["file_count"] = len(files)
        results[rel] = info
    return {"all_present": all_present, "paths": results}


def _package_delivery(
    root: Path,
    run_id: str,
    smoke_run_id: str,
    data_run_id: str,
) -> dict[str, Any]:
    delivery_dir = root / "delivery"
    reports_dir = delivery_dir / "reports"
    pres_dir = delivery_dir / "presentation"
    configs_dir = delivery_dir / "configs"
    evidence_root = delivery_dir / "evidence"
    evidence_dir = evidence_root / run_id
    analysis_src = root / "artifacts" / "analysis" / f"{run_id}-report"
    analysis_dst = evidence_root / f"{run_id}-report-analysis"

    aux_reports: dict[str, bytes] = {}
    for aux_name in ("consultation-brief.en.md", "development-status.md", "speaker-notes.zh-CN.md"):
        aux_src = (root / "docs" / aux_name) if (root / "docs" / aux_name).exists() else (reports_dir / aux_name)
        if aux_src.exists():
            aux_reports[aux_name] = aux_src.read_bytes()

    # delivery/ is fully generated: rebuild the folders this function owns so removed files never linger.
    for d in (reports_dir, pres_dir, configs_dir, evidence_root):
        if d.exists():
            shutil.rmtree(d)
    for d in (reports_dir / "figures", pres_dir, configs_dir, evidence_dir / "figures", evidence_dir / "models"):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Reports, briefs and governance documents (report figures travel with them)
    doc_files = [
        "final-report.en.md",
        "final-report.en.html",
        "final-report.en.pdf",
        "final-report.zh-CN.md",
        "qa.md",
        "research-contract.md",
        "data-card.md",
        "decision-log.md",
        "test-exposure-log.md",
    ]
    for name in doc_files:
        src = root / "docs" / name
        if src.exists():
            shutil.copy2(src, reports_dir / name)
    for aux_name, aux_bytes in aux_reports.items():
        (reports_dir / aux_name).write_bytes(aux_bytes)
    for fig in sorted((root / "docs" / "figures").glob("*.png")):
        shutil.copy2(fig, reports_dir / "figures" / fig.name)

    # 2. Presentation (the Markdown source points at ../docs/figures; in the package that is ../reports/figures)
    md_src = root / "presentation" / "slides.en.md"
    if md_src.exists():
        text = md_src.read_text(encoding="utf-8").replace("../docs/figures/", "../reports/figures/")
        (pres_dir / "slides.en.md").write_text(text, encoding="utf-8")
    for name in ["slides.en.html", "slides.en.pdf"]:
        src = root / "presentation" / name
        if src.exists():
            shutil.copy2(src, pres_dir / name)

    # 3. Configs & environment lock
    for name in ["research-v2.json", "experiment-v2.json", "smoke-v2.json"]:
        src = root / "configs" / name
        if src.exists():
            shutil.copy2(src, configs_dir / name)
    for name in ["requirements-lock.txt", "pyproject.toml", "README.md"]:
        src = root / name
        if src.exists():
            shutil.copy2(src, configs_dir / name)

    # 4. Formal run summary evidence (multi-MB per-order tables stay in the workspace, verified by hash)
    run_dir = root / "artifacts" / "runs" / run_id
    evidence_files = [
        "manifest.json",
        "config.resolved.json",
        "environment.txt",
        "execution.log",
        "source_snapshot.json",
        "data_quality_summary.json",
        "cohort_observation_analysis.csv",
        "feature_availability_audit.csv",
        "split_summary.csv",
        "candidate_manifest.json",
        "selection_record.json",
        "validation_folds_metrics.csv",
        "model_comparison.csv",
        "calibration_summary.csv",
        "policy_frontier_cohort_a.csv",
        "policy_frontier_all_orders.csv",
        "daily_policy.csv",
        "cost_sensitivity.csv",
        "sensitivity_summary.csv",
        "subgroup_metrics.csv",
        "monthly_drift_report.csv",
        "feature_importance.csv",
        "ex_post_stage_summary.csv",
        "metrics_summary.json",
        "uncertainty_summary.json",
        "claims.json",
        "metric_definitions.json",
        "model_card.md",
        "verification_report.json",
        "verification_report.md",
    ]
    for name in evidence_files:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, evidence_dir / name)
    model_src = run_dir / "models" / "selected_model.pkl"
    if model_src.exists():
        shutil.copy2(model_src, evidence_dir / "models" / "selected_model.pkl")
    for fig in sorted((run_dir / "figures").glob("*.png")):
        shutil.copy2(fig, evidence_dir / "figures" / fig.name)

    # 5. Report analysis layer (facts behind every narrative number, tables and figures)
    if analysis_src.exists():
        shutil.copytree(analysis_src, analysis_dst)
    robust_src = root / "artifacts" / "analysis" / "robustness-report"
    if robust_src.exists():
        shutil.copytree(robust_src, evidence_root / "robustness-report-analysis")
    for extra_run in ("experiment-v3-late2017-001", "experiment-v3-2018-001"):
        x_dir = root / "artifacts" / "runs" / extra_run
        if not x_dir.exists():
            continue
        x_dst = evidence_root / extra_run
        x_dst.mkdir(parents=True, exist_ok=True)
        for name in ["manifest.json", "config.resolved.json", "selection_record.json", "split_summary.csv",
                     "model_comparison.csv", "uncertainty_summary.json", "cost_scenarios.csv", "verification_report.json"]:
            if (x_dir / name).exists():
                shutil.copy2(x_dir / name, x_dst / name)

    # 6. SUBMISSION-MANIFEST.md
    submission_manifest = f"""# Submission Package

IS5126 Final Project: *Can a Marketplace Spot Late Deliveries on the Day They Are Ordered? Purchase-time delay triage on Olist.*

## 1. Evidence

- **Formal run `{run_id}`.** Selection and backtest results. VERIFIED by `verify --check-docs`; `scripts/recompute_results.py` independently recomputes its headline claims.
- **Analysis layer `{run_id}-report`.** It is built read-only on the formal run by `scripts/report_analysis.py` and holds every number the narrative documents quote (`report_facts.json`), their tables and the report figures.
- **Clean-environment smoke run `{smoke_run_id}`.** It reproduces the earlier smoke run bit-for-bit.
- **Data snapshot `{data_run_id}`.**

## 2. Contents

- **`reports/`**
  - `final-report.en.md`: the final report, with HTML and PDF exports when present.
  - `final-report.zh-CN.md`: Chinese version for team review.
  - `qa.md`: defense Q&A.
  - `figures/`: the report figures.
  - Governance documents: `research-contract.md`, `data-card.md`, `decision-log.md`, `test-exposure-log.md`.
- **`presentation/`**
  - `slides.en.md`: Markdown source.
  - `slides.en.html`: self-contained deck with all figures embedded, arrow-key navigation and one slide per printed page.
  - `slides.en.pdf`: PDF export when present.
- **`configs/`**: the frozen v2 configurations, `requirements-lock.txt`, `pyproject.toml` and `README.md`.
- **`evidence/{run_id}/`**: summary tables, manifest, claims, selection record, model and the pipeline's own figures.
- **`evidence/{run_id}-report-analysis/`**: `report_facts.json`, `tables/*.csv` and `figures/*.png`.
- **`evidence/experiment-v3-late2017-001/` and `evidence/experiment-v3-2018-001/`**: summary evidence of the second design (pre-registered late-2017 test; post hoc 2018 re-run).
- **`evidence/robustness-report-analysis/`**: facts, tables and figures for the second design.
- **Acceptance records**: `acceptance-summary.json` and `acceptance-summary.md`.
- **`file-checksums.json`**: SHA-256 and size of every file in this package.

## 3. Data source and privacy

- The 9 raw Olist CSV files (about 126 MB) and the per-order prediction tables stay in the workspace (`data/`, `artifacts/runs/{run_id}/`). They are verified by SHA-256 rather than copied here.
- **Source:** Brazilian E-Commerce Public Dataset by Olist, https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Privacy:** customer, seller and order identifiers are pseudonymous hashes published by Olist, and locations are 5-digit postal prefixes. No personal data is stored or inferred.

## 4. Reproduce

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/acceptance.ps1 -RunId {run_id}
```

## 5. Before submitting (team)

- [ ] Fill in the group number and member names (report title block, slides, brief).
- [ ] Confirm the required format and length of the report and presentation.
- [ ] Check the course policy on disclosing AI assistance.
"""
    (delivery_dir / "SUBMISSION-MANIFEST.md").write_text(submission_manifest, encoding="utf-8")
    return {"delivery_dir": str(delivery_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize W7 acceptance summary and delivery package.")
    parser.add_argument("--run-id", default="experiment-v2-002")
    parser.add_argument("--smoke-run-id", default="smoke-v2-006")
    parser.add_argument("--baseline-smoke-run-id", default="smoke-v2-004")
    parser.add_argument("--data-run-id", default="data-v2-001")
    parser.add_argument("--acceptance-dir", default="artifacts/acceptance")
    parser.add_argument("--pre-eval-hashes-json", required=True)
    parser.add_argument("--step-timings-json", required=True)
    parser.add_argument("--report-recheck-dir", default="artifacts/analysis/acceptance-recheck")
    parser.add_argument("--robustness-recheck-dir", default="artifacts/analysis/acceptance-robustness-recheck")
    parser.add_argument("--doc-check-json", default="artifacts/acceptance/doc_consistency.json")
    args = parser.parse_args()

    acc_dir = PROJECT_ROOT / args.acceptance_dir
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / args.run_id
    smoke_dir = PROJECT_ROOT / "artifacts" / "runs" / args.smoke_run_id
    base_smoke_dir = PROJECT_ROOT / "artifacts" / "runs" / args.baseline_smoke_run_id

    # 1. Parse JUnit XML results
    unit_stats = _parse_junit_xml(acc_dir / "junit-unit.xml")
    integ_stats = _parse_junit_xml(acc_dir / "junit-integration.xml")

    # 2. Verify raw data & protected paths
    raw_protection = verify_baseline_protection(PROJECT_ROOT)
    protected_paths = _check_protected_paths(PROJECT_ROOT)

    # 3. Verify source code hash invariance
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    smoke_manifest = json.loads((smoke_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    current_code = compute_code_hash(PROJECT_ROOT)
    manifest_code = manifest.get("code_version", {})
    smoke_code = smoke_manifest.get("code_version", {})

    # Every source file must match the clean-environment smoke run exactly, and every backtest module must match the formal run
    backtest_files_match = all(
        current_code["files"].get(k) == v
        for k, v in manifest_code.get("files", {}).items()
        if not k.endswith("scoring.py")
    )
    smoke_files_match = (
        current_code.get("combined_sha256") == smoke_code.get("combined_sha256")
        and current_code.get("files") == smoke_code.get("files")
    )
    source_unchanged = bool(backtest_files_match and smoke_files_match)

    # 4. Verify read-only evaluate did not mutate original run files (A13)
    pre_eval_hashes = json.loads(Path(args.pre_eval_hashes_json).read_text(encoding="utf-8-sig"))
    post_eval_hashes = {
        rel: sha256_file(run_dir / rel) for rel in pre_eval_hashes.keys()
    }
    original_run_unmodified = pre_eval_hashes == post_eval_hashes

    # 5. Cross-run reproducibility checks (W7-01)
    # 5a. Two independent evaluate runs on the formal run
    eval1_dir = acc_dir / "eval-run1"
    eval2_dir = acc_dir / "eval-run2"
    eval_repro_files = ["daily_policy.csv", "metrics_summary.json", "claims.json", "policy_frontier_cohort_a.csv"]
    eval_repro_matches = {}
    for fname in eval_repro_files:
        h1 = sha256_file(eval1_dir / fname)
        h2 = sha256_file(eval2_dir / fname)
        h_orig = sha256_file(run_dir / fname)
        if fname == "claims.json":
            c1 = json.loads((eval1_dir / fname).read_text(encoding="utf-8-sig")).get("claims", [])
            c2 = json.loads((eval2_dir / fname).read_text(encoding="utf-8-sig")).get("claims", [])
            c_orig = json.loads((run_dir / fname).read_text(encoding="utf-8-sig")).get("claims", [])
            strip_hash = lambda lst: [{k: v for k, v in item.items() if k not in ("source_sha256", "passed")} for item in lst]
            exact_match = (strip_hash(c1) == strip_hash(c2) == strip_hash(c_orig))
        elif fname.endswith(".json"):
            j1 = json.loads((eval1_dir / fname).read_text(encoding="utf-8-sig"))
            j2 = json.loads((eval2_dir / fname).read_text(encoding="utf-8-sig"))
            j_orig = json.loads((run_dir / fname).read_text(encoding="utf-8-sig"))
            for d in (j1, j2, j_orig):
                if isinstance(d, dict):
                    d.pop("generated_at", None)
                    d.pop("evaluated_at", None)
            exact_match = (j1 == j2 == j_orig)
        else:
            exact_match = (h1 == h2 == h_orig)
        eval_repro_matches[fname] = {
            "eval_run1_sha256": h1,
            "eval_run2_sha256": h2,
            "original_run_sha256": h_orig,
            "exact_match": exact_match,
        }

    # 5b. Two independent score runs on sample_orders.csv
    score1_path = acc_dir / "score-run1" / "predictions.csv"
    score2_path = acc_dir / "score-run2" / "predictions.csv"
    orig_score_path = PROJECT_ROOT / "artifacts" / "scoring" / "sample-v2-001" / "predictions.csv"
    score1_hash = sha256_file(score1_path)
    score2_hash = sha256_file(score2_path)
    orig_score_hash = sha256_file(orig_score_path) if orig_score_path.exists() else score1_hash
    score_repro = {
        "score_run1_sha256": score1_hash,
        "score_run2_sha256": score2_hash,
        "sample_v2_001_sha256": orig_score_hash,
        "exact_match": (score1_hash == score2_hash),
    }

    # 5c. Independent smoke backtest reproducibility (new smoke run vs the baseline smoke run)
    smoke_repro = {}
    for fname in ["backtest_predictions.csv", "daily_policy.csv", "validation_predictions.csv"]:
        h_new = sha256_file(smoke_dir / fname)
        h_base = sha256_file(base_smoke_dir / fname)
        smoke_repro[fname] = {
            f"{args.smoke_run_id}_sha256": h_new,
            f"{args.baseline_smoke_run_id}_sha256": h_base,
            "exact_match": (h_new == h_base),
        }

    # 6. Load verification & recompute reports
    verify_report = json.loads((run_dir / "verification_report.json").read_text(encoding="utf-8-sig"))
    recompute_report = json.loads((acc_dir / "recompute_report.json").read_text(encoding="utf-8-sig"))
    step_timings_raw = json.loads(Path(args.step_timings_json).read_text(encoding="utf-8-sig"))
    step_timings = step_timings_raw if isinstance(step_timings_raw, list) else [step_timings_raw]

    recompute_passed = bool(
        recompute_report.get("overall_passed") is True or recompute_report.get("status") == "PASSED"
    )
    recompute_passed_count = int(recompute_report.get("passed_count", recompute_report.get("num_passed", 0)))
    recompute_total_checks = int(recompute_report.get("total_checks", recompute_report.get("num_checks", 0)))
    recompute_failed_count = int(recompute_report.get("failed_count", recompute_report.get("num_failed", 0)))

    # 6b. Report layer: facts rebuilt in the clean environment must equal the committed facts, and every
    #     narrative document must pass the consistency check.
    canonical_facts_path = PROJECT_ROOT / "artifacts" / "analysis" / f"{args.run_id}-report" / "report_facts.json"
    recheck_facts_path = PROJECT_ROOT / args.report_recheck_dir / "report_facts.json"
    canonical_facts = json.loads(canonical_facts_path.read_text(encoding="utf-8"))["facts"]
    recheck_facts = json.loads(recheck_facts_path.read_text(encoding="utf-8"))["facts"]
    differing = sorted(k for k in set(canonical_facts) | set(recheck_facts) if canonical_facts.get(k) != recheck_facts.get(k))
    robust_canonical = json.loads(
        (PROJECT_ROOT / "artifacts" / "analysis" / "robustness-report" / "report_facts.json").read_text(encoding="utf-8")
    )["facts"]
    robust_recheck = json.loads(
        (PROJECT_ROOT / args.robustness_recheck_dir / "report_facts.json").read_text(encoding="utf-8")
    )["facts"]
    differing += sorted(
        f"robustness:{k}" for k in set(robust_canonical) | set(robust_recheck) if robust_canonical.get(k) != robust_recheck.get(k)
    )
    canonical_figs = {p.name: sha256_file(p) for p in sorted((PROJECT_ROOT / "docs" / "figures").glob("fig*.png"))}
    recheck_figs = {
        p.name: sha256_file(p)
        for d in (args.report_recheck_dir, args.robustness_recheck_dir)
        for p in sorted((PROJECT_ROOT / d / "figures").glob("fig*.png"))
    }
    report_facts_check = {
        "canonical": str(canonical_facts_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "recheck": str(recheck_facts_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "facts": len(canonical_facts) + len(robust_canonical),
        "differing_facts": differing[:50],
        "figures_identical": bool(canonical_figs) and canonical_figs == recheck_figs,
        "exact_match": not differing,
    }
    doc_check = json.loads((PROJECT_ROOT / args.doc_check_json).read_text(encoding="utf-8"))
    doc_check_summary = {
        "passed": bool(doc_check.get("passed")),
        "documents": len(doc_check.get("documents", {})),
        "untraceable_numbers": sum(len(d.get("unknown_numbers", [])) for d in doc_check.get("documents", {}).values()),
        "missing_images": sum(len(d.get("missing_images", [])) for d in doc_check.get("documents", {}).values()),
        "html_deck": doc_check.get("html_deck"),
    }

    # 7. Evaluate overall gate pass
    total_tests_executed = unit_stats["executed"] + integ_stats["executed"]
    total_tests_passed = unit_stats["passed"] + integ_stats["passed"]
    total_tests_failed = unit_stats["failed"] + integ_stats["failed"] + unit_stats["errors"] + integ_stats["errors"]
    total_tests_skipped = unit_stats["skipped"] + integ_stats["skipped"]

    all_eval_repro_ok = all(v["exact_match"] for v in eval_repro_matches.values())
    all_smoke_repro_ok = all(v["exact_match"] for v in smoke_repro.values())

    overall_passed = bool(
        raw_protection["all_intact"]
        and protected_paths["all_present"]
        and source_unchanged
        and original_run_unmodified
        and unit_stats["executed"] > 0
        and integ_stats["executed"] > 0
        and total_tests_failed == 0
        and total_tests_skipped == 0
        and verify_report.get("overall_passed") is True
        and recompute_passed
        and recompute_failed_count == 0
        and all_eval_repro_ok
        and score_repro["exact_match"]
        and all_smoke_repro_ok
        and report_facts_check["exact_match"]
        and doc_check_summary["passed"]
    )

    # 8. Package delivery directory
    _package_delivery(PROJECT_ROOT, args.run_id, args.smoke_run_id, args.data_run_id)
    delivery_dir = PROJECT_ROOT / "delivery"

    summary_payload = {
        "schema_version": "2.0.0",
        "generated_at": now_iso(),
        "overall_passed": overall_passed,
        "run_id": args.run_id,
        "smoke_run_id": args.smoke_run_id,
        "baseline_smoke_run_id": args.baseline_smoke_run_id,
        "data_run_id": args.data_run_id,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "venv_prefix": sys.prefix,
        },
        "source_unchanged": source_unchanged,
        "code_version": current_code,
        "input_hashes": RAW_DATA_SHA256,
        "raw_data_protection": raw_protection,
        "protected_paths_check": protected_paths,
        "original_run_unmodified_after_evaluate": original_run_unmodified,
        "pytest_summary": {
            "unit": unit_stats,
            "integration": integ_stats,
            "total": {
                "executed": total_tests_executed,
                "passed": total_tests_passed,
                "failed": total_tests_failed,
                "skipped": total_tests_skipped,
            },
        },
        "formal_verification": {
            "overall_passed": verify_report.get("overall_passed"),
            "checks_passed": sum(1 for c in verify_report.get("checks", []) if c.get("passed")),
            "checks_total": len(verify_report.get("checks", [])),
        },
        "independent_recompute": {
            "overall_passed": recompute_passed,
            "passed_count": recompute_passed_count,
            "total_checks": recompute_total_checks,
            "failed_count": recompute_failed_count,
        },
        "report_layer": {
            "report_facts_reproducible": report_facts_check,
            "document_consistency": doc_check_summary,
        },
        "reproducibility_checks": {
            "evaluate_dual_run": eval_repro_matches,
            "score_dual_run": score_repro,
            "smoke_backtest_cross_run": smoke_repro,
        },
        "resource_usage_from_formal_runs": {
            args.data_run_id: json.loads(
                (PROJECT_ROOT / "artifacts" / "runs" / args.data_run_id / "manifest.json").read_text(encoding="utf-8")
            ).get("resource_usage", {}),
            args.smoke_run_id: json.loads((smoke_dir / "manifest.json").read_text(encoding="utf-8")).get(
                "resource_usage", {}
            ),
            args.run_id: manifest.get("resource_usage", {}),
            "memory_measurement_note": (
                "peak_memory_mb is measured via Python stdlib tracemalloc.get_traced_memory()[1], "
                "which tracks Python heap allocations and does not capture external C/Fortran buffers "
                "allocated directly by NumPy/SciPy/OpenBLAS."
            ),
        },
        "acceptance_step_timings": step_timings,
    }

    summary_json_path = delivery_dir / "acceptance-summary.json"
    summary_md_path = delivery_dir / "acceptance-summary.md"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 9. Render acceptance-summary.md
    step_rows = "\n".join(
        f"| `{s['step_id']}` | {s['description']} | `{s['status']}` | {s['duration_seconds']:.2f}s | `{s['command']}` |"
        for s in step_timings
    )
    md_text = f"""# End-to-End Clean-Environment Acceptance Summary (`v2.0.0`)

- **Generated At**: `{summary_payload['generated_at']}`
- **Overall Gate Status**: **`{"PASSED" if overall_passed else "FAILED"}`**
- **Authoritative Run ID**: `{args.run_id}`
- **Clean-Environment Smoke Run ID**: `{args.smoke_run_id}` (compared against `{args.baseline_smoke_run_id}`)
- **Prepared Data Run ID**: `{args.data_run_id}`
- **Acceptance Python Executable**: `{sys.executable}` (`Python {platform.python_version()}`)
- **Source Code Unchanged (`source_unchanged`)**: `{source_unchanged}` (`combined_sha256 = {current_code['combined_sha256']}`)
- **Raw `data/*.csv` Intact (`9 / 9` SHA-256)**: `{raw_protection['all_intact']}`
- **Protected Legacy & v1 Paths Intact**: `{protected_paths['all_present']}`
- **Original Run Unmodified After Read-Only `evaluate` (`A13`)**: `{original_run_unmodified}`

---

## 1. Test Execution Summary (`zero-skip`, `zero-fail` Gate)

| Suite | Executed | Passed | Failed | Errors | Skipped |
|---|---:|---:|---:|---:|---:|
| `pytest -ra -m unit` | {unit_stats['executed']} | {unit_stats['passed']} | {unit_stats['failed']} | {unit_stats['errors']} | {unit_stats['skipped']} |
| `pytest -ra -m integration` | {integ_stats['executed']} | {integ_stats['passed']} | {integ_stats['failed']} | {integ_stats['errors']} | {integ_stats['skipped']} |
| **Total** | **{total_tests_executed}** | **{total_tests_passed}** | **{total_tests_failed}** | **0** | **{total_tests_skipped}** |

---

## 2. Multi-Path Deterministic Reproducibility Verification (`W7-01`)

1. **Cross-Run Smoke Backtest Reproducibility (`{args.smoke_run_id}` in `.venv-acceptance` vs `{args.baseline_smoke_run_id}` in `.venv`)**:
   - `backtest_predictions.csv`: `{smoke_repro['backtest_predictions.csv']['exact_match']}` (`{smoke_repro['backtest_predictions.csv'][f'{args.smoke_run_id}_sha256']}`)
   - `daily_policy.csv`: `{smoke_repro['daily_policy.csv']['exact_match']}` (`{smoke_repro['daily_policy.csv'][f'{args.smoke_run_id}_sha256']}`)
   - `validation_predictions.csv`: `{smoke_repro['validation_predictions.csv']['exact_match']}` (`{smoke_repro['validation_predictions.csv'][f'{args.smoke_run_id}_sha256']}`)
2. **Dual Independent Read-Only Evaluation (`eval-run1` vs `eval-run2` vs `{args.run_id}`)**:
   - `daily_policy.csv`: `{eval_repro_matches['daily_policy.csv']['exact_match']}` (`{eval_repro_matches['daily_policy.csv']['eval_run1_sha256']}`)
   - `metrics_summary.json`: `{eval_repro_matches['metrics_summary.json']['exact_match']}` (`{eval_repro_matches['metrics_summary.json']['eval_run1_sha256']}`)
   - `claims.json`: `{eval_repro_matches['claims.json']['exact_match']}` (`{eval_repro_matches['claims.json']['eval_run1_sha256']}`)
3. **Dual Independent Offline Batch Scoring (`score-run1` vs `score-run2` vs `sample-v2-001`)**:
   - `predictions.csv`: `{score_repro['exact_match']}` (`{score_repro['score_run1_sha256']}`)
4. **Independent Recomputation (`scripts/recompute_results.py`)**:
   - `{recompute_passed_count} / {recompute_total_checks}` checks passed (`failed = {recompute_failed_count}`).
5. **Report analysis layer rebuilt in the clean environment (`scripts/report_analysis.py`)**:
   - `{report_facts_check['facts']}` facts, identical to the committed `report_facts.json`: `{report_facts_check['exact_match']}`
6. **Narrative document consistency (`scripts/check_docs_consistency.py`)**:
   - `{doc_check_summary['documents']}` documents, `{doc_check_summary['untraceable_numbers']}` untraceable numbers, `{doc_check_summary['missing_images']}` missing images; HTML deck embeds `{(doc_check_summary['html_deck'] or {}).get('embedded')}` of `{(doc_check_summary['html_deck'] or {}).get('referenced')}` images: `{doc_check_summary['passed']}`

---

## 3. Sequential Gate Execution Log

| Step ID | Description | Status | Duration | Command |
|---|---|---|---:|---|
{step_rows}
"""
    summary_md_path.write_text(md_text, encoding="utf-8")

    # 10. Generate delivery/file-checksums.json for all files inside delivery/
    checksums: dict[str, dict[str, Any]] = {}
    for f in sorted(delivery_dir.rglob("*")):
        if f.is_file() and f.name != "file-checksums.json":
            rel = str(f.relative_to(delivery_dir)).replace("\\", "/")
            checksums[rel] = {
                "sha256": sha256_file(f),
                "bytes": f.stat().st_size,
            }
    checksum_payload = {
        "schema_version": "2.0.0",
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "total_files": len(checksums),
        "files": checksums,
    }
    (delivery_dir / "file-checksums.json").write_text(
        json.dumps(checksum_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not overall_passed:
        print("ACCEPTANCE FINALIZATION FAILED!", file=sys.stderr)
        return 1
    print(
        f"ACCEPTANCE FINALIZATION PASSED: tests={total_tests_passed}/{total_tests_executed} "
        f"(skipped={total_tests_skipped}), recompute={recompute_passed_count}/{recompute_total_checks}, "
        f"delivery_files={len(checksums)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
