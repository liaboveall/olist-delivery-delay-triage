"""Run lifecycle, manifest management, non-overwriting output guards, and baseline integrity."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import tracemalloc
import time
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .config import PROJECT_ROOT, resolve_project_path


SGT = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(SGT).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


RAW_DATA_SHA256: dict[str, str] = {
    "olist_customers_dataset.csv": "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "olist_geolocation_dataset.csv": "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
    "olist_orders_dataset.csv": "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "olist_order_items_dataset.csv": "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "olist_order_payments_dataset.csv": "4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0",
    "olist_order_reviews_dataset.csv": "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7",
    "olist_products_dataset.csv": "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "olist_sellers_dataset.csv": "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "product_category_name_translation.csv": "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
}


def compute_code_hash(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    pkg_dir = root / "src" / "olist_delay"
    files = sorted(pkg_dir.glob("*.py"))
    per_file = {}
    combined = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(root)).replace("\\", "/")
        digest = sha256_file(f)
        per_file[rel] = digest
        combined.update(f"{rel}:{digest}\n".encode("utf-8"))
    return {
        "package_version": __version__,
        "combined_sha256": combined.hexdigest(),
        "files": per_file,
    }


def compute_input_hashes(data_dir: Path) -> dict[str, str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    hashes = {}
    for p in sorted(data_dir.glob("*.csv")):
        hashes[p.name] = sha256_file(p)
    if not hashes:
        raise FileNotFoundError(f"No CSV files found in data directory: {data_dir}")
    return hashes


def verify_baseline_protection(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify that all 9 raw dataset CSV files in data/ remain intact."""
    data_dir = root / "data"
    mismatches = []
    checked = 0
    for filename, exp_hash in RAW_DATA_SHA256.items():
        rel_norm = f"data/{filename}"
        check_path = data_dir / filename
        if not check_path.exists():
            mismatches.append({"file": rel_norm, "error": "missing", "checked_path": str(check_path)})
            continue
        actual = sha256_file(check_path)
        if actual != exp_hash:
            mismatches.append({
                "file": rel_norm,
                "expected": exp_hash,
                "actual": actual,
                "checked_path": str(check_path),
            })
        checked += 1

    return {
        "total_expected": len(RAW_DATA_SHA256),
        "total_checked": checked,
        "all_intact": len(mismatches) == 0 and checked == len(RAW_DATA_SHA256),
        "mismatches": mismatches,
    }


PROTECTED_RELATIVE_PATHS = (
    "data",
    "artifacts/review-2026-10-19",
    "artifacts/runs/review-20261019-backtest-v1",
    "artifacts/runs/review-20261019-data-v1",
    "configs/research-v1.json",
    "configs/experiment-v1.json",
)


def assert_writable_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Ensure a target path does not touch any protected raw data, legacy review, or v1 config path."""
    resolved = resolve_project_path(path, root=root)
    for rel in PROTECTED_RELATIVE_PATHS:
        prot = (root / Path(rel)).resolve()
        if resolved == prot or resolved.is_relative_to(prot):
            raise PermissionError(f"Refusing to write to protected path: {resolved} (protected root: {prot})")
    return resolved


def verify_model_artifact_hash(run_dir: Path, model_path: Path) -> str:
    """Refuse to deserialize a model whose SHA-256 differs from the digest bound in the run manifest.

    Pickle files execute code on load, so the digest is checked first. Returns the verified digest.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rel = str(model_path.relative_to(run_dir)).replace("\\", "/")
    entry = manifest.get("artifacts", {}).get(rel)
    expected = entry.get("sha256") if isinstance(entry, dict) else None
    if not expected:
        raise ValueError(f"Manifest of {run_dir.name} has no sha256 for {rel}; refusing to load the model")
    actual = sha256_file(model_path)
    if actual != expected:
        raise ValueError(f"{rel} sha256 {actual} does not match manifest {expected}; refusing to load the model")
    return actual


def atomic_write_json(path: Path, data: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
    assert_writable_path(path, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def collect_environment_info() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "olist_delay": __version__,
    }


class RunContext:
    """Manages non-overwriting run directory, execution log, resource tracking, and atomic manifest."""

    PROTECTED_NAMES = {
        "data",
        "review-20261019-backtest-v1",
        "review-20261019-data-v1",
    }

    def __init__(
        self,
        run_id: str,
        config: dict[str, Any],
        command: str,
        root: Path = PROJECT_ROOT,
        allow_existing_for_update: bool = False,
        parent_run_id: str | None = None,
        rerun_reason: str | None = None,
    ) -> None:
        if (
            not run_id
            or any(sep in run_id for sep in ("/", "\\", ".."))
            or run_id in self.PROTECTED_NAMES
            or run_id.startswith("review-20261019-")
        ):
            raise ValueError(f"Invalid or protected run_id: {run_id!r}")
        self.run_id = run_id
        self.config = config
        self.command = command
        self.root = root
        self.allow_existing_for_update = allow_existing_for_update
        self.parent_run_id = parent_run_id
        self.rerun_reason = rerun_reason

        runs_base = resolve_project_path(config.get("paths", {}).get("runs_dir", "artifacts/runs"), root=root)
        self.run_dir = (runs_base / run_id).resolve()
        assert_writable_path(self.run_dir, root=root)

        self.manifest_path = self.run_dir / "manifest.json"
        self.log_path = self.run_dir / "execution.log"
        self.figures_dir = self.run_dir / "figures"
        self.models_dir = self.run_dir / "models"
        self._start_time = 0.0
        self.manifest: dict[str, Any] = {}

    def __enter__(self) -> "RunContext":
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not self.allow_existing_for_update:
                raise FileExistsError(
                    f"Run directory '{self.run_id}' already exists with status={existing.get('status')!r}. "
                    "Completed or existing runs cannot be overwritten; specify a new --run-id."
                )
            if existing.get("verification_status") == "VERIFIED":
                raise PermissionError(
                    f"Run '{self.run_id}' is already VERIFIED and frozen; cannot mutate in-place."
                )
            self.manifest = existing
        else:
            if self.allow_existing_for_update:
                raise FileNotFoundError(f"Run '{self.run_id}' does not exist at {self.run_dir}")
            if self.run_dir.exists() and any(self.run_dir.iterdir()):
                raise FileExistsError(
                    f"Run directory '{self.run_id}' already exists and is non-empty at {self.run_dir}. "
                    "Specify a new --run-id."
                )

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        tracemalloc.start()

        data_dir = resolve_project_path(self.config.get("paths", {}).get("data_dir", "data"), root=self.root)
        env_info = collect_environment_info()
        code_ver = compute_code_hash(self.root)

        if not self.allow_existing_for_update:
            self.manifest = {
                "schema_version": str(self.config.get("schema_version", "2.0.0")),
                "run_id": self.run_id,
                "status": "RUNNING",
                "verification_status": "NOT_RUN",
                "smoke_only": bool(self.config.get("smoke_only", False)),
                "started_at": now_iso(),
                "completed_at": None,
                "timezone": "Asia/Singapore (UTC+08:00)",
                "command": self.command,
                "parent_run_id": self.parent_run_id,
                "rerun_reason": self.rerun_reason,
                "config_version": self.config.get("config_version"),
                "config_hash": self.config.get("_config_hash"),
                "code_version": code_ver,
                "input_hashes": compute_input_hashes(data_dir),
                "environment": env_info,
                "label_version": self.config.get(
                    "primary_label", self.config.get("label_contract", {}).get("primary_label", "calendar_day")
                ),
                "cohort_definitions": {
                    "path_A": "conditional_delivered_cohort (order_status == 'delivered' and valid key timestamps)",
                    "path_B": "all_scorable_orders_cohort (all orders placed up to decision_at without future status filtering)",
                },
                "feature_version": str(self.config.get("feature_version", "base_v2")),
                "split_protocol": self.config.get("split_contract", {}),
                "random_seed": self.config.get("random_seed", 5126),
                "policy_contract": self.config.get("policy_contract", {}),
                "artifacts": {},
                "resource_usage": {},
                "failure_reason": None,
                "history": [
                    {"timestamp": now_iso(), "action": self.command, "status": "RUNNING"}
                ],
            }
            atomic_write_json(self.run_dir / "config.resolved.json", self.config, root=self.root)
            atomic_write_json(
                self.run_dir / "source_snapshot.json",
                {
                    "run_id": self.run_id,
                    "recorded_at": now_iso(),
                    "config_version": self.config.get("config_version"),
                    "schema_version": self.config.get("schema_version", "2.0.0"),
                    "smoke_only": bool(self.config.get("smoke_only", False)),
                    "config_hash": self.config.get("_config_hash"),
                    "code_version": code_ver,
                },
                root=self.root,
            )
            env_lines = [f"{k}={v}" for k, v in env_info.items()]
            (self.run_dir / "environment.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        else:
            self.manifest.setdefault("history", []).append(
                {"timestamp": now_iso(), "action": self.command, "status": "UPDATING"}
            )

        atomic_write_json(self.manifest_path, self.manifest, root=self.root)
        self.log(f"[{self.command}] Started run_id={self.run_id}")
        return self

    def log(self, message: str) -> None:
        line = f"[{now_iso()}] {message}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def refresh_artifact_hashes(self) -> dict[str, dict[str, Any]]:
        artifacts: dict[str, dict[str, Any]] = {}
        for p in sorted(self.run_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name in ("manifest.json", "manifest.json.tmp") or p.suffix == ".tmp":
                continue
            rel = str(p.relative_to(self.run_dir)).replace("\\", "/")
            artifacts[rel] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
        self.manifest["artifacts"] = artifacts
        return artifacts

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        usage = self.manifest.get("resource_usage", {})
        usage[self.command] = {
            "wall_time_seconds": round(elapsed, 4),
            "peak_memory_mb": round(peak_bytes / (1024 * 1024), 3),
        }
        self.manifest["resource_usage"] = usage

        if exc_type is not None:
            self.manifest["status"] = "FAILED"
            self.manifest["failure_reason"] = f"{exc_type.__name__}: {exc_val}"
            self.manifest["completed_at"] = now_iso()
            self.manifest.setdefault("history", []).append(
                {"timestamp": now_iso(), "action": self.command, "status": "FAILED", "error": str(exc_val)}
            )
            self.log(f"[{self.command}] FAILED: {exc_type.__name__}: {exc_val}")
            self.refresh_artifact_hashes()
            atomic_write_json(self.manifest_path, self.manifest, root=self.root)
            return False

        self.log(
            f"[{self.command}] Finished successfully in {elapsed:.2f}s (peak memory {peak_bytes / (1024 * 1024):.1f} MB)"
        )
        self.refresh_artifact_hashes()
        self.manifest["status"] = "COMPLETED"
        self.manifest["completed_at"] = now_iso()
        self.manifest["failure_reason"] = None
        self.manifest.setdefault("history", []).append(
            {"timestamp": now_iso(), "action": self.command, "status": "COMPLETED"}
        )
        atomic_write_json(self.manifest_path, self.manifest, root=self.root)
        return False
