"""Configuration loading, merging, validation, and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(rel_or_abs: str | Path, root: Path = PROJECT_ROOT) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(cfg)).hexdigest()


ALLOWED_TOP_LEVEL_KEYS = {
    "config_version",
    "schema_version",
    "research_config",
    "smoke_only",
    "description",
    "random_seed",
    "paths",
    "cohort_contract",
    "label_contract",
    "primary_label",
    "sensitivity_label",
    "feature_version",
    "available_feature_versions",
    "geography_contract",
    "history_contract",
    "calibration_contract",
    "split_contract",
    "policy_contract",
    "selection_contract",
    "baseline_rules",
    "model_grid",
    "models",
    "calibration",
    "uncertainty_contract",
    "cost_scenarios",
    "tolerances",
}

ALLOWED_BASELINE_RULES = {
    "rule_random",
    "rule_short_promised_lead",
    "rule_seller_history",
    "rule_region_lead",
    "global_prior_baseline",
    # v3 rules (pre-registered in configs/experiment-v3*.json)
    "rule_route_recent_lead",
    "rule_route_recent_ca_lead",
    "rule_promise_slack",
}

ALLOWED_LABEL_KNOWN_RULES = ("delivered_at", "delivered_or_overdue")


def expand_baseline_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return structured baseline specifications from configuration."""
    rules = list(cfg.get("baseline_rules", []))
    feat_ver = str(cfg.get("feature_version", "base_v2"))
    specs: list[dict[str, Any]] = []
    for r in rules:
        if r not in ALLOWED_BASELINE_RULES:
            raise ValueError(f"Unsupported baseline rule in baseline_rules: {r!r}")
        specs.append(
            {
                "candidate_id": r,
                "model_name": r,
                "model_family": "baseline_rule",
                "feature_version": feat_ver,
                "hyperparameters": {},
                "is_baseline_rule": True,
            }
        )
    return specs


def expand_model_grid(cfg: dict[str, Any], enforce_limit: bool = True) -> list[dict[str, Any]]:
    """Deterministically expand `model_grid` into candidate specifications and enforce `max_fitted_model_configs`."""
    grid = cfg.get("model_grid")
    if not grid:
        return []
    if not isinstance(grid, dict):
        raise ValueError("'model_grid' must be a JSON object.")

    allowed_grid_keys = {"logistic_regression", "decision_tree", "feature_variants"}
    unknown_grid_keys = set(grid.keys()) - allowed_grid_keys
    if unknown_grid_keys:
        raise ValueError(f"Unknown keys in model_grid: {sorted(unknown_grid_keys)}")

    feat_variants = grid.get("feature_variants", [cfg.get("feature_version", "base_v2")])
    if not isinstance(feat_variants, list) or not feat_variants:
        raise ValueError("'model_grid.feature_variants' must be a non-empty list of feature version names.")

    avail_feats = cfg.get("available_feature_versions")
    if isinstance(avail_feats, list) and avail_feats:
        invalid_feats = [fv for fv in feat_variants if fv not in avail_feats]
        if invalid_feats:
            raise ValueError(
                f"model_grid.feature_variants {invalid_feats} not in available_feature_versions {avail_feats}"
            )

    candidates: list[dict[str, Any]] = []

    if "logistic_regression" in grid:
        lr_spec = grid["logistic_regression"]
        c_list = lr_spec.get("C", [1.0])
        cw_list = lr_spec.get("class_weight", [None])
        max_iter = int(lr_spec.get("max_iter", 1500))
        if not isinstance(c_list, list) or not c_list or not isinstance(cw_list, list) or not cw_list:
            raise ValueError("logistic_regression grid requires non-empty lists for 'C' and 'class_weight'.")
        for fv in feat_variants:
            for c_val in c_list:
                if float(c_val) <= 0:
                    raise ValueError(f"logistic_regression C must be > 0, got {c_val}")
                for cw in cw_list:
                    if cw not in (None, "balanced"):
                        raise ValueError(f"Unsupported class_weight for logistic_regression: {cw!r}")
                    cw_tag = "balanced" if cw == "balanced" else "none"
                    c_tag = str(c_val).replace(".", "p")
                    cid = f"lr_C{c_tag}_cw_{cw_tag}__{fv}"
                    hp = {
                        "C": float(c_val),
                        "class_weight": cw,
                        "max_iter": max_iter,
                    }
                    cal_default = str(cfg.get("calibration_contract", {}).get("default_active_method", "platt"))
                    comp_rank = 10 + (0 if str(fv) == "base_v2" else 10)
                    candidates.append(
                        {
                            "candidate_id": cid,
                            "model_name": cid,
                            "model_family": "logistic_regression",
                            "feature_version": str(fv),
                            "hyperparameters": hp,
                            "hyperparams": dict(hp),
                            "hyperparams_json": json.dumps(hp, sort_keys=True, separators=(",", ":")),
                            "calibration_method": cal_default,
                            "complexity_rank": comp_rank,
                            "is_baseline_rule": False,
                        }
                    )

    if "decision_tree" in grid:
        dt_spec = grid["decision_tree"]
        depth_list = dt_spec.get("max_depth", [3])
        leaf_list = dt_spec.get("min_samples_leaf", [150])
        cw_list = dt_spec.get("class_weight", [None])
        if (
            not isinstance(depth_list, list)
            or not depth_list
            or not isinstance(leaf_list, list)
            or not leaf_list
            or not isinstance(cw_list, list)
            or not cw_list
        ):
            raise ValueError(
                "decision_tree grid requires non-empty lists for 'max_depth', 'min_samples_leaf', and 'class_weight'."
            )
        for fv in feat_variants:
            for depth in depth_list:
                if int(depth) < 1 or int(depth) > 10:
                    raise ValueError(f"decision_tree max_depth must be in [1, 10], got {depth}")
                for leaf in leaf_list:
                    if int(leaf) < 1:
                        raise ValueError(f"decision_tree min_samples_leaf must be >= 1, got {leaf}")
                    for cw in cw_list:
                        if cw not in (None, "balanced"):
                            raise ValueError(f"Unsupported class_weight for decision_tree: {cw!r}")
                        cw_tag = "balanced" if cw == "balanced" else "none"
                        cid = f"dt_d{int(depth)}_leaf{int(leaf)}_cw_{cw_tag}__{fv}"
                        hp = {
                            "max_depth": int(depth),
                            "min_samples_leaf": int(leaf),
                            "class_weight": cw,
                        }
                        cal_default = str(cfg.get("calibration_contract", {}).get("default_active_method", "platt"))
                        comp_rank = (20 if int(depth) <= 3 else 25) + (0 if str(fv) == "base_v2" else 10)
                        candidates.append(
                            {
                                "candidate_id": cid,
                                "model_name": cid,
                                "model_family": "decision_tree",
                                "feature_version": str(fv),
                                "hyperparameters": hp,
                                "hyperparams": dict(hp),
                                "hyperparams_json": json.dumps(hp, sort_keys=True, separators=(",", ":")),
                                "calibration_method": cal_default,
                                "complexity_rank": comp_rank,
                                "is_baseline_rule": False,
                            }
                        )

    max_configs = int(cfg.get("selection_contract", {}).get("max_fitted_model_configs", 24))
    if enforce_limit and len(candidates) > max_configs:
        raise ValueError(
            f"Expanded model_grid has {len(candidates)} candidate configurations, exceeding "
            f"selection_contract.max_fitted_model_configs={max_configs}."
        )
    return candidates


def expand_model_candidates(cfg: dict[str, Any], enforce_limit: bool = True) -> list[dict[str, Any]]:
    """Expand model_grid into candidate specifications."""
    return expand_model_grid(cfg, enforce_limit=enforce_limit)


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a JSON object.")

    public_keys = {k for k in cfg.keys() if not k.startswith("_")}
    unknown_keys = public_keys - ALLOWED_TOP_LEVEL_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown_keys)}")

    if "config_version" not in cfg or not isinstance(cfg["config_version"], str) or not cfg["config_version"].strip():
        raise ValueError("Configuration missing required non-empty string 'config_version'.")
    if "random_seed" not in cfg or not isinstance(cfg["random_seed"], int) or isinstance(cfg["random_seed"], bool):
        raise ValueError("Configuration missing integer 'random_seed'.")

    label_contract = cfg.get("label_contract", {})
    if not isinstance(label_contract, dict):
        raise ValueError("'label_contract' must be a JSON object.")
    primary_label = cfg.get("primary_label", label_contract.get("primary_label"))
    if primary_label not in ("calendar_day", "timestamp"):
        raise ValueError(f"Unsupported primary_label: {primary_label!r}")
    sensitivity_label = cfg.get("sensitivity_label", label_contract.get("sensitivity_label", "timestamp"))
    if sensitivity_label not in ("calendar_day", "timestamp"):
        raise ValueError(f"Unsupported sensitivity_label: {sensitivity_label!r}")
    known_rule = label_contract.get("label_known_rule", "delivered_at")
    if known_rule not in ALLOWED_LABEL_KNOWN_RULES:
        raise ValueError(
            f"Unsupported label_contract.label_known_rule: {known_rule!r}; expected one of {ALLOWED_LABEL_KNOWN_RULES}"
        )

    split_contract = cfg.get("split_contract", {})
    if not isinstance(split_contract, dict) or "validation_folds" not in split_contract or "backtest" not in split_contract:
        raise ValueError("Configuration missing required 'split_contract' with 'validation_folds' and 'backtest'.")
    val_folds = split_contract.get("validation_folds", [])
    if not isinstance(val_folds, list) or len(val_folds) == 0:
        raise ValueError("'split_contract.validation_folds' must be a non-empty list.")
    for fold in val_folds:
        if not isinstance(fold, dict) or not {"fold_id", "train_at", "val_start", "val_end"}.issubset(fold.keys()):
            raise ValueError(f"Invalid validation fold specification: {fold!r}")
        if not (str(fold["train_at"]) <= str(fold["val_start"]) < str(fold["val_end"])):
            raise ValueError(f"Non-chronological validation fold boundaries in {fold['fold_id']!r}")

    bt = split_contract.get("backtest", {})
    if not isinstance(bt, dict) or not {"fold_id", "train_at", "test_start", "test_end"}.issubset(bt.keys()):
        raise ValueError("Invalid 'split_contract.backtest' specification.")
    if not (str(bt["train_at"]) <= str(bt["test_start"]) < str(bt["test_end"])):
        raise ValueError("Non-chronological backtest boundaries in split_contract.backtest.")

    geo = cfg.get("geography_contract", {})
    if geo:
        for bound_key in ("brazil_lat_bounds", "brazil_lng_bounds"):
            if bound_key in geo:
                b = geo[bound_key]
                if not isinstance(b, list) or len(b) != 2 or float(b[0]) >= float(b[1]):
                    raise ValueError(f"Invalid geography_contract.{bound_key}: {b!r}")

    cal = cfg.get("calibration_contract", {})
    if cal:
        core_f = float(cal.get("core_fraction", 0.80))
        tail_f = float(cal.get("tail_fraction", 0.20))
        if core_f <= 0 or tail_f <= 0 or abs((core_f + tail_f) - 1.0) > 1e-6:
            raise ValueError(
                f"calibration_contract core_fraction ({core_f}) and tail_fraction ({tail_f}) must be positive and sum to 1.0."
            )

    pol = cfg.get("policy_contract", {})
    if pol:
        pb = float(pol.get("primary_budget", 0.10))
        if not (0.0 < pb < 1.0):
            raise ValueError(f"policy_contract.primary_budget must be in (0, 1), got {pb}")
        qr = pol.get("quota_rounding", "floor")
        if qr not in ("floor", "round", "ceil"):
            raise ValueError(f"Unsupported policy_contract.quota_rounding: {qr!r}")

    if "baseline_rules" in cfg:
        if not isinstance(cfg["baseline_rules"], list) or len(cfg["baseline_rules"]) == 0:
            raise ValueError("'baseline_rules' must be a non-empty list.")
        expand_baseline_specs(cfg)

    if "model_grid" in cfg:
        enforce_cap = str(cfg.get("config_version", "")) != "experiment-v1.0"
        expand_model_grid(cfg, enforce_limit=enforce_cap)


def load_config(config_path: str | Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = resolve_project_path(config_path, root=root)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config structure in {path}")
    validate_raw_keys = {k for k in raw.keys() if not k.startswith("_")}
    unknown_raw = validate_raw_keys - ALLOWED_TOP_LEVEL_KEYS
    if unknown_raw:
        raise ValueError(f"Unknown top-level configuration keys in {path.name}: {sorted(unknown_raw)}")

    # Merge base research_config if referenced by experiment config
    if "research_config" in raw:
        base_path = resolve_project_path(raw["research_config"], root=root)
        if not base_path.exists():
            raise FileNotFoundError(f"Referenced research_config not found: {base_path}")
        base_cfg = json.loads(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_cfg, dict):
            raise ValueError(f"Invalid base config structure in {base_path}")
        validate_config(base_cfg)
        merged = dict(base_cfg)
        for k, v in raw.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                sub = dict(merged[k])
                sub.update(v)
                merged[k] = sub
            else:
                merged[k] = v
    else:
        merged = dict(raw)

    validate_config(merged)
    merged["_resolved_from"] = str(path.relative_to(root) if path.is_relative_to(root) else path).replace("\\", "/")
    merged["_config_hash"] = compute_config_hash({k: v for k, v in merged.items() if not k.startswith("_")})
    return merged


def inspect_config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a structured summary of a loaded configuration for `python -m olist_delay inspect-config`."""
    enforce_cap = str(cfg.get("config_version", "")) != "experiment-v1.0"
    baselines = expand_baseline_specs(cfg) if "baseline_rules" in cfg else []
    model_candidates = expand_model_grid(cfg, enforce_limit=enforce_cap) if "model_grid" in cfg else []
    max_configs = int(cfg.get("selection_contract", {}).get("max_fitted_model_configs", 24))
    split_c = cfg.get("split_contract", {})
    return {
        "config_version": cfg.get("config_version"),
        "schema_version": cfg.get("schema_version", "1.0.0"),
        "smoke_only": bool(cfg.get("smoke_only", False)),
        "resolved_from": cfg.get("_resolved_from"),
        "config_hash": cfg.get("_config_hash"),
        "random_seed": cfg.get("random_seed"),
        "primary_label": cfg.get("primary_label", cfg.get("label_contract", {}).get("primary_label")),
        "sensitivity_label": cfg.get("sensitivity_label", cfg.get("label_contract", {}).get("sensitivity_label")),
        "feature_version": cfg.get("feature_version", "base_v2"),
        "available_feature_versions": cfg.get("available_feature_versions", ["base_v2"]),
        "baseline_rules_count": len(baselines),
        "baseline_rules": [b["candidate_id"] for b in baselines],
        "expanded_model_candidates_count": len(model_candidates),
        "max_fitted_model_configs": max_configs,
        "within_model_budget": len(model_candidates) <= max_configs,
        "expanded_model_candidates": [
            {
                "candidate_id": c["candidate_id"],
                "model_family": c["model_family"],
                "feature_version": c["feature_version"],
                "hyperparameters": c["hyperparameters"],
            }
            for c in model_candidates
        ],
        "validation_folds": [f["fold_id"] for f in split_c.get("validation_folds", [])],
        "selection_at": split_c.get("selection_at"),
        "backtest": split_c.get("backtest", {}),
        "policy_contract": cfg.get("policy_contract", {}),
        "calibration_contract": cfg.get("calibration_contract", {}),
    }

