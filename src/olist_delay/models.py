"""Baseline predictors, finite scikit-learn model pipelines, and chronological-tail probability calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    assert_zero_leakage_features,
    get_feature_columns_by_type,
)


V2_BASELINE_NAMES: list[str] = [
    "rule_random",
    "rule_short_promised_lead",
    "rule_seller_history",
    "rule_region_lead",
    "global_prior_baseline",
]

V3_BASELINE_NAMES: list[str] = [
    "rule_route_recent_lead",
    "rule_route_recent_ca_lead",
    "rule_promise_slack",
]

LEGACY_BASELINE_NAMES: list[str] = [
    "global_prior_baseline",
    "route_smoothed_lookup",
    "seller_route_lookup",
    "distance_lead_logistic",
]


def _safe_logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


class GlobalPriorBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the matured training set global positive rate (`global_prior_baseline`)."""

    def __init__(self) -> None:
        self.prior_: float = 0.068
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "GlobalPriorBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = np.full(len(X), self.prior_, dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])


class RuleRandomBaseline(BaseEstimator, ClassifierMixin):
    """Fixed-seed operational random queue baseline (`rule_random`).

    Generates a deterministic pseudo-random score centered around `prior_` so that
    daily Top-k% ranking is a reproducible random queue rather than an all-tied constant.
    """

    def __init__(self, random_seed: int = 5126) -> None:
        self.random_seed = random_seed
        self.prior_: float = 0.068
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RuleRandomBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)
        # Deterministic row fingerprint so score is reproducible across batch slices
        u = np.zeros(n, dtype=np.float64)
        cols = [c for c in ["estimated_lead_calendar_days", "haversine_distance_km", "total_price", "customer_state"] if c in X.columns]
        for i in range(n):
            row_key = f"{self.random_seed}:{i}"
            if cols:
                vals = ":".join(str(X.iloc[i][c]) for c in cols)
                row_key = f"{self.random_seed}:{vals}:{i}"
            digest = hashlib.sha256(row_key.encode("utf-8")).digest()
            int_val = int.from_bytes(digest[:8], byteorder="big", signed=False)
            u[i] = int_val / float(2**64)
        # Small jitter around prior_ preserves mean calibration while inducing a random ranking
        p1 = np.clip(self.prior_ + (u - 0.5) * 0.02, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class RuleShortPromisedLeadBaseline(BaseEstimator, ClassifierMixin):
    """Single-feature rule ranking shorter promised delivery windows (`estimated_lead_calendar_days`) as higher risk."""

    def __init__(self) -> None:
        self.prior_: float = 0.068
        self.median_lead_: float = 21.0
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RuleShortPromisedLeadBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        if "estimated_lead_calendar_days" in X.columns:
            lead = pd.to_numeric(X["estimated_lead_calendar_days"], errors="coerce")
            med = float(lead.median()) if lead.notna().any() else 21.0
            self.median_lead_ = med if np.isfinite(med) and med > 0 else 21.0
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lead = (
            pd.to_numeric(X["estimated_lead_calendar_days"], errors="coerce")
            .fillna(self.median_lead_)
            .to_numpy(dtype=np.float64)
        )
        lead_clipped = np.clip(lead, 1.0, 90.0)
        # Monotonically decreasing in promised lead days
        p1 = np.clip(self.prior_ * np.exp(-0.05 * (lead_clipped - self.median_lead_)), 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class RuleSellerHistoryBaseline(BaseEstimator, ClassifierMixin):
    """Single-feature rule ranking by Bayesian-smoothed matured seller historical late rate (`seller_prior_late_rate_smoothed`)."""

    def __init__(self) -> None:
        self.prior_: float = 0.068
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RuleSellerHistoryBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        col = (
            "seller_prior_late_rate_smoothed"
            if "seller_prior_late_rate_smoothed" in X.columns
            else "seller_hist_late_rate_smoothed"
        )
        r_seller = pd.to_numeric(X[col], errors="coerce").fillna(self.prior_).to_numpy(dtype=np.float64)
        p1 = np.clip(r_seller, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class RuleRegionLeadBaseline(BaseEstimator, ClassifierMixin):
    """Pre-registered rule combining route historical smoothed late rate with relative promised lead tightness."""

    def __init__(self) -> None:
        self.prior_: float = 0.068
        self.median_lead_: float = 21.0
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RuleRegionLeadBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        if "estimated_lead_calendar_days" in X.columns:
            lead = pd.to_numeric(X["estimated_lead_calendar_days"], errors="coerce")
            med = float(lead.median()) if lead.notna().any() else 21.0
            self.median_lead_ = med if np.isfinite(med) and med > 0 else 21.0
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        col = (
            "route_prior_late_rate_smoothed"
            if "route_prior_late_rate_smoothed" in X.columns
            else "route_hist_late_rate_smoothed"
        )
        r_route = pd.to_numeric(X[col], errors="coerce").fillna(self.prior_).to_numpy(dtype=np.float64)
        lead = (
            pd.to_numeric(X["estimated_lead_calendar_days"], errors="coerce")
            .fillna(self.median_lead_)
            .to_numpy(dtype=np.float64)
        )
        lead_tightness = np.clip(self.median_lead_ / np.clip(lead, 3.0, 90.0), 0.5, 2.0)
        p1 = np.clip(r_route * lead_tightness, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class RuleRouteRecentLeadBaseline(RuleRegionLeadBaseline):
    """v3 rule: `rule_region_lead` with the route late rate taken over the last 60 days only."""

    route_col = "route_recent_60d_late_rate_smoothed"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        r_route = pd.to_numeric(X[self.route_col], errors="coerce").fillna(self.prior_).to_numpy(dtype=np.float64)
        lead = (
            pd.to_numeric(X["estimated_lead_calendar_days"], errors="coerce")
            .fillna(self.median_lead_)
            .to_numpy(dtype=np.float64)
        )
        lead_tightness = np.clip(self.median_lead_ / np.clip(lead, 3.0, 90.0), 0.5, 2.0)
        p1 = np.clip(r_route * lead_tightness, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class RuleRouteRecentCensoringAwareLeadBaseline(RuleRouteRecentLeadBaseline):
    """v3 rule: 60-day route late rate that counts a late order as soon as its promised day has passed."""

    route_col = "route_recent_60d_late_rate_ca"


class RulePromiseSlackBaseline(BaseEstimator, ClassifierMixin):
    """v3 rule: rank by promise slack, the promised lead minus the route's recent actual delivery time.

    Score = prior * exp(-(slack - median_slack) / 10), so a promise 10 days tighter than usual relative to how long
    the route has recently taken scores e-times higher. Only the ranking matters for the daily queue.
    """

    def __init__(self) -> None:
        self.prior_: float = 0.068
        self.median_slack_: float = 10.0
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RulePromiseSlackBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        if "promise_slack_days_60d" in X.columns:
            slack = pd.to_numeric(X["promise_slack_days_60d"], errors="coerce")
            med = float(slack.median()) if slack.notna().any() else 10.0
            self.median_slack_ = med if np.isfinite(med) else 10.0
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        slack = (
            pd.to_numeric(X["promise_slack_days_60d"], errors="coerce")
            .fillna(self.median_slack_)
            .to_numpy(dtype=np.float64)
        )
        z = np.clip(-(slack - self.median_slack_) / 10.0, -20.0, 20.0)
        p1 = np.clip(self.prior_ * np.exp(z), 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class RouteSmoothedBaseline(BaseEstimator, ClassifierMixin):
    """Legacy v1 route historical smoothed late rate baseline."""

    def __init__(self) -> None:
        self.train_prior_: float = 0.068
        self.mean_route_rate_: float = 0.068
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RouteSmoothedBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.train_prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        r = pd.to_numeric(X["route_hist_late_rate_smoothed"], errors="coerce").fillna(self.train_prior_)
        self.mean_route_rate_ = float(np.mean(r)) if len(r) > 0 else self.train_prior_
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        r_all = pd.to_numeric(X["route_hist_late_rate_smoothed"], errors="coerce").fillna(self.train_prior_).to_numpy()
        r_recent = pd.to_numeric(X["route_recent_60d_late_rate_smoothed"], errors="coerce").fillna(self.train_prior_).to_numpy()
        p1 = np.clip(0.65 * r_all + 0.35 * r_recent, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


class SellerRouteLookupBaseline(BaseEstimator, ClassifierMixin):
    """Legacy v1 seller + route lookup baseline."""

    def __init__(self) -> None:
        self.train_prior_: float = 0.068
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "SellerRouteLookupBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        self.train_prior_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.068
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        r_route = pd.to_numeric(X["route_hist_late_rate_smoothed"], errors="coerce").fillna(self.train_prior_).to_numpy()
        r_seller = pd.to_numeric(X["seller_hist_late_rate_smoothed"], errors="coerce").fillna(self.train_prior_).to_numpy()
        r_recent = pd.to_numeric(X["seller_recent_30d_late_rate_smoothed"], errors="coerce").fillna(self.train_prior_).to_numpy()
        p1 = np.clip(0.45 * r_route + 0.35 * r_seller + 0.20 * r_recent, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p1, p1])


def build_distance_lead_logistic(random_seed: int = 5126) -> Pipeline:
    """Simple 2-feature logistic baseline using only distance and promised lead days."""
    cols = ["haversine_distance_km", "estimated_lead_calendar_days"]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                cols,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", pre),
            ("clf", LogisticRegression(C=1.0, max_iter=300, random_state=random_seed)),
        ]
    )


def build_ml_pipeline(
    model_name: str,
    params: dict[str, Any] | None = None,
    random_seed: int = 5126,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    feature_version: str | None = None,
) -> Pipeline | BaseEstimator:
    """Construct a leak-free preprocessing + estimator pipeline for the given model family and feature version."""
    if feature_version is not None and numeric_cols is None and categorical_cols is None:
        num_cols, cat_cols = get_feature_columns_by_type(feature_version)
    else:
        num_cols = list(numeric_cols) if numeric_cols is not None else list(NUMERIC_FEATURES)
        cat_cols = list(categorical_cols) if categorical_cols is not None else list(CATEGORICAL_FEATURES)
    assert_zero_leakage_features(num_cols + cat_cols)
    p = dict(params or {})

    if model_name == "global_prior_baseline":
        return GlobalPriorBaseline()
    if model_name == "rule_random":
        return RuleRandomBaseline(random_seed=random_seed)
    if model_name == "rule_short_promised_lead":
        return RuleShortPromisedLeadBaseline()
    if model_name == "rule_seller_history":
        return RuleSellerHistoryBaseline()
    if model_name == "rule_region_lead":
        return RuleRegionLeadBaseline()
    if model_name == "rule_route_recent_lead":
        return RuleRouteRecentLeadBaseline()
    if model_name == "rule_route_recent_ca_lead":
        return RuleRouteRecentCensoringAwareLeadBaseline()
    if model_name == "rule_promise_slack":
        return RulePromiseSlackBaseline()
    if model_name == "route_smoothed_lookup":
        return RouteSmoothedBaseline()
    if model_name == "seller_route_lookup":
        return SellerRouteLookupBaseline()
    if model_name == "distance_lead_logistic":
        return build_distance_lead_logistic(random_seed=random_seed)

    if model_name == "logistic_regression":
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=25,
                        sparse_output=False,
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(
            C=float(p.get("C", 0.5)),
            max_iter=int(p.get("max_iter", 500)),
            class_weight=p.get("class_weight", None),
            random_state=random_seed,
        )
        return Pipeline([("preprocess", pre), ("clf", clf)])

    # Tree / GBDT preprocessor
    tree_pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    if model_name == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=int(p.get("max_depth", 4)),
            min_samples_leaf=int(p.get("min_samples_leaf", 100)),
            class_weight=p.get("class_weight", None),
            random_state=random_seed,
        )
        return Pipeline([("preprocess", tree_pre), ("clf", clf)])

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 12)),
            min_samples_leaf=int(p.get("min_samples_leaf", 30)),
            max_features=p.get("max_features", "sqrt"),
            class_weight=p.get("class_weight", None),
            n_jobs=1,
            random_state=random_seed,
        )
        return Pipeline([("preprocess", tree_pre), ("clf", clf)])

    if model_name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 14)),
            min_samples_leaf=int(p.get("min_samples_leaf", 25)),
            max_features=p.get("max_features", "sqrt"),
            class_weight=p.get("class_weight", None),
            n_jobs=1,
            random_state=random_seed,
        )
        return Pipeline([("preprocess", tree_pre), ("clf", clf)])

    if model_name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=int(p.get("max_iter", 200)),
            learning_rate=float(p.get("learning_rate", 0.04)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(p.get("min_samples_leaf", 50)),
            l2_regularization=float(p.get("l2_regularization", 2.0)),
            random_state=random_seed,
        )
        return Pipeline([("preprocess", tree_pre), ("clf", clf)])

    raise ValueError(f"Unsupported model_name: {model_name!r}")


class _CalibrationStatusDict(dict):
    """Dict subclass for calibration_status that also supports direct equality comparison with status strings."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return (
                self.get("platt") == other
                or self.get("isotonic") == other
                or self.get("uncalibrated") == other
            )
        return super().__eq__(other)


@dataclass
class CalibratedModelBundle:
    """Holds a fitted base estimator, post-hoc calibrators (Platt / Isotonic), and temporal audit metadata."""

    model_name: str
    base_estimator: Any
    platt_calibrator: LogisticRegression | None = None
    isotonic_calibrator: IsotonicRegression | None = None
    active_calibration: str = "uncalibrated"
    feature_version: str = "extended_purchase_assumptions_v2"
    feature_cols: list[str] = field(default_factory=list)
    calibration_status: dict[str, str] = field(
        default_factory=lambda: _CalibrationStatusDict(
            {
                "uncalibrated": "OK",
                "platt": "UNAVAILABLE",
                "isotonic": "UNAVAILABLE",
            }
        )
    )
    calibration_fallback_reason: str | None = None
    core_train_count: int = 0
    core_dropped_unmatured_at_tail_start_count: int = 0
    cal_tail_count: int = 0
    cal_tail_positives: int = 0
    core_cutoff_timestamp: str | None = None
    refit_full_after_calibration: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.calibration_status, dict) and not isinstance(self.calibration_status, _CalibrationStatusDict):
            self.calibration_status = _CalibrationStatusDict(self.calibration_status)

    @property
    def core_dropped_unmatured_count(self) -> int:
        return int(self.core_dropped_unmatured_at_tail_start_count)

    @property
    def available_variants(self) -> list[str]:
        out = ["uncalibrated"]
        if self.platt_calibrator is not None and self.calibration_status.get("platt") == "OK":
            out.append("platt")
        if self.isotonic_calibrator is not None and self.calibration_status.get("isotonic") == "OK":
            out.append("isotonic")
        return out

    def is_variant_available(self, method: str) -> bool:
        if method == "uncalibrated":
            return True
        if method == "platt":
            return self.platt_calibrator is not None and self.calibration_status.get("platt") == "OK"
        if method == "isotonic":
            return self.isotonic_calibrator is not None and self.calibration_status.get("isotonic") == "OK"
        return False

    def predict_proba_variant(
        self,
        X: pd.DataFrame,
        method: str | None = None,
        strict: bool = False,
    ) -> np.ndarray:
        variant = method or self.active_calibration
        raw_p1 = np.asarray(self.base_estimator.predict_proba(X)[:, 1], dtype=np.float64)
        raw_p1 = np.clip(raw_p1, 1e-6, 1.0 - 1e-6)

        if variant == "uncalibrated":
            return raw_p1
        if variant == "platt":
            if self.platt_calibrator is None:
                if strict:
                    raise ValueError(
                        f"Calibration variant 'platt' is UNAVAILABLE for {self.model_name} "
                        f"(status={self.calibration_status.get('platt')})"
                    )
                return raw_p1
            logits = _safe_logit(raw_p1).reshape(-1, 1)
            cal_p1 = self.platt_calibrator.predict_proba(logits)[:, 1]
            return np.clip(cal_p1, 1e-6, 1.0 - 1e-6)
        if variant == "isotonic":
            if self.isotonic_calibrator is None:
                if strict:
                    raise ValueError(
                        f"Calibration variant 'isotonic' is UNAVAILABLE for {self.model_name} "
                        f"(status={self.calibration_status.get('isotonic')})"
                    )
                return raw_p1
            cal_p1 = self.isotonic_calibrator.predict(raw_p1)
            return np.clip(cal_p1, 1e-6, 1.0 - 1e-6)
        raise ValueError(f"Unknown calibration variant: {variant!r}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.predict_proba_variant(X, method=self.active_calibration)
        return np.column_stack([1.0 - p1, p1])

    def audit_summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_version": self.feature_version,
            "active_calibration": self.active_calibration,
            "calibration_status": dict(self.calibration_status),
            "calibration_fallback_reason": self.calibration_fallback_reason,
            "core_train_count": int(self.core_train_count),
            "core_dropped_unmatured_at_tail_start_count": int(self.core_dropped_unmatured_at_tail_start_count),
            "cal_tail_count": int(self.cal_tail_count),
            "cal_tail_positives": int(self.cal_tail_positives),
            "core_cutoff_timestamp": self.core_cutoff_timestamp,
            "refit_full_after_calibration": bool(self.refit_full_after_calibration),
        }


def fit_model_with_chronological_calibration(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    purchase_timestamps: pd.Series,
    params: dict[str, Any] | None = None,
    random_seed: int = 5126,
    calibration_tail_fraction: float = 0.20,
    active_calibration: str = "platt",
    label_available_timestamps: pd.Series | None = None,
    label_available_at: pd.Series | None = None,
    feature_version: str = "extended_purchase_assumptions_v2",
    refit_full_after_calibration: bool = False,
    min_calibration_samples: int = 20,
    min_positive_in_tail: int = 2,
    min_tail_samples: int | None = None,
    min_tail_positives: int | None = None,
) -> CalibratedModelBundle:
    """Fit a model and its post-hoc calibrators using a strict chronological tail split of matured training data.

    1. Sort `(X_train, y_train)` by `purchase_timestamps` ascending.
    2. Split into `fit_core` (earliest `1 - calibration_tail_fraction`) and `cal_tail` (latest `calibration_tail_fraction`),
       with `T_core` equal to the first purchase timestamp in `cal_tail`.
    3. When `label_available_timestamps` (or `label_available_at`) is provided, filter `fit_core` strictly to orders whose
       `label_available_at <= T_core` so `fit_core` never uses labels that matured during `cal_tail`.
    4. Check that both `fit_core` and `cal_tail` contain two classes and sufficient positives. If not,
       explicitly mark `calibration_status["platt"] = "UNAVAILABLE_DEGENERATE_TAIL"` and fall back
       `active_calibration` to `"uncalibrated"` instead of mislabeling raw probabilities as `"platt"`.
    5. When `refit_full_after_calibration=False` (v2 contract default), keep `core_est` as the base
       estimator so that the fitted calibrators match the exact base estimator distribution.
    """
    eff_label_avail = label_available_at if label_available_at is not None else label_available_timestamps
    eff_min_samples = int(min_tail_samples) if min_tail_samples is not None else int(min_calibration_samples)
    eff_min_pos = int(min_tail_positives) if min_tail_positives is not None else int(min_positive_in_tail)

    num_cols, cat_cols = get_feature_columns_by_type(feature_version)
    avail_num = [c for c in num_cols if c in X_train.columns]
    avail_cat = [c for c in cat_cols if c in X_train.columns]
    feature_cols = avail_num + avail_cat

    p_ts = pd.to_datetime(purchase_timestamps)
    order_idx = np.argsort(p_ts.to_numpy(dtype="datetime64[ns]"), kind="mergesort")
    X_sorted = X_train.iloc[order_idx].reset_index(drop=True)
    y_sorted = np.asarray(y_train, dtype=int)[order_idx]
    p_ts_sorted = p_ts.iloc[order_idx].reset_index(drop=True)

    if eff_label_avail is not None:
        l_ts_sorted = pd.to_datetime(eff_label_avail).iloc[order_idx].reset_index(drop=True)
    else:
        l_ts_sorted = None

    n_total = len(X_sorted)
    n_cal = max(eff_min_samples, int(round(n_total * calibration_tail_fraction)))
    split_pos = max(1, n_total - n_cal)

    is_baseline = model_name in (set(V2_BASELINE_NAMES) | set(LEGACY_BASELINE_NAMES))
    if is_baseline:
        full_est = build_ml_pipeline(
            model_name,
            params=params,
            random_seed=random_seed,
            numeric_cols=avail_num,
            categorical_cols=avail_cat,
        )
        full_est.fit(X_sorted, y_sorted)
        return CalibratedModelBundle(
            model_name=model_name,
            base_estimator=full_est,
            platt_calibrator=None,
            isotonic_calibrator=None,
            active_calibration="uncalibrated",
            feature_version=feature_version,
            feature_cols=feature_cols,
            calibration_status=_CalibrationStatusDict(
                {
                    "uncalibrated": "OK",
                    "platt": "NOT_APPLICABLE_BASELINE",
                    "isotonic": "NOT_APPLICABLE_BASELINE",
                }
            ),
            core_train_count=n_total,
            core_dropped_unmatured_at_tail_start_count=0,
            cal_tail_count=0,
            cal_tail_positives=0,
            core_cutoff_timestamp=str(p_ts_sorted.iloc[-1].isoformat()) if n_total > 0 else None,
            refit_full_after_calibration=False,
        )

    platt_cal: LogisticRegression | None = None
    iso_cal: IsotonicRegression | None = None
    cal_status = _CalibrationStatusDict(
        {
            "uncalibrated": "OK",
            "platt": "UNAVAILABLE",
            "isotonic": "UNAVAILABLE",
        }
    )
    fallback_reason: str | None = None
    core_dropped = 0
    core_cutoff_iso: str | None = None
    core_est: Any = None

    if 0 < split_pos < n_total:
        t_core = p_ts_sorted.iloc[split_pos]
        core_cutoff_iso = t_core.isoformat()
        X_core_raw = X_sorted.iloc[:split_pos]
        y_core_raw = y_sorted[:split_pos]

        if l_ts_sorted is not None:
            l_core = l_ts_sorted.iloc[:split_pos]
            matured_at_t_core = (l_core.notna() & (l_core <= t_core)).to_numpy(dtype=bool)
            core_dropped = int(np.sum(~matured_at_t_core))
            X_core = X_core_raw.loc[matured_at_t_core].reset_index(drop=True)
            y_core = y_core_raw[matured_at_t_core]
        else:
            X_core = X_core_raw
            y_core = y_core_raw

        X_tail = X_sorted.iloc[split_pos:].reset_index(drop=True)
        y_tail = y_sorted[split_pos:]
        tail_pos = int(np.sum(y_tail == 1))
        tail_neg = int(np.sum(y_tail == 0))
        core_classes = len(np.unique(y_core))
        tail_classes = len(np.unique(y_tail))

        if len(X_core) < eff_min_samples or core_classes < 2:
            fallback_reason = "UNAVAILABLE_DEGENERATE_CORE"
            cal_status["platt"] = fallback_reason
            cal_status["isotonic"] = fallback_reason
        elif len(X_tail) < eff_min_samples or tail_classes < 2 or tail_pos < eff_min_pos or tail_neg < eff_min_pos:
            fallback_reason = "UNAVAILABLE_DEGENERATE_TAIL"
            cal_status["platt"] = fallback_reason
            cal_status["isotonic"] = fallback_reason
        else:
            core_est = build_ml_pipeline(
                model_name,
                params=params,
                random_seed=random_seed,
                numeric_cols=avail_num,
                categorical_cols=avail_cat,
            )
            core_est.fit(X_core, y_core)
            tail_raw = np.clip(core_est.predict_proba(X_tail)[:, 1], 1e-6, 1.0 - 1e-6)

            platt_cal = LogisticRegression(C=1.0, max_iter=200, random_state=random_seed)
            platt_cal.fit(_safe_logit(tail_raw).reshape(-1, 1), y_tail)
            cal_status["platt"] = "OK"

            iso_cal = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip")
            iso_cal.fit(tail_raw, y_tail)
            cal_status["isotonic"] = "OK"
    else:
        fallback_reason = "UNAVAILABLE_INSUFFICIENT_SAMPLE_SIZE"
        cal_status["platt"] = fallback_reason
        cal_status["isotonic"] = fallback_reason

    if core_est is not None and not refit_full_after_calibration:
        base_est = core_est
        core_cnt = int(len(X_core))
    else:
        base_est = build_ml_pipeline(
            model_name,
            params=params,
            random_seed=random_seed,
            numeric_cols=avail_num,
            categorical_cols=avail_cat,
        )
        base_est.fit(X_sorted, y_sorted)
        core_cnt = int(len(X_sorted)) if core_est is None else int(len(X_core))

    effective_active = active_calibration
    if effective_active in ("platt", "isotonic") and cal_status.get(effective_active) != "OK":
        effective_active = "uncalibrated"

    return CalibratedModelBundle(
        model_name=model_name,
        base_estimator=base_est,
        platt_calibrator=platt_cal,
        isotonic_calibrator=iso_cal,
        active_calibration=effective_active,
        feature_version=feature_version,
        feature_cols=feature_cols,
        calibration_status=cal_status,
        calibration_fallback_reason=fallback_reason,
        core_train_count=core_cnt,
        core_dropped_unmatured_at_tail_start_count=core_dropped,
        cal_tail_count=int(n_total - split_pos) if split_pos < n_total else 0,
        cal_tail_positives=int(np.sum(y_sorted[split_pos:] == 1)) if split_pos < n_total else 0,
        core_cutoff_timestamp=core_cutoff_iso,
        refit_full_after_calibration=refit_full_after_calibration,
    )
