"""Report analysis layer built read-only on top of a frozen, verified formal run.

The formal pipeline (`python -m olist_delay backtest`) is frozen together with its source-code hash, so the
narrative documents are fed from this separate layer instead of from hand-copied numbers. It reads:

- the formal run's per-order predictions and summary tables (`artifacts/runs/<run-id>/`), and
- the raw Olist CSVs in `data/` (SHA-256 checked against `RAW_DATA_SHA256`),

and writes an isolated output directory containing

- `report_facts.json`: every number the final report, slides, Q&A and briefs are allowed to quote,
- `tables/*.csv`: the tables the report reproduces,
- `figures/*.png`: the report figures (also copied to `docs/figures/`).

Nothing under `artifacts/runs/` is modified. Two analyses are explicitly post hoc (designed after the backtest
window had been inspected) and are labelled as such in the facts file: the Milestone-2 shipping-limit signal and
the paired comparison against `rule_short_promised_lead`. The layer's own weekly block bootstrap (2,000 draws)
adds paired comparisons the pipeline does not compute; the pipeline's intervals are reported as `boot.formal.*`.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from olist_delay.artifacts import RAW_DATA_SHA256, sha256_file  # noqa: E402
from olist_delay.features import get_feature_columns  # noqa: E402
from olist_delay.policy import allocate_daily_alerts  # noqa: E402

ANALYSIS_VERSION = "report-analysis-v1"
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]
SELECTED = "rule_region_lead"
SHORT_LEAD = "rule_short_promised_lead"
LEAD_BANDS = [(-np.inf, 10, "<=10d"), (10, 20, "11-20d"), (20, 30, "21-30d"), (30, np.inf, ">30d")]
PERIOD_VALIDATION = ("2018-01-01", "2018-05-01")  # purchase months of the four validation folds
TIMELINE = ("2017-01-01", "2018-09-01")

# Chart tokens (reference palette, light mode; first three categorical slots validated all-pairs).
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"


# --------------------------------------------------------------------------------------------- facts registry
class Facts:
    """Ordered registry of every number the narrative documents may quote."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def add(self, key: str, value: Any, source: str, note: str = "", post_hoc: bool = False) -> Any:
        if key in self.items:
            raise KeyError(f"Duplicate fact key: {key}")
        if isinstance(value, (np.integer, np.bool_)):
            value = value.item()
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 6)
        entry: dict[str, Any] = {"value": value, "source": source}
        if note:
            entry["note"] = note
        if post_hoc:
            entry["post_hoc"] = True
        self.items[key] = entry
        return value


def _require_close(label: str, got: float, expected: float, tol: float = 1e-6) -> None:
    if abs(float(got) - float(expected)) > tol:
        raise AssertionError(f"{label}: recomputed {got} != frozen artifact {expected}")


# --------------------------------------------------------------------------------------------- loading
def load_inputs(run_dir: Path, data_dir: Path) -> dict[str, Any]:
    for name, digest in RAW_DATA_SHA256.items():
        actual = sha256_file(data_dir / name)
        if actual != digest:
            raise ValueError(f"Raw data hash mismatch for {name}: {actual}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("smoke_only") or manifest.get("verification_status") != "VERIFIED":
        raise ValueError("Report analysis requires a non-smoke run whose manifest is VERIFIED")

    bt = pd.read_csv(run_dir / "backtest_predictions.csv", low_memory=False)
    val = pd.read_csv(run_dir / "validation_predictions.csv", low_memory=False)
    for df in (bt, val):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
        df["in_cohort_a"] = df["in_cohort_a"].astype(str).str.lower().isin(["true", "1"])

    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})
    audit = pd.read_csv(run_dir / "cohort_audit.csv", low_memory=False)
    audit["order_purchase_timestamp"] = pd.to_datetime(audit["order_purchase_timestamp"])
    audit["in_cohort_a"] = audit["in_cohort_a"].astype(str).str.lower().isin(["true", "1"])

    cand = json.loads((run_dir / "candidate_manifest.json").read_text(encoding="utf-8-sig"))
    candidates = [c["candidate_id"] for c in cand["baselines"]] + [c["candidate_id"] for c in cand["model_candidates"]]
    families = {c["candidate_id"]: c["model_family"] for c in cand["model_candidates"]}
    families.update({c["candidate_id"]: "rule" for c in cand["baselines"]})

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "bt": bt,
        "val": val,
        "orders": orders,
        "items": items,
        "sellers": sellers,
        "customers": customers,
        "audit": audit,
        "candidates": candidates,
        "families": families,
        "config": json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8-sig")),
        "claims": json.loads((run_dir / "claims.json").read_text(encoding="utf-8-sig"))["claims"],
        "selection": json.loads((run_dir / "selection_record.json").read_text(encoding="utf-8-sig")),
        "metrics": json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8-sig")),
        "uncertainty": json.loads((run_dir / "uncertainty_summary.json").read_text(encoding="utf-8-sig")),
        "split_summary": pd.read_csv(run_dir / "split_summary.csv"),
        "ex_post": pd.read_csv(run_dir / "ex_post_stage_summary.csv"),
        "dq": json.loads((run_dir / "data_quality_summary.json").read_text(encoding="utf-8-sig")),
    }


def order_attributes(orders: pd.DataFrame, items: pd.DataFrame, sellers: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Per-order route and Milestone-2 attributes rebuilt from raw tables with the pipeline's definitions."""
    it = items.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    # Primary seller = item with highest price, then freight, then lowest item id (src/olist_delay/data.py).
    primary = it.sort_values(
        ["order_id", "price", "freight_value", "order_item_id"], ascending=[True, False, False, True], kind="mergesort"
    ).drop_duplicates("order_id", keep="first")[["order_id", "seller_state"]]
    limit = it.groupby("order_id", as_index=False)["shipping_limit_date"].max()
    out = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    out = out.merge(primary, on="order_id", how="left").merge(limit, on="order_id", how="left")
    out["same_state"] = out["seller_state"] == out["customer_state"]
    out["lead_days"] = (
        out["order_estimated_delivery_date"].dt.normalize() - out["order_purchase_timestamp"].dt.normalize()
    ).dt.days
    return out


def lead_band(days: pd.Series) -> pd.Series:
    labels = pd.Series(pd.NA, index=days.index, dtype="object")
    for lo, hi, name in LEAD_BANDS:
        labels[(days > lo) & (days <= hi)] = name
    return labels


# --------------------------------------------------------------------------------------------- policy helpers
def daily_floor_selected(df: pd.DataFrame, score_col: str, k: float) -> np.ndarray:
    """Boolean mask of daily `floor(k * N_d)` alerts, using the production allocator and tie-break."""
    alloc = allocate_daily_alerts(
        df=df[["order_id", "order_purchase_timestamp", "order_estimated_delivery_date"]].assign(
            order_purchase_timestamp=df["order_purchase_timestamp"].astype(str)
        ),
        scores=df[score_col].to_numpy(dtype=float),
        budget_k=k,
        exclude_known_delivered_before_decision=False,
        separate_already_overdue_before_decision=True,
    )
    sel = dict(zip(alloc["order_id"].astype(str), alloc["selected_top10"].astype(bool)))
    return df["order_id"].astype(str).map(sel).fillna(False).to_numpy(dtype=bool)


def policy_metrics(y: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    alerts = int(selected.sum())
    tp = int((y[selected] == 1).sum())
    pos = int(y.sum())
    prec = tp / alerts if alerts else 0.0
    return {
        "alerts": alerts,
        "tp": tp,
        "fp": alerts - tp,
        "positives": pos,
        "precision": prec,
        "recall": tp / pos if pos else 0.0,
        "lift": prec / (pos / len(y)) if pos else 0.0,
    }


# --------------------------------------------------------------------------------------------- analyses
def add_claims(f: Facts, inp: dict[str, Any]) -> dict[str, Any]:
    claims = {c["claim_id"]: c["value"] for c in inp["claims"]}
    for cid, value in claims.items():
        f.add(f"claim.{cid}", value, "claims.json")
    f.add(
        "claim.april_unmatured_late_share",
        claims["C05_VAL_APRIL_UNMATURED_LATE_COUNT"] / claims["C04_VAL_APRIL_UNMATURED_AT_SELECTION"],
        "claims.json (C05 / C04)",
    )
    return claims


def population_and_motivation(f: Facts, inp: dict[str, Any]) -> None:
    dq = inp["dq"]
    audit = inp["audit"]
    a = audit[audit["in_cohort_a"]]
    f.add("pop.orders_with_items", int(audit["has_items"].astype(str).str.lower().isin(["true", "1"]).sum()), "cohort_audit.csv")
    f.add("pop.orders_without_items", int((~audit["has_items"].astype(str).str.lower().isin(["true", "1"])).sum()), "cohort_audit.csv")
    f.add("pop.cohort_a_late_calendar", int(a["late_calendar_day"].sum()), "cohort_audit.csv")
    f.add("pop.cohort_a_late_calendar_rate", float(a["late_calendar_day"].mean()), "cohort_audit.csv")
    f.add("pop.cohort_a_late_timestamp", int(a["late_timestamp"].sum()), "cohort_audit.csv")
    f.add("pop.cohort_a_late_timestamp_rate", float(a["late_timestamp"].mean()), "cohort_audit.csv")
    f.add("pop.same_day_after_midnight", int(a["late_timestamp"].sum() - a["late_calendar_day"].sum()), "cohort_audit.csv")
    f.add("pop.cohort_a_on_time", int(len(a) - a["late_calendar_day"].sum()), "cohort_audit.csv")
    f.add("pop.cohort_a_on_time_rate", float(1 - a["late_calendar_day"].mean()), "cohort_audit.csv")
    n_all = len(audit)
    has_items = audit["has_items"].astype(str).str.lower().isin(["true", "1"])
    f.add("pop.cohort_a_share", len(a) / n_all, "cohort_audit.csv")
    f.add("pop.unknown_share", float((~audit["in_cohort_a"]).mean()), "cohort_audit.csv")
    f.add("pop.without_items_share", float((~has_items).mean()), "cohort_audit.csv")
    f.add("pop.unknown_with_items", int((~audit["in_cohort_a"] & has_items).sum()), "cohort_audit.csv")
    f.add("pop.unknown_with_items_share", float((~audit["in_cohort_a"] & has_items).mean()), "cohort_audit.csv")
    lr = dq["label_reconciliation"]["backtest_window"]
    f.add("pop.backtest_same_day_after_midnight", int(lr["same_day_after_midnight_count"]), "data_quality_summary.json")
    status = audit[~audit["in_cohort_a"]]["order_status"].value_counts()
    for s, n in status.items():
        f.add(f"pop.unknown_status.{s}", int(n), "cohort_audit.csv")
    for k, v in dq["anomaly_counts"].items():
        f.add(f"dq.{k}", int(v), "data_quality_summary.json")
    for side in ("customer", "seller"):
        for k, v in dq["geo_fallback_counts"][side].items():
            f.add(f"dq.geo.{side}.{k}", int(v), "data_quality_summary.json")
    f.add("dq.geo.out_of_bounds_rows_dropped", int(dq["geo_fallback_counts"]["out_of_bounds_geolocation_rows_dropped"]), "data_quality_summary.json")
    f.add("pop.first_purchase", str(audit["order_purchase_timestamp"].min().date()), "cohort_audit.csv")
    f.add("pop.last_purchase", str(audit["order_purchase_timestamp"].max().date()), "cohort_audit.csv")

    ex = inp["ex_post"].set_index("label_value")
    for lbl, tag in [(0, "on_time"), (1, "late")]:
        row = ex.loc[lbl]
        f.add(f"stage.{tag}.n", int(row["order_count"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.review_mean", float(row["post_delivery_review_score_mean"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.one_star_rate", float(row["one_star_review_rate"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.approval_hours_mean", float(row["approval_hours_mean"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.approval_hours_median", float(row["approval_hours_median"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.seller_days_mean", float(row["seller_to_carrier_days_mean"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.seller_days_median", float(row["seller_to_carrier_days_median"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.carrier_days_mean", float(row["carrier_to_customer_days_mean"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.carrier_days_median", float(row["carrier_to_customer_days_median"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.total_days_mean", float(row["total_fulfillment_days_mean"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.total_days_median", float(row["total_fulfillment_days_median"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.promised_days_mean", float(row["promised_lead_calendar_days_mean"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.shipping_limit_breach_rate", float(row["seller_exceeded_shipping_limit_rate"]), "ex_post_stage_summary.csv")
        f.add(f"stage.{tag}.shipping_limit_breach_n", int(row["seller_exceeded_shipping_limit_count"]), "ex_post_stage_summary.csv")
    on, late = ex.loc[0], ex.loc[1]
    f.add("stage.one_star_ratio", float(late["one_star_review_rate"] / on["one_star_review_rate"]), "ex_post_stage_summary.csv")
    f.add("stage.review_gap", float(on["post_delivery_review_score_mean"] - late["post_delivery_review_score_mean"]), "ex_post_stage_summary.csv")
    gap_seller = late["seller_to_carrier_days_mean"] - on["seller_to_carrier_days_mean"]
    gap_carrier = late["carrier_to_customer_days_mean"] - on["carrier_to_customer_days_mean"]
    gap_total = late["total_fulfillment_days_mean"] - on["total_fulfillment_days_mean"]
    f.add("stage.gap_seller_days", float(gap_seller), "ex_post_stage_summary.csv")
    f.add("stage.gap_carrier_days", float(gap_carrier), "ex_post_stage_summary.csv")
    f.add("stage.gap_total_days", float(gap_total), "ex_post_stage_summary.csv")
    f.add("stage.gap_carrier_share", float(gap_carrier / gap_total), "ex_post_stage_summary.csv")
    f.add("stage.gap_seller_share", float(gap_seller / gap_total), "ex_post_stage_summary.csv")
    f.add("stage.breach_enrichment", float(late["seller_exceeded_shipping_limit_rate"] / on["seller_exceeded_shipping_limit_rate"]), "ex_post_stage_summary.csv")
    breach_tp = int(late["seller_exceeded_shipping_limit_count"])
    breach_fp = int(on["seller_exceeded_shipping_limit_count"])
    f.add("stage.breach_precision_full_sample", breach_tp / (breach_tp + breach_fp), "ex_post_stage_summary.csv")


def raw_table_facts(f: Facts, inp: dict[str, Any], data_dir: Path) -> None:
    """Row counts and structural facts quoted by the data card."""
    for name in RAW_DATA_SHA256:
        n = len(pd.read_csv(data_dir / name, usecols=[0]))
        f.add(f"raw.rows.{name.removesuffix('.csv')}", n, name)
    geo = pd.read_csv(data_dir / "olist_geolocation_dataset.csv")
    lat_ok = geo["geolocation_lat"].between(-33.8, 5.4) & geo["geolocation_lng"].between(-74.1, -28.8)
    dedup = geo.drop_duplicates()
    dedup_ok = dedup["geolocation_lat"].between(-33.8, 5.4) & dedup["geolocation_lng"].between(-74.1, -28.8)
    f.add("raw.geo.exact_duplicates", int(geo.duplicated().sum()), "olist_geolocation_dataset.csv")
    f.add("raw.geo.out_of_bounds_raw", int((~lat_ok).sum()), "olist_geolocation_dataset.csv")
    f.add("raw.geo.unique_rows", len(dedup), "olist_geolocation_dataset.csv")
    f.add("raw.geo.out_of_bounds_unique", int((~dedup_ok).sum()), "olist_geolocation_dataset.csv")
    f.add("raw.geo.valid_unique_rows", int(dedup_ok.sum()), "olist_geolocation_dataset.csv")
    f.add("raw.geo.valid_zip_prefixes", int(dedup.loc[dedup_ok, "geolocation_zip_code_prefix"].nunique()), "olist_geolocation_dataset.csv")
    items = inp["items"]
    sellers_per_order = items.groupby("order_id")["seller_id"].nunique()
    f.add("raw.items.multi_seller_orders", int((sellers_per_order > 1).sum()), "olist_order_items_dataset.csv")
    f.add("raw.items.single_seller_orders", int((sellers_per_order == 1).sum()), "olist_order_items_dataset.csv")
    f.add("raw.items.max_sellers_per_order", int(sellers_per_order.max()), "olist_order_items_dataset.csv")
    f.add("raw.items.order_seller_pairs", len(items[["order_id", "seller_id"]].drop_duplicates()), "olist_order_items_dataset.csv")
    audit = inp["audit"]
    has = audit["has_items"].astype(str).str.lower().isin(["true", "1"])
    unk = audit[~audit["in_cohort_a"]]
    for (status, with_items), n in unk.groupby([unk["order_status"], has[~audit["in_cohort_a"]]]).size().items():
        f.add(f"pop.unknown_status_items.{status}.{'with' if with_items else 'without'}", int(n), "cohort_audit.csv")


def monthly_timeline(f: Facts, inp: dict[str, Any], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    audit = inp["audit"]
    a = audit[audit["in_cohort_a"] & (audit["order_purchase_timestamp"] >= TIMELINE[0]) & (audit["order_purchase_timestamp"] < TIMELINE[1])]
    m = a.groupby(a["order_purchase_timestamp"].dt.to_period("M")).agg(
        orders=("late_calendar_day", "size"), late=("late_calendar_day", "sum")
    )
    m["late_rate"] = m["late"] / m["orders"]
    m.index = m.index.astype(str)
    for month, row in m.iterrows():
        f.add(f"month.{month}.orders", int(row["orders"]), "cohort_audit.csv (eventual labels)")
        f.add(f"month.{month}.late_rate", float(row["late_rate"]), "cohort_audit.csv (eventual labels)")
    p2017 = a[a["order_purchase_timestamp"] < "2018-01-01"]
    f.add("month.2017.late_rate", float(p2017["late_calendar_day"].mean()), "cohort_audit.csv")
    ss = inp["split_summary"].set_index("split_id")
    tables["monthly_late_rate"] = m.reset_index(names="purchase_month")
    return m


def split_facts(f: Facts, inp: dict[str, Any]) -> None:
    ss = inp["split_summary"]
    for _, r in ss.iterrows():
        sid = r["split_id"]
        for col in r.index:
            if col == "split_id":
                continue
            v = r[col]
            if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and np.isfinite(v)):
                f.add(f"split.{sid}.{col}", v, "split_summary.csv")


def leaderboards(f: Facts, inp: dict[str, Any], tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    val, bt = inp["val"], inp["bt"]
    candidates, families = inp["candidates"], inp["families"]
    rows = []
    for cid in candidates:
        col = f"score_{cid}"
        rec, prec, aps, rocs = [], [], [], []
        per_fold = {}
        for sid, g in val.groupby("split_id"):
            g = g[g["in_cohort_a"] & g["late"].notna()]
            y = g["late"].astype(int).to_numpy()
            pm = policy_metrics(y, daily_floor_selected(g, col, 0.10))
            rec.append(pm["recall"])
            prec.append(pm["precision"])
            aps.append(average_precision_score(y, g[col]))
            rocs.append(roc_auc_score(y, g[col]))
            per_fold[sid] = pm["recall"]
        a = bt[bt["in_cohort_a"]]
        yb = a["late_calendar_day"].astype(int).to_numpy()
        pb = policy_metrics(yb, daily_floor_selected(a, col, 0.10))
        rows.append(
            {
                "candidate_id": cid,
                "family": families[cid],
                "val_mean_daily_top10_recall": float(np.mean(rec)),
                "val_mean_daily_top10_precision": float(np.mean(prec)),
                "val_min_fold_recall": float(np.min(rec)),
                "val_mean_ap": float(np.mean(aps)),
                "val_mean_roc_auc": float(np.mean(rocs)),
                **{f"val_recall_{k}": v for k, v in per_fold.items()},
                "bt_ap": float(average_precision_score(yb, a[col])),
                "bt_roc_auc": float(roc_auc_score(yb, a[col])),
                "bt_daily_top10_alerts": pb["alerts"],
                "bt_daily_top10_tp": pb["tp"],
                "bt_daily_top10_recall": pb["recall"],
                "bt_daily_top10_precision": pb["precision"],
            }
        )
    lb = pd.DataFrame(rows).sort_values("val_mean_daily_top10_recall", ascending=False).reset_index(drop=True)
    lb.insert(0, "val_rank", np.arange(1, len(lb) + 1))
    lb["bt_rank_by_recall"] = lb["bt_daily_top10_recall"].rank(ascending=False, method="min").astype(int)
    tables["leaderboard"] = lb

    sel = inp["selection"]
    sel_row = lb.set_index("candidate_id").loc[SELECTED]
    _require_close("selected val recall", sel_row["val_mean_daily_top10_recall"], sel["selected_candidate"]["val_mean_daily_top10_recall"])
    best_ml = sel["best_ml_candidate"]["candidate_id"]
    _require_close("best ML val recall", lb.set_index("candidate_id").loc[best_ml, "val_mean_daily_top10_recall"], sel["best_ml_candidate"]["val_mean_daily_top10_recall"])

    trees = lb[lb["family"] == "decision_tree"]
    best_tree = trees.iloc[0]["candidate_id"]
    best_tree_bt = trees.sort_values("bt_ap", ascending=False).iloc[0]["candidate_id"]
    keyed = lb.set_index("candidate_id")
    named = {
        "selected": SELECTED,
        "best_lr": best_ml,
        "best_tree": best_tree,
        "best_tree_bt": best_tree_bt,
        "short_lead": SHORT_LEAD,
        "seller_history": "rule_seller_history",
        "random": "rule_random",
        "global_prior": "global_prior_baseline",
    }
    for tag, cid in named.items():
        r = keyed.loc[cid]
        f.add(f"lb.{tag}.id", cid, "leaderboard.csv")
        f.add(f"lb.{tag}.val_rank", int(r["val_rank"]), "leaderboard.csv")
        for col in [
            "val_mean_daily_top10_recall",
            "val_mean_daily_top10_precision",
            "val_min_fold_recall",
            "val_mean_ap",
            "val_mean_roc_auc",
            "bt_ap",
            "bt_roc_auc",
            "bt_daily_top10_tp",
            "bt_daily_top10_recall",
            "bt_daily_top10_precision",
            "bt_rank_by_recall",
        ]:
            f.add(f"lb.{tag}.{col}", r[col], "leaderboard.csv (recomputed from per-order predictions)")
        for sid in sorted(val["split_id"].unique()):
            f.add(f"lb.{tag}.val_recall.{sid}", r[f"val_recall_{sid}"], "leaderboard.csv")
    for _, r in lb.iterrows():
        for col in ["val_rank", "val_mean_daily_top10_recall", "val_min_fold_recall", "val_mean_ap", "val_mean_roc_auc", "bt_daily_top10_tp", "bt_daily_top10_recall", "bt_ap", "bt_roc_auc", "bt_rank_by_recall"]:
            f.add(f"lb.all.{r['candidate_id']}.{col}", r[col], "leaderboard.csv")
    ml = lb[lb["family"] != "rule"]
    f.add("lb.n_candidates", len(lb), "candidate_manifest.json")
    f.add("lb.n_ml", len(ml), "candidate_manifest.json")
    f.add("lb.n_rules", int((lb["family"] == "rule").sum()), "candidate_manifest.json")
    f.add("lb.n_lr", int((lb["family"] == "logistic_regression").sum()), "candidate_manifest.json")
    f.add("lb.n_tree", int((lb["family"] == "decision_tree").sum()), "candidate_manifest.json")
    f.add("lb.ml_bt_recall_max", float(ml["bt_daily_top10_recall"].max()), "leaderboard.csv")
    f.add("lb.ml_bt_ap_max", float(ml["bt_ap"].max()), "leaderboard.csv")
    f.add("lb.val_gap_selected_minus_best_ml", float(sel_row["val_mean_daily_top10_recall"] - keyed.loc[best_ml, "val_mean_daily_top10_recall"]), "leaderboard.csv")
    f.add("lb.ml_beating_selected_on_bt_recall", int((ml["bt_daily_top10_recall"] > sel_row["bt_daily_top10_recall"]).sum()), "leaderboard.csv")
    return named


def capacity_frontier(f: Facts, inp: dict[str, Any], tables: dict[str, pd.DataFrame], named: dict[str, str]) -> pd.DataFrame:
    bt = inp["bt"]
    a = bt[bt["in_cohort_a"]]
    y = a["late_calendar_day"].astype(int).to_numpy()
    yt = a["late_timestamp"].astype(int).to_numpy()
    rows = []
    for tag in ["selected", "short_lead", "best_lr", "random"]:
        cid = named[tag]
        for k in BUDGETS:
            sel = daily_floor_selected(a, f"score_{cid}", k)
            pm = policy_metrics(y, sel)
            rows.append({"policy": tag, "candidate_id": cid, "k": k, **pm})
            kk = int(round(k * 100))
            for m in ["alerts", "tp", "fp", "precision", "recall", "lift"]:
                f.add(f"frontier.{tag}.k{kk}.{m}", pm[m], "capacity_frontier.csv (daily floor, Cohort A)")
            if tag == "selected" and kk == 10:
                pt = policy_metrics(yt, sel)
                f.add("frontier.selected.k10.timestamp_tp", pt["tp"], "capacity_frontier.csv")
                f.add("frontier.selected.k10.timestamp_precision", pt["precision"], "capacity_frontier.csv")
                f.add("frontier.selected.k10.timestamp_recall", pt["recall"], "capacity_frontier.csv")
                f.add("frontier.selected.k10.fp_per_tp", pm["fp"] / pm["tp"], "capacity_frontier.csv")
                f.add("frontier.selected.k10.alerts_per_tp", pm["alerts"] / pm["tp"], "capacity_frontier.csv")
    fr = pd.DataFrame(rows)
    tables["capacity_frontier"] = fr
    claims = {c["claim_id"]: c["value"] for c in inp["claims"]}
    s10 = fr[(fr["policy"] == "selected") & (fr["k"] == 0.10)].iloc[0]
    _require_close("C12 daily alerts", s10["alerts"], claims["C12_SELECTED_DAILY_TOP10_ALERTS_COHORT_A"])
    _require_close("C13 daily TP", s10["tp"], claims["C13_SELECTED_DAILY_TOP10_TP_COHORT_A"])
    f.add("bt.base_rate", float(y.mean()), "backtest_predictions.csv")
    f.add("bt.base_rate_timestamp", float(yt.mean()), "backtest_predictions.csv")
    f.add("bt.positives_timestamp", int(yt.sum()), "backtest_predictions.csv")
    f.add("bt.days", int(a["order_purchase_timestamp"].dt.normalize().nunique()), "backtest_predictions.csv")
    may = a[a["order_purchase_timestamp"] < "2018-06-01"]
    f.add("bt.may_orders", len(may), "backtest_predictions.csv")
    f.add("bt.may_late", int(may["late_calendar_day"].sum()), "backtest_predictions.csv")
    for mon, g in a.groupby(a["order_purchase_timestamp"].dt.to_period("M").astype(str)):
        f.add(f"bt.month.{mon}.late_rate", float(g["late_calendar_day"].mean()), "backtest_predictions.csv")
    f.add("bt.orders_per_day_path_b", len(bt) / bt["order_purchase_timestamp"].dt.normalize().nunique(), "backtest_predictions.csv")

    # Path B: all scorable orders, unknown outcomes kept in the queue.
    sel_b = daily_floor_selected(bt, f"score_{SELECTED}", 0.10)
    q = bt[sel_b]
    unlabeled = q[~q["in_cohort_a"]]
    _require_close("C17 Path B alerts", len(q), claims["C17_PATH_B_DAILY_ALERTS_TOTAL"])
    _require_close("C18 Path B unlabeled", len(unlabeled), claims["C18_PATH_B_DAILY_ALERTS_UNLABELED"])
    tp_b = int(q.loc[q["in_cohort_a"], "late_calendar_day"].sum())
    f.add("pathb.k10.alerts_per_day", len(q) / bt["order_purchase_timestamp"].dt.normalize().nunique(), "backtest_predictions.csv")
    f.add("pathb.k10.cohort_a_alerts", int(q["in_cohort_a"].sum()), "backtest_predictions.csv")
    f.add("pathb.k10.tp", tp_b, "backtest_predictions.csv")
    f.add("pathb.k10.precision_lower", tp_b / len(q), "backtest_predictions.csv")
    f.add("pathb.k10.precision_upper", (tp_b + len(unlabeled)) / len(q), "backtest_predictions.csv")
    f.add("pathb.k10.unlabeled_share", len(unlabeled) / len(q), "backtest_predictions.csv")
    f.add("pathb.unknown_share_population", float((~bt["in_cohort_a"]).mean()), "backtest_predictions.csv")
    f.add("pathb.unknown_population", int((~bt["in_cohort_a"]).sum()), "backtest_predictions.csv")
    for s, n in unlabeled["order_status"].value_counts().items():
        f.add(f"pathb.k10.unlabeled_status.{s}", int(n), "backtest_predictions.csv")
    return fr


def calibration(f: Facts, inp: dict[str, Any], tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for period, df, ycol in [("backtest", inp["bt"], "late_calendar_day"), ("validation", inp["val"], "late")]:
        a = df[df["in_cohort_a"] & df[ycol].notna()]
        s = a[f"score_{SELECTED}"].astype(float)
        y = a[ycol].astype(int)
        bins = pd.qcut(s.rank(method="first"), 10, labels=False)
        rel = pd.DataFrame({"bin": bins, "pred": s, "obs": y}).groupby("bin").agg(mean_pred=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size"))
        rel.insert(0, "period", period)
        out[period] = rel
        f.add(f"calib.{period}.mean_pred", float(s.mean()), "backtest/validation predictions")
        f.add(f"calib.{period}.observed", float(y.mean()), "backtest/validation predictions")
        f.add(f"calib.{period}.ratio", float(s.mean() / y.mean()), "backtest/validation predictions")
        f.add(f"calib.{period}.top_decile_pred", float(rel["mean_pred"].iloc[-1]), "calibration table")
        f.add(f"calib.{period}.top_decile_observed", float(rel["observed"].iloc[-1]), "calibration table")
    tables["calibration_deciles"] = pd.concat(out.values()).reset_index()
    m = inp["metrics"]["selected_model_backtest_metrics"]
    _require_close("backtest mean pred", f.items["calib.backtest.mean_pred"]["value"], m["mean_predicted_prob"], 1e-5)
    f.add("calib.backtest.brier", m["brier_score"], "metrics_summary.json")
    f.add("calib.backtest.ece10", m.get("ece_quantile_10", m.get("ece_10", np.nan)), "metrics_summary.json")
    return out


def concentration(f: Facts, inp: dict[str, Any], attrs: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Where delays sit by promised lead x route, for the validation months and for the backtest window."""
    bt = inp["bt"]
    a_bt = bt[bt["in_cohort_a"]][["order_id", "late_calendar_day", "same_state", "estimated_lead_calendar_days", "customer_state", "primary_seller_state"]]
    chk = a_bt.merge(attrs[["order_id", "same_state", "lead_days", "seller_state", "customer_state"]], on="order_id", suffixes=("", "_raw"))
    same_ok = (chk["same_state"].astype(str).str.lower().isin(["true", "1"]) == chk["same_state_raw"]).mean()
    lead_ok = (chk["estimated_lead_calendar_days"].astype(float) == chk["lead_days"].astype(float)).mean()
    if same_ok < 1.0 or lead_ok < 1.0:
        raise AssertionError(f"Raw rebuild of route/lead disagrees with frozen predictions (same_state={same_ok}, lead={lead_ok})")

    audit = inp["audit"][inp["audit"]["in_cohort_a"]][["order_id", "order_purchase_timestamp", "late_calendar_day"]]
    periods = {
        "validation": audit[(audit["order_purchase_timestamp"] >= PERIOD_VALIDATION[0]) & (audit["order_purchase_timestamp"] < PERIOD_VALIDATION[1])],
        "backtest": audit[audit["order_id"].isin(a_bt["order_id"])],
    }
    rows = []
    for period, p in periods.items():
        d = p.merge(attrs[["order_id", "same_state", "lead_days", "seller_state", "customer_state"]], on="order_id", how="left")
        d["band"] = lead_band(d["lead_days"].astype(float))
        d["route"] = np.where(d["same_state"], "intra_state", "inter_state")
        total_late = int(d["late_calendar_day"].sum())
        f.add(f"conc.{period}.orders", len(d), "cohort_audit.csv + raw tables")
        f.add(f"conc.{period}.late", total_late, "cohort_audit.csv + raw tables")
        f.add(f"conc.{period}.late_rate", float(d["late_calendar_day"].mean()), "cohort_audit.csv + raw tables")
        for route, g in d.groupby("route"):
            f.add(f"conc.{period}.{route}.orders", len(g), "concentration.csv")
            f.add(f"conc.{period}.{route}.late_rate", float(g["late_calendar_day"].mean()), "concentration.csv")
            f.add(f"conc.{period}.{route}.late_share", float(g["late_calendar_day"].sum() / total_late), "concentration.csv")
        for band, g in d.groupby("band"):
            key = band.replace("<=", "le").replace(">", "gt").replace("-", "_")
            f.add(f"conc.{period}.band_{key}.orders", len(g), "concentration.csv")
            f.add(f"conc.{period}.band_{key}.order_share", len(g) / len(d), "concentration.csv")
            f.add(f"conc.{period}.band_{key}.late", int(g["late_calendar_day"].sum()), "concentration.csv")
            f.add(f"conc.{period}.band_{key}.late_rate", float(g["late_calendar_day"].mean()), "concentration.csv")
            f.add(f"conc.{period}.band_{key}.late_share", float(g["late_calendar_day"].sum() / total_late), "concentration.csv")
            f.add(f"conc.{period}.band_{key}.intra_share", float(g["same_state"].mean()), "concentration.csv")
        for (band, route), g in d.groupby(["band", "route"]):
            rows.append({"period": period, "lead_band": band, "route": route, "orders": len(g), "late": int(g["late_calendar_day"].sum()), "late_rate": float(g["late_calendar_day"].mean())})
        spsp = d[(d["seller_state"] == "SP") & (d["customer_state"] == "SP")]
        spsp_short = spsp[spsp["lead_days"] <= 10]
        f.add(f"conc.{period}.sp_sp.orders", len(spsp), "concentration.csv")
        f.add(f"conc.{period}.sp_sp.late_rate", float(spsp["late_calendar_day"].mean()), "concentration.csv")
        f.add(f"conc.{period}.sp_sp_le10.orders", len(spsp_short), "concentration.csv")
        f.add(f"conc.{period}.sp_sp_le10.late", int(spsp_short["late_calendar_day"].sum()), "concentration.csv")
        f.add(f"conc.{period}.sp_sp_le10.late_rate", float(spsp_short["late_calendar_day"].mean()), "concentration.csv")
        f.add(f"conc.{period}.sp_sp_le10.late_share", float(spsp_short["late_calendar_day"].sum() / total_late), "concentration.csv")
        state = d.groupby("customer_state")["late_calendar_day"].agg(["size", "mean", "sum"])
        f.add(f"conc.{period}.states_lt100", int((state["size"] < 100).sum()), "concentration.csv")
        for st in ["SP", "RJ", "MG", "BA"]:
            if st in state.index:
                f.add(f"conc.{period}.state_{st}.orders", int(state.loc[st, "size"]), "concentration.csv")
                f.add(f"conc.{period}.state_{st}.late_rate", float(state.loc[st, "mean"]), "concentration.csv")
                f.add(f"conc.{period}.state_{st}.late_share", float(state.loc[st, "sum"] / total_late), "concentration.csv")
    conc = pd.DataFrame(rows)
    tables["delay_concentration"] = conc
    f.add("conc.raw_rebuild_agreement", float(min(same_ok, lead_ok)), "backtest_predictions.csv vs raw tables")
    return conc


def milestone2(f: Facts, inp: dict[str, Any], attrs: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Post hoc: flag an order when the seller has not handed it to the carrier by its shipping-limit deadline."""
    bt = inp["bt"]
    att = attrs.set_index("order_id")
    d = bt[["order_id", "order_purchase_timestamp", "order_estimated_delivery_date", "in_cohort_a", "late_calendar_day", "order_status", f"score_{SELECTED}", f"score_{SHORT_LEAD}"]].copy()
    d = d.join(att[["shipping_limit_date", "order_delivered_carrier_date"]], on="order_id")
    has_limit = d["shipping_limit_date"].notna()
    not_handed = d["order_delivered_carrier_date"].isna() | (d["order_delivered_carrier_date"] > d["shipping_limit_date"])
    d["m2_flag"] = has_limit & not_handed
    a = d[d["in_cohort_a"]].copy()
    carrier_missing = int(a["order_delivered_carrier_date"].isna().sum())
    # A delivered order with no carrier timestamp cannot be judged at the deadline: not flagged (conservative).
    a.loc[a["order_delivered_carrier_date"].isna(), "m2_flag"] = False
    y = a["late_calendar_day"].astype(int).to_numpy()
    m2 = a["m2_flag"].to_numpy(dtype=bool)
    pm = policy_metrics(y, m2)
    for k, v in pm.items():
        f.add(f"m2.bt.{k}", v, "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.carrier_missing_cohort_a", carrier_missing, "raw orders", post_hoc=True)
    f.add("m2.bt.alert_rate", float(m2.mean()), "milestone2 (post hoc)", post_hoc=True)
    flagged = a[m2]
    per_day = flagged.groupby(flagged["shipping_limit_date"].dt.normalize()).size()
    days = a["order_purchase_timestamp"].dt.normalize().nunique()
    f.add("m2.bt.alerts_per_day", len(flagged) / days, "milestone2 (post hoc)", post_hoc=True)
    slack = (flagged["order_estimated_delivery_date"] - flagged["shipping_limit_date"]).dt.total_seconds() / 86400
    slack_tp = slack[flagged["late_calendar_day"] == 1]
    f.add("m2.bt.slack_days_median", float(slack.median()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.slack_days_median_tp", float(slack_tp.median()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.flag_after_promise", int((slack <= 0).sum()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.flag_after_promise_share", float((slack <= 0).mean()), "milestone2 (post hoc)", post_hoc=True)

    # Equal-volume comparison with the purchase-time rule (global top-N within the backtest window).
    n_alert = int(m2.sum())
    order = a.sort_values([f"score_{SELECTED}", "order_purchase_timestamp", "order_id"], ascending=[False, True, True], kind="mergesort")
    top = np.zeros(len(a), dtype=bool)
    top[a.index.get_indexer(order.index[:n_alert])] = True
    eq = policy_metrics(y, top)
    f.add("m2.bt.equal_volume_purchase_precision", eq["precision"], "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.equal_volume_purchase_recall", eq["recall"], "milestone2 (post hoc)", post_hoc=True)
    p10_metrics = policy_metrics(y, daily_floor_selected(a, f"score_{SELECTED}", 0.10))
    f.add("m2.bt.precision_gain_vs_purchase_k10", pm["precision"] - p10_metrics["precision"], "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.precision_ratio_vs_purchase_k10", pm["precision"] / p10_metrics["precision"], "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.recall_ratio_vs_purchase_k10", pm["recall"] / p10_metrics["recall"], "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.alerts_per_tp", pm["alerts"] / pm["tp"], "milestone2 (post hoc)", post_hoc=True)

    # Overlap with the purchase-time daily top-10% queue.
    p10 = daily_floor_selected(a, f"score_{SELECTED}", 0.10)
    late = y == 1
    f.add("m2.bt.late_caught_both", int((late & p10 & m2).sum()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.late_caught_purchase_only", int((late & p10 & ~m2).sum()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.late_caught_m2_only", int((late & ~p10 & m2).sum()), "milestone2 (post hoc)", post_hoc=True)
    union = p10 | m2
    pu = policy_metrics(y, union)
    for k in ["alerts", "tp", "precision", "recall"]:
        f.add(f"m2.bt.union.{k}", pu[k], "milestone2 (post hoc)", post_hoc=True)

    # Path B view: never-handed-over orders with unknown outcome.
    unk = d[~d["in_cohort_a"]]
    f.add("m2.bt.unknown_flagged", int(unk["m2_flag"].sum()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.bt.unknown_total", len(unk), "milestone2 (post hoc)", post_hoc=True)

    # Same rule on the validation months (no parameters were fitted, so it is directly comparable).
    audit = inp["audit"][inp["audit"]["in_cohort_a"]][["order_id", "order_purchase_timestamp", "late_calendar_day"]]
    vp = audit[(audit["order_purchase_timestamp"] >= PERIOD_VALIDATION[0]) & (audit["order_purchase_timestamp"] < PERIOD_VALIDATION[1])]
    vp = vp.join(att[["shipping_limit_date", "order_delivered_carrier_date"]], on="order_id")
    vflag = vp["shipping_limit_date"].notna() & vp["order_delivered_carrier_date"].notna() & (vp["order_delivered_carrier_date"] > vp["shipping_limit_date"])
    pv = policy_metrics(vp["late_calendar_day"].astype(int).to_numpy(), vflag.to_numpy())
    for k in ["alerts", "tp", "precision", "recall", "lift"]:
        f.add(f"m2.val.{k}", pv[k], "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.val.alert_rate", float(vflag.mean()), "milestone2 (post hoc)", post_hoc=True)
    f.add("m2.val.base_rate", float(vp["late_calendar_day"].mean()), "milestone2 (post hoc)", post_hoc=True)

    tables["milestone2_summary"] = pd.DataFrame(
        [
            {"policy": "purchase_time_daily_top10", **policy_metrics(y, p10)},
            {"policy": "milestone2_shipping_limit_breach", **pm},
            {"policy": "union", **pu},
            {"policy": "purchase_time_equal_volume", **eq},
        ]
    )
    return {"a": a, "m2": m2, "p10": p10}


def bootstrap(f: Facts, inp: dict[str, Any], m2: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    """Weekly block bootstrap (calendar weeks starting Monday) with the configured replicate count."""
    uc = inp["config"]["uncertainty_contract"]
    reps = int(uc["block_bootstrap_replicates"])
    seed = int(inp["config"]["random_seed"])
    f.add("boot.reps_configured", reps, "config.resolved.json")
    f.add("boot.reps_formal_run", int(inp["uncertainty"]["weekly_block_bootstrap"]["n_bootstrap"]), "uncertainty_summary.json")
    f.add("boot.weeks", int(inp["uncertainty"]["weekly_block_bootstrap"]["n_blocks"]), "uncertainty_summary.json")
    wb = inp["uncertainty"]["weekly_block_bootstrap"]["weekly_block_bootstrap"]
    for m in ["daily_policy_recall_top10", "daily_policy_precision_top10", "pr_auc"]:
        f.add(f"boot.formal.{m}.low", wb[m]["ci95_low"], "uncertainty_summary.json")
        f.add(f"boot.formal.{m}.high", wb[m]["ci95_high"], "uncertainty_summary.json")
    iid = inp["uncertainty"]["iid_bootstrap"]["metrics"]
    f.add("boot.formal.iid_roc.low", iid["roc_auc"]["ci95_low"], "uncertainty_summary.json")
    f.add("boot.formal.iid_roc.high", iid["roc_auc"]["ci95_high"], "uncertainty_summary.json")

    a = m2["a"]
    y = a["late_calendar_day"].astype(int).to_numpy()
    week = a["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
    weeks = np.unique(week)
    idx = {w: np.where(week == w)[0] for w in weeks}
    sel10 = m2["p10"]
    short10 = daily_floor_selected(a, f"score_{SHORT_LEAD}", 0.10)
    s_sel = a[f"score_{SELECTED}"].to_numpy(dtype=float)
    flag = m2["m2"]
    rng = np.random.default_rng(seed + 2026)
    out = {k: [] for k in ["recall", "precision", "ap", "roc", "short_minus_sel_recall", "m2_precision", "m2_recall", "m2_minus_sel_precision"]}
    for _ in range(reps):
        drawn = rng.choice(weeks, size=len(weeks), replace=True)
        b = np.concatenate([idx[w] for w in drawn])
        yb = y[b]
        pos = yb.sum()
        if pos == 0 or pos == len(yb):
            continue
        tp_sel = (yb & sel10[b]).sum()
        out["recall"].append(tp_sel / pos)
        out["precision"].append(tp_sel / sel10[b].sum())
        out["ap"].append(average_precision_score(yb, s_sel[b]))
        out["roc"].append(roc_auc_score(yb, s_sel[b]))
        out["short_minus_sel_recall"].append(((yb & short10[b]).sum() - tp_sel) / pos)
        m2p = (yb & flag[b]).sum() / max(1, flag[b].sum())
        out["m2_precision"].append(m2p)
        out["m2_recall"].append((yb & flag[b]).sum() / pos)
        out["m2_minus_sel_precision"].append(m2p - tp_sel / sel10[b].sum())
    rows = []
    for k, v in out.items():
        lo, hi = np.quantile(v, [0.025, 0.975])
        f.add(f"boot.wb2000.{k}.low", float(lo), "weekly block bootstrap (this layer)", post_hoc=k.startswith(("short", "m2")))
        f.add(f"boot.wb2000.{k}.high", float(hi), "weekly block bootstrap (this layer)", post_hoc=k.startswith(("short", "m2")))
        f.add(f"boot.wb2000.{k}.share_above_zero", float(np.mean(np.asarray(v) > 0)), "weekly block bootstrap (this layer)", post_hoc=k.startswith(("short", "m2")))
        rows.append({"metric": k, "valid_draws": len(v), "ci95_low": lo, "ci95_high": hi, "mean": float(np.mean(v))})
    f.add("boot.wb2000.valid_draws", len(out["recall"]), "weekly block bootstrap (this layer)")
    tables["bootstrap_weekly_2000"] = pd.DataFrame(rows)


def cost_scenarios(f: Facts, inp: dict[str, Any], frontier: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pre-registered cost scenarios from the frozen config, applied to the daily-floor frontier and Milestone 2."""
    n = len(inp["bt"][inp["bt"]["in_cohort_a"]])
    rows = []
    sel = frontier[frontier["policy"] == "selected"]
    m2 = {k: f.items[f"m2.bt.{k}"]["value"] for k in ["alerts", "tp", "fp", "precision"]}
    for sc in inp["config"]["cost_scenarios"]:
        sid, b, c, fp_cost, fn_cost = sc["scenario_id"], sc["benefit_per_tp"], sc["cost_per_contact"], sc["cost_per_fp"], sc["cost_per_fn"]
        breakeven = (c + fp_cost) / (b + fp_cost)
        f.add(f"cost.{sid}.benefit", b, "config.resolved.json")
        f.add(f"cost.{sid}.contact", c, "config.resolved.json")
        f.add(f"cost.{sid}.fp_cost", fp_cost, "config.resolved.json")
        f.add(f"cost.{sid}.fn_cost", fn_cost, "config.resolved.json")
        f.add(f"cost.{sid}.breakeven_precision", breakeven, "cost_scenarios.csv")
        f.add(f"cost.{sid}.breakeven_precision_with_fn", (c + fp_cost) / (b + fn_cost + fp_cost), "cost_scenarios.csv")
        best_k, best_v = 0.0, 0.0
        for _, r in sel.iterrows():
            v = 1000.0 * (r["tp"] * b - r["alerts"] * c - r["fp"] * fp_cost) / n
            rows.append({"scenario_id": sid, "policy": "purchase_time_daily_floor", "k": r["k"], "alerts": r["alerts"], "tp": r["tp"], "precision": r["precision"], "net_per_1000_orders": v})
            kk = int(round(r["k"] * 100))
            f.add(f"cost.{sid}.k{kk}.net_per_1000", v, "cost_scenarios.csv")
            if v > best_v:
                best_k, best_v = r["k"], v
        f.add(f"cost.{sid}.best_k", int(round(best_k * 100)), "cost_scenarios.csv")
        f.add(f"cost.{sid}.best_net_per_1000", best_v, "cost_scenarios.csv")
        v_m2 = 1000.0 * (m2["tp"] * b - m2["alerts"] * c - m2["fp"] * fp_cost) / n
        rows.append({"scenario_id": sid, "policy": "milestone2_shipping_limit_breach", "k": m2["alerts"] / n, "alerts": m2["alerts"], "tp": m2["tp"], "precision": m2["precision"], "net_per_1000_orders": v_m2})
        f.add(f"cost.{sid}.m2.net_per_1000", v_m2, "cost_scenarios.csv", post_hoc=True)
    cs = pd.DataFrame(rows)
    tables["cost_scenarios"] = cs
    return cs


def model_facts(f: Facts, inp: dict[str, Any]) -> None:
    base_num, base_cat, base_all = get_feature_columns("base_v2")
    ext_num, ext_cat, ext_all = get_feature_columns("extended_purchase_assumptions_v2")
    f.add("features.base_total", len(base_all), "src/olist_delay/features.py")
    f.add("features.base_numeric", len(base_num), "src/olist_delay/features.py")
    f.add("features.base_categorical", len(base_cat), "src/olist_delay/features.py")
    f.add("features.extended_total", len(ext_all), "src/olist_delay/features.py")
    f.add("features.extended_added", len(ext_all) - len(base_all), "src/olist_delay/features.py")
    grid = inp["config"]["model_grid"]
    f.add("grid.lr_C", grid["logistic_regression"]["C"], "config.resolved.json")
    f.add("grid.tree_depth", grid["decision_tree"]["max_depth"], "config.resolved.json")
    f.add("grid.tree_leaf", grid["decision_tree"]["min_samples_leaf"], "config.resolved.json")
    f.add("rule.tightness_clip_low", 0.5, "src/olist_delay/models.py:192")
    f.add("rule.tightness_clip_high", 2.0, "src/olist_delay/models.py:192")
    f.add("rule.lead_clip_low", 3.0, "src/olist_delay/models.py:192")
    f.add("rule.lead_clip_high", 90.0, "src/olist_delay/models.py:192")
    f.add("rule.route_recent_window_days", 60, "src/olist_delay/history.py (route_recent_60d_late_rate_smoothed)")
    f.add("rule.route_smoothing_m", inp["config"]["history_contract"]["smoothing_alpha"], "config.resolved.json (history_contract.smoothing_alpha)")
    bt = inp["bt"]
    f.add("rule.route_pairs_backtest", int((bt["primary_seller_state"].astype(str) + "->" + bt["customer_state"].astype(str)).nunique()), "backtest_predictions.csv")
    f.add("rule.backtest_median_lead", float(bt.loc[bt["in_cohort_a"], "estimated_lead_calendar_days"].median()), "backtest_predictions.csv")
    # Fitted parameters of the selected rule, read only after the pickle matches the hash bound in the manifest.
    model_path = inp["run_dir"] / "models" / "selected_model.pkl"
    expected = inp["manifest"]["artifacts"]["models/selected_model.pkl"]["sha256"]
    if sha256_file(model_path) != expected:
        raise ValueError("selected_model.pkl does not match the manifest hash; refusing to deserialize")
    with model_path.open("rb") as fh:
        rule = pickle.load(fh)["selected_bundle"].base_estimator
    f.add("rule.fitted_median_lead", float(rule.median_lead_), "models/selected_model.pkl (hash-verified)")
    f.add("rule.fitted_prior", float(rule.prior_), "models/selected_model.pkl (hash-verified)")


# --------------------------------------------------------------------------------------------- figures
def _style(ax: plt.Axes, ygrid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0, labelsize=9)
    ax.set_facecolor(SURFACE)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _new(ncols: int = 1, width: float = 10.0, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def _pct(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def make_figures(fig_dir: Path, f: Facts, tables: dict[str, pd.DataFrame], inp: dict[str, Any], named: dict[str, str], m2: dict[str, Any]) -> list[str]:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold"})
    V = lambda k: f.items[k]["value"]  # noqa: E731
    names: list[str] = []

    # F1 — population and why lateness matters
    fig, (a1, a2) = _new(2, 10.5, 3.8)
    labels = ["Source orders", "With item rows", "Cohort A\n(delivered, evaluable)", "Unknown outcome\n(kept as NA)"]
    vals = [V("claim.C01_SOURCE_ORDERS_TOTAL"), V("pop.orders_with_items"), V("claim.C02_COHORT_A_EVALUABLE_TOTAL"), V("claim.C03_UNKNOWN_OUTCOME_EXCLUDED_TOTAL")]
    yy = np.arange(len(vals))[::-1]
    a1.barh(yy, vals, height=0.42, color=[BLUE, BLUE, BLUE, ORANGE])
    a1.set_yticks(yy, labels)
    a1.xaxis.grid(True, color=GRID)
    a1.yaxis.grid(False)
    for yv, v in zip(yy, vals):
        a1.text(v + 1200, yv, f"{v:,}", va="center", color=INK, fontsize=9)
    a1.set_xlim(0, 118000)
    a1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    a1.set_title("A. Every order is accounted for", loc="left")
    groups = ["On time", "Late (≥1 calendar day)"]
    one = [V("stage.on_time.one_star_rate"), V("stage.late.one_star_rate")]
    xs = np.arange(2)
    a2.bar(xs, one, width=0.38, color=[BLUE, ORANGE])
    a2.set_xticks(xs, groups)
    _pct(a2)
    a2.set_ylim(0, 0.62)
    for x, v, mean in zip(xs, one, [V("stage.on_time.review_mean"), V("stage.late.review_mean")]):
        a2.text(x, v + 0.015, f"{v:.1%} one-star\nmean score {mean:.2f}", ha="center", color=INK, fontsize=9)
    a2.set_ylabel("Share of reviews with 1 star")
    a2.set_title("B. Late orders draw far more 1-star reviews", loc="left")
    _save(fig, fig_dir / "fig01_population_and_motivation.png")
    names.append("fig01_population_and_motivation.png")

    # F2 — monthly late rate timeline with the evaluation windows
    m = tables["monthly_late_rate"]
    fig, (ax,) = _new(1, 10.5, 3.6)
    x = np.arange(len(m))
    ax.axvspan(list(m["purchase_month"]).index("2018-01") - 0.5, list(m["purchase_month"]).index("2018-04") + 0.5, color=BLUE, alpha=0.08, lw=0)
    ax.axvspan(list(m["purchase_month"]).index("2018-05") - 0.5, list(m["purchase_month"]).index("2018-08") + 0.5, color=ORANGE, alpha=0.08, lw=0)
    ax.plot(x, m["late_rate"], color=INK2, lw=2, marker="o", ms=4, mfc=INK2, mec=SURFACE, mew=1.5)
    ax.set_xticks(x[::2], [s[2:] for s in m["purchase_month"]][::2], rotation=0)
    _pct(ax)
    ax.set_ylabel("Late rate (calendar day, Cohort A)")
    top = m["late_rate"].max()
    ax.text(list(m["purchase_month"]).index("2018-01") - 0.3, top * 1.02, "Validation folds\n(model selection)", color=INK2, fontsize=9, va="bottom")
    ax.text(list(m["purchase_month"]).index("2018-05") - 0.3, top * 1.02, "Backtest window\n(26 May – 29 Aug)", color=INK2, fontsize=9, va="bottom")
    ax.set_ylim(0, top * 1.28)
    ax.set_title("Monthly late-delivery rate by purchase month, 2017-01 to 2018-08", loc="left")
    _save(fig, fig_dir / "fig02_monthly_late_rate.png")
    names.append("fig02_monthly_late_rate.png")

    # F3 — validation vs backtest daily top-10% recall for all candidates
    lb = tables["leaderboard"]
    fig, (ax,) = _new(1, 7.2, 5.0)
    fam_style = {"rule": (BLUE, "o", "Rule baseline"), "logistic_regression": (ORANGE, "s", "Logistic regression"), "decision_tree": (AQUA, "^", "Shallow decision tree")}
    for fam, (col, mk, lab) in fam_style.items():
        g = lb[lb["family"] == fam]
        ax.scatter(g["val_mean_daily_top10_recall"], g["bt_daily_top10_recall"], s=46, color=col, marker=mk, edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
    lim = (0, max(lb["val_mean_daily_top10_recall"].max(), lb["bt_daily_top10_recall"].max()) * 1.12)
    ax.plot(lim, lim, color=AXIS, lw=1, zorder=1)
    ax.axhline(0.10, color=GRID, lw=1, zorder=1)
    ax.axvline(0.10, color=GRID, lw=1, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _pct(ax, "y")
    _pct(ax, "x")
    ax.xaxis.grid(True, color=GRID)
    ax.set_xlabel("Validation (Jan–Apr 2018): mean daily top-10% recall")
    ax.set_ylabel("Backtest (May–Aug 2018): daily top-10% recall")
    k = lb.set_index("candidate_id")
    for cid, text, dx, dy in [
        (SELECTED, "rule_region_lead (selected)", -0.062, 0.010),
        (SHORT_LEAD, "rule_short_promised_lead", 0.006, 0.004),
        (named["best_lr"], "best logistic regression", -0.030, -0.022),
        ("rule_random", "random", 0.004, -0.010),
    ]:
        r = k.loc[cid]
        ax.annotate(text, (r["val_mean_daily_top10_recall"], r["bt_daily_top10_recall"]), xytext=(r["val_mean_daily_top10_recall"] + dx, r["bt_daily_top10_recall"] + dy), color=INK, fontsize=8.5)
    ax.text(lim[1] * 0.98, lim[1] * 0.93, "y = x", color=MUTED, fontsize=8, ha="right")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    ax.set_title("Validation ranking did not carry over to the backtest", loc="left")
    _save(fig, fig_dir / "fig03_validation_vs_backtest.png")
    names.append("fig03_validation_vs_backtest.png")

    # F4 — PR and ROC curves on the backtest
    a = inp["bt"][inp["bt"]["in_cohort_a"]]
    y = a["late_calendar_day"].astype(int).to_numpy()
    fig, (p1, p2) = _new(2, 10.5, 4.2)
    for cid, col, lab in [(SELECTED, BLUE, "rule_region_lead (selected)"), (SHORT_LEAD, ORANGE, "rule_short_promised_lead"), (named["best_lr"], AQUA, "best logistic regression")]:
        s = a[f"score_{cid}"].to_numpy(dtype=float)
        pr, rc, _ = precision_recall_curve(y, s)
        p1.plot(rc, pr, color=col, lw=2, label=lab)
        fpr, tpr, _ = roc_curve(y, s)
        p2.plot(fpr, tpr, color=col, lw=2, label=lab)
    base = y.mean()
    p1.axhline(base, color=MUTED, lw=1)
    p1.text(0.98, base - 0.016, f"base rate {base:.1%}", color=MUTED, fontsize=8, ha="right")
    p1.set_ylim(0, 0.4)
    p1.set_xlabel("Recall")
    p1.set_ylabel("Precision")
    _pct(p1)
    _pct(p1, "x")
    p1.set_title("A. Precision–recall (backtest, Cohort A)", loc="left")
    p2.plot([0, 1], [0, 1], color=MUTED, lw=1)
    p2.set_xlabel("False positive rate")
    p2.set_ylabel("True positive rate")
    p2.set_title("B. ROC (backtest, Cohort A)", loc="left")
    p2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    _save(fig, fig_dir / "fig04_backtest_pr_roc.png")
    names.append("fig04_backtest_pr_roc.png")

    # F5 — calibration flips between regimes
    cal = tables["calibration_deciles"]
    fig, (ax,) = _new(1, 6.4, 4.6)
    for period, col, lab in [("validation", BLUE, "Validation months (Jan–Apr 2018)"), ("backtest", ORANGE, "Backtest window (May–Aug 2018)")]:
        g = cal[cal["period"] == period]
        ax.plot(g["mean_pred"], g["observed"], color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=lab)
    mx = max(cal["mean_pred"].max(), cal["observed"].max()) * 1.05
    ax.plot([0, mx], [0, mx], color=AXIS, lw=1)
    ax.text(mx * 0.97, mx * 0.9, "perfect calibration", color=MUTED, fontsize=8, ha="right")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    _pct(ax)
    _pct(ax, "x")
    ax.xaxis.grid(True, color=GRID)
    ax.set_xlabel("Mean score in decile (rule_region_lead)")
    ax.set_ylabel("Observed late rate in decile")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    ax.set_title("Scores are useful for ranking, not as probabilities", loc="left")
    _save(fig, fig_dir / "fig05_calibration_by_regime.png")
    names.append("fig05_calibration_by_regime.png")

    # F6 — capacity frontier under the daily floor policy
    fr = tables["capacity_frontier"]
    fig, (r1, r2) = _new(2, 10.5, 4.0)
    for tag, col, lab in [("selected", BLUE, "rule_region_lead (selected)"), ("short_lead", ORANGE, "rule_short_promised_lead"), ("random", GRAY, "random ranking")]:
        g = fr[fr["policy"] == tag]
        r1.plot(g["k"], g["recall"], color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=lab)
        r2.plot(g["k"], g["precision"], color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=lab)
    r2.axhline(V("bt.base_rate"), color=MUTED, lw=1)
    r2.text(0.205, V("bt.base_rate") - 0.004, "base rate", color=MUTED, fontsize=8, ha="center")
    for ax_, t in [(r1, "A. Share of late orders caught"), (r2, "B. Share of alerts that were late")]:
        ax_.set_xticks(BUDGETS, [f"{int(b*100)}%" for b in BUDGETS])
        ax_.set_xlabel("Daily alert capacity k (share of the day's orders)")
        _pct(ax_)
        ax_.set_title(t, loc="left")
    r2.set_ylim(0, None)
    r1.set_ylim(0, None)
    r1.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    _save(fig, fig_dir / "fig06_capacity_frontier.png")
    names.append("fig06_capacity_frontier.png")

    # F7 — delay concentration by promised lead and route, two regimes
    conc = tables["delay_concentration"]
    fig, axes = _new(2, 10.5, 4.0)
    bands = [b[2] for b in LEAD_BANDS]
    ymax = conc["late_rate"].max() * 1.18
    for ax_, period, title in [(axes[0], "validation", "A. Validation months (Jan–Apr 2018)"), (axes[1], "backtest", "B. Backtest window (May–Aug 2018)")]:
        g = conc[conc["period"] == period]
        xs = np.arange(len(bands))
        for j, (route, col, lab) in enumerate([("intra_state", BLUE, "Seller and customer in same state"), ("inter_state", ORANGE, "Cross-state")]):
            gg = g[g["route"] == route].set_index("lead_band").reindex(bands)
            ax_.bar(xs + (j - 0.5) * 0.34, gg["late_rate"], width=0.32, color=col, label=lab)
        n_band = g.groupby("lead_band")["orders"].sum().reindex(bands)
        ax_.set_xticks(xs, [f"{b}\nn={n:,}" for b, n in zip(bands, n_band)])
        ax_.set_xlabel("Promised lead time at purchase (orders in band)")
        ax_.set_ylim(0, ymax)
        _pct(ax_)
        ax_.set_title(title, loc="left")
    axes[0].set_ylabel("Late rate")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    _save(fig, fig_dir / "fig07_delay_concentration.png")
    names.append("fig07_delay_concentration.png")

    # F8 — where the extra time goes (ex post, never used for prediction)
    fig, (s1, s2) = _new(2, 10.5, 3.8)
    stages = ["Payment approval", "Seller → carrier", "Carrier → customer"]
    on = [V("stage.on_time.approval_hours_mean") / 24, V("stage.on_time.seller_days_mean"), V("stage.on_time.carrier_days_mean")]
    lt = [V("stage.late.approval_hours_mean") / 24, V("stage.late.seller_days_mean"), V("stage.late.carrier_days_mean")]
    yy = np.arange(len(stages))[::-1]
    s1.barh(yy + 0.17, on, height=0.32, color=BLUE, label="On time")
    s1.barh(yy - 0.17, lt, height=0.32, color=ORANGE, label="Late")
    s1.set_yticks(yy, stages)
    s1.yaxis.grid(False)
    s1.xaxis.grid(True, color=GRID)
    s1.set_xlabel("Mean days in stage")
    for yv, v in zip(yy - 0.17, lt):
        s1.text(v + 0.4, yv, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    for yv, v in zip(yy + 0.17, on):
        s1.text(v + 0.4, yv, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    s1.set_xlim(0, max(lt) * 1.12)
    s1.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    s1.set_title("A. Most of the extra time is carrier transit", loc="left")
    br = [V("stage.on_time.shipping_limit_breach_rate"), V("stage.late.shipping_limit_breach_rate")]
    s2.bar([0, 1], br, width=0.38, color=[BLUE, ORANGE])
    s2.set_xticks([0, 1], ["On time", "Late"])
    _pct(s2)
    s2.set_ylim(0, 0.35)
    for x_, v in zip([0, 1], br):
        s2.text(x_, v + 0.008, f"{v:.1%}", ha="center", color=INK, fontsize=9)
    s2.set_ylabel("Seller missed shipping-limit date")
    s2.set_title("B. Late orders often miss the seller deadline", loc="left")
    _save(fig, fig_dir / "fig08_fulfillment_stages.png")
    names.append("fig08_fulfillment_stages.png")

    # F9 — purchase-time queue vs Milestone-2 signal (post hoc)
    ms = tables["milestone2_summary"].set_index("policy")
    pols = [("purchase_time_daily_top10", "Purchase-time\ntop 10% per day", BLUE), ("milestone2_shipping_limit_breach", "Seller missed\nshipping limit", ORANGE), ("union", "Either signal", AQUA)]
    fig, axes = _new(3, 11.0, 3.6)
    for ax_, metric, title, fmt in [(axes[0], "precision", "A. Precision (share of alerts late)", "{:.1%}"), (axes[1], "recall", "B. Recall (share of late orders flagged)", "{:.1%}"), (axes[2], "alerts", "C. Alerts in the backtest window", "{:,.0f}")]:
        vals = [ms.loc[p, metric] for p, _, _ in pols]
        ax_.bar(range(3), vals, width=0.4, color=[c for _, _, c in pols])
        ax_.set_xticks(range(3), [lab for _, lab, _ in pols], fontsize=8.5)
        for x_, v in enumerate(vals):
            ax_.text(x_, v * 1.02, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        if metric != "alerts":
            _pct(ax_)
        else:
            ax_.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}"))
        ax_.set_ylim(0, max(vals) * 1.2)
        ax_.set_title(title, loc="left", fontsize=10)
    axes[0].axhline(V("bt.base_rate"), color=MUTED, lw=1)
    _save(fig, fig_dir / "fig09_milestone2_vs_purchase_time.png")
    names.append("fig09_milestone2_vs_purchase_time.png")

    # F10 — pre-registered cost scenarios
    cs = tables["cost_scenarios"]
    fig, (ax,) = _new(1, 8.0, 4.2)
    colors = [BLUE, ORANGE, AQUA]
    for (sid, g), col in zip(cs.groupby("scenario_id", sort=False), colors):
        p = g[g["policy"] == "purchase_time_daily_floor"]
        sc = next(s for s in inp["config"]["cost_scenarios"] if s["scenario_id"] == sid)
        lab = f"{sid.replace('_', ' ')} (benefit {sc['benefit_per_tp']:.2f}, contact {sc['cost_per_contact']:.2f}, false alarm {sc['cost_per_fp']:.2f})"
        ax.plot(p["k"], p["net_per_1000_orders"], color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=lab)
        q = g[g["policy"] == "milestone2_shipping_limit_breach"].iloc[0]
        ax.scatter([q["k"]], [q["net_per_1000_orders"]], marker="D", s=50, color=col, edgecolor=INK, linewidth=0.8, zorder=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(BUDGETS, [f"{int(b*100)}%" for b in BUDGETS])
    ax.set_xlabel("Alert rate (share of Cohort A orders flagged)")
    ax.set_ylabel("Net value per 1,000 orders (config units)")
    ax.text(0.302, ax.get_ylim()[1] * 0.92, "◆ = seller missed shipping limit (post hoc)", fontsize=8, color=INK2, ha="right")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)
    ax.set_title("Net value under the three pre-registered cost scenarios (hypothetical)", loc="left")
    _save(fig, fig_dir / "fig10_cost_scenarios.png")
    names.append("fig10_cost_scenarios.png")
    return names


# --------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="experiment-v2-002")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None, help="Default: artifacts/analysis/<run-id>-report")
    parser.add_argument("--figures-dest", default="docs/figures", help="Where the report figures are copied for the documents")
    parser.add_argument("--replace", action="store_true", help="Replace an existing output directory of this analysis")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "artifacts" / "runs" / args.run_id
    out_dir = PROJECT_ROOT / (args.output_dir or f"artifacts/analysis/{args.run_id}-report")
    analysis_root = (PROJECT_ROOT / "artifacts" / "analysis").resolve()
    if analysis_root not in out_dir.resolve().parents:
        raise ValueError(f"Output directory must live under {analysis_root}")
    if out_dir.exists():
        if not args.replace:
            raise FileExistsError(f"{out_dir} exists; pass --replace to rebuild this analysis")
        shutil.rmtree(out_dir)
    (out_dir / "tables").mkdir(parents=True)
    (out_dir / "figures").mkdir()

    inp = load_inputs(run_dir, PROJECT_ROOT / args.data_dir)
    attrs = order_attributes(inp["orders"], inp["items"], inp["sellers"], inp["customers"])
    f = Facts()
    tables: dict[str, pd.DataFrame] = {}
    add_claims(f, inp)
    population_and_motivation(f, inp)
    raw_table_facts(f, inp, PROJECT_ROOT / args.data_dir)
    split_facts(f, inp)
    monthly_timeline(f, inp, tables)
    named = leaderboards(f, inp, tables)
    frontier = capacity_frontier(f, inp, tables, named)
    calibration(f, inp, tables)
    concentration(f, inp, attrs, tables)
    m2 = milestone2(f, inp, attrs, tables)
    bootstrap(f, inp, m2, tables)
    cost_scenarios(f, inp, frontier, tables)
    model_facts(f, inp)

    for name, df in tables.items():
        df.to_csv(out_dir / "tables" / f"{name}.csv", index=False, float_format="%.6f", lineterminator="\n")
    figures = make_figures(out_dir / "figures", f, tables, inp, named, m2)

    dest = PROJECT_ROOT / args.figures_dest
    dest.mkdir(parents=True, exist_ok=True)
    for name in figures:
        shutil.copy2(out_dir / "figures" / name, dest / name)

    inputs = {
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p)
        for p in sorted(run_dir.glob("*.csv")) + sorted(run_dir.glob("*.json"))
    }
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "run_id": args.run_id,
        "run_verification_status": inp["manifest"].get("verification_status"),
        "run_code_sha256": inp["manifest"].get("code_version", {}).get("combined_sha256"),
        "input_sha256": inputs,
        "raw_data_sha256": RAW_DATA_SHA256,
        "figures": figures,
        "tables": sorted(f"{n}.csv" for n in tables),
        "facts": f.items,
    }
    (out_dir / "report_facts.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "facts": len(f.items), "figures": figures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
