"""Robustness layer: the pre-registered second evaluation period and the post hoc v3 re-run of the 2018 window.

Reads (read-only) three frozen runs:

- `--period2-run` (default `experiment-v3-late2017-001`): configuration frozen before any model had been scored on
  that period, so it is a genuine out-of-sample test of the v2 method, the three v3 rules, censoring-aware labels
  and the shipping-limit signal found post hoc in 2018.
- `--posthoc-run` (default `experiment-v3-2018-001`): the same v3 candidates on the already-inspected 2018 window.
  Everything from it is labelled post hoc.
- `--main-run` (default `experiment-v2-002`): the formal v2 run, for comparisons.

Writes `artifacts/analysis/robustness-report/` (`report_facts.json`, tables, figures) and copies the figures to
`docs/figures/`. Fact keys are prefixed `p2.` (second period) and `ph.` (post hoc 2018) so they never collide with
the main report facts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from olist_delay.artifacts import RAW_DATA_SHA256, sha256_file  # noqa: E402

_spec = importlib.util.spec_from_file_location("report_analysis", Path(__file__).with_name("report_analysis.py"))
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

KEY_CANDIDATES = {
    "region_lead": "rule_region_lead",
    "route_recent": "rule_route_recent_lead",
    "route_recent_ca": "rule_route_recent_ca_lead",
    "promise_slack": "rule_promise_slack",
    "short_lead": "rule_short_promised_lead",
    "seller_history": "rule_seller_history",
    "random": "rule_random",
}


def _load_run(run_id: str) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("smoke_only") or manifest.get("verification_status") != "VERIFIED":
        raise ValueError(f"{run_id} must be a VERIFIED non-smoke run")
    bt = pd.read_csv(run_dir / "backtest_predictions.csv", low_memory=False)
    val = pd.read_csv(run_dir / "validation_predictions.csv", low_memory=False)
    for df in (bt, val):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
        df["in_cohort_a"] = df["in_cohort_a"].astype(str).str.lower().isin(["true", "1"])
    cand = json.loads((run_dir / "candidate_manifest.json").read_text(encoding="utf-8-sig"))
    families = {c["candidate_id"]: c["model_family"] for c in cand["model_candidates"]}
    families.update({c["candidate_id"]: "rule" for c in cand["baselines"]})
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "manifest": manifest,
        "bt": bt,
        "val": val,
        "families": families,
        "candidates": [c["candidate_id"] for c in cand["baselines"]] + [c["candidate_id"] for c in cand["model_candidates"]],
        "selection": json.loads((run_dir / "selection_record.json").read_text(encoding="utf-8-sig")),
        "uncertainty": json.loads((run_dir / "uncertainty_summary.json").read_text(encoding="utf-8-sig")),
        "split_summary": pd.read_csv(run_dir / "split_summary.csv"),
        "config": json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8-sig")),
    }


def leaderboard(run: dict[str, Any]) -> pd.DataFrame:
    val, bt = run["val"], run["bt"]
    a = bt[bt["in_cohort_a"]]
    yb = a["late_calendar_day"].astype(int).to_numpy()
    rows = []
    for cid in run["candidates"]:
        col = f"score_{cid}"
        rec, aps = [], []
        for _, g in val.groupby("split_id"):
            g = g[g["in_cohort_a"] & g["late"].notna()]
            y = g["late"].astype(int).to_numpy()
            rec.append(ra.policy_metrics(y, ra.daily_floor_selected(g, col, 0.10))["recall"])
            aps.append(average_precision_score(y, g[col]))
        pb = ra.policy_metrics(yb, ra.daily_floor_selected(a, col, 0.10))
        rows.append(
            {
                "candidate_id": cid,
                "family": run["families"][cid],
                "val_mean_daily_top10_recall": float(np.mean(rec)),
                "val_min_fold_recall": float(np.min(rec)),
                "val_mean_ap": float(np.mean(aps)),
                "bt_daily_top10_tp": pb["tp"],
                "bt_daily_top10_recall": pb["recall"],
                "bt_daily_top10_precision": pb["precision"],
                "bt_ap": float(average_precision_score(yb, a[col])),
                "bt_roc_auc": float(roc_auc_score(yb, a[col])),
            }
        )
    lb = pd.DataFrame(rows).sort_values("val_mean_daily_top10_recall", ascending=False).reset_index(drop=True)
    lb.insert(0, "val_rank", np.arange(1, len(lb) + 1))
    lb["bt_rank_by_recall"] = lb["bt_daily_top10_recall"].rank(ascending=False, method="min").astype(int)
    sel = run["selection"]
    got = lb.set_index("candidate_id").loc[sel["selected_candidate_id"], "val_mean_daily_top10_recall"]
    ra._require_close(f"{run['run_id']} selected val recall", got, sel["selected_candidate"]["val_mean_daily_top10_recall"])
    return lb


def run_facts(f: ra.Facts, prefix: str, run: dict[str, Any], attrs: pd.DataFrame, audit: pd.DataFrame, post_hoc: bool) -> pd.DataFrame:
    src = f"{run['run_id']}"
    lb = leaderboard(run)
    keyed = lb.set_index("candidate_id")
    sel = run["selection"]
    f.add(f"{prefix}.selected_id", sel["selected_candidate_id"], src, post_hoc=post_hoc)
    f.add(f"{prefix}.selection_outcome", sel["selection_outcome"], src, post_hoc=post_hoc)
    f.add(f"{prefix}.best_rule_id", sel["best_simple_baseline"]["candidate_id"], src, post_hoc=post_hoc)
    f.add(f"{prefix}.best_ml_id", sel["best_ml_candidate"]["candidate_id"], src, post_hoc=post_hoc)
    f.add(f"{prefix}.ml_gain_over_best_rule", sel.get("ml_gain_over_best_rule", np.nan), src, post_hoc=post_hoc)
    f.add(f"{prefix}.n_candidates", len(lb), src, post_hoc=post_hoc)
    named = dict(KEY_CANDIDATES)
    named["selected"] = sel["selected_candidate_id"]
    named["best_ml"] = sel["best_ml_candidate"]["candidate_id"]
    for tag, cid in named.items():
        if cid not in keyed.index:
            continue
        r = keyed.loc[cid]
        for col in ["val_rank", "val_mean_daily_top10_recall", "val_min_fold_recall", "val_mean_ap", "bt_daily_top10_tp",
                    "bt_daily_top10_recall", "bt_daily_top10_precision", "bt_ap", "bt_roc_auc", "bt_rank_by_recall"]:
            f.add(f"{prefix}.lb.{tag}.{col}", r[col], src, post_hoc=post_hoc)
    ml = lb[lb["family"] != "rule"]
    f.add(f"{prefix}.ml_bt_recall_max", float(ml["bt_daily_top10_recall"].max()), src, post_hoc=post_hoc)
    f.add(f"{prefix}.ml_beating_region_lead_bt", int((ml["bt_daily_top10_recall"] > keyed.loc["rule_region_lead", "bt_daily_top10_recall"]).sum()), src, post_hoc=post_hoc)

    bt = run["bt"]
    a = bt[bt["in_cohort_a"]]
    y = a["late_calendar_day"].astype(int).to_numpy()
    f.add(f"{prefix}.bt.path_b", len(bt), src, post_hoc=post_hoc)
    f.add(f"{prefix}.bt.cohort_a", len(a), src, post_hoc=post_hoc)
    f.add(f"{prefix}.bt.late", int(y.sum()), src, post_hoc=post_hoc)
    f.add(f"{prefix}.bt.base_rate", float(y.mean()), src, post_hoc=post_hoc)
    f.add(f"{prefix}.bt.days", int(a["order_purchase_timestamp"].dt.normalize().nunique()), src, post_hoc=post_hoc)
    f.add(f"{prefix}.bt.first_day", str(a["order_purchase_timestamp"].min().date()), src, post_hoc=post_hoc)
    f.add(f"{prefix}.bt.last_day", str(a["order_purchase_timestamp"].max().date()), src, post_hoc=post_hoc)
    sel10 = ra.daily_floor_selected(a, f"score_{sel['selected_candidate_id']}", 0.10)
    pm = ra.policy_metrics(y, sel10)
    for k in ["alerts", "tp", "precision", "recall", "lift"]:
        f.add(f"{prefix}.selected.k10.{k}", pm[k], src, post_hoc=post_hoc)

    wb = run["uncertainty"]["weekly_block_bootstrap"]
    f.add(f"{prefix}.boot.reps", int(wb["weekly_block_bootstrap"]["n_weekly_bootstrap"]), src, post_hoc=post_hoc)
    f.add(f"{prefix}.boot.weeks", int(wb["n_blocks"]), src, post_hoc=post_hoc)
    for m in ["daily_policy_recall_top10", "daily_policy_precision_top10"]:
        f.add(f"{prefix}.boot.{m}.low", wb["weekly_block_bootstrap"][m]["ci95_low"], src, post_hoc=post_hoc)
        f.add(f"{prefix}.boot.{m}.high", wb["weekly_block_bootstrap"][m]["ci95_high"], src, post_hoc=post_hoc)

    ss = run["split_summary"].set_index("split_id")
    bt_row = ss.loc["historical_backtest"]
    f.add(f"{prefix}.split.train_matured", int(bt_row["train_matured_count"]), src, post_hoc=post_hoc)
    f.add(f"{prefix}.split.train_in_transit", int(bt_row["train_dropped_in_transit_count"]), src, post_hoc=post_hoc)
    f.add(f"{prefix}.split.train_late_rate", float(bt_row["train_matured_late_rate"]), src, post_hoc=post_hoc)
    vals = run["split_summary"][run["split_summary"]["split_type"] == "validation"].sort_values("eval_end")
    for _, r in vals.iterrows():
        sid = r["split_id"]
        f.add(f"{prefix}.split.{sid}.eval_matured_late_rate", float(r["eval_matured_late_rate"]), src, post_hoc=post_hoc)
        f.add(f"{prefix}.split.{sid}.eval_full_late_rate", float(r["eval_full_cohort_a_late_rate"]), src, post_hoc=post_hoc)
        f.add(f"{prefix}.split.{sid}.eval_unmatured", int(r["eval_unmatured_at_selection_count"]), src, post_hoc=post_hoc)
        f.add(f"{prefix}.split.{sid}.eval_unmatured_late", int(r["eval_full_cohort_a_positives"] - r["eval_matured_positives"]), src, post_hoc=post_hoc)
        f.add(f"{prefix}.split.{sid}.train_matured", int(r["train_matured_count"]), src, post_hoc=post_hoc)

    # Shipping-limit signal on this run's backtest orders.
    att = attrs.set_index("order_id")
    d = a[["order_id", "order_purchase_timestamp", "order_estimated_delivery_date", "late_calendar_day"]].join(
        att[["shipping_limit_date", "order_delivered_carrier_date"]], on="order_id"
    )
    flag = (
        d["shipping_limit_date"].notna()
        & d["order_delivered_carrier_date"].notna()
        & (d["order_delivered_carrier_date"] > d["shipping_limit_date"])
    ).to_numpy()
    m2 = ra.policy_metrics(y, flag)
    for k in ["alerts", "tp", "precision", "recall", "lift"]:
        f.add(f"{prefix}.m2.{k}", m2[k], src + " + raw tables", post_hoc=post_hoc)
    f.add(f"{prefix}.m2.alerts_per_day", m2["alerts"] / max(1, f.items[f"{prefix}.bt.days"]["value"]), src, post_hoc=post_hoc)
    f.add(f"{prefix}.m2.alert_rate", float(flag.mean()), src, post_hoc=post_hoc)
    f.add(f"{prefix}.m2.precision_gain_vs_selected", m2["precision"] - pm["precision"], src, post_hoc=post_hoc)
    f.add(f"{prefix}.m2.precision_ratio_vs_selected", m2["precision"] / pm["precision"] if pm["precision"] else np.nan, src, post_hoc=post_hoc)
    slack = (d.loc[flag, "order_estimated_delivery_date"] - d.loc[flag, "shipping_limit_date"]).dt.total_seconds() / 86400
    late_flag = d.loc[flag, "late_calendar_day"].astype(int) == 1
    f.add(f"{prefix}.m2.slack_days_median_tp", float(slack[late_flag].median()), src, post_hoc=post_hoc)
    union = ra.policy_metrics(y, flag | sel10)
    for k in ["alerts", "tp", "precision", "recall"]:
        f.add(f"{prefix}.m2.union.{k}", union[k], src, post_hoc=post_hoc)
    # Paired weekly block bootstrap (calendar weeks starting Monday) on this backtest.
    reps = int(run["config"]["uncertainty_contract"]["block_bootstrap_replicates"])
    rng = np.random.default_rng(int(run["config"]["random_seed"]) + 2027)
    week = a["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
    weeks = np.unique(week)
    idx = {w: np.where(week == w)[0] for w in weeks}
    reg10 = ra.daily_floor_selected(a, "score_rule_region_lead", 0.10)
    diffs_rec, diffs_prec = [], []
    for _ in range(reps):
        b = np.concatenate([idx[w] for w in rng.choice(weeks, size=len(weeks), replace=True)])
        yb = y[b]
        pos = yb.sum()
        if pos == 0:
            continue
        diffs_rec.append(((yb & sel10[b]).sum() - (yb & reg10[b]).sum()) / pos)
        p_sel = (yb & sel10[b]).sum() / max(1, sel10[b].sum())
        p_m2 = (yb & flag[b]).sum() / max(1, flag[b].sum())
        diffs_prec.append(p_m2 - p_sel)
    for name, vals in [("selected_minus_region_lead_recall", diffs_rec), ("m2_minus_selected_precision", diffs_prec)]:
        lo, hi = np.quantile(vals, [0.025, 0.975])
        f.add(f"{prefix}.paired.{name}.point", float(np.mean(vals)), src, post_hoc=post_hoc)
        f.add(f"{prefix}.paired.{name}.low", float(lo), src, post_hoc=post_hoc)
        f.add(f"{prefix}.paired.{name}.high", float(hi), src, post_hoc=post_hoc)
        f.add(f"{prefix}.paired.{name}.share_above_zero", float(np.mean(np.asarray(vals) > 0)), src, post_hoc=post_hoc)
    n_a = len(a)
    for sc in run["config"].get("cost_scenarios", []):
        b, c, fpc = sc["benefit_per_tp"], sc["cost_per_contact"], sc["cost_per_fp"]
        for tag, mm in [("selected_k10", pm), ("m2", m2)]:
            net = 1000.0 * (mm["tp"] * b - mm["alerts"] * c - mm["fp"] * fpc) / n_a
            f.add(f"{prefix}.cost.{sc['scenario_id']}.{tag}.net_per_1000", net, src, post_hoc=post_hoc)
    reg = ra.policy_metrics(y, reg10)
    f.add(f"{prefix}.selected_minus_region_lead_recall", pm["recall"] - reg["recall"], src, post_hoc=post_hoc)
    lb.insert(0, "run_id", run["run_id"])
    return lb


def make_figures(fig_dir: Path, f: ra.Facts, lb2: pd.DataFrame, lbh: pd.DataFrame) -> list[str]:
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold"})
    names = []

    # F11 — second period: validation vs backtest daily recall for every candidate
    fig, (ax,) = ra._new(1, 7.2, 5.0)
    new_rules = set(KEY_CANDIDATES[k] for k in ["route_recent", "route_recent_ca", "promise_slack"])
    styles = [
        ("rule", lambda d: d["family"].eq("rule") & ~d["candidate_id"].isin(new_rules), ra.BLUE, "o", "v2 rule"),
        ("new", lambda d: d["candidate_id"].isin(new_rules), ra.INK2, "D", "v3 rule (pre-registered)"),
        ("lr", lambda d: d["family"].eq("logistic_regression"), ra.ORANGE, "s", "Logistic regression"),
        ("dt", lambda d: d["family"].eq("decision_tree"), ra.AQUA, "^", "Shallow decision tree"),
    ]
    for _, sel_fn, col, mk, lab in styles:
        g = lb2[sel_fn(lb2)]
        ax.scatter(g["val_mean_daily_top10_recall"], g["bt_daily_top10_recall"], s=48, color=col, marker=mk,
                   edgecolor=ra.SURFACE, linewidth=1.5, label=lab, zorder=3)
    lim = (0, max(lb2["val_mean_daily_top10_recall"].max(), lb2["bt_daily_top10_recall"].max()) * 1.15)
    ax.plot(lim, lim, color=ra.AXIS, lw=1, zorder=1)
    ax.axhline(0.10, color=ra.GRID, lw=1, zorder=1)
    ax.axvline(0.10, color=ra.GRID, lw=1, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ra._pct(ax)
    ra._pct(ax, "x")
    ax.xaxis.grid(True, color=ra.GRID)
    k = lb2.set_index("candidate_id")
    sel_id = f.items["p2.selected_id"]["value"]
    for cid, text in [(sel_id, f"{sel_id} (selected)"), ("rule_region_lead", "rule_region_lead (v2 winner)")]:
        if cid in k.index:
            r = k.loc[cid]
            ax.annotate(text, (r["val_mean_daily_top10_recall"], r["bt_daily_top10_recall"]),
                        xytext=(6, 6), textcoords="offset points", color=ra.INK, fontsize=8.5)
    ax.set_xlabel("Validation (Jun–Sep 2017): mean daily top-10% recall")
    ax.set_ylabel("Backtest (26 Oct–31 Dec 2017): daily top-10% recall")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=ra.INK2)
    ax.set_title("Second period: validation vs backtest for all candidates", loc="left")
    ra._save(fig, fig_dir / "fig11_second_period_candidates.png")
    names.append("fig11_second_period_candidates.png")

    # F12 — key rules and the shipping-limit signal across the two backtests
    rows = [
        ("rule_region_lead", "route × promise\n(v2 winner)"),
        ("rule_route_recent_lead", "60-day route\n× promise"),
        ("rule_route_recent_ca_lead", "60-day route (CA)\n× promise"),
        ("rule_promise_slack", "promise\nslack"),
        ("rule_short_promised_lead", "shortest\npromise"),
    ]
    k2, kh = lb2.set_index("candidate_id"), lbh.set_index("candidate_id")
    fig, (ax,) = ra._new(1, 10.5, 4.2)
    xs = np.arange(len(rows) + 1)
    v2017 = [k2.loc[c, "bt_daily_top10_recall"] for c, _ in rows] + [f.items["p2.m2.recall"]["value"]]
    v2018 = [kh.loc[c, "bt_daily_top10_recall"] for c, _ in rows] + [f.items["ph.m2.recall"]["value"]]
    ax.bar(xs - 0.18, v2017, width=0.34, color=ra.BLUE, label="Backtest 26 Oct–31 Dec 2017 (pre-registered)")
    ax.bar(xs + 0.18, v2018, width=0.34, color=ra.GRAY, label="Backtest 26 May–29 Aug 2018 (post hoc for v3 rules)")
    for x, v in zip(xs - 0.18, v2017):
        ax.text(x, v + 0.008, f"{v:.0%}", ha="center", fontsize=8, color=ra.INK)
    for x, v in zip(xs + 0.18, v2018):
        ax.text(x, v + 0.008, f"{v:.0%}", ha="center", fontsize=8, color=ra.INK)
    ax.set_xticks(xs, [lab for _, lab in rows] + ["seller missed\nshipping limit"], fontsize=8.5)
    ax.axhline(0.10, color=ra.MUTED, lw=1)
    ax.text(len(rows) - 0.5, 0.103, "random ranking ≈ 10%", color=ra.MUTED, fontsize=8, ha="center")
    ra._pct(ax)
    ax.set_ylabel("Share of late orders flagged (recall)")
    ax.set_ylim(0, max(v2017 + v2018) * 1.18)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=ra.INK2)
    ax.set_title("Purchase-time rules (daily top 10%) vs the shipping-limit signal in two periods", loc="left")
    ra._save(fig, fig_dir / "fig12_rules_across_periods.png")
    names.append("fig12_rules_across_periods.png")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period2-run", default="experiment-v3-late2017-001")
    parser.add_argument("--posthoc-run", default="experiment-v3-2018-001")
    parser.add_argument("--main-run", default="experiment-v2-002")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts/analysis/robustness-report")
    parser.add_argument("--figures-dest", default="docs/figures")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    analysis_root = (PROJECT_ROOT / "artifacts" / "analysis").resolve()
    if analysis_root not in out_dir.resolve().parents:
        raise ValueError(f"Output directory must live under {analysis_root}")
    if out_dir.exists():
        if not args.replace:
            raise FileExistsError(f"{out_dir} exists; pass --replace to rebuild")
        shutil.rmtree(out_dir)
    (out_dir / "tables").mkdir(parents=True)
    (out_dir / "figures").mkdir()

    data_dir = PROJECT_ROOT / args.data_dir
    for name, digest in RAW_DATA_SHA256.items():
        if sha256_file(data_dir / name) != digest:
            raise ValueError(f"Raw data hash mismatch for {name}")
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"])
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})
    attrs = ra.order_attributes(orders, items, sellers, customers)

    p2 = _load_run(args.period2_run)
    ph = _load_run(args.posthoc_run)
    main_run = _load_run(args.main_run)
    audit = pd.read_csv(p2["run_dir"] / "cohort_audit.csv", low_memory=False)

    f = ra.Facts()
    lb2 = run_facts(f, "p2", p2, attrs, audit, post_hoc=False)
    lbh = run_facts(f, "ph", ph, attrs, audit, post_hoc=True)

    # Effect of censoring-aware maturity on the 2018 contract (v3 vs v2, same splits).
    ss_v2 = main_run["split_summary"].set_index("split_id")
    ss_v3 = ph["split_summary"].set_index("split_id")
    f.add("ca.v2.train_matured_backtest", int(ss_v2.loc["historical_backtest", "train_matured_count"]), args.main_run)
    f.add("ca.v3.train_matured_backtest", int(ss_v3.loc["historical_backtest", "train_matured_count"]), args.posthoc_run, post_hoc=True)
    f.add("ca.added_train_backtest", int(ss_v3.loc["historical_backtest", "train_matured_count"] - ss_v2.loc["historical_backtest", "train_matured_count"]), "split summaries", post_hoc=True)
    for sid in ["val_2018_01", "val_2018_04"]:
        f.add(f"ca.v2.{sid}.train_matured", int(ss_v2.loc[sid, "train_matured_count"]), args.main_run)
        f.add(f"ca.v3.{sid}.train_matured", int(ss_v3.loc[sid, "train_matured_count"]), args.posthoc_run, post_hoc=True)
    f.add("ca.added_train_val_2018_04", int(ss_v3.loc["val_2018_04", "train_matured_count"] - ss_v2.loc["val_2018_04", "train_matured_count"]), "split summaries", post_hoc=True)
    orig_unc = PROJECT_ROOT / "artifacts" / "runs" / "experiment-v2-001" / "uncertainty_summary.json"
    if orig_unc.exists():
        f.add("orig.v2_001.boot_reps", int(json.loads(orig_unc.read_text(encoding="utf-8-sig"))["weekly_block_bootstrap"]["n_bootstrap"]), "experiment-v2-001/uncertainty_summary.json")
    f.add("ca.v2.val_2018_04.eval_matured_late",int(ss_v2.loc["val_2018_04", "eval_matured_positives"]), args.main_run)
    f.add("ca.v3.val_2018_04.eval_matured_late", int(ss_v3.loc["val_2018_04", "eval_matured_positives"]), args.posthoc_run, post_hoc=True)
    f.add("ca.v2.val_2018_04.eval_matured_late_rate", float(ss_v2.loc["val_2018_04", "eval_matured_late_rate"]), args.main_run)
    f.add("ca.v3.val_2018_04.eval_matured_late_rate", float(ss_v3.loc["val_2018_04", "eval_matured_late_rate"]), args.posthoc_run, post_hoc=True)
    # Same candidate, same 2018 window, v2 labels vs censoring-aware labels.
    kh = lbh.set_index("candidate_id")
    f.add("ca.v3.region_lead_bt_recall", float(kh.loc["rule_region_lead", "bt_daily_top10_recall"]), args.posthoc_run, post_hoc=True)

    for name, df in {"leaderboard_period2": lb2, "leaderboard_posthoc_2018": lbh}.items():
        df.to_csv(out_dir / "tables" / f"{name}.csv", index=False, float_format="%.6f", lineterminator="\n")
    figures = make_figures(out_dir / "figures", f, lb2, lbh)
    dest = PROJECT_ROOT / args.figures_dest
    dest.mkdir(parents=True, exist_ok=True)
    for name in figures:
        shutil.copy2(out_dir / "figures" / name, dest / name)

    payload = {
        "analysis_version": "robustness-analysis-v1",
        "runs": {"period2": args.period2_run, "posthoc_2018": args.posthoc_run, "main": args.main_run},
        "run_code_sha256": {r["run_id"]: r["manifest"].get("code_version", {}).get("combined_sha256") for r in (p2, ph, main_run)},
        "figures": figures,
        "facts": f.items,
    }
    (out_dir / "report_facts.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "facts": len(f.items), "figures": figures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
