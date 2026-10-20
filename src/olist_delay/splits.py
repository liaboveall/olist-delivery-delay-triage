"""Rolling expanding-window split generator with strict label-maturity filtering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitMasks:
    split_id: str
    split_type: str  # "validation" or "backtest"
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    selection_at: pd.Timestamp
    train_all_cohort_a_mask: pd.Series
    train_matured_mask: pd.Series
    train_in_transit_dropped_mask: pd.Series
    eval_cohort_a_all_mask: pd.Series
    eval_matured_at_selection_mask: pd.Series
    eval_unmatured_at_selection_mask: pd.Series
    eval_path_b_all_mask: pd.Series


def build_split_masks(
    df: pd.DataFrame,
    split_id: str,
    split_type: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp,
    selection_at: str | pd.Timestamp,
    known_col: str = "label_available_at",
) -> SplitMasks:
    """Construct boolean masks for a rolling validation fold or historical backtest split.

    Enforces:
    - Train purchase window `[train_start, train_end)` AND `label_available_at <= train_end`.
    - Orders with `order_purchase_timestamp < train_end` and `label_available_at > train_end` are
      excluded from training (`train_in_transit_dropped_mask`).
    - For validation folds, selection evaluation uses only validation orders matured by `selection_at`
      (`label_available_at <= selection_at`).
    - `known_col` names the column holding the time an order's label became known. The v2 contract uses
      `label_available_at` (delivery); a censoring-aware contract passes `label_known_at`.
    """
    t_start = pd.Timestamp(train_start)
    t_end = pd.Timestamp(train_end)
    e_start = pd.Timestamp(eval_start)
    e_end = pd.Timestamp(eval_end)
    sel_at = pd.Timestamp(selection_at)

    purch = pd.to_datetime(df["order_purchase_timestamp"])
    avail = pd.to_datetime(df[known_col])
    in_a = df["in_cohort_a"].fillna(False).astype(bool)

    train_window_a = in_a & (purch >= t_start) & (purch < t_end)
    train_matured = train_window_a & avail.notna() & (avail <= t_end)
    train_in_transit = train_window_a & (avail.isna() | (avail > t_end))

    eval_window_all_b = (purch >= e_start) & (purch < e_end)
    eval_window_a = in_a & eval_window_all_b

    if split_type == "validation":
        eval_matured = eval_window_a & avail.notna() & (avail <= sel_at)
        eval_unmatured = eval_window_a & (avail.isna() | (avail > sel_at))
    else:
        # For historical backtest, ex-post evaluation is on Cohort A in [test_start, test_end)
        eval_matured = eval_window_a
        eval_unmatured = pd.Series(False, index=df.index, dtype=bool)

    # Assert strict chronological ordering and zero overlap
    overlap = train_matured & eval_matured
    if overlap.any():
        raise AssertionError(f"Split {split_id} has overlapping train and eval orders!")
    if train_matured.any() and eval_matured.any():
        max_train_ts = purch[train_matured].max()
        min_eval_ts = purch[eval_matured].min()
        if not (max_train_ts < min_eval_ts):
            raise AssertionError(
                f"Split {split_id} violates chronological ordering: max_train={max_train_ts} >= min_eval={min_eval_ts}"
            )

    return SplitMasks(
        split_id=split_id,
        split_type=split_type,
        train_start=t_start,
        train_end=t_end,
        eval_start=e_start,
        eval_end=e_end,
        selection_at=sel_at,
        train_all_cohort_a_mask=train_window_a,
        train_matured_mask=train_matured,
        train_in_transit_dropped_mask=train_in_transit,
        eval_cohort_a_all_mask=eval_window_a,
        eval_matured_at_selection_mask=eval_matured,
        eval_unmatured_at_selection_mask=eval_unmatured,
        eval_path_b_all_mask=eval_window_all_b,
    )


def generate_all_splits(
    df: pd.DataFrame, split_contract: dict[str, Any], known_col: str = "label_available_at"
) -> list[SplitMasks]:
    """Build all rolling validation folds and the final historical backtest split from config."""
    splits: list[SplitMasks] = []
    default_sel_at = split_contract.get("selection_at", "2018-05-26 18:18:03")
    for fold_cfg in split_contract.get("validation_folds", []):
        t_start = fold_cfg.get("train_start", "2016-09-01 00:00:00")
        t_end = fold_cfg.get("train_end", fold_cfg.get("train_at"))
        splits.append(
            build_split_masks(
                df=df,
                split_id=fold_cfg["fold_id"],
                split_type="validation",
                train_start=t_start,
                train_end=t_end,
                eval_start=fold_cfg["val_start"],
                eval_end=fold_cfg["val_end"],
                selection_at=fold_cfg.get("selection_at", default_sel_at),
                known_col=known_col,
            )
        )

    bt_cfg = split_contract["backtest"]
    bt_t_start = bt_cfg.get("train_start", "2016-09-01 00:00:00")
    bt_t_end = bt_cfg.get("train_end", bt_cfg.get("train_at"))
    splits.append(
        build_split_masks(
            df=df,
            split_id=bt_cfg.get("split_id", bt_cfg.get("fold_id", "historical_backtest")),
            split_type="backtest",
            train_start=bt_t_start,
            train_end=bt_t_end,
            eval_start=bt_cfg["test_start"],
            eval_end=bt_cfg["test_end"],
            selection_at=bt_cfg.get("decision_freeze_at", bt_t_end),
            known_col=known_col,
        )
    )
    return splits


def summarize_splits(
    df: pd.DataFrame,
    splits: list[SplitMasks],
    label_col: str = "late",
    known_col: str = "label_available_at",
) -> pd.DataFrame:
    """Create the `split_summary.csv` table documenting counts, maturity filtering, and positive rates."""
    rows: list[dict[str, Any]] = []
    for sp in splits:
        y_train = df.loc[sp.train_matured_mask, label_col].astype(float)
        y_eval_mat = df.loc[sp.eval_matured_at_selection_mask, label_col].astype(float)
        y_eval_all = df.loc[sp.eval_cohort_a_all_mask, label_col].astype(float)

        purch = pd.to_datetime(df["order_purchase_timestamp"])
        max_train_purch = purch[sp.train_matured_mask].max() if sp.train_matured_mask.any() else pd.NaT
        min_eval_purch = purch[sp.eval_matured_at_selection_mask].min() if sp.eval_matured_at_selection_mask.any() else pd.NaT
        max_train_avail = (
            pd.to_datetime(df.loc[sp.train_matured_mask, known_col]).max()
            if sp.train_matured_mask.any()
            else pd.NaT
        )

        rows.append(
            {
                "split_id": sp.split_id,
                "split_type": sp.split_type,
                "train_start": str(sp.train_start),
                "train_end": str(sp.train_end),
                "eval_start": str(sp.eval_start),
                "eval_end": str(sp.eval_end),
                "selection_at": str(sp.selection_at),
                "train_cohort_a_total": int(sp.train_all_cohort_a_mask.sum()),
                "train_matured_count": int(sp.train_matured_mask.sum()),
                "train_dropped_in_transit_count": int(sp.train_in_transit_dropped_mask.sum()),
                "train_matured_positives": int(y_train.sum()) if len(y_train) else 0,
                "train_matured_late_rate": round(float(y_train.mean()), 6) if len(y_train) else 0.0,
                "eval_path_b_total": int(sp.eval_path_b_all_mask.sum()),
                "eval_cohort_a_total": int(sp.eval_cohort_a_all_mask.sum()),
                "eval_matured_at_selection_count": int(sp.eval_matured_at_selection_mask.sum()),
                "eval_unmatured_at_selection_count": int(sp.eval_unmatured_at_selection_mask.sum()),
                "eval_matured_positives": int(y_eval_mat.sum()) if len(y_eval_mat) else 0,
                "eval_matured_late_rate": round(float(y_eval_mat.mean()), 6) if len(y_eval_mat) else 0.0,
                "eval_full_cohort_a_positives": int(y_eval_all.sum()) if len(y_eval_all) else 0,
                "eval_full_cohort_a_late_rate": round(float(y_eval_all.mean()), 6) if len(y_eval_all) else 0.0,
                "max_train_purchase_ts": str(max_train_purch),
                "min_eval_purchase_ts": str(min_eval_purch),
                "max_train_label_available_at": str(max_train_avail),
                "chronological_strict_pass": bool(max_train_purch < min_eval_purch),
                "maturity_strict_pass": bool(max_train_avail <= sp.train_end),
            }
        )
    return pd.DataFrame(rows)


def build_cohort_observation_analysis(
    df: pd.DataFrame,
    splits: list[SplitMasks],
    label_col: str = "late",
) -> pd.DataFrame:
    """Summarize maturity selection bias and Cohort A vs Path B observation counts across splits."""
    rows: list[dict[str, Any]] = []
    for sp in splits:
        y_mat = df.loc[sp.eval_matured_at_selection_mask, label_col].astype(float)
        y_unmat = df.loc[sp.eval_unmatured_at_selection_mask, label_col].astype(float)
        y_all_a = df.loc[sp.eval_cohort_a_all_mask, label_col].astype(float)
        path_b_n = int(sp.eval_path_b_all_mask.sum())
        cohort_a_n = int(sp.eval_cohort_a_all_mask.sum())
        rows.append(
            {
                "split_id": sp.split_id,
                "split_type": sp.split_type,
                "selection_at": str(sp.selection_at),
                "eval_path_b_total": path_b_n,
                "eval_cohort_a_total": cohort_a_n,
                "eval_non_cohort_a_count": path_b_n - cohort_a_n,
                "eval_matured_at_selection_count": int(sp.eval_matured_at_selection_mask.sum()),
                "eval_matured_positives": int(y_mat.sum()) if len(y_mat) else 0,
                "eval_matured_late_rate": round(float(y_mat.mean()), 6) if len(y_mat) else 0.0,
                "eval_unmatured_at_selection_count": int(sp.eval_unmatured_at_selection_mask.sum()),
                "eval_unmatured_positives": int(y_unmat.sum()) if len(y_unmat) else 0,
                "eval_unmatured_late_rate": round(float(y_unmat.mean()), 6) if len(y_unmat) else 0.0,
                "eval_full_cohort_a_positives": int(y_all_a.sum()) if len(y_all_a) else 0,
                "eval_full_cohort_a_late_rate": round(float(y_all_a.mean()), 6) if len(y_all_a) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_split_membership_table(
    df: pd.DataFrame,
    splits: list[SplitMasks],
) -> pd.DataFrame:
    """Create an order-level split membership audit table."""
    out = pd.DataFrame(
        {
            "order_id": df["order_id"],
            "order_purchase_timestamp": df["order_purchase_timestamp"],
            "label_available_at": df["label_available_at"],
            "in_cohort_a": df["in_cohort_a"],
        }
    )
    for sp in splits:
        role = np.full(len(df), "none", dtype=object)
        role[sp.train_in_transit_dropped_mask.to_numpy(dtype=bool)] = "train_dropped_in_transit"
        role[sp.train_matured_mask.to_numpy(dtype=bool)] = "train_matured"
        non_a_eval = sp.eval_path_b_all_mask & (~sp.eval_cohort_a_all_mask)
        role[non_a_eval.to_numpy(dtype=bool)] = "eval_non_cohort_a"
        role[sp.eval_unmatured_at_selection_mask.to_numpy(dtype=bool)] = "eval_unmatured"
        role[sp.eval_matured_at_selection_mask.to_numpy(dtype=bool)] = "eval_matured"
        out[f"{sp.split_id}_role"] = role
    return out

