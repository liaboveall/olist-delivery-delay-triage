"""Unit tests for the report layer: document consistency checker, report-analysis helpers and deck builder."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load("check_docs_consistency")
analysis = _load("report_analysis")
deck = _load("build_presentation")

ALLOWED = sorted({0.173591, 117.0, 1885.0, 0.042123, -0.134171, 2000.0, 0.062069})


@pytest.mark.unit
def test_checker_flags_number_absent_from_facts() -> None:
    text = "The queue caught 117 of 1,885 alerts (17.4% recall), and 44.96% at 30%."
    bad = [p["token"] for p in checker.scan_numbers(text, ALLOWED)]
    assert bad == ["44.96%", "30%"]


@pytest.mark.unit
def test_checker_accepts_rounded_forms_and_units() -> None:
    text = "Recall 17.36% (0.1736), gap 4.2 percentage points, interval −13.4 pp, 2,000 draws, 6.2%."
    assert checker.scan_numbers(text, ALLOWED) == []


@pytest.mark.unit
def test_checker_ignores_dates_identifiers_and_structure() -> None:
    text = (
        "## 4.2 Results\nOn 2018-05-26 18:18:03 the run experiment-v2-001 scored val_2018_04 (C05) "
        "in 4 folds over 2018; see Section 3.5 and 4.6 节, per 1,000 orders.\n"
        "```\npython -m olist_delay verify --run-id 12345\n```\n"
    )
    assert checker.scan_numbers(text, ALLOWED) == []


@pytest.mark.unit
def test_checker_honours_ignore_marker() -> None:
    text = "Retracted values 0.1539 and 0.7303. <!-- check:ignore -->\nKept 0.9999."
    bad = [p["token"] for p in checker.scan_numbers(text, ALLOWED)]
    assert bad == ["0.9999"]


@pytest.mark.unit
def test_checker_reports_missing_image_and_deck_mismatch(tmp_path: Path) -> None:
    (tmp_path / "ok.png").write_bytes(b"x")
    md = tmp_path / "doc.md"
    md.write_text("![a](ok.png)\n![b](missing.png)\n", encoding="utf-8")
    assert checker.check_images(md) == ["missing.png"]
    html = tmp_path / "deck.html"
    html.write_text('<img src="x" />', encoding="utf-8")
    result = checker.check_html_deck(md, html)
    assert result == {"referenced": 2, "embedded": 1, "ok": False}


@pytest.mark.unit
def test_checker_run_end_to_end(tmp_path: Path) -> None:
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"facts": {"a": {"value": 0.25}, "b": {"value": [3, 5]}}}), encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"budget": 0.1, "reps": 400}), encoding="utf-8")
    good = tmp_path / "good.md"
    good.write_text("A quarter (25%) at depth 3 or 5 with 400 draws and a 10% budget.", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("We ran 500 draws.", encoding="utf-8")
    assert checker.run([good], facts, [cfg], None, None)["passed"] is True
    report = checker.run([good, bad], facts, [cfg], None, None)
    assert report["passed"] is False
    assert report["documents"][str(bad)]["unknown_numbers"][0]["token"] == "500"


@pytest.mark.unit
def test_lead_band_edges() -> None:
    bands = analysis.lead_band(pd.Series([0, 10, 11, 20, 21, 30, 31, np.nan]))
    assert bands.tolist()[:7] == ["<=10d", "<=10d", "11-20d", "11-20d", "21-30d", "21-30d", ">30d"]
    assert pd.isna(bands.iloc[7])


@pytest.mark.unit
def test_policy_metrics_counts() -> None:
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    sel = np.array([True, True, False, False, False, False, False, False, False, False])
    m = analysis.policy_metrics(y, sel)
    assert (m["alerts"], m["tp"], m["fp"], m["positives"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["lift"] == pytest.approx(2.5)


@pytest.mark.unit
def test_daily_floor_selected_uses_floor_per_day() -> None:
    rows = []
    for day, n in [("2018-06-01", 15), ("2018-06-02", 9), ("2018-06-03", 20)]:
        for i in range(n):
            rows.append(
                {
                    "order_id": f"{day}-{i:02d}",
                    "order_purchase_timestamp": pd.Timestamp(f"{day} 10:00:00") + pd.Timedelta(minutes=i),
                    "order_estimated_delivery_date": pd.Timestamp(day) + pd.Timedelta(days=20),
                    "score": float(i),
                }
            )
    df = pd.DataFrame(rows)
    sel = analysis.daily_floor_selected(df, "score", 0.10)
    per_day = df.assign(sel=sel).groupby(df["order_purchase_timestamp"].dt.date)["sel"].sum().tolist()
    assert per_day == [1, 0, 2]
    # Highest score of the day wins.
    assert sorted(df.loc[sel, "order_id"]) == ["2018-06-01-14", "2018-06-03-18", "2018-06-03-19"]


@pytest.mark.unit
def test_deck_builder_fails_on_missing_image(tmp_path: Path) -> None:
    (tmp_path / "presentation").mkdir()
    (tmp_path / "presentation" / "slides.en.md").write_text(
        '---\ntitle: "T"\n---\n\n# Slide\n\n![x](../docs/figures/nope.png)\n', encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        deck.build_html_slides(tmp_path)
