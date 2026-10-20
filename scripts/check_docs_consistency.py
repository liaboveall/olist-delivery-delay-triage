"""Check that the narrative documents only quote numbers that trace to the report facts, and that every figure resolves.

`verify --check-docs` (V08) only confirms that five headline tokens appear in the two final reports. This checker
closes the gap left by that gate:

1. Every image link in the checked Markdown files must resolve to an existing file.
2. The self-contained HTML deck must embed exactly as many images as `slides.en.md` references.
3. Every number in the prose and tables must equal (after rounding to the precision written) a value in
   `report_facts.json`, a numeric leaf of the frozen configs, or a small structural integer (<= 31, years).

Rule 3 does not prove that a number is used in the right place, but it rejects numbers that exist in no artifact,
which is how the v2.0.0 reports went wrong (fabricated frontier rows, subgroup tables and medians).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOCS = [
    "docs/final-report.en.md",
    "docs/final-report.zh-CN.md",
    "presentation/slides.en.md",
    "docs/qa.md",
    "docs/data-card.md",
    "docs/research-contract.md",
    "docs/decision-log.md",
    "docs/test-exposure-log.md",
]
DEFAULT_FACTS = [
    "artifacts/analysis/experiment-v2-002-report/report_facts.json",
    "artifacts/analysis/robustness-report/report_facts.json",
]
DEFAULT_CONFIGS = ["configs/experiment-v2.json", "configs/research-v2.json"]
YEARS = {2016, 2017, 2018, 2026}
IGNORE_MARKER = "check:ignore"
STRUCTURAL = {1000.0}  # normalisation constant in "per 1,000 orders"
SMALL_INT_MAX = 31

IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE_RE = re.compile(r"```.*?```", re.S)
STRIP_PATTERNS = [
    re.compile(r"<!--.*?-->", re.S),  # HTML comments
    re.compile(r"\]\([^)]*\)"),  # link / image targets
    re.compile(r"https?://\S+"),
    re.compile(r"\b[0-9a-f]{64}\b"),  # SHA-256 digests
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"),  # dates and timestamps
    re.compile(r"\b\d{4}-\d{2}\b"),  # year-month
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"),  # times of day
    re.compile(r"\b(?:19|20)\d{2}[–-](?:19|20)?\d{2}\b"),  # year ranges such as 2016–2018
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_\-]*\d[A-Za-z0-9_\-.]*\b"),  # identifiers: C01, val_2018_01, fig03_x.png, v2
    re.compile(r"^\s{0,3}#{1,6}\s+(?:\d+(?:\.\d+)*\.?\s+)?", re.M),  # heading numbering
    re.compile(r"(?:Section|Sections|§|Slide|Slides|Figure|Figures|Table|Tables|Fig\.)\s*\d+(?:\.\d+)*", re.I),
    re.compile(r"(?:图|表|第)\s*\d+(?:\.\d+)*"),
    re.compile(r"\d+(?:\.\d+)*\s*节"),  # Chinese section references such as "4.2 节"
]
NUMBER_RE = re.compile(
    r"(?<![\w.,])(?P<sign>[-−+]?)(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<unit>\s?%|\s?pp\b|\s(?:percentage\s)?points?\b|\s?个百分点|x\b|×)?"
)


def _numeric_leaves(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numeric_leaves(v)


def load_allowed_values(facts_paths: Path | list[Path], config_paths: list[Path]) -> list[float]:
    values: set[float] = set()
    for facts_path in [facts_paths] if isinstance(facts_paths, Path) else facts_paths:
        payload = json.loads(facts_path.read_text(encoding="utf-8"))
        for entry in payload["facts"].values():
            values.update(_numeric_leaves(entry["value"]))
    for cp in config_paths:
        values.update(_numeric_leaves(json.loads(cp.read_text(encoding="utf-8"))))
    return sorted(values)


def _matches(token_value: float, decimals: int, is_pct: bool, allowed: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for c in allowed:
        cand = c * 100.0 if is_pct else c
        if abs(abs(cand) - token_value) <= tol:
            return True
    return False


def scan_numbers(text: str, allowed: list[float]) -> list[dict[str, Any]]:
    """Return the numbers in `text` that match no allowed value, with 1-based line numbers."""
    # A line carrying the marker `check:ignore` (e.g. inside an HTML comment) quotes numbers on purpose,
    # such as a retraction of figures that exist in no artifact.
    text = "\n".join("" if IGNORE_MARKER in line else line for line in text.split("\n"))
    body = FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    for pat in STRIP_PATTERNS:
        body = pat.sub(lambda m: " " * len(m.group(0)) if "\n" not in m.group(0) else "\n" * m.group(0).count("\n"), body)
    problems = []
    for m in NUMBER_RE.finditer(body):
        raw = m.group("num")
        unit = (m.group("unit") or "").strip()
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        is_pct = unit in ("%", "pp", "个百分点") or unit.endswith(("point", "points"))
        if decimals == 0 and not unit and (value <= SMALL_INT_MAX or int(value) in YEARS):
            continue
        if decimals == 0 and not unit and value in STRUCTURAL:
            continue
        if decimals == 0 and is_pct and value in (0, 100):
            continue
        if _matches(value, decimals, is_pct, allowed):
            continue
        line = body.count("\n", 0, m.start()) + 1
        problems.append({"line": line, "token": m.group(0).strip()})
    return problems


def check_images(doc: Path) -> list[str]:
    missing = []
    for target in IMAGE_RE.findall(doc.read_text(encoding="utf-8")):
        if not (doc.parent / target).resolve().exists():
            missing.append(target)
    return missing


def check_html_deck(md: Path, html: Path) -> dict[str, Any]:
    refs = len(IMAGE_RE.findall(md.read_text(encoding="utf-8")))
    embedded = html.read_text(encoding="utf-8").count("<img ") if html.exists() else 0
    return {"referenced": refs, "embedded": embedded, "ok": html.exists() and refs == embedded}


def run(
    docs: list[Path], facts: Path | list[Path], configs: list[Path], deck_md: Path | None, deck_html: Path | None
) -> dict[str, Any]:
    allowed = load_allowed_values(facts, configs)
    facts_list = [facts] if isinstance(facts, Path) else list(facts)
    report: dict[str, Any] = {"facts": [str(p) for p in facts_list], "documents": {}, "passed": True}
    for doc in docs:
        if not doc.exists():
            report["documents"][str(doc)] = {"error": "missing"}
            report["passed"] = False
            continue
        unknown = scan_numbers(doc.read_text(encoding="utf-8"), allowed)
        missing = check_images(doc)
        report["documents"][str(doc)] = {"unknown_numbers": unknown, "missing_images": missing}
        if unknown or missing:
            report["passed"] = False
    if deck_md is not None and deck_html is not None:
        deck = check_html_deck(deck_md, deck_html)
        report["html_deck"] = deck
        report["passed"] = report["passed"] and deck["ok"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check narrative documents against report facts and figure files.")
    parser.add_argument("--facts", nargs="+", default=DEFAULT_FACTS)
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--deck-md", default="presentation/slides.en.md")
    parser.add_argument("--deck-html", default="presentation/slides.en.html")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rel = lambda p: PROJECT_ROOT / p  # noqa: E731
    report = run(
        [rel(d) for d in args.docs],
        [rel(p) for p in args.facts],
        [rel(c) for c in args.configs],
        rel(args.deck_md) if args.deck_md else None,
        rel(args.deck_html) if args.deck_html else None,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        out = rel(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    for doc, res in report["documents"].items():
        name = Path(doc).name
        if "error" in res:
            print(f"[FAIL] {name}: {res['error']}")
            continue
        bad = res["unknown_numbers"]
        status = "PASS" if not bad and not res["missing_images"] else "FAIL"
        print(f"[{status}] {name}: {len(bad)} untraceable number(s), {len(res['missing_images'])} missing image(s)")
        for item in bad[:40]:
            print(f"    line {item['line']}: {item['token']}")
        for img in res["missing_images"]:
            print(f"    missing image: {img}")
    if "html_deck" in report:
        d = report["html_deck"]
        print(f"[{'PASS' if d['ok'] else 'FAIL'}] slides.en.html embeds {d['embedded']} of {d['referenced']} referenced images")
    print("OVERALL:", "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
