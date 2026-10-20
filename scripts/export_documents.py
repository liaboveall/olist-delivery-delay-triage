"""Export the final report and slide deck to submission formats (HTML and PDF).

- `docs/final-report.en.md` → `docs/final-report.en.html` (pandoc, figures embedded) → `docs/final-report.en.pdf`
- `presentation/slides.en.html` → `presentation/slides.en.pdf` (one 16:9 slide per page)

PDF printing uses a local Chromium-based browser (Microsoft Edge or Google Chrome) in headless mode. The tools are
looked up on PATH and in their standard Windows install locations; the script exits with code 2 if one is missing.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

REPORT_CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --rule:#e1e0d9; --accent:#2a78d6; --wash:#eef4fc; }
html { background:#fff; }
body { font-family:"Segoe UI", system-ui, -apple-system, sans-serif; color:var(--ink); max-width:860px;
       margin:0 auto; padding:24px 28px 48px; font-size:11pt; line-height:1.5; background:#fff; }
h1 { font-size:22pt; line-height:1.2; margin:0 0 6px; }
h2 { font-size:15pt; margin:28px 0 8px; border-bottom:2px solid var(--accent); padding-bottom:4px; break-after:avoid; }
h3 { font-size:12pt; margin:18px 0 6px; break-after:avoid; }
p, li { orphans:3; widows:3; }
img { max-width:100%; height:auto; display:block; margin:8px auto 2px; }
p:has(> img) { break-inside:avoid; break-after:avoid; }
table { border-collapse:collapse; width:100%; font-size:9.5pt; margin:8px 0 14px; break-inside:avoid; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--rule); padding:4px 6px; vertical-align:top; }
th { border-bottom:1.5px solid var(--ink2); text-align:left; }
blockquote { margin:10px 0; padding:8px 14px; background:var(--wash); border-left:4px solid var(--accent); }
code { font-family:Consolas, monospace; font-size:9pt; background:#f2f2ef; padding:0 3px; border-radius:3px; }
hr { border:none; border-top:1px solid var(--rule); margin:20px 0; }
em { color:var(--ink2); }
@page { size:A4; margin:16mm 14mm; }
@media print { body { padding:0; max-width:none; } }
"""


def _find_browser() -> str | None:
    for name in ("msedge", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return next((p for p in BROWSER_CANDIDATES if Path(p).exists()), None)


def _print_pdf(browser: str, html_path: Path, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Browser did not produce {pdf_path}")


def export_report(pandoc: str, browser: str) -> list[Path]:
    md = PROJECT_ROOT / "docs" / "final-report.en.md"
    html_out = PROJECT_ROOT / "docs" / "final-report.en.html"
    pdf_out = PROJECT_ROOT / "docs" / "final-report.en.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        css = Path(tmp) / "report.css"
        css.write_text(REPORT_CSS, encoding="utf-8")
        subprocess.run(
            [
                pandoc,
                str(md),
                "--from=markdown-implicit_figures",
                "--to=html5",
                "--standalone",
                "--embed-resources",
                f"--resource-path={md.parent}",
                f"--css={css}",
                "--metadata=pagetitle:Purchase-Time Delay Triage on Olist",
                f"--output={html_out}",
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    _print_pdf(browser, html_out, pdf_out)
    return [html_out, pdf_out]


def export_slides(browser: str) -> list[Path]:
    html_in = PROJECT_ROOT / "presentation" / "slides.en.html"
    pdf_out = PROJECT_ROOT / "presentation" / "slides.en.pdf"
    _print_pdf(browser, html_in, pdf_out)
    return [pdf_out]


def main() -> int:
    pandoc = shutil.which("pandoc")
    browser = _find_browser()
    if not pandoc or not browser:
        print(f"Missing tool: pandoc={pandoc!r}, browser={browser!r}", file=sys.stderr)
        return 2
    outputs = export_report(pandoc, browser) + export_slides(browser)
    for p in outputs:
        print(f"wrote {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
