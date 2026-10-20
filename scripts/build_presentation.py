"""Build the self-contained HTML deck from presentation/slides.en.md.

Supported Markdown subset per slide: `# title`, `### subtitle`, paragraphs, flat bullet / numbered lists,
pipe tables, `> takeaway` call-outs and `![alt](path)` images (embedded as base64). A slide that has both text
and an image is laid out in two columns. A missing image is an error, so the deck can never silently ship
without a figure it references.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

SLIDE_W, SLIDE_H = 1280, 720


def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def _png_aspect(path: Path) -> float:
    """Width / height read from the PNG header."""
    head = path.read_bytes()[16:24]
    width, height = int.from_bytes(head[:4], "big"), int.from_bytes(head[4:], "big")
    return width / height if height else 1.0


def _render_slide(slide_md: str, md_dir: Path) -> tuple[str, str, str]:
    title, subtitle = "", ""
    blocks: list[str] = []  # text and tables, in source order
    images: list[tuple[str, float]] = []
    list_buf: list[str] = []
    table_rows: list[list[str]] = []

    def flush_list() -> None:
        if list_buf:
            blocks.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in list_buf) + "</ul>")
            list_buf.clear()

    def flush_table() -> None:
        if table_rows:
            head, body = table_rows[0], table_rows[2:]
            aligns = ["right" if c.strip().endswith(":") and not c.strip().startswith(":") else "left" for c in table_rows[1]]
            th = "".join(f'<th style="text-align:{a}">{_inline(c)}</th>' for c, a in zip(head, aligns))
            trs = "".join("<tr>" + "".join(f'<td style="text-align:{a}">{_inline(c)}</td>' for c, a in zip(r, aligns)) + "</tr>" for r in body)
            blocks.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
            table_rows.clear()

    for line in slide_md.splitlines():
        s = line.strip()
        if not s:
            flush_list()
            flush_table()
            continue
        if s.startswith("|") and s.endswith("|"):
            flush_list()
            table_rows.append([c.strip() for c in s.strip("|").split("|")])
            continue
        flush_table()
        img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if img:
            flush_list()
            path = (md_dir / img.group(2)).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Slide image not found: {img.group(2)} (resolved to {path})")
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            images.append((f'<img src="data:image/png;base64,{b64}" alt="{html.escape(img.group(1))}" />', _png_aspect(path)))
        elif s.startswith("# "):
            title = _inline(s[2:])
        elif s.startswith("### "):
            subtitle = _inline(s[4:])
        elif s.startswith("> "):
            flush_list()
            blocks.append(f'<div class="takeaway">{_inline(s[2:])}</div>')
        elif re.match(r"^([-*]|\d+\.)\s+", s):
            list_buf.append(re.sub(r"^([-*]|\d+\.)\s+", "", s))
        else:
            flush_list()
            blocks.append(f"<p>{_inline(s)}</p>")
    flush_list()
    flush_table()

    content = "\n".join(blocks)
    imgs = "".join(tag for tag, _ in images)
    if images and blocks:
        wide = max(a for _, a in images) > 1.8
        layout = "stack" if wide else "cols"
        body = f'<div class="{layout}"><div class="txt">{content}</div><div class="fig">{imgs}</div></div>'
    elif images:
        body = f'<div class="stack"><div class="fig">{imgs}</div></div>'
    else:
        body = content
    return title, subtitle, body


def build_html_slides(root: Path) -> Path:
    md_path = root / "presentation" / "slides.en.md"
    out_path = root / "presentation" / "slides.en.html"
    text = md_path.read_text(encoding="utf-8")
    deck_title = "Presentation"
    if text.startswith("---"):
        _, front, text = text.split("---", 2)
        m = re.search(r'^title:\s*"?(.+?)"?$', front, re.M)
        deck_title = m.group(1) if m else deck_title
    raw_slides = [s.strip() for s in re.split(r"\n---\n", text) if s.strip()]

    sections = []
    for idx, slide_md in enumerate(raw_slides, start=1):
        title, subtitle, body = _render_slide(slide_md, md_path.parent)
        cls = "slide title-slide" if idx == 1 else "slide"
        sub = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
        sections.append(
            f'<section class="{cls}" id="slide-{idx}"><header><h1>{title}</h1>{sub}</header>'
            f'<div class="body">{body}</div>'
            f'<footer><span>Olist delivery-delay triage · IS5126</span><span>{idx} / {len(raw_slides)}</span></footer></section>'
        )

    css = f"""
:root {{ --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; --accent:#2a78d6; --wash:#eef4fc; --page:#f2f2ef; --surface:#ffffff; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family:"Segoe UI", system-ui, -apple-system, sans-serif; }}
.slide {{ width:{SLIDE_W}px; height:{SLIDE_H}px; margin:24px auto; background:var(--surface); position:relative; overflow:hidden;
          padding:40px 56px 56px; display:flex; flex-direction:column; box-shadow:0 2px 10px rgba(0,0,0,.08); }}
header h1 {{ font-size:30px; line-height:1.2; margin:0; font-weight:600; letter-spacing:-.2px; }}
header .subtitle {{ font-size:17px; color:var(--ink2); margin:8px 0 0; }}
header {{ border-bottom:2px solid var(--accent); padding-bottom:12px; margin-bottom:18px; }}
.body {{ flex:1; min-height:0; font-size:19px; line-height:1.45; }}
.body p {{ margin:0 0 10px; }}
.body ul {{ margin:0 0 10px; padding-left:22px; }}
.body li {{ margin-bottom:8px; }}
.body strong {{ font-weight:600; }}
code {{ font-family:Consolas, monospace; font-size:.85em; background:#f2f2ef; padding:1px 5px; border-radius:4px; }}
.cols {{ display:grid; grid-template-columns: 46% 54%; gap:28px; height:100%; }}
.cols .txt {{ min-width:0; }}
.cols .fig {{ display:flex; align-items:center; justify-content:center; min-width:0; min-height:0; }}
.cols .fig img {{ max-width:100%; max-height:100%; object-fit:contain; }}
.stack {{ display:flex; flex-direction:column; height:100%; gap:8px; }}
.stack .txt {{ flex:none; }}
.stack .fig {{ flex:1; min-height:0; display:flex; justify-content:center; align-items:flex-end; }}
.stack .fig img {{ max-width:100%; max-height:100%; object-fit:contain; }}
table {{ border-collapse:collapse; width:100%; font-size:16px; margin:6px 0 12px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--rule); }}
th {{ font-weight:600; color:var(--ink2); border-bottom:1.5px solid var(--ink2); }}
.takeaway {{ background:var(--wash); border-left:4px solid var(--accent); padding:10px 14px; margin:8px 0 12px; border-radius:4px; }}
footer {{ position:absolute; left:56px; right:56px; bottom:18px; display:flex; justify-content:space-between; font-size:13px; color:var(--muted); }}
.title-slide header {{ border:none; margin-top:150px; }}
.title-slide header h1 {{ font-size:42px; }}
.title-slide header .subtitle {{ font-size:22px; }}
.title-slide .body {{ border-top:2px solid var(--accent); padding-top:18px; flex:none; }}
@page {{ size:{SLIDE_W}px {SLIDE_H}px; margin:0; }}
@media print {{ body {{ background:#fff; }} .slide {{ margin:0; box-shadow:none; page-break-after:always; break-after:page; }} }}
"""
    js = """
document.addEventListener('keydown', (e) => {
  const slides = [...document.querySelectorAll('.slide')];
  const y = window.scrollY + 40;
  let cur = slides.findIndex(s => s.offsetTop + s.offsetHeight > y);
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { cur = Math.min(cur + 1, slides.length - 1); }
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) { cur = Math.max(cur - 1, 0); }
  else { return; }
  e.preventDefault();
  slides[cur].scrollIntoView({ behavior: 'smooth', block: 'center' });
});
"""
    html_doc = (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8" />\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"<title>{html.escape(deck_title)}</title>\n<style>{css}</style>\n</head>\n<body>\n"
        + "\n".join(sections)
        + f"\n<script>{js}</script>\n</body>\n</html>\n"
    )
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    out = build_html_slides(root_dir)
    print(f"Built HTML presentation: {out} ({out.stat().st_size:,} bytes)")
