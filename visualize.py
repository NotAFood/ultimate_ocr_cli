"""Generate an HTML visualization of Unlimited-OCR bounding-box output.

Usage:
    uv run python visualize.py samples/doc.pages.json samples/doc.pdf -o samples/doc.html
"""

import argparse
import base64
import json
import re
from pathlib import Path

import pymupdf

BLOCK_RE = re.compile(
    r"<\|det\|>(\w+)\s*\[([^\]]+)\]<\|/det\|>(.*?)(?=<\|det\|>|\Z)",
    re.DOTALL,
)

LABEL_COLORS = {
    "title":        "#58a6ff",
    "text":         "#3fb950",
    "table":        "#d29922",
    "image":        "#bc8cff",
    "image_caption":"#f78166",
    "header":       "#6e7681",
    "footer":       "#6e7681",
    "page_number":  "#6e7681",
}
DEFAULT_COLOR = "#6e7681"


def parse_blocks(raw: str) -> list[dict]:
    blocks = []
    for m in BLOCK_RE.finditer(raw):
        label = m.group(1)
        try:
            coords = [int(x.strip()) for x in m.group(2).split(",")]
        except ValueError:
            continue
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords
        content = m.group(3).strip()
        blocks.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "content": content})
    return blocks


def render_pages(pdf_path: Path, dpi: int = 120) -> list[str]:
    doc = pymupdf.open(pdf_path)
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=72)).decode()
        images.append(f"data:image/jpeg;base64,{b64}")
    doc.close()
    return images


def build_html(pages_json: list[dict], pdf_path: Path, dpi: int = 120) -> str:
    n = len(pages_json)
    print(f"Rendering {n} pages at {dpi} DPI...")
    images = render_pages(pdf_path, dpi=dpi)

    page_data = []
    for i, entry in enumerate(pages_json):
        blocks = parse_blocks(entry.get("raw", ""))
        page_data.append({"img": images[i] if i < len(images) else "", "blocks": blocks})

    # Truncate very long block content for the tooltip
    for p in page_data:
        for b in p["blocks"]:
            if len(b["content"]) > 600:
                b["content"] = b["content"][:600] + "…"

    data_js = json.dumps(page_data, ensure_ascii=False)
    colors_js = json.dumps(LABEL_COLORS, ensure_ascii=False)

    # Legend HTML (deduplicate footer/header into one "chrome" entry for brevity)
    legend_labels = [
        ("title", "Title"),
        ("text", "Text"),
        ("table", "Table"),
        ("image", "Image"),
        ("image_caption", "Caption"),
        ("header", "Chrome"),
    ]
    legend_html = "".join(
        f'<span class="leg"><span class="swatch" style="background:{LABEL_COLORS[k]}"></span>{name}</span>'
        for k, name in legend_labels
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR — {pdf_path.name}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}

:root{{
  --bg:#0d1117;
  --surface:#161b22;
  --border:#30363d;
  --text:#e6edf3;
  --muted:#8b949e;
  --accent:#388bfd;
}}

body{{
  font-family:system-ui,-apple-system,sans-serif;
  background:var(--bg);
  color:var(--text);
  height:100vh;
  display:flex;
  flex-direction:column;
  overflow:hidden;
}}

/* ── Header bar ── */
#bar{{
  flex-shrink:0;
  display:flex;
  align-items:center;
  gap:16px;
  padding:0 20px;
  height:52px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
}}

#filename{{
  font-family:ui-monospace,monospace;
  font-size:12px;
  color:var(--muted);
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  max-width:260px;
}}

.nav{{display:flex;align-items:center;gap:6px}}
.nav button{{
  background:transparent;
  border:1px solid var(--border);
  color:var(--text);
  padding:4px 12px;
  border-radius:6px;
  font-size:12px;
  cursor:pointer;
  line-height:1.5;
  transition:background .12s,border-color .12s;
}}
.nav button:hover:not(:disabled){{background:var(--border);border-color:#484f58}}
.nav button:disabled{{opacity:.35;cursor:default}}
#pager{{
  font-size:12px;
  color:var(--muted);
  min-width:72px;
  text-align:center;
  font-variant-numeric:tabular-nums;
}}

.legend{{
  display:flex;
  gap:12px;
  flex-wrap:wrap;
  margin-left:auto;
}}
.leg{{
  display:flex;
  align-items:center;
  gap:5px;
  font-size:11px;
  color:var(--muted);
  font-family:ui-monospace,monospace;
}}
.swatch{{
  width:9px;height:9px;
  border-radius:2px;
  flex-shrink:0;
}}

/* ── Stage ── */
#stage{{
  flex:1;
  overflow:auto;
  display:flex;
  justify-content:center;
  align-items:flex-start;
  padding:24px;
}}

#wrap{{
  position:relative;
  display:inline-block;
  box-shadow:0 4px 24px rgba(0,0,0,.6);
  line-height:0;
}}

#img{{
  display:block;
  max-width:min(860px,88vw);
  height:auto;
  user-select:none;
}}

#overlay{{
  position:absolute;
  inset:0;
}}

/* ── Boxes ── */
.box{{
  position:absolute;
  border:1.5px solid;
  border-radius:2px;
  cursor:default;
  transition:opacity .1s;
  opacity:.45;
}}
.box:hover{{opacity:1;z-index:20}}

/* ── Tooltip ── */
.tip{{
  display:none;
  position:absolute;
  left:0;top:calc(100% + 4px);
  background:#1c2128;
  border:1px solid var(--border);
  border-radius:8px;
  padding:10px 12px;
  width:min(340px,80vw);
  max-height:220px;
  overflow-y:auto;
  z-index:100;
  pointer-events:none;
  box-shadow:0 8px 24px rgba(0,0,0,.5);
}}
/* flip up if near bottom */
.box.flip-up .tip{{top:auto;bottom:calc(100% + 4px)}}

.box:hover .tip{{display:block}}
.tip-lbl{{
  font-family:ui-monospace,monospace;
  font-size:10px;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:.06em;
  margin-bottom:6px;
}}
.tip-txt{{
  font-size:12px;
  line-height:1.55;
  color:#c9d1d9;
  white-space:pre-wrap;
  word-break:break-word;
}}
</style>
</head>
<body>

<div id="bar">
  <span id="filename">{pdf_path.name}</span>
  <div class="nav">
    <button id="prev" onclick="go(-1)">&#8592;</button>
    <span id="pager">1 / {n}</span>
    <button id="next" onclick="go(1)">&#8594;</button>
  </div>
  <div class="legend">{legend_html}</div>
</div>

<div id="stage">
  <div id="wrap">
    <img id="img" alt="Document page" src="">
    <div id="overlay"></div>
  </div>
</div>

<script>
const PAGES  = {data_js};
const COLORS = {colors_js};
const DEFAULT = "{DEFAULT_COLOR}";
let cur = 0;

function render(idx) {{
  const p = PAGES[idx];
  document.getElementById('img').src = p.img;
  document.getElementById('pager').textContent = `${{idx + 1}} / ${{PAGES.length}}`;
  document.getElementById('prev').disabled = idx === 0;
  document.getElementById('next').disabled = idx === PAGES.length - 1;

  const ov = document.getElementById('overlay');
  ov.innerHTML = '';

  // Determine page height in screen px for flip-up heuristic (after img load)
  const imgEl = document.getElementById('img');

  p.blocks.forEach(b => {{
    const color = COLORS[b.label] || DEFAULT;
    const box = document.createElement('div');
    box.className = 'box';
    box.style.cssText = [
      `left:${{b.x1/999*100}}%`,
      `top:${{b.y1/999*100}}%`,
      `width:${{(b.x2-b.x1)/999*100}}%`,
      `height:${{(b.y2-b.y1)/999*100}}%`,
      `border-color:${{color}}`,
      `background:${{color}}1a`,
    ].join(';');

    // flip tooltip up if box is in the lower 40% of page
    if (b.y1 / 999 > 0.6) box.classList.add('flip-up');

    const tip = document.createElement('div');
    tip.className = 'tip';

    const lbl = document.createElement('div');
    lbl.className = 'tip-lbl';
    lbl.style.color = color;
    lbl.textContent = b.label;
    tip.appendChild(lbl);

    const txt = document.createElement('div');
    txt.className = 'tip-txt';
    txt.textContent = b.content || '—';
    tip.appendChild(txt);

    box.appendChild(tip);
    ov.appendChild(box);
  }});
}}

function go(d) {{
  const next = cur + d;
  if (next < 0 || next >= PAGES.length) return;
  cur = next;
  render(cur);
  document.getElementById('stage').scrollTop = 0;
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') go(1);
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   go(-1);
}});

render(0);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize Unlimited-OCR bounding boxes as HTML")
    parser.add_argument("pages_json", type=Path, help="Path to .pages.json from main.py --save-raw")
    parser.add_argument("pdf", type=Path, help="Original PDF file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output HTML file")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for page rendering (default: 120)")
    args = parser.parse_args()

    pages_json = json.loads(args.pages_json.read_text(encoding="utf-8"))
    html = build_html(pages_json, args.pdf, dpi=args.dpi)
    args.output.write_text(html, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1_048_576
    print(f"Saved {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
