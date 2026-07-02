import argparse
import base64
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pymupdf
import requests

SERVER_IP = "100.122.71.69"
PORT = "8000"
API_URL = f"http://{SERVER_IP}:{PORT}/v1/chat/completions"


# Block types to drop entirely (repeated page furniture, no image data from vLLM)
_DROP_LABELS = {"header", "footer", "page_number", "image"}

# Matches one grounded block: <|det|>label [coords]<|/det|>CONTENT
# Content runs until the next <|det|> tag or end of string
_BLOCK_RE = re.compile(
    r"<\|det\|>(\w+)\s*\[[^\]]+\]<\|/det\|>(.*?)(?=<\|det\|>|\Z)",
    re.DOTALL,
)

# Heading-level heuristic based on common datasheet title patterns
_CHAPTER_RE = re.compile(r"^(chapter\s+\d+|appendix)", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\d+\.\d+")


class _TableParser(HTMLParser):
    """Convert an HTML <table> to a GitHub-Flavored Markdown table."""

    def __init__(self):
        super().__init__()
        self._rows: list[list[str]] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("tr",):
            self._rows.append([])
        elif tag in ("td", "th"):
            self._cell = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._rows[-1].append("".join(self._cell).strip())
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)

    def to_markdown(self) -> str:
        if not self._rows:
            return ""
        rows = self._rows
        header = rows[0]
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows[1:]:
            # pad short rows
            while len(row) < len(header):
                row.append("")
            if any(c.strip() for c in row):
                lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def _html_table_to_md(html: str) -> str:
    parser = _TableParser()
    parser.feed(html)
    return parser.to_markdown()


def _title_level(text: str) -> str:
    if _CHAPTER_RE.match(text):
        return "#"
    if _SECTION_RE.match(text):
        return "###"
    return "##"


def clean_ocr(raw: str) -> str:
    """Strip <|det|> grounding tags and convert block types to clean markdown."""
    parts: list[str] = []
    for m in _BLOCK_RE.finditer(raw):
        label = m.group(1)
        content = m.group(2).strip()
        if not content or label in _DROP_LABELS:
            continue
        if label == "title":
            level = _title_level(content)
            # collapse multi-line titles to single line
            parts.append(f"{level} {' '.join(content.splitlines())}")
        elif label == "image_caption":
            parts.append(f"*{content}*")
        elif label == "table":
            table_html = re.search(r"<table>.*?</table>", content, re.DOTALL)
            if table_html:
                parts.append(_html_table_to_md(table_html.group()))
            else:
                parts.append(content)
        else:
            parts.append(content)
    return "\n\n".join(parts)


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def query_ocr(base64_image: str, multi_page: bool = False) -> str | None:
    window_size = 1024 if multi_page else 128
    payload = {
        "model": "baidu/Unlimited-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image>document parsing."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "skip_special_tokens": False,
        "extra_body": {
            "custom_params": {
                "ngram_size": 35,
                "window_size": window_size,
            }
        },
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        return None


def pdf_to_pages(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("jpeg"))
    doc.close()
    return pages


def process_pdf(pdf_path: Path, dpi: int = 150, save_raw_to: Path | None = None) -> str:
    print(f"Converting {pdf_path.name} to images at {dpi} DPI...")
    pages = pdf_to_pages(pdf_path, dpi=dpi)
    n = len(pages)
    print(f"Processing {n} page{'s' if n != 1 else ''}...")

    raw_pages: list[str] = []
    for i, page_bytes in enumerate(pages, 1):
        print(f"  OCR page {i}/{n}...", end=" ", flush=True)
        b64 = encode_image_bytes(page_bytes)
        text = query_ocr(b64, multi_page=(n > 1))
        if text:
            raw_pages.append(text)
            print("done")
        else:
            raw_pages.append("")
            print("failed")

    if save_raw_to is not None:
        import json
        save_raw_to.write_text(
            json.dumps([{"page": i + 1, "raw": r} for i, r in enumerate(raw_pages)]),
            encoding="utf-8",
        )
        print(f"Raw OCR saved to {save_raw_to}")

    return clean_ocr("\n\n".join(raw_pages))


def process_image(image_path: Path) -> str | None:
    with open(image_path, "rb") as f:
        b64 = encode_image_bytes(f.read())
    raw = query_ocr(b64, multi_page=False)
    return clean_ocr(raw) if raw else None


def main():
    parser = argparse.ArgumentParser(description="Baidu Unlimited-OCR wrapper")
    parser.add_argument("input", type=Path, help="PDF or image file to OCR")
    parser.add_argument("-o", "--output", type=Path, help="Write output to file instead of stdout")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF rendering (default: 150)")
    parser.add_argument("--save-raw", action="store_true", help="Save per-page raw OCR to <output>.pages.json (required for visualize.py)")
    args = parser.parse_args()

    path: Path = args.input
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_path = args.output.with_suffix(".pages.json") if args.save_raw and args.output else None
        result = process_pdf(path, dpi=args.dpi, save_raw_to=raw_path)
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        print(f"OCR-ing {path.name}...")
        result = process_image(path)
    else:
        print(f"Unsupported file type: {suffix}", file=sys.stderr)
        sys.exit(1)

    if result is None:
        print("OCR failed.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print("\n--- OCR Output ---")
        print(result)


if __name__ == "__main__":
    main()
