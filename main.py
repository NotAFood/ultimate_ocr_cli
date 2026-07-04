import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

import pymupdf
import requests

SERVER_IP = "localhost"
PORT = "8001"
WORKERS = 4


def build_api_url(server_ip: str, port: str) -> str:
    return f"http://{server_ip}:{port}/v1/chat/completions"


API_URL = build_api_url(SERVER_IP, PORT)


class OcrPageError(Exception):
    """Raised when one or more pages fail OCR; the whole document is aborted."""

    def __init__(self, failed_pages: list[int], total_pages: int):
        self.failed_pages = failed_pages
        self.total_pages = total_pages
        pages = ", ".join(str(p) for p in failed_pages)
        super().__init__(f"OCR failed on page(s) {pages} of {total_pages}")


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

# OCR emits LaTeX math wrapped in \( \) (inline) and \[ \] (display).
# Convert to $ $ / $$ $$ for MathJax-based markdown readers.
_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)


def _convert_math_delimiters(text: str) -> str:
    text = _DISPLAY_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", text)
    text = _INLINE_MATH_RE.sub(lambda m: f"${m.group(1)}$", text)
    return text


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
        content = _convert_math_delimiters(content)
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


def query_ocr(
    base64_image: str,
    multi_page: bool = False,
    api_url: str = API_URL,
    idle_timeout: float = 180,
) -> str | None:
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
        "stream": True,
        "extra_body": {
            "custom_params": {
                "ngram_size": 35,
                "window_size": window_size,
            }
        },
    }

    # timeout=(connect, read) — read is measured between chunks, not over the
    # whole request, so a slow page keeps resetting it as tokens stream in.
    response = requests.post(
        api_url, json=payload, timeout=(10, idle_timeout), stream=True
    )
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        return None

    parts = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            break
        delta = json.loads(data)["choices"][0]["delta"].get("content")
        if delta:
            parts.append(delta)
    return "".join(parts)


def process_pdf(
    pdf_path: Path,
    dpi: int = 150,
    api_url: str = API_URL,
    workers: int = WORKERS,
    idle_timeout: float = 180,
) -> tuple[str, list[str]]:
    """Return (cleaned_markdown, raw_pages) where raw_pages is one string per page.

    Rendering and OCR are pipelined: each page is rasterized on the main thread
    and its OCR request submitted to a worker pool immediately, so rendering page
    i+1 overlaps with in-flight OCR calls for earlier pages. If any page fails,
    remaining work is cancelled and the whole document is aborted (no partial
    output) rather than returning a document with holes in it.
    """
    doc = pymupdf.open(pdf_path)
    n = len(doc)
    print(f"Processing {n} page{'s' if n != 1 else ''} with {workers} worker{'s' if workers != 1 else ''}...")

    raw_pages: list[str | None] = [None] * n
    failed_pages: list[int] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures: dict[Future, int] = {}
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc, 1):
            print(f"  page {i}/{n} rendering...", flush=True)
            page_bytes = page.get_pixmap(matrix=mat).tobytes("jpeg")
            b64 = encode_image_bytes(page_bytes)
            futures[
                pool.submit(
                    query_ocr,
                    b64,
                    multi_page=(n > 1),
                    api_url=api_url,
                    idle_timeout=idle_timeout,
                )
            ] = i
            print(f"  page {i}/{n} queued for OCR", flush=True)

        for future in as_completed(futures):
            page_num = futures[future]
            text = future.result()
            if text is None:
                failed_pages.append(page_num)
                print(f"  page {page_num}/{n} failed", file=sys.stderr, flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break
            raw_pages[page_num - 1] = text
            print(f"  page {page_num}/{n} done", flush=True)
    finally:
        # wait=False: an in-flight requests.post() can't be cancelled and won't
        # return until it times out. Don't block program exit on it — see the
        # KeyboardInterrupt handler in main() for the other half of this.
        pool.shutdown(wait=False, cancel_futures=True)
        doc.close()

    if failed_pages:
        raise OcrPageError(sorted(failed_pages), n)

    return clean_ocr("\n\n".join(raw_pages)), raw_pages  # type: ignore[arg-type]


def process_image(image_path: Path, api_url: str = API_URL) -> str | None:
    with open(image_path, "rb") as f:
        b64 = encode_image_bytes(f.read())
    raw = query_ocr(b64, multi_page=False, api_url=api_url)
    return clean_ocr(raw) if raw else None


def main():
    parser = argparse.ArgumentParser(description="Baidu Unlimited-OCR wrapper")
    parser.add_argument("input", type=Path, help="PDF or image file to OCR")
    parser.add_argument(
        "-o", "--output", type=Path, help="Write output to file instead of stdout"
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="DPI for PDF rendering (default: 150)"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save per-page raw OCR to <output>.pages.json",
    )
    parser.add_argument(
        "--gen-viz",
        action="store_true",
        help="Generate self-contained HTML visualizer at <output>.html (implies --save-raw)",
    )
    parser.add_argument(
        "--server-ip",
        default=SERVER_IP,
        help=f"OCR server host/IP (default: {SERVER_IP})",
    )
    parser.add_argument(
        "--port", default=PORT, help=f"OCR server port (default: {PORT})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=f"Concurrent OCR requests for PDF pages (default: {WORKERS})",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=180,
        help="Max seconds of silence between streamed tokens before a page "
        "request is abandoned (default: 180). Raise this if you run many "
        "concurrent workers against one server, since per-request token "
        "cadence drops as concurrency rises.",
    )
    args = parser.parse_args()

    path: Path = args.input
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    api_url = build_api_url(args.server_ip, args.port)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            result, raw_pages = process_pdf(
                path,
                dpi=args.dpi,
                api_url=api_url,
                workers=args.workers,
                idle_timeout=args.idle_timeout,
            )
        except OcrPageError as e:
            print(f"{e} — aborting, no output written.", file=sys.stderr)
            sys.exit(1)
        if (args.save_raw or args.gen_viz) and args.output:
            raw_path = args.output.with_suffix(".pages.json")
            raw_path.write_text(
                json.dumps(
                    [{"page": i + 1, "raw": r} for i, r in enumerate(raw_pages)]
                ),
                encoding="utf-8",
            )
            print(f"Raw OCR saved to {raw_path}")
        if args.gen_viz and args.output:
            from visualize import build_html

            pages_data = [{"page": i + 1, "raw": r} for i, r in enumerate(raw_pages)]
            html = build_html(pages_data, path)
            viz_path = args.output.with_suffix(".html")
            viz_path.write_text(html, encoding="utf-8")
            print(
                f"Visualizer saved to {viz_path} ({viz_path.stat().st_size / 1_048_576:.1f} MB)"
            )
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        print(f"OCR-ing {path.name}...")
        result = process_image(path, api_url=api_url)
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
    try:
        main()
    except KeyboardInterrupt:
        # Worker threads may be blocked inside a blocking requests.post() call,
        # which can't be cancelled and keeps the interpreter alive at normal
        # exit until it times out. Bypass that and exit immediately.
        print("\nInterrupted.", file=sys.stderr)
        os._exit(130)
