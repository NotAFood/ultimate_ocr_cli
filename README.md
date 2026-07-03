# unlimited-ocr-local

Python client for [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), a state-of-the-art document parsing model that extracts structured markdown from PDFs and images — preserving tables, headings, and layout.

Includes a self-contained HTML visualizer that overlays the model's bounding-box output on the original document pages.

## Server setup

The model runs via vLLM on a machine with an NVIDIA GPU (≥8 GB VRAM for most documents).

```bash
cd server
# Edit docker-compose.yml and replace YOUR_HOST_IP with your server's IP
docker compose up -d
```

First startup takes 10–20 minutes: the model weights download and CUDA graphs compile. Subsequent starts are fast. Check readiness with:

```bash
curl http://YOUR_HOST_IP:8000/health
```

> **GPU notes:** The default image targets RTX/A-series GPUs. For H100/H200, use `vllm/vllm-openai:unlimited-ocr-cu129` in `docker-compose.yml`.

## Client setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv venv && uv sync
```

Set your server address in `main.py`:

```python
SERVER_IP = "YOUR_HOST_IP"
PORT = "8000"
```

## Usage

### OCR a PDF or image

```bash
uv run python main.py document.pdf -o document.md
```

Output is clean markdown: headings, paragraphs, and GFM tables. Headers, footers, and page numbers are stripped.

### Generate a bounding-box visualizer

Add `--gen-viz` to produce a self-contained HTML viewer alongside the markdown:

```bash
uv run python main.py document.pdf -o document.md --gen-viz
```

This writes `document.md`, `document.pages.json` (raw OCR intermediate), and `document.html`. Open the HTML in any browser — all page images and OCR data are embedded. Navigate pages with `←` `→` or the arrow keys; hover any box to see its block type and extracted content.

To regenerate the visualizer from an existing `.pages.json` without re-running OCR:

```bash
uv run python visualize.py document.pages.json document.pdf -o document.html
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o OUTPUT` | stdout | Write markdown output to file |
| `--dpi DPI` | 150 | PDF render resolution (higher = better quality, slower) |
| `--save-raw` | off | Save per-page raw OCR to `<output>.pages.json` |
| `--gen-viz` | off | Generate HTML visualizer at `<output>.html` (implies `--save-raw`) |
