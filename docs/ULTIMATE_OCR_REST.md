# Unlimited-OCR REST API

The container exposes an OpenAI-compatible API via vLLM. Most standard `/v1/chat/completions` tooling works, but the model has three hard requirements that differ from a normal chat model.

## Base URL

```
http://<HOST>:8000
```

No authentication by default. Restrict access at the network level (bind to a specific IP in `docker-compose.yml`).

---

## Endpoints

### `GET /health`

Returns `200 OK` when the model is loaded and ready. Useful for startup probes.

```bash
curl http://HOST:8000/health
```

### `GET /v1/models`

Lists available models. Confirms the container is serving `baidu/Unlimited-OCR`.

```json
{
  "object": "list",
  "data": [
    {
      "id": "baidu/Unlimited-OCR",
      "object": "model",
      "max_model_len": 32768,
      "owned_by": "vllm"
    }
  ]
}
```

### `POST /v1/chat/completions`

The primary inference endpoint.

---

## Request format

```json
{
  "model": "baidu/Unlimited-OCR",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<image>document parsing."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<BASE64>"
          }
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 4096,
  "skip_special_tokens": false,
  "extra_body": {
    "custom_params": {
      "ngram_size": 35,
      "window_size": 128
    }
  }
}
```

### Critical rules

**1. The text prompt must start with `<image>`**
The string `<image>document parsing.` is the correct prompt. Any other prefix produces degraded output. The `<image>` token is a model-level instruction, not a placeholder.

**2. `skip_special_tokens` must be `false`**
The model emits structural tokens (`<|det|>`, `<|/det|>`, `<|ref|>`, `<|/ref|>`) that encode layout information. Setting this to `true` strips them and breaks all post-processing.

**3. Pass `custom_params` via `extra_body`**
Without `ngram_size` and `window_size`, the model can enter infinite generation loops on dense or repetitive content.

| Parameter | Single image | Multi-page / PDF |
|-----------|-------------|-----------------|
| `ngram_size` | 35 | 35 |
| `window_size` | 128 | 1024 |

Use `window_size: 1024` when processing pages from a multi-page document to give the model broader context for repetition detection.

### Image encoding

Images must be sent as base64-encoded data URIs. JPEG is recommended:

```
data:image/jpeg;base64,<BASE64_STRING>
```

PNG is also accepted (`data:image/png;base64,...`). For PDFs, render each page to an image first — the endpoint accepts one image per request.

### Recommended parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | `0` | Deterministic output; higher values introduce noise |
| `max_tokens` | `4096` | Sufficient for most pages; dense tables may need more |

---

## Response format

Standard OpenAI chat completion response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "baidu/Unlimited-OCR",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<|det|>title [86, 79, 395, 99]<|/det|>Chapter 1 Introduction\n<|det|>text [85, 163, 913, 210]<|/det|>RKNanoD is a low-cost..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1280,
    "completion_tokens": 847,
    "total_tokens": 2127
  }
}
```

The raw content string contains grounding tokens that must be post-processed (see below).

---

## Output format

Each detected block appears as:

```
<|det|>LABEL [x1, y1, x2, y2]<|/det|>CONTENT
```

Tables include inline HTML:

```
<|det|>table [140, 102, 860, 221]<|/det|><table><tr><td>...</td></tr></table>
```

### Block type labels

| Label | Description |
|-------|-------------|
| `title` | Section heading |
| `text` | Body paragraph |
| `table` | Tabular data (HTML `<table>` in content) |
| `image` | Image region (no text content; crop from source if needed) |
| `image_caption` | Caption associated with a figure |
| `header` | Repeating page header |
| `footer` | Repeating page footer |
| `page_number` | Page number |

### Bounding box coordinates

Coordinates are integers in `[0, 999]`, normalized to the image dimensions:

```
pixel_x = x1 / 999 * image_width
pixel_y = y1 / 999 * image_height
```

Format is top-left → bottom-right: `[x1, y1, x2, y2]`.

---

## Post-processing

Strip grounding tokens to get clean text. The minimal transform:

```python
import re

BLOCK_RE = re.compile(
    r"<\|det\|>(\w+)\s*\[[^\]]+\]<\|/det\|>(.*?)(?=<\|det\|>|\Z)",
    re.DOTALL,
)

for m in BLOCK_RE.finditer(raw_content):
    label   = m.group(1)   # e.g. "title", "text", "table"
    content = m.group(2).strip()
```

See `main.py` (`clean_ocr`) and `visualize.py` (`parse_blocks`) for complete implementations including heading conversion, GFM table rendering, and bounding-box visualization.
