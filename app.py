import asyncio
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from browser_use import BrowserSession



DATA_PATH = Path(
   "/Users/saitejagudidevini/Documents/Dev/StarshipChatbot/output/hierarchical_crawl_www_mhsindiana_com_20251005_172659.json"
)

app = FastAPI(title="Starship Semantic Scraper")


def load_entries() -> List[Dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    candidates: Iterable[Dict[str, str]] = []
    if isinstance(payload, dict):
        elements = payload.get("semantic_elements")
        if isinstance(elements, dict):
            candidates = elements.values()
        elif isinstance(elements, list):
            candidates = elements
        elif all(isinstance(value, dict) for value in payload.values()):
            candidates = payload.values()
    elif isinstance(payload, list):
        candidates = payload

    entries = [
        item
        for item in candidates
        if isinstance(item, dict)
        and item.get("semantic_path")
        and item.get("element_type")
    ]
    if not entries:
        raise RuntimeError("No valid entries in crawl JSON.")
    return entries


ENTRIES = load_entries()
ENTRIES_BY_PATH = {entry["semantic_path"]: entry for entry in ENTRIES}


def infer_topic(semantic_path: str) -> str:
    tail = semantic_path.rsplit("/", 1)[-1]
    topic = tail.replace("-", " ").replace("_", " ").strip()
    return topic or semantic_path


def normalize_url(raw: str) -> str:
    return raw.replace(" ", "%20")



async def scrape_semantic_paragraph(entry: Dict[str, str]) -> str:
    url = normalize_url(entry.get("original_url") or entry["semantic_path"])
    topic = infer_topic(entry["semantic_path"])
    element_type = entry.get("element_type", "heading").lower()

    browser = BrowserSession(headless=True)
    try:
        await browser.start()
        page = await browser.new_page()
        await page.goto(url)
        await asyncio.sleep(1.5)

        selector_map = {
            "heading": ["h1", "h2", "h3", "h4", "h5", "h6"],
            "link": ["a", "[role=\"link\"]"],
            "button": [
                "button",
                "[role=\"button\"]",
                "a[role=\"button\"]",
                "input[type=\"button\"]",
                "input[type=\"submit\"]",
            ],
        }
        selectors = selector_map.get(element_type, [element_type])

        script = """(topic, selectors) => {
            const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
            const target = normalize(topic);
            const list = Array.isArray(selectors) ? selectors : [selectors];

            const matchesSelector = (node) => list.some((selector) => {
                try {
                    return node.matches(selector);
                } catch (error) {
                    return node.tagName && node.tagName.toLowerCase() === selector.toLowerCase();
                }
            });

            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
            let anchorFound = false;

            while (walker.nextNode()) {
                const node = walker.currentNode;
                if (!(node instanceof HTMLElement)) {
                    continue;
                }

                if (!anchorFound) {
                    if (!matchesSelector(node)) {
                        continue;
                    }
                    const text = normalize(node.textContent);
                    if (!text) {
                        continue;
                    }
                    if (text === target || text.includes(target)) {
                        anchorFound = true;
                    }
                    continue;
                }

                if (node.tagName.toLowerCase() === 'p') {
                    const paragraphText = (node.textContent || '').trim();
                    if (paragraphText) {
                        return paragraphText;
                    }
                }
            }
            return '';
        }"""

        paragraph = ""
        for _ in range(5):
            paragraph = (await page.evaluate(script, topic, selectors)) or ""
            paragraph = paragraph.strip()
            if paragraph:
                break
            await asyncio.sleep(0.8)
    finally:
        try:
            await browser.stop()
        except Exception:
            pass

    if not paragraph:
        raise RuntimeError(f"Could not extract paragraph beneath '{topic}' on {url}.")

    print(f"[scraper] {topic}: {paragraph[:120]}")
    return paragraph



@app.get("/", response_class=HTMLResponse)
async def index():
    buttons = "".join(
        f'<li><button class="path-btn" data-path="{html.escape(entry["semantic_path"], quote=True)}">'
        f'{html.escape(entry["semantic_path"])}</button></li>'
        for entry in ENTRIES
    )
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Starship Semantic Explorer</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; }}
#paths {{ list-style: none; padding: 0; }}
#paths li {{ margin-bottom: 0.5rem; }}
.path-btn {{ padding: 0.4rem 0.75rem; cursor: pointer; }}
#result {{ margin-top: 2rem; }}
#error {{ color: #b00020; }}
#paragraph {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>Semantic Paths</h1>
<ul id="paths">{buttons}</ul>
<section id="result">
<h2>Extracted Paragraph</h2>
<p id="selected"></p>
<pre id="paragraph"></pre>
<p id="error"></p>
</section>
<script>
const paragraph = document.getElementById("paragraph");
const error = document.getElementById("error");
const selected = document.getElementById("selected");
document.querySelectorAll(".path-btn").forEach((button) => {{
    button.addEventListener("click", async () => {{
        const semanticPath = button.dataset.path;
        paragraph.textContent = "";
        error.textContent = "";
        selected.textContent = "Loading…";
        try {{
            const response = await fetch(`/extract?semantic_path=${{encodeURIComponent(semanticPath)}}`);
            if (!response.ok) {{
                const issue = await response.json().catch(() => ({{ detail: response.statusText }}));
                throw new Error(issue.detail || "Unknown error");
            }}
            const data = await response.json();
            console.log('Extract response', data);
            paragraph.textContent = data.paragraph;
            selected.textContent = 'Topic: ' + data.topic + ' — Source: ' + data.source;
        }} catch (err) {{
            selected.textContent = 'Topic: ' + semanticPath;
            error.textContent = err.message;
        }}
    }});
}});
</script>
</body>
</html>"""
    return HTMLResponse(html_doc)


@app.get("/paths", response_class=JSONResponse)
async def list_paths():
    return ENTRIES


@app.get("/extract")
async def extract_paragraph(
    semantic_path: str = Query(..., description="Semantic path to extract.")
):
    entry = ENTRIES_BY_PATH.get(semantic_path)
    if not entry:
        raise HTTPException(status_code=404, detail="Semantic path not found.")
    try:
        paragraph = await scrape_semantic_paragraph(entry)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {
        "semantic_path": semantic_path,
        "source": entry.get("original_url") or entry["semantic_path"],
        "topic": infer_topic(entry["semantic_path"]),
        "element_type": entry.get("element_type"),
        "paragraph": paragraph,
    }