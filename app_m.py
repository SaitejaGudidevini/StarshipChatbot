import asyncio
import html
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from browser_use import Agent
from browser_use.browser import BrowserSession
from browser_use import ChatGoogle
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



DATA_PATH = Path(
   "/Users/saitejagudidevini/Documents/Dev/StarshipChatbot/output/hierarchical_crawl_pytorch_org_20250930_213016_filtered.json"
)

app = FastAPI(title="Starship AI-Powered Semantic Scraper")

# Initialize AI clients
groq_client = None
google_llm = None

def initialize_ai_clients():
    """Initialize Groq and Google AI clients"""
    global groq_client, google_llm
    
    # Initialize Groq client
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    
    # Initialize Google LLM for browser-use
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        google_llm = ChatGoogle(
            model="gemini-2.5-pro",
            api_key=google_api_key,
            temperature=0.1
        )

# Initialize clients on startup
initialize_ai_clients()


async def summarize_with_groq(text: str, topic: str, max_length: int = 150) -> Dict[str, str]:
    """Summarize content using Groq API"""
    if not groq_client:
        return {"summary": text[:max_length] + "...", "error": "Groq client not initialized"}
    
    try:
        prompt = f"""
Summarize the following content about '{topic}' in approximately {max_length} words. 
Focus on key information and main points.

Content:
{text}

Summary:"""
        
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=200
        )
        
        summary = completion.choices[0].message.content.strip()
        return {
            "summary": summary,
            "model_used": "llama3-8b-8192",
            "processing_time": "<1s"
        }
    except Exception as e:
        # Handle different text types
        fallback_text = str(text)[:max_length] if text else "No content available"
        return {
            "summary": fallback_text + "...",
            "error": f"Groq summarization failed: {str(e)}"
        }


async def extract_content_with_ai_agent(entry: Dict[str, str]) -> Dict[str, str]:
    """Extract content using browser-use AI Agent"""
    url = normalize_url(entry.get("original_url") or entry["semantic_path"])
    topic = infer_topic(entry["semantic_path"])
    
    if not google_llm:
        # Fallback to basic scraping if Google LLM not available
        paragraph = await scrape_semantic_paragraph(entry)
        return {
            "content": paragraph,
            "extraction_method": "basic_scraping",
            "error": "Google LLM not available"
        }
    
    try:
        # Create AI agent with specific task
        task = f"""
Navigate to {url} and extract comprehensive information about '{topic}'.
Focus on:
1. Main content related to '{topic}'
2. Key details and descriptions
3. Any relevant context or background information
4. Important facts or statistics if present

Provide a detailed but concise summary of the relevant content."""
        
        use_cloud = os.getenv('USE_CLOUD_BROWSER', 'false').lower() == 'true'
        
        agent = Agent(
            task=task,
            llm=google_llm,
            use_cloud=use_cloud
        )
        
        start_time = time.time()
        result = await agent.run()
        processing_time = time.time() - start_time
        
        # Convert AgentHistoryList to string if needed
        content_text = str(result) if result else "No content extracted"
        
        return {
            "content": content_text,
            "extraction_method": "ai_agent",
            "processing_time": f"{processing_time:.1f}s",
            "use_cloud": use_cloud
        }
        
    except Exception as e:
        # Fallback to basic scraping on AI agent failure
        try:
            paragraph = await scrape_semantic_paragraph(entry)
            return {
                "content": paragraph,
                "extraction_method": "basic_scraping_fallback",
                "error": f"AI agent failed: {str(e)}"
            }
        except Exception as fallback_error:
            return {
                "content": f"Failed to extract content for {topic}",
                "extraction_method": "failed",
                "error": f"Both AI and basic extraction failed: {str(e)}, {str(fallback_error)}"
            }


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
<title>Starship AI-Powered Semantic Explorer</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 1200px; }}
.header {{ margin-bottom: 2rem; }}
.controls {{ 
    background: #f5f5f5; 
    padding: 1rem; 
    border-radius: 8px; 
    margin-bottom: 2rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
}}
.control-group {{ display: flex; align-items: center; gap: 0.5rem; }}
#paths {{ list-style: none; padding: 0; }}
#paths li {{ margin-bottom: 0.5rem; }}
.path-btn {{ 
    padding: 0.4rem 0.75rem; 
    cursor: pointer; 
    border: 1px solid #ddd;
    background: white;
    border-radius: 4px;
    transition: background-color 0.2s;
}}
.path-btn:hover {{ background: #f0f0f0; }}
#result {{ margin-top: 2rem; }}
#error {{ color: #b00020; margin-top: 1rem; }}
#paragraph {{ 
    white-space: pre-wrap; 
    background: #f9f9f9; 
    padding: 1rem; 
    border-radius: 4px;
    margin-top: 1rem;
}}
#summary {{ 
    background: #e3f2fd; 
    padding: 1rem; 
    border-radius: 4px; 
    margin-top: 1rem;
    border-left: 4px solid #2196f3;
}}
.extraction-info {{ 
    font-size: 0.9em; 
    color: #666; 
    margin-top: 0.5rem; 
}}
.loading {{ opacity: 0.6; }}
label {{ font-weight: bold; }}
input[type="checkbox"] {{ margin-right: 0.5rem; }}
input[type="range"] {{ width: 100px; }}
#summaryLengthValue {{ font-weight: bold; color: #2196f3; }}
</style>
</head>
<body>
<div class="header">
<h1>🚀 Starship AI-Powered Semantic Explorer</h1>
<p>Extract and summarize content using AI-powered browser automation and Groq summarization</p>
</div>

<div class="controls">
<div class="control-group">
    <label><input type="checkbox" id="useAI" checked> Use AI Extraction</label>
</div>
<div class="control-group">
    <label><input type="checkbox" id="useSummary" checked> Generate Summary</label>
</div>
<div class="control-group">
    <label for="summaryLength">Summary Length:</label>
    <input type="range" id="summaryLength" min="50" max="300" value="150">
    <span id="summaryLengthValue">150</span> words
</div>
</div>

<h2>Semantic Paths</h2>
<ul id="paths">{buttons}</ul>

<section id="result">
<h2>Extracted Content</h2>
<p id="selected"></p>
<div id="extraction-info" class="extraction-info"></div>
<div id="summary-section" style="display: none;">
    <h3>🤖 AI Summary</h3>
    <div id="summary"></div>
</div>
<div id="content-section">
    <h3>📄 Full Content</h3>
    <pre id="paragraph"></pre>
</div>
<p id="error"></p>
</section>

<script>
const paragraph = document.getElementById("paragraph");
const summary = document.getElementById("summary");
const summarySection = document.getElementById("summary-section");
const contentSection = document.getElementById("content-section");
const error = document.getElementById("error");
const selected = document.getElementById("selected");
const extractionInfo = document.getElementById("extraction-info");
const useAICheckbox = document.getElementById("useAI");
const useSummaryCheckbox = document.getElementById("useSummary");
const summaryLengthSlider = document.getElementById("summaryLength");
const summaryLengthValue = document.getElementById("summaryLengthValue");

// Update summary length display
summaryLengthSlider.addEventListener("input", () => {{
    summaryLengthValue.textContent = summaryLengthSlider.value;
}});

document.querySelectorAll(".path-btn").forEach((button) => {{
    button.addEventListener("click", async () => {{
        const semanticPath = button.dataset.path;
        
        // Clear previous results
        paragraph.textContent = "";
        summary.textContent = "";
        error.textContent = "";
        extractionInfo.textContent = "";
        summarySection.style.display = "none";
        
        // Show loading state
        selected.textContent = "🔄 Processing...";
        document.body.classList.add("loading");
        
        try {{
            // Build query parameters
            const params = new URLSearchParams({{
                semantic_path: semanticPath,
                use_ai: useAICheckbox.checked,
                summarize: useSummaryCheckbox.checked,
                summary_length: summaryLengthSlider.value
            }});
            
            const response = await fetch(`/extract?${{params}}`);
            if (!response.ok) {{
                const issue = await response.json().catch(() => ({{ detail: response.statusText }}));
                throw new Error(issue.detail || "Unknown error");
            }}
            
            const data = await response.json();
            console.log('Extract response', data);
            
            // Display results
            paragraph.textContent = data.paragraph || data.content;
            selected.textContent = `📍 Topic: ${{data.topic}} — 🔗 Source: ${{data.source}}`;
            
            // Show extraction info
            if (data.extraction_info) {{
                const info = data.extraction_info;
                extractionInfo.innerHTML = `
                    <strong>Extraction Method:</strong> ${{info.extraction_method || 'unknown'}} 
                    ${{info.processing_time ? `| <strong>Time:</strong> ${{info.processing_time}}` : ''}}
                    ${{info.use_cloud ? '| <strong>Cloud Browser:</strong> Yes' : ''}}
                    ${{info.error ? `| <strong>Note:</strong> ${{info.error}}` : ''}}
                `;
            }}
            
            // Show summary if available
            if (data.summary) {{
                summarySection.style.display = "block";
                if (typeof data.summary === 'object') {{
                    summary.innerHTML = `
                        <div>${{data.summary.summary}}</div>
                        <div style="margin-top: 0.5rem; font-size: 0.8em; color: #666;">
                            Model: ${{data.summary.model_used || 'unknown'}} 
                            | Time: ${{data.summary.processing_time || 'unknown'}}
                            ${{data.summary.error ? `| Error: ${{data.summary.error}}` : ''}}
                        </div>
                    `;
                }} else {{
                    summary.textContent = data.summary;
                }}
            }}
            
        }} catch (err) {{
            selected.textContent = `❌ Error processing: ${{semanticPath}}`;
            error.textContent = err.message;
        }} finally {{
            document.body.classList.remove("loading");
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
    semantic_path: str = Query(..., description="Semantic path to extract."),
    use_ai: bool = Query(False, description="Use AI-powered extraction"),
    summarize: bool = Query(False, description="Generate summary using Groq"),
    summary_length: int = Query(150, description="Summary length in words")
):
    entry = ENTRIES_BY_PATH.get(semantic_path)
    if not entry:
        raise HTTPException(status_code=404, detail="Semantic path not found.")
    
    try:
        if use_ai:
            # Use AI-powered extraction
            extraction_result = await extract_content_with_ai_agent(entry)
            content = extraction_result["content"]
            extraction_info = {
                "extraction_method": extraction_result.get("extraction_method"),
                "processing_time": extraction_result.get("processing_time"),
                "use_cloud": extraction_result.get("use_cloud"),
                "error": extraction_result.get("error")
            }
        else:
            # Use basic scraping
            content = await scrape_semantic_paragraph(entry)
            extraction_info = {"extraction_method": "basic_scraping"}
        
        response = {
            "semantic_path": semantic_path,
            "source": entry.get("original_url") or entry["semantic_path"],
            "topic": infer_topic(entry["semantic_path"]),
            "element_type": entry.get("element_type"),
            "paragraph": content,
            "extraction_info": extraction_info
        }
        
        # Add summary if requested
        if summarize:
            summary_result = await summarize_with_groq(content, infer_topic(entry["semantic_path"]), summary_length)
            response["summary"] = summary_result
        
        return response
        
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/extract/ai")
async def extract_with_ai_only(
    semantic_path: str = Query(..., description="Semantic path to extract."),
    summarize: bool = Query(True, description="Generate summary using Groq"),
    summary_length: int = Query(150, description="Summary length in words")
):
    """AI-powered extraction endpoint with automatic summarization"""
    entry = ENTRIES_BY_PATH.get(semantic_path)
    if not entry:
        raise HTTPException(status_code=404, detail="Semantic path not found.")
    
    try:
        # Use AI-powered extraction
        extraction_result = await extract_content_with_ai_agent(entry)
        content = extraction_result["content"]
        
        response = {
            "semantic_path": semantic_path,
            "source": entry.get("original_url") or entry["semantic_path"],
            "topic": infer_topic(entry["semantic_path"]),
            "element_type": entry.get("element_type"),
            "content": content,
            "extraction_info": extraction_result
        }
        
        # Add summary
        if summarize:
            summary_result = await summarize_with_groq(content, infer_topic(entry["semantic_path"]), summary_length)
            response["summary"] = summary_result
        
        return response
        
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/bulk-extract")
async def bulk_extract_with_ai(
    use_ai: bool = Query(True, description="Use AI-powered extraction"),
    summarize: bool = Query(True, description="Generate summaries"),
    limit: int = Query(5, description="Number of entries to process")
):
    """Process multiple semantic paths with AI extraction and summarization"""
    results = []
    
    for i, entry in enumerate(ENTRIES[:limit]):
        try:
            if use_ai:
                extraction_result = await extract_content_with_ai_agent(entry)
                content = extraction_result["content"]
                extraction_info = extraction_result
            else:
                content = await scrape_semantic_paragraph(entry)
                extraction_info = {"extraction_method": "basic_scraping"}
            
            result = {
                "semantic_path": entry["semantic_path"],
                "topic": infer_topic(entry["semantic_path"]),
                "content": content,
                "extraction_info": extraction_info
            }
            
            if summarize:
                summary_result = await summarize_with_groq(content, infer_topic(entry["semantic_path"]))
                result["summary"] = summary_result
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "semantic_path": entry["semantic_path"],
                "error": str(e)
            })
    
    return {
        "processed": len(results),
        "total_available": len(ENTRIES),
        "results": results
    }