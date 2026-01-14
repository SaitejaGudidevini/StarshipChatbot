"""
Multi-Agent Pipeline using LangGraph
=====================================

Complete end-to-end content processing pipeline with hierarchical web crawling.

Agents (4 Total):
1. CrawlerAgent (OPTIONAL) - Hierarchical web crawler with semantic path generation
2. LoadJSONAgent - Loads and validates semantic elements from crawler or file
3. BrowserAgent - Extracts detailed web content using browser_use + Groq AI
4. QAGeneratorAgent - Generates 10 Q&A pairs per semantic path using Pydantic validation

Architecture:
- Uses LangGraph for agent orchestration
- Hierarchical crawler discovers semantic elements (headings/links)
- Conditional routing loops through all items
- Each item goes through: Crawl → Load → Extract → Generate Q&A → Next item
- AsyncSqliteSaver checkpointing for automatic resume from failures

Flow (WITH Crawler):
START → crawler → load_json → browser_agent → qa_generator → should_continue?
    (crawl site)              ↑                                    |
                              |------------- YES -----------------|
                              NO → END

Flow (WITHOUT Crawler):
START → load_json → browser_agent → qa_generator → should_continue?
                         ↑                              |
                         |---------- YES --------------|
                         NO → END

Crawler Features:
- Hierarchical priority: headings first, then links
- Semantic path generation: single slash (/) for headings, double slash (//) for links
- Noise removal: strips headers/footers before extraction
- Relationship-aware: understands heading↔content↔links

Checkpointing:
- Automatic state persistence at each node execution
- Resume from exactly where it stopped using same thread_id
- No manual database tracking needed
- Fault-tolerant: successful nodes don't re-run on resume
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, TypedDict, Optional, Tuple
from dataclasses import dataclass

# Suppress verbose browser_use logs BEFORE importing (must be set before import)
os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'warning'  # Only show final results

from browser_use import Agent, BrowserSession, BrowserProfile, ChatGroq
from dotenv import load_dotenv
import glob as glob_module
from langchain_groq import ChatGroq as LangChainChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from hierarchical_crawler import HierarchicalWebCrawler
from pydantic import BaseModel, Field

# ============================================================================
# BROWSER PATH DETECTION (for Railway/Docker compatibility)
# ============================================================================

def get_chromium_path() -> Optional[str]:
    """
    Find Playwright's Chromium binary. Works locally and in Docker/Railway.
    Returns None if not found (browser_use will try its own detection).
    """
    search_patterns = [
        # Linux (Docker/Railway) - new Playwright path structure
        "/root/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
        "/root/.cache/ms-playwright/chromium-*/chrome-linux64/chrome",
        # Linux alternative paths
        "/home/*/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
        # macOS
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"),
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium"),
    ]

    for pattern in search_patterns:
        matches = glob_module.glob(pattern)
        if matches:
            binary = matches[0]
            if os.path.isfile(binary) and os.access(binary, os.X_OK):
                logger.info(f"Found Chromium at: {binary}")
                return binary

    return None


def create_browser_session(headless: bool = True) -> BrowserSession:
    """
    Create BrowserSession with proper browser path for both local and Docker.
    """
    chromium_path = get_chromium_path()

    if chromium_path:
        # Found Chromium - pass explicit path to bypass LocalBrowserWatchdog
        profile = BrowserProfile(
            headless=headless,
            executable_path=chromium_path,
            chromium_sandbox=False  # Required for Docker
        )
        return BrowserSession(browser_profile=profile)
    else:
        # Let browser_use auto-detect (local dev with Chrome installed)
        return BrowserSession(headless=headless)


# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

@dataclass
class ParallelConfig:
    """Configuration for parallel browser processing"""
    num_workers: int = 3          # Reduced from 5 to avoid Groq rate limits
    batch_size: int = 10          # Items to process per batch
    max_steps_per_item: int = 10  # Reduced from 30 - prevents endless scrolling
    step_timeout: int = 30        # Seconds per step (reduced from 60)
    rate_limit_delay: float = 2.0 # Increased from 0.5 to avoid API rate limits
    save_interval: int = 5        # Save results every N items

# Global config - can be overridden
PARALLEL_CONFIG = ParallelConfig()


# ============================================================================
# SSE PROGRESS TRACKER (NEW)
# ============================================================================
class ParallelProgressTracker:
    """
    Tracks and broadcasts progress for parallel processing via an asyncio.Queue.
    Designed to be a global instance for easy access from SSE endpoints.
    """
    def __init__(self, total_items: int, num_workers: int,
                 current_batch: int = 1, total_batches: int = 1,
                 overall_completed: int = 0, overall_total: int = 0):
        self.queue = asyncio.Queue()
        self.total_items = total_items  # Items in this batch
        self.num_workers = num_workers
        self.completed_items = 0
        self.failed_items = 0
        self.workers = {i: {"status": "idle", "item": None, "items_completed": 0} for i in range(num_workers)}
        self._start_time = time.time()
        self._active = True
        # Batch tracking
        self.current_batch = current_batch
        self.total_batches = total_batches
        self.overall_completed = overall_completed  # Items completed in previous batches
        self.overall_total = overall_total  # Total items across all batches

    async def _send_event(self, event_type: str, data: Dict):
        """Helper to format and send an event to the queue."""
        if not self._active:
            return
        # Calculate overall progress including previous batches
        total_completed_overall = self.overall_completed + self.completed_items
        progress_pct = (total_completed_overall / self.overall_total * 100) if self.overall_total > 0 else 0

        event_data = {
            "type": event_type,
            "timestamp": time.time(),
            # Batch-level progress
            "batch_completed": self.completed_items,
            "batch_failed": self.failed_items,
            "batch_total": self.total_items,
            # Overall progress
            "completed": total_completed_overall,
            "failed": self.failed_items,
            "total": self.overall_total,
            "progress_pct": round(progress_pct, 1),
            # Batch info
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            # Workers
            "workers": self.workers,
            **data,
        }
        await self.queue.put(json.dumps(event_data))

    async def update_worker_status(self, worker_id: int, status: str, item: Optional[Dict] = None):
        """Update a worker's status and broadcast the change."""
        self.workers[worker_id]["status"] = status
        self.workers[worker_id]["item"] = item.get("topic") if item else None
        await self._send_event(
            "worker_update",
            {"worker_id": worker_id, "status": status, "item": self.workers[worker_id]["item"]},
        )

    async def item_completed(self, worker_id: int, success: bool):
        """Mark an item as completed, update counts, and broadcast."""
        if success:
            self.completed_items += 1
        else:
            self.failed_items += 1

        self.workers[worker_id]["status"] = "idle"
        self.workers[worker_id]["items_completed"] = self.workers[worker_id].get("items_completed", 0) + 1
        await self._send_event(
            "item_completed",
            {"worker_id": worker_id, "success": success},
        )

    async def send_heartbeat(self):
        """Send a heartbeat event to keep the SSE connection alive."""
        await self._send_event("heartbeat", {})

    async def finish(self):
        """Signal that the batch is complete and close the tracker."""
        elapsed_time = time.time() - self._start_time
        await self._send_event(
            "batch_completed",
            {"status": "completed", "elapsed_time": elapsed_time},
        )
        # Add a sentinel value to signal the end of the stream
        await self.queue.put(None)
        self._active = False

    async def get_updates(self):
        """Async generator to yield updates from the queue."""
        while self._active:
            try:
                # Wait for an update or timeout for heartbeat
                update = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                if update is None:  # End of stream sentinel
                    break
                yield update
            except asyncio.TimeoutError:
                # No activity, send a heartbeat
                await self.send_heartbeat()
        
# Global instance for the tracker - to be accessed by the SSE endpoint
progress_tracker: Optional[ParallelProgressTracker] = None


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('browser_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party logs (rate limit messages, HTTP requests, etc.)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)


# Set logging level for all browser_use loggers
logging.getLogger('browser_use').setLevel(logging.WARNING)
logging.getLogger('cdp_use').setLevel(logging.WARNING)
logging.getLogger('bubus').setLevel(logging.WARNING)

# Or suppress all loggers globally
logging.basicConfig(level=logging.WARNING)


# ============================================================================
# PYDANTIC MODELS FOR Q&A GENERATION
# ============================================================================

class QAPair(BaseModel):
    """Single question-answer pair"""
    question: str = Field(description="A plausible question a user might ask about the topic")
    answer: str = Field(description="Comprehensive answer based on the extracted content")


class QAList(BaseModel):
    """List of Q&A pairs generated from content"""
    qa_pairs: List[QAPair] = Field(
        description="List of 10 question-answer pairs covering different aspects of the topic",
        min_length=8,   # Allow slight flexibility (at least 8)
        max_length=12   # Allow up to 12 if LLM generates more
    )


# ============================================================================
# STATE SCHEMA DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Shared state that flows through the agent graph"""
    # Crawler inputs (NEW)
    start_url: str                   # Website URL to crawl
    max_depth: int                   # Maximum crawl depth
    max_pages: int                   # Maximum pages to crawl
    crawler_output_path: str         # Path to crawler JSON output

    # Processing state
    semantic_paths: List[Dict]      # All items from JSON
    current_item: Dict               # Current item being processed
    current_index: int               # Index of current item
    browser_content: str             # Extracted content from browser_use
    extraction_method: str           # How content was extracted
    processing_time: float           # Time taken for extraction
    qa_pairs: List[Dict]             # Generated Q&A pairs (list of {"question": str, "answer": str})
    qa_generation_status: str        # Status: completed/failed/skipped
    qa_generation_time: float        # Time taken for Q&A generation
    status: str                      # Current status: pending/loading/extracting/complete/failed
    error_message: str               # Error details if any
    total_items: int                 # Total number of items to process
    processed_items: List[Dict]      # List of completed items
    max_items: int                   # Maximum items to process (optional limit)
    json_path: str                   # Path to JSON file (fallback if crawler not used)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_final_content(result) -> str:
    """
    Extract clean final content from browser_use AgentHistoryList
    Reused from app_automation_test.py
    """
    try:
        # Method 1: Try direct extracted_content attribute
        if hasattr(result, 'extracted_content') and result.extracted_content:
            content = result.extracted_content
            if isinstance(content, str):
                logger.debug("Extracted content via extracted_content attribute")
                return content

        # Method 2: Try final_result() method
        if hasattr(result, 'final_result') and callable(result.final_result):
            final = result.final_result()
            if final and isinstance(final, str):
                logger.debug("Extracted content via final_result() method")
                return final

        # Method 3: Try action_results attribute
        if hasattr(result, 'action_results'):
            action_results = result.action_results() if callable(result.action_results) else result.action_results
            if action_results and hasattr(action_results, '__iter__'):
                for action_result in reversed(list(action_results)):
                    if hasattr(action_result, 'is_done') and action_result.is_done:
                        if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                            logger.debug("Extracted content via action_results")
                            return action_result.extracted_content

        # Method 4: Try history attribute
        if hasattr(result, 'history'):
            history = result.history() if callable(result.history) else result.history
            if history and hasattr(history, '__iter__'):
                for item in reversed(list(history)):
                    if hasattr(item, 'result') and hasattr(item.result, 'is_done') and item.result.is_done:
                        if hasattr(item.result, 'extracted_content'):
                            logger.debug("Extracted content via history")
                            return item.result.extracted_content

        # Fallback: convert to string
        logger.warning("Could not extract clean content, using string conversion")
        return str(result)

    except Exception as e:
        logger.error(f"Error extracting final content: {e}")
        return str(result)


def infer_topic(semantic_path: str) -> str:
    """Extract topic from semantic path"""
    topic = semantic_path.split('/')[-1] if '/' in semantic_path else semantic_path
    topic = topic.replace('-', ' ').replace('_', ' ').strip()
    return topic or semantic_path


def create_qa_generation_prompt(content: str, item: dict, all_items: list = None) -> str:
    """
    Content-driven Q&A generation - reads content first, generates Q&A based on actual content.

    Args:
        content: Extracted page content
        item: Full item dict with semantic_path, depth, parent_url, element_type, etc.
        all_items: List of all items (optional, for context)
    """
    # === EXTRACT CONTEXT FROM ITEM (for reference only) ===
    semantic_path = item.get('semantic_path', '')
    parent_url = item.get('parent_url', '')

    # Build readable path for context
    path_parts = semantic_path.split('/')
    clean_parts = []
    for part in path_parts:
        part = part.strip()
        if part and not part.startswith('http') and ':' not in part:
            clean_parts.append(part.replace('-', ' ').replace('_', ' '))
    readable_path = ' → '.join(clean_parts) if clean_parts else 'Root'

    # === BUILD CONTENT-DRIVEN PROMPT ===
    return f"""You are generating Q&A pairs for a chatbot knowledge base.

════════════════════════════════════════════════════════════
STEP 1: READ AND UNDERSTAND THIS CONTENT
════════════════════════════════════════════════════════════
{content[:4000]}

════════════════════════════════════════════════════════════
STEP 2: GENERATE 10 Q&A PAIRS BASED ON THE CONTENT ABOVE
════════════════════════════════════════════════════════════

RULES:
1. BASE QUESTIONS ON WHAT THE CONTENT ACTUALLY DISCUSSES
   - Identify the main subject(s) from the content
   - Generate questions about those subjects
   - Do NOT invent information - only use facts from the content

2. MAKE QUESTIONS SELF-CONTAINED
   - Include the subject name in each question
   - A user reading just the question should understand what it's about
   ✓ GOOD: "What is torchforge and how does it enable scalable RL?"
   ✓ GOOD: "What are the key features of Weaver?"
   ✗ BAD: "What is it?" (unclear what "it" refers to)
   ✗ BAD: "How does this work?" (unclear what "this" is)

3. EXTRACT SPECIFIC DETAILS
   - Include URLs, dates, version numbers, names if mentioned
   - Be specific and factual
   - Example: "When was torchforge released?" → "January 9, 2026"

4. AVOID GENERIC FILLER QUESTIONS
   - Skip questions like "Where can I find more information?"
   - Skip questions like "Who should I contact?"
   - Focus on substantive content

CONTEXT (for reference - do NOT force these into questions):
- Source path: {readable_path}
- This helps you understand the domain/website structure

════════════════════════════════════════════════════════════

Generate 10 Q&A pairs now. Read the content carefully and ask questions a real user would ask about what's discussed."""


# ============================================================================
# BROWSER_USE TOOL INTEGRATION
# ============================================================================

async def extract_with_browser_use(url: str, topic: str) -> Dict[str, any]:
    """
    Extract content using browser_use with Groq AI

    Args:
        url: URL to extract content from
        topic: Topic/subject of the content

    Returns:
        Dict with content, extraction_method, and processing_time
    """
    groq_api_key = os.getenv('GROQ_API_KEY')

    if not groq_api_key:
        return {
            "content": f"AI extraction not available for {topic}",
            "extraction_method": "ai_unavailable",
            "processing_time": 0,
            "error": "GROQ_API_KEY not found in environment"
        }

    try:
        # Initialize Groq LLM for browser_use
        groq_llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct"
        )

        # Create extraction task based on element type
        # For headings: stay on original page and find the section
        # For links: navigate to the linked page
        task = f"""
Navigate to {url} and extract content about '{topic}'.

IMPORTANT: Stay on this exact page. Do NOT navigate away to search for '{topic}' elsewhere.
Find the section or content about '{topic}' ON THIS PAGE and extract:

1. What '{topic}' means in this context
2. Key details, features, or explanations about '{topic}'
3. Any relevant links, resources, or references mentioned
4. Important facts, dates, or contact information if present

If '{topic}' is a section heading on the page, extract the content under that heading.
Do NOT leave this domain or search for '{topic}' on other websites.

Provide a structured response with the information found.
"""

        # Create browser session (uses helper to find Chromium in Docker)
        browser_session = create_browser_session(headless=True)

        # Create browser_use agent with intelligent error handling
        agent = Agent(
            task=task,
            llm=groq_llm,
            use_vision=False,
            use_cloud=False,
            browser_session=browser_session,
            # Intelligent error handling and loop prevention
            max_failures=2,              # Fail faster on stuck elements (default: 3)
            step_timeout=60,             # Timeout faster on unresponsive elements (default: 120s)
            max_actions_per_step=3,      # More focused, deliberate actions (default: 4)
            use_thinking=True,           # Enable reasoning to recognize stuck patterns (default: True)
            final_response_after_failure=True  # Always try to extract something (default: True)
        )

        # Execute extraction with reasonable step limit
        logger.info(f"🌐 Extracting content for: {topic}")
        start_time = time.time()
        result = await agent.run(max_steps=30)  # Prevent infinite loops (default: 100)
        processing_time = time.time() - start_time

        # Cleanup browser session
        try:
            await browser_session.close()
        except Exception as e:
            logger.debug(f"Browser session cleanup warning: {e}")

        # Extract final clean content
        content_text = extract_final_content(result) if result else "No content extracted"

        logger.info(f"✅ Extracted {len(content_text)} characters in {processing_time:.1f}s")

        return {
            "content": content_text,
            "extraction_method": "ai_agent_groq",
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"❌ AI extraction failed for {topic}: {e}")
        return {
            "content": f"AI extraction failed for {topic}",
            "extraction_method": "ai_failed",
            "processing_time": 0,
            "error": str(e)
        }


# ============================================================================
# PARALLEL PROCESSING FUNCTIONS
# ============================================================================

async def extract_single_item_parallel(
    item: Dict,
    worker_id: int,
    semaphore: asyncio.Semaphore,
    config: ParallelConfig = None,
    tracker: Optional[ParallelProgressTracker] = None,
    all_items: List[Dict] = None,
) -> Dict:
    """
    Extract content for a single item with dedicated browser instance.
    Uses semaphore to limit concurrent workers and reports progress to tracker.

    Args:
        all_items: Full list of items for sibling awareness in Q&A generation
    """
    if config is None:
        config = PARALLEL_CONFIG

    async with semaphore:
        success = False
        # Use worker slot (0 to num_workers-1) for tracker, item index for logging
        worker_slot = worker_id % config.num_workers
        try:
            url = item.get("original_url", item.get("semantic_path", ""))
            semantic_path = item.get("semantic_path", url)
            topic = infer_topic(semantic_path)

            logger.info(f"🔄 [Worker {worker_slot}] Processing item {worker_id}: {topic[:50]}...")
            if tracker:
                await tracker.update_worker_status(worker_slot, "processing", {"topic": topic})


            # Add small delay to avoid API rate limits
            await asyncio.sleep(config.rate_limit_delay * (worker_id % config.num_workers))

            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                return {
                    "semantic_path": semantic_path,
                    "original_url": url,
                    "topic": topic,
                    "browser_content": "API key not available",
                    "extraction_method": "api_unavailable",
                    "processing_time": 0,
                    "status": "failed",
                    "error": "GROQ_API_KEY not found",
                    "qa_pairs": [],
                    "qa_generation_status": "skipped"
                }

            browser_session = None

            # Create isolated browser session for this worker (uses helper to find Chromium in Docker)
            browser_session = create_browser_session(headless=True)

            # Initialize Groq LLM
            groq_llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct"
            )

            # Simplified extraction task for speed
            # IMPORTANT: Tell agent to stay on page and not navigate away
            task = f"""Navigate to {url} and extract content about '{topic}'.
IMPORTANT: Stay on this exact page. Do NOT navigate away or search elsewhere.
Find the section about '{topic}' ON THIS PAGE and extract key information.
If '{topic}' is a heading, extract content under that heading.
Be concise and extract only essential information."""

            # Create agent with reduced steps for speed
            agent = Agent(
                task=task,
                llm=groq_llm,
                use_cloud=False,
                use_vision=False,
                browser_session=browser_session,
                max_failures=2,
                step_timeout=config.step_timeout,
                max_actions_per_step=2,
            )

            # Extract content with limited steps
            start_time = time.time()
            result = await agent.run(max_steps=config.max_steps_per_item)
            extraction_time = time.time() - start_time

            content_text = extract_final_content(result) if result else "No content extracted"

            logger.info(f"✅ [Worker {worker_slot}] Extracted {len(content_text)} chars for item {worker_id} in {extraction_time:.1f}s")

            # Generate Q&A pairs inline
            qa_pairs = []
            qa_status = "skipped"
            qa_time = 0

            if content_text and content_text not in ["No content extracted", "AI extraction failed"]:
                try:
                    llm = LangChainChatGroq(
                        model="meta-llama/llama-4-maverick-17b-128e-instruct",
                        api_key=groq_api_key,
                        temperature=0.3,
                        max_tokens=2000
                    )
                    structured_llm = llm.with_structured_output(QAList)
                    # Pass full item dict and all_items for context-aware prompt generation
                    prompt = create_qa_generation_prompt(content_text, item, all_items)

                    qa_start = time.time()
                    qa_result = structured_llm.invoke(prompt)
                    qa_time = time.time() - qa_start

                    qa_pairs = [{"question": qa.question, "answer": qa.answer} for qa in qa_result.qa_pairs]
                    qa_status = "completed"
                    logger.info(f"✅ [Worker {worker_slot}] Generated {len(qa_pairs)} Q&A pairs for item {worker_id}")

                except Exception as qa_error:
                    logger.warning(f"⚠️ [Worker {worker_slot}] Q&A generation failed for item {worker_id}: {qa_error}")
                    qa_status = "failed"
            
            success = True
            return {
                "semantic_path": semantic_path,
                "original_url": url,
                "topic": topic,
                "browser_content": content_text,
                "extraction_method": "ai_agent_groq_parallel",
                "processing_time": extraction_time,
                "status": "completed",
                "qa_pairs": qa_pairs,
                "qa_generation_status": qa_status,
                "qa_generation_time": qa_time,
                "qa_count": len(qa_pairs),
                "worker_id": worker_id
            }

        except Exception as e:
            logger.error(f"❌ [Worker {worker_slot}] Failed for item {worker_id} ({topic}): {e}")
            success = False
            return {
                "semantic_path": semantic_path,
                "original_url": url,
                "topic": topic,
                "browser_content": "",
                "extraction_method": "ai_failed",
                "processing_time": 0,
                "status": "failed",
                "error": str(e),
                "qa_pairs": [],
                "qa_generation_status": "skipped",
                "worker_id": worker_id
            }
        finally:
            if tracker:
                await tracker.item_completed(worker_slot, success=success)
            if browser_session:
                try:
                    await browser_session.close()
                except Exception:
                    pass


async def process_batch_parallel(
    items: List[Dict],
    config: ParallelConfig = None,
    current_batch: int = 1,
    total_batches: int = 1,
    overall_completed: int = 0,
    overall_total: int = 0
) -> List[Dict]:
    """
    Process a batch of items in parallel using multiple browser workers.
    Creates and manages a ParallelProgressTracker for SSE streaming.
    """
    global progress_tracker
    if config is None:
        config = PARALLEL_CONFIG

    logger.info(f"🚀 Starting parallel processing: Batch {current_batch}/{total_batches} ({len(items)} items) with {config.num_workers} workers")

    # Create and assign the global tracker with batch info
    progress_tracker = ParallelProgressTracker(
        total_items=len(items),
        num_workers=config.num_workers,
        current_batch=current_batch,
        total_batches=total_batches,
        overall_completed=overall_completed,
        overall_total=overall_total
    )
    
    try:
        # Create semaphore to limit concurrent workers
        semaphore = asyncio.Semaphore(config.num_workers)

        # Create tasks for all items, passing the tracker and all_items for sibling awareness
        tasks = [
            extract_single_item_parallel(item, i, semaphore, config, tracker=progress_tracker, all_items=items)
            for i, item in enumerate(items)
        ]

        # Process with progress tracking
        results = []
        completed_count = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed_count += 1

            if completed_count % config.save_interval == 0:
                logger.info(f"📊 Progress: {completed_count}/{len(items)} items completed")

        # Sort results to maintain original order
        item_order = {item.get("semantic_path", ""): i for i, item in enumerate(items)}
        results.sort(key=lambda x: item_order.get(x.get("semantic_path", ""), 999999))

        succeeded = sum(1 for r in results if r.get("status") == "completed")
        logger.info(f"✅ Parallel processing complete: {succeeded}/{len(items)} succeeded")

        return results
    finally:
        if progress_tracker:
            await progress_tracker.finish()
            progress_tracker = None # Clear the global instance


async def parallel_batch_node(state: AgentState) -> AgentState:
    """
    LangGraph Node: Process items in parallel batches with checkpointing.
    Processes BATCH_SIZE items at a time, then returns to save checkpoint.
    Graph loops back for next batch until all items are done.
    """
    BATCH_SIZE = 50  # Process 50 items per batch, checkpoint after each

    semantic_paths = state.get("semantic_paths", [])
    current_index = state.get("current_index", 0)
    processed_items = state.get("processed_items", [])
    total_items = len(semantic_paths)

    # Get next batch of items to process
    batch_end = min(current_index + BATCH_SIZE, total_items)
    batch_items = semantic_paths[current_index:batch_end]

    if not batch_items:
        logger.info("✅ All items already processed")
        return {
            **state,
            "status": "complete",
            "current_index": total_items
        }

    # Calculate batch info for progress tracking
    total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    current_batch = (current_index // BATCH_SIZE) + 1

    logger.info(f"🚀 Processing batch {current_batch}/{total_batches}: items {current_index+1}-{batch_end} of {total_items} ({len(batch_items)} items)")

    # Process this batch in parallel with batch info for SSE
    results = await process_batch_parallel(
        batch_items,
        PARALLEL_CONFIG,
        current_batch=current_batch,
        total_batches=total_batches,
        overall_completed=current_index,  # Items already completed before this batch
        overall_total=total_items
    )

    # Merge results with existing processed items
    processed_items.extend(results)
    new_index = current_index + len(results)

    # Determine if we're done or need another batch
    is_complete = new_index >= total_items
    status = "complete" if is_complete else "processing"

    if is_complete:
        logger.info(f"✅ All {total_items} items processed! Total Q&A: {sum(r.get('qa_count', 0) for r in processed_items)}")
    else:
        logger.info(f"📊 Batch complete. Progress: {new_index}/{total_items} ({100*new_index//total_items}%) - Checkpoint saved")

    return {
        **state,
        "processed_items": processed_items,
        "current_index": new_index,
        "total_items": total_items,
        "status": status,
        "total_processed": len(processed_items)
    }


# ============================================================================
# LANGGRAPH NODES (SEQUENTIAL - Original)
# ============================================================================

async def crawler_node(state: AgentState) -> AgentState:
    """
    Node: Crawl website using hierarchical crawler
    This is the FIRST node - discovers all semantic elements to process
    """
    start_url = state.get("start_url", "")
    max_depth = state.get("max_depth", 2)
    max_pages = state.get("max_pages", 100)

    if not start_url:
        logger.error("❌ No start_url provided for crawler")
        return {
            **state,
            "status": "failed",
            "error_message": "No start_url provided for crawler"
        }

    logger.info(f"🕷️ Starting hierarchical crawler for: {start_url}")
    logger.info(f"📊 Max depth: {max_depth}, Max pages: {max_pages}")

    try:
        # Initialize crawler
        crawler = HierarchicalWebCrawler(
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            headless=True,
            timeout=30000
        )

        # Run hierarchical crawl
        crawl_nodes = await crawler.crawl_hierarchical()

        # Save results
        output_path = crawler.save_hierarchical_results()

        logger.info(f"✅ Crawler completed!")
        logger.info(f"📄 Total nodes discovered: {len(crawl_nodes)}")
        logger.info(f"💾 Saved to: {output_path}")

        return {
            **state,
            "crawler_output_path": output_path,
            "json_path": output_path,  # Set as json_path for load_json_node
            "status": "crawled"
        }

    except Exception as e:
        logger.error(f"❌ Crawler failed: {e}")
        return {
            **state,
            "status": "failed",
            "error_message": f"Crawler failed: {str(e)}"
        }


def load_json_node(state: AgentState) -> AgentState:
    """
    Node: Load semantic paths from JSON file
    Handles both crawler output and pre-existing JSON files
    """
    try:
        logger.info("📂 Loading JSON data...")

        # Get json_path (set by crawler_node or provided directly)
        # Use DATA_DIR environment variable or current directory
        data_dir = os.getenv('DATA_DIR', '.')
        default_json = os.path.join(data_dir, 'CSU.json')
        json_path = state.get("json_path", default_json)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract semantic elements (handle multiple formats)
        if isinstance(data, dict) and 'semantic_elements' in data:
            # Crawler output format: {"semantic_elements": {...}}
            semantic_elements = data['semantic_elements']

            # Check if semantic_elements is a dict (keys are semantic paths)
            if isinstance(semantic_elements, dict):
                # Convert dict to list format, preserving all fields
                semantic_elements = [
                    {
                        "semantic_path": k,
                        **v  # Includes: original_url, source_type, element_type
                    }
                    for k, v in semantic_elements.items()
                ]
                logger.debug(f"Converted dict semantic_elements to list format")

                # Log crawler metadata if available
                if 'crawl_metadata' in data:
                    metadata = data['crawl_metadata']
                    logger.info(f"🕷️ Crawler metadata:")
                    logger.info(f"   Domain: {metadata.get('domain', 'N/A')}")
                    logger.info(f"   Total elements: {metadata.get('total_elements', 'N/A')}")
                    logger.info(f"   Crawl type: {metadata.get('crawl_type', 'N/A')}")
            # Otherwise it's already a list

        elif isinstance(data, list):
            # Pre-existing list format
            semantic_elements = data
        else:
            # Assume it's a dict of semantic paths (no 'semantic_elements' key)
            semantic_elements = [{"semantic_path": k, **v} for k, v in data.items()]

        # Check if max_items limit is set
        max_items = state.get("max_items")
        original_count = len(semantic_elements)
        if max_items and max_items < original_count:
            semantic_elements = semantic_elements[:max_items]
            logger.info(f"📋 Limited to first {max_items} items (out of {original_count} total)")

        total_items = len(semantic_elements)
        logger.info(f"✅ Loaded {total_items} semantic elements")
        logger.info(f"🚀 Starting batch processing with conditional routing...")

        # Preserve checkpoint values if resuming (don't reset!)
        current_index = state.get("current_index", 0)
        processed_items = state.get("processed_items", [])

        # Log if resuming from checkpoint
        if current_index > 0:
            logger.info(f"✅ Preserving checkpoint: continuing from item {current_index+1}")

        return {
            **state,
            "semantic_paths": semantic_elements,
            "current_item": state.get("current_item", {}),
            "current_index": current_index,        # Preserve from checkpoint
            "total_items": total_items,
            "status": "loaded",
            "processed_items": processed_items     # Preserve from checkpoint
        }

    except Exception as e:
        logger.error(f"❌ Failed to load JSON: {e}")
        return {
            **state,
            "status": "failed",
            "error_message": f"JSON loading failed: {str(e)}"
        }


async def browser_agent_node(state: AgentState) -> AgentState:
    """
    Node: Extract content using browser_use for ONE item
    Uses conditional routing to loop through all items
    """
    try:
        current_idx = state["current_index"]
        semantic_paths = state["semantic_paths"]

        # Check if we have items to process
        if not semantic_paths or current_idx >= len(semantic_paths):
            logger.warning("⚠️ No more items to process")
            return {
                **state,
                "status": "complete"
            }

        # Get current item
        current_item = semantic_paths[current_idx]

        # Extract URL and topic
        url = current_item.get("original_url", current_item.get("semantic_path", ""))
        semantic_path = current_item.get("semantic_path", url)
        topic = infer_topic(semantic_path)

        logger.info(f"🔄 Processing [{current_idx + 1}/{state['total_items']}]: {topic}")

        # Extract content with browser_use
        result = await extract_with_browser_use(url, topic)

        # Create processed item record
        processed_item = {
            "semantic_path": semantic_path,
            "original_url": url,
            "topic": topic,
            "browser_content": result["content"],
            "extraction_method": result["extraction_method"],
            "processing_time": result["processing_time"],
            "status": "completed" if not result.get("error") else "failed",
            "error": result.get("error", "")
        }

        # Update processed items list
        processed_items = state.get("processed_items", [])
        processed_items.append(processed_item)

        # Increment index for next iteration
        next_index = current_idx + 1

        return {
            **state,
            "current_item": current_item,
            "current_index": next_index,  # Move to next item
            "browser_content": result["content"],
            "extraction_method": result["extraction_method"],
            "processing_time": result["processing_time"],
            "status": "processing" if next_index < len(semantic_paths) else "complete",
            "processed_items": processed_items,
            "error_message": result.get("error", "")
        }

    except Exception as e:
        logger.error(f"❌ Browser agent failed: {e}")

        # Still increment index to avoid infinite loop
        next_index = state["current_index"] + 1

        return {
            **state,
            "current_index": next_index,
            "browser_content": "",
            "status": "processing" if next_index < len(state["semantic_paths"]) else "failed",
            "error_message": f"Browser extraction failed: {str(e)}"
        }


async def qa_generator_node(state: AgentState) -> AgentState:
    """
    Node: Generate Q&A pairs from browser_use content
    Uses LangChain's structured output with Pydantic models for validation
    """
    browser_content = state.get("browser_content", "")
    current_item = state.get("current_item", {})
    topic = current_item.get("topic", infer_topic(current_item.get("semantic_path", "")))

    # Skip if no content
    if not browser_content or browser_content in ["No content extracted", "AI extraction failed"]:
        logger.warning(f"⚠️ Skipping Q&A generation for {topic}: No valid content")

        # Update the last processed item
        processed_items = state.get("processed_items", [])
        if processed_items:
            processed_items[-1]["qa_pairs"] = []
            processed_items[-1]["qa_generation_status"] = "skipped"

        return {
            **state,
            "qa_pairs": [],
            "qa_generation_status": "skipped",
            "qa_generation_time": 0.0,
            "processed_items": processed_items
        }

    logger.info(f"❓ Generating Q&A pairs for: {topic}")

    try:
        # Initialize LangChain ChatGroq with structured output
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        # Create LangChain ChatGroq instance
        llm = LangChainChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key,
            temperature=0.3,  # Slightly creative but mostly factual
            max_tokens=3000   # Allow longer responses for 10 Q&A pairs
        )

        # Add structured output with Pydantic validation
        structured_llm = llm.with_structured_output(QAList)

        # Create Q&A generation prompt with full context
        # Pass current_item and all semantic_paths for sibling awareness
        all_items = state.get("semantic_paths", [])
        prompt = create_qa_generation_prompt(browser_content, current_item, all_items)

        # Generate Q&A pairs with validation
        start_time = time.time()
        result = structured_llm.invoke(prompt)  # Returns QAList object
        qa_generation_time = time.time() - start_time

        # Convert Pydantic models to dictionaries
        qa_pairs = [
            {"question": qa.question, "answer": qa.answer}
            for qa in result.qa_pairs
        ]

        logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs for {topic} ({qa_generation_time:.1f}s)")

        # Update the last processed item with Q&A pairs
        processed_items = state.get("processed_items", [])
        if processed_items:
            processed_items[-1]["qa_pairs"] = qa_pairs
            processed_items[-1]["qa_generation_status"] = "completed"
            processed_items[-1]["qa_generation_time"] = round(qa_generation_time, 2)
            processed_items[-1]["qa_model"] = "meta-llama/llama-4-maverick-17b-128e-instruct"
            processed_items[-1]["qa_count"] = len(qa_pairs)

        return {
            **state,
            "qa_pairs": qa_pairs,
            "qa_generation_status": "completed",
            "qa_generation_time": qa_generation_time,
            "processed_items": processed_items
        }

    except Exception as e:
        logger.error(f"❌ Q&A generation failed for {topic}: {e}")

        # Return empty Q&A list on failure
        processed_items = state.get("processed_items", [])
        if processed_items:
            processed_items[-1]["qa_pairs"] = []
            processed_items[-1]["qa_generation_status"] = "failed"
            processed_items[-1]["qa_generation_error"] = str(e)

        return {
            **state,
            "qa_pairs": [],
            "qa_generation_status": "failed",
            "qa_generation_time": 0.0,
            "processed_items": processed_items,
            "error_message": f"Q&A generation failed: {str(e)}"
        }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Router function: Decide whether to continue processing or end

    Returns:
        "continue" - Loop back to browser_agent to process next item
        "end" - All items processed, go to END
    """
    current_idx = state["current_index"]
    total_items = state["total_items"]

    if current_idx < total_items:
        logger.info(f"📊 Progress: {current_idx}/{total_items} - Continuing to next item")
        return "continue"
    else:
        logger.info(f"✅ All {total_items} items processed!")
        return "end"


def should_continue_parallel(state: AgentState) -> str:
    """
    Router function for parallel batch processing.
    Loops back for next batch or ends when complete.

    Returns:
        "continue" - More batches to process, loop back
        "end" - All batches complete, go to END
    """
    status = state.get("status", "")
    if status == "complete":
        return "end"
    else:
        return "continue"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_browser_agent_graph(
    checkpointer: Optional[AsyncSqliteSaver] = None,
    enable_crawler: bool = False,
    enable_parallel: bool = False
):
    """
    Build LangGraph workflow for Multi-Agent Pipeline

    SEQUENTIAL MODE (enable_parallel=False):
    START → [crawler] → load_json → browser_agent → qa_generator → should_continue?
                                          ↑                              |
                                          |------------ YES -------------|
                                          NO → END

    PARALLEL MODE (enable_parallel=True):
    START → [crawler] → load_json → parallel_batch_node → should_continue_parallel?
                                          ↑                              |
                                          |------------ YES -------------|
                                          NO → END
    (Processes 50 items per batch with 3 workers, checkpoints after each batch)

    Args:
        checkpointer: Optional AsyncSqliteSaver for state persistence
        enable_crawler: If True, includes crawler_node as first step
        enable_parallel: If True, uses parallel batch processing (5x-10x faster)
    """
    # Create graph builder
    builder = StateGraph(AgentState)

    # Add nodes based on mode
    if enable_crawler:
        builder.add_node("crawler", crawler_node)
    builder.add_node("load_json", load_json_node)

    if enable_parallel:
        # PARALLEL MODE: Single node processes all items concurrently
        builder.add_node("parallel_batch", parallel_batch_node)
        logger.info("🚀 PARALLEL MODE enabled - using concurrent browser workers")
    else:
        # SEQUENTIAL MODE: Loop through items one by one
        builder.add_node("browser_agent", browser_agent_node)
        builder.add_node("qa_generator", qa_generator_node)

    # Define edges
    if enable_crawler:
        builder.add_edge(START, "crawler")
        builder.add_edge("crawler", "load_json")
    else:
        builder.add_edge(START, "load_json")

    if enable_parallel:
        # PARALLEL: load_json → parallel_batch → loop/end (with checkpointing per batch)
        builder.add_edge("load_json", "parallel_batch")
        builder.add_conditional_edges(
            "parallel_batch",
            should_continue_parallel,
            {
                "continue": "parallel_batch",  # Loop back for next batch
                "end": END
            }
        )
    else:
        # SEQUENTIAL: load_json → browser_agent → qa_generator → loop/end
        builder.add_edge("load_json", "browser_agent")
        builder.add_edge("browser_agent", "qa_generator")
        builder.add_conditional_edges(
            "qa_generator",
            should_continue,
            {
                "continue": "browser_agent",
                "end": END
            }
        )

    # Compile graph
    if checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        logger.info(f"✅ Graph compiled with checkpointing enabled")
    else:
        graph = builder.compile()
        logger.info(f"✅ Graph compiled without checkpointing")

    if enable_crawler:
        logger.info("🕷️ Crawler node enabled")
    if enable_parallel:
        logger.info(f"⚡ Parallel config: {PARALLEL_CONFIG.num_workers} workers, {PARALLEL_CONFIG.max_steps_per_item} max steps/item")

    return graph


# ============================================================================
# EXECUTION
# ============================================================================

async def run_browser_agent(
    json_path: str = None,
    output_file: str = "browser_agent_output.json",
    max_items: int = None,
    thread_id: str = "default-workflow",
    checkpoint_db: str = "browser_agent_checkpoints.db",
    enable_checkpointing: bool = True,
    # Crawler parameters
    enable_crawler: bool = False,
    start_url: str = None,
    max_depth: int = 2,
    max_pages: int = 100,
    # Parallel processing
    enable_parallel: bool = False
):
    """
    Execute Multi-Agent Pipeline (Crawler + Browser + Q&A Generator)

    Args:
        json_path: Path to JSON file with semantic paths (used if crawler disabled)
        output_file: Where to save extracted content
        max_items: Maximum number of items to process (None = process all)
        thread_id: Thread ID for checkpointing (same ID = resume from last checkpoint)
        checkpoint_db: Path to SQLite checkpoint database
        enable_checkpointing: Enable/disable checkpointing and resume functionality
        enable_crawler: Enable/disable hierarchical web crawler as first step
        start_url: Website URL to crawl (required if enable_crawler=True)
        max_depth: Maximum crawl depth (default: 2)
        max_pages: Maximum pages to crawl (default: 100)
        enable_parallel: Enable parallel processing (5x-10x faster, uses 5 concurrent browsers)

    Returns:
        Final state with extracted content and Q&A pairs
    """
    logger.info("🚀 Starting Multi-Agent Pipeline")

    if enable_parallel:
        logger.info(f"⚡ PARALLEL MODE: {PARALLEL_CONFIG.num_workers} concurrent browsers")
        logger.info(f"   Max steps per item: {PARALLEL_CONFIG.max_steps_per_item}")
        logger.info(f"   Estimated speedup: 5x-10x faster than sequential")
    else:
        logger.info("🔄 Sequential mode (use enable_parallel=True for faster processing)")

    if enable_crawler:
        logger.info(f"🕷️ Crawler enabled: {start_url}")
        if not start_url:
            logger.error("❌ start_url required when enable_crawler=True")
            raise ValueError("start_url is required when enable_crawler=True")
    else:
        logger.info(f"📂 Using pre-existing JSON: {json_path}")

    if max_items:
        logger.info(f"📋 Limiting processing to {max_items} items")

    # Initial state
    initial_state = {
        # Crawler parameters (NEW)
        "start_url": start_url or "",
        "max_depth": max_depth,
        "max_pages": max_pages,
        "crawler_output_path": "",

        # Processing state
        "semantic_paths": [],
        "current_item": {},
        "current_index": 0,
        "browser_content": "",
        "extraction_method": "",
        "processing_time": 0.0,
        "qa_pairs": [],                  # Generated Q&A pairs
        "qa_generation_status": "",      # Q&A generation status
        "qa_generation_time": 0.0,       # Q&A generation timing
        "status": "pending",
        "error_message": "",
        "total_items": 0,
        "processed_items": [],
        "json_path": json_path or os.path.join(os.getenv('DATA_DIR', '.'), 'mhs_indianafiltered_data.json'),
        "max_items": max_items
    }

    # Configuration with thread_id for checkpointing and recursion limit
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 1000  # Allow up to 1000 loop iterations (500 items)
    } if enable_checkpointing else {
        "recursion_limit": 1000
    }

    # Branch based on checkpointing enabled/disabled
    if enable_checkpointing:
        logger.info(f"🔄 Checkpointing enabled with thread_id: {thread_id}")
        logger.info(f"💡 If interrupted, run again with same thread_id to resume")

        # Use async context manager for AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
            logger.info(f"✅ Async checkpointer initialized with database: {checkpoint_db}")

            # Build graph with checkpointer (inside context manager)
            graph = build_browser_agent_graph(
                checkpointer=checkpointer,
                enable_crawler=enable_crawler,
                enable_parallel=enable_parallel
            )

            # Check if resuming from previous state
            is_resuming = False
            try:
                existing_state = await graph.aget_state(config)
                if existing_state and existing_state.values:
                    current_idx = existing_state.values.get("current_index", 0)
                    total = existing_state.values.get("total_items", 0)
                    if current_idx > 0 and current_idx < total:
                        logger.info(f"🔄 RESUMING from checkpoint: Item {current_idx}/{total}")
                        logger.info(f"✅ Already processed: {current_idx} items")
                        is_resuming = True
            except Exception as e:
                logger.debug(f"No previous state found (this is normal for first run): {e}")

            # Run graph
            # IMPORTANT: Pass None when resuming to load state from checkpoint
            # Pass initial_state only when starting fresh
            try:
                if is_resuming:
                    logger.info("📥 Loading state from checkpoint (passing None to ainvoke)")
                    result = await graph.ainvoke(None, config=config)
                else:
                    logger.info("🆕 Starting fresh workflow with initial state")
                    result = await graph.ainvoke(initial_state, config=config)

                # Log and save results
                _log_and_save_results(result, output_file)

                return result

            except Exception as e:
                logger.error(f"❌ Browser Agent execution failed: {e}")
                logger.info(f"💡 Progress saved to checkpoint. Run again with thread_id='{thread_id}' to resume")
                raise

    else:
        # No checkpointing - build graph without checkpointer
        logger.info("⚠️ Checkpointing disabled - progress will not be saved")
        graph = build_browser_agent_graph(
            checkpointer=None,
            enable_crawler=enable_crawler,
            enable_parallel=enable_parallel
        )

        try:
            result = await graph.ainvoke(initial_state, config=config)

            # Log and save results
            _log_and_save_results(result, output_file)

            return result

        except Exception as e:
            logger.error(f"❌ Browser Agent execution failed: {e}")
            raise


def _log_and_save_results(result: dict, output_file: str):
    """Helper function to log results and save output"""
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Multi-Agent Pipeline Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Status: {result['status']}")
    logger.info(f"Items Processed: {len(result.get('processed_items', []))}")

    # Show Q&A generation statistics
    processed_items = result.get('processed_items', [])
    qa_generated_count = sum(1 for item in processed_items if item.get('qa_generation_status') == 'completed')
    skipped_count = sum(1 for item in processed_items if item.get('qa_generation_status') == 'skipped')
    total_qa_pairs = sum(item.get('qa_count', 0) for item in processed_items)

    logger.info(f"Q&A Generated: {qa_generated_count}/{len(processed_items)}")
    logger.info(f"Total Q&A Pairs: {total_qa_pairs}")
    logger.info(f"Skipped: {skipped_count}/{len(processed_items)}")
    logger.info(f"Processing Time: {result.get('processing_time', 0):.1f}s")

    if result.get('error_message'):
        logger.warning(f"⚠️ Errors: {result['error_message']}")

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.get('processed_items', []), f, indent=2, ensure_ascii=False)

    logger.info(f"📁 Results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point for testing"""
    print("\n" + "="*60)
    print("MULTI-AGENT PIPELINE")
    print("Crawler → Browser → Q&A Generator")
    print("LangGraph Conditional Routing + Checkpointing")
    print("="*60)

    # Ask about crawler
    print("\n🕷️ Crawler Options:")
    print("   1. Use hierarchical web crawler (crawl website first)")
    print("   2. Use existing JSON file (skip crawling)")

    crawler_choice = input("\nChoose option (1/2): ").strip()
    enable_crawler = crawler_choice == "1"

    start_url = None
    max_depth = 2
    max_pages = 100
    json_path = None

    if enable_crawler:
        start_url = input("   Enter website URL to crawl: ").strip()
        max_depth_input = input("   Max crawl depth (press Enter for 2): ").strip()
        max_depth = int(max_depth_input) if max_depth_input else 2
        max_pages_input = input("   Max pages to crawl (press Enter for 100): ").strip()
        max_pages = int(max_pages_input) if max_pages_input else 100
        print(f"   🕷️ Will crawl: {start_url} (depth={max_depth}, max_pages={max_pages})\n")
    else:
        json_path = input("   Enter path to JSON file (press Enter for default): ").strip()
        if not json_path:
            data_dir = os.getenv('DATA_DIR', '.')
            json_path = os.path.join(data_dir, 'mhs_indianafiltered_data.json')
        print(f"   📂 Will use: {json_path}\n")

    # Ask about checkpointing
    print("🔄 Checkpointing & Resume:")
    print("   Enable checkpointing to automatically save progress and resume from failures")
    enable_checkpointing = input("   Enable checkpointing? (Y/n): ").strip().lower() != 'n'

    thread_id = "default-workflow"
    if enable_checkpointing:
        thread_id = input("   Thread ID (press Enter for 'default-workflow'): ").strip() or "default-workflow"
        print(f"   💡 Using thread_id: {thread_id}")
        print(f"   💡 If interrupted, run again with same thread_id to resume\n")

    # Ask about parallel processing
    print("\n⚡ Processing Mode:")
    print("   1. PARALLEL (5 concurrent browsers - 5x-10x FASTER)")
    print("   2. Sequential (one at a time - slower but safer)")

    mode_choice = input("\nChoose mode (1/2, default=1): ").strip()
    enable_parallel = mode_choice != "2"

    if enable_parallel:
        print("   ⚡ Using PARALLEL mode with 5 concurrent browsers")
    else:
        print("   🔄 Using sequential mode")

    # Ask user how many items to process
    print("\n🔧 Item Limit:")
    print("   1. Process ALL items (full automation)")
    print("   2. Process specific number of items (e.g., 5 or 10)")

    choice = input("\nChoose option (1/2): ").strip()

    max_items = None
    if choice == "2":
        max_items = int(input("How many items to process? "))

    result = await run_browser_agent(
        json_path=json_path,
        output_file="browser_agent_test_output.json",
        max_items=max_items,
        thread_id=thread_id,
        enable_checkpointing=enable_checkpointing,
        enable_crawler=enable_crawler,
        start_url=start_url,
        max_depth=max_depth,
        max_pages=max_pages,
        enable_parallel=enable_parallel
    )

    print("\n" + "="*60)
    print("MULTI-AGENT PIPELINE COMPLETE")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Processed: {len(result.get('processed_items', []))} items")

    # Show Q&A generation stats
    processed_items = result.get('processed_items', [])
    qa_generated = sum(1 for item in processed_items if item.get('qa_generation_status') == 'completed')
    total_qa_pairs = sum(item.get('qa_count', 0) for item in processed_items)
    print(f"Q&A Generated: {qa_generated}/{len(processed_items)}")
    print(f"Total Q&A Pairs: {total_qa_pairs}")
    print(f"Output saved to: browser_agent_test_output.json")

    if enable_crawler and result.get('crawler_output_path'):
        print(f"\n🕷️ Crawler output: {result['crawler_output_path']}")

    if enable_checkpointing:
        print(f"\n💾 Checkpoint saved with thread_id: {thread_id}")
        print(f"   To start a new workflow, use a different thread_id")


if __name__ == "__main__":
    asyncio.run(main())
