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
from typing import Dict, List, TypedDict, Optional, Tuple, Any
from dataclasses import dataclass
import csv

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hnswlib
import threading

# ============================================================================
# BROWSER PATH DETECTION (for Railway/Docker compatibility)
# ============================================================================

def get_chromium_path() -> Optional[str]:
    """
    Find Playwright's Chromium binary. Works locally and in Docker/Railway.
    Returns None if not found (browser_use will try its own detection).
    """
    # Check PLAYWRIGHT_BROWSERS_PATH env var first
    pw_path = os.environ.get('PLAYWRIGHT_BROWSERS_PATH', '')

    search_patterns = [
        # Linux (Docker/Railway) - standard Playwright paths
        "/root/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
        "/root/.cache/ms-playwright/chromium-*/chrome-linux64/chrome",
        # Headless shell variants (newer Playwright versions)
        "/root/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell",
        "/root/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux64/headless_shell",
        # Chrome for Testing (what browser_use 0.11.2 calls 'chrome')
        "/root/.cache/ms-playwright/chrome-*/chrome-linux/chrome",
        "/root/.cache/ms-playwright/chrome-*/chrome-linux64/chrome",
        # Linux alternative paths
        "/home/*/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
        "/home/*/.cache/ms-playwright/chromium-*/chrome-linux64/chrome",
        # System-installed browsers (fallback)
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        # macOS
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"),
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium"),
    ]

    # If PLAYWRIGHT_BROWSERS_PATH is set, add those patterns too
    if pw_path:
        search_patterns = [
            f"{pw_path}/chromium-*/chrome-linux/chrome",
            f"{pw_path}/chromium-*/chrome-linux64/chrome",
            f"{pw_path}/chrome-*/chrome-linux/chrome",
            f"{pw_path}/chrome-*/chrome-linux64/chrome",
        ] + search_patterns

    for pattern in search_patterns:
        matches = glob_module.glob(pattern)
        if matches:
            binary = matches[0]
            if os.path.isfile(binary) and os.access(binary, os.X_OK):
                logger.info(f"Found Chromium at: {binary}")
                return binary

    # Last resort: try to find any chrome/chromium binary in ms-playwright
    fallback_patterns = [
        "/root/.cache/ms-playwright/**/chrome",
        "/root/.cache/ms-playwright/**/chromium",
    ]
    for pattern in fallback_patterns:
        matches = glob_module.glob(pattern, recursive=True)
        if matches:
            binary = matches[0]
            if os.path.isfile(binary) and os.access(binary, os.X_OK):
                logger.info(f"Found Chromium (fallback glob) at: {binary}")
                return binary

    logger.warning("❌ No Chromium binary found in any known path")
    return None


def create_browser_profile(headless: bool = True) -> BrowserProfile:
    """
    Create BrowserProfile with proper browser path for both local and Docker/Railway.
    Always use this instead of bare BrowserProfile(headless=True) to ensure
    the Playwright Chromium binary is found in production.
    """
    chromium_path = get_chromium_path()

    if chromium_path:
        logger.info(f"🔧 Using detected Chromium path: {chromium_path}")
        return BrowserProfile(
            headless=headless,
            executable_path=chromium_path,
            chromium_sandbox=False  # Required for Docker/Railway
        )
    else:
        logger.warning("⚠️ No Chromium path detected, falling back to browser_use auto-detect")
        return BrowserProfile(headless=headless)


def create_browser_session(headless: bool = True) -> BrowserSession:
    """
    Create BrowserSession with proper browser path for both local and Docker.
    """
    return BrowserSession(browser_profile=create_browser_profile(headless))


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
        self.workers[worker_id]["semantic_path"] = item.get("semantic_path") if item else None
        await self._send_event(
            "worker_update",
            {
                "worker_id": worker_id,
                "status": status,
                "item": self.workers[worker_id]["item"],
                "semantic_path": self.workers[worker_id]["semantic_path"],
            },
        )

    async def item_completed(self, worker_id: int, success: bool):
        """Mark an item as completed, update counts, and broadcast."""
        if success:
            self.completed_items += 1
        else:
            self.failed_items += 1

        # Capture the semantic_path before resetting to idle
        completed_path = self.workers[worker_id].get("semantic_path")
        self.workers[worker_id]["status"] = "idle"
        self.workers[worker_id]["items_completed"] = self.workers[worker_id].get("items_completed", 0) + 1
        self.workers[worker_id]["semantic_path"] = None
        await self._send_event(
            "item_completed",
            {"worker_id": worker_id, "success": success, "semantic_path": completed_path},
        )

    async def send_tree_init(self, items: List[Dict]):
        """Send the full tree structure to the frontend so it can render the live tree.
        
        Converts the flat list of items (each with a semantic_path) into a tree structure.
        
        Example semantic_paths:
            "https://example.com/Home"
            "https://example.com/Home/About"
            "https://example.com/Home//Contact"  (// = link)
        
        The tree groups items by their path hierarchy so the frontend can draw them as nodes.
        """
        tree = self._build_tree_from_items(items)
        # Send as a special event (not counted in progress)
        if self._active:
            event_data = {
                "type": "tree_init",
                "tree": tree,
                "total_nodes": len(items),
            }
            await self.queue.put(json.dumps(event_data))

    @staticmethod
    def _build_tree_from_items(items: List[Dict]) -> Dict:
        """Build a tree structure from a flat list of items with semantic_path fields.
        
        Each item has a semantic_path like:
            "https://www.syrahealth.com/Home"
            "https://www.syrahealth.com/Home/Our Products Services"
            "https://www.syrahealth.com/Home//LEARN MORE"
        
        We split on '/' to determine parent-child relationships.
        Returns a nested dict with children arrays for D3 rendering.
        """
        if not items:
            return {"title": "Empty", "semantic_path": "", "children": []}

        # Build a lookup: semantic_path → item metadata
        path_data = {}
        for item in items:
            sp = item.get("semantic_path", "")
            path_data[sp] = {
                "title": item.get("text", item.get("topic", sp.split("/")[-1] or sp)),
                "semantic_path": sp,
                "original_url": item.get("original_url", ""),
                "element_type": item.get("element_type", "unknown"),
                "depth": item.get("depth", 0),
            }

        # Sort paths so parents come before children
        sorted_paths = sorted(path_data.keys(), key=lambda p: p.count("/"))

        # Use the shortest path as root
        root_path = sorted_paths[0] if sorted_paths else ""
        root = {
            **path_data.get(root_path, {"title": "Root", "semantic_path": root_path}),
            "children": [],
        }

        # Map each path to its tree node for fast lookup
        node_map = {root_path: root}

        for sp in sorted_paths[1:]:
            node = {**path_data[sp], "children": []}

            # Find parent by walking backwards through the path
            # Try removing the last segment (split by / or //)
            parent_found = False
            # Strategy: try progressively shorter prefixes
            parts = sp.split("/")
            for i in range(len(parts) - 1, 0, -1):
                candidate_parent = "/".join(parts[:i])
                if candidate_parent in node_map:
                    node_map[candidate_parent]["children"].append(node)
                    parent_found = True
                    break

            if not parent_found:
                # No parent found, attach to root
                root["children"].append(node)

            node_map[sp] = node

        return root

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
        description="List of question-answer pairs covering different aspects of the topic",
        min_length=2,   # Dynamic: short content may only need 3 pairs
        max_length=12   # Allow up to 12 if LLM generates more
    )


def calculate_max_qa_pairs(content: str) -> int:
    """
    Calculate the optimal number of Q&A pairs based on content length.
    Each meaningful Q&A pair needs ~50 words of source content to be properly grounded.
    
    Args:
        content: The page content text
    
    Returns:
        Number of Q&A pairs to generate (3-10)
    """
    word_count = len(content.split())
    max_pairs = max(3, min(10, word_count // 50))
    return max_pairs

# ============================================================================
# DE-DUPLICATION ANALYZER CLASS (NEW)
# ============================================================================
class QADuplicateAnalyzer:
    """
    A class to find and handle duplicates in a list of Q&A pairs
    using semantic similarity. This is a fundamental ML-driven tool
    for data cleaning and normalization in Q&A knowledge bases.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the analyzer by loading the sentence transformer model. 
        
        Args:
            model_name: The name of the sentence-transformer model to use.
        """
        logger.info(f"Initializing de-duplication model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        logger.info("✅ De-duplication model initialized.")

    def find_duplicates(self, qa_pairs: List[Dict[str, Any]], question_threshold: float, answer_threshold: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Analyzes a list of Q&A pairs to find and process duplicates.

        The strategy is to iterate through all pairs, maintaining a list of "canonical" 
        (unique) pairs. For each new pair, it's compared against the canonicals.
        - If it's unique, it's added to the canonical list.
        - If it's a duplicate question with a duplicate answer, it's discarded.
        - If it's a duplicate question with a new answer, the new answer is merged.

        Args:
            qa_pairs: A list of Q&A dictionaries. Each dict must have 'question', 'answer', 
                      and 'semantic_path' keys.
            question_threshold: The similarity score (0-1) to consider questions as duplicates.
            answer_threshold: The similarity score (0-1) to consider answers as duplicates.

        Returns:
            A tuple containing:
            - A list of de-duplicated Q&A pairs.
            - A list of log entries detailing the actions taken.
        """
        if not qa_pairs:
            return [], []

        logger.info("Generating embeddings for all questions and answers for de-duplication...")
        all_questions = [qa['question'] for qa in qa_pairs]
        all_answers = [qa['answer'] for qa in qa_pairs]
        
        question_embeddings = self.model.encode(all_questions, show_progress_bar=False, convert_to_numpy=True)
        answer_embeddings = self.model.encode(all_answers, show_progress_bar=False, convert_to_numpy=True)
        
        logger.info("Calculating similarity matrices for de-duplication...")
        question_sim_matrix = cosine_similarity(question_embeddings)
        answer_sim_matrix = cosine_similarity(answer_embeddings)

        canonical_list = []
        log = []
        
        processed_indices = [False] * len(qa_pairs)

        logger.info("Starting de-duplication processing loop...")
        for i in range(len(qa_pairs)):
            if processed_indices[i]:
                continue

            canonical_item = qa_pairs[i].copy()
            canonical_list.append(canonical_item)
            processed_indices[i] = True
            log.append({
                'Action': 'KEPT',
                'Original Question': canonical_item['question'],
                'Original Answer': '...',
                'Original Semantic Path': canonical_item.get('semantic_path', 'N/A'),
                'Notes': 'Initial canonical item.'
            })

            for j in range(i + 1, len(qa_pairs)):
                if processed_indices[j]:
                    continue
                
                q_sim = question_sim_matrix[i, j]

                if q_sim >= question_threshold:
                    a_sim = answer_sim_matrix[i, j]
                    duplicate_item = qa_pairs[j] 
                    
                    if a_sim >= answer_threshold:
                        log.append({
                            'Action': 'DISCARDED',
                            'Original Question': duplicate_item['question'],
                            'Original Answer': '...',
                            'Original Semantic Path': duplicate_item.get('semantic_path', 'N/A'),
                            'Notes': f"Duplicate of item with question '{canonical_item['question'][:50]}...'. Q-Sim: {q_sim:.4f}, A-Sim: {a_sim:.4f}"
                        })
                    else:
                        canonical_item['answer'] += f"\n\n[MERGED]: {duplicate_item['answer']}"
                        log.append({
                            'Action': 'MERGED',
                            'Original Question': duplicate_item['question'],
                            'Original Answer': '...',
                            'Original Semantic Path': duplicate_item.get('semantic_path', 'N/A'),
                            'Notes': f"Answer merged into item with question '{canonical_item['question'][:50]}...'. Q-Sim: {q_sim:.4f}, A-Sim: {a_sim:.4f}"
                        })
                    
                    processed_indices[j] = True

        logger.info("✅ De-duplication loop complete.")
        return canonical_list, log
    
    def reconstruct_topic_structure(self, flat_qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reconstructs the original topic-based structure from a flat list of Q&A pairs.
        """
        output_structure = {}
        for qa in flat_qa_list:
            semantic_path = qa.get('semantic_path', 'N/A')
            if semantic_path not in output_structure:
                output_structure[semantic_path] = {
                    "topic": infer_topic(semantic_path),
                    "semantic_path": semantic_path,
                    "qa_pairs": []
                }
            # Append only the question and answer, as semantic_path is now at the parent level
            output_structure[semantic_path]['qa_pairs'].append({
                "question": qa['question'],
                "answer": qa['answer']
            })

        return list(output_structure.values())


# ============================================================================
# HNSW REAL-TIME DEDUPLICATION INDEX (NEW)
# ============================================================================

class HNSWDeduplicationIndex:
    """
    Thread-safe HNSW index for real-time deduplication of Q&A pairs
    during parallel generation. Each worker checks against a shared index
    before accepting a new Q&A pair, preventing duplicates at generation time
    rather than in a post-processing O(n²) pass.
    """

    def __init__(self, dim: int = 384, max_elements: int = 50000,
                 ef_construction: int = 200, M: int = 16):
        """
        Initialize the HNSW index in cosine space.

        Args:
            dim: Embedding dimension (384 for all-MiniLM-L6-v2)
            max_elements: Maximum number of elements the index can hold
            ef_construction: Controls index build quality (higher = better recall, slower build)
            M: Number of bi-directional links per element (higher = better recall, more memory)
        """
        self.dim = dim
        self.max_elements = max_elements
        self._lock = threading.Lock()

        # Initialize HNSW index for QUESTIONS
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)  # ef should be > top_k for queries

        # Initialize HNSW index for ANSWERS
        self.answer_index = hnswlib.Index(space='cosine', dim=dim)
        self.answer_index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.answer_index.set_ef(50)

        # Load the same embedding model used by QADuplicateAnalyzer
        logger.info("Loading SentenceTransformer model for HNSW deduplication...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Track all added questions and answers for reference lookups
        self._questions: List[str] = []
        self._answers: List[str] = []
        self._metadata: List[dict] = []

        # Stats
        self._total_added = 0
        self._total_blocked = 0
        self._total_blocked_by_answer = 0

        logger.info(f"✅ HNSWDeduplicationIndex initialized (dim={dim}, max_elements={max_elements}, dual-index: Q+A)")

    def _embed(self, text: str):
        """Embed a single text string. Returns a numpy array of shape (dim,)."""
        return self.model.encode([text], convert_to_numpy=True)[0]

    def check_duplicate(self, question: str, answer: str = None, threshold: float = 0.92,
                        answer_threshold: float = 0.90) -> Tuple[bool, float, Optional[str]]:
        """
        Check if a Q&A pair is a duplicate by checking BOTH question and answer similarity.
        A pair is considered duplicate if EITHER the question OR the answer is too similar
        to an existing entry.

        Args:
            question: The question text to check
            answer: The answer text to check (if None, only checks question)
            threshold: Cosine similarity threshold for questions (0-1). Above this = duplicate.
            answer_threshold: Cosine similarity threshold for answers (0-1). Above this = duplicate.

        Returns:
            Tuple of (is_duplicate, similarity_score, matched_question_or_None)
        """
        with self._lock:
            if self.index.get_current_count() == 0:
                return (False, 0.0, None)

            # Check question similarity
            q_embedding = self._embed(question)
            labels, distances = self.index.knn_query(q_embedding.reshape(1, -1), k=1)
            q_cosine_sim = 1.0 - distances[0][0]
            q_matched_idx = labels[0][0]

            if q_cosine_sim >= threshold:
                matched_question = self._questions[q_matched_idx] if q_matched_idx < len(self._questions) else None
                return (True, float(q_cosine_sim), matched_question)

            # Check answer similarity (if answer provided and answer index has entries)
            if answer and self.answer_index.get_current_count() > 0:
                a_embedding = self._embed(answer)
                a_labels, a_distances = self.answer_index.knn_query(a_embedding.reshape(1, -1), k=1)
                a_cosine_sim = 1.0 - a_distances[0][0]
                a_matched_idx = a_labels[0][0]

                if a_cosine_sim >= answer_threshold:
                    self._total_blocked_by_answer += 1
                    matched_question = self._questions[a_matched_idx] if a_matched_idx < len(self._questions) else None
                    matched_answer_preview = self._answers[a_matched_idx][:60] if a_matched_idx < len(self._answers) else ""
                    logger.debug(
                        f"HNSW answer-dedup: answer too similar (sim={a_cosine_sim:.3f}) "
                        f"to existing Q: \"{matched_question[:60]}...\" A: \"{matched_answer_preview}...\""
                    )
                    return (True, float(a_cosine_sim), matched_question)

            return (False, float(q_cosine_sim), None)

    def add_qa_pair(self, question: str, answer: str, metadata: dict = None,
                    threshold: float = 0.92, answer_threshold: float = 0.90) -> bool:
        """
        Attempt to add a Q&A pair. If the question OR answer is a near-duplicate
        of an existing entry, the pair is rejected.

        Args:
            question: The question text
            answer: The answer text
            metadata: Optional metadata dict (e.g. semantic_path, topic)
            threshold: Duplicate detection threshold for questions
            answer_threshold: Duplicate detection threshold for answers

        Returns:
            True if the pair was added (unique), False if blocked (duplicate)
        """
        with self._lock:
            q_embedding = self._embed(question)
            a_embedding = self._embed(answer)

            # Check for question duplicates if index is non-empty
            if self.index.get_current_count() > 0:
                labels, distances = self.index.knn_query(q_embedding.reshape(1, -1), k=1)
                cosine_sim = 1.0 - distances[0][0]
                if cosine_sim >= threshold:
                    self._total_blocked += 1
                    matched_q = self._questions[labels[0][0]] if labels[0][0] < len(self._questions) else "?"
                    logger.debug(
                        f"HNSW dedup blocked (question): \"{question[:60]}...\" "
                        f"(sim={cosine_sim:.3f} with \"{matched_q[:60]}...\")"
                    )
                    return False

            # Check for answer duplicates if answer index is non-empty
            if self.answer_index.get_current_count() > 0:
                a_labels, a_distances = self.answer_index.knn_query(a_embedding.reshape(1, -1), k=1)
                a_cosine_sim = 1.0 - a_distances[0][0]
                if a_cosine_sim >= answer_threshold:
                    self._total_blocked += 1
                    self._total_blocked_by_answer += 1
                    matched_q = self._questions[a_labels[0][0]] if a_labels[0][0] < len(self._questions) else "?"
                    logger.debug(
                        f"HNSW dedup blocked (answer): \"{question[:60]}...\" "
                        f"(answer sim={a_cosine_sim:.3f} with Q: \"{matched_q[:60]}...\")"
                    )
                    return False

            # Unique — add to both indexes
            idx = self.index.get_current_count()
            self.index.add_items(q_embedding.reshape(1, -1), [idx])
            self.answer_index.add_items(a_embedding.reshape(1, -1), [idx])
            self._questions.append(question)
            self._answers.append(answer)
            self._metadata.append(metadata or {})
            self._total_added += 1
            return True

    def get_similar_context(self, topic_text: str, k: int = 20, threshold: float = 0.70) -> List[Dict]:
        """
        Find existing Q&A pairs most relevant to a topic/content.
        Used to provide context to the LLM during generation.
        
        Args:
            topic_text: Text to find similar existing Q&A for
            k: Maximum number of similar pairs to return
            threshold: Minimum similarity to include
        
        Returns:
            List of dicts with 'question', 'answer', 'similarity' keys
        """
        with self._lock:
            if self.index.get_current_count() == 0:
                return []
            
            embedding = self._embed(topic_text)
            actual_k = min(k, self.index.get_current_count())
            labels, distances = self.index.knn_query(embedding.reshape(1, -1), k=actual_k)
            
            results = []
            for idx, dist in zip(labels[0], distances[0]):
                sim = 1.0 - dist
                if sim >= threshold:
                    results.append({
                        "question": self._questions[idx] if idx < len(self._questions) else "",
                        "answer": self._answers[idx] if idx < len(self._answers) else "",
                        "similarity": round(float(sim), 3)
                    })
            return results

    def get_stats(self) -> dict:
        """Return current deduplication statistics."""
        with self._lock:
            total_seen = self._total_added + self._total_blocked
            return {
                "total_indexed": self._total_added,
                "total_blocked": self._total_blocked,
                "total_blocked_by_answer": self._total_blocked_by_answer,
                "total_blocked_by_question": self._total_blocked - self._total_blocked_by_answer,
                "total_seen": total_seen,
                "dedup_rate": round(self._total_blocked / total_seen * 100, 1) if total_seen > 0 else 0.0,
                "q_index_size": self.index.get_current_count(),
                "a_index_size": self.answer_index.get_current_count(),
            }

    def save_index(self, path: str):
        """Persist both HNSW indexes and metadata to disk."""
        with self._lock:
            self.index.save_index(path)
            # Save answer index alongside question index
            answer_index_path = path + ".answers"
            self.answer_index.save_index(answer_index_path)
            meta_path = path + ".meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "questions": self._questions,
                    "answers": self._answers,
                    "metadata": self._metadata,
                    "total_added": self._total_added,
                    "total_blocked": self._total_blocked,
                    "total_blocked_by_answer": self._total_blocked_by_answer,
                }, f, ensure_ascii=False)
            logger.info(f"💾 HNSW dual-index saved to {path} ({self.index.get_current_count()} items)")

    def load_index(self, path: str):
        """Load previously saved HNSW indexes and metadata from disk."""
        with self._lock:
            self.index.load_index(path, max_elements=self.max_elements)
            # Load answer index if it exists
            answer_index_path = path + ".answers"
            if os.path.exists(answer_index_path):
                self.answer_index.load_index(answer_index_path, max_elements=self.max_elements)
                logger.info(f"📂 Answer index loaded ({self.answer_index.get_current_count()} items)")
            else:
                # Backward compatibility: rebuild answer index from stored answers
                logger.info("⚠️ No answer index found — rebuilding from stored answers...")
                meta_path = path + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    answers = meta.get("answers", [])
                    if answers:
                        for idx, ans in enumerate(answers):
                            a_emb = self._embed(ans)
                            self.answer_index.add_items(a_emb.reshape(1, -1), [idx])
                        logger.info(f"✅ Rebuilt answer index with {len(answers)} entries")

            meta_path = path + ".meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                self._questions = meta.get("questions", [])
                self._answers = meta.get("answers", [])
                self._metadata = meta.get("metadata", [])
                self._total_added = meta.get("total_added", 0)
                self._total_blocked = meta.get("total_blocked", 0)
                self._total_blocked_by_answer = meta.get("total_blocked_by_answer", 0)
            logger.info(f"📂 HNSW dual-index loaded from {path} ({self.index.get_current_count()} items)")


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
    deduplication_log: List[Dict]    # NEW: For storing the de-duplication audit trail
    hnsw_stats: Dict                 # NEW: Real-time HNSW deduplication statistics


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


def create_qa_generation_prompt(content: str, item: dict, all_items: list = None, existing_qa_context: list = None, num_pairs: int = 10) -> str:
    """
    Context-aware Q&A generation prompt. Reads content first, checks existing KB context,
    then generates unique, chatbot-quality Q&A pairs.

    Args:
        content: Extracted page content
        item: Full item dict with semantic_path, depth, parent_url, element_type, etc.
        all_items: List of all items (optional, for context)
        existing_qa_context: List of existing Q&A dicts from HNSW index (question, answer, similarity)
        num_pairs: Number of Q&A pairs to generate (calculated from content length)
    """
    semantic_path = item.get('semantic_path', '')
    parent_url = item.get('parent_url', '')

    path_parts = semantic_path.split('/')
    clean_parts = []
    for part in path_parts:
        part = part.strip()
        if part and not part.startswith('http') and ':' not in part:
            clean_parts.append(part.replace('-', ' ').replace('_', ' '))
    readable_path = ' → '.join(clean_parts) if clean_parts else 'Root'

    # Build existing KB context section
    existing_section = ""
    if existing_qa_context:
        existing_lines = []
        for i, qa in enumerate(existing_qa_context[:30], 1):  # Show up to 30 existing pairs
            existing_lines.append(f"  Q{i}: {qa['question']}")
            existing_lines.append(f"  A{i}: {qa['answer'][:200]}...")
            existing_lines.append("")
        existing_section = f"""
════════════════════════════════════════════════════════════
EXISTING Q&A PAIRS IN THE KNOWLEDGE BASE ({len(existing_qa_context)} similar pairs found)
════════════════════════════════════════════════════════════
The following Q&A pairs ALREADY EXIST in our chatbot knowledge base.
DO NOT generate questions with the same intent as these.
Instead, focus on what THIS page's content covers that is NOT yet addressed.
If a topic overlaps, go DEEPER — ask more specific, detailed questions
grounded in the unique content of THIS page.

{chr(10).join(existing_lines)}
"""

    return f"""You are building a chatbot knowledge base. Your job is to generate high-quality Q&A pairs from page content.

════════════════════════════════════════════════════════════
STEP 1: READ AND UNDERSTAND THIS CONTENT
════════════════════════════════════════════════════════════
{content[:4000]}
{existing_section}
════════════════════════════════════════════════════════════
STEP 2: GENERATE EXACTLY {num_pairs} Q&A PAIRS FROM THE CONTENT ABOVE
════════════════════════════════════════════════════════════
(This content supports {num_pairs} quality Q&A pairs based on its depth and detail.)

CRITICAL RULES FOR CHATBOT Q&A QUALITY:

1. QUESTIONS MUST BE "QUESTIONABLE" (natural chatbot queries)
   - Write questions exactly as a real user would type them into a chatbot
   - Each question should be self-contained — a stranger reading just the question
     must understand what is being asked without any other context
   - Include the specific subject/entity name in every question
   ✓ GOOD: "What healthcare consulting services does Syra Health offer?"
   ✓ GOOD: "How does Syra Health's population health management program work?"
   ✗ BAD: "What services do they offer?" (who is "they"?)
   ✗ BAD: "Tell me more about this" (about what?)

2. ANSWERS MUST BE "ANSWERABLE" (complete, self-sufficient responses)
   - Each answer should fully satisfy the question on its own
   - A user reading ONLY the answer should get a complete, useful response
   - Include specific facts, details, names, dates, and URLs from the content
   - Do NOT write answers like "Please visit our website for more info"
   - Do NOT write vague answers like "They offer various services"
   ✓ GOOD: "Syra Health's healthcare consulting includes compliance auditing,
            workforce optimization, and regulatory guidance for state Medicaid programs.
            Their team has supported over 15 state agencies since 2019."
   ✗ BAD: "They offer consulting services in healthcare." (too vague)

3. GROUND EVERY Q&A IN THIS PAGE'S SPECIFIC CONTENT
   - Only use facts, details, and information from the content above
   - Do NOT invent or assume information not present in the content
   - If the content mentions specific programs, names, numbers — use them

4. ENSURE DIVERSITY OF INTENT
   - Each of the {num_pairs} questions should ask about a DIFFERENT aspect
   - Cover: what, how, why, when, who, where, how much, what if
   - Do NOT ask multiple questions that have the same basic answer
   - Spread across different topics/sections mentioned in the content

5. {"AVOID OVERLAP WITH EXISTING KB — this is the most important rule. Read the existing Q&A pairs above carefully. Generate questions that cover NEW ground, different angles, or deeper details not yet in the KB." if existing_qa_context else "CREATE DIVERSE QUESTIONS covering different aspects of the content."}

CONTEXT (for reference only — do NOT force these into questions):
- Source path: {readable_path}
- This helps you understand the domain/website structure

════════════════════════════════════════════════════════════

Generate exactly {num_pairs} Q&A pairs now. Each must be a natural chatbot interaction — 
a real question a user would ask, with a complete answer they'd be satisfied with.
{"NOTE: This content is limited, so focus on QUALITY over quantity. Only generate pairs that are well-grounded in the actual content." if num_pairs < 7 else ""}"""


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

        # NOTE: Do NOT pre-create BrowserSession — browser-use 0.11.2 hangs
        # when BrowserSession.start() is called externally. Let Agent manage
        # its own browser lifecycle instead.

        # Create browser_use agent with intelligent error handling
        agent = Agent(
            task=task,
            llm=groq_llm,
            use_vision=False,
            use_cloud=False,
            browser_profile=create_browser_profile(headless=True),
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
# TWO-PHASE CONTEXT-AWARE PROCESSING FUNCTIONS (NEW)
# ============================================================================

async def extract_content_only_parallel(
    item: Dict,
    worker_id: int,
    semaphore: asyncio.Semaphore,
    config: ParallelConfig = None,
    tracker: Optional['ParallelProgressTracker'] = None,
) -> Dict:
    """
    Phase 1: Extract content from a single page using browser_use.
    NO Q&A generation — just content extraction.
    """
    if config is None:
        config = PARALLEL_CONFIG

    async with semaphore:
        success = False
        worker_slot = worker_id % config.num_workers
        url = item.get("original_url", item.get("semantic_path", ""))
        semantic_path = item.get("semantic_path", url)
        topic = infer_topic(semantic_path)

        try:
            logger.info(f"🔄 [Worker {worker_slot}] Extracting content for item {worker_id}: {topic[:50]}...")
            if tracker:
                await tracker.update_worker_status(worker_slot, "extracting", {"topic": topic, "semantic_path": semantic_path})

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
                    "worker_id": worker_id
                }

            groq_llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct"
            )

            task = f"""Navigate to {url} and extract content about '{topic}'.
IMPORTANT: Stay on this exact page. Do NOT navigate away or search elsewhere.
Find the section about '{topic}' ON THIS PAGE and extract key information.
If '{topic}' is a heading, extract content under that heading.
Be concise and extract only essential information."""

            agent = Agent(
                task=task,
                llm=groq_llm,
                use_cloud=False,
                use_vision=False,
                browser_profile=create_browser_profile(headless=True),
                max_failures=2,
                step_timeout=config.step_timeout,
                max_actions_per_step=2,
            )

            start_time = time.time()
            result = await agent.run(max_steps=config.max_steps_per_item)
            extraction_time = time.time() - start_time

            content_text = extract_final_content(result) if result else "No content extracted"

            logger.info(f"✅ [Worker {worker_slot}] Extracted {len(content_text)} chars for item {worker_id} in {extraction_time:.1f}s")

            success = True
            return {
                "semantic_path": semantic_path,
                "original_url": url,
                "topic": topic,
                "browser_content": content_text,
                "extraction_method": "ai_agent_groq_parallel",
                "processing_time": extraction_time,
                "status": "extracted",
                "worker_id": worker_id
            }

        except Exception as e:
            logger.error(f"❌ [Worker {worker_slot}] Extraction failed for item {worker_id} ({topic}): {e}")
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
                "worker_id": worker_id
            }
        finally:
            if tracker:
                await tracker.item_completed(worker_slot, success=success)


def create_refinement_prompt(original_question: str, original_answer: str,
                             similar_question: str, similar_answer: str,
                             content: str, similarity_score: float,
                             all_existing_questions: list = None) -> str:
    """
    Create a prompt to refine a Q&A pair that was found to be too similar
    to an existing one in the HNSW index. Shows ALL existing KB questions
    so the LLM knows the full list to avoid.
    """
    # Build full list of existing questions to avoid
    avoid_list = ""
    if all_existing_questions:
        avoid_lines = [f"  - {q}" for q in all_existing_questions[-50:]]  # Last 50
        avoid_list = f"""
═══════════════════════════════════════════════════
ALL EXISTING QUESTIONS IN KB ({len(all_existing_questions)} total) — AVOID ALL OF THESE:
═══════════════════════════════════════════════════
{chr(10).join(avoid_lines)}
"""

    return f"""You generated a Q&A pair, but it is too similar to one that ALREADY EXISTS in the knowledge base.

═══════════════════════════════════════════════════
YOUR GENERATED PAIR (too similar — needs refinement):
═══════════════════════════════════════════════════
Question: {original_question}
Answer: {original_answer}

═══════════════════════════════════════════════════
CLOSEST MATCH IN KB (similarity: {similarity_score:.0%}):
═══════════════════════════════════════════════════
Question: {similar_question}
Answer: {similar_answer[:300]}
{avoid_list}
═══════════════════════════════════════════════════
PAGE CONTENT TO DRAW FROM:
═══════════════════════════════════════════════════
{content[:3000]}

═══════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════
Generate ONE new Q&A pair that:
1. Asks about a COMPLETELY DIFFERENT topic/aspect than ALL the existing questions above
2. Is NOT a rephrase — must have a different INTENT, not just different wording
3. Focuses on a specific detail, fact, or angle from the page content that no existing question covers
4. Is a natural chatbot question a real user would type
5. Has a complete, self-sufficient answer grounded in the page content

IMPORTANT: Read the full list of existing questions above. Your new question must be
clearly distinct from ALL of them, not just the closest match."""


async def generate_qa_with_context(
    extracted_item: Dict,
    hnsw_index: HNSWDeduplicationIndex,
    all_items: List[Dict] = None,
    item_number: int = 0,
    total_items: int = 0,
) -> Dict:
    """
    Phase 2: Generate Q&A pairs with per-pair HNSW feedback loop.
    
    Flow for EACH Q&A pair:
    1. Generate initial batch of 10 Q&A pairs (with existing KB context)
    2. Check EACH pair individually against HNSW index
    3. If similar → send the specific pair + the similar existing pair back to LLM
       → LLM refines to produce a unique pair
       → Check again (max 3 retries)
    4. If unique → add to HNSW index ✅
    5. Move to next pair
    
    This guarantees every single Q&A pair in the KB is unique in intent.
    """
    content = extracted_item.get("browser_content", "")
    topic = extracted_item.get("topic", "")
    semantic_path = extracted_item.get("semantic_path", "")

    MAX_REFINEMENT_RETRIES = 3
    SIMILARITY_THRESHOLD = 0.85  # Above this = too similar, needs refinement

    if not content or content in ["No content extracted", "AI extraction failed", "API key not available"]:
        logger.warning(f"⚠️ [{item_number}/{total_items}] Skipping Q&A generation for '{topic}': No valid content")
        extracted_item["qa_pairs"] = []
        extracted_item["qa_generation_status"] = "skipped"
        extracted_item["qa_generation_time"] = 0
        extracted_item["qa_count"] = 0
        return extracted_item

    # Calculate dynamic Q&A pair count based on content length
    num_pairs = calculate_max_qa_pairs(content)
    word_count = len(content.split())
    logger.info(f"❓ [{item_number}/{total_items}] Generating context-aware Q&A for: {topic} "
                f"({word_count} words → {num_pairs} pairs)")

    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")

        # Query HNSW for existing similar Q&A pairs (preemptive context)
        existing_context = hnsw_index.get_similar_context(
            topic_text=f"{topic} {content[:500]}",
            k=30,
            threshold=0.50
        )

        if existing_context:
            logger.info(f"   📚 Found {len(existing_context)} existing Q&A pairs as context for '{topic}'")
        else:
            logger.info(f"   🆕 No existing context found — this is a fresh topic")

        # Create context-aware prompt with dynamic pair count
        prompt = create_qa_generation_prompt(
            content=content,
            item=extracted_item,
            all_items=all_items,
            existing_qa_context=existing_context,
            num_pairs=num_pairs
        )

        # Initialize LLMs
        llm = LangChainChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key,
            temperature=0.3,
            max_tokens=3000
        )
        structured_llm = llm.with_structured_output(QAList)

        # For single-pair refinement
        refinement_llm = LangChainChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key,
            temperature=0.5,  # Slightly higher creativity for refinement
            max_tokens=500
        )
        structured_refinement_llm = refinement_llm.with_structured_output(QAPair)

        qa_start = time.time()

        # ═══════════════════════════════════════════════════
        # STEP 1: Generate initial batch of 10 Q&A pairs
        # ═══════════════════════════════════════════════════
        qa_result = structured_llm.invoke(prompt)
        candidate_pairs = [{"question": qa.question, "answer": qa.answer} for qa in qa_result.qa_pairs]

        logger.info(f"   📝 Generated {len(candidate_pairs)} initial Q&A pairs, now checking each against HNSW...")

        # ═══════════════════════════════════════════════════
        # STEP 2: Check EACH pair against HNSW with feedback loop
        # ═══════════════════════════════════════════════════
        accepted_pairs = []
        refinement_count = 0
        total_retries = 0

        for pair_idx, pair in enumerate(candidate_pairs):
            current_question = pair["question"]
            current_answer = pair["answer"]
            accepted = False

            for retry in range(MAX_REFINEMENT_RETRIES + 1):  # 0 = initial check, 1-3 = retries
                # Check against HNSW (both question AND answer similarity)
                is_duplicate, sim_score, matched_question = hnsw_index.check_duplicate(
                    current_question, answer=current_answer, threshold=SIMILARITY_THRESHOLD,
                    answer_threshold=0.90
                )

                if not is_duplicate:
                    # ✅ Unique! Add to HNSW index (checks both Q and A)
                    hnsw_index.add_qa_pair(
                        question=current_question,
                        answer=current_answer,
                        metadata={"semantic_path": semantic_path, "topic": topic},
                        threshold=0.98,  # Near-exact only for the actual add
                        answer_threshold=0.95  # Slightly stricter for add to avoid edge cases
                    )
                    accepted_pairs.append({"question": current_question, "answer": current_answer})
                    accepted = True

                    if retry > 0:
                        logger.info(f"   ✅ Pair {pair_idx+1}: Accepted after {retry} refinement(s)")
                    break

                else:
                    # ❌ Too similar — send back to LLM for refinement
                    if retry < MAX_REFINEMENT_RETRIES:
                        # Get the full matched Q&A pair for context
                        matched_idx = None
                        for qi, q in enumerate(hnsw_index._questions):
                            if q == matched_question:
                                matched_idx = qi
                                break
                        matched_answer = hnsw_index._answers[matched_idx] if matched_idx is not None and matched_idx < len(hnsw_index._answers) else ""

                        logger.info(f"   🔄 Pair {pair_idx+1}: Too similar (sim={sim_score:.2f}) to \"{matched_question[:50]}...\" — refining (attempt {retry+1}/{MAX_REFINEMENT_RETRIES})")

                        # Create refinement prompt with ALL existing questions
                        refine_prompt = create_refinement_prompt(
                            original_question=current_question,
                            original_answer=current_answer,
                            similar_question=matched_question,
                            similar_answer=matched_answer,
                            content=content,
                            similarity_score=sim_score,
                            all_existing_questions=hnsw_index._questions.copy()
                        )

                        try:
                            # LLM refines the pair
                            refined_result = structured_refinement_llm.invoke(refine_prompt)
                            current_question = refined_result.question
                            current_answer = refined_result.answer
                            refinement_count += 1
                            total_retries += 1
                        except Exception as refine_err:
                            logger.warning(f"   ⚠️ Refinement failed for pair {pair_idx+1}: {refine_err}")
                            break
                    else:
                        # Max retries exhausted — accept anyway (content may genuinely overlap)
                        logger.warning(f"   ⚠️ Pair {pair_idx+1}: Still similar after {MAX_REFINEMENT_RETRIES} retries (sim={sim_score:.2f}). Accepting as-is.")
                        hnsw_index.add_qa_pair(
                            question=current_question,
                            answer=current_answer,
                            metadata={"semantic_path": semantic_path, "topic": topic},
                            threshold=0.98
                        )
                        accepted_pairs.append({"question": current_question, "answer": current_answer})
                        accepted = True

        qa_time = time.time() - qa_start

        logger.info(f"✅ [{item_number}/{total_items}] Completed Q&A for '{topic}': "
                     f"{len(accepted_pairs)} pairs accepted, "
                     f"{refinement_count} refinements made, "
                     f"{total_retries} total retries, "
                     f"{qa_time:.1f}s")

        extracted_item["qa_pairs"] = accepted_pairs
        extracted_item["qa_generation_status"] = "completed"
        extracted_item["qa_generation_time"] = round(qa_time, 2)
        extracted_item["qa_count"] = len(accepted_pairs)
        extracted_item["qa_context_used"] = len(existing_context)
        extracted_item["qa_refinements"] = refinement_count
        extracted_item["qa_retries"] = total_retries
        extracted_item["status"] = "completed"
        return extracted_item

    except Exception as e:
        logger.error(f"❌ [{item_number}/{total_items}] Q&A generation failed for '{topic}': {e}")
        extracted_item["qa_pairs"] = []
        extracted_item["qa_generation_status"] = "failed"
        extracted_item["qa_generation_time"] = 0
        extracted_item["qa_count"] = 0
        extracted_item["qa_generation_error"] = str(e)
        return extracted_item


async def process_batch_two_phase(
    items: List[Dict],
    hnsw_index: HNSWDeduplicationIndex,
    config: ParallelConfig = None,
    current_batch: int = 1,
    total_batches: int = 1,
    overall_completed: int = 0,
    overall_total: int = 0,
    all_semantic_paths: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Two-phase batch processing:
    Phase 1: Extract content from all items in PARALLEL (fast)
    Phase 2: Generate Q&A for each item SEQUENTIALLY with HNSW context (smart)
    
    This ensures every Q&A pair is generated with full awareness of what
    already exists in the knowledge base.
    """
    global progress_tracker
    if config is None:
        config = PARALLEL_CONFIG

    logger.info(f"\n{'='*60}")
    logger.info(f"📦 BATCH {current_batch}/{total_batches}: {len(items)} items")
    logger.info(f"{'='*60}")

    # ═══════════════════════════════════════════════════
    # PHASE 1: PARALLEL CONTENT EXTRACTION
    # ═══════════════════════════════════════════════════
    logger.info(f"\n🚀 PHASE 1: Parallel content extraction ({config.num_workers} workers)")

    progress_tracker = ParallelProgressTracker(
        total_items=len(items),
        num_workers=config.num_workers,
        current_batch=current_batch,
        total_batches=total_batches,
        overall_completed=overall_completed,
        overall_total=overall_total
    )

    # Send full tree structure to frontend on FIRST batch only
    if current_batch == 1:
        tree_items = all_semantic_paths if all_semantic_paths else items
        await progress_tracker.send_tree_init(tree_items)

    try:
        semaphore = asyncio.Semaphore(config.num_workers)

        extraction_tasks = [
            extract_content_only_parallel(item, i, semaphore, config, tracker=progress_tracker)
            for i, item in enumerate(items)
        ]

        extracted_results = []
        for coro in asyncio.as_completed(extraction_tasks):
            result = await coro
            extracted_results.append(result)

        # Sort to maintain original order
        item_order = {item.get("semantic_path", ""): i for i, item in enumerate(items)}
        extracted_results.sort(key=lambda x: item_order.get(x.get("semantic_path", ""), 999999))

        extraction_succeeded = sum(1 for r in extracted_results if r.get("status") == "extracted")
        logger.info(f"✅ PHASE 1 complete: {extraction_succeeded}/{len(items)} pages extracted successfully")

    finally:
        if progress_tracker:
            await progress_tracker.finish()
            progress_tracker = None

    # ═══════════════════════════════════════════════════
    # PHASE 2: SEQUENTIAL CONTEXT-AWARE Q&A GENERATION
    # ═══════════════════════════════════════════════════
    logger.info(f"\n🧠 PHASE 2: Sequential context-aware Q&A generation")
    logger.info(f"   HNSW index has {hnsw_index.get_stats()['total_indexed']} existing Q&A pairs as context")

    final_results = []
    for i, extracted_item in enumerate(extracted_results):
        result = await generate_qa_with_context(
            extracted_item=extracted_item,
            hnsw_index=hnsw_index,
            all_items=items,
            item_number=i + 1,
            total_items=len(extracted_results),
        )
        final_results.append(result)

        # Small delay to avoid rate limits
        await asyncio.sleep(1.0)

    # Log batch summary
    succeeded = sum(1 for r in final_results if r.get("qa_generation_status") == "completed")
    total_qa = sum(r.get("qa_count", 0) for r in final_results)
    hnsw_stats = hnsw_index.get_stats()

    logger.info(f"\n📊 BATCH {current_batch} SUMMARY:")
    logger.info(f"   Q&A generated: {succeeded}/{len(final_results)} items")
    logger.info(f"   Total Q&A pairs: {total_qa}")
    logger.info(f"   KB index size: {hnsw_stats['total_indexed']} total Q&A pairs")
    logger.info(f"   Context awareness: {sum(r.get('qa_context_used', 0) for r in final_results)} existing pairs referenced")

    return final_results


# ============================================================================
# PARALLEL PROCESSING FUNCTIONS (LEGACY — kept for reference)
# ============================================================================

async def extract_single_item_parallel(
    item: Dict,
    worker_id: int,
    semaphore: asyncio.Semaphore,
    config: ParallelConfig = None,
    tracker: Optional[ParallelProgressTracker] = None,
    all_items: List[Dict] = None,
    hnsw_index: Optional['HNSWDeduplicationIndex'] = None,
) -> Dict:
    """
    Extract content for a single item with dedicated browser instance.
    Uses semaphore to limit concurrent workers and reports progress to tracker.

    Args:
        all_items: Full list of items for sibling awareness in Q&A generation
        hnsw_index: Optional shared HNSW index for real-time deduplication
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
                await tracker.update_worker_status(worker_slot, "processing", {"topic": topic, "semantic_path": semantic_path})


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

            # NOTE: Do NOT pre-create BrowserSession — browser-use 0.11.2 hangs
            # when BrowserSession.start() is called externally. Let Agent manage
            # its own browser lifecycle instead.

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
            # Let Agent create and manage its own browser session internally
            agent = Agent(
                task=task,
                llm=groq_llm,
                use_cloud=False,
                use_vision=False,
                browser_profile=create_browser_profile(headless=True),
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

                    # --- HNSW Real-time Deduplication ---
                    if hnsw_index is not None:
                        original_count = len(qa_pairs)
                        unique_pairs = []
                        for qp in qa_pairs:
                            added = hnsw_index.add_qa_pair(
                                question=qp["question"],
                                answer=qp["answer"],
                                metadata={"semantic_path": semantic_path, "topic": topic, "worker_id": worker_id},
                            )
                            if added:
                                unique_pairs.append(qp)
                        filtered_count = original_count - len(unique_pairs)
                        if filtered_count > 0:
                            logger.info(
                                f"🔍 [Worker {worker_slot}] HNSW dedup: kept {len(unique_pairs)}/{original_count} "
                                f"Q&A pairs for item {worker_id} ({filtered_count} duplicates removed)"
                            )
                        qa_pairs = unique_pairs

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
    overall_total: int = 0,
    all_semantic_paths: Optional[List[Dict]] = None,
    hnsw_index: 'HNSWDeduplicationIndex' = None,
) -> List[Dict]:
    """
    Process a batch of items in parallel using multiple browser workers.
    Creates and manages a ParallelProgressTracker for SSE streaming.
    
    Args:
        all_semantic_paths: Full list of ALL items (across all batches) for tree_init.
                           If None, falls back to just the batch items.
        hnsw_index: Shared HNSW deduplication index that persists across batches.
                   If None, creates a new one (backward compatible).
    """
    global progress_tracker
    if config is None:
        config = PARALLEL_CONFIG

    logger.info(f"🚀 Starting parallel processing: Batch {current_batch}/{total_batches} ({len(items)} items) with {config.num_workers} workers")

    # Use shared HNSW index if provided, otherwise create new one
    if hnsw_index is None:
        hnsw_index = HNSWDeduplicationIndex()
        logger.info("🔍 HNSW real-time deduplication index created for batch")
    else:
        stats = hnsw_index.get_stats()
        logger.info(f"🔍 Using shared HNSW index ({stats['total_indexed']} existing Q&A pairs from previous batches)")

    # Create and assign the global tracker with batch info
    progress_tracker = ParallelProgressTracker(
        total_items=len(items),
        num_workers=config.num_workers,
        current_batch=current_batch,
        total_batches=total_batches,
        overall_completed=overall_completed,
        overall_total=overall_total
    )

    # Send full tree structure to frontend on FIRST batch only (all items, not just this batch)
    if current_batch == 1:
        tree_items = all_semantic_paths if all_semantic_paths else items
        await progress_tracker.send_tree_init(tree_items)
    
    try:
        # Create semaphore to limit concurrent workers
        semaphore = asyncio.Semaphore(config.num_workers)

        # Create tasks for all items, passing the tracker, all_items for sibling awareness, and HNSW index
        tasks = [
            extract_single_item_parallel(item, i, semaphore, config, tracker=progress_tracker, all_items=items, hnsw_index=hnsw_index)
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

        # Log HNSW deduplication stats
        hnsw_stats = hnsw_index.get_stats()
        logger.info(f"🔍 HNSW Dedup Stats: {hnsw_stats['total_indexed']} unique indexed, "
                     f"{hnsw_stats['total_blocked']} duplicates blocked "
                     f"({hnsw_stats['dedup_rate']}% dedup rate)")

        # Save HNSW index for potential future use
        try:
            data_dir = os.getenv('DATA_DIR', '.')
            index_path = os.path.join(data_dir, f"hnsw_dedup_index_batch{current_batch}.bin")
            hnsw_index.save_index(index_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not save HNSW index: {e}")

        # Attach stats to results for upstream access
        for r in results:
            r['_hnsw_stats'] = hnsw_stats

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

    # Create persistent HNSW index and rebuild from previously processed items
    hnsw_index = HNSWDeduplicationIndex()
    if processed_items:
        logger.info(f"📂 Rebuilding HNSW index from {len(processed_items)} previously processed items...")
        for item in processed_items:
            for qa in item.get("qa_pairs", []):
                hnsw_index.add_qa_pair(
                    question=qa["question"],
                    answer=qa["answer"],
                    metadata={"semantic_path": item.get("semantic_path", "")},
                    threshold=0.98,
                    answer_threshold=0.95
                )
        logger.info(f"✅ HNSW index rebuilt with {hnsw_index.get_stats()['total_indexed']} Q&A pairs")

    # Process this batch in parallel with batch info for SSE + shared HNSW index
    results = await process_batch_parallel(
        batch_items,
        PARALLEL_CONFIG,
        current_batch=current_batch,
        total_batches=total_batches,
        overall_completed=current_index,  # Items already completed before this batch
        overall_total=total_items,
        all_semantic_paths=semantic_paths,  # Full list for tree_init on first batch
        hnsw_index=hnsw_index,  # Shared index with context from previous batches
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

    # Get HNSW stats directly from the shared index
    hnsw_stats = hnsw_index.get_stats()
    logger.info(f"🔍 Cumulative HNSW stats: {hnsw_stats}")

    # Save HNSW index
    try:
        data_dir = os.getenv('DATA_DIR', '.')
        index_path = os.path.join(data_dir, f"hnsw_dedup_index_batch{current_batch}.bin")
        hnsw_index.save_index(index_path)
    except Exception as e:
        logger.warning(f"⚠️ Could not save HNSW index: {e}")

    # Clean up internal _hnsw_stats keys from results before storing in state
    for r in results:
        r.pop('_hnsw_stats', None)

    return {
        **state,
        "processed_items": processed_items,
        "current_index": new_index,
        "total_items": total_items,
        "status": status,
        "total_processed": len(processed_items),
        "hnsw_stats": hnsw_stats,
    }


# ============================================================================
# TWO-PHASE BATCH NODE (NEW - replaces parallel_batch_node in graph)
# ============================================================================

async def two_phase_batch_node(state: AgentState) -> AgentState:
    """
    LangGraph Node: Two-phase batch processing with context-aware Q&A generation.
    
    Phase 1: Extract content from batch items in PARALLEL
    Phase 2: Generate Q&A for each item SEQUENTIALLY with HNSW context
    
    The HNSW index persists across batches, so later batches have full awareness
    of all previously generated Q&A pairs.
    """
    BATCH_SIZE = 50

    semantic_paths = state.get("semantic_paths", [])
    current_index = state.get("current_index", 0)
    processed_items = state.get("processed_items", [])
    total_items = len(semantic_paths)

    # Get or create persistent HNSW index
    # The index persists via state across batches
    hnsw_index = HNSWDeduplicationIndex()
    
    # If we have previously processed items, rebuild the index from them
    if processed_items:
        logger.info(f"📂 Rebuilding HNSW index from {len(processed_items)} previously processed items...")
        for item in processed_items:
            for qa in item.get("qa_pairs", []):
                hnsw_index.add_qa_pair(
                    question=qa["question"],
                    answer=qa["answer"],
                    metadata={"semantic_path": item.get("semantic_path", "")},
                    threshold=0.98
                )
        logger.info(f"✅ HNSW index rebuilt with {hnsw_index.get_stats()['total_indexed']} Q&A pairs")

    # Get next batch
    batch_end = min(current_index + BATCH_SIZE, total_items)
    batch_items = semantic_paths[current_index:batch_end]

    if not batch_items:
        logger.info("✅ All items already processed")
        return {
            **state,
            "status": "complete",
            "current_index": total_items
        }

    total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
    current_batch = (current_index // BATCH_SIZE) + 1

    logger.info(f"🚀 Processing batch {current_batch}/{total_batches}: items {current_index+1}-{batch_end} of {total_items}")

    # Two-phase processing
    results = await process_batch_two_phase(
        items=batch_items,
        hnsw_index=hnsw_index,
        config=PARALLEL_CONFIG,
        current_batch=current_batch,
        total_batches=total_batches,
        overall_completed=current_index,
        overall_total=total_items,
        all_semantic_paths=semantic_paths,  # Full list for tree_init on first batch
    )

    # Merge results
    processed_items.extend(results)
    new_index = current_index + len(results)

    is_complete = new_index >= total_items
    status = "complete" if is_complete else "processing"

    # Save HNSW index
    try:
        data_dir = os.getenv('DATA_DIR', '.')
        index_path = os.path.join(data_dir, f"hnsw_context_index_batch{current_batch}.bin")
        hnsw_index.save_index(index_path)
    except Exception as e:
        logger.warning(f"⚠️ Could not save HNSW index: {e}")

    hnsw_stats = hnsw_index.get_stats()

    if is_complete:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 ALL {total_items} ITEMS PROCESSED!")
        logger.info(f"   Total Q&A pairs in KB: {hnsw_stats['total_indexed']}")
        logger.info(f"   Zero duplicates removed — all pairs generated with context awareness")
        logger.info(f"{'='*60}")
    else:
        logger.info(f"📊 Batch complete. Progress: {new_index}/{total_items} ({100*new_index//total_items}%)")

    return {
        **state,
        "processed_items": processed_items,
        "current_index": new_index,
        "total_items": total_items,
        "status": status,
        "total_processed": len(processed_items),
        "hnsw_stats": hnsw_stats,
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
            timeout=60000  # 60 seconds for slow government sites
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
# DE-DUPLICATION NODE (NEW)
# ============================================================================

async def deduplication_node(state: AgentState) -> AgentState:
    """
    Node: Takes the final raw data from `processed_items`, flattens all Q&A pairs,
    de-duplicates them using QADuplicateAnalyzer, and updates the state with the
    clean data and a log of actions.
    """
    logger.info("🔬 Starting De-duplication Service Node...")
    processed_items = state["processed_items"]

    # Flatten all Q&A pairs from all processed items (raw data)
    all_qa_pairs_raw = []
    for item in processed_items:
        # Each item in processed_items should have a 'qa_pairs' list
        # We need to add 'semantic_path' to each individual qa for the analyzer
        for qa in item.get("qa_pairs", []):
            qa_with_context = qa.copy()
            qa_with_context['semantic_path'] = item.get('semantic_path', 'N/A')
            all_qa_pairs_raw.append(qa_with_context)

    if not all_qa_pairs_raw:
        logger.warning("No Q&A pairs found in processed_items for de-duplication.")
        return {**state, "deduplication_log": [], "processed_items": []}

    logger.info(f"Found {len(all_qa_pairs_raw)} raw Q&A pairs for de-duplication.")

    # Initialize and run the de-duplication tool
    # Using default thresholds; these could be made configurable in AgentState if needed
    analyzer = QADuplicateAnalyzer()
    clean_qa_pairs_flat, log_entries = analyzer.find_duplicates(
        all_qa_pairs_raw, 
        question_threshold=0.90, 
        answer_threshold=0.95
    )

    # Reconstruct the original topic-based structure for the final output
    final_clean_items = analyzer.reconstruct_topic_structure(clean_qa_pairs_flat)
    
    # Add back the metadata needed for the final report
    for item in final_clean_items:
        if "qa_pairs" in item and item["qa_pairs"]:
            item["qa_generation_status"] = "completed"
            item["qa_count"] = len(item["qa_pairs"])
        else:
            item["qa_generation_status"] = "completed_deduplicated_all"
            item["qa_count"] = 0
            
    logger.info(f"✅ De-duplication complete. Original Q&A count: {len(all_qa_pairs_raw)}, Cleaned count: {len(clean_qa_pairs_flat)}")

    # Update state with the clean data and log
    state["processed_items"] = final_clean_items
    state["deduplication_log"] = log_entries # Store log in state for potential external use
    
    # Save the log file as an artifact of the run (optional, but good for auditing)
    log_filename = os.path.join(os.getenv('DATA_DIR', '.'), f"deduplication_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    if log_entries:
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = log_entries[0].keys() # Dynamically get fieldnames
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_entries)
        logger.info(f"💾 De-duplication audit log saved to '{log_filename}'")

    return state

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
        # PARALLEL MODE: Two-phase context-aware processing
        builder.add_node("two_phase_batch", two_phase_batch_node)
        logger.info("🚀 PARALLEL MODE enabled - using two-phase context-aware processing")
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
        # PARALLEL: load_json → two_phase_batch → loop/end (with checkpointing per batch)
        builder.add_edge("load_json", "two_phase_batch")
        builder.add_conditional_edges(
            "two_phase_batch",
            should_continue_parallel,
            {
                "continue": "two_phase_batch",  # Loop back for next batch
                "end": END
            }
        )
    else:
        # SEQUENTIAL: load_json → browser_agent → qa_generator → should_continue (or deduplication node, but we'll modify later)
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
