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
from typing import Dict, List, TypedDict, Optional

from browser_use import Agent, BrowserSession, ChatGroq
from dotenv import load_dotenv
from langchain_groq import ChatGroq as LangChainChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from hierarchical_crawler import HierarchicalWebCrawler
from pydantic import BaseModel, Field

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


def create_qa_generation_prompt(content: str, topic: str) -> str:
    """
    Create prompt for Q&A pair generation
    Generates 10 plausible questions and comprehensive answers based on content
    """
    return f"""You are a helpful assistant that generates question-answer pairs for chatbot training.

Your task: Analyze the following content about "{topic}" and generate 10 diverse, plausible questions that users might ask, along with comprehensive answers.

CRITICAL REQUIREMENTS:
1. Generate EXACTLY 10 question-answer pairs
2. Cover different question types:
   - What/Definition questions (e.g., "What is {topic}?")
   - How-to/Procedural questions (e.g., "How do I use {topic}?")
   - Why/Reasoning questions (e.g., "Why should I use {topic}?")
   - When/Timing questions (e.g., "When is {topic} available?")
   - Where/Location questions (e.g., "Where can I find {topic}?")
   - Who/Contact questions (e.g., "Who should I contact about {topic}?")

3. Make questions natural and conversational (how real users would ask)
4. Answers must be comprehensive and include:
   - ALL important URLs and links
   - ALL dates, times, and contact information
   - ALL procedures and steps
   - ALL requirements and prerequisites
   - Clear, actionable information

5. Keep answers easy to understand but comprehensive
6. Base all answers ONLY on the provided content (don't make up information)

CONTENT ABOUT "{topic}":
{content}

Generate 10 question-answer pairs now."""


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

        # Create extraction task
        task = f"""
Navigate to the webpage at {url}.

Thoroughly read and analyze the entire webpage about '{topic}'. Extract comprehensive information including:

1. A clear explanation of what '{topic}' is and its purpose
2. All relevant links mentioned on the page (URLs and what they link to)
3. Any instructions, phone numbers, people names, addresses, or related resources mentioned
4. Key features, capabilities, or important details
5. Any other relevant information that helps understand '{topic}'

Provide a detailed, structured response that covers all aspects found on the page.
"""

        # Create browser session
        browser_session = BrowserSession(headless=True)

        # Create browser_use agent
        agent = Agent(
            task=task,
            llm=groq_llm,
            use_cloud=False,
            browser_session=browser_session
        )

        # Execute extraction
        logger.info(f"🌐 Extracting content for: {topic}")
        start_time = time.time()
        result = await agent.run()
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
# LANGGRAPH NODES
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
        json_path = state.get("json_path", "/Users/saiteja/Documents/Dev/StarshipChatbot/mhs_indianafiltered_data.json")

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

        # Create Q&A generation prompt
        prompt = create_qa_generation_prompt(browser_content, topic)

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


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_browser_agent_graph(checkpointer: Optional[AsyncSqliteSaver] = None, enable_crawler: bool = False):
    """
    Build LangGraph workflow for Multi-Agent Pipeline with Conditional Routing

    Flow (with crawler):
    START → crawler_node → load_json → browser_agent → qa_generator → should_continue?
                                             ↑                              |
                                             |------------ YES -------------|
                                             |
                                             NO
                                             ↓
                                            END

    Flow (without crawler):
    START → load_json → browser_agent → qa_generator → should_continue?
                             ↑                              |
                             |------------ YES -------------|
                             |
                             NO
                             ↓
                            END

    Args:
        checkpointer: Optional AsyncSqliteSaver for state persistence and resume functionality
        enable_crawler: If True, includes crawler_node as first step
    """
    # Create graph builder
    builder = StateGraph(AgentState)

    # Add nodes
    if enable_crawler:
        builder.add_node("crawler", crawler_node)
    builder.add_node("load_json", load_json_node)
    builder.add_node("browser_agent", browser_agent_node)
    builder.add_node("qa_generator", qa_generator_node)

    # Define edges
    if enable_crawler:
        builder.add_edge(START, "crawler")         # START → crawler
        builder.add_edge("crawler", "load_json")   # crawler → load_json
    else:
        builder.add_edge(START, "load_json")       # START → load_json (skip crawler)

    builder.add_edge("load_json", "browser_agent")
    builder.add_edge("browser_agent", "qa_generator")  # Browser → Q&A Generator

    # Conditional edge: Loop back or finish
    builder.add_conditional_edges(
        "qa_generator",       # From qa_generator node
        should_continue,      # Call this router function
        {
            "continue": "browser_agent",  # Loop back to process next item
            "end": END                     # All done, go to END
        }
    )

    # Compile graph with optional checkpointer
    if checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        logger.info(f"✅ Multi-Agent graph compiled with checkpointing enabled")
    else:
        graph = builder.compile()
        logger.info(f"✅ Multi-Agent graph compiled without checkpointing")

    if enable_crawler:
        logger.info("🕷️ Crawler node enabled as first step")

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
    # Crawler parameters (NEW)
    enable_crawler: bool = False,
    start_url: str = None,
    max_depth: int = 2,
    max_pages: int = 100
):
    """
    Execute Multi-Agent Pipeline (Crawler + Browser + Paraphraser) with conditional routing

    Args:
        json_path: Path to JSON file with semantic paths (used if crawler disabled)
        output_file: Where to save extracted and paraphrased content
        max_items: Maximum number of items to process (None = process all)
        thread_id: Thread ID for checkpointing (same ID = resume from last checkpoint)
        checkpoint_db: Path to SQLite checkpoint database
        enable_checkpointing: Enable/disable checkpointing and resume functionality
        enable_crawler: Enable/disable hierarchical web crawler as first step
        start_url: Website URL to crawl (required if enable_crawler=True)
        max_depth: Maximum crawl depth (default: 2)
        max_pages: Maximum pages to crawl (default: 100)

    Returns:
        Final state with extracted and paraphrased content
    """
    logger.info("🚀 Starting Multi-Agent Pipeline")

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
        "json_path": json_path or "/Users/saiteja/Documents/Dev/StarshipChatbot/mhs_indianafiltered_data.json",
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
                enable_crawler=enable_crawler
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
            enable_crawler=enable_crawler
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
            json_path = "/Users/saiteja/Documents/Dev/StarshipChatbot/mhs_indianafiltered_data.json"
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

    # Ask user how many items to process
    print("🔧 Processing Options:")
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
        max_pages=max_pages
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
