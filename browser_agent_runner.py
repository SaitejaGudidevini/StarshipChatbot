"""
Browser Agent Runner - Async Wrapper for Background Execution
==============================================================

Wraps browser_agent.py to run as async background task with real-time progress tracking.
Enables running data generation from web UI with SSE (Server-Sent Events) updates.

Features:
- Async execution (non-blocking)
- Real-time progress tracking
- Error handling and recovery
- Mock mode for testing
- Saves results incrementally

Usage:
    from browser_agent_runner import create_runner

    runner = create_runner()
    async for progress in runner.run("https://example.com", max_pages=10):
        print(progress)  # {'status': 'running', 'current': 3, 'total': 10, ...}
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# MOCK RUNNER (For Testing)
# ============================================================================

class MockBrowserAgentRunner:
    """
    Mock runner that simulates browser_agent.py behavior

    Use this for testing the UI without running actual scraping.
    Generates fake data and provides realistic progress updates.
    """

    def __init__(self, output_file: str = "browser_agent_test_output.json"):
        self.output_file = output_file
        self.progress = {
            'status': 'idle',
            'current': 0,
            'total': 0,
            'current_url': '',
            'current_topic': '',
            'qa_generated': 0,
            'topics_completed': 0,
            'error': None,
            'started_at': None,
            'completed_at': None,
            'elapsed_seconds': 0
        }
        self._start_time = None
        self._is_running = False

    async def run(
        self,
        url: str,
        max_pages: int = 10,
        use_crawler: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Simulate browser agent execution

        Args:
            url: Starting URL
            max_pages: Number of pages to process
            use_crawler: Whether to use crawler (ignored in mock)

        Yields:
            Progress updates every ~2 seconds
        """
        self._is_running = True
        self.progress['status'] = 'initializing'
        self.progress['started_at'] = datetime.now().isoformat()
        self._start_time = datetime.now()
        self.progress['total'] = max_pages

        yield self.progress

        await asyncio.sleep(1)

        self.progress['status'] = 'running'
        yield self.progress

        # Simulate processing pages
        for i in range(max_pages):
            if not self._is_running:
                self.progress['status'] = 'cancelled'
                yield self.progress
                return

            # Simulate work (2 seconds per page)
            await asyncio.sleep(2)

            self.progress['current'] = i + 1
            self.progress['current_url'] = f"{url}/page-{i+1}"
            self.progress['current_topic'] = f"Topic {i+1}: Example Content"
            self.progress['qa_generated'] = (i + 1) * 10
            self.progress['topics_completed'] = i + 1

            if self._start_time:
                elapsed = (datetime.now() - self._start_time).total_seconds()
                self.progress['elapsed_seconds'] = int(elapsed)

            yield self.progress

        # Generate mock data
        logger.info(f"Generating mock data with {max_pages} topics")

        mock_data = []
        for i in range(max_pages):
            topic_data = {
                "topic": f"Topic {i+1}",
                "semantic_path": f"{url}/Topic {i+1}",
                "original_url": f"{url}/page-{i+1}",
                "browser_content": f"This is mock content for topic {i+1}. " * 10,
                "extraction_method": "mock",
                "processing_time": 2.0,
                "status": "completed",
                "error": "",
                "qa_pairs": [
                    {
                        "question": f"What is question {j+1} about topic {i+1}?",
                        "answer": f"This is answer {j+1} for topic {i+1}. It provides detailed information about the subject matter."
                    }
                    for j in range(10)
                ]
            }
            mock_data.append(topic_data)

        # Save mock data
        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mock_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Mock data saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save mock data: {e}")
            self.progress['status'] = 'error'
            self.progress['error'] = f"Failed to save results: {str(e)}"
            yield self.progress
            return

        # Completion
        self.progress['status'] = 'completed'
        self.progress['completed_at'] = datetime.now().isoformat()
        self._is_running = False

        yield self.progress

        logger.info(f"Mock generation completed. Generated {self.progress['qa_generated']} Q&A pairs")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress snapshot"""
        if self._start_time and self.progress['status'] == 'running':
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.progress['elapsed_seconds'] = int(elapsed)

        return self.progress.copy()

    def is_running(self) -> bool:
        """Check if generation is currently running"""
        return self._is_running

    def cancel(self):
        """Cancel running generation"""
        logger.info("Cancelling mock generation")
        self._is_running = False


# ============================================================================
# REAL RUNNER (Wraps browser_agent.py)
# ============================================================================

class BrowserAgentRunner:
    """
    Real runner that wraps browser_agent.py for async execution

    Note: Requires browser_agent.py and all dependencies to be available.
    Falls back to MockBrowserAgentRunner if not available.
    """

    def __init__(self, output_file: str = "browser_agent_test_output.json"):
        self.output_file = output_file
        self.progress = {
            'status': 'idle',
            'current': 0,
            'total': 0,
            'current_url': '',
            'current_topic': '',
            'qa_generated': 0,
            'topics_completed': 0,
            'error': None,
            'started_at': None,
            'completed_at': None,
            'elapsed_seconds': 0
        }
        self._start_time = None
        self._is_running = False
        self._graph = None

    async def run(
        self,
        url: str,
        max_pages: int = 10,
        use_crawler: bool = True,
        max_depth: int = 2,
        max_items: int = None,
        thread_id: str = None,
        enable_checkpointing: bool = True,
        json_filename: str = None,
        enable_parallel: bool = True  # Default to parallel for speed
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run browser agent pipeline

        Args:
            url: Starting URL to scrape
            max_pages: Maximum number of pages to process
            use_crawler: Whether to use hierarchical crawler
            max_depth: Maximum crawl depth (1-5)
            max_items: Limit items to process (None = all)
            thread_id: Custom thread ID for checkpointing (None = auto-generate)
            enable_checkpointing: Enable/disable checkpointing
            enable_parallel: Use parallel processing (5 concurrent browsers, 5x-10x faster)

        Yields:
            Progress updates as processing occurs
        """
        self._is_running = True

        try:
            # Import browser_agent components
            from browser_agent import build_browser_agent_graph
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self.progress['status'] = 'initializing'
            self.progress['started_at'] = datetime.now().isoformat()
            self._start_time = datetime.now()
            yield self.progress

            logger.info(f"Starting browser agent for URL: {url}")

            # Build graph with checkpointing
            self.progress['status'] = 'building_graph'
            yield self.progress

            # Use async context manager for checkpointer (LangGraph requirement)
            async with AsyncSqliteSaver.from_conn_string("browser_agent_checkpoints.db") as checkpointer:
                self._graph = build_browser_agent_graph(
                    checkpointer,
                    enable_crawler=use_crawler,
                    enable_parallel=enable_parallel
                )
                logger.info("Graph built successfully")

                # Store checkpointer reference for the graph execution
                self._checkpointer = checkpointer

                # Prepare initial state (matching browser_agent.py AgentState schema)
                # If json_filename provided, get full path
                json_path = ""
                if json_filename:
                    import os
                    data_dir = os.getenv('DATA_DIR', '.')
                    json_path = os.path.join(data_dir, json_filename)

                initial_state = {
                    # Crawler parameters
                    "start_url": url,
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
                    "qa_pairs": [],
                    "qa_generation_status": "",
                    "qa_generation_time": 0.0,
                    "status": "pending",
                    "error_message": "",
                    "total_items": 0,
                    "processed_items": [],
                    "max_items": max_items,
                    "json_path": json_path
                }

                self.progress['total'] = max_pages
                self.progress['status'] = 'running'
                yield self.progress

                # Run the graph
                # Generate thread_id if not provided
                if thread_id is None:
                    thread_id = f"generation_{datetime.now().timestamp()}"

                config = {
                    "configurable": {
                        "thread_id": thread_id
                    },
                    "recursion_limit": 1000  # Allow deep recursion for loops
                } if enable_checkpointing else {
                    "recursion_limit": 1000
                }

                # Check if resuming from checkpoint
                stream_input = initial_state
                if enable_checkpointing:
                    try:
                        existing_state = await self._graph.aget_state(config)
                        if existing_state and existing_state.values:
                            current_idx = existing_state.values.get("current_index", 0)
                            total = existing_state.values.get("total_items", 0)
                            if current_idx > 0 and current_idx < total:
                                logger.info(f"🔄 RESUMING from checkpoint: {current_idx}/{total}")
                                stream_input = None
                    except Exception as e:
                        logger.debug(f"No previous checkpoint found: {e}")

                async for event in self._graph.astream(stream_input, config):
                    if not self._is_running:
                        self.progress['status'] = 'cancelled'
                        yield self.progress
                        return

                    # Process event and update progress
                    await self._handle_event(event)

                    # Update elapsed time
                    if self._start_time:
                        elapsed = (datetime.now() - self._start_time).total_seconds()
                        self.progress['elapsed_seconds'] = int(elapsed)

                    # Yield progress
                    yield self.progress

                    # Small delay
                    await asyncio.sleep(0.1)

                # After stream ends, get final state as fallback for results
                try:
                    final_state = await self._graph.aget_state(config)
                    if final_state and final_state.values:
                        final_items = final_state.values.get("processed_items", [])
                        if final_items:
                            final_qa = sum(r.get('qa_count', 0) for r in final_items if isinstance(r, dict))
                            logger.info(f"[Final state] processed_items={len(final_items)}, total_qa={final_qa}")
                            # Update progress if stream events missed it
                            if final_qa > self.progress['qa_generated']:
                                logger.info(f"[Final state] Updating qa_generated from {self.progress['qa_generated']} to {final_qa}")
                                self.progress['qa_generated'] = final_qa
                                self.progress['topics_completed'] = len(final_items)
                            # Always save final results (ensures JSON file is written)
                            await self._save_results(final_items)
                except Exception as e:
                    logger.warning(f"[Final state] Could not read final graph state: {e}")

                # Completion
                self.progress['status'] = 'completed'
                self.progress['completed_at'] = datetime.now().isoformat()
                self._is_running = False
                yield self.progress

                logger.info(f"Browser agent completed. Generated {self.progress['qa_generated']} Q&A pairs")

        except ImportError as e:
            logger.error(f"Browser agent import failed: {e}")
            self.progress['status'] = 'error'
            self.progress['error'] = f"browser_agent.py not available: {str(e)}"
            self._is_running = False
            yield self.progress

        except Exception as e:
            logger.error(f"Browser agent error: {e}", exc_info=True)
            self.progress['status'] = 'error'
            self.progress['error'] = str(e)
            self.progress['completed_at'] = datetime.now().isoformat()
            self._is_running = False
            yield self.progress

    async def _handle_event(self, event: Dict[str, Any]):
        """Process LangGraph events and update progress"""
        for node_name, node_data in event.items():
            if node_name == "__start__":
                continue

            logger.info(f"[_handle_event] node={node_name}, keys={list(node_data.keys()) if isinstance(node_data, dict) else type(node_data).__name__}")

            # Handle crawler node (new schema: semantic_paths)
            if node_name == "crawler":
                if 'semantic_paths' in node_data:
                    self.progress['total'] = len(node_data['semantic_paths'])
                    logger.info(f"Crawler found {self.progress['total']} elements")

            # Handle load_json node
            elif node_name == "load_json":
                if 'total_items' in node_data:
                    self.progress['total'] = node_data['total_items']
                current_idx = node_data.get('current_index', 0)
                self.progress['current'] = current_idx

            # Handle browser_agent node
            elif node_name == "browser_agent":
                current_idx = node_data.get('current_index', 0)
                self.progress['current'] = current_idx

                if 'current_item' in node_data:
                    item = node_data['current_item']
                    self.progress['current_url'] = item.get('original_url', '')
                    self.progress['current_topic'] = item.get('topic', '')[:50]

            # Handle qa_generator node (sequential mode)
            elif node_name == "qa_generator":
                if 'processed_items' in node_data:
                    results = node_data['processed_items']
                    total_qa = sum(r.get('qa_count', 0) for r in results if isinstance(r, dict))
                    self.progress['qa_generated'] = total_qa
                    self.progress['topics_completed'] = len(results)

                    # Save incrementally
                    await self._save_results(results)

            # Handle parallel_batch or two_phase_batch node (parallel mode)
            elif node_name in ("parallel_batch", "two_phase_batch"):
                logger.info(f"[two_phase_batch] Event received. has processed_items={'processed_items' in node_data}, "
                           f"has total_items={'total_items' in node_data}")

                # Update total from semantic_paths if available
                if 'total_items' in node_data:
                    self.progress['total'] = node_data['total_items']

                # Update current progress
                current_idx = node_data.get('current_index', 0)
                self.progress['current'] = current_idx

                # Update processed items and Q&A count
                if 'processed_items' in node_data:
                    results = node_data['processed_items']
                    logger.info(f"[two_phase_batch] processed_items count={len(results)}, "
                               f"types={[type(r).__name__ for r in results[:3]]}, "
                               f"qa_counts={[r.get('qa_count', 'MISSING') for r in results if isinstance(r, dict)]}")
                    total_qa = sum(r.get('qa_count', 0) for r in results if isinstance(r, dict))
                    self.progress['qa_generated'] = total_qa
                    self.progress['topics_completed'] = len(results)

                    # Get current topic from last processed item
                    if results:
                        last_item = results[-1]
                        if isinstance(last_item, dict):
                            self.progress['current_url'] = last_item.get('original_url', '')
                            self.progress['current_topic'] = last_item.get('topic', '')[:50]

                    # Save incrementally
                    await self._save_results(results)

                logger.info(f"Parallel batch progress: {current_idx}/{self.progress['total']} items, {self.progress['qa_generated']} Q&A pairs")

    async def _save_results(self, results: list):
        """Save results to JSON file"""
        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress"""
        if self._start_time and self.progress['status'] == 'running':
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.progress['elapsed_seconds'] = int(elapsed)
        return self.progress.copy()

    def is_running(self) -> bool:
        """Check if running"""
        return self._is_running

    def cancel(self):
        """Cancel generation"""
        logger.info("Cancelling browser agent")
        self._is_running = False


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_runner(output_file: str = "browser_agent_test_output.json", use_mock: bool = False):
    """
    Create appropriate runner

    Args:
        output_file: Where to save generated data
        use_mock: Force use of mock runner (for testing)

    Returns:
        BrowserAgentRunner or MockBrowserAgentRunner
    """
    if use_mock:
        logger.info("Using MockBrowserAgentRunner (forced)")
        return MockBrowserAgentRunner(output_file)

    try:
        # Try to import browser_agent
        from browser_agent import build_browser_agent_graph
        logger.info("Using BrowserAgentRunner (real)")
        return BrowserAgentRunner(output_file)
    except ImportError as e:
        logger.warning(f"browser_agent.py not available: {e}, using MockBrowserAgentRunner")
        return MockBrowserAgentRunner(output_file)


# ============================================================================
# TESTING
# ============================================================================

async def test_runner():
    """Test the runner"""
    print("="*60)
    print("Testing Browser Agent Runner")
    print("="*60)

    runner = create_runner(use_mock=True)

    print(f"\nStarting mock generation for 5 pages...")

    async for progress in runner.run("https://example.com", max_pages=5):
        status = progress['status']
        current = progress['current']
        total = progress['total']
        qa = progress['qa_generated']
        elapsed = progress['elapsed_seconds']

        print(f"[{elapsed}s] {status.upper()}: {current}/{total} pages | {qa} Q&A pairs")

        if progress['status'] == 'completed':
            print("\n✅ Generation completed successfully!")
            print(f"   Total Q&A pairs: {progress['qa_generated']}")
            print(f"   Topics: {progress['topics_completed']}")
            print(f"   Time: {progress['elapsed_seconds']}s")
        elif progress['status'] == 'error':
            print(f"\n❌ Error: {progress['error']}")

    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_runner())
