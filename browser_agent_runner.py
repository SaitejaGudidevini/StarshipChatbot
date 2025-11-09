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
        use_crawler: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run browser agent pipeline

        Args:
            url: Starting URL to scrape
            max_pages: Maximum number of pages to process
            use_crawler: Whether to use hierarchical crawler

        Yields:
            Progress updates as processing occurs
        """
        self._is_running = True

        try:
            # Import browser_agent components
            from WorkingFiles.browser_agent import build_graph
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self.progress['status'] = 'initializing'
            self.progress['started_at'] = datetime.now().isoformat()
            self._start_time = datetime.now()
            yield self.progress

            logger.info(f"Starting browser agent for URL: {url}")

            # Build graph with checkpointing
            self.progress['status'] = 'building_graph'
            yield self.progress

            checkpointer = await AsyncSqliteSaver.from_conn_string(
                "browser_agent_checkpoints.db"
            )

            self._graph = await build_graph(checkpointer)
            logger.info("Graph built successfully")

            # Prepare initial state
            initial_state = {
                "start_url": url,
                "max_depth": 2,
                "max_pages": max_pages,
                "use_crawler": use_crawler,
                "semantic_elements": [] if use_crawler else [
                    {"text": url, "url": url, "semantic_path": url, "type": "manual"}
                ],
                "current_index": 0,
                "results": [],
                "errors": []
            }

            self.progress['total'] = max_pages
            self.progress['status'] = 'running'
            yield self.progress

            # Run the graph
            config = {
                "configurable": {
                    "thread_id": f"generation_{datetime.now().timestamp()}"
                }
            }

            async for event in self._graph.astream(initial_state, config):
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

            logger.debug(f"Processing node: {node_name}")

            if node_name == "crawler" and 'semantic_elements' in node_data:
                self.progress['total'] = len(node_data['semantic_elements'])
                logger.info(f"Crawler found {self.progress['total']} elements")

            elif node_name == "load_json":
                current_idx = node_data.get('current_index', 0)
                self.progress['current'] = current_idx

                if 'semantic_elements' in node_data and current_idx < len(node_data['semantic_elements']):
                    element = node_data['semantic_elements'][current_idx]
                    self.progress['current_url'] = element.get('url', '')
                    self.progress['current_topic'] = element.get('text', '')[:50]

            elif node_name == "qa_generator":
                if 'results' in node_data:
                    results = node_data['results']
                    total_qa = sum(len(r.get('qa_pairs', [])) for r in results if isinstance(r, dict))
                    self.progress['qa_generated'] = total_qa
                    self.progress['topics_completed'] = len(results)

                    # Save incrementally
                    await self._save_results(results)

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
        from WorkingFiles.browser_agent import build_graph
        logger.info("Using BrowserAgentRunner (real)")
        return BrowserAgentRunner(output_file)
    except ImportError:
        logger.warning("browser_agent.py not available, using MockBrowserAgentRunner")
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
