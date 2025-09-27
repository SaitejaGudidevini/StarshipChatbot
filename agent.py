"""
Agentic Web Navigator - Main Agent
Combines crawling and labeling to create semantic navigation paths
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from crawler import WebCrawler, NavigationNode
from labeling import SemanticLabeler
from summarizer import SemanticSummaryService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticWebNavigator:
    """
    Main agent that orchestrates crawling and semantic labeling
    to produce navigation paths for websites
    """
    
    def __init__(
        self,
        start_url: str,
        groq_api_key: Optional[str] = None,
        max_depth: int = 3,
        max_pages: int = 100,
        headless: bool = True,
        timeout: int = 30000
    ):
        self.start_url = start_url
        
        # Initialize components
        self.crawler = WebCrawler(
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            headless=headless,
            timeout=timeout
        )
        
        self.labeler = SemanticLabeler(groq_api_key=groq_api_key)
        
        self.summarizer = SemanticSummaryService(
            groq_api_key=groq_api_key,
            model=os.getenv("SUMMARY_MODEL", "llama-3.1-8b-instant"),
            enabled=os.getenv("GENERATE_SUMMARIES", "true").lower() == "true"
        )
        
        # Results storage
        self.navigation_paths = {}
        self.metadata = {}
    
    async def _process_navigation_node(
        self,
        node: NavigationNode,
        navigation_map: Dict[str, NavigationNode]
    ) -> Dict:
        """
        Process a single navigation node to create its semantic path
        """
        # Build the label chain from root to current node
        labels_chain = []
        current = node
        
        while current and current.parent_url:
            parent = navigation_map.get(current.parent_url)
            if parent:
                # Get semantic label for the transition
                label = await self.labeler.get_semantic_label(
                    text=current.label,
                    href=current.url,
                    html_context="",  # Would be passed from crawler in full implementation
                    parent_label=parent.label if parent else None
                )
                labels_chain.append(label)
                current = parent
            else:
                break
        
        # Reverse to get path from root
        labels_chain.reverse()
        
        # Create semantic path
        semantic_path = self.labeler.create_semantic_path(labels_chain)
        
        return {
            "url": node.url,
            "semantic_path": semantic_path,
            "depth": node.depth,
            "parent_url": node.parent_url
        }
    
    async def navigate(self) -> Dict:
        """
        Main method to perform web navigation and generate semantic paths
        """
        logger.info(f"Starting Agentic Web Navigation for: {self.start_url}")
        start_time = datetime.now()
        
        try:
            # Step 1: Crawl the website
            logger.info("Step 1: Crawling website structure...")
            navigation_map = await self.crawler.crawl()
            
            # Step 2: Generate semantic paths for each URL
            logger.info("Step 2: Generating semantic paths...")
            for url, node in navigation_map.items():
                if node.depth == 0:
                    # Root node
                    self.navigation_paths[url] = {
                        "url": url,
                        "semantic_path": ["Home"],
                        "depth": 0,
                        "parent_url": None
                    }
                else:
                    # Process non-root nodes
                    path_data = await self._process_navigation_node(node, navigation_map)
                    self.navigation_paths[url] = path_data
            
            # Step 3: Generate summaries for pages with content
            logger.info("Step 3: Generating page summaries...")
            for url, node in navigation_map.items():
                if node.page_text:
                    node.summary = await self.summarizer.summarize(node.page_text, node.url)
                    # Add summary to navigation paths
                    if url in self.navigation_paths:
                        self.navigation_paths[url]["summary"] = node.summary
                    else:
                        # Root node case
                        self.navigation_paths[url]["summary"] = node.summary
            
            # Step 4: Compile metadata
            end_time = datetime.now()
            self.metadata = {
                "start_url": self.start_url,
                "domain": self.crawler.domain,
                "total_urls_discovered": len(self.navigation_paths),
                "max_depth_reached": max((p["depth"] for p in self.navigation_paths.values()), default=0),
                "crawl_duration_seconds": (end_time - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "crawler_stats": self.crawler.get_statistics()
            }
            
            logger.info(f"Navigation complete. Discovered {len(self.navigation_paths)} URLs")
            
        except Exception as e:
            logger.error(f"Error during navigation: {e}")
            raise
        
        return {
            "metadata": self.metadata,
            "navigation_paths": self.navigation_paths
        }
    
    def save_results(self, output_dir: str = "output"):
        """
        Save navigation results to JSON file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_clean = self.crawler.domain.replace(".", "_")
        filename = f"{output_dir}/navigation_{domain_clean}_{timestamp}.json"
        
        # Prepare output data
        output_data = {
            "metadata": self.metadata,
            "navigation_paths": self.navigation_paths
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {filename}")
        return filename
    
    def get_example_output(self, limit: int = 5) -> str:
        """
        Get example output for demonstration
        """
        examples = []
        
        for url, path_data in list(self.navigation_paths.items())[:limit]:
            examples.append({
                "url": url,
                "semantic_path": " → ".join(path_data["semantic_path"]) if path_data["semantic_path"] else "Home",
                "depth": path_data["depth"]
            })
        
        return json.dumps(examples, indent=2)


async def main():
    """
    Main entry point for the Agentic Web Navigator
    """
    # Configuration from environment or defaults
    start_url = os.getenv("START_URL", "https://www.mhsindiana.com")
    groq_api_key = os.getenv("GROQ_API_KEY")
    max_depth = int(os.getenv("MAX_DEPTH", 3))
    max_pages = int(os.getenv("MAX_PAGES", 100))
    headless = os.getenv("HEADLESS", "true").lower() == "true"
    
    # Create and run the agent
    agent = AgenticWebNavigator(
        start_url=start_url,
        groq_api_key=groq_api_key,
        max_depth=max_depth,
        max_pages=max_pages,
        headless=headless
    )
    
    # Navigate and generate semantic paths
    results = await agent.navigate()
    
    # Save results
    output_file = agent.save_results()
    
    # Display example output
    print("\n" + "="*50)
    print("NAVIGATION COMPLETE")
    print("="*50)
    print(f"\nMetadata:")
    print(json.dumps(results["metadata"], indent=2))
    print(f"\nExample Navigation Paths:")
    print(agent.get_example_output(limit=10))
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())