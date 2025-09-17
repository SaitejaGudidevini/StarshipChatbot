"""
Agentic Web Navigator - Crawler Module
Handles browser automation and page navigation using Playwright
"""

import asyncio
import json
import logging
from typing import Dict, Set, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass, field
from collections import deque

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NavigationNode:
    """Represents a navigation point in the website structure"""
    url: str
    semantic_path: List[str]
    depth: int
    parent_url: Optional[str] = None
    label: str = ""
    visited: bool = False


class WebCrawler:
    """
    Robust web crawler using Playwright with isolated browser contexts
    """
    
    def __init__(
        self,
        start_url: str,
        max_depth: int = 3,
        max_pages: int = 100,
        headless: bool = True,
        timeout: int = 30000
    ):
        self.start_url = self._normalize_url(start_url)
        self.domain = urlparse(self.start_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.headless = headless
        self.timeout = timeout
        
        # Tracking structures
        self.visited_urls: Set[str] = set()
        self.url_queue: deque = deque()
        self.navigation_map: Dict[str, NavigationNode] = {}
        
        # Browser instances
        self.playwright = None
        self.browser: Optional[Browser] = None
        
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent comparison"""
        parsed = urlparse(url)
        
        # Remove fragment
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/') if parsed.path != '/' else '/',
            parsed.params,
            parsed.query,
            ''
        ))
        
        return normalized
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        return urlparse(url).netloc == self.domain
    
    async def initialize(self):
        """Initialize Playwright and browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        logger.info(f"Browser initialized for domain: {self.domain}")
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser resources cleaned up")
    
    async def _create_isolated_context(self) -> BrowserContext:
        """Create a new isolated browser context"""
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        return context
    
    async def _extract_links(self, page: Page, current_url: str) -> List[Tuple[str, str, str]]:
        """
        Extract all navigational links from the page
        Returns: List of (url, text, html_context) tuples
        """
        links = []
        
        try:
            # Wait for the page to be fully loaded
            await page.wait_for_load_state('networkidle', timeout=self.timeout)
            
            # Extract all anchor tags
            link_elements = await page.query_selector_all('a[href]')
            
            for element in link_elements:
                try:
                    href = await element.get_attribute('href')
                    if not href or href.startswith('#') or href.startswith('javascript:'):
                        continue
                    
                    # Get absolute URL
                    absolute_url = urljoin(current_url, href)
                    normalized_url = self._normalize_url(absolute_url)
                    
                    # Check if URL is within the same domain
                    if not self._is_same_domain(normalized_url):
                        continue
                    
                    # Extract link context for labeling
                    text = await element.inner_text() or ""
                    
                    # Get surrounding HTML for context
                    html_context = await element.evaluate('''
                        (element) => {
                            const parent = element.parentElement;
                            return parent ? parent.outerHTML.substring(0, 500) : element.outerHTML;
                        }
                    ''')
                    
                    links.append((normalized_url, text.strip(), html_context))
                    
                except Exception as e:
                    logger.debug(f"Error extracting link: {e}")
                    continue
            
        except PlaywrightTimeout:
            logger.warning(f"Timeout while extracting links from {current_url}")
        except Exception as e:
            logger.error(f"Error extracting links from {current_url}: {e}")
        
        return links
    
    async def _crawl_page(self, node: NavigationNode) -> List[NavigationNode]:
        """
        Crawl a single page in an isolated context
        Returns list of discovered child nodes
        """
        if node.url in self.visited_urls or len(self.visited_urls) >= self.max_pages:
            return []
        
        self.visited_urls.add(node.url)
        child_nodes = []
        
        # Create isolated context for this page
        context = await self._create_isolated_context()
        page = await context.new_page()
        
        try:
            logger.info(f"Crawling: {node.url} (depth: {node.depth})")
            
            # Navigate to the page
            response = await page.goto(
                node.url,
                wait_until='networkidle',
                timeout=self.timeout
            )
            
            if not response or response.status >= 400:
                logger.warning(f"Failed to load {node.url}: Status {response.status if response else 'None'}")
                return []
            
            # Extract links if we haven't reached max depth
            if node.depth < self.max_depth:
                links = await self._extract_links(page, node.url)
                
                for url, text, html_context in links:
                    if url not in self.visited_urls:
                        # Create child node (label will be determined by the labeling system)
                        child_node = NavigationNode(
                            url=url,
                            semantic_path=node.semantic_path.copy(),
                            depth=node.depth + 1,
                            parent_url=node.url,
                            label=text  # Temporary label, will be refined
                        )
                        child_nodes.append(child_node)
            
        except Exception as e:
            logger.error(f"Error crawling {node.url}: {e}")
        
        finally:
            await page.close()
            await context.close()
        
        return child_nodes
    
    async def crawl(self) -> Dict[str, NavigationNode]:
        """
        Main crawling method - iteratively crawls the website
        Returns the complete navigation map
        """
        await self.initialize()
        
        try:
            # Create root node
            root_node = NavigationNode(
                url=self.start_url,
                semantic_path=[],
                depth=0,
                parent_url=None,
                label="Home"
            )
            
            self.url_queue.append(root_node)
            self.navigation_map[self.start_url] = root_node
            
            # Iterative crawling
            while self.url_queue and len(self.visited_urls) < self.max_pages:
                current_node = self.url_queue.popleft()
                
                if current_node.visited:
                    continue
                
                current_node.visited = True
                
                # Crawl the page and get child nodes
                child_nodes = await self._crawl_page(current_node)
                
                # Add child nodes to queue and navigation map
                for child in child_nodes:
                    if child.url not in self.navigation_map:
                        self.navigation_map[child.url] = child
                        self.url_queue.append(child)
                
                # Small delay to be respectful to the server
                await asyncio.sleep(0.5)
            
            logger.info(f"Crawling complete. Visited {len(self.visited_urls)} pages")
            
        finally:
            await self.cleanup()
        
        return self.navigation_map
    
    def get_statistics(self) -> Dict:
        """Get crawling statistics"""
        return {
            "total_pages_visited": len(self.visited_urls),
            "total_pages_discovered": len(self.navigation_map),
            "max_depth_reached": max((node.depth for node in self.navigation_map.values()), default=0),
            "domain": self.domain
        }


async def main():
    """Example usage"""
    crawler = WebCrawler(
        start_url="https://www.example.com",
        max_depth=2,
        max_pages=50,
        headless=True
    )
    
    navigation_map = await crawler.crawl()
    stats = crawler.get_statistics()
    
    print(f"\nCrawling Statistics:")
    print(json.dumps(stats, indent=2))
    
    print(f"\nDiscovered {len(navigation_map)} URLs")
    for url, node in list(navigation_map.items())[:5]:
        print(f"  - {url} (depth: {node.depth})")


if __name__ == "__main__":
    asyncio.run(main())