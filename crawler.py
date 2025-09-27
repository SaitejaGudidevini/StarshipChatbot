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
    page_text: str = ""
    summary: str = ""
    # NEW: Enhanced content capture fields
    main_content: str = ""  # Clean main content for summarization
    content_metadata: Dict = field(default_factory=dict)  # Content analysis metadata
    # NEW: Element identification data
    page_elements: Dict = field(default_factory=dict)  # All identified elements by type
    element_count: int = 0  # Total number of elements found


class WebCrawler:
    """
    Robust web crawler using Playwright with isolated browser contexts
    Enhanced with element identification capabilities
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
        
        # Element identification configuration
        self.element_selectors = {
            'headings': 'h1, h2, h3, h4, h5, h6',
            'links': 'a[href]',
            'buttons': 'button, input[type="button"], input[type="submit"]',
            'forms': 'form',
            'navigation': 'nav, .nav, .navigation, .menu',
            'content': 'article, .content, main, .main-content',
            'lists': 'ul, ol',
            'images': 'img[alt]',
            'inputs': 'input, textarea, select',
            'sections': 'section, div.section, .container'
        }
        
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
    
    async def _extract_main_content(self, page: Page) -> Tuple[str, Dict]:
        """
        Extract clean main content from page, removing navigation, ads, etc.
        Returns: (main_content_text, metadata_dict)
        """
        try:
            # Strategy 1: Look for main content containers
            main_selectors = [
                'main', 'article', '.content', '#content', '.main-content', 
                '.page-content', '.entry-content', '.post-content'
            ]
            
            main_content = ""
            extraction_method = "unknown"
            
            # Try each selector until we find substantial content
            for selector in main_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        content_parts = []
                        for element in elements:
                            text = await element.inner_text()
                            if text and len(text.strip()) > 100:  # Only substantial content
                                content_parts.append(text.strip())
                        
                        if content_parts:
                            main_content = "\n\n".join(content_parts)
                            extraction_method = f"selector_{selector}"
                            break
                except Exception:
                    continue
            
            # Fallback: Extract from body but remove nav/footer/sidebar
            if not main_content or len(main_content) < 200:
                try:
                    # Get all text but exclude navigation areas
                    exclude_selectors = [
                        'nav', 'header', 'footer', '.navigation', '.nav', 
                        '.sidebar', '.menu', '.breadcrumb', '.breadcrumbs',
                        '.ads', '.advertisement', '.social', '.share'
                    ]
                    
                    # Remove unwanted elements
                    for selector in exclude_selectors:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            try:
                                await element.evaluate('element => element.remove()')
                            except Exception:
                                pass
                    
                    # Get remaining content
                    body_text = await page.inner_text('body')
                    if len(body_text.strip()) > len(main_content.strip()):
                        main_content = body_text.strip()
                        extraction_method = "body_filtered"
                        
                except Exception:
                    # Final fallback
                    main_content = await page.inner_text('body')
                    extraction_method = "body_raw"
            
            # Clean up content
            main_content = self._clean_content(main_content)
            
            # Generate metadata
            metadata = {
                "extraction_method": extraction_method,
                "word_count": len(main_content.split()),
                "char_count": len(main_content),
                "estimated_read_time_minutes": max(1, len(main_content.split()) // 200)
            }
            
            return main_content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting main content: {e}")
            # Ultimate fallback
            try:
                fallback_content = await page.inner_text('body')
                return self._clean_content(fallback_content), {
                    "extraction_method": "error_fallback",
                    "word_count": len(fallback_content.split())
                }
            except Exception:
                return "", {"extraction_method": "failed", "word_count": 0}
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text"""
        if not content:
            return ""
        
        import re
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'Skip to content|Skip to main content|Skip navigation', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Share this page|Print this page|Email this page', '', content, flags=re.IGNORECASE)
        content = re.sub(r'JavaScript must be enabled|Enable JavaScript', '', content, flags=re.IGNORECASE)
        
        # Clean up formatting
        content = content.strip()
        
        # Limit length for summarization (keep most important content)
        if len(content) > 4000:
            # Keep first 4000 chars but try to end at sentence boundary
            truncated = content[:4000]
            last_period = truncated.rfind('.')
            if last_period > 3000:  # Only truncate at sentence if it's not too early
                content = truncated[:last_period + 1]
            else:
                content = truncated
        
        return content
    
    async def _identify_page_elements(self, page: Page, url: str) -> Tuple[Dict, int]:
        """
        Identify all elements and their text content on the page
        Returns: (elements_by_type_dict, total_element_count)
        """
        logger.info(f"🔍 Identifying elements on: {url}")
        
        page_elements = {}
        total_elements = 0
        
        # Extract each type of element
        for element_type, selector in self.element_selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                element_data = []
                
                for i, element in enumerate(elements):
                    try:
                        # Get element info
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        text_content = await element.inner_text()
                        
                        # Get additional attributes based on element type
                        extra_info = {}
                        if element_type == 'links':
                            href = await element.get_attribute('href')
                            extra_info['href'] = href
                        elif element_type == 'images':
                            alt = await element.get_attribute('alt')
                            src = await element.get_attribute('src')
                            extra_info.update({'alt': alt, 'src': src})
                        elif element_type == 'buttons':
                            button_type = await element.get_attribute('type')
                            extra_info['type'] = button_type
                        elif element_type == 'inputs':
                            input_type = await element.get_attribute('type')
                            placeholder = await element.get_attribute('placeholder')
                            extra_info.update({'type': input_type, 'placeholder': placeholder})
                        elif element_type == 'forms':
                            action = await element.get_attribute('action')
                            method = await element.get_attribute('method')
                            extra_info.update({'action': action, 'method': method})
                        
                        element_info = {
                            'index': i,
                            'tag': tag_name,
                            'text': text_content.strip()[:300] if text_content else '',  # Limit text length
                            **extra_info
                        }
                        
                        element_data.append(element_info)
                        
                    except Exception as e:
                        logger.debug(f"Error extracting {element_type} element {i}: {e}")
                
                page_elements[element_type] = element_data
                total_elements += len(element_data)
                
                # Log summary for this element type
                if element_data:
                    logger.info(f"📋 Found {len(element_data)} {element_type}")
                    
            except Exception as e:
                logger.debug(f"Error finding {element_type}: {e}")
                page_elements[element_type] = []
        
        logger.info(f"✅ Total elements identified: {total_elements}")
        return page_elements, total_elements
    
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
            
            # Capture page text content
            try:
                node.page_text = await page.inner_text("body")
                
                # DEBUG: Print captured content
                print(f"\n{'='*80}")
                print(f"🔍 CAPTURED CONTENT FROM: {node.url}")
                print(f"{'='*80}")
                print(f"📄 Content length: {len(node.page_text)} characters")
                print(f"📝 First 500 characters:")
                print(f"'{node.page_text[:500]}...'")
                print(f"{'='*80}\n")
                
            except Exception as e:
                logger.debug(f"Failed to extract page text from {node.url}: {e}")
                node.page_text = ""
            
            # NEW: Capture clean main content for summarization
            try:
                main_content, content_meta = await self._extract_main_content(page)
                node.main_content = main_content
                # Add this:
                print(f"🎯 CLEANED MAIN CONTENT ({len(main_content)} chars):")
                print(f"'{main_content[:300]}...'")
                node.content_metadata = content_meta
                print(f" Extraction Metadata: {content_meta}")
                logger.debug(f"Captured {len(main_content)} chars of main content from {node.url}")
            except Exception as e:
                logger.debug(f"Failed to extract main content from {node.url}: {e}")
                node.main_content = node.page_text  # Fallback to full page text
                node.content_metadata = {"extraction_method": "fallback", "word_count": len(node.page_text.split())}
            
            # NEW: Identify all elements on the page
            try:
                page_elements, element_count = await self._identify_page_elements(page, node.url)
                node.page_elements = page_elements
                node.element_count = element_count
                print(f"🔍 IDENTIFIED ELEMENTS: {element_count} total elements")
                print(f"📊 Element breakdown: {', '.join([f'{k}: {len(v)}' for k, v in page_elements.items() if v])}")
                logger.debug(f"Identified {element_count} elements from {node.url}")
            except Exception as e:
                logger.debug(f"Failed to identify elements from {node.url}: {e}")
                node.page_elements = {}
                node.element_count = 0
            
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
        """Get crawling statistics including element analysis"""
        total_elements = sum(node.element_count for node in self.navigation_map.values())
        
        # Element type breakdown
        element_counts = {}
        for node in self.navigation_map.values():
            for element_type, elements in node.page_elements.items():
                element_counts[element_type] = element_counts.get(element_type, 0) + len(elements)
        
        return {
            "total_pages_visited": len(self.visited_urls),
            "total_pages_discovered": len(self.navigation_map),
            "max_depth_reached": max((node.depth for node in self.navigation_map.values()), default=0),
            "domain": self.domain,
            "total_elements_identified": total_elements,
            "elements_by_type": element_counts,
            "avg_elements_per_page": total_elements / len(self.navigation_map) if self.navigation_map else 0
        }


def save_crawl_results_to_json(navigation_map: Dict[str, NavigationNode], stats: Dict, filename: str = None):
    """
    Save complete crawl results including element data to a JSON file
    """
    if filename is None:
        from datetime import datetime
        domain = stats['domain'].replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_training_{domain}_{timestamp}.json"
    
    # Convert navigation map to serializable format
    crawl_data = {
        "crawl_metadata": {
            "timestamp": stats.get('timestamp', ''),
            "domain": stats['domain'],
            "total_pages": stats['total_pages_visited'],
            "total_elements": stats.get('total_elements_identified', 0),
            "elements_by_type": stats.get('elements_by_type', {}),
            "avg_elements_per_page": stats.get('avg_elements_per_page', 0)
        },
        "pages": {}
    }
    
    # Add page data with elements
    for url, node in navigation_map.items():
        crawl_data["pages"][url] = {
            "url": node.url,
            "depth": node.depth,
            "parent_url": node.parent_url,
            "label": node.label,
            "page_text": node.page_text,
            "main_content": node.main_content,
            "content_metadata": node.content_metadata,
            "page_elements": node.page_elements,
            "element_count": node.element_count,
            "semantic_path": node.semantic_path
        }
    
    # Save to file
    output_dir = "output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(crawl_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Complete crawl data with elements saved to: {filepath}")
    print(f"📊 Saved {len(navigation_map)} pages with {stats.get('total_elements_identified', 0)} total elements")
    return filepath


async def main():
    """Example usage with JSON file saving"""
    crawler = WebCrawler(
        start_url="https://www.mhsindiana.com/",
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
    
    # Save results to JSON file
    saved_file = save_crawl_results_to_json(navigation_map, stats)
    
    # Example of loading the saved data
    print(f"\nExample: Loading saved data from {saved_file}")
    with open(saved_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    print(f"Loaded {len(loaded_data['navigation_map'])} URLs from file")
    print(f"Crawl timestamp: {loaded_data['crawl_metadata']['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())