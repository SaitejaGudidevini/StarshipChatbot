"""
Hierarchical Web Crawler - Architecture Implementation
Follows heading-first, then links priority system with semantic path generation

Architecture Flow:
1. Start at homepage -> Extract ALL headings first
2. Create semantic paths: https://domain/heading_text (single slash)
3. Visit each heading page -> Extract headings, then links
4. Create link semantic paths: https://domain//link_text (double slash)
5. Depth-first traversal with backtracking
6. Return to homepage, process next main link branch
"""

import asyncio
import json
import logging
from typing import Dict, Set, List, Optional, Tuple, Deque
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeout
from labeling import SemanticLabeler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlNode:
    """Represents a discovered page with its semantic information"""
    original_url: str
    semantic_path: str
    source_type: str  # 'heading' or 'link'
    parent_url: Optional[str] = None
    depth: int = 0
    visited: bool = False
    page_content: str = ""
    page_elements: Dict = field(default_factory=dict)
    element_count: int = 0
    discovered_headings: List[str] = field(default_factory=list)
    discovered_links: List[str] = field(default_factory=list)


class HierarchicalWebCrawler:
    """
    Advanced web crawler that processes headings before links
    and maintains hierarchical navigation structure
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
        
        # Hierarchical tracking structures
        self.heading_queue: Deque[CrawlNode] = deque()
        self.link_queue: Deque[CrawlNode] = deque()
        self.processed_urls: Set[str] = set()
        self.crawl_nodes: Dict[str, CrawlNode] = {}
        self.navigation_tree: Dict[str, List[str]] = {}
        
        # Browser instances
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.labeler = SemanticLabeler()
        
        # Statistics
        self.stats = {
            "total_headings_found": 0,
            "total_links_found": 0,
            "heading_pages_crawled": 0,
            "link_pages_crawled": 0,
            "backtrack_count": 0
        }
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent comparison"""
        parsed = urlparse(url)
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
        logger.info(f"🚀 Hierarchical crawler initialized for: {self.domain}")
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("🧹 Browser resources cleaned up")
    
    async def _create_isolated_context(self) -> BrowserContext:
        """Create a new isolated browser context"""
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        return context
    
    async def _extract_page_data(self, page: Page, url: str) -> Tuple[Dict, str, int]:
        """
        Extract comprehensive page data including elements and content
        Returns: (page_elements, main_content, element_count)
        """
        try:
            # Wait for page to be fully loaded
            await page.wait_for_load_state('networkidle', timeout=self.timeout)
            
            # FIRST: Remove header and footer elements before ANY extraction
            await page.evaluate('''
                () => {
                    // Remove common header selectors
                    const headerSelectors = ['header', '.header', '#header', 'nav', '.nav', '.navigation', '.navbar', '.topbar', '.top-bar'];
                    headerSelectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => el.remove());
                    });
                    
                    // Remove common footer selectors + MHS-specific selectors
                    const footerSelectors = [
                        'footer', '.footer', '#footer', '.bottom', '.copyright', '.site-footer',
                        '.footer-nav', '.global-footer', '.pdffooter', '.pdffooter-desktop', 
                        '.pdffooter-mobile', '.socialintegration', '.footerlinlist-wrapper'
                    ];
                    footerSelectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => el.remove());
                    });
                }
            ''')
            
            # THEN: Extract page elements with their associated content from cleaned DOM
            page_elements = {}
            
            # Define the main element types we want to focus on
            main_selectors = {
                'headings': 'h1, h2, h3, h4, h5, h6',
                'links': 'a[href]',
                'buttons': 'button, input[type="button"], input[type="submit"]'
            }
            
            total_elements = 0
            for element_type, selector in main_selectors.items():
                try:
                    elements = await page.query_selector_all(selector)
                    element_data = []
                    
                    for i, element in enumerate(elements):
                        try:
                            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                            text_content = await element.inner_text()
                            
                            # Get following content for this element
                            following_content = await element.evaluate('''
                                (el) => {
                                    let content = '';
                                    let current = el.nextElementSibling;
                                    let maxElements = 3; // Look at next 3 elements max
                                    let elementCount = 0;
                                    
                                    while (current && elementCount < maxElements) {
                                        const tagName = current.tagName.toLowerCase();
                                        
                                        // Stop if we hit another heading, link, or button
                                        if (tagName.match(/^h[1-6]$/) || tagName === 'a' || tagName === 'button') {
                                            break;
                                        }
                                        
                                        // Collect content from paragraphs, divs, spans
                                        if (tagName === 'p' || tagName === 'div' || tagName === 'span') {
                                            const text = current.innerText || current.textContent || '';
                                            if (text.trim().length > 10) {
                                                content += text.trim() + ' ';
                                            }
                                        }
                                        
                                        current = current.nextElementSibling;
                                        elementCount++;
                                    }
                                    
                                    return content.trim();
                                }
                            ''')
                            
                            # Get additional attributes based on element type
                            extra_info = {}
                            if element_type == 'links':
                                href = await element.get_attribute('href')
                                extra_info['href'] = href
                            elif element_type == 'buttons':
                                button_type = await element.get_attribute('type')
                                extra_info['type'] = button_type
                            
                            element_info = {
                                'text': text_content.strip() if text_content else '',
                                'content': following_content[:500] if following_content else '',  # Limit content length
                                **extra_info
                            }
                            
                            # Only include elements that have meaningful text
                            if element_info['text'] and len(element_info['text']) > 2:
                                element_data.append(element_info)
                            
                        except Exception:
                            continue
                    
                    page_elements[element_type] = element_data
                    total_elements += len(element_data)
                    
                except Exception:
                    page_elements[element_type] = []
            
            # Extract main content (headers/footers already removed)
            try:
                main_content = await page.inner_text("body")
                # Clean the content
                main_content = self._clean_content(main_content)
            except Exception:
                main_content = ""
            
            return page_elements, main_content, total_elements
            
        except Exception as e:
            logger.error(f"Error extracting page data from {url}: {e}")
            return {}, "", 0
    
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
        
        # Limit length for summarization
        if len(content) > 4000:
            truncated = content[:4000]
            last_period = truncated.rfind('.')
            if last_period > 3000:
                content = truncated[:last_period + 1]
            else:
                content = truncated
        
        return content
    
    async def _discover_headings_and_links(self, page: Page, current_url: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Discover headings and links from current page
        Returns: ([(heading_text, semantic_path), ...], [(link_text, link_url, semantic_path), ...])
        """
        discovered_headings = []
        discovered_links = []
        
        try:
            # Extract headings first (priority 1)
            heading_elements = await page.query_selector_all('h1, h2, h3, h4, h5, h6')
            for element in heading_elements:
                try:
                    heading_text = await element.inner_text()
                    if heading_text and len(heading_text.strip()) > 2:
                        #check: Does this heading have paragraphs after it?
                        has_paragraphs = await element.evaluate('''
                        (heading)=> {
                            let nextElement = heading.nextElementSibling;
                            let foundParagraph = false;
                            
                            // Look for <p> tags in next 5 siblings
                            for (let i = 0; i < 5 && nextElement; i++) {
                                if (nextElement.tagName === 'P') {
                                    foundParagraph = true;
                                    break;
                                }
                                nextElement = nextElement.nextElementSibling;
                            }
                            return foundParagraph;
                        }
                    ''')
                    #Only process heading that have paragraphs
                    if has_paragraphs:
                        
                        # Create semantic path for heading (single slash)
                        clean_text = self.labeler._clean_text_natural(heading_text.strip())
                        semantic_path = f"https://{self.domain}/{clean_text}"
                        discovered_headings.append((heading_text.strip(), semantic_path))
                except Exception:
                    continue
            
            # Extract meaningful links (priority 2)
            link_elements = await page.query_selector_all('a[href]')
            for element in link_elements:
                try:
                    href = await element.get_attribute('href')
                    link_text = await element.inner_text()
                    
                    if not href or href.startswith('#') or href.startswith('javascript:'):
                        continue
                    
                    # Get absolute URL
                    absolute_url = urljoin(current_url, href)
                    normalized_url = self._normalize_url(absolute_url)
                    
                    # Check if URL is within the same domain
                    if not self._is_same_domain(normalized_url):
                        continue
                    
                    # Skip if already processed
                    if normalized_url in self.processed_urls:
                        continue
                    
                    # Create semantic path for link (double slash)
                    if link_text and len(link_text.strip()) > 2:
                        clean_text = self.labeler._clean_text_natural(link_text.strip())
                        semantic_path = f"https://{self.domain}//{clean_text}"
                        discovered_links.append((link_text.strip(), normalized_url, semantic_path))
                    
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"Error discovering headings/links from {current_url}: {e}")
        
        return discovered_headings, discovered_links
    
    async def _crawl_single_page(self, node: CrawlNode) -> bool:
        """
        Crawl a single page and update the node with discovered content
        Returns True if successful, False otherwise
        """
        if node.original_url in self.processed_urls or len(self.processed_urls) >= self.max_pages:
            return False
        
        self.processed_urls.add(node.original_url)
        
        # Create isolated context for this page
        context = await self._create_isolated_context()
        page = await context.new_page()
        
        try:
            logger.info(f"🔍 [{node.source_type.upper()}] Crawling: {node.semantic_path}")
            logger.info(f"    📍 Original: {node.original_url}")
            
            # Navigate to the page
            response = await page.goto(
                node.original_url,
                wait_until='networkidle',
                timeout=self.timeout
            )
            
            if not response or response.status >= 400:
                logger.warning(f"❌ Failed to load {node.original_url}: Status {response.status if response else 'None'}")
                return False
            
            # Extract comprehensive page data
            page_elements, main_content, element_count = await self._extract_page_data(page, node.original_url)
            
            # Update node with extracted data
            node.page_elements = page_elements
            node.page_content = main_content
            node.element_count = element_count
            node.visited = True
            
            # Discover headings and links for further crawling
            discovered_headings, discovered_links = await self._discover_headings_and_links(page, node.original_url)
            
            # Add discovered headings to heading queue (if within depth limit)
            if node.depth < self.max_depth:
                for heading_text, semantic_path in discovered_headings:
                    if semantic_path not in self.crawl_nodes:
                        heading_node = CrawlNode(
                            original_url=node.original_url,  # Headings reference same page
                            semantic_path=semantic_path,
                            source_type='heading',
                            parent_url=node.original_url,
                            depth=node.depth + 1
                        )
                        self.crawl_nodes[semantic_path] = heading_node
                        self.heading_queue.append(heading_node)
                        self.stats["total_headings_found"] += 1
                
                # Add discovered links to link queue
                for link_text, link_url, semantic_path in discovered_links:
                    if semantic_path not in self.crawl_nodes:
                        link_node = CrawlNode(
                            original_url=link_url,
                            semantic_path=semantic_path,
                            source_type='link',
                            parent_url=node.original_url,
                            depth=node.depth + 1
                        )
                        self.crawl_nodes[semantic_path] = link_node
                        self.link_queue.append(link_node)
                        self.stats["total_links_found"] += 1
            
            # Update statistics
            if node.source_type == 'heading':
                self.stats["heading_pages_crawled"] += 1
            else:
                self.stats["link_pages_crawled"] += 1
            
            logger.info(f"✅ Discovered: {len(discovered_headings)} headings, {len(discovered_links)} links")
            
            # Small delay to be respectful to the server
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crawling {node.original_url}: {e}")
            return False
        
        finally:
            await page.close()
            await context.close()
    
    async def crawl_hierarchical(self) -> Dict[str, CrawlNode]:
        """
        Main hierarchical crawling method following the architecture:
        1. Start at homepage
        2. Extract headings first, then links
        3. Process all headings before links (depth-first)
        4. Backtrack and continue with links
        """
        await self.initialize()
        
        try:
            logger.info(f"🎯 Starting hierarchical crawl of: {self.start_url}")
            logger.info(f"📊 Max depth: {self.max_depth}, Max pages: {self.max_pages}")
            
            # Create root node for homepage
            root_semantic = f"https://{self.domain}/Home"
            root_node = CrawlNode(
                original_url=self.start_url,
                semantic_path=root_semantic,
                source_type='homepage',
                depth=0
            )
            
            self.crawl_nodes[root_semantic] = root_node
            
            # Phase 1: Crawl homepage and discover initial headings/links
            logger.info("\n🚀 PHASE 1: Homepage Discovery")
            success = await self._crawl_single_page(root_node)
            
            if not success:
                logger.error("❌ Failed to crawl homepage. Aborting.")
                return self.crawl_nodes
            
            # Phase 2: Process all headings first (depth-first)
            logger.info(f"\n📰 PHASE 2: Processing {len(self.heading_queue)} Headings (Priority)")
            while self.heading_queue and len(self.processed_urls) < self.max_pages:
                heading_node = self.heading_queue.popleft()
                if not heading_node.visited:
                    await self._crawl_single_page(heading_node)
            
            # Phase 3: Process all links (depth-first)
            logger.info(f"\n🔗 PHASE 3: Processing {len(self.link_queue)} Links")
            while self.link_queue and len(self.processed_urls) < self.max_pages:
                link_node = self.link_queue.popleft()
                if not link_node.visited:
                    await self._crawl_single_page(link_node)
            
            logger.info(f"\n✅ Hierarchical crawling complete!")
            self._print_crawl_summary()
            
        finally:
            await self.cleanup()
        
        return self.crawl_nodes
    
    def _print_crawl_summary(self):
        """Print comprehensive crawl statistics"""
        logger.info("📊 HIERARCHICAL CRAWL SUMMARY:")
        logger.info(f"   🌐 Domain: {self.domain}")
        logger.info(f"   📄 Total pages processed: {len(self.processed_urls)}")
        logger.info(f"   📰 Heading-based entries: {self.stats['heading_pages_crawled']}")
        logger.info(f"   🔗 Link-based entries: {self.stats['link_pages_crawled']}")
        logger.info(f"   🔍 Total headings discovered: {self.stats['total_headings_found']}")
        logger.info(f"   🔗 Total links discovered: {self.stats['total_links_found']}")
    
    def _clean_text_for_path(self, text: str) -> str:
        """Clean text for use in semantic path"""
        import re
        # Remove special characters but keep spaces and basic punctuation
        cleaned = re.sub(r'[^\w\s\-\.\,\(\)]', '', text)
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def save_hierarchical_results(self, filename: str = None) -> str:
        """Save hierarchical crawl results to JSON file with individual element entries"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain_clean = self.domain.replace('.', '_')
            filename = f"output/hierarchical_crawl_{domain_clean}_{timestamp}.json"
        
        # Convert crawl nodes to flattened element format
        crawl_data = {
            "crawl_metadata": {
                "domain": self.domain,
                "start_url": self.start_url,
                "crawl_type": "hierarchical_flattened",
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "total_nodes": len(self.crawl_nodes),
                "format_description": "Each element (heading/link/button) has its own semantic path entry"
            },
            "semantic_elements": {}
        }
        
        total_elements = 0
        
        # Process each crawl node and create individual entries for each element
        for semantic_path, node in self.crawl_nodes.items():
            if not node.visited or not node.page_elements:
                continue
            
            # Create entries for each element type
            for element_type in ['headings', 'links', 'buttons']:
                elements = node.page_elements.get(element_type, [])
                
                for element in elements:
                    if element.get('text'):
                        # Create semantic path for this specific element
                        element_text = element['text'].strip()
                        # Clean element text for use in path
                        clean_text = self._clean_text_for_path(element_text)
                        
                        element_semantic_path = f"{semantic_path}/{clean_text}"
                        
                        element_entry = {
                            "semantic_path": element_semantic_path,
                            "original_url": node.original_url,
                            "source_type": element_type[:-1],  # Remove 's' (heading, link, button)
                            "parent_url": node.parent_url,
                            "parent_semantic_path": semantic_path,
                            "depth": node.depth + 1,
                            "element_text": element_text,
                            "element_content": element.get('content', ''),
                            "element_type": element_type[:-1]
                        }
                        
                        # Add specific attributes
                        if element_type == 'links' and 'href' in element:
                            element_entry['href'] = element['href']
                        elif element_type == 'buttons' and 'type' in element:
                            element_entry['button_type'] = element['type']
                        
                        crawl_data["semantic_elements"][element_semantic_path] = element_entry
                        total_elements += 1
        
        # Update metadata with element count
        crawl_data["crawl_metadata"]["total_elements"] = total_elements
        
        # Save to file
        import os
        os.makedirs("output", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(crawl_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Hierarchical results saved to: {filename}")
        return filename


async def main():
    """Example usage of hierarchical crawler"""
    crawler = HierarchicalWebCrawler(
        start_url="https://www.mhsindiana.com/",
        max_depth=2,
        max_pages=20,  # Only crawl homepage
        headless=True
    )
    
    # Run hierarchical crawl
    crawl_nodes = await crawler.crawl_hierarchical()
    
    # Save results
    filename = crawler.save_hierarchical_results()
    
    # Show sample results
    print(f"\n🎯 SAMPLE HIERARCHICAL RESULTS:")
    print("=" * 60)
    
    for i, (semantic_path, node) in enumerate(list(crawl_nodes.items())[:5], 1):
        slash_type = "HEADING (single /)" if "//" not in semantic_path else "LINK (double //)"
        print(f"{i}. {semantic_path} [{slash_type}]")
        print(f"    📄 Source: {node.source_type} | Depth: {node.depth}")
        print(f"    🔗 Original: {node.original_url}")
        if node.page_content:
            print(f"    📝 Content: {len(node.page_content)} chars")
        print()


if __name__ == "__main__":
    asyncio.run(main())