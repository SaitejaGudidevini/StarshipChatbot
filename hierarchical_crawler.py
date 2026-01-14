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
    text: str = ""    # The text content of the heading or link
    parent_url: Optional[str] = None
    depth: int = 0
    visited: bool = False
    page_content: str = ""
    page_elements: Dict = field(default_factory=dict)
    element_count: int = 0
    discovered_headings: List[str] = field(default_factory=list)
    discovered_links: List[str] = field(default_factory=list)
    page_primary_heading: str = ""  # The main heading/title of this page


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
        timeout: int = 60000
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

        # 🔑 GLOBAL REGISTRY: "First-Discovery-Wins" Gatekeeper
        # Maps normalized URL to its canonical semantic path (first discovered)
        self.url_registry: Dict[str, str] = {}

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
            "backtrack_count": 0,
            "duplicates_blocked": 0,  # Total duplicates rejected by registry
            "heading_duplicates_blocked": 0,  # Heading duplicates blocked
            "link_duplicates_blocked": 0  # Link duplicates blocked
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

    def _register_url(self, url: str, semantic_path: str) -> bool:
        """
        🔑 GATEKEEPER: Atomic "First-Discovery-Wins" registration

        Attempts to claim a URL for a specific semantic path.
        This is the core of the duplicate prevention system.

        Args:
            url: The URL to register (will be normalized)
            semantic_path: The semantic path claiming this URL

        Returns:
            True: URL successfully claimed (first discovery) - proceed with processing
            False: URL already claimed by another path (duplicate) - reject immediately
        """
        normalized_url = self._normalize_url(url)

        # Check if URL already claimed by another discovery
        if normalized_url in self.url_registry:
            canonical_path = self.url_registry[normalized_url]
            logger.debug(f"⏭️  URL already claimed: {normalized_url}")
            logger.debug(f"    Canonical path: {canonical_path}")
            logger.debug(f"    Rejected path: {semantic_path}")
            return False  # Duplicate - reject

        # Claim the URL (First Discovery!)
        self.url_registry[normalized_url] = semantic_path
        logger.debug(f"🔒 URL claimed: {normalized_url} → {semantic_path}")
        return True  # Success - proceed with this discovery

    def _register_heading(self, heading_text: str, url: str, semantic_path: str) -> bool:
        """
        🔑 GATEKEEPER: Heading duplicate detection

        For headings, we check for duplicates based on:
        - Heading text (normalized)
        - Parent URL (the page it appears on)

        This prevents the same heading from appearing multiple times on the same page.

        Args:
            heading_text: The heading text
            url: The parent page URL
            semantic_path: The semantic path for this heading

        Returns:
            True: Heading is unique (first discovery) - proceed with processing
            False: Heading already exists (duplicate) - reject immediately
        """
        normalized_url = self._normalize_url(url)

        # Create a signature: normalized text + parent URL
        # This allows same heading text on DIFFERENT pages (different context)
        # But blocks same heading text on SAME page (true duplicate)
        clean_text = heading_text.strip().lower()[:100]  # First 100 chars, normalized
        heading_signature = f"{clean_text}@{normalized_url}"

        # Check if this heading signature already exists
        if heading_signature in self.url_registry:
            canonical_path = self.url_registry[heading_signature]
            logger.debug(f"⏭️  Heading already claimed: '{heading_text[:50]}'")
            logger.debug(f"    On page: {normalized_url}")
            logger.debug(f"    Canonical path: {canonical_path}")
            logger.debug(f"    Rejected path: {semantic_path}")
            return False  # Duplicate - reject

        # Claim this heading (First Discovery!)
        self.url_registry[heading_signature] = semantic_path
        logger.debug(f"🔒 Heading claimed: '{heading_text[:50]}' on {normalized_url}")
        return True  # Success - proceed with this discovery

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
    async def _navigate_safely(self, page: Page, url: str) -> Optional[object]:
        """
        Navigate to a URL with robust fallback strategies.
        1. Try networkidle (ideal for SPAs)
        2. Fallback to domcontentloaded (if networkidle times out)
        3. Fallback to load (if all else fails)
        """
        try:
            # Strategy 1: Network Idle (Best for dynamic content)
            response = await page.goto(url, wait_until='networkidle', timeout=self.timeout)
            return response
        except PlaywrightTimeout:
            logger.warning(f"⚠️ Timeout waiting for networkidle on {url}. Retrying with domcontentloaded...")
            try:
                # Strategy 2: DOM Content Loaded (Faster, less strict)
                response = await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                return response
            except Exception as e:
                logger.error(f"❌ Failed to navigate to {url} even with fallback: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ Error navigating to {url}: {e}")
            return None
    
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
                            
                            # Get following content for this element using multiple strategies
                            following_content = await element.evaluate('''
                                (el) => {
                                    let content = '';
                                    
                                    // Strategy 1: Look at immediate siblings
                                    let current = el.nextElementSibling;
                                    let elementCount = 0;
                                    let maxElements = 5; // Increased to look deeper
                                    
                                    while (current && elementCount < maxElements) {
                                        const tagName = current.tagName.toLowerCase();
                                        
                                        // Collect content from paragraphs, divs, spans (more permissive)
                                        if (tagName === 'p' || tagName === 'div' || tagName === 'span') {
                                            const text = current.innerText || current.textContent || '';
                                            if (text.trim().length > 10) {
                                                content += text.trim() + ' ';
                                            }
                                        }
                                        
                                        // Only stop if we hit a heading (allow links to continue)
                                        if (tagName.match(/^h[1-6]$/)) {
                                            break;
                                        }
                                        
                                        current = current.nextElementSibling;
                                        elementCount++;
                                    }
                                    
                                    // Strategy 2: If no content found, look within parent container
                                    if (content.trim().length === 0) {
                                        const parent = el.parentElement;
                                        if (parent) {
                                            // Look for content divs in the same parent
                                            const contentDivs = parent.querySelectorAll('div.excerpt, div.content, div.description, .summary, .abstract');
                                            for (let div of contentDivs) {
                                                const text = div.innerText || div.textContent || '';
                                                if (text.trim().length > 10) {
                                                    content += text.trim() + ' ';
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Strategy 3: If still no content, look for nearby paragraph content
                                    if (content.trim().length === 0) {
                                        const parent = el.parentElement;
                                        if (parent) {
                                            const paragraphs = parent.querySelectorAll('p');
                                            for (let p of paragraphs) {
                                                const text = p.innerText || p.textContent || '';
                                                if (text.trim().length > 10) {
                                                    content += text.trim() + ' ';
                                                    break; // Take first meaningful paragraph
                                                }
                                            }
                                        }
                                    }
                                    
                                    return content.trim();
                                }
                            ''')
                            
                            # Get additional attributes based on element type
                            extra_info = {}
                            if element_type == 'links':
                                href = await element.get_attribute('href')
                                class_attr = await element.get_attribute('class')
                                aria_label = await element.get_attribute('aria-label')
                                data_style = await element.get_attribute('data-style')
                                extra_info['href'] = href
                                extra_info['class'] = class_attr
                                extra_info['aria_label'] = aria_label
                                extra_info['data_style'] = data_style
                            elif element_type == 'buttons':
                                button_type = await element.get_attribute('type')
                                class_attr = await element.get_attribute('class')
                                extra_info['type'] = button_type
                                extra_info['class'] = class_attr
                            
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
    
    async def _extract_page_data_with_relationships(self, page: Page, url: str):
        """
        Extract page data by understanding element relationships within containers
        Uses Playwright locators for better relationship detection
        """
        structured_content = []

        try:
            # Step 1: Identify content containers (articles, posts, sections)
            container_selectors = [
                'article',
                '.post',
                '.blog-entry',
                '.card',
                'section[class*="content"]',
                'div[class*="item"]',
                '.row .col',  # Grid-based layouts
                'div[class*="post-"][class*="type-post"]'  # WordPress-style posts
            ]

            # Track what we've already processed to avoid duplicates
            processed_texts = set()

            for selector in container_selectors:
                containers = page.locator(selector)
                count = await containers.count()

                if count > 0:
                    logger.debug(f"Found {count} containers with selector: {selector}")

                for i in range(count):
                    container = containers.nth(i)
                    container_data = {
                        'selector': selector,
                        'elements': {},
                        'relationships': {}
                    }

                    try:
                        # Find the main heading within this container
                        heading_locator = container.locator('h1, h2, h3, h4, h5, h6').first
                        if await heading_locator.count() > 0:
                            heading_text = await heading_locator.inner_text()

                            # Skip if we've already processed this heading
                            if heading_text in processed_texts:
                                continue
                            processed_texts.add(heading_text)

                            container_data['elements']['heading'] = {
                                'text': heading_text.strip(),
                                'tag': await heading_locator.evaluate('el => el.tagName.toLowerCase()')
                            }

                            # Find content related to this heading using multiple strategies

                            # Strategy 1: Look for immediate sibling after heading
                            sibling_content = heading_locator.locator('xpath=./following-sibling::*[1][self::p or self::div]')
                            if await sibling_content.count() > 0:
                                content_text = await sibling_content.inner_text()
                                container_data['elements']['sibling_content'] = content_text.strip()

                            # Strategy 2: Look for excerpt/content divs within the container
                            excerpt_locator = container.locator('.excerpt, .content, .description, .summary').first
                            if await excerpt_locator.count() > 0:
                                excerpt_text = await excerpt_locator.inner_text()
                                container_data['elements']['excerpt'] = excerpt_text.strip()

                            # Strategy 3: Get all paragraphs within container
                            paragraphs = container.locator('p')
                            p_count = await paragraphs.count()
                            if p_count > 0:
                                p_texts = []
                                for j in range(min(p_count, 3)):  # Limit to first 3 paragraphs
                                    p_text = await paragraphs.nth(j).inner_text()
                                    if p_text and len(p_text.strip()) > 20:
                                        p_texts.append(p_text.strip())
                                if p_texts:
                                    container_data['elements']['paragraphs'] = p_texts

                        # Find the main link within this container
                        link_locator = container.locator('a[href]').first
                        if await link_locator.count() > 0:
                            link_data = {
                                'href': await link_locator.get_attribute('href'),
                                'text': await link_locator.inner_text(),
                                'class': await link_locator.get_attribute('class'),
                                'aria_label': await link_locator.get_attribute('aria-label')
                            }
                            container_data['elements']['main_link'] = link_data

                        # Find metadata (date, author, category)
                        metadata = {}

                        # Date
                        date_locator = container.locator('time, .date, .post-date, [datetime]').first
                        if await date_locator.count() > 0:
                            metadata['date'] = await date_locator.inner_text()

                        # Author
                        author_locator = container.locator('.author, .by-author, [rel="author"]').first
                        if await author_locator.count() > 0:
                            metadata['author'] = await author_locator.inner_text()

                        # Category
                        category_locator = container.locator('.category, .tag, [rel="category"]').first
                        if await category_locator.count() > 0:
                            metadata['category'] = await category_locator.inner_text()

                        if metadata:
                            container_data['elements']['metadata'] = metadata

                        # Build relationships
                        if 'heading' in container_data['elements']:
                            heading_text = container_data['elements']['heading']['text']

                            # Map heading to its content
                            content = (container_data['elements'].get('sibling_content') or
                                     container_data['elements'].get('excerpt') or
                                     (container_data['elements'].get('paragraphs', [''])[0] if 'paragraphs' in container_data['elements'] else ''))

                            if content:
                                container_data['relationships']['heading_to_content'] = {
                                    heading_text: content
                                }

                            # Map heading to its link
                            if 'main_link' in container_data['elements']:
                                container_data['relationships']['heading_to_link'] = {
                                    heading_text: container_data['elements']['main_link']['href']
                                }

                        # Only add if we found meaningful content
                        if container_data['elements']:
                            structured_content.append(container_data)

                    except Exception as e:
                        logger.debug(f"Error processing container: {e}")
                        continue

            return structured_content

        except Exception as e:
            logger.error(f"Error in relationship extraction from {url}: {e}")
            return []

    async def _extract_article_cards(self, page: Page):
        """
        Extract structured data from article cards (common blog/news pattern)
        Specifically handles patterns like PyTorch's blog cards
        """
        article_cards = []

        try:
            # Look for common article card patterns
            card_selectors = [
                'a.post[class*="type-post"]',  # PyTorch specific pattern
                'article.card',
                'div.blog-card',
                '.article-preview',
                'a[class*="col"][class*="post-"]',  # Grid-based post links
                '.post-item',
                '.news-item'
            ]

            for selector in card_selectors:
                cards = page.locator(selector)
                card_count = await cards.count()

                if card_count > 0:
                    logger.debug(f"Found {card_count} article cards with selector: {selector}")

                    for i in range(card_count):
                        card = cards.nth(i)

                        try:
                            # Extract all attributes from the card
                            card_data = {
                                'selector_used': selector,
                                'href': await card.get_attribute('href'),
                                'class': await card.get_attribute('class'),
                                'aria_label': await card.get_attribute('aria-label'),
                                'id': await card.get_attribute('id'),
                                'data_attributes': {},
                                'heading': None,
                                'excerpt': None,
                                'image': None,
                                'metadata': {}
                            }

                            # Get all data-* attributes
                            # We'll extract these using JavaScript for efficiency
                            data_attrs = await card.evaluate('''
                                (el) => {
                                    const attrs = {};
                                    for (let attr of el.attributes) {
                                        if (attr.name.startsWith('data-')) {
                                            attrs[attr.name] = attr.value;
                                        }
                                    }
                                    return attrs;
                                }
                            ''')
                            card_data['data_attributes'] = data_attrs

                            # Find heading within card (try multiple selectors)
                            heading_selectors = ['h1', 'h2', 'h3', 'h4', '.title', '.post-title', '.entry-title']
                            for h_sel in heading_selectors:
                                card_heading = card.locator(h_sel).first
                                if await card_heading.count() > 0:
                                    card_data['heading'] = await card_heading.inner_text()
                                    break

                            # Find excerpt within card
                            excerpt_selectors = ['.excerpt', '.description', '.content', '.summary', 'p']
                            for e_sel in excerpt_selectors:
                                card_excerpt = card.locator(e_sel).first
                                if await card_excerpt.count() > 0:
                                    excerpt_text = await card_excerpt.inner_text()
                                    if excerpt_text and len(excerpt_text.strip()) > 20:
                                        card_data['excerpt'] = excerpt_text.strip()
                                        break

                            # Find image
                            img_locator = card.locator('img').first
                            if await img_locator.count() > 0:
                                card_data['image'] = {
                                    'src': await img_locator.get_attribute('src'),
                                    'alt': await img_locator.get_attribute('alt')
                                }

                            # Extract date
                            date_locator = card.locator('time, .date, .post-date').first
                            if await date_locator.count() > 0:
                                card_data['metadata']['date'] = await date_locator.inner_text()

                            # Extract author
                            author_locator = card.locator('.author, .by-author').first
                            if await author_locator.count() > 0:
                                card_data['metadata']['author'] = await author_locator.inner_text()

                            # Extract category/tags
                            category_locator = card.locator('.category, .tag').first
                            if await category_locator.count() > 0:
                                card_data['metadata']['category'] = await category_locator.inner_text()

                            # Only add if we have meaningful data
                            if card_data['href'] or card_data['heading']:
                                article_cards.append(card_data)

                        except Exception as e:
                            logger.debug(f"Error processing article card: {e}")
                            continue

            return article_cards

        except Exception as e:
            logger.error(f"Error extracting article cards: {e}")
            return []

    def _build_element_relationships(self, structured_content):
        """
        Build a map of how elements relate to each other
        """
        relationships = {
            'heading_to_content': {},  # heading_text -> content_text
            'heading_to_links': {},    # heading_text -> [links]
            'container_groups': [],     # Groups of related elements
            'content_hierarchy': {}     # Nested structure
        }

        try:
            for container in structured_content:
                if 'relationships' in container and container['relationships']:
                    # Merge heading-to-content mappings
                    if 'heading_to_content' in container['relationships']:
                        relationships['heading_to_content'].update(
                            container['relationships']['heading_to_content']
                        )

                    # Merge heading-to-link mappings
                    if 'heading_to_link' in container['relationships']:
                        for heading, link in container['relationships']['heading_to_link'].items():
                            if heading not in relationships['heading_to_links']:
                                relationships['heading_to_links'][heading] = []
                            relationships['heading_to_links'][heading].append(link)

                # Group related elements from the same container
                if 'elements' in container and container['elements']:
                    group = {
                        'container_type': container.get('selector', 'unknown'),
                        'elements': container['elements']
                    }
                    relationships['container_groups'].append(group)

        except Exception as e:
            logger.error(f"Error building relationships: {e}")

        return relationships

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
    
    async def _discover_headings_and_links_with_locators(self, page: Page, current_url: str, current_semantic_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Discover headings and links using a robust Content-Aware approach.
        Identifies 'Topics' by looking for elements (Headings, Links, Divs) followed by descriptive content.
        Returns: ([(heading_text, semantic_path), ...], [(link_text, link_url, semantic_path), ...])
        """
        discovered_headings = []
        discovered_links = []

        try:
            # Execute JS to analyze the DOM structure and group links under topics
            # This matches the logic verified in tree_scraper_demo.py
            page_structure = await page.evaluate('''() => {
                // FIRST: Remove Footer and Header elements using Semantic HTML & ARIA Roles
                
                // 1. Semantic Tags (The "HTML Perspective")
                const semanticSelectors = [
                    'header',       // <header> tag
                    'footer',       // <footer> tag
                    'nav',          // <nav> tag
                    '[role="banner"]',      // ARIA role for header
                    '[role="contentinfo"]', // ARIA role for footer
                    '[role="navigation"]'   // ARIA role for nav
                ];
                
                semanticSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });

                // 2. Fallback: Common structural classes (Only if semantic tags missed them)
                // We keep these as a backup because not all sites use semantic HTML
                const fallbackSelectors = [
                    '.site-footer', '.global-footer', '#footer',
                    '.site-header', '.global-header', '#header',
                    // Utility links that often live outside headers
                    '.skip-link', '.skip-to-content', '#skip-to-content', 
                    '.language-picker', '.utility-nav', '.usa-banner' // Common on gov sites
                ];
                fallbackSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });

                const headings = [];
                const links = [];
                const processedLinks = new Set();
                
                // 1. Collect Headings (LEAVES)
                // We strictly treat H1-H6 as content leaves on the current page
                const headingElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
                headingElements.forEach(el => {
                    const text = el.innerText.trim();
                    if (text && text.length > 3) {
                        headings.push({
                            text: text,
                            type: el.tagName
                        });
                    }
                });

                // 2. Collect Links (BRANCHES)
                // We strictly treat Links as navigation to new sub-trees
                const linkElements = document.querySelectorAll('a[href]');
                linkElements.forEach(el => {
                    // Skip if already processed
                    if (processedLinks.has(el.href)) return;
                    
                    // Skip common non-content links
                    if (el.innerText.match(/Skip to|Menu|Search|Login/i)) return;

                    const text = el.innerText.trim();
                    if (text && text.length > 2) {
                        links.push({
                            text: text,
                            url: el.href,
                            is_topic: false // Links are always branches now
                        });
                        processedLinks.add(el.href);
                    }
                });
                
                return { headings, links };
            }''')

            # Process Headings (LEAVES)
            for h in page_structure['headings']:
                clean_text = self.labeler._clean_text_natural(h['text'])
                # Single slash / denotes content on current page
                # Use current_semantic_path as base
                semantic_path = f"{current_semantic_path}/{clean_text}"
                discovered_headings.append((h['text'], semantic_path))

            # Process Links (BRANCHES)
            for l in page_structure['links']:
                try:
                    normalized_url = self._normalize_url(l['url'])
                    
                    # Check domain
                    if not self._is_same_domain(normalized_url):
                        continue
                        
                    # Skip if already processed (global check)
                    if normalized_url in self.processed_urls:
                        continue

                    clean_text = self.labeler._clean_text_natural(l['text'])
                    # Double slash // denotes navigation to new page
                    # Use current_semantic_path as base
                    semantic_path = f"{current_semantic_path}//{clean_text}"
                    
                    discovered_links.append((l['text'], l['url'], semantic_path))
                    
                except Exception as e:
                    logger.warning(f"Error processing link {l}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in content-aware discovery: {e}")
            # Fallback to basic extraction if JS fails
            return [], []

        return discovered_headings, discovered_links

    async def _discover_headings_and_links(self, page: Page, current_url: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Discover headings and links from current page
        Now uses the improved locator-based method
        Returns: ([(heading_text, semantic_path), ...], [(link_text, link_url, semantic_path), ...])
        """
        # Use the new locator-based discovery method
        return await self._discover_headings_and_links_with_locators(page, current_url)

    async def _discover_headings_and_links_legacy(self, page: Page, current_url: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Legacy discovery method using query_selector_all
        Kept for backward compatibility
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
            
            # Extract comprehensive page data using BOTH methods
            # Method 1: Original extraction (for backward compatibility)
            page_elements, main_content, element_count = await self._extract_page_data(page, node.original_url)

            # Method 2: New relationship-aware extraction
            structured_content = await self._extract_page_data_with_relationships(page, node.original_url)
            article_cards = await self._extract_article_cards(page)
            relationships = self._build_element_relationships(structured_content)

            # Combine both extraction methods
            enhanced_elements = {
                **page_elements,  # Keep original extracted elements
                'structured_content': structured_content,
                'article_cards': article_cards,
                'relationships': relationships
            }

            # Update node with extracted data
            node.page_elements = enhanced_elements
            node.page_content = main_content
            node.element_count = element_count
            node.visited = True

            # Extract and store the page's primary heading/title
            page_primary_heading = ""

            # Strategy 1: Try to get the first H1 heading
            if 'headings' in page_elements and page_elements['headings']:
                for heading in page_elements['headings']:
                    if heading.get('text'):
                        page_primary_heading = heading['text'].strip()
                        break

            # Strategy 2: Try structured content for main heading
            if not page_primary_heading and 'structured_content' in enhanced_elements:
                for container in enhanced_elements['structured_content']:
                    if 'elements' in container and 'heading' in container['elements']:
                        heading_elem = container['elements']['heading']
                        if heading_elem.get('text') and heading_elem.get('tag') in ['h1', 'h2']:
                            page_primary_heading = heading_elem['text'].strip()
                            break

            # Strategy 3: Extract from the node's semantic_path as fallback
            if not page_primary_heading:
                # For links: extract from semantic path (e.g., "https://domain//Text" -> "Text")
                if '//' in node.semantic_path:
                    page_primary_heading = node.semantic_path.split('//')[-1].split('/')[0]
                else:
                    # For headings or other: extract last segment
                    page_primary_heading = node.semantic_path.split('/')[-1]

            # Store the primary heading in the node
            node.page_primary_heading = page_primary_heading
            logger.debug(f"Page primary heading identified: '{page_primary_heading}'")

            # Discover headings and links for further crawling
            discovered_headings, discovered_links = await self._discover_headings_and_links_with_locators(page, node.original_url, node.semantic_path)

            # DEBUG: Print what was discovered
            logger.info(f"🔍 DEBUG: Discovered {len(discovered_headings)} headings and {len(discovered_links)} links from {node.original_url}")
            if discovered_links:
                logger.info(f"   📋 Links found: {[link[0][:50] for link in discovered_links[:5]]}")  # Show first 5 link texts

            # Add discovered headings to heading queue (with Global Registry gatekeeper)
            if node.depth < self.max_depth:
                headings_added_count = 0
                heading_duplicates_blocked = 0

                for heading_text, semantic_path in discovered_headings:
                    # 🔑 GATEKEEPER: Check if this heading is a duplicate
                    if not self._register_heading(heading_text, node.original_url, semantic_path):
                        # Heading already exists on this page - duplicate
                        heading_duplicates_blocked += 1
                        logger.debug(f"⏭️  Blocked duplicate heading: '{heading_text[:50]}'")
                        continue  # Discard immediately - do not create node, do not queue

                    # Heading is unique - proceed with creation
                    if semantic_path not in self.crawl_nodes:
                        heading_node = CrawlNode(
                            original_url=node.original_url,  # Headings reference same page
                            semantic_path=semantic_path,
                            source_type='heading',
                            text=heading_text,
                            parent_url=node.original_url,
                            depth=node.depth + 1
                        )
                        self.crawl_nodes[semantic_path] = heading_node
                        self.heading_queue.appendleft(heading_node)  # DFS: Add to FRONT for depth-first
                        self.stats["total_headings_found"] += 1
                        headings_added_count += 1

                # Log heading processing results
                if heading_duplicates_blocked > 0:
                    logger.info(f"📰 Added {headings_added_count} unique headings, blocked {heading_duplicates_blocked} duplicates")
                    self.stats["heading_duplicates_blocked"] = self.stats.get("heading_duplicates_blocked", 0) + heading_duplicates_blocked

                # Add discovered links to link queue (with Global Registry gatekeeper)
                links_added_count = 0
                duplicates_blocked = 0

                for link_text, link_url, semantic_path in discovered_links:
                    # 🔑 GATEKEEPER: Try to claim this URL in the registry
                    if not self._register_url(link_url, semantic_path):
                        # URL already claimed by another path - this is a duplicate
                        duplicates_blocked += 1
                        logger.debug(f"⏭️  Blocked duplicate: '{link_text[:50]}' → {link_url}")
                        continue  # Discard immediately - do not create node, do not queue

                    # URL successfully claimed! This is the canonical path for this URL
                    # Now proceed with normal node creation
                    if semantic_path not in self.crawl_nodes:
                        link_node = CrawlNode(
                            original_url=link_url,
                            semantic_path=semantic_path,
                            source_type='link',
                            text=link_text,
                            parent_url=node.original_url,
                            depth=node.depth + 1
                        )
                        self.crawl_nodes[semantic_path] = link_node
                        self.link_queue.appendleft(link_node)  # DFS: Add to FRONT for depth-first
                        self.stats["total_links_found"] += 1
                        links_added_count += 1

                # Enhanced logging with duplicate count
                logger.info(f"✅ Added {links_added_count} unique links, blocked {duplicates_blocked} duplicates (depth {node.depth} → {node.depth + 1})")
                self.stats["link_duplicates_blocked"] = self.stats.get("link_duplicates_blocked", 0) + duplicates_blocked
                self.stats["duplicates_blocked"] = self.stats.get("duplicates_blocked", 0) + duplicates_blocked

                # DEBUG: Show tree structure for first few links
                if links_added_count > 0 and node.depth <= 2:
                    indent = "  " * node.depth
                    logger.info(f"🌳 TREE: {indent}└─> Parent: {node.semantic_path.split('/')[-1][:40]}")
                    for link_text, link_url, semantic_path in list(discovered_links)[:3]:
                        if semantic_path in self.crawl_nodes:
                            logger.info(f"🌳 TREE: {indent}    ├─> Child: {link_text[:40]} (depth {node.depth + 1})")
            else:
                # Track "extra deep" elements that exceeded max depth
                for heading_text, semantic_path in discovered_headings:
                    if semantic_path not in self.crawl_nodes:
                        extra_deep_heading = CrawlNode(
                            original_url=node.original_url,
                            semantic_path=semantic_path,
                            source_type='heading',
                            text=heading_text,
                            parent_url=node.original_url,
                            depth=node.depth + 1,
                            visited=False,
                            page_content="extra deep"
                        )
                        self.crawl_nodes[semantic_path] = extra_deep_heading
                        self.stats["total_headings_found"] += 1
                
                # Track "extra deep" links that exceeded max depth
                for link_text, link_url, semantic_path in discovered_links:
                    if semantic_path not in self.crawl_nodes:
                        extra_deep_link = CrawlNode(
                            original_url=link_url,
                            semantic_path=semantic_path,
                            source_type='link',
                            text=link_text,
                            parent_url=node.original_url,
                            depth=node.depth + 1,
                            visited=False,
                            page_content="extra deep"
                        )
                        self.crawl_nodes[semantic_path] = extra_deep_link
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

            # 🔑 REGISTRY: Pre-register the homepage to prevent re-discovery
            # (e.g., from logo links, footer links, etc.)
            self.url_registry[self.start_url] = root_semantic
            logger.info(f"🔒 Registry initialized: {self.start_url} → {root_semantic}")

            # Phase 1: Crawl homepage and discover initial headings/links
            logger.info("\n🚀 PHASE 1: Homepage Discovery")
            success = await self._crawl_single_page(root_node)
            
            if not success:
                logger.error("❌ Failed to crawl homepage. Aborting.")
                return self.crawl_nodes
            
            # Phase 2: Process all headings first (depth-first - goes deep on each branch)
            logger.info(f"\n📰 PHASE 2: Processing {len(self.heading_queue)} Headings (Priority)")
            while self.heading_queue and len(self.processed_urls) < self.max_pages:
                heading_node = self.heading_queue.popleft()
                if not heading_node.visited:
                    await self._crawl_single_page(heading_node)

            # Phase 3: Process all links (depth-first - goes deep on each branch)
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
        logger.info(f"   🚫 Total duplicates blocked by registry: {self.stats.get('duplicates_blocked', 0)}")
        logger.info(f"      📰 Heading duplicates blocked: {self.stats.get('heading_duplicates_blocked', 0)}")
        logger.info(f"      🔗 Link duplicates blocked: {self.stats.get('link_duplicates_blocked', 0)}")
        logger.info(f"   ✅ Unique items registered: {len(self.url_registry)}")
    
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
            "semantic_elements": {},
            "article_cards": [],
            "structured_content": [],
            "element_relationships": {}
        }

        total_elements = 0
        total_article_cards = 0

        # Process each crawl node and create individual entries for each element
        for semantic_path, node in self.crawl_nodes.items():
            # 1. Add the node itself as a semantic element
            # This ensures every discovered node (heading or link) is in the output
            crawl_data["semantic_elements"][semantic_path] = {
                "text": node.text if node.text else node.page_primary_heading,
                "element_type": node.source_type,
                "original_url": node.original_url,
                "parent_url": node.parent_url,
                "depth": node.depth,
                "content": node.page_content if node.visited else None
            }

            # 2. Handle visited nodes (add extra content)
            if node.visited and node.page_elements:
                # Add new structured content if available
                if 'structured_content' in node.page_elements:
                    crawl_data["structured_content"].extend(node.page_elements['structured_content'])

                # Add article cards if available
                if 'article_cards' in node.page_elements:
                    for card in node.page_elements['article_cards']:
                        card['found_on_page'] = node.original_url
                        crawl_data["article_cards"].append(card)
                        total_article_cards += 1

                # Add relationships if available
                if 'relationships' in node.page_elements:
                    rel = node.page_elements['relationships']
                    if 'heading_to_content' in rel:
                        crawl_data["element_relationships"].update(rel['heading_to_content'])
            
            total_elements += 1

        
        # Update metadata with all counts
        crawl_data["crawl_metadata"]["total_elements"] = total_elements
        crawl_data["crawl_metadata"]["total_article_cards"] = total_article_cards
        crawl_data["crawl_metadata"]["total_structured_containers"] = len(crawl_data["structured_content"])
        crawl_data["crawl_metadata"]["total_relationships"] = len(crawl_data["element_relationships"])
        
        # Save to file
        import os
        os.makedirs("output", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(crawl_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Hierarchical results saved to: {filename}")
        
        # Also save tree structure for visualization
        try:
            logger.info("🌳 Starting tree structure save...")
            tree_filename = self.save_tree_structure(filename)
            logger.info(f"🌳 Tree structure saved to: {tree_filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save tree structure: {e}", exc_info=True)
        
        return filename
    
    def save_tree_structure(self, base_filename: str = None) -> str:
        """
        Save hierarchical tree structure for D3.js visualization
        Converts flat crawl_nodes into nested parent-child tree structure
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain_clean = self.domain.replace('.', '_')
            base_filename = f"output/hierarchical_crawl_{domain_clean}_{timestamp}.json"
        
        # Create tree filename
        tree_filename = base_filename.replace('.json', '_tree.json')
        
        # Find root node (depth 0)
        root_node = None
        for node in self.crawl_nodes.values():
            if node.depth == 0:
                root_node = node
                break
        
        if not root_node:
            logger.warning("No root node found, cannot build tree")
            return tree_filename
        
        # Build hierarchical tree iteratively (avoids recursion depth issues)
        def build_tree_node_iterative(root_node: CrawlNode) -> dict:
            """Build tree structure iteratively to avoid recursion depth errors"""
            from collections import deque
            
            logger.info(f"🔨 Building tree from {len(self.crawl_nodes)} total nodes...")
            
            # Create root tree node
            def create_tree_node(node: CrawlNode) -> dict:
                # Extract title from semantic path
                if '//' in node.semantic_path:
                    title = node.semantic_path.split('//')[-1].split('/')[0]
                else:
                    title = node.semantic_path.split('/')[-1]
                
                display_title = node.text if node.text else title
                
                return {
                    "title": display_title,
                    "url": node.original_url,
                    "semantic_path": node.semantic_path,
                    "source_type": node.source_type,
                    "depth": node.depth,
                    "visited": node.visited,
                    "has_content": bool(node.page_content),
                    "children": [],
                    "_crawl_node": node  # Temporary reference
                }
            
            # Build tree iteratively using DFS
            root_tree = create_tree_node(root_node)
            queue = deque([root_tree])
            processed = 0
            visited_paths = {root_node.semantic_path}  # Track visited semantic paths to prevent duplicates

            logger.info(f"🔄 Starting DFS tree building...")

            while queue:
                current_tree_node = queue.popleft()
                current_crawl_node = current_tree_node["_crawl_node"]
                processed += 1

                if processed % 10 == 0:
                    logger.info(f"   Processed {processed} nodes, queue size: {len(queue)}, visited: {len(visited_paths)}")

                # Find children
                children = []
                for child_node in self.crawl_nodes.values():
                    if (child_node.parent_url == current_crawl_node.original_url and
                        child_node.semantic_path != current_crawl_node.semantic_path and
                        child_node.semantic_path not in visited_paths):  # Prevent duplicates using semantic_path
                        children.append(child_node)
                        visited_paths.add(child_node.semantic_path)  # Mark semantic_path as visited
                
                # Sort children
                children.sort(key=lambda n: (n.source_type != 'heading', n.depth, n.semantic_path))
                
                # Add children to tree
                for child in children:
                    child_tree_node = create_tree_node(child)
                    current_tree_node["children"].append(child_tree_node)
                    queue.appendleft(child_tree_node)  # DFS: Add to FRONT for depth-first tree building

            logger.info(f"✅ DFS complete. Processed {processed} nodes total.")
            logger.info(f"🧹 Cleaning temporary references...")
            
            # Remove temporary references iteratively
            clean_queue = deque([root_tree])
            cleaned = 0
            while clean_queue:
                node = clean_queue.popleft()
                if "_crawl_node" in node:
                    del node["_crawl_node"]
                    cleaned += 1
                for child in node.get("children", []):
                    clean_queue.append(child)
            
            logger.info(f"✅ Cleaned {cleaned} nodes.")
            return root_tree
        
        # Build the complete tree
        tree_data = build_tree_node_iterative(root_node)
        
        # Add metadata
        tree_output = {
            "metadata": {
                "domain": self.domain,
                "start_url": self.start_url,
                "timestamp": datetime.now().isoformat(),
                "total_nodes": len(self.crawl_nodes),
                "max_depth": self.max_depth,
                "format": "hierarchical_tree_for_d3js"
            },
            "tree": tree_data
        }
        
        # Save to file
        import os
        os.makedirs("output", exist_ok=True)
        
        with open(tree_filename, 'w', encoding='utf-8') as f:
            json.dump(tree_output, f, indent=2, ensure_ascii=False)
        
        return tree_filename


async def main():
    """Example usage of hierarchical crawler"""
    crawler = HierarchicalWebCrawler(
        start_url="https://pytorch.org",
        max_depth=10,
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