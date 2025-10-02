"""
Playwright Content Extractor Service
Dynamically extracts content from web pages based on element type and target text
"""

import asyncio
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ContentExtractor:
    """
    Service to extract live content from web pages using Playwright
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self.playwright = None
        self.browser: Optional[Browser] = None
        
    async def initialize(self):
        """Initialize Playwright browser"""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled']
            )
            logger.info("🎭 Content Extractor initialized")
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def extract_content(
        self, 
        url: str, 
        element_type: str, 
        target_text: str
    ) -> str:
        """
        Extract content from a webpage based on element type and target text
        
        Args:
            url: The webpage URL to visit
            element_type: Type of element (heading, link, button, div, etc.)
            target_text: The text to search for within that element type
            
        Returns:
            Extracted content string or error message
        """
        try:
            await self.initialize()
            
            # Create isolated context
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            try:
                print(f"🌐 Navigating to: {url}")
                response = await page.goto(url, wait_until='networkidle', timeout=self.timeout)
                
                if not response or response.status >= 400:
                    return f"Failed to load page: {response.status if response else 'No response'}"
                
                # Remove header/footer elements for cleaner extraction
                await self._remove_header_footer(page)
                
                # Find the target element based on type and text
                target_element = await self._find_target_element(page, element_type, target_text)
                
                if not target_element:
                    return "Content not found"
                
                # Extract content following the target element
                content = await self._extract_following_content(target_element)
                
                return content if content else "No content found after element"
                
            finally:
                await page.close()
                await context.close()
                
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return f"Error: {str(e)}"
    
    async def _remove_header_footer(self, page: Page):
        """Remove header and footer elements for cleaner content"""
        await page.evaluate('''
            () => {
                // Remove headers
                const headerSelectors = ['header', '.header', '#header', 'nav', '.nav', '.navigation', '.navbar', '.topbar', '.top-bar'];
                headerSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });
                
                // Remove footers
                const footerSelectors = [
                    'footer', '.footer', '#footer', '.bottom', '.copyright', '.site-footer',
                    '.footer-nav', '.global-footer', '.pdffooter'
                ];
                footerSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });
            }
        ''')
    
    async def _find_target_element(self, page: Page, element_type: str, target_text: str):
        """Find the target element based on type and text content"""
        
        # Define selectors based on element type
        type_selectors = {
            'heading': 'h1, h2, h3, h4, h5, h6',
            'link': 'a[href]',
            'button': 'button, input[type="button"], input[type="submit"]',
            'div': 'div',
            'span': 'span',
            'paragraph': 'p',
            'article': 'article',
            'section': 'section'
        }
        
        selector = type_selectors.get(element_type.lower(), '*')
        
        try:
            elements = await page.query_selector_all(selector)
            
            for element in elements:
                try:
                    element_text = await element.inner_text()
                    
                    # Check if the element contains the target text (case-insensitive partial match)
                    if target_text.lower() in element_text.lower():
                        print(f"✅ Found {element_type}: '{element_text[:50]}...'")
                        return element
                        
                except Exception:
                    continue
                    
            # If exact match not found, try more flexible matching
            print(f"🔍 Exact match not found, trying flexible search for '{target_text}'")
            return await self._flexible_search(page, target_text)
            
        except Exception as e:
            logger.error(f"Error finding target element: {e}")
            return None
    
    async def _flexible_search(self, page: Page, target_text: str):
        """More flexible search when exact match fails"""
        try:
            # Search in all text-containing elements
            all_elements = await page.query_selector_all('*')
            
            best_match = None
            highest_score = 0
            
            for element in all_elements:
                try:
                    element_text = await element.inner_text()
                    
                    if element_text and len(element_text.strip()) > 5:
                        # Simple scoring based on word overlap
                        target_words = set(target_text.lower().split())
                        element_words = set(element_text.lower().split())
                        
                        if target_words & element_words:  # If there's any word overlap
                            score = len(target_words & element_words) / len(target_words)
                            
                            if score > highest_score and score > 0.3:  # At least 30% match
                                highest_score = score
                                best_match = element
                                
                except Exception:
                    continue
            
            if best_match:
                text = await best_match.inner_text()
                print(f"📍 Best flexible match (score: {highest_score:.2f}): '{text[:50]}...'")
                
            return best_match
            
        except Exception as e:
            logger.error(f"Error in flexible search: {e}")
            return None
    
    async def _extract_following_content(self, element) -> str:
        """Extract content that follows the target element"""
        try:
            content = await element.evaluate('''
                (el) => {
                    let content = '';
                    
                    // Strategy 1: Look at immediate siblings
                    let current = el.nextElementSibling;
                    let elementCount = 0;
                    
                    while (current && elementCount < 5) {
                        const tagName = current.tagName.toLowerCase();
                        
                        // Collect content from content elements
                        if (tagName === 'p' || tagName === 'div' || tagName === 'span' || tagName === 'section') {
                            const text = current.innerText || current.textContent || '';
                            if (text.trim().length > 10) {
                                content += text.trim() + '\\n\\n';
                            }
                        }
                        
                        // Stop if we hit another heading
                        if (tagName.match(/^h[1-6]$/)) {
                            break;
                        }
                        
                        current = current.nextElementSibling;
                        elementCount++;
                    }
                    
                    // Strategy 2: Look within parent container if no content found
                    if (content.trim().length === 0) {
                        const parent = el.parentElement;
                        if (parent) {
                            const contentElements = parent.querySelectorAll('p, div.content, div.excerpt, div.description, .summary');
                            for (let elem of contentElements) {
                                const text = elem.innerText || elem.textContent || '';
                                if (text.trim().length > 20 && !text.includes(el.innerText)) {
                                    content += text.trim() + '\\n\\n';
                                }
                            }
                        }
                    }
                    
                    // Strategy 3: Get content from article/main containers
                    if (content.trim().length === 0) {
                        const containers = document.querySelectorAll('article, main, .content, .post-content');
                        for (let container of containers) {
                            if (container.contains(el)) {
                                const text = container.innerText || container.textContent || '';
                                if (text.trim().length > 50) {
                                    // Extract a relevant portion
                                    const sentences = text.split('.').slice(0, 3);
                                    content = sentences.join('.') + '.';
                                    break;
                                }
                            }
                        }
                    }
                    
                    return content.trim();
                }
            ''')
            
            # Clean and limit the content
            if content:
                # Remove extra whitespace and limit length
                content = ' '.join(content.split())
                if len(content) > 800:
                    content = content[:800] + "..."
                    
            return content
            
        except Exception as e:
            logger.error(f"Error extracting following content: {e}")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the content extractor"""
        try:
            await self.initialize()
            return {
                "status": "healthy",
                "browser": "ready" if self.browser else "not initialized"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Example usage
async def main():
    """Test the content extractor"""
    extractor = ContentExtractor(headless=False)
    
    try:
        content = await extractor.extract_content(
            url="https://pytorch.org/join-ecosystem",
            element_type="heading",
            target_text="Application Process"
        )
        
        print("📄 Extracted Content:")
        print(content)
        
    finally:
        await extractor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())