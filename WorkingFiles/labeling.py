"""
Agentic Web Navigator - Semantic Labeling Module
Handles intelligent labeling of navigation elements using heuristics and LLM
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from html.parser import HTMLParser
from groq import Groq
from urllib.parse import urlparse, urljoin
import os

logger = logging.getLogger(__name__)


class HTMLContextExtractor(HTMLParser):
    """Extract relevant context from HTML for labeling"""
    
    def __init__(self):
        super().__init__()
        self.reset()
        self.text_content = []
        
    def handle_data(self, data):
        if data.strip():
            self.text_content.append(data.strip())
    
    def get_text(self):
        return ' '.join(self.text_content)


class SemanticLabeler:
    """
    Hierarchical labeling system:
    1. Heuristic analysis (primary)
    2. LLM interpretation via Groq (secondary)
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_client = None
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
            logger.info("Groq client initialized for LLM labeling")
        else:
            logger.warning("No Groq API key provided. Falling back to heuristics only.")
        
        # Common generic terms to avoid as labels
        self.generic_terms = {
            'click here', 'read more', 'learn more', 'more info',
            'view', 'see', 'go', 'link', 'here', 'this', 'that',
            'continue', 'next', 'previous', 'back', 'forward'
        }
    
    def _is_generic_label(self, label: str) -> bool:
        """Check if a label is too generic"""
        if not label:
            return True
        
        label_lower = label.lower().strip()
        
        # Check exact matches
        if label_lower in self.generic_terms:
            return True
        
        # Check if it's just a number or single character
        if len(label) <= 1 or label.isdigit():
            return True
        
        # Check if it contains only generic terms
        for term in self.generic_terms:
            if term == label_lower:
                return True
        
        return False
    
    def _extract_heuristic_label(
        self,
        text: str,
        href: str,
        html_context: str
    ) -> Optional[str]:
        """
        Extract label using heuristic analysis
        Priority order:
        1. Clean inner text
        2. ARIA labels
        3. Title attributes
        4. Alt text (for image links)
        5. URL path analysis
        """
        
        # 1. Check inner text first
        if text and not self._is_generic_label(text):
            # Clean and truncate if necessary
            clean_text = re.sub(r'\s+', ' ', text).strip()
            if len(clean_text) <= 50:  # Reasonable length for a label
                return clean_text
        
        # 2. Check for ARIA label
        aria_match = re.search(r'aria-label=["\']([^"\']+)["\']', html_context, re.IGNORECASE)
        if aria_match:
            aria_label = aria_match.group(1)
            if not self._is_generic_label(aria_label):
                return aria_label
        
        # 3. Check for title attribute
        title_match = re.search(r'title=["\']([^"\']+)["\']', html_context, re.IGNORECASE)
        if title_match:
            title = title_match.group(1)
            if not self._is_generic_label(title):
                return title
        
        # 4. Check for alt text (image links)
        alt_match = re.search(r'alt=["\']([^"\']+)["\']', html_context, re.IGNORECASE)
        if alt_match:
            alt_text = alt_match.group(1)
            if not self._is_generic_label(alt_text):
                return alt_text
        
        # 5. Analyze URL path as last resort
        if href:
            path_parts = href.strip('/').split('/')
            if path_parts:
                last_part = path_parts[-1]
                # Clean up URL part (remove query params, hyphens to spaces)
                last_part = last_part.split('?')[0].split('#')[0]
                last_part = last_part.replace('-', ' ').replace('_', ' ')
                if last_part and not self._is_generic_label(last_part):
                    return last_part.title()
        
        return None
    
    async def _get_llm_label(
        self,
        href: str,
        html_context: str
    ) -> Optional[str]:
        """
        Use Groq LLM to determine semantic label from HTML context
        """
        if not self.groq_client:
            return None
        
        try:
            # Prepare concise context for LLM
            parser = HTMLContextExtractor()
            parser.feed(html_context)
            text_context = parser.get_text()[:200]  # Limit context size
            
            prompt = f"""Analyze this HTML link and provide a concise (1-3 words) semantic label.

URL: {href}
Context: {text_context}
HTML: {html_context[:300]}

Rules:
- Focus on what the link navigates TO, not the action
- Use proper nouns when available
- Avoid generic terms like "click here" or "learn more"
- If it's a category or section, name the category
- If it's a product or service, name it specifically

Respond with ONLY the label, nothing else."""

            response = self.groq_client.chat.completions.create(
                model="llama-3.2-3b-preview",
                messages=[
                    {"role": "system", "content": "You are a web navigation expert. Provide only the requested label."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            label = response.choices[0].message.content.strip()
            
            # Validate LLM response
            if label and len(label) <= 50 and not self._is_generic_label(label):
                logger.debug(f"LLM generated label: {label} for {href}")
                return label
            
        except Exception as e:
            logger.error(f"Error getting LLM label: {e}")
        
        return None
    
    async def get_semantic_label(
        self,
        text: str,
        href: str,
        html_context: str,
        parent_label: Optional[str] = None
    ) -> str:
        """
        Main method to determine semantic label using hierarchical approach
        """
        # Step 1: Try heuristic extraction
        heuristic_label = self._extract_heuristic_label(text, href, html_context)
        
        if heuristic_label:
            logger.debug(f"Using heuristic label: {heuristic_label}")
            return heuristic_label
        
        # Step 2: Fallback to LLM if available
        if self.groq_client:
            llm_label = await self._get_llm_label(href, html_context)
            if llm_label:
                logger.debug(f"Using LLM label: {llm_label}")
                return llm_label
        
        # Step 3: Last resort - use cleaned text or URL part
        if text and len(text) < 100:
            return text[:50]
        
        # Extract from URL
        url_parts = href.strip('/').split('/')
        if url_parts:
            return url_parts[-1].replace('-', ' ').replace('_', ' ').title()[:50]
        
        return "Page"
    
    async def generate_url_heading_labels_from_crawl(
        self,
        crawl_data_file: str
    ) -> Dict[str, str]:
        """
        Generate URL+heading labels from crawler output file
        Returns mapping of original_url -> semantic_path
        """
        import json
        
        with open(crawl_data_file, 'r') as f:
            crawl_data = json.load(f)
        
        # Extract domain from metadata
        domain = crawl_data.get('crawl_metadata', {}).get('domain', 'unknown.com')
        
        semantic_paths = {}
        
        # Process each page
        for url, page_data in crawl_data.get('pages', {}).items():
            # Generate semantic path
            semantic_path = self.generate_url_heading_semantic_path(
                url,
                page_data.get('page_elements', {}),
                domain
            )
            
            semantic_paths[url] = semantic_path
            
            print(f"🏷️  {url}")
            print(f"   → {semantic_path}")
        
        return semantic_paths
    
    def create_semantic_path(
        self,
        labels_chain: list[str]
    ) -> list[str]:
        """
        Create a clean semantic path from a chain of labels
        Removes duplicates and cleans up the path
        """
        semantic_path = []
        
        for label in labels_chain:
            if not label:
                continue
            
            # Clean the label
            clean_label = re.sub(r'\s+', ' ', label).strip()
            
            # Avoid consecutive duplicates
            if semantic_path and semantic_path[-1].lower() == clean_label.lower():
                continue
            
            semantic_path.append(clean_label)
        
        return semantic_path
    
    def generate_url_heading_semantic_path(
        self,
        url: str,
        page_elements: Dict,
        domain: str
    ) -> str:
        """
        Generate clean semantic path based on content priority:
        - For headings (h1, h2): https://www.mhsindiana.com/Find a provider (single slash)
        - For links: https://www.mhsindiana.com//For Members (double slash)
        """
        
        # First priority: Extract primary heading (h1, h2, etc.)
        primary_heading = self._extract_primary_heading(page_elements)
        if primary_heading:
            # Clean heading text but keep natural spacing
            clean_heading = self._clean_text_natural(primary_heading)
            semantic_path = f"https://{domain}/{clean_heading}"
            return semantic_path
        
        # Second priority: Extract meaningful link text from page links
        meaningful_link = self._extract_meaningful_link_text(page_elements)
        if meaningful_link:
            # Clean link text but keep natural spacing
            clean_link = self._clean_text_natural(meaningful_link)
            semantic_path = f"https://{domain}//{clean_link}"
            return semantic_path
        
        # Fallback: Use cleaned URL path with single slash
        parsed_url = urlparse(url)
        url_path = parsed_url.path.strip('/')
        if not url_path:
            clean_path = "Home"
        else:
            clean_path = self._clean_url_path_natural(url_path)
        
        semantic_path = f"https://{domain}/{clean_path}"
        return semantic_path
    
    def _extract_primary_heading(self, page_elements: Dict) -> Optional[str]:
        """
        Extract the most relevant heading (h1 first, then h2, etc.)
        """
        headings = page_elements.get('headings', [])
        if not headings:
            return None
        
        # Priority order: h1 > h2 > h3...
        for heading in headings:
            if heading.get('tag') == 'h1' and heading.get('text'):
                return heading['text'][:50]  # Limit length
        
        for heading in headings:
            if heading.get('tag') == 'h2' and heading.get('text'):
                return heading['text'][:50]
        
        # Fallback to first available heading
        for heading in headings:
            if heading.get('text'):
                return heading['text'][:50]
        
        return None
    
    def _clean_text_natural(self, text: str) -> str:
        """
        Clean text while preserving natural spacing and readability
        """
        if not text:
            return ''
        
        # Remove only problematic characters, keep natural spaces
        clean_text = re.sub(r'[^\w\s-]', '', text)
        # Normalize multiple spaces to single space
        clean_text = re.sub(r'\s+', ' ', clean_text.strip())
        
        return clean_text[:50]  # Reasonable length limit
    
    def _clean_url_path_natural(self, path: str) -> str:
        """
        Convert URL path to natural readable format
        """
        if not path:
            return 'Home'
        
        # Convert hyphens and underscores to spaces
        clean_path = path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
        # Remove file extensions
        clean_path = re.sub(r'\.(html|php|jsp|asp)$', '', clean_path)
        # Capitalize first letter of each word
        clean_path = clean_path.title().strip()
        
        return clean_path[:50]
    
    def _extract_meaningful_link_text(self, page_elements: Dict) -> Optional[str]:
        """
        Extract meaningful link text from page links
        Prioritizes navigation-style links over generic ones
        """
        links = page_elements.get('links', [])
        if not links:
            return None
        
        # Look for meaningful link text (not generic terms)
        meaningful_links = []
        for link in links:
            text = link.get('text', '').strip()
            if text and not self._is_generic_label(text) and len(text) > 3:
                # Prioritize navigation-style links
                if any(nav_word in text.lower() for nav_word in ['for', 'about', 'services', 'members', 'providers']):
                    return text[:50]
                meaningful_links.append(text)
        
        # Return first meaningful link if found
        if meaningful_links:
            return meaningful_links[0][:50]
        
        return None
    
    def categorize_link_type(self, url: str, domain: str) -> str:
        """
        Categorize links as internal vs external based on domain
        """
        parsed_url = urlparse(url)
        
        if not parsed_url.netloc:  # Relative URL
            return 'internal'
        elif parsed_url.netloc == domain:
            return 'internal'
        else:
            return 'external'
    
    def generate_semantic_paths_for_crawl_data(
        self,
        navigation_map: Dict,
        domain: str
    ) -> Dict[str, Dict]:
        """
        Generate semantic paths for all pages in crawl data
        Returns enhanced navigation map with semantic paths
        """
        enhanced_map = {}
        
        for url, node in navigation_map.items():
            # Generate semantic path using URL + heading
            semantic_path = self.generate_url_heading_semantic_path(
                url, 
                node.page_elements, 
                domain
            )
            
            # Categorize link type
            link_type = self.categorize_link_type(url, domain)
            
            # Create enhanced node data
            enhanced_map[url] = {
                'original_url': url,
                'semantic_path': semantic_path,
                'link_type': link_type,
                'primary_heading': self._extract_primary_heading(node.page_elements),
                'depth': node.depth,
                'parent_url': node.parent_url,
                'element_count': node.element_count,
                'page_elements': node.page_elements,
                'main_content': node.main_content
            }
            
            logger.info(f"Generated semantic path: {semantic_path} [{link_type}]")
        
        return enhanced_map