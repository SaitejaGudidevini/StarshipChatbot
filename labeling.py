"""
Agentic Web Navigator - Semantic Labeling Module
Handles intelligent labeling of navigation elements using heuristics and LLM
"""

import re
import logging
from typing import Optional, Dict, Any
from html.parser import HTMLParser
from groq import Groq
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
                model="llama-3.3-70b-versatile",
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