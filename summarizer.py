"""
SemanticSummaryService - Generates business-focused page summaries using Groq
Enhanced with response generation for chatbot training
"""

import os
import logging
import re
from typing import Optional, List, Dict, Any
from groq import Groq

logger = logging.getLogger(__name__)


class SemanticSummaryService:
    """
    Simple service to generate page summaries using Groq API
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 220,
        enabled: bool = True
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.enabled = enabled
        
        if self.enabled and groq_api_key:
            try:
                self.client = Groq(api_key=groq_api_key)
                logger.info(f"SemanticSummaryService initialized with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}. Summaries disabled.")
                self.enabled = False
                self.client = None
        else:
            self.client = None
            if not groq_api_key:
                logger.info("No Groq API key provided. Summaries disabled.")
    
    async def summarize(self, text: str, url: str) -> str:
        """
        Generate a business-focused summary of page content
        
        Args:
            text: Page content to summarize
            url: URL of the page (for context)
            
        Returns:
            Summary string or empty string if disabled/failed
        """
        if not self.enabled or not self.client or not text.strip():
            return ""
        
        try:
            # Truncate input to manageable size
            truncated_text = text[:3500] if len(text) > 3500 else text
            
            # Debug: Print the text blob that Groq will see
            print(f"\n{'='*60}")
            print(f"DEBUG: Text blob for URL: {url}")
            print(f"{'='*60}")
            print(f"Raw text length: {len(text)} characters")
            print(f"Truncated length: {len(truncated_text)} characters")
            print(f"First 500 chars: {repr(truncated_text[:500])}")
            print(f"{'='*60}\n")
            
            # Simple, focused prompt
            prompt = f"""Analyze this webpage and provide a concise business summary in 2-3 sentences. Focus on: 1) What specific services or features are offered, 2) Who the target audience is, 3) What actions users can take on this page.

URL: {url}
Content: {truncated_text}

Business Summary:"""

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.choices[0].message.content.strip()
            logger.debug(f"Generated summary for {url}: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate summary for {url}: {e}")
            return ""
    
    async def generate_response(
        self, 
        content: str, 
        semantic_path: List[str], 
        url: str,
        domain_type: str = "general"
    ) -> str:
        """
        Generate a chatbot response from page content (100 words max)
        
        Args:
            content: Clean main content from page
            semantic_path: Navigation path (e.g., ["Home", "Benefits", "Health Screening"])
            url: Page URL
            domain_type: Domain category (e-commerce, healthcare, etc.)
            
        Returns:
            Formatted response string (≤100 words)
        """
        # DEBUG: Show actual reasons for fallback instead of generic response
        if not self.enabled:
            return f"ERROR: Response generator disabled - no Groq API key provided"
        if not self.client:
            return f"ERROR: Groq client not initialized - API key issue"
        if not content.strip():
            return f"ERROR: No content provided - content extraction failed for {url}"
        
        try:
            # Prepare context for response generation
            path_context = " → ".join(semantic_path) if semantic_path else "Main Page"
            page_topic = semantic_path[-1] if semantic_path else "information"
            
            # Domain-specific response styling
            style_instructions = self._get_domain_style(domain_type)
            
            # Truncate content for processing
            clean_content = content[:2000] if len(content) > 2000 else content
            
            # Create response-focused prompt
            prompt = f"""You are a helpful chatbot assistant. Generate a direct, informative response about this topic.

Context:
- Page: {path_context}
- Topic: {page_topic}
- Domain: {domain_type}
- URL: {url}

Content to explain:
{clean_content}

Requirements:
- Write EXACTLY 100 words or less
- Answer as if responding to: "What is {page_topic}?" or "Tell me about {page_topic}"
- {style_instructions}
- Be specific about what services/benefits are offered
- Include actionable information (links, next steps, requirements)
- Write in second person ("you can", "your", etc.)
- Be conversational but professional

Response:"""

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=150,  # Allow some buffer for 100 words
                temperature=0.2,  # Keep responses consistent
                messages=[
                    {"role": "system", "content": "You are a helpful customer service chatbot. Always stay within word limits and provide actionable information."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Validate and clean response
            validated_response = self._validate_response(generated_response, page_topic, url)
            
            logger.debug(f"Generated response for {page_topic}: {len(validated_response.split())} words")
            return validated_response
            
        except Exception as e:
            logger.warning(f"Failed to generate response for {url}: {e}")
            return f"ERROR: Response generation failed for {url} - {str(e)[:100]}..."
    
    def _get_domain_style(self, domain_type: str) -> str:
        """Get domain-specific response styling instructions"""
        styles = {
            "e-commerce": "Focus on products, prices, availability, and how to purchase",
            "healthcare": "Use professional medical tone, mention insurance/coverage, be empathetic",
            "education": "Focus on courses, enrollment, requirements, and academic benefits", 
            "government": "Use formal tone, mention requirements, deadlines, and official processes",
            "finance": "Be precise about fees, requirements, and regulatory compliance",
            "technology": "Explain technical concepts clearly, mention features and capabilities"
        }
        return styles.get(domain_type, "Provide clear, helpful information about the topic")
    
    def _validate_response(self, response: str, topic: str, url: str) -> str:
        """Validate and clean the generated response"""
        if not response:
            return f"Information about {topic} is available on this page."
        
        # Clean up common LLM artifacts
        response = re.sub(r'^(Response:|Answer:|Summary:)\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # Check word count and truncate if needed
        words = response.split()
        if len(words) > 100:
            response = ' '.join(words[:100]) + "..."
            logger.info(f"Truncated response to 100 words for {topic}")
        
        # Ensure it ends properly
        if not response.endswith(('.', '!', '?', '...')):
            response += "."
        
        # Add URL reference if it seems helpful and there's space
        if len(words) < 90 and url:
            response += f" For more details, visit: {url}"
        
        return response
    
    def _generate_fallback_response(self, semantic_path: List[str], url: str) -> str:
        """Generate a basic fallback response when LLM fails"""
        if semantic_path:
            topic = semantic_path[-1]
            path_str = " → ".join(semantic_path)
            return f"Information about {topic} is available in the {path_str} section. Visit {url} for complete details and requirements."
        else:
            return f"This page contains important information and resources. Visit {url} for full details."
    
    async def batch_generate_responses(
        self, 
        content_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple content items efficiently
        
        Args:
            content_items: List of dicts with keys: content, semantic_path, url, domain_type
            
        Returns:
            List of enhanced items with 'response' field added
        """
        enhanced_items = []
        
        for item in content_items:
            try:
                response = await self.generate_response(
                    content=item.get('content', ''),
                    semantic_path=item.get('semantic_path', []),
                    url=item.get('url', ''),
                    domain_type=item.get('domain_type', 'general')
                )
                
                enhanced_item = item.copy()
                enhanced_item['response'] = response
                enhanced_item['response_word_count'] = len(response.split())
                enhanced_items.append(enhanced_item)
                
            except Exception as e:
                logger.error(f"Failed to process item {item.get('url', 'unknown')}: {e}")
                # Add item with error details instead of generic fallback
                enhanced_item = item.copy()
                enhanced_item['response'] = f"ERROR: Batch processing failed for {item.get('url', 'unknown')} - {str(e)[:100]}..."
                enhanced_item['response_word_count'] = len(enhanced_item['response'].split())
                enhanced_items.append(enhanced_item)
        
        return enhanced_items