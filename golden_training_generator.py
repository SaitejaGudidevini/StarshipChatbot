"""
Golden Image Based Universal Training Phrase Generator
Domain-agnostic system that generates high-quality training phrases
validated against golden image standards
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, field
from enum import Enum
import random

from groq import Groq

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Enumeration of supported domain types"""
    ECOMMERCE = "e-commerce"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    GOVERNMENT = "government"
    FINANCE = "finance"
    TRAVEL = "travel"
    FOOD = "food"
    TECHNOLOGY = "technology"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"


@dataclass
class GoldenImage:
    """Represents the golden image (ideal output) for a domain type"""
    domain_type: DomainType
    min_phrases: int
    max_phrases: int
    required_intents: List[str]
    phrase_patterns: Dict[str, List[str]]
    quality_metrics: Dict[str, Any]
    terminology_style: str
    includes_variations: Dict[str, bool]
    
    def validate_output(self, generated_phrases: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate generated phrases against this golden image
        Returns: (passes_validation, list_of_gaps)
        """
        gaps = []
        all_phrases = generated_phrases.get("training_phrases", [])
        
        # Check phrase count
        if len(all_phrases) < self.min_phrases:
            gaps.append(f"Too few phrases: {len(all_phrases)} < {self.min_phrases}")
        elif len(all_phrases) > self.max_phrases:
            gaps.append(f"Too many phrases: {len(all_phrases)} > {self.max_phrases}")
        
        # Check required intents
        covered_intents = set(generated_phrases.get("intent_classification", {}).keys())
        missing_intents = set(self.required_intents) - covered_intents
        if missing_intents:
            gaps.append(f"Missing intents: {', '.join(missing_intents)}")
        
        # Check for variations
        if self.includes_variations.get("typos", False):
            has_typos = any(self._looks_like_typo(p) for p in all_phrases)
            if not has_typos:
                gaps.append("No phrases with natural typos/casual speech")
        
        if self.includes_variations.get("emotional", False):
            has_emotion = any(self._has_emotional_context(p) for p in all_phrases)
            if not has_emotion:
                gaps.append("No phrases with emotional context")
        
        # Check naturalness score
        naturalness = self._calculate_naturalness(all_phrases)
        if naturalness < self.quality_metrics.get("min_naturalness", 0.7):
            gaps.append(f"Low naturalness score: {naturalness:.2f}")
        
        return len(gaps) == 0, gaps
    
    def _looks_like_typo(self, phrase: str) -> bool:
        """Check if phrase contains intentional typos or casual speech"""
        typo_indicators = ["cant", "wont", "dont", "wheres", "hows", "umm", "uhh", "plz", "pls", "u", "ur"]
        return any(indicator in phrase.lower() for indicator in typo_indicators)
    
    def _has_emotional_context(self, phrase: str) -> bool:
        """Check if phrase contains emotional context"""
        emotion_indicators = ["urgent", "quickly", "frustrated", "confused", "happy", "please help", 
                             "can't find", "lost", "need help", "asap", "right now", "immediately"]
        return any(indicator in phrase.lower() for indicator in emotion_indicators)
    
    def _calculate_naturalness(self, phrases: List[str]) -> float:
        """Calculate how natural the phrases sound (0-1 score)"""
        natural_score = 0.0
        
        for phrase in phrases:
            # Check for conversational markers
            if any(marker in phrase.lower() for marker in ["i need", "i want", "can you", "do you", "where"]):
                natural_score += 1
            # Check for complete sentences
            if phrase[0].isupper() and phrase[-1] in ".?!":
                natural_score += 0.5
            # Check for variety in length
            if 3 <= len(phrase.split()) <= 15:
                natural_score += 0.5
        
        return min(natural_score / len(phrases), 1.0) if phrases else 0.0


class GoldenImageLibrary:
    """Library of golden images for different domain types"""
    
    def __init__(self):
        self.golden_images = self._initialize_golden_images()
    
    def _initialize_golden_images(self) -> Dict[DomainType, GoldenImage]:
        """Initialize golden images for each domain type"""
        return {
            DomainType.ECOMMERCE: GoldenImage(
                domain_type=DomainType.ECOMMERCE,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "availability", "price", "features", "location"],
                phrase_patterns={
                    "navigation": ["Where is/are {item}", "How to find {item}", "Show me {item}"],
                    "availability": ["Do you have {item}", "Is {item} in stock", "Are {item} available"],
                    "price": ["How much is {item}", "Price of {item}", "Cost of {item}"],
                    "features": ["What {item} do you have", "Types of {item}", "Brands of {item}"]
                },
                quality_metrics={"min_naturalness": 0.8, "intent_coverage": 0.9},
                terminology_style="casual",
                includes_variations={"typos": True, "emotional": True, "brands": True}
            ),
            
            DomainType.HEALTHCARE: GoldenImage(
                domain_type=DomainType.HEALTHCARE,
                min_phrases=7,
                max_phrases=7,
                required_intents=["appointment", "symptoms", "doctor", "insurance", "emergency"],
                phrase_patterns={
                    "appointment": ["Book appointment for {condition}", "Schedule {service}"],
                    "symptoms": ["I have {symptom}", "What to do for {symptom}"],
                    "doctor": ["Which doctor for {condition}", "Find specialist for {condition}"],
                    "insurance": ["Is {service} covered", "Insurance for {treatment}"]
                },
                quality_metrics={"min_naturalness": 0.85, "includes_urgency": True},
                terminology_style="professional",
                includes_variations={"typos": False, "emotional": True, "urgency": True}
            ),
            
            DomainType.EDUCATION: GoldenImage(
                domain_type=DomainType.EDUCATION,
                min_phrases=7,
                max_phrases=7,
                required_intents=["enrollment", "courses", "schedule", "resources", "information"],
                phrase_patterns={
                    "enrollment": ["How to enroll in {course}", "Register for {course}"],
                    "courses": ["What {subject} courses", "Prerequisites for {course}"],
                    "schedule": ["When does {course} start", "Class times for {course}"],
                    "resources": ["Materials for {course}", "Textbook for {course}"]
                },
                quality_metrics={"min_naturalness": 0.75, "formality_range": "mixed"},
                terminology_style="academic",
                includes_variations={"typos": True, "emotional": False, "academic_terms": True}
            ),
            
            DomainType.TECHNOLOGY: GoldenImage(
                domain_type=DomainType.TECHNOLOGY,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {item}", "Navigate to {item}", "Find {item}"],
                    "information": ["Tell me about {item}", "What is {item}", "Explain {item}"],
                    "help": ["Help with {item}", "I need {item}", "Looking for {item}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="technical",
                includes_variations={"typos": True, "emotional": False, "technical_terms": True}
            ),
            
            DomainType.FINANCE: GoldenImage(
                domain_type=DomainType.FINANCE,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {item}", "Navigate to {item}", "Find {item}"],
                    "information": ["Tell me about {item}", "What is {item}", "Explain {item}"],
                    "help": ["Help with {item}", "I need {item}", "Looking for {item}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="professional",
                includes_variations={"typos": True, "emotional": False, "financial_terms": True}
            ),
            
            DomainType.TRAVEL: GoldenImage(
                domain_type=DomainType.TRAVEL,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {item}", "Navigate to {item}", "Find {item}"],
                    "information": ["Tell me about {item}", "What is {item}", "Explain {item}"],
                    "help": ["Help with {item}", "I need {item}", "Looking for {item}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="casual",
                includes_variations={"typos": True, "emotional": True, "travel_terms": True}
            ),
            
            DomainType.FOOD: GoldenImage(
                domain_type=DomainType.FOOD,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {item}", "Navigate to {item}", "Find {item}"],
                    "information": ["Tell me about {item}", "What is {item}", "Explain {item}"],
                    "help": ["Help with {item}", "I need {item}", "Looking for {item}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="casual",
                includes_variations={"typos": True, "emotional": True, "food_terms": True}
            ),
            
            DomainType.GOVERNMENT: GoldenImage(
                domain_type=DomainType.GOVERNMENT,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {item}", "Navigate to {item}", "Find {item}"],
                    "information": ["Tell me about {item}", "What is {item}", "Explain {item}"],
                    "help": ["Help with {item}", "I need {item}", "Looking for {item}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="formal",
                includes_variations={"typos": False, "emotional": False, "government_terms": True}
            ),
            
            DomainType.ENTERTAINMENT: GoldenImage(
                domain_type=DomainType.ENTERTAINMENT,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {item}", "Navigate to {item}", "Find {item}"],
                    "information": ["Tell me about {item}", "What is {item}", "Explain {item}"],
                    "help": ["Help with {item}", "I need {item}", "Looking for {item}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="casual",
                includes_variations={"typos": True, "emotional": True, "entertainment_terms": True}
            ),
            
            DomainType.GENERAL: GoldenImage(
                domain_type=DomainType.GENERAL,
                min_phrases=7,
                max_phrases=7,
                required_intents=["navigation", "information", "help"],
                phrase_patterns={
                    "navigation": ["Where is {page}", "Go to {section}", "Find {content}"],
                    "information": ["Tell me about {topic}", "What is {subject}"],
                    "help": ["Help with {task}", "How to {action}"]
                },
                quality_metrics={"min_naturalness": 0.7},
                terminology_style="neutral",
                includes_variations={"typos": True, "emotional": False, "brands": False}
            )
        }
    
    def get_golden_image(self, domain_type: DomainType) -> GoldenImage:
        """Get golden image for a specific domain type"""
        return self.golden_images.get(domain_type, self.golden_images[DomainType.GENERAL])
    
    def identify_domain_type(self, url: str, semantic_paths: List[List[str]]) -> DomainType:
        """Identify domain type from URL and semantic paths"""
        # Analyze URL
        domain = urlparse(url).netloc.lower()
        path_text = " ".join([" ".join(path).lower() for path in semantic_paths[:5]])
        
        # Check for domain indicators
        if any(term in domain + path_text for term in ["shop", "store", "product", "cart", "checkout", "price"]):
            return DomainType.ECOMMERCE
        elif any(term in domain + path_text for term in ["health", "medical", "doctor", "clinic", "hospital", "treatment"]):
            return DomainType.HEALTHCARE
        elif any(term in domain + path_text for term in ["course", "class", "student", "education", "learn", "school"]):
            return DomainType.EDUCATION
        elif any(term in domain + path_text for term in [".gov", "government", "service", "permit", "license"]):
            return DomainType.GOVERNMENT
        elif any(term in domain + path_text for term in ["bank", "finance", "loan", "investment", "account"]):
            return DomainType.FINANCE
        elif any(term in domain + path_text for term in ["travel", "flight", "hotel", "booking", "destination"]):
            return DomainType.TRAVEL
        elif any(term in domain + path_text for term in ["restaurant", "food", "menu", "order", "delivery", "cuisine"]):
            return DomainType.FOOD
        else:
            return DomainType.GENERAL


class UniversalPhraseGenerator:
    """
    Universal training phrase generator that works with any domain
    Uses golden images for quality validation
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.golden_library = GoldenImageLibrary()
        self.domain_cache = {}
    
    async def analyze_domain(self, url: str, semantic_paths: List[List[str]]) -> Dict[str, Any]:
        """Analyze domain using LLM for deep understanding"""
        if not self.groq_client:
            # Fallback to rule-based identification
            domain_type = self.golden_library.identify_domain_type(url, semantic_paths)
            return {
                "domain_type": domain_type.value,
                "confidence": 0.7,
                "key_entities": [],
                "terminology_style": "neutral"
            }
        
        # Check cache
        domain = urlparse(url).netloc
        if domain in self.domain_cache:
            return self.domain_cache[domain]
        
        # Prepare context for LLM
        sample_paths = semantic_paths[:10] if len(semantic_paths) > 10 else semantic_paths
        paths_text = "\n".join([" → ".join(path) for path in sample_paths])
        
        prompt = f"""Analyze this website and identify its characteristics:

URL: {url}
Navigation Paths:
{paths_text}

Identify:
1. Domain type (e-commerce, healthcare, education, government, finance, travel, food, technology, entertainment, general)
2. Key entities or products (max 5)
3. Communication style (formal, casual, technical, professional)
4. Common user goals on this site

Respond in JSON format:
{{
    "domain_type": "type here",
    "key_entities": ["entity1", "entity2"],
    "terminology_style": "style here",
    "user_goals": ["goal1", "goal2"]
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a domain analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            self.domain_cache[domain] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Error in domain analysis: {e}")
            # Fallback
            domain_type = self.golden_library.identify_domain_type(url, semantic_paths)
            return {"domain_type": domain_type.value, "confidence": 0.5}
    
    async def generate_phrases(
        self,
        url: str,
        semantic_path: List[str],
        domain_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate training phrases for a semantic path"""
        
        # Get domain type and golden image
        if not domain_analysis:
            domain_analysis = await self.analyze_domain(url, [semantic_path])
        
        domain_type = DomainType(domain_analysis.get("domain_type", "general"))
        golden_image = self.golden_library.get_golden_image(domain_type)
        
        # Extract target item
        target = semantic_path[-1] if semantic_path else "page"
        parent = semantic_path[-2] if len(semantic_path) > 1 else None
        
        # Generate phrases using multiple strategies
        phrases = []
        
        # 1. Template-based generation
        template_phrases = self._generate_from_templates(target, parent, golden_image)
        phrases.extend(template_phrases)
        
        # 2. LLM-based generation (if available)
        if self.groq_client:
            llm_phrases = await self._generate_with_llm(
                semantic_path, url, golden_image, domain_analysis
            )
            phrases.extend(llm_phrases)
        
        # 3. Variation generation
        variation_phrases = self._generate_variations(target, golden_image)
        phrases.extend(variation_phrases)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                unique_phrases.append(phrase)
        
        # Classify phrases by intent
        classified = self._classify_intents(unique_phrases, golden_image.required_intents)
        
        # Validate against golden image
        result = {
            "url": url,
            "semantic_path": semantic_path,
            "target": target,
            "domain_type": domain_type.value,
            "training_phrases": unique_phrases[:golden_image.max_phrases],
            "intent_classification": classified,
            "total_phrases": len(unique_phrases)
        }
        
        # Check validation
        passes, gaps = golden_image.validate_output(result)
        result["meets_golden_image"] = passes
        result["quality_gaps"] = gaps
        
        # If doesn't meet golden image, try to fill gaps
        if not passes and "Too few phrases" in str(gaps):
            additional = self._generate_gap_fillers(target, parent, golden_image, gaps)
            result["training_phrases"].extend(additional)
            result["total_phrases"] = len(result["training_phrases"])
            # Re-validate
            passes, gaps = golden_image.validate_output(result)
            result["meets_golden_image"] = passes
            result["quality_gaps"] = gaps
        
        return result
    
    def _generate_from_templates(
        self,
        target: str,
        parent: Optional[str],
        golden_image: GoldenImage
    ) -> List[str]:
        """Generate phrases from golden image templates"""
        phrases = []
        
        for intent, templates in golden_image.phrase_patterns.items():
            for template in templates:
                if "{item}" in template:
                    phrases.append(template.replace("{item}", target))
                if "{condition}" in template:
                    phrases.append(template.replace("{condition}", target))
                if "{service}" in template:
                    phrases.append(template.replace("{service}", target))
                if "{course}" in template:
                    phrases.append(template.replace("{course}", target))
                if "{page}" in template:
                    phrases.append(template.replace("{page}", target))
        
        # Add parent context phrases
        if parent:
            phrases.extend([
                f"Where is {target} in {parent}?",
                f"Show me {target} under {parent}",
                f"I'm looking for {target} in the {parent} section",
                f"Navigate to {target} from {parent}"
            ])
        
        return phrases
    
    async def _generate_with_llm(
        self,
        semantic_path: List[str],
        url: str,
        golden_image: GoldenImage,
        domain_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate phrases using LLM"""
        if not self.groq_client:
            return []
        
        path_str = " → ".join(semantic_path)
        target = semantic_path[-1]
        
        # Build domain-aware prompt
        prompt = f"""Generate natural training phrases for a {golden_image.domain_type.value} chatbot.

Navigation path: {path_str}
Target: {target}
URL: {url}
Domain style: {golden_image.terminology_style}

Generate 7 diverse phrases that real users would say to navigate to or ask about "{target}".

Requirements:
- Include these intents: {', '.join(golden_image.required_intents[:3])}
- Mix formal and informal language
- Include some with typos or casual speech
- Include emotional context (frustrated, urgent, confused)
- Make them sound natural and conversational

Generate exactly 7 phrases, one per line:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at generating natural user queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.8
            )
            
            # Parse response
            lines = response.choices[0].message.content.strip().split('\n')
            phrases = []
            for line in lines:
                cleaned = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip().strip('"\'')
                if cleaned and len(cleaned) > 3:
                    phrases.append(cleaned)
            
            return phrases[:7]
            
        except Exception as e:
            logger.error(f"Error generating LLM phrases: {e}")
            return []
    
    def _generate_variations(self, target: str, golden_image: GoldenImage) -> List[str]:
        """Generate variations based on golden image requirements"""
        variations = []
        
        # Add typos/casual speech if required
        if golden_image.includes_variations.get("typos", False):
            variations.extend([
                f"wheres the {target}",
                f"cant find {target}",
                f"i need {target} plz",
                f"do u have {target}",
                f"umm looking for {target}"
            ])
        
        # Add emotional context if required
        if golden_image.includes_variations.get("emotional", False):
            variations.extend([
                f"I urgently need {target}",
                f"Can't find {target} anywhere, help!",
                f"Getting frustrated, where is {target}?",
                f"Please help me find {target}",
                f"I'm lost, need {target}"
            ])
        
        # Add domain-specific variations
        if golden_image.domain_type == DomainType.ECOMMERCE:
            variations.extend([
                f"Is {target} on sale?",
                f"Best {target} you have?",
                f"Cheapest {target} available?",
                f"Reviews for {target}?"
            ])
        elif golden_image.domain_type == DomainType.HEALTHCARE:
            variations.extend([
                f"Emergency {target} needed",
                f"Is {target} covered by insurance?",
                f"Appointment for {target} today?",
                f"Specialist for {target}?"
            ])
        
        return variations
    
    def _classify_intents(self, phrases: List[str], required_intents: List[str]) -> Dict[str, List[str]]:
        """Classify phrases by intent"""
        classified = {intent: [] for intent in required_intents}
        
        # Intent keywords mapping
        intent_keywords = {
            "navigation": ["where", "find", "locate", "show", "navigate", "go to"],
            "availability": ["have", "available", "stock", "sell", "carry", "offer"],
            "price": ["cost", "price", "how much", "expensive", "cheap", "sale"],
            "information": ["what", "tell", "about", "explain", "details", "info"],
            "appointment": ["book", "schedule", "appointment", "availability", "slot"],
            "symptoms": ["feel", "pain", "symptom", "suffering", "experiencing"],
            "enrollment": ["enroll", "register", "join", "sign up", "apply"],
            "help": ["help", "assist", "support", "can't", "problem", "issue"]
        }
        
        # Classify each phrase
        for phrase in phrases:
            phrase_lower = phrase.lower()
            classified_to_intent = False
            
            for intent in required_intents:
                if intent in intent_keywords:
                    if any(kw in phrase_lower for kw in intent_keywords[intent]):
                        classified[intent].append(phrase)
                        classified_to_intent = True
                        break
            
            # Default to first intent if not classified
            if not classified_to_intent and required_intents:
                classified[required_intents[0]].append(phrase)
        
        return classified
    
    def _generate_gap_fillers(
        self,
        target: str,
        parent: Optional[str],
        golden_image: GoldenImage,
        gaps: List[str]
    ) -> List[str]:
        """Generate additional phrases to fill identified gaps"""
        fillers = []
        
        # Analyze gaps and generate targeted phrases
        for gap in gaps:
            if "Missing intents" in gap:
                # Extract missing intents
                missing = gap.split(": ")[1].split(", ")
                for intent in missing:
                    if intent == "navigation":
                        fillers.extend([f"How do I get to {target}?", f"Where is {target} located?"])
                    elif intent == "availability":
                        fillers.extend([f"Do you have {target}?", f"Is {target} available?"])
                    elif intent == "price":
                        fillers.extend([f"How much does {target} cost?", f"Price for {target}?"])
            
            elif "typos" in gap.lower():
                fillers.extend([
                    f"were can i find {target}",
                    f"need {target} quik",
                    f"lookin for {target}"
                ])
            
            elif "emotional" in gap.lower():
                fillers.extend([
                    f"Really need {target} urgently!",
                    f"So confused, where's {target}?",
                    f"Happy to finally find {target}"
                ])
        
        return fillers
    
    async def process_navigation_data(
        self,
        navigation_data: Dict[str, Any],
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process entire navigation data to generate training phrases"""
        
        # Extract all paths for domain analysis
        all_paths = []
        for path_data in navigation_data.get("navigation_paths", {}).values():
            all_paths.append(path_data.get("semantic_path", []))
        
        # Analyze domain once
        first_url = list(navigation_data.get("navigation_paths", {}).keys())[0] if navigation_data.get("navigation_paths") else ""
        domain_analysis = await self.analyze_domain(first_url, all_paths)
        
        # Determine domain type
        domain_type = DomainType(domain_analysis.get("domain_type", "general"))
        golden_image = self.golden_library.get_golden_image(domain_type)
        
        # Process results
        enhanced_data = {
            "metadata": {
                **navigation_data.get("metadata", {}),
                "domain_analysis": domain_analysis,
                "golden_image_type": domain_type.value,
                "quality_standards": {
                    "min_phrases": golden_image.min_phrases,
                    "max_phrases": golden_image.max_phrases,
                    "required_intents": golden_image.required_intents
                }
            },
            "training_data": {},
            "quality_report": {
                "total_paths": 0,
                "paths_meeting_golden_image": 0,
                "total_phrases_generated": 0,
                "average_quality_score": 0.0
            }
        }
        
        # Process each path
        paths_to_process = list(navigation_data.get("navigation_paths", {}).items())
        if sample_size:
            paths_to_process = paths_to_process[:sample_size]
        
        quality_scores = []
        
        for url, path_data in paths_to_process:
            # Generate training phrases
            result = await self.generate_phrases(
                url=url,
                semantic_path=path_data.get("semantic_path", []),
                domain_analysis=domain_analysis
            )
            
            enhanced_data["training_data"][url] = result
            
            # Update statistics
            enhanced_data["quality_report"]["total_paths"] += 1
            if result["meets_golden_image"]:
                enhanced_data["quality_report"]["paths_meeting_golden_image"] += 1
            enhanced_data["quality_report"]["total_phrases_generated"] += len(result["training_phrases"])
            
            # Calculate quality score
            quality_score = 1.0 if result["meets_golden_image"] else 0.5
            quality_scores.append(quality_score)
        
        # Calculate average quality
        if quality_scores:
            enhanced_data["quality_report"]["average_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        return enhanced_data


# Example usage and testing
async def main():
    """Example usage of the Golden Training Generator"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Sample navigation data
    sample_data = {
        "navigation_paths": {
            "https://www.heb.com/category/shop/health-beauty/cotton-balls-swabs": {
                "semantic_path": ["Home", "Health & Beauty", "Cotton balls & swabs"],
                "depth": 2
            },
            "https://www.heb.com/category/shop/pharmacy/vitamins": {
                "semantic_path": ["Home", "Pharmacy", "Vitamins"],
                "depth": 2
            }
        }
    }
    
    # Initialize generator
    generator = UniversalPhraseGenerator(groq_api_key=os.getenv("GROQ_API_KEY"))
    
    # Process and generate training data
    result = await generator.process_navigation_data(sample_data)
    
    # Display results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())