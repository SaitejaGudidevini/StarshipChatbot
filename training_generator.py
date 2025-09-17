"""
Training Phrase Generator for Semantic Navigation Paths
Generates conversational training phrases for chatbot training
"""

import re
import json
import random
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from groq import Groq
import logging

logger = logging.getLogger(__name__)


class TrainingPhraseGenerator:
    """
    Generates diverse training phrases for each navigation path
    using templates and LLM augmentation
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, store_name: str = "the store"):
        self.groq_client = None
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        
        self.store_name = store_name
        
        # Template categories for different intent types
        self.templates = {
            "navigation": [
                "Where can I find {item}?",
                "Where are {item} located?",
                "How do I get to {item}?",
                "Can you help me find {item}?",
                "I'm looking for {item}",
                "Show me where {item} are",
                "Take me to {item}",
                "Navigate to {item}",
                "I need to find {item}",
                "Where would {item} be?"
            ],
            "availability": [
                "Do you have {item}?",
                "Are {item} available?",
                "Do you sell {item}?",
                "Can I get {item} at {store}?",
                "Are {item} in stock?",
                "Do you carry {item}?",
                "Is there {item} available?",
                "Can I buy {item} here?",
                "Do you stock {item}?",
                "Are there any {item}?"
            ],
            "information": [
                "Tell me about {item}",
                "What {item} do you have?",
                "Show me {item} options",
                "What types of {item} are available?",
                "I need information about {item}",
                "What brands of {item} do you carry?",
                "Show me all {item}",
                "List {item} products",
                "What kind of {item} can I find?",
                "Give me details about {item}"
            ],
            "location": [
                "Which aisle has {item}?",
                "What section are {item} in?",
                "Where in the store are {item}?",
                "Which department sells {item}?",
                "What aisle number for {item}?",
                "In which section can I find {item}?",
                "Where exactly are the {item}?",
                "Point me to the {item} section",
                "Which area has {item}?",
                "Guide me to {item}"
            ],
            "specific": [
                "I need {item} for {purpose}",
                "Looking for {item} to {action}",
                "Need {item} for my {context}",
                "Want to buy {item} for {reason}",
                "Searching for {item} that can {feature}",
                "Do you have {item} that are {attribute}?",
                "I want {attribute} {item}",
                "Show me {item} under ${price}",
                "Need {quantity} {item}",
                "Looking for {brand} {item}"
            ]
        }
        
        # Context-specific attributes for different categories
        self.context_attributes = {
            "health-beauty": {
                "purposes": ["skincare", "makeup removal", "first aid", "baby care", "cleaning"],
                "attributes": ["hypoallergenic", "organic", "sensitive skin", "unscented", "biodegradable"],
                "actions": ["clean", "remove makeup", "apply medicine", "care for baby", "treat wounds"]
            },
            "grocery": {
                "purposes": ["cooking", "meal prep", "snacking", "party", "lunch"],
                "attributes": ["fresh", "organic", "local", "gluten-free", "sugar-free"],
                "actions": ["cook", "prepare meals", "pack lunch", "make dinner", "bake"]
            },
            "pharmacy": {
                "purposes": ["cold relief", "pain management", "allergies", "vitamins", "prescriptions"],
                "attributes": ["generic", "brand name", "children's", "adult", "extra strength"],
                "actions": ["treat symptoms", "manage pain", "boost immunity", "get prescription", "feel better"]
            }
        }
    
    def _extract_category_from_path(self, semantic_path: List[str], url: str) -> str:
        """Extract the general category from the semantic path or URL"""
        path_str = " ".join(semantic_path).lower()
        url_lower = url.lower()
        
        # Check for known categories
        if any(term in path_str or term in url_lower for term in ["health", "beauty", "personal", "hygiene"]):
            return "health-beauty"
        elif any(term in path_str or term in url_lower for term in ["pharmacy", "medicine", "prescription", "drug"]):
            return "pharmacy"
        elif any(term in path_str or term in url_lower for term in ["grocery", "food", "produce", "dairy", "meat"]):
            return "grocery"
        else:
            return "general"
    
    def _get_item_variations(self, item: str) -> List[str]:
        """Generate variations of the item name"""
        variations = [item]
        
        # Singular/plural variations
        if item.endswith('s'):
            variations.append(item[:-1])  # Remove 's' for singular
        else:
            variations.append(item + 's')  # Add 's' for plural
        
        # With/without ampersand
        if '&' in item:
            variations.append(item.replace('&', 'and'))
        elif ' and ' in item:
            variations.append(item.replace(' and ', ' & '))
        
        # Abbreviated versions
        if len(item.split()) > 2:
            words = item.split()
            variations.append(f"{words[0]} {words[-1]}")  # First and last word
        
        return list(set(variations))
    
    def _generate_template_phrases(
        self,
        item: str,
        category: str,
        semantic_path: List[str]
    ) -> List[str]:
        """Generate phrases using templates"""
        phrases = []
        item_variations = self._get_item_variations(item)
        
        # Get category-specific context
        context = self.context_attributes.get(category, {})
        
        for variation in item_variations:
            # Navigation phrases
            for template in random.sample(self.templates["navigation"], min(3, len(self.templates["navigation"]))):
                phrases.append(template.format(item=variation))
            
            # Availability phrases
            for template in random.sample(self.templates["availability"], min(2, len(self.templates["availability"]))):
                phrases.append(template.format(item=variation, store=self.store_name))
            
            # Information phrases
            for template in random.sample(self.templates["information"], min(2, len(self.templates["information"]))):
                phrases.append(template.format(item=variation))
            
            # Location phrases
            for template in random.sample(self.templates["location"], min(2, len(self.templates["location"]))):
                phrases.append(template.format(item=variation))
            
            # Context-specific phrases
            if context:
                for template in random.sample(self.templates["specific"], min(3, len(self.templates["specific"]))):
                    if "{purpose}" in template and context.get("purposes"):
                        purpose = random.choice(context["purposes"])
                        phrases.append(template.format(item=variation, purpose=purpose))
                    elif "{action}" in template and context.get("actions"):
                        action = random.choice(context["actions"])
                        phrases.append(template.format(item=variation, action=action))
                    elif "{attribute}" in template and context.get("attributes"):
                        attribute = random.choice(context["attributes"])
                        phrases.append(template.format(item=variation, attribute=attribute))
                    elif "{context}" in template:
                        phrases.append(template.format(item=variation, context="bathroom"))
                    elif "{reason}" in template:
                        phrases.append(template.format(item=variation, reason="home use"))
                    elif "{feature}" in template:
                        phrases.append(template.format(item=variation, feature="work well"))
                    elif "{price}" in template:
                        phrases.append(template.replace("{item}", variation).replace("${price}", "$10"))
                    elif "{quantity}" in template:
                        phrases.append(template.format(quantity="some", item=variation))
                    elif "{brand}" in template:
                        phrases.append(template.format(brand="any brand", item=variation))
        
        # Add path-based phrases
        if len(semantic_path) > 1:
            parent = semantic_path[-2] if len(semantic_path) > 1 else "products"
            phrases.append(f"Show me {item} in {parent}")
            phrases.append(f"Where are {item} in the {parent} section?")
            phrases.append(f"Take me to {item} under {parent}")
        
        return phrases
    
    async def _generate_llm_phrases(
        self,
        item: str,
        semantic_path: List[str],
        url: str,
        existing_phrases: List[str]
    ) -> List[str]:
        """Use LLM to generate additional creative phrases"""
        if not self.groq_client:
            return []
        
        try:
            # Prepare context for LLM
            path_str = " → ".join(semantic_path)
            existing_sample = random.sample(existing_phrases, min(5, len(existing_phrases)))
            
            prompt = f"""Generate 10 diverse, natural conversational phrases a customer might use when looking for "{item}" in a store.

Context:
- Navigation path: {path_str}
- Store: {self.store_name}
- URL indicates this is in: {self._extract_category_from_path(semantic_path, url)} section

Examples of style (but create different ones):
{chr(10).join(existing_sample[:3])}

Requirements:
- Natural, conversational language
- Mix of formal and informal styles
- Include some with typos or casual speech
- Some should be complete sentences, others fragments
- Include emotional context (frustrated, confused, happy, etc.)
- Some should mention related needs or contexts

Generate exactly 10 unique phrases, one per line:"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding how real customers ask for products in stores. Generate natural, diverse phrases."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.8
            )
            
            # Parse LLM response
            llm_phrases = response.choices[0].message.content.strip().split('\n')
            # Clean and filter
            llm_phrases = [p.strip().strip('"').strip("'") for p in llm_phrases if p.strip()]
            
            return llm_phrases[:10]  # Limit to 10
            
        except Exception as e:
            logger.error(f"Error generating LLM phrases: {e}")
            return []
    
    async def generate_training_phrases(
        self,
        url: str,
        semantic_path: List[str],
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training phrases for a navigation path
        
        Returns:
        {
            "url": "...",
            "semantic_path": [...],
            "primary_item": "cotton balls & swabs",
            "training_phrases": [...],
            "phrase_categories": {
                "navigation": [...],
                "availability": [...],
                "information": [...]
            }
        }
        """
        # Extract the primary item (usually the last meaningful part of the path)
        primary_item = semantic_path[-1] if semantic_path else "products"
        
        # Determine category
        category = self._extract_category_from_path(semantic_path, url)
        
        # Generate template-based phrases
        template_phrases = self._generate_template_phrases(primary_item, category, semantic_path)
        
        # Generate LLM phrases if enabled
        llm_phrases = []
        if use_llm and self.groq_client:
            llm_phrases = await self._generate_llm_phrases(
                primary_item,
                semantic_path,
                url,
                template_phrases
            )
        
        # Combine and deduplicate
        all_phrases = list(set(template_phrases + llm_phrases))
        
        # Categorize phrases for structured training
        categorized = self._categorize_phrases(all_phrases)
        
        return {
            "url": url,
            "semantic_path": semantic_path,
            "primary_item": primary_item,
            "category": category,
            "training_phrases": all_phrases,
            "phrase_categories": categorized,
            "total_phrases": len(all_phrases)
        }
    
    def _categorize_phrases(self, phrases: List[str]) -> Dict[str, List[str]]:
        """Categorize phrases by intent type"""
        categorized = {
            "navigation": [],
            "availability": [],
            "information": [],
            "location": [],
            "specific": []
        }
        
        # Keywords for categorization
        nav_keywords = ["where", "find", "looking", "navigate", "take me", "show me where"]
        avail_keywords = ["do you have", "available", "stock", "sell", "carry", "do you"]
        info_keywords = ["tell me", "what", "types", "brands", "options", "list", "show me all"]
        loc_keywords = ["aisle", "section", "department", "area", "which", "exact"]
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            
            if any(kw in phrase_lower for kw in nav_keywords):
                categorized["navigation"].append(phrase)
            elif any(kw in phrase_lower for kw in avail_keywords):
                categorized["availability"].append(phrase)
            elif any(kw in phrase_lower for kw in info_keywords):
                categorized["information"].append(phrase)
            elif any(kw in phrase_lower for kw in loc_keywords):
                categorized["location"].append(phrase)
            else:
                categorized["specific"].append(phrase)
        
        return categorized
    
    def generate_variations_with_context(
        self,
        base_phrases: List[str],
        context_modifiers: Dict[str, List[str]]
    ) -> List[str]:
        """
        Add context modifiers to base phrases
        E.g., "urgently need", "for my baby", "that's on sale"
        """
        variations = base_phrases.copy()
        
        modifiers = {
            "urgency": ["urgently", "quickly", "right now", "immediately", "asap"],
            "user_context": ["for my baby", "for my mother", "for travel", "for guests", "for work"],
            "preference": ["cheap", "best quality", "on sale", "popular", "recommended"],
            "quantity": ["a few", "lots of", "bulk", "single", "family size"]
        }
        
        for phrase in base_phrases[:10]:  # Limit to avoid explosion
            for mod_type, mod_values in modifiers.items():
                if random.random() < 0.3:  # 30% chance to add modifier
                    modifier = random.choice(mod_values)
                    if mod_type == "urgency":
                        variations.append(f"I {modifier} need {phrase.lower()}")
                    elif mod_type == "user_context":
                        variations.append(f"{phrase} {modifier}")
                    elif mod_type == "preference":
                        variations.append(phrase.replace("find", f"find {modifier}").replace("need", f"need {modifier}"))
                    elif mod_type == "quantity":
                        variations.append(phrase.replace("I need", f"I need {modifier}"))
        
        return variations


class TrainingDataEnhancer:
    """
    Enhances navigation data with training phrases
    """
    
    def __init__(self, generator: TrainingPhraseGenerator):
        self.generator = generator
    
    async def enhance_navigation_data(
        self,
        navigation_data: Dict[str, Any],
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Enhance existing navigation data with training phrases
        """
        enhanced_data = {
            "metadata": navigation_data.get("metadata", {}),
            "navigation_paths": {},
            "training_data": {
                "total_phrases_generated": 0,
                "urls_processed": 0,
                "categories_found": set()
            }
        }
        
        # Process each URL
        paths_to_process = list(navigation_data.get("navigation_paths", {}).items())
        if sample_size:
            paths_to_process = paths_to_process[:sample_size]
        
        for url, path_data in paths_to_process:
            # Generate training phrases
            enhanced_path = await self.generator.generate_training_phrases(
                url=url,
                semantic_path=path_data.get("semantic_path", []),
                use_llm=True
            )
            
            # Combine original path data with training phrases
            enhanced_path.update({
                "depth": path_data.get("depth", 0),
                "parent_url": path_data.get("parent_url")
            })
            
            enhanced_data["navigation_paths"][url] = enhanced_path
            
            # Update statistics
            enhanced_data["training_data"]["total_phrases_generated"] += len(enhanced_path["training_phrases"])
            enhanced_data["training_data"]["urls_processed"] += 1
            enhanced_data["training_data"]["categories_found"].add(enhanced_path["category"])
        
        # Convert set to list for JSON serialization
        enhanced_data["training_data"]["categories_found"] = list(enhanced_data["training_data"]["categories_found"])
        
        return enhanced_data
    
    def export_for_dialogflow(self, enhanced_data: Dict[str, Any]) -> List[Dict]:
        """
        Export training phrases in Dialogflow format
        """
        training_examples = []
        
        for url, path_data in enhanced_data["navigation_paths"].items():
            for phrase in path_data["training_phrases"]:
                training_examples.append({
                    "text": phrase,
                    "intent": f"navigate_to_{path_data['primary_item'].replace(' ', '_').replace('&', 'and')}",
                    "entities": [
                        {
                            "entity": "product",
                            "value": path_data["primary_item"]
                        },
                        {
                            "entity": "category",
                            "value": path_data["category"]
                        }
                    ],
                    "action": "navigate",
                    "parameters": {
                        "url": url,
                        "semantic_path": path_data["semantic_path"]
                    }
                })
        
        return training_examples


# Example usage
async def main():
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Example navigation data
    navigation_data = {
        "navigation_paths": {
            "https://www.heb.com/category/shop/health-beauty/cotton-balls-swabs/490021/490085": {
                "url": "https://www.heb.com/category/shop/health-beauty/cotton-balls-swabs/490021/490085",
                "semantic_path": ["Home", "Health & Beauty", "Cotton balls & swabs"],
                "depth": 2,
                "parent_url": "https://www.heb.com/category/shop/health-beauty"
            }
        }
    }
    
    # Initialize generator
    generator = TrainingPhraseGenerator(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        store_name="HEB"
    )
    
    # Generate training phrases for a single path
    result = await generator.generate_training_phrases(
        url="https://www.heb.com/category/shop/health-beauty/cotton-balls-swabs/490021/490085",
        semantic_path=["Home", "Health & Beauty", "Cotton balls & swabs"],
        use_llm=True
    )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())