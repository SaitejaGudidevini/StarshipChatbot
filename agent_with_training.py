"""
Enhanced Agentic Web Navigator with Golden Image Training Generation
Extends the base agent to include training phrase generation
"""

import asyncio
import json
import logging
import os
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent import AgenticWebNavigator
from golden_training_generator import UniversalPhraseGenerator, DomainType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedWebNavigator(AgenticWebNavigator):
    """
    Extended navigator that adds golden image based training phrase generation
    """
    
    def __init__(
        self,
        start_url: str,
        groq_api_key: Optional[str] = None,
        generate_training: bool = True,
        validate_golden_image: bool = True,
        **kwargs
    ):
        """
        Initialize enhanced navigator with training capabilities
        
        Args:
            start_url: Website to navigate
            groq_api_key: API key for Groq LLM
            generate_training: Whether to generate training phrases
            validate_golden_image: Whether to validate against golden image standards
            **kwargs: Additional arguments for parent class
        """
        super().__init__(start_url, groq_api_key, **kwargs)
        
        self.generate_training = generate_training
        self.validate_golden_image = validate_golden_image
        
        if self.generate_training:
            self.training_generator = UniversalPhraseGenerator(groq_api_key=groq_api_key)
            logger.info("Training phrase generation enabled with golden image validation")
    
    async def navigate_with_training(self) -> Dict:
        """
        Navigate website and generate training phrases validated against golden images
        """
        # Step 1: Perform standard navigation
        logger.info(f"Step 1: Navigating {self.start_url}")
        navigation_results = await self.navigate()
        
        if not self.generate_training:
            return navigation_results
        
        # Step 2: Generate training phrases with golden image validation
        logger.info(f"Step 2: Generating training phrases for {len(self.navigation_paths)} URLs")
        logger.info("Using golden image validation for quality assurance")
        
        # Process navigation data through training generator
        enhanced_results = await self.training_generator.process_navigation_data(
            navigation_results,
            sample_size=None  # Process all URLs
        )
        
        # Combine navigation and training results
        final_results = {
            "metadata": enhanced_results["metadata"],
            "navigation_paths": navigation_results["navigation_paths"],
            "training_data": enhanced_results["training_data"],
            "quality_report": enhanced_results["quality_report"]
        }
        
        # Log quality report
        self._log_quality_report(enhanced_results["quality_report"])
        
        return final_results
    
    def _log_quality_report(self, report: Dict):
        """Log the quality report for training generation"""
        logger.info("="*50)
        logger.info("TRAINING GENERATION QUALITY REPORT")
        logger.info("="*50)
        logger.info(f"Total paths processed: {report['total_paths']}")
        logger.info(f"Paths meeting golden image: {report['paths_meeting_golden_image']}")
        logger.info(f"Quality compliance: {report['paths_meeting_golden_image']/report['total_paths']*100:.1f}%")
        logger.info(f"Total phrases generated: {report['total_phrases_generated']}")
        logger.info(f"Average quality score: {report['average_quality_score']:.2f}")
        logger.info("="*50)
    
    def save_enhanced_results(self, results: Dict, output_dir: str = "output") -> str:
        """
        Save complete enhanced results in a single file
        
        Returns:
            Path to the saved file
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_clean = self.crawler.domain.replace(".", "_")
        
        # Save complete enhanced data in one file
        complete_file = f"{output_dir}/complete_{domain_clean}_{timestamp}.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to: {complete_file}")
        return complete_file
    
    def _extract_training_data(self, results: Dict) -> Dict:
        """Extract training data in a format ready for chatbot training"""
        training_data = {
            "metadata": {
                "domain": self.crawler.domain,
                "timestamp": datetime.now().isoformat(),
                "golden_image_type": results.get("metadata", {}).get("golden_image_type", "unknown"),
                "total_urls": len(results.get("training_data", {}))
            },
            "training_examples": []
        }
        
        # Extract all training phrases with their contexts
        for url, data in results.get("training_data", {}).items():
            for phrase in data.get("training_phrases", []):
                training_data["training_examples"].append({
                    "text": phrase,
                    "intent": f"navigate_to_{data.get('target', 'page').replace(' ', '_')}",
                    "entities": {
                        "target": data.get("target", ""),
                        "semantic_path": data.get("semantic_path", []),
                        "url": url
                    }
                })
        
        training_data["metadata"]["total_examples"] = len(training_data["training_examples"])
        
        return training_data
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate a human-readable summary report"""
        lines = []
        lines.append("="*60)
        lines.append("ENHANCED WEB NAVIGATION SUMMARY")
        lines.append("="*60)
        
        # Domain information
        domain_analysis = results.get("metadata", {}).get("domain_analysis", {})
        lines.append(f"\nDomain: {self.crawler.domain}")
        lines.append(f"Domain Type: {domain_analysis.get('domain_type', 'unknown')}")
        lines.append(f"Golden Image Type: {results.get('metadata', {}).get('golden_image_type', 'unknown')}")
        
        # Navigation statistics
        lines.append(f"\nNavigation Statistics:")
        lines.append(f"  URLs Discovered: {len(results.get('navigation_paths', {}))}")
        lines.append(f"  Max Depth Reached: {results.get('metadata', {}).get('max_depth_reached', 0)}")
        
        # Training generation statistics
        quality_report = results.get("quality_report", {})
        lines.append(f"\nTraining Generation Statistics:")
        lines.append(f"  Total Phrases Generated: {quality_report.get('total_phrases_generated', 0)}")
        lines.append(f"  Paths Meeting Golden Image: {quality_report.get('paths_meeting_golden_image', 0)}/{quality_report.get('total_paths', 0)}")
        lines.append(f"  Average Quality Score: {quality_report.get('average_quality_score', 0):.2f}")
        
        # Sample training phrases
        lines.append(f"\nSample Training Phrases:")
        sample_count = 0
        for url, data in list(results.get("training_data", {}).items())[:3]:
            if sample_count >= 3:
                break
            lines.append(f"\nFor: {' → '.join(data.get('semantic_path', []))}")
            for phrase in data.get("training_phrases", [])[:3]:
                lines.append(f"  • {phrase}")
            sample_count += 1
        
        # Quality gaps if any
        has_gaps = False
        for url, data in results.get("training_data", {}).items():
            if not data.get("meets_golden_image", True):
                if not has_gaps:
                    lines.append(f"\nQuality Improvements Needed:")
                    has_gaps = True
                lines.append(f"  {data.get('target', url)}: {', '.join(data.get('quality_gaps', []))}")
                if has_gaps and sample_count >= 3:
                    break
                sample_count += 1
        
        lines.append("\n" + "="*60)
        
        return "\n".join(lines)


async def main():
    """Main execution function"""
    # Configuration from environment
    start_url = os.getenv("START_URL", "https://www.mhsindiana.com/")
    groq_api_key = os.getenv("GROQ_API_KEY")
    max_depth = int(os.getenv("MAX_DEPTH", 2))
    max_pages = int(os.getenv("MAX_PAGES", 20))
    
    print("\n" + "="*60)
    print("ENHANCED AGENTIC WEB NAVIGATOR")
    print("with Golden Image Training Generation")
    print("="*60)
    
    # Create enhanced navigator
    navigator = EnhancedWebNavigator(
        start_url=start_url,
        groq_api_key=groq_api_key,
        max_depth=max_depth,
        max_pages=max_pages,
        generate_training=True,
        validate_golden_image=True,
        headless=True
    )
    
    print(f"\nConfiguration:")
    print(f"  URL: {start_url}")
    print(f"  Max Depth: {max_depth}")
    print(f"  Max Pages: {max_pages}")
    print(f"  Golden Image Validation: Enabled")
    print(f"  LLM: {'Enabled' if groq_api_key else 'Disabled (using templates only)'}")
    
    try:
        # Navigate and generate training data
        print("\nStarting enhanced navigation with training generation...")
        results = await navigator.navigate_with_training()
        
        # Save results to single file
        output_file = navigator.save_enhanced_results(results)
        
        # Generate and display summary
        summary = navigator.generate_summary_report(results)
        print(summary)
        
        print(f"\nFile saved: {output_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())