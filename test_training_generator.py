"""
Test script for Golden Image Training Generator
Validates that the training generator meets quality standards
"""

import asyncio
import json
import os
from dotenv import load_dotenv

from golden_training_generator import UniversalPhraseGenerator, GoldenImageLibrary, DomainType
from agent_with_training import EnhancedWebNavigator

# Load environment variables
load_dotenv()


async def test_golden_image_validation():
    """Test golden image validation logic"""
    print("\n" + "="*60)
    print("TEST 1: Golden Image Validation")
    print("="*60)
    
    library = GoldenImageLibrary()
    
    # Test e-commerce golden image
    ecommerce_golden = library.get_golden_image(DomainType.ECOMMERCE)
    
    # Test case 1: Good output (meets golden image)
    good_output = {
        "training_phrases": [
            "Where can I find cotton balls?",
            "Do you have cotton balls?",
            "How much are cotton balls?",
            "Show me cotton balls",
            "Cotton balls in stock?",
            "Which aisle has cotton balls?",
            "I need cotton balls",
            "Looking for cotton balls",
            "Cotton balls near me",
            "Best cotton balls",
            "Cheap cotton balls",
            "Cotton balls on sale?",
            "Navigate to cotton balls",
            "Take me to cotton balls",
            "Find cotton balls",
            "Cotton balls section",
            "Do you sell cotton balls?",
            "Are cotton balls available?",
            "Cotton balls location",
            "I urgently need cotton balls",
            "cant find cotton balls"
        ],
        "intent_classification": {
            "navigation": ["Where can I find cotton balls?", "Navigate to cotton balls"],
            "availability": ["Do you have cotton balls?", "Cotton balls in stock?"],
            "price": ["How much are cotton balls?", "Cheap cotton balls"],
            "features": ["Best cotton balls"],
            "location": ["Which aisle has cotton balls?", "Cotton balls location"]
        }
    }
    
    passes, gaps = ecommerce_golden.validate_output(good_output)
    print(f"Good output validation: {'PASSED' if passes else 'FAILED'}")
    if gaps:
        print(f"  Gaps: {gaps}")
    
    # Test case 2: Bad output (doesn't meet golden image)
    bad_output = {
        "training_phrases": [
            "Cotton balls",
            "Find cotton balls",
            "Where cotton balls"
        ],
        "intent_classification": {
            "navigation": ["Find cotton balls", "Where cotton balls"]
        }
    }
    
    passes, gaps = ecommerce_golden.validate_output(bad_output)
    print(f"Bad output validation: {'PASSED' if passes else 'FAILED'}")
    print(f"  Expected gaps found: {len(gaps) > 0}")
    print(f"  Gaps: {gaps[:3]}")  # Show first 3 gaps


async def test_domain_identification():
    """Test domain type identification"""
    print("\n" + "="*60)
    print("TEST 2: Domain Identification")
    print("="*60)
    
    library = GoldenImageLibrary()
    
    test_cases = [
        {
            "url": "https://www.walmart.com/products",
            "paths": [["Home", "Products", "Electronics"], ["Home", "Grocery", "Fruits"]],
            "expected": DomainType.ECOMMERCE
        },
        {
            "url": "https://www.mayoclinic.org",
            "paths": [["Home", "Symptoms", "Headache"], ["Home", "Doctors", "Cardiology"]],
            "expected": DomainType.HEALTHCARE
        },
        {
            "url": "https://www.coursera.org",
            "paths": [["Home", "Courses", "Computer Science"], ["Home", "Enrollment", "Spring 2025"]],
            "expected": DomainType.EDUCATION
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        identified = library.identify_domain_type(test["url"], test["paths"])
        result = "✓" if identified == test["expected"] else "✗"
        print(f"Test {i}: {test['url']}")
        print(f"  Expected: {test['expected'].value}")
        print(f"  Identified: {identified.value}")
        print(f"  Result: {result}")


async def test_phrase_generation():
    """Test training phrase generation"""
    print("\n" + "="*60)
    print("TEST 3: Phrase Generation")
    print("="*60)
    
    generator = UniversalPhraseGenerator(groq_api_key=os.getenv("GROQ_API_KEY"))
    
    # Test with a sample path
    test_path = {
        "url": "https://www.heb.com/category/shop/health-beauty/cotton-balls-swabs",
        "semantic_path": ["Home", "Health & Beauty", "Cotton balls & swabs"]
    }
    
    print(f"Generating phrases for: {' → '.join(test_path['semantic_path'])}")
    
    result = await generator.generate_phrases(
        url=test_path["url"],
        semantic_path=test_path["semantic_path"]
    )
    
    print(f"\nResults:")
    print(f"  Domain Type: {result['domain_type']}")
    print(f"  Total Phrases: {result['total_phrases']}")
    print(f"  Meets Golden Image: {'✓' if result['meets_golden_image'] else '✗'}")
    
    if not result['meets_golden_image']:
        print(f"  Quality Gaps: {result['quality_gaps']}")
    
    print(f"\nSample Phrases (first 5):")
    for phrase in result['training_phrases'][:5]:
        print(f"  • {phrase}")
    
    print(f"\nIntent Distribution:")
    for intent, phrases in result['intent_classification'].items():
        if phrases:
            print(f"  {intent}: {len(phrases)} phrases")


async def test_full_pipeline():
    """Test the complete pipeline with a small website"""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline Test")
    print("="*60)
    
    # Use a small test website
    test_url = os.getenv("TEST_URL", "https://books.toscrape.com")
    
    print(f"Testing with: {test_url}")
    print("This will:")
    print("  1. Crawl the website (limited to 10 pages)")
    print("  2. Generate semantic paths")
    print("  3. Create training phrases")
    print("  4. Validate against golden images")
    
    navigator = EnhancedWebNavigator(
        start_url=test_url,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_depth=2,
        max_pages=10,  # Limited for testing
        generate_training=True,
        validate_golden_image=True,
        headless=True
    )
    
    try:
        print("\nRunning navigation and training generation...")
        results = await navigator.navigate_with_training()
        
        # Display summary
        quality_report = results.get("quality_report", {})
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  URLs processed: {quality_report.get('total_paths', 0)}")
        print(f"  Phrases generated: {quality_report.get('total_phrases_generated', 0)}")
        print(f"  Golden image compliance: {quality_report.get('paths_meeting_golden_image', 0)}/{quality_report.get('total_paths', 0)}")
        print(f"  Average quality score: {quality_report.get('average_quality_score', 0):.2f}")
        
        # Save test results
        saved_files = navigator.save_enhanced_results(results, "test_output")
        print(f"\nTest results saved to test_output/")
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_golden_image_compliance():
    """Test that generated phrases actually meet golden image standards"""
    print("\n" + "="*60)
    print("TEST 5: Golden Image Compliance")
    print("="*60)
    
    generator = UniversalPhraseGenerator(groq_api_key=os.getenv("GROQ_API_KEY"))
    
    # Test different domain types
    test_cases = [
        {
            "domain": DomainType.ECOMMERCE,
            "path": ["Home", "Electronics", "Laptops", "Gaming Laptops"],
            "url": "https://store.com/electronics/laptops/gaming"
        },
        {
            "domain": DomainType.HEALTHCARE,
            "path": ["Home", "Services", "Cardiology", "Heart Checkup"],
            "url": "https://clinic.com/services/cardiology/checkup"
        },
        {
            "domain": DomainType.EDUCATION,
            "path": ["Home", "Courses", "Computer Science", "Python Programming"],
            "url": "https://university.edu/courses/cs/python"
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting {test['domain'].value} domain:")
        print(f"  Path: {' → '.join(test['path'])}")
        
        # Force domain type for testing
        domain_analysis = {"domain_type": test["domain"].value}
        
        result = await generator.generate_phrases(
            url=test["url"],
            semantic_path=test["path"],
            domain_analysis=domain_analysis
        )
        
        if result["meets_golden_image"]:
            print(f"  ✓ Meets golden image standards")
        else:
            print(f"  ✗ Does not meet golden image")
            print(f"    Gaps: {', '.join(result['quality_gaps'][:2])}")
        
        print(f"  Generated {len(result['training_phrases'])} phrases")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GOLDEN IMAGE TRAINING GENERATOR TEST SUITE")
    print("="*60)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\nWARNING: No GROQ_API_KEY found.")
        print("Tests will run with template-based generation only.")
        print("For full testing, add GROQ_API_KEY to your .env file.\n")
    
    # Run tests
    await test_golden_image_validation()
    await test_domain_identification()
    await test_phrase_generation()
    await test_golden_image_compliance()
    
    # Optional: Run full pipeline test (takes longer)
    print("\n" + "="*60)
    response = input("Run full pipeline test? This will crawl a website. (y/n): ")
    if response.lower() == 'y':
        await test_full_pipeline()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())