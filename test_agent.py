"""
Test script for the Agentic Web Navigator
"""

import asyncio
import os
from dotenv import load_dotenv
from agent import AgenticWebNavigator

# Load environment variables
load_dotenv()


async def test_simple_website():
    """Test with a simple website"""
    print("="*60)
    print("TESTING AGENTIC WEB NAVIGATOR")
    print("="*60)
    
    # Test URL - using a simple website for initial testing
    test_url = "https://www.heb.com/"  # A simple test website for web scraping
    
    # Get Groq API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("\nWARNING: No GROQ_API_KEY found in environment.")
        print("The agent will run with heuristics only (no LLM labeling).")
        print("To enable LLM labeling, create a .env file with GROQ_API_KEY=your_key_here")
        input("\nPress Enter to continue with heuristics-only mode...")
    
    # Create agent with conservative settings for testing
    agent = AgenticWebNavigator(
        start_url=test_url,
        groq_api_key=groq_api_key,
        max_depth=2,  # Shallow depth for testing
        max_pages=20,  # Limited pages for testing
        headless=True,
        timeout=30000
    )
    
    print(f"\nStarting navigation of: {test_url}")
    print(f"Configuration:")
    print(f"  - Max Depth: 2")
    print(f"  - Max Pages: 20")
    print(f"  - LLM Labeling: {'Enabled' if groq_api_key else 'Disabled (heuristics only)'}")
    print("\nThis may take a minute...\n")
    
    try:
        # Navigate and generate semantic paths
        results = await agent.navigate()
        
        # Save results
        output_file = agent.save_results()
        
        # Display results
        print("\n" + "="*60)
        print("NAVIGATION COMPLETE")
        print("="*60)
        
        print("\nStatistics:")
        print(f"  - URLs Discovered: {results['metadata']['total_urls_discovered']}")
        print(f"  - Max Depth Reached: {results['metadata']['max_depth_reached']}")
        print(f"  - Duration: {results['metadata']['crawl_duration_seconds']:.2f} seconds")
        
        print("\nExample Navigation Paths:")
        print("-"*40)
        
        # Show first 5 navigation paths
        for i, (url, path_data) in enumerate(list(results['navigation_paths'].items())[:5], 1):
            semantic_path_str = " → ".join(path_data['semantic_path']) if path_data['semantic_path'] else "Home"
            print(f"{i}. {semantic_path_str}")
            print(f"   URL: {url}")
            print()
        
        print(f"Full results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting Agentic Web Navigator Test...")
    asyncio.run(test_simple_website())