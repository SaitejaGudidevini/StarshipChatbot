# Agentic Web Navigator

An intelligent web crawling agent that autonomously explores websites and generates human-readable semantic navigation paths. This Phase 1 implementation creates the foundational data layer for the StarshipChatbot project.

## Overview

Traditional web crawlers identify what URLs exist but fail to capture the context of how users navigate to them. The Agentic Web Navigator solves this by:

- **Autonomously exploring** target websites using headless browser automation
- **Understanding navigation structure** through intelligent link analysis
- **Generating semantic paths** that describe the interaction sequence to reach specific URLs

For example, instead of just knowing `https://www.cinemark.com/movies/now-playing` exists, the system understands it's reached via: **Home → Movies → Now Playing**

## Features

- **Robust Navigation**: Isolated browser contexts prevent state pollution and JavaScript errors
- **Hierarchical Labeling System**:
  - Primary: Heuristic analysis (inner text, ARIA labels, title attributes)
  - Secondary: LLM interpretation via Groq API for ambiguous elements
- **Intelligent Path Generation**: Creates human-readable navigation sequences
- **Scalable Architecture**: Iterative crawling with configurable depth and page limits
- **JSON Output Format**: Standardized format for easy integration

## Technology Stack

- **Python 3.10+**: Core programming language
- **Playwright**: Browser automation for reliable navigation
- **Groq API**: Ultra-fast LLM inference for contextual analysis
- **AsyncIO**: Asynchronous operations for efficiency

## Installation

### Prerequisites

- Python 3.10 or higher
- Groq API key (optional, for LLM labeling)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd StarshipChatbot
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

4. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

### Basic Usage

Run the agent with default settings:

```bash
python agent.py
```

### Test Script

Run the test script to verify installation:

```bash
python test_agent.py
```

### Programmatic Usage

```python
import asyncio
from agent import AgenticWebNavigator

async def main():
    agent = AgenticWebNavigator(
        start_url="https://example.com",
        groq_api_key="your_api_key",  # Optional
        max_depth=3,
        max_pages=100,
        headless=True
    )
    
    results = await agent.navigate()
    output_file = agent.save_results()
    print(f"Results saved to: {output_file}")

asyncio.run(main())
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_url` | The website URL to navigate | Required |
| `groq_api_key` | Groq API key for LLM labeling | None (heuristics only) |
| `max_depth` | Maximum navigation depth | 3 |
| `max_pages` | Maximum pages to crawl | 100 |
| `headless` | Run browser in headless mode | True |
| `timeout` | Page load timeout (ms) | 30000 |

## Output Format

The agent generates JSON output with the following structure:

```json
{
  "metadata": {
    "start_url": "https://example.com",
    "domain": "example.com",
    "total_urls_discovered": 47,
    "max_depth_reached": 3,
    "crawl_duration_seconds": 125.4,
    "timestamp": "2025-09-14T10:30:00",
    "crawler_stats": {...}
  },
  "navigation_paths": {
    "https://example.com/page": {
      "url": "https://example.com/page",
      "semantic_path": ["Home", "Section", "Page"],
      "depth": 2,
      "parent_url": "https://example.com/section"
    }
  }
}
```

## Architecture

### Components

1. **WebCrawler** (`crawler.py`)
   - Manages browser automation with Playwright
   - Implements iterative crawling with isolated contexts
   - Extracts links and maintains navigation queue

2. **SemanticLabeler** (`labeling.py`)
   - Hierarchical labeling system
   - Heuristic analysis for high-confidence labels
   - LLM fallback for ambiguous elements

3. **AgenticWebNavigator** (`agent.py`)
   - Orchestrates crawling and labeling
   - Generates semantic navigation paths
   - Produces structured JSON output

### Workflow

1. **Initialize**: Set up browser and prepare crawling queue
2. **Crawl**: Iteratively visit pages in isolated contexts
3. **Extract**: Identify navigational links on each page
4. **Label**: Determine semantic labels using heuristics/LLM
5. **Build Paths**: Construct navigation sequences from root
6. **Output**: Generate structured JSON with all paths

## Limitations

### In Scope
- Standard navigation links (`<a>` tags)
- Same-domain navigation
- Dynamic content handling
- Basic JavaScript-rendered pages

### Out of Scope
- Form submissions and authentication
- Page content extraction
- Complex anti-bot measures/CAPTCHAs
- Cross-domain navigation

## Development

### Project Structure
```
StarshipChatbot/
├── agent.py           # Main orchestration agent
├── crawler.py         # Web crawling logic
├── labeling.py        # Semantic labeling system
├── test_agent.py      # Test script
├── requirements.txt   # Python dependencies
├── .env.example       # Environment configuration template
├── example_output.json # Sample output format
└── README.md          # This file
```

### Running Tests

```bash
# Run the test script
python test_agent.py

# Test with a specific website
START_URL=https://example.com python agent.py
```

## Troubleshooting

### Common Issues

1. **No GROQ_API_KEY**: The agent will run with heuristics only
2. **Timeout errors**: Increase the `timeout` parameter
3. **Memory issues**: Reduce `max_pages` or `max_depth`
4. **Playwright issues**: Ensure Chromium is installed: `playwright install chromium`

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Support for form interactions
- Multi-domain crawling
- Parallel crawling for performance
- Advanced anti-bot handling
- Real-time progress monitoring
- REST API interface

## License

[Specify License]

## Support

For issues or questions, please contact the StarshipChatbot Project Team.

---

*Phase 1 Implementation - Agentic Web Navigation and Semantic Pathing*