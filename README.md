# StarshipChatbot - Production RAG Pipeline with LangGraph

A production-grade multi-agent pipeline built with **LangGraph orchestration** for intelligent web content extraction and Q&A generation. This system autonomously crawls websites, extracts detailed content using AI agents, and generates structured question-answer pairs for chatbot training.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   LangGraph Multi-Agent Pipeline              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  START → [Crawler] → [LoadJSON] → [Browser] → [QA Generator] │
│                          ↑              ↓            ↓        │
│                          └──────── Loop Back ────────┘        │
│                                                               │
│  Features:                                                    │
│  ✓ Conditional routing for batch processing                  │
│  ✓ AsyncSqliteSaver checkpointing (auto-resume)             │
│  ✓ Fault-tolerant execution                                  │
│  ✓ Pydantic-validated structured output                      │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### 🔄 LangGraph Orchestration
- **4-Agent Pipeline**: Crawler → Loader → Browser → Q&A Generator
- **Conditional Routing**: Intelligent looping through hundreds of items
- **State Management**: Shared state flows through all agents
- **Graph Compilation**: Production-ready workflow execution

### 💾 Fault-Tolerant Checkpointing
- **AsyncSqliteSaver**: Automatic state persistence at each node
- **Auto-Resume**: Pick up exactly where you left off after failures
- **Thread-Based**: Multiple workflows with unique thread IDs
- **Zero Manual Tracking**: No need for manual database management

### 🤖 Intelligent Content Extraction
- **browser_use Integration**: AI-powered browser automation
- **Groq AI**: Ultra-fast LLM inference (Llama 4 Maverick 17B)
- **Headless Browsing**: Scalable content extraction
- **Smart Extraction**: Comprehensive page analysis with context

### 📝 Structured Q&A Generation
- **Pydantic Validation**: Type-safe Q&A pair generation
- **10 Q&A per Page**: Diverse question types (What, How, Why, When, Where)
- **Chatbot-Ready**: Pre-formatted for training conversational AI
- **Quality Assurance**: Min/max length validation (8-12 pairs)

### 🕷️ Hierarchical Web Crawler
- **Semantic Path Generation**: Human-readable navigation paths
- **Priority System**: Headings first, then links
- **Noise Removal**: Strips headers/footers before extraction
- **Relationship-Aware**: Understands heading↔content↔links

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Agent workflow management |
| **Checkpointing** | AsyncSqliteSaver | State persistence & resume |
| **Browser Automation** | browser_use + Playwright | Intelligent web scraping |
| **LLM Provider** | Groq AI | Ultra-fast inference |
| **Model** | Llama 4 Maverick 17B | Content extraction & Q&A generation |
| **Validation** | Pydantic | Structured output validation |
| **Crawling** | Custom Hierarchical Crawler | Semantic element discovery |
| **Language** | Python 3.10+ | Core implementation |

## Installation

### Prerequisites

- Python 3.10 or higher
- Groq API key ([Get one here](https://console.groq.com))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/StarshipChatbot.git
cd StarshipChatbot
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
playwright install chromium
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

### Quick Start (Existing JSON)

Process a pre-existing JSON file with semantic elements:

```bash
python browser_agent.py
# Select option 2 (Use existing JSON file)
# Choose to enable checkpointing (recommended)
```

### Full Pipeline (With Crawler)

Crawl a website and process all discovered content:

```bash
python browser_agent.py
# Select option 1 (Use hierarchical web crawler)
# Enter website URL (e.g., https://example.com)
# Configure max depth and pages
# Enable checkpointing
```

### Programmatic Usage

```python
import asyncio
from browser_agent import run_browser_agent

async def main():
    # Option 1: Use existing JSON file
    result = await run_browser_agent(
        json_path="path/to/semantic_data.json",
        output_file="output.json",
        max_items=10,  # Process 10 items
        thread_id="my-workflow-1",
        enable_checkpointing=True
    )

    # Option 2: Crawl website first, then process
    result = await run_browser_agent(
        enable_crawler=True,
        start_url="https://example.com",
        max_depth=2,
        max_pages=50,
        output_file="output.json",
        thread_id="my-workflow-2",
        enable_checkpointing=True
    )

    print(f"Processed: {len(result['processed_items'])} items")
    print(f"Generated: {sum(item.get('qa_count', 0) for item in result['processed_items'])} Q&A pairs")

asyncio.run(main())
```

### Resume from Checkpoint

If the pipeline is interrupted, resume with the same thread_id:

```python
# Same thread_id will automatically resume from last checkpoint
result = await run_browser_agent(
    json_path="path/to/data.json",
    thread_id="my-workflow-1",  # Same thread_id as before
    enable_checkpointing=True
)
```

## Configuration

### Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `json_path` | Path to JSON file with semantic paths | Required (if crawler disabled) |
| `output_file` | Output file for processed results | `browser_agent_output.json` |
| `max_items` | Maximum items to process (None = all) | `None` |
| `thread_id` | Unique thread ID for checkpointing | `default-workflow` |
| `checkpoint_db` | SQLite database for checkpoints | `browser_agent_checkpoints.db` |
| `enable_checkpointing` | Enable/disable checkpoint functionality | `True` |

### Crawler Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_crawler` | Enable hierarchical web crawler | `False` |
| `start_url` | Website URL to crawl | Required (if crawler enabled) |
| `max_depth` | Maximum crawl depth | `2` |
| `max_pages` | Maximum pages to crawl | `100` |

## Output Format

The pipeline generates JSON output with processed items:

```json
[
  {
    "semantic_path": "https://example.com/page-title",
    "original_url": "https://example.com/page",
    "topic": "Page Title",
    "browser_content": "Extracted content...",
    "extraction_method": "ai_agent_groq",
    "processing_time": 12.5,
    "status": "completed",
    "qa_pairs": [
      {
        "question": "What is Page Title?",
        "answer": "Page Title is..."
      }
    ],
    "qa_generation_status": "completed",
    "qa_count": 10,
    "qa_model": "meta-llama/llama-4-maverick-17b-128e-instruct"
  }
]
```

## Project Structure

```
StarshipChatbot/
├── browser_agent.py              # Main LangGraph pipeline
├── hierarchical_crawler.py       # Hierarchical web crawler
├── labeling.py                   # Semantic labeling system
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
│
├── output/                       # Crawler output directory
│   └── hierarchical_crawl_*.json
│
└── browser_agent_checkpoints.db  # Checkpoint database (auto-generated)
```

## Architecture Details

### Agent Workflow

1. **CrawlerAgent** (Optional)
   - Hierarchical web crawling
   - Semantic path generation
   - Element relationship mapping
   - Outputs: `crawler_output_path`

2. **LoadJSONAgent**
   - Loads semantic elements from JSON
   - Handles both crawler output and pre-existing files
   - Converts formats and validates data
   - Outputs: `semantic_paths`, `total_items`

3. **BrowserAgent**
   - Extracts content using browser_use + Groq AI
   - Processes ONE item per execution
   - Creates detailed extraction records
   - Outputs: `browser_content`, `extraction_method`

4. **QAGeneratorAgent**
   - Generates 10 Q&A pairs per item
   - Uses Pydantic validation for quality
   - Supports diverse question types
   - Outputs: `qa_pairs`, `qa_generation_status`

### Conditional Routing

```python
def should_continue(state: AgentState) -> str:
    """
    Router function: Decides whether to continue or end

    Returns:
        "continue" - Loop back to browser_agent for next item
        "end" - All items processed, go to END
    """
    current_idx = state["current_index"]
    total_items = state["total_items"]

    return "continue" if current_idx < total_items else "end"
```

### Checkpointing System

- **Automatic**: State saved after each node execution
- **Transparent**: No code changes needed for resume
- **Thread-Based**: Multiple independent workflows
- **Efficient**: Only unsuccessful nodes re-run on resume

## Use Cases

### 1. Chatbot Training Data Generation
Extract website content and generate Q&A pairs for training conversational AI systems.

### 2. Knowledge Base Creation
Build comprehensive knowledge bases from multiple websites with structured Q&A format.

### 3. Content Analysis at Scale
Process hundreds of pages with fault-tolerant execution and automatic resume.

### 4. Semantic Web Mapping
Generate semantic navigation paths for understanding website structure.

## Performance

- **Processing Speed**: ~12-15 seconds per page (extraction + Q&A)
- **Scalability**: Handles 100+ pages with checkpointing
- **Fault Tolerance**: Auto-resume on any failure
- **Memory Efficient**: Isolated browser contexts
- **Concurrent Safe**: Thread-based checkpoint isolation

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Copy `.env.example` to `.env`
   - Add your Groq API key

2. **Checkpoint database locked**
   - Ensure no other instances are running
   - Delete `.db-shm` and `.db-wal` files

3. **Browser timeout errors**
   - Increase timeout in `hierarchical_crawler.py`
   - Check internet connection

4. **Pydantic validation errors**
   - LLM may generate < 8 Q&A pairs
   - These items are marked as "failed" and logged

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## Roadmap

- [ ] Support for multiple LLM providers (OpenAI, Anthropic)
- [ ] Parallel processing for faster execution
- [ ] Web UI for monitoring progress
- [ ] REST API interface
- [ ] Docker containerization
- [ ] Vector database integration (ChromaDB, Pinecone)
- [ ] Fine-tuning support for custom models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify License - MIT, Apache 2.0, etc.]

## Acknowledgments

- **LangGraph** by LangChain for orchestration framework
- **Groq** for ultra-fast LLM inference
- **browser_use** for AI-powered browser automation
- **Playwright** for reliable browser automation

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with LangGraph** | Production-Ready RAG Pipeline | 2025
