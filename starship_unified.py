"""
StarshipChatbot - Unified Control Center
=========================================

Single FastAPI server integrating all 3 systems:
1. Data Generation (browser_agent.py via browser_agent_runner.py)
2. Data Editor (langgraph_chatbot.py)
3. Production Chatbot (json_chatbot_engine.py)

Features:
- Unified web UI with tabbed interface
- Real-time SSE updates for data generation
- All endpoints in one server
- Integrated state management

Port: 8000

Usage:
    python starship_unified.py

    Open: http://localhost:8000
"""

import logging
import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

# CRITICAL: Load .env FIRST, before any browser_use imports!
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import shutil

# Import our systems (after load_dotenv so browser_use sees BROWSER_USE_LOGGING_LEVEL)
from json_chatbot_engine import JSONChatbotEngine
from browser_agent_runner import create_runner
import browser_agent  # Import module to access progress_tracker dynamically

# Try to import LangGraph editor
try:
    from langgraph_chatbot import QAWorkflowManager, build_merge_qa_graph, SimplifyAgent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("⚠️  LangGraph editor not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PERSISTENT STORAGE CONFIGURATION
# ============================================================================

# Data directory for persistent storage (Railway volume)
# Local: uses current directory
# Railway: uses /app/data (mounted volume)
DATA_DIR = os.getenv('DATA_DIR', '.')
os.makedirs(DATA_DIR, exist_ok=True)

logger.info(f"📁 Data directory: {DATA_DIR}")

# Helper function to get data file path
def get_data_path(filename: str) -> str:
    """Get full path to file in data directory"""
    return os.path.join(DATA_DIR, filename)


def load_qa_json(json_path: str) -> tuple:
    """
    Load Q&A JSON supporting both formats.
    Returns: (full_data, topics_array)
    - New format: {"tree": {...}, "topics": [...]} -> (data, data['topics'])
    - Legacy format: [...] -> (None, data)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'topics' in data:
        return data, data['topics']
    else:
        return None, data


def save_qa_json(json_path: str, full_data: dict, topics: list):
    """
    Save Q&A JSON preserving format.
    - If full_data is not None (new format), update topics in place
    - If full_data is None (legacy format), save topics array directly
    """
    if full_data is not None:
        full_data['topics'] = topics
        output = full_data
    else:
        output = topics

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# Global state
chatbot_engine: Optional[JSONChatbotEngine] = None
qa_modifier: Optional[Any] = None
browser_runner: Optional[Any] = None
generation_task: Optional[asyncio.Task] = None

# ============================================================================
# ARCHITECTURE SWITCH: Set to True for V2 (parallel-fused), False for V1 (sequential)
# ============================================================================
USE_V2_ARCHITECTURE = False  # Change to True to enable V2

# Current JSON file (stored in data directory)
default_json_name = os.getenv('JSON_DATA_PATH', 'CSU_Progress.json')
# Strip directory if provided, we'll use DATA_DIR
default_json_name = os.path.basename(default_json_name)
current_json_file: str = get_data_path(default_json_name)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class GenerateRequest(BaseModel):
    """Start data generation"""
    url: Optional[str] = None               # URL to crawl (required if use_crawler=True)
    max_pages: int = 10
    use_crawler: bool = False
    max_depth: int = 2                      # Max crawl depth (1-5)
    max_items: Optional[int] = None         # Limit items to process (None = all)
    thread_id: Optional[str] = None         # Custom thread ID for checkpointing
    enable_checkpointing: bool = True       # Enable/disable checkpointing
    output_filename: Optional[str] = None   # Custom output filename (None = auto-generate)
    json_filename: Optional[str] = None     # JSON file to use (required if use_crawler=False)


class ChatRequest(BaseModel):
    """Chat question"""
    question: str
    session_id: str = "default"


class SimplifyRequest(BaseModel):
    """Simplify topic"""
    topic_index: int


class MergeRequest(BaseModel):
    """Merge Q&A pairs"""
    topic_index: int
    qa_indices: List[int]
    user_request: str = "Merge these Q&A pairs"


class DeleteRequest(BaseModel):
    """Delete Q&A pairs"""
    topic_index: int
    qa_indices: List[int]


class EditRequest(BaseModel):
    """Edit Q&A pair"""
    topic_index: int
    qa_index: int
    new_question: str
    new_answer: str


class SwitchFileRequest(BaseModel):
    """Switch JSON file"""
    filename: str


class CancelRequest(BaseModel):
    """Cancel generation request"""
    save_data: bool = True  # Default to saving data


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all systems on startup"""
    global chatbot_engine, qa_modifier, current_json_file

    logger.info("="*60)
    logger.info("🚀 STARSHIP CHATBOT - UNIFIED SERVER STARTUP")
    logger.info("="*60)

    # Copy initial JSON files to data directory if they don't exist (first deploy)
    initial_files = ['CSU_Progress.json', 'browser_agent_test_output.json', 'SYRAHEALTHDEMOFINAL.json']
    for filename in initial_files:
        data_file = get_data_path(filename)
        if not os.path.exists(data_file):
            # Check if file exists in app directory (from Docker COPY)
            if os.path.exists(filename):
                import shutil
                shutil.copy(filename, data_file)
                logger.info(f"📋 Copied {filename} to data directory")
            else:
                logger.warning(f"⚠️  Initial file {filename} not found")

    json_path = current_json_file

    # Ensure json_path exists
    if not os.path.exists(json_path):
        logger.error(f"❌ JSON file not found: {json_path}")
        logger.info(f"Available files in DATA_DIR: {os.listdir(DATA_DIR)}")

    # Initialize chatbot engine
    try:
        logger.info("Initializing chatbot engine...")
        chatbot_engine = JSONChatbotEngine(
            json_path=json_path,
            enable_rephrasing=os.getenv('GROQ_API_KEY') is not None
        )

        # Select architecture: V1 (default), V2 (parallel-fused), V3 (Gemini + Q&A)
        # Set CHATBOT_VERSION=v1, v2, or v3 in environment
        chatbot_version = os.getenv('CHATBOT_VERSION', 'v1').lower()

        if chatbot_version == 'v3':
            logger.info("🚀 Enabling V3 Architecture (Gemini + Q&A)...")
            chatbot_engine.enable_v3_architecture()
            if hasattr(chatbot_engine, 'v3_enabled') and chatbot_engine.v3_enabled:
                logger.info("✅ Chatbot engine ready (V3: Gemini + Q&A)")
            else:
                logger.info("⚠️  V3 not available - falling back to V1")

        elif chatbot_version == 'v2':
            logger.info("🚀 Enabling V2 Architecture (Parallel-Fused)...")
            chatbot_engine.enable_v2_architecture()
            if chatbot_engine.v2_enabled:
                logger.info("✅ Chatbot engine ready (V2: Parallel-Fused)")
            else:
                logger.info("⚠️  V2 not available - falling back to V1")

        else:
            logger.info("✅ Chatbot engine ready (V1: Sequential Search)")

    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")

    # Initialize LangGraph editor
    if LANGGRAPH_AVAILABLE:
        try:
            logger.info("Initializing LangGraph editor...")
            from langgraph_chatbot import build_selective_modifier_graph

            # Build all required graphs
            merge_graph = build_merge_qa_graph()
            selective_graph = build_selective_modifier_graph()

            # Initialize workflow manager with both graphs
            qa_modifier = QAWorkflowManager(
                merge_graph=merge_graph,
                selective_graph=selective_graph
            )
            logger.info("✅ LangGraph editor ready (merge + selective workflows)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize editor: {e}")

    logger.info("="*60)
    logger.info("✅ STARSHIP UNIFIED SERVER READY")
    logger.info(f"   📊 Chatbot: {'Ready' if chatbot_engine else 'Failed'}")
    logger.info(f"   ✏️  Editor: {'Ready' if qa_modifier else 'Not available'}")
    logger.info(f"   🤖 Generator: Ready (on-demand)")
    logger.info("="*60)
    port = os.getenv('PORT', '8000')
    logger.info(f"🌐 Open: http://localhost:{port}")
    logger.info("="*60)

    yield

    logger.info("Shutting down...")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="StarshipChatbot - Unified Control Center",
    description="Complete Q&A system: Generate, Edit, Chat",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, update to specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if frontend build exists
FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")
USE_BUILT_FRONTEND = False
if os.path.exists(FRONTEND_BUILD_DIR):
    # Mount assets directory for JS/CSS files
    assets_dir = os.path.join(FRONTEND_BUILD_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    
    # Mount data directory for tree.json, if it was moved there
    data_dir = os.path.join(FRONTEND_BUILD_DIR, "data")
    if os.path.exists(data_dir):
        app.mount("/data", StaticFiles(directory=data_dir), name="data")

    logger.info(f"✅ Serving frontend build from {FRONTEND_BUILD_DIR}")
    USE_BUILT_FRONTEND = True
else:
    logger.info("ℹ️  No frontend build found. Using embedded HTML.")
    USE_BUILT_FRONTEND = False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/tree/data")
async def get_tree_data():
    """
    Serves tree data for the active Q&A JSON file.
    Priority: 1) Embedded in Q&A JSON, 2) Separate _tree.json file
    """
    global current_json_file

    if not current_json_file or not os.path.exists(current_json_file):
        raise HTTPException(status_code=404, detail="Active JSON file not set or not found.")

    # First, check if tree is embedded in the Q&A JSON (new format)
    try:
        with open(current_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'tree' in data:
            # New format: tree embedded in Q&A JSON
            logger.info(f"🌳 Serving embedded tree from: {current_json_file}")
            return JSONResponse({
                "metadata": data.get("metadata", {}),
                "tree": data["tree"]
            })
    except Exception as e:
        logger.warning(f"Could not check for embedded tree: {e}")

    # Fallback: Look for separate tree file
    base_path = os.path.splitext(current_json_file)[0]
    tree_filename = f"{base_path}_tree.json"

    if os.path.exists(tree_filename):
        logger.info(f"🌳 Serving separate tree file: {tree_filename}")
        return FileResponse(tree_filename, media_type='application/json')

    # Fallback: Search for *any* _tree.json file in multiple locations
    tree_files = []

    # 1. DATA_DIR (Railway volume: /app/data/)
    if os.path.exists(DATA_DIR):
        tree_files.extend([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_tree.json')])

    # 2. Local output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if os.path.exists(output_dir):
        tree_files.extend([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_tree.json')])

    if tree_files:
        # Serve the most recent one
        latest_tree_file = max(tree_files, key=os.path.getmtime)
        logger.info(f"🌳 Serving fallback tree: {latest_tree_file}")
        return FileResponse(latest_tree_file, media_type='application/json')

    raise HTTPException(status_code=404, detail=f"Tree not found. Checked: embedded in {current_json_file}, {tree_filename}, DATA_DIR, output/")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web UI - either built frontend or embedded HTML"""

    # If frontend build exists, serve index.html
    if USE_BUILT_FRONTEND:
        index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())

    # Otherwise, serve embedded HTML (backward compatible)
    # Get stats for UI
    total_topics = len(chatbot_engine.dataset.topics) if chatbot_engine else 0
    total_qa = len(chatbot_engine.dataset.all_qa_pairs) if chatbot_engine else 0

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StarshipChatbot - Control Center</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: #333;
        }

        /* Header */
        .header {
            background: rgba(0, 0, 0, 0.2);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
        }
        .header h1 { font-size: 1.8em; font-weight: 600; }
        .header .stats {
            display: flex;
            gap: 30px;
            font-size: 0.9em;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            opacity: 0.9;
            font-size: 0.85em;
        }

        /* Tabs */
        .tabs {
            display: flex;
            background: rgba(255, 255, 255, 0.1);
            padding: 0 40px;
            gap: 5px;
        }
        .tab {
            padding: 15px 30px;
            background: transparent;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }
        .tab:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        .tab.active {
            background: white;
            color: #1e3c72;
            border-bottom-color: #4CAF50;
        }

        /* Container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px;
        }

        /* Tab Content */
        .tab-content {
            display: none;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            min-height: 600px;
        }
        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Dashboard Cards */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .dashboard-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        .dashboard-card h3 {
            font-size: 1.1em;
            margin-bottom: 15px;
            opacity: 0.95;
        }
        .dashboard-card .value {
            font-size: 2.8em;
            font-weight: bold;
            margin: 15px 0;
        }
        .dashboard-card .label {
            opacity: 0.85;
            font-size: 0.95em;
        }

        /* Forms */
        .form-group {
            margin-bottom: 20px;
        }
        .form-label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        .form-input, .form-textarea {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            font-family: inherit;
            transition: border-color 0.3s ease;
        }
        .form-input:focus, .form-textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .form-textarea {
            min-height: 120px;
            resize: vertical;
        }

        /* Buttons */
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-success {
            background: #4CAF50;
            color: white;
        }
        .btn-success:hover {
            background: #45a049;
        }
        .btn-danger {
            background: #f44336;
            color: white;
        }
        .btn-danger:hover {
            background: #da190b;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        /* Progress Bar */
        .progress-container {
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-bar {
            height: 30px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.9em;
        }

        /* Generation Progress */
        .progress-box {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            display: none;
        }
        .progress-box.active {
            display: block;
        }
        .progress-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .progress-stat {
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .progress-stat-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        .progress-stat-value {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
        }

        /* Chat */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 15px 20px;
            border-radius: 12px;
            max-width: 75%;
            line-height: 1.6;
        }
        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .message.bot {
            background: white;
            border: 1px solid #e0e0e0;
        }
        .message .confidence {
            font-size: 0.85em;
            margin-top: 8px;
            opacity: 0.8;
            font-style: italic;
        }
        .chat-input-area {
            display: flex;
            gap: 10px;
        }
        .chat-input-area input {
            flex: 1;
        }

        /* Topics List */
        .topics-grid {
            display: grid;
            gap: 20px;
            margin-top: 20px;
        }
        .topic-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }
        .topic-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }
        .topic-card h3 {
            margin-bottom: 10px;
            color: #333;
            font-size: 1.2em;
        }
        .topic-card .meta {
            color: #666;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .topic-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        /* Status Messages */
        .status-message {
            padding: 15px 20px;
            border-radius: 8px;
            margin: 15px 0;
            display: none;
        }
        .status-message.active {
            display: block;
        }
        .status-message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status-message.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }

        /* Loading Spinner */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Tree Visualization Styles */
        #tree-container {
            width: 100%;
            height: 600px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: white;
            overflow: hidden;
            position: relative;
        }
        .tree-node circle {
            fill: #fff;
            stroke: #667eea;
            stroke-width: 2px;
            cursor: pointer;
        }
        .tree-node circle:hover {
            stroke: #764ba2;
            stroke-width: 3px;
        }
        .tree-node text {
            font: 12px sans-serif;
            cursor: pointer;
            fill: #333;
        }
        .tree-link {
            fill: none;
            stroke: #ccc;
            stroke-width: 1.5px;
        }
        .tree-tooltip {
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 12px sans-serif;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            pointer-events: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
            max-width: 300px;
        }
        .tree-controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 100;
            display: flex;
            gap: 10px;
        }
        .tree-info {
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 0.9em;
        }

        /* Quick Actions */
        .quick-actions {
            display: flex;
            gap: 15px;
            margin-top: 30px;
            flex-wrap: wrap;
        }

        h2 {
            margin-bottom: 10px;
            color: #333;
            font-size: 1.8em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.05em;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <div>
            <h1>🚀 StarshipChatbot Control Center</h1>
            <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px;">
                <label style="font-size: 0.9em; opacity: 0.9;">Active JSON File:</label>
                <select id="jsonFileSelector" onchange="switchJsonFile()" style="padding: 5px 10px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.3); background: rgba(255,255,255,0.1); color: white; cursor: pointer; font-size: 0.9em;">
                    <option>Loading...</option>
                </select>
                <button onclick="document.getElementById('jsonUploadInput').click()" style="padding: 5px 15px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.3); background: rgba(76, 175, 80, 0.2); color: white; cursor: pointer; font-size: 0.9em; font-weight: 600;">
                    ➕ Upload JSON
                </button>
                <input type="file" id="jsonUploadInput" accept=".json" style="display: none;" onchange="uploadJsonFile(event)">
            </div>
        </div>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="headerTopics">""" + str(total_topics) + """</div>
                <div class="stat-label">Topics</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="headerQA">""" + str(total_qa) + """</div>
                <div class="stat-label">Q&A Pairs</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: #4CAF50;">●</div>
                <div class="stat-label">Online</div>
            </div>
        </div>
    </div>

    <!-- Tabs -->
    <div class="tabs">
        <button class="tab active" onclick="showTab('dashboard')">📊 Dashboard</button>
        <button class="tab" onclick="showTab('generate')">🤖 Generate Data</button>
        <button class="tab" onclick="showTab('edit')">✏️ Edit Data</button>
        <button class="tab" onclick="showTab('chat')">💬 Chat</button>
        <button class="tab" onclick="showTab('tree')">🌳 Tree View</button>
        <button class="tab" onclick="showTab('settings')">⚙️ Settings</button>
    </div>

    <!-- Container -->
    <div class="container">

        <!-- TAB 1: DASHBOARD -->
        <div id="dashboard" class="tab-content active">
            <h2>System Dashboard</h2>
            <p class="subtitle">Overview of your StarshipChatbot system</p>

            <div class="dashboard-grid">
                <div class="dashboard-card">
                    <h3>📚 Q&A Database</h3>
                    <div class="value" id="dashQA">""" + str(total_qa) + """</div>
                    <div class="label">Question-answer pairs</div>
                </div>
                <div class="dashboard-card">
                    <h3>📑 Topics</h3>
                    <div class="value" id="dashTopics">""" + str(total_topics) + """</div>
                    <div class="label">Organized topics</div>
                </div>
                <div class="dashboard-card">
                    <h3>🎯 Chatbot</h3>
                    <div class="value">Ready</div>
                    <div class="label">Semantic search active</div>
                </div>
                <div class="dashboard-card">
                    <h3>🤖 Generator</h3>
                    <div class="value">Ready</div>
                    <div class="label">On-demand scraping</div>
                </div>
            </div>

            <div class="quick-actions">
                <button class="btn btn-primary" onclick="showTab('chat')">💬 Start Chatting</button>
                <button class="btn btn-success" onclick="showTab('edit')">✏️ Edit Q&A Data</button>
                <button class="btn btn-primary" onclick="showTab('generate')">🤖 Generate New Data</button>
            </div>
        </div>

        <!-- TAB 2: GENERATE DATA -->
        <div id="generate" class="tab-content">
            <h2>🤖 Generate Q&A Data from Website</h2>
            <p class="subtitle">Automatically scrape websites and generate Q&A pairs using AI</p>

            <!-- Mode Selection -->
            <div class="form-group">
                <label class="form-label">Generation Mode</label>
                <div style="display: flex; gap: 20px; margin-top: 10px;">
                    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="radio" name="generationMode" value="crawler" checked onchange="toggleGenerationMode()">
                        <span>🕷️ Run Crawler (Discover + Generate)</span>
                    </label>
                    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="radio" name="generationMode" value="existing" onchange="toggleGenerationMode()">
                        <span>📂 Use Existing Crawler JSON (Generate Only)</span>
                    </label>
                </div>
            </div>

            <!-- CRAWLER MODE FIELDS -->
            <div id="crawlerModeFields">
                <div class="form-group">
                    <label class="form-label">Website URL</label>
                    <input type="text" id="generateUrl" class="form-input" placeholder="https://pytorch.org" value="https://pytorch.org" />
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div class="form-group">
                        <label class="form-label">Maximum Pages to Crawl</label>
                        <input type="number" id="generateMaxPages" class="form-input" value="20" min="1" max="500" />
                    </div>
                    <div class="form-group">
                        <label class="form-label">Maximum Depth (Levels)</label>
                        <input type="number" id="generateMaxDepth" class="form-input" value="3" min="1" max="10" />
                    </div>
                </div>
            </div>

            <!-- EXISTING JSON MODE FIELDS -->
            <div id="existingJsonFields" style="display: none;">
                <div class="form-group">
                    <label class="form-label">Select Crawler Output JSON</label>
                    <select id="crawlerJsonSelector" class="form-input" onchange="loadCrawlerJsonPreview()">
                        <option value="">-- Select a file --</option>
                    </select>
                    <button class="btn btn-secondary" onclick="refreshCrawlerJsonList()" style="margin-top: 10px;">
                        🔄 Refresh List
                    </button>
                </div>

                <div id="crawlerJsonPreview" style="display: none; background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px;">
                    <h4 style="margin: 0 0 10px 0;">File Preview:</h4>
                    <div style="font-size: 14px; color: #666;">
                        <div><strong>Domain:</strong> <span id="previewDomain">-</span></div>
                        <div><strong>Total Elements:</strong> <span id="previewElements">-</span></div>
                        <div><strong>Crawled:</strong> <span id="previewTimestamp">-</span></div>
                    </div>
                </div>
            </div>

            <!-- COMMON FIELDS -->
            <div class="form-group">
                <label class="form-label">Max Items to Process (Optional)</label>
                <input type="number" id="generateMaxItems" class="form-input" placeholder="Leave empty to process all" min="1" />
                <small style="color: #666; font-size: 12px;">Limit how many semantic paths to process (useful for testing)</small>
            </div>

            <div style="display: flex; gap: 15px;">
                <button class="btn btn-primary" onclick="startGeneration()" id="startBtn">
                    <span id="startBtnText">▶️ Start Generation</span>
                </button>
                <button class="btn btn-danger" onclick="cancelGeneration()" id="cancelBtn" style="display:none;">
                    ⏹️ Cancel
                </button>
            </div>

            <div id="statusMessage" class="status-message"></div>

            <div id="progressBox" class="progress-box">
                <h3>Generation Progress</h3>

                <div class="progress-container">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>

                <div class="progress-stats">
                    <div class="progress-stat">
                        <div class="progress-stat-label">Status</div>
                        <div class="progress-stat-value" id="progStatus">Idle</div>
                    </div>
                    <div class="progress-stat">
                        <div class="progress-stat-label">Progress</div>
                        <div class="progress-stat-value" id="progProgress">0 / 0</div>
                    </div>
                    <div class="progress-stat">
                        <div class="progress-stat-label">Q&A Generated</div>
                        <div class="progress-stat-value" id="progQA">0</div>
                    </div>
                    <div class="progress-stat">
                        <div class="progress-stat-label">Elapsed Time</div>
                        <div class="progress-stat-value" id="progTime">0s</div>
                    </div>
                </div>

                <div style="margin-top: 15px; padding: 15px; background: white; border-radius: 8px;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Current URL:</div>
                    <div style="font-weight: 600; color: #333; word-break: break-all;" id="progUrl">-</div>
                </div>
            </div>
        </div>

        <!-- TAB 3: EDIT DATA -->
        <div id="edit" class="tab-content">
            <h2>✏️ Edit Q&A Database</h2>
            <p class="subtitle">Browse and edit your Q&A database with AI assistance</p>

            <div id="topicsList" class="topics-grid">
                <div style="text-align: center; padding: 40px; color: #666;">
                    <div class="spinner" style="border-color: #667eea; border-top-color: transparent; width: 40px; height: 40px;"></div>
                    <p style="margin-top: 15px;">Loading topics...</p>
                </div>
            </div>
        </div>

        <!-- TAB 4: CHAT -->
        <div id="chat" class="tab-content">
            <h2>💬 Interactive Chat</h2>
            <p class="subtitle">Test your chatbot with real questions</p>

            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot">
                        <strong>StarshipBot</strong>
                        <p>Hello! I'm ready to answer questions from the Q&A database. Ask me anything!</p>
                    </div>
                </div>

                <div class="chat-input-area">
                    <input type="text" id="chatInput" class="form-input" placeholder="Type your question here..." />
                    <button class="btn btn-primary" onclick="sendChat()" id="chatSendBtn">Send</button>
                </div>
            </div>
        </div>

        <!-- TAB 5: TREE VISUALIZATION -->
        <div id="tree" class="tab-content">
            <h2>🌳 Crawler Tree Visualization</h2>
            <p class="subtitle">Interactive hierarchical view of crawled website structure</p>

            <div style="position: relative;">
                <div id="tree-container">
                    <div style="text-align: center; padding: 100px 40px; color: #666;">
                        <div class="spinner" style="border-color: #667eea; border-top-color: transparent; width: 40px; height: 40px; margin: 0 auto;"></div>
                        <p style="margin-top: 20px;">Loading tree data...</p>
                    </div>
                </div>
                <div class="tree-tooltip" id="treeTooltip"></div>
            </div>

            <div class="tree-info" id="treeInfo">
                <strong>Instructions:</strong> Scroll to zoom • Drag to pan • Click nodes to expand/collapse children
            </div>
        </div>

        <!-- TAB 6: SETTINGS -->
        <div id="settings" class="tab-content">
            <h2>⚙️ System Settings</h2>
            <p class="subtitle">Configure your StarshipChatbot system</p>

            <div class="form-group">
                <label class="form-label">Active JSON Data File</label>
                <input type="text" id="settingsJsonPath" class="form-input" value="Loading..." readonly />
                <p style="margin-top: 5px; font-size: 0.85em; color: #666;">Use the dropdown in the header to switch between JSON files</p>
            </div>

            <div class="form-group">
                <label class="form-label">Groq API Key Status</label>
                <input type="text" class="form-input" value="Configured ✓" readonly />
            </div>

            <div style="margin-top: 40px;">
                <h3 style="margin-bottom: 20px;">System Information</h3>
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px;">
                    <p style="margin-bottom: 15px;"><strong>Chatbot Engine:</strong> <span style="color: #4CAF50;">✓ Active</span></p>
                    <p style="margin-bottom: 15px;"><strong>Data Editor:</strong> <span style="color: #4CAF50;">✓ Active</span></p>
                    <p style="margin-bottom: 15px;"><strong>Data Generator:</strong> <span style="color: #4CAF50;">✓ Ready</span></p>
                    <p><strong>Real-time Updates:</strong> <span style="color: #4CAF50;">✓ SSE Enabled</span></p>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Global state
        let currentEventSource = null;

        // Tab switching
        function showTab(tabName) {
            // Remove active class from all tabs and content
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Add active class to selected tab
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');

            // Load data if needed
            if (tabName === 'edit') {
                loadTopics();
            } else if (tabName === 'tree') {
                loadTreeVisualization();
            }
        }

        // ═══════════════════════════════════════════════════════════
        // DATA GENERATION
        // ═══════════════════════════════════════════════════════════

        function toggleGenerationMode() {
            const mode = document.querySelector('input[name="generationMode"]:checked').value;
            const crawlerFields = document.getElementById('crawlerModeFields');
            const existingFields = document.getElementById('existingJsonFields');

            if (mode === 'crawler') {
                crawlerFields.style.display = 'block';
                existingFields.style.display = 'none';
            } else {
                crawlerFields.style.display = 'none';
                existingFields.style.display = 'block';
                refreshCrawlerJsonList();
            }
        }

        async function refreshCrawlerJsonList() {
            try {
                const response = await fetch('/api/crawler-files/list');
                const data = await response.json();

                const selector = document.getElementById('crawlerJsonSelector');
                selector.innerHTML = '<option value="">-- Select a file --</option>';

                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.textContent = file;
                    selector.appendChild(option);
                });
            } catch (error) {
                showStatus('Error loading crawler files: ' + error.message, 'error');
            }
        }

        async function loadCrawlerJsonPreview() {
            const filename = document.getElementById('crawlerJsonSelector').value;
            const preview = document.getElementById('crawlerJsonPreview');

            if (!filename) {
                preview.style.display = 'none';
                return;
            }

            try {
                const response = await fetch(`/api/crawler-files/preview?filename=${encodeURIComponent(filename)}`);
                const data = await response.json();

                document.getElementById('previewDomain').textContent = data.domain || 'N/A';
                document.getElementById('previewElements').textContent = data.total_elements || 'N/A';
                document.getElementById('previewTimestamp').textContent = data.timestamp ? new Date(data.timestamp).toLocaleString() : 'N/A';

                preview.style.display = 'block';
            } catch (error) {
                showStatus('Error loading preview: ' + error.message, 'error');
            }
        }

        async function startGeneration() {
            const mode = document.querySelector('input[name="generationMode"]:checked').value;
            const maxItems = document.getElementById('generateMaxItems').value;

            let requestBody = {
                max_items: maxItems ? parseInt(maxItems) : null,
                enable_checkpointing: true
            };

            // Build request based on mode
            if (mode === 'crawler') {
                const url = document.getElementById('generateUrl').value.trim();
                const maxPages = parseInt(document.getElementById('generateMaxPages').value);
                const maxDepth = parseInt(document.getElementById('generateMaxDepth').value);

                if (!url) {
                    showStatus('Please enter a URL', 'error');
                    return;
                }

                requestBody.use_crawler = true;
                requestBody.url = url;
                requestBody.max_pages = maxPages;
                requestBody.max_depth = maxDepth;
            } else {
                const jsonFilename = document.getElementById('crawlerJsonSelector').value;

                if (!jsonFilename) {
                    showStatus('Please select a crawler JSON file', 'error');
                    return;
                }

                requestBody.use_crawler = false;
                requestBody.json_filename = jsonFilename;
            }

            // Start generation
            try {
                const response = await fetch('/api/generate/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to start generation');
                }

                // Show progress box
                document.getElementById('progressBox').classList.add('active');
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('cancelBtn').style.display = 'inline-block';

                showStatus('Generation started! Connecting to live updates...', 'info');

                // Connect to SSE stream
                connectToGenerationStream();

            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }

        function connectToGenerationStream() {
            // Close existing connection
            if (currentEventSource) {
                currentEventSource.close();
            }

            // Create new SSE connection
            currentEventSource = new EventSource('/api/generate/stream');

            currentEventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);

                // Update UI
                updateGenerationProgress(data);

                // Handle completion
                if (data.status === 'completed') {
                    showStatus('✅ Generation completed! Reloading data...', 'success');
                    currentEventSource.close();
                    document.getElementById('startBtn').style.display = 'inline-block';
                    document.getElementById('cancelBtn').style.display = 'none';

                    // Reload page after 2 seconds
                    setTimeout(() => location.reload(), 2000);
                } else if (data.status === 'error') {
                    showStatus('❌ Error: ' + data.error, 'error');
                    currentEventSource.close();
                    document.getElementById('startBtn').style.display = 'inline-block';
                    document.getElementById('cancelBtn').style.display = 'none';
                }
            };

            currentEventSource.onerror = () => {
                showStatus('Connection lost. Retrying...', 'error');
            };
        }

        function updateGenerationProgress(data) {
            // Update progress bar
            const percent = data.total > 0 ? (data.current / data.total) * 100 : 0;
            const bar = document.getElementById('progressBar');
            bar.style.width = percent + '%';
            bar.textContent = Math.round(percent) + '%';

            // Update stats
            document.getElementById('progStatus').textContent = data.status;
            document.getElementById('progProgress').textContent = `${data.current} / ${data.total}`;
            document.getElementById('progQA').textContent = data.qa_generated;
            document.getElementById('progTime').textContent = data.elapsed_seconds + 's';
            document.getElementById('progUrl').textContent = data.current_url || '-';
        }

        async function cancelGeneration() {
            if (!confirm('Cancel generation?')) return;

            try {
                await fetch('/api/generate/cancel', {method: 'POST'});
                showStatus('Generation cancelled', 'info');

                if (currentEventSource) {
                    currentEventSource.close();
                }

                document.getElementById('startBtn').style.display = 'inline-block';
                document.getElementById('cancelBtn').style.display = 'none';
            } catch (error) {
                showStatus('Error cancelling: ' + error.message, 'error');
            }
        }

        function showStatus(message, type) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.className = `status-message ${type} active`;
            statusDiv.textContent = message;

            if (type === 'success') {
                setTimeout(() => statusDiv.classList.remove('active'), 5000);
            }
        }

        // ═══════════════════════════════════════════════════════════
        // EDITOR
        // ═══════════════════════════════════════════════════════════

        async function loadTopics() {
            const container = document.getElementById('topicsList');

            try {
                const response = await fetch('/api/editor/topics');
                const topics = await response.json();

                if (topics.length === 0) {
                    container.innerHTML = '<div style="text-align:center;padding:40px;color:#666;">No topics found. Generate some data first!</div>';
                    return;
                }

                container.innerHTML = topics.map((topic, idx) => `
                    <div class="topic-card">
                        <h3>${topic.name}</h3>
                        <div class="meta">${topic.qa_count} Q&A pairs</div>
                        <div class="topic-actions">
                            <button class="btn btn-primary" onclick="viewTopic(${idx})">👁️ View Details</button>
                            <button class="btn btn-success" onclick="simplifyTopicQuick(${idx}, '${topic.name.replace(/'/g, "\\'")}')">✨ Simplify</button>
                        </div>
                    </div>
                `).join('');

            } catch (error) {
                container.innerHTML = '<div style="color:red;padding:40px;text-align:center;">Error loading topics</div>';
            }
        }

        async function simplifyTopicQuick(topicIndex, topicName) {
            if (!confirm(`Simplify topic "${topicName}"? This will create intelligent buckets with grouped questions.`)) return;

            try {
                showStatus('Simplifying topic... This may take a moment.', 'info');

                const response = await fetch('/api/editor/simplify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({topic_index: topicIndex})
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`✅ ${result.message} - Created ${result.buckets_created} buckets with ${result.new_count} Q&A pairs`, 'success');
                    loadTopics(); // Reload topic list
                } else {
                    showStatus(`❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }

        async function viewTopic(idx) {
            try {
                const response = await fetch(`/api/editor/topics/${idx}`);
                const topicData = await response.json();

                // Create modal HTML
                const modalHtml = `
                    <div id="topicModal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; display: flex; align-items: center; justify-content: center;">
                        <div style="background: white; border-radius: 15px; max-width: 900px; max-height: 90vh; width: 90%; overflow: hidden; display: flex; flex-direction: column;">
                            <!-- Header -->
                            <div style="padding: 25px; border-bottom: 2px solid #e0e0e0; display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h2 style="margin: 0; color: #333;">${topicData.topic}</h2>
                                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">${topicData.qa_count} Q&A pairs</p>
                                </div>
                                <button onclick="closeTopicModal()" style="background: none; border: none; font-size: 28px; cursor: pointer; color: #666;">&times;</button>
                            </div>

                            <!-- Action Buttons -->
                            <div style="padding: 15px 25px; background: #f8f9fa; border-bottom: 1px solid #e0e0e0; display: flex; gap: 10px; flex-wrap: wrap;">
                                <button class="btn btn-success" onclick="simplifyCurrentTopic(${idx})" style="font-size: 0.9em; padding: 8px 16px;">✨ Simplify All</button>
                                <button class="btn btn-primary" onclick="mergeSelectedQA(${idx})" style="font-size: 0.9em; padding: 8px 16px;">🔀 Merge Selected</button>
                                <button class="btn btn-danger" onclick="deleteSelectedQA(${idx})" style="font-size: 0.9em; padding: 8px 16px;">🗑️ Delete Selected</button>
                                <div style="flex: 1;"></div>
                                <span id="selectionCount" style="align-self: center; color: #666; font-size: 0.9em;">0 selected</span>
                            </div>

                            <!-- Q&A List -->
                            <div style="flex: 1; overflow-y: auto; padding: 25px;">
                                <div id="qaList">
                                    ${topicData.qa_pairs.map((qa, qaIdx) => `
                                        <div class="qa-item" style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 4px solid #667eea;">
                                            <div style="display: flex; gap: 15px;">
                                                <input type="checkbox" class="qa-checkbox" data-qa-index="${qaIdx}" style="width: 20px; height: 20px; cursor: pointer;" onchange="updateSelectionCount()">
                                                <div style="flex: 1;">
                                                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                                                        <strong style="color: #333; font-size: 1.05em;">Q${qaIdx + 1}: ${qa.question}</strong>
                                                        <button class="btn btn-primary" onclick="editQAPair(${idx}, ${qaIdx}, \`${qa.question.replace(/`/g, '\\`')}\`, \`${qa.answer.replace(/`/g, '\\`')}\`, event)" style="font-size: 0.8em; padding: 5px 12px;">✏️ Edit</button>
                                                    </div>
                                                    <p style="color: #555; margin: 10px 0 0 0; line-height: 1.6;">${qa.answer}</p>
                                                    ${qa.is_bucketed ? `<div style="margin-top: 8px; font-size: 0.85em; color: #667eea;">🗂️ ${qa.bucket_id}</div>` : ''}
                                                    ${qa.is_unified ? `<div style="margin-top: 8px; font-size: 0.85em; color: #4CAF50;">✨ Unified Answer</div>` : ''}
                                                </div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                // Insert modal
                document.body.insertAdjacentHTML('beforeend', modalHtml);

            } catch (error) {
                alert('Error loading topic details: ' + error.message);
            }
        }

        function closeTopicModal() {
            const modal = document.getElementById('topicModal');
            if (modal) modal.remove();
        }

        function updateSelectionCount() {
            const checkboxes = document.querySelectorAll('.qa-checkbox:checked');
            const count = checkboxes.length;
            const countSpan = document.getElementById('selectionCount');
            if (countSpan) {
                countSpan.textContent = `${count} selected`;
                countSpan.style.color = count > 0 ? '#667eea' : '#666';
                countSpan.style.fontWeight = count > 0 ? 'bold' : 'normal';
            }
        }

        async function simplifyCurrentTopic(topicIndex) {
            if (!confirm('Simplify all Q&A pairs in this topic? This will create intelligent buckets with grouped questions.')) return;

            try {
                showStatus('Simplifying topic... This may take a moment.', 'info');

                const response = await fetch('/api/editor/simplify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({topic_index: topicIndex})
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`✅ ${result.message} - Created ${result.buckets_created} buckets with ${result.new_count} Q&A pairs`, 'success');
                    closeTopicModal();
                    loadTopics(); // Reload topic list
                } else {
                    showStatus(`❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }

        async function mergeSelectedQA(topicIndex) {
            const checkboxes = document.querySelectorAll('.qa-checkbox:checked');
            const selectedIndices = Array.from(checkboxes).map(cb => parseInt(cb.dataset.qaIndex));

            if (selectedIndices.length < 2) {
                alert('Please select at least 2 Q&A pairs to merge');
                return;
            }

            if (!confirm(`Merge ${selectedIndices.length} Q&A pairs into one? The originals will be kept and a merged version will be added.`)) return;

            try {
                showStatus('Merging Q&A pairs... This may take a moment.', 'info');

                const response = await fetch('/api/editor/merge', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        topic_index: topicIndex,
                        qa_indices: selectedIndices,
                        user_request: 'Merge these Q&A pairs intelligently'
                    })
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`✅ ${result.message}`, 'success');
                    closeTopicModal();
                    loadTopics(); // Reload topic list
                } else {
                    showStatus(`❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }

        async function deleteSelectedQA(topicIndex) {
            const checkboxes = document.querySelectorAll('.qa-checkbox:checked');
            const selectedIndices = Array.from(checkboxes).map(cb => parseInt(cb.dataset.qaIndex));

            if (selectedIndices.length === 0) {
                alert('Please select Q&A pairs to delete');
                return;
            }

            if (!confirm(`Delete ${selectedIndices.length} Q&A pair(s)? This cannot be undone.`)) return;

            try {
                const response = await fetch('/api/editor/delete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        topic_index: topicIndex,
                        qa_indices: selectedIndices
                    })
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(`✅ ${result.message}`, 'success');
                    closeTopicModal();
                    loadTopics(); // Reload topic list
                } else {
                    showStatus(`❌ Error: ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            }
        }

        async function editQAPair(topicIndex, qaIndex, currentQuestion, currentAnswer, event) {
            event.stopPropagation();

            // Create edit modal
            const editModalHtml = `
                <div id="editModal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1100; display: flex; align-items: center; justify-content: center;">
                    <div style="background: white; border-radius: 15px; max-width: 600px; width: 90%; padding: 30px;">
                        <h3 style="margin: 0 0 20px 0;">Edit Q&A Pair</h3>

                        <div class="form-group">
                            <label class="form-label">Question:</label>
                            <textarea id="editQuestion" class="form-textarea" style="min-height: 80px;">${currentQuestion}</textarea>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Answer:</label>
                            <textarea id="editAnswer" class="form-textarea" style="min-height: 120px;">${currentAnswer}</textarea>
                        </div>

                        <div style="display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px;">
                            <button class="btn" onclick="closeEditModal()" style="background: #ccc;">Cancel</button>
                            <button class="btn btn-primary" onclick="saveQAEdit(${topicIndex}, ${qaIndex})">Save Changes</button>
                        </div>
                    </div>
                </div>
            `;

            document.body.insertAdjacentHTML('beforeend', editModalHtml);
        }

        function closeEditModal() {
            const modal = document.getElementById('editModal');
            if (modal) modal.remove();
        }

        async function saveQAEdit(topicIndex, qaIndex) {
            const newQuestion = document.getElementById('editQuestion').value.trim();
            const newAnswer = document.getElementById('editAnswer').value.trim();

            if (!newQuestion || !newAnswer) {
                alert('Question and answer cannot be empty');
                return;
            }

            try {
                const response = await fetch('/api/editor/edit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        topic_index: topicIndex,
                        qa_index: qaIndex,
                        new_question: newQuestion,
                        new_answer: newAnswer
                    })
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus('✅ Q&A pair updated successfully', 'success');
                    closeEditModal();
                    closeTopicModal();
                    loadTopics();
                } else {
                    alert('Error: ' + result.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        // ═══════════════════════════════════════════════════════════
        // CHAT
        // ═══════════════════════════════════════════════════════════

        async function sendChat() {
            const input = document.getElementById('chatInput');
            const question = input.value.trim();

            if (!question) return;

            // Add user message
            addMessage(question, 'user');
            input.value = '';

            // Disable send button
            const sendBtn = document.getElementById('chatSendBtn');
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="spinner"></span>';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });

                const result = await response.json();

                // Add bot response
                const botHTML = `
                    <strong>StarshipBot</strong>
                    <p>${result.answer}</p>
                    <div class="confidence">
                        Confidence: ${(result.confidence * 100).toFixed(1)}% |
                        Matched by: ${result.matched_by}
                        ${result.source_topic ? ' | Topic: ' + result.source_topic : ''}
                    </div>
                `;
                addMessage(botHTML, 'bot', true);

            } catch (error) {
                addMessage('Sorry, an error occurred: ' + error.message, 'bot');
            }

            // Re-enable send button
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        }

        function addMessage(content, type, isHTML = false) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;

            if (isHTML) {
                messageDiv.innerHTML = content;
            } else {
                if (type === 'user') {
                    messageDiv.innerHTML = `<strong>You</strong><p>${content}</p>`;
                } else {
                    messageDiv.textContent = content;
                }
            }

            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // Enter key to send chat
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChat();
        });

        // ═══════════════════════════════════════════════════════════
        // TREE VISUALIZATION
        // ═══════════════════════════════════════════════════════════

        let treeDataGlobal = null;
        let treeRendered = false;

        async function loadTreeVisualization() {
            // Only load once
            if (treeRendered) return;

            try {
                const response = await fetch('/api/tree/data');
                
                if (!response.ok) {
                    const error = await response.json();
                    document.getElementById('tree-container').innerHTML = `
                        <div style="text-align: center; padding: 100px 40px; color: #e74c3c;">
                            <h3>❌ No Tree Data Found</h3>
                            <p>${error.detail}</p>
                            <p style="margin-top: 15px;">Generate data first using the "Generate Data" tab.</p>
                        </div>
                    `;
                    return;
                }

                const data = await response.json();
                treeDataGlobal = data;
                
                // Update info
                const metadata = data.metadata || {};
                document.getElementById('treeInfo').innerHTML = `
                    <strong>Domain:</strong> ${metadata.domain || 'Unknown'} | 
                    <strong>Total Nodes:</strong> ${metadata.total_nodes || 0} | 
                    <strong>Max Depth:</strong> ${metadata.max_depth || 0}<br>
                    <strong>Instructions:</strong> Scroll to zoom • Drag to pan • Click nodes to expand/collapse children
                `;

                // Render tree
                renderTree(data.tree);
                treeRendered = true;

            } catch (error) {
                console.error('Error loading tree:', error);
                document.getElementById('tree-container').innerHTML = `
                    <div style="text-align: center; padding: 100px 40px; color: #e74c3c;">
                        <h3>❌ Error Loading Tree</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }

        function renderTree(treeData) {
            // Clear container
            const container = document.getElementById('tree-container');
            container.innerHTML = '';

            // Set up dimensions
            const width = container.clientWidth;
            const height = container.clientHeight;

            // Create SVG
            const svg = d3.select('#tree-container').append('svg')
                .attr('width', width)
                .attr('height', height);

            // Create zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 3])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                });

            svg.call(zoom);

            // Create main group
            const g = svg.append('g')
                .attr('transform', `translate(100, ${height / 2})`);

            // Create controls
            const controlsDiv = d3.select('#tree-container').append('div')
                .attr('class', 'tree-controls');

            controlsDiv.append('button')
                .attr('class', 'btn btn-primary')
                .style('font-size', '0.85em')
                .style('padding', '6px 12px')
                .text('Expand All')
                .on('click', () => {
                    expandAll(root);
                    update(root);
                });

            controlsDiv.append('button')
                .attr('class', 'btn')
                .style('font-size', '0.85em')
                .style('padding', '6px 12px')
                .style('background', '#95a5a6')
                .text('Collapse All')
                .on('click', () => {
                    if (root.children) {
                        root.children.forEach(collapse);
                    }
                    update(root);
                });

            let i = 0;
            const duration = 750;
            let root;

            // Create tree layout
            const treeLayout = d3.tree().nodeSize([25, 200]);

            // Create hierarchy
            root = d3.hierarchy(treeData, d => d.children);
            root.x0 = height / 2;
            root.y0 = 0;

            // Collapse children initially
            if (root.children) {
                root.children.forEach(collapse);
            }

            update(root);

            function collapse(d) {
                if (d.children) {
                    d._children = d.children;
                    d._children.forEach(collapse);
                    d.children = null;
                }
            }

            function expandAll(d) {
                if (d._children) {
                    d.children = d._children;
                    d._children = null;
                }
                if (d.children) {
                    d.children.forEach(expandAll);
                }
            }

            function update(source) {
                // Compute new tree layout
                const treeData = treeLayout(root);
                const nodes = treeData.descendants();
                const links = treeData.descendants().slice(1);

                // Normalize for fixed-depth
                nodes.forEach(d => { d.y = d.depth * 250; });

                // Update nodes
                const node = g.selectAll('g.tree-node')
                    .data(nodes, d => d.id || (d.id = ++i));

                // Enter new nodes
                const nodeEnter = node.enter().append('g')
                    .attr('class', 'tree-node')
                    .attr('transform', d => `translate(${source.y0},${source.x0})`)
                    .on('click', click)
                    .on('mouseover', function(event, d) {
                        const tooltip = d3.select('#treeTooltip');
                        tooltip.transition().duration(200).style('opacity', 0.9);
                        tooltip.html(`
                            <strong>${d.data.title}</strong><br/>
                            <em>${d.data.source_type}</em><br/>
                            Depth: ${d.data.depth}<br/>
                            ${d.data.url ? `URL: ${d.data.url.substring(0, 50)}...` : ''}
                        `)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                    })
                    .on('mouseout', function() {
                        d3.select('#treeTooltip').transition().duration(500).style('opacity', 0);
                    });

                // Add circles
                nodeEnter.append('circle')
                    .attr('r', 1e-6)
                    .style('fill', d => {
                        if (d.data.source_type === 'homepage') return '#667eea';
                        if (d.data.source_type === 'heading') return '#f093fb';
                        return d._children ? '#4facfe' : '#fff';
                    });

                // Add labels
                nodeEnter.append('text')
                    .attr('dy', '.35em')
                    .attr('x', d => d.children || d._children ? -13 : 13)
                    .attr('text-anchor', d => d.children || d._children ? 'end' : 'start')
                    .text(d => d.data.title.length > 30 ? d.data.title.substring(0, 30) + '...' : d.data.title);

                // Update
                const nodeUpdate = nodeEnter.merge(node);

                nodeUpdate.transition()
                    .duration(duration)
                    .attr('transform', d => `translate(${d.y},${d.x})`);

                nodeUpdate.select('circle')
                    .attr('r', 6)
                    .style('fill', d => {
                        if (d.data.source_type === 'homepage') return '#667eea';
                        if (d.data.source_type === 'heading') return '#f093fb';
                        return d._children ? '#4facfe' : '#fff';
                    });

                // Remove exiting nodes
                const nodeExit = node.exit().transition()
                    .duration(duration)
                    .attr('transform', d => `translate(${source.y},${source.x})`)
                    .remove();

                nodeExit.select('circle').attr('r', 1e-6);
                nodeExit.select('text').style('fill-opacity', 1e-6);

                // Update links
                const link = g.selectAll('path.tree-link')
                    .data(links, d => d.id);

                const linkEnter = link.enter().insert('path', 'g')
                    .attr('class', 'tree-link')
                    .attr('d', d => {
                        const o = {x: source.x0, y: source.y0};
                        return diagonal(o, o);
                    });

                const linkUpdate = linkEnter.merge(link);

                linkUpdate.transition()
                    .duration(duration)
                    .attr('d', d => diagonal(d, d.parent));

                const linkExit = link.exit().transition()
                    .duration(duration)
                    .attr('d', d => {
                        const o = {x: source.x, y: source.y};
                        return diagonal(o, o);
                    })
                    .remove();

                // Store old positions
                nodes.forEach(d => {
                    d.x0 = d.x;
                    d.y0 = d.y;
                });

                function diagonal(s, d) {
                    return `M ${s.y} ${s.x}
                            C ${(s.y + d.y) / 2} ${s.x},
                              ${(s.y + d.y) / 2} ${d.x},
                              ${d.y} ${d.x}`;
                }

                function click(event, d) {
                    if (d.children) {
                        d._children = d.children;
                        d.children = null;
                    } else {
                        d.children = d._children;
                        d._children = null;
                    }
                    update(d);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════
        // JSON FILE MANAGEMENT
        // ═══════════════════════════════════════════════════════════

        async function loadJsonFiles() {
            try {
                const response = await fetch('/api/json-files/list');
                const data = await response.json();

                const selector = document.getElementById('jsonFileSelector');
                selector.innerHTML = '';

                // Populate dropdown
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file.filename;
                    option.textContent = `${file.filename} (${file.topics} topics, ${file.qa_pairs} Q&A)`;
                    option.selected = file.is_active;
                    selector.appendChild(option);
                });

                // Update settings display
                const settingsInput = document.getElementById('settingsJsonPath');
                if (settingsInput) {
                    settingsInput.value = data.current;
                }
            } catch (error) {
                console.error('Error loading JSON files:', error);
            }
        }

        async function switchJsonFile() {
            const selector = document.getElementById('jsonFileSelector');
            const selectedFile = selector.value;

            if (!selectedFile) return;

            try {
                showStatus(`Switching to ${selectedFile}... This will reload the chatbot engine.`, 'info');

                const response = await fetch('/api/json-files/switch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ filename: selectedFile })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to switch JSON file');
                }

                const result = await response.json();
                showStatus(`✅ ${result.message} - ${result.topics} topics, ${result.qa_pairs} Q&A pairs`, 'success');

                // Reload the page to refresh all data
                setTimeout(() => {
                    location.reload();
                }, 1500);

            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
                // Reload file list to restore previous selection
                loadJsonFiles();
            }
        }

        async function uploadJsonFile(event) {
            const file = event.target.files[0];
            if (!file) return;

            // Validate file type
            if (!file.name.endsWith('.json')) {
                showStatus('❌ Please select a .json file', 'error');
                event.target.value = ''; // Reset input
                return;
            }

            try {
                showStatus(`Uploading ${file.name}... Please wait.`, 'info');

                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/api/json-files/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.detail || 'Upload failed');
                }

                showStatus(`✅ ${result.message} - ${result.topics} topics, ${result.qa_pairs} Q&A pairs`, 'success');

                // Reload JSON file list
                await loadJsonFiles();

                // Ask user if they want to switch to the new file
                if (confirm(`File uploaded successfully! Would you like to switch to "${result.filename}" now?`)) {
                    // Update selector and trigger switch
                    const selector = document.getElementById('jsonFileSelector');
                    selector.value = result.filename;
                    await switchJsonFile();
                }

            } catch (error) {
                showStatus('❌ Upload error: ' + error.message, 'error');
            } finally {
                // Reset file input
                event.target.value = '';
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            console.log('StarshipChatbot UI loaded!');
            loadJsonFiles(); // Load JSON file list on startup
        });
    </script>
</body>
</html>
    """

    return HTMLResponse(content=html_content)


# Serve static files from frontend build (vite.svg, etc.)
@app.get("/{filename}")
async def serve_static_file(filename: str):
    """Serve static files from frontend build root (like vite.svg)"""
    if USE_BUILT_FRONTEND and not filename.startswith("api"):
        file_path = os.path.join(FRONTEND_BUILD_DIR, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
    # If not found or API route, let other routes handle it
    raise HTTPException(status_code=404, detail="Not found")


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/api/dashboard")
async def get_dashboard():
    """Get system dashboard statistics"""
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    return {
        'total_topics': len(chatbot_engine.dataset.topics),
        'total_qa_pairs': len(chatbot_engine.dataset.all_qa_pairs),
        'chatbot_ready': chatbot_engine is not None,
        'editor_ready': qa_modifier is not None,
        'generator_ready': True,
        'generation_running': browser_runner is not None and browser_runner.is_running() if browser_runner else False,
        'rephrasing_enabled': chatbot_engine.rephraser is not None if chatbot_engine else False,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# DATA GENERATION ENDPOINTS
# ============================================================================

@app.post("/api/generate/start")
async def start_generation(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start browser agent data generation in background"""
    global browser_runner, generation_task

    # Check if already running
    if browser_runner and browser_runner.is_running():
        raise HTTPException(status_code=400, detail="Generation already running")

    # Validate inputs based on mode
    if request.use_crawler:
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required when using crawler")
        logger.info(f"Starting generation with crawler for URL: {request.url}")
    else:
        if not request.json_filename:
            raise HTTPException(status_code=400, detail="JSON filename is required when not using crawler")
        json_path = get_data_path(request.json_filename)
        if not os.path.exists(json_path):
            raise HTTPException(status_code=400, detail=f"JSON file not found: {request.json_filename}")
        logger.info(f"Starting generation from JSON file: {request.json_filename}")

    # Create output filename (use custom or auto-generate with timestamp)
    if request.output_filename:
        output_filename = get_data_path(request.output_filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = get_data_path(f"generated_{timestamp}.json")
    logger.info(f"Output will be saved to: {output_filename}")

    # Create runner with output filename
    browser_runner = create_runner(
        output_file=output_filename,
        use_mock=False  # Real mode - actually scrapes websites
    )

    # Start background task
    async def run_generation():
        try:
            async for progress in browser_runner.run(
                url=request.url or "",
                max_pages=request.max_pages,
                use_crawler=request.use_crawler,
                max_depth=request.max_depth,
                max_items=request.max_items,
                thread_id=request.thread_id,
                enable_checkpointing=request.enable_checkpointing,
                json_filename=request.json_filename
            ):
                logger.debug(f"Generation progress: {progress['status']} - {progress['current']}/{progress['total']}")
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)

    generation_task = asyncio.create_task(run_generation())

    return {
        'message': 'Generation started',
        'url': request.url,
        'max_pages': request.max_pages,
        'use_crawler': request.use_crawler
    }


@app.get("/api/generate/stream")
async def generation_stream():
    """SSE endpoint for real-time generation progress.
    Streams from the new ParallelProgressTracker if available, otherwise falls back to polling.
    Loops across batches — when one tracker finishes, waits for the next one.
    """

    async def event_generator():
        """Generate SSE events with progress updates"""
        try:
            # Give the tracker a moment to be created if generation just started
            await asyncio.sleep(0.5)

            # Track which tracker instances we've already consumed
            last_tracker = None

            while True:
                tracker = browser_agent.progress_tracker

                # If we have an active tracker we haven't consumed yet, stream from it
                if tracker and tracker._active and tracker is not last_tracker:
                    logger.info(f"SSE stream connected to ParallelProgressTracker (batch {tracker.current_batch}/{tracker.total_batches}).")
                    last_tracker = tracker
                    try:
                        async for update in tracker.get_updates():
                            yield {"data": update}
                        logger.info("SSE stream: batch tracker finished, waiting for next...")
                        # Send a transition event so the frontend knows we're between batches
                        if tracker.current_batch < tracker.total_batches:
                            yield {"data": json.dumps({
                                "type": "batch_transition",
                                "completed_batch": tracker.current_batch,
                                "total_batches": tracker.total_batches,
                                "message": f"Batch {tracker.current_batch}/{tracker.total_batches} complete. Starting next batch..."
                            })}
                    except asyncio.CancelledError:
                        logger.info("SSE stream (tracker) cancelled by client.")
                        return
                    except Exception as e:
                        logger.error(f"SSE stream (tracker) error: {e}", exc_info=True)
                        yield {"event": "error", "data": json.dumps({'error': str(e)})}
                    continue  # Loop back to check for next batch tracker

                # Check if the generation task is done (no more batches coming)
                if browser_runner:
                    progress = browser_runner.get_progress()
                    if progress['status'] in ['completed', 'error', 'cancelled']:
                        logger.info(f"Generation {progress['status']} — closing SSE stream")
                        yield {"data": json.dumps(progress)}
                        break

                # Wait for next tracker or task completion
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled by client.")
        except Exception as e:
            logger.error(f"Top-level SSE event_generator error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({'error': str(e)})
            }

    return EventSourceResponse(event_generator())


@app.get("/api/generate/status")
async def get_generation_status():
    """Get current generation status (polling alternative to SSE)"""
    if not browser_runner:
        return {
            'status': 'idle',
            'message': 'No generation running'
        }

    return browser_runner.get_progress()


@app.post("/api/generate/cancel")
async def cancel_generation(request: CancelRequest):
    """Cancel running generation gracefully with option to save data."""
    global browser_runner

    if not browser_runner or not browser_runner.is_running():
        raise HTTPException(status_code=400, detail="No generation running")

    logger.info(f"Requesting cancellation (save_data={request.save_data})...")
    result = browser_runner.cancel(save_data=request.save_data)

    if request.save_data:
        return {
            'message': 'Cancellation requested. Data will be saved.',
            'saved': True,
            'output_file': result.get('output_file') if result else None
        }
    else:
        return {
            'message': 'Cancellation requested. Data discarded.',
            'saved': False
        }


# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat question and return answer (V3 > V2 > V1 based on what's enabled)"""
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    try:
        # Use V3 if enabled, then V2, then V1
        if hasattr(chatbot_engine, 'v3_enabled') and chatbot_engine.v3_enabled:
            result = chatbot_engine.process_question_v3(request.question)
        elif hasattr(chatbot_engine, 'v2_enabled') and chatbot_engine.v2_enabled:
            result = chatbot_engine.process_question_v2(request.question)
        else:
            result = chatbot_engine.process_question(
                user_question=request.question,
                session_id=request.session_id
            )

        # Handle source_qa being either dict or object
        source_qa = result.get('source_qa')
        if isinstance(source_qa, dict):
            source_question = source_qa.get('question')
            source_topic = source_qa.get('topic')
        elif source_qa:
            source_question = source_qa.question
            source_topic = getattr(source_qa, 'topic', None)
        else:
            source_question = None
            source_topic = None

        return {
            'answer': result['answer'],
            'matched_by': result['matched_by'],
            'confidence': result['confidence'],
            'source_question': source_question,
            'source_topic': result.get('source_topic') or source_topic,
            'source_url': result.get('source_url'),
            'suggested_questions': result.get('suggested_questions'),
            'pipeline_info': result.get('pipeline_info', {})
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/stats")
async def get_chat_stats():
    """Get chatbot statistics"""
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    return {
        'total_qa_pairs': len(chatbot_engine.dataset.all_qa_pairs),
        'total_topics': len(chatbot_engine.dataset.topics),
        'rephrasing_enabled': chatbot_engine.rephraser is not None,
        'thresholds': {
            'ideal': chatbot_engine.search_engine.SIMILARITY_THRESHOLD_IDEAL,
            'minimal': chatbot_engine.search_engine.SIMILARITY_THRESHOLD
        }
    }


# ============================================================================
# EDITOR ENDPOINTS
# ============================================================================

@app.get("/api/editor/topics")
async def get_editor_topics():
    """Get all topics for editor"""
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    return chatbot_engine.get_all_topics()


@app.get("/api/editor/topics/{topic_index}")
async def get_topic_details(topic_index: int):
    """Get detailed Q&A pairs for a specific topic"""
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    if topic_index < 0 or topic_index >= len(chatbot_engine.dataset.topics):
        raise HTTPException(status_code=404, detail="Topic not found")

    topic = chatbot_engine.dataset.topics[topic_index]

    return {
        'topic': topic.topic,
        'url': topic.original_url,
        'qa_count': len(topic.qa_pairs),
        'qa_pairs': [
            {
                'question': qa.question,
                'answer': qa.answer,
                'qa_index': qa.qa_index,
                'is_bucketed': qa.is_bucketed,
                'bucket_id': qa.bucket_id
            }
            for qa in topic.qa_pairs
        ]
    }


@app.post("/api/editor/simplify")
async def simplify_topic(request: SimplifyRequest):
    """Simplify Q&A pairs in a topic using SimplifyAgent"""
    if not qa_modifier:
        raise HTTPException(
            status_code=503,
            detail="LangGraph editor not available. Install required dependencies."
        )

    try:
        # Load current JSON (supports both new and legacy format)
        json_path = current_json_file
        full_data, topics = load_qa_json(json_path)

        if request.topic_index < 0 or request.topic_index >= len(topics):
            raise HTTPException(status_code=404, detail="Topic not found")

        topic = topics[request.topic_index]
        topic_name = topic.get('topic', 'Unknown Topic')
        qa_pairs = topic.get('qa_pairs', [])

        if not qa_pairs:
            raise HTTPException(status_code=400, detail="No Q&A pairs to simplify")

        logger.info(f"Simplifying topic '{topic_name}' with {len(qa_pairs)} Q&A pairs")

        # Use SimplifyAgent to create bucketed Q&A pairs
        bucketed_pairs = await SimplifyAgent.dynamic_adjust(qa_pairs)

        # Update topic with bucketed pairs
        topic['qa_pairs'] = bucketed_pairs
        topic['qa_count'] = len(bucketed_pairs)

        # Save back to JSON (preserves tree if present)
        save_qa_json(json_path, full_data, topics)

        # Reload chatbot engine
        global chatbot_engine
        chatbot_engine = JSONChatbotEngine(
            json_path=json_path,
            enable_rephrasing=os.getenv('GROQ_API_KEY') is not None
        )

        # Re-enable architecture after reload
        if USE_V2_ARCHITECTURE:
            try:
                chatbot_engine.enable_v2_architecture()
                if chatbot_engine.v2_enabled:
                    logger.info("✅ V2 re-enabled after simplify")
            except Exception as v2_error:
                logger.warning(f"⚠️ V2 re-enable failed: {v2_error}")
        else:
            logger.info("ℹ️  Using V1 sequential search architecture")

        # Count unique buckets
        unique_buckets = len(set(pair.get("bucket_id", "") for pair in bucketed_pairs if pair.get("is_bucketed")))

        logger.info(f"✅ Simplification complete: {len(bucketed_pairs)} Q&A pairs in {unique_buckets} buckets")

        return {
            'message': f'Successfully simplified topic "{topic_name}"',
            'topic_index': request.topic_index,
            'original_count': len(qa_pairs),
            'new_count': len(bucketed_pairs),
            'buckets_created': unique_buckets
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simplify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/editor/merge")
async def merge_qa_pairs(request: MergeRequest):
    """Merge multiple Q&A pairs into one using LangGraph workflow"""
    if not qa_modifier:
        raise HTTPException(status_code=503, detail="LangGraph editor not available")

    if len(request.qa_indices) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 Q&A pairs to merge")

    try:
        # Load current JSON (supports both new and legacy format)
        json_path = current_json_file
        full_data, topics = load_qa_json(json_path)

        if request.topic_index < 0 or request.topic_index >= len(topics):
            raise HTTPException(status_code=404, detail="Topic not found")

        topic = topics[request.topic_index]
        topic_name = topic.get('topic', 'Unknown Topic')
        qa_pairs = topic.get('qa_pairs', [])

        # Validate indices
        for idx in request.qa_indices:
            if idx < 0 or idx >= len(qa_pairs):
                raise HTTPException(status_code=400, detail=f"Invalid Q&A index: {idx}")

        logger.info(f"Merging {len(request.qa_indices)} Q&A pairs from topic '{topic_name}'")

        # Use QAWorkflowManager to merge Q&A pairs
        result = await qa_modifier.merge_qa_pairs(
            topic_index=request.topic_index,
            selected_qa_indices=request.qa_indices,
            all_data=topics,
            user_request=request.user_request
        )

        if result.get('error'):
            raise HTTPException(status_code=500, detail=result['error'])

        # Save the modified data (preserves tree if present)
        modified_topics = result.get('modified_data', topics)
        save_qa_json(json_path, full_data, modified_topics)

        # Reload chatbot engine
        global chatbot_engine
        chatbot_engine = JSONChatbotEngine(
            json_path=json_path,
            enable_rephrasing=os.getenv('GROQ_API_KEY') is not None
        )

        # Re-enable architecture after reload
        if USE_V2_ARCHITECTURE:
            try:
                chatbot_engine.enable_v2_architecture()
                if chatbot_engine.v2_enabled:
                    logger.info("✅ V2 re-enabled after merge")
            except Exception as v2_error:
                logger.warning(f"⚠️ V2 re-enable failed: {v2_error}")
        else:
            logger.info("ℹ️  Using V1 sequential search architecture")

        logger.info(f"✅ Merge complete: {result.get('agent_response')}")

        return {
            'message': result.get('agent_response', 'Merge completed'),
            'topic_index': request.topic_index,
            'qa_indices': request.qa_indices,
            'merged_count': len(request.qa_indices),
            'new_total': len(modified_topics[request.topic_index].get('qa_pairs', []))
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/editor/delete")
async def delete_qa_pairs(request: DeleteRequest):
    """Delete Q&A pairs from a topic"""
    try:
        # Load current JSON (supports both new and legacy format)
        json_path = current_json_file
        full_data, topics = load_qa_json(json_path)

        if request.topic_index < 0 or request.topic_index >= len(topics):
            raise HTTPException(status_code=404, detail="Topic not found")

        topic = topics[request.topic_index]

        # Delete specified Q&A pairs
        qa_pairs = topic.get('qa_pairs', [])

        # Sort indices in reverse to delete from end first
        for idx in sorted(request.qa_indices, reverse=True):
            if 0 <= idx < len(qa_pairs):
                qa_pairs.pop(idx)

        # Save back (preserves tree if present)
        save_qa_json(json_path, full_data, topics)

        # Reload chatbot engine
        global chatbot_engine
        chatbot_engine = JSONChatbotEngine(
            json_path=json_path,
            enable_rephrasing=os.getenv('GROQ_API_KEY') is not None
        )

        # Re-enable architecture after reload
        if USE_V2_ARCHITECTURE:
            try:
                logger.info("Re-enabling V2 parallel-fused architecture...")
                chatbot_engine.enable_v2_architecture()
                if chatbot_engine.v2_enabled:
                    logger.info("✅ V2 architecture re-enabled after delete")
            except Exception as v2_error:
                logger.warning(f"⚠️ V2 re-enable failed: {v2_error}")
        else:
            logger.info("ℹ️  Using V1 sequential search architecture")

        return {
            'message': f'Deleted {len(request.qa_indices)} Q&A pairs',
            'deleted_count': len(request.qa_indices)
        }
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/editor/edit")
async def edit_qa_pair(request: EditRequest):
    """Edit a single Q&A pair"""
    try:
        # Load current JSON (supports both new and legacy format)
        json_path = current_json_file
        full_data, topics = load_qa_json(json_path)

        if request.topic_index < 0 or request.topic_index >= len(topics):
            raise HTTPException(status_code=404, detail="Topic not found")

        topic = topics[request.topic_index]
        qa_pairs = topic.get('qa_pairs', [])

        if request.qa_index < 0 or request.qa_index >= len(qa_pairs):
            raise HTTPException(status_code=404, detail="Q&A pair not found")

        # Update Q&A pair
        qa_pairs[request.qa_index]['question'] = request.new_question
        qa_pairs[request.qa_index]['answer'] = request.new_answer

        # Save (preserves tree if present)
        save_qa_json(json_path, full_data, topics)

        # Reload chatbot
        global chatbot_engine
        chatbot_engine = JSONChatbotEngine(
            json_path=json_path,
            enable_rephrasing=os.getenv('GROQ_API_KEY') is not None
        )

        # Re-enable architecture after reload
        if USE_V2_ARCHITECTURE:
            try:
                logger.info("Re-enabling V2 parallel-fused architecture...")
                chatbot_engine.enable_v2_architecture()
                if chatbot_engine.v2_enabled:
                    logger.info("✅ V2 architecture re-enabled after edit")
            except Exception as v2_error:
                logger.warning(f"⚠️ V2 re-enable failed: {v2_error}")
        else:
            logger.info("ℹ️  Using V1 sequential search architecture")

        return {
            'message': 'Q&A pair updated successfully',
            'topic_index': request.topic_index,
            'qa_index': request.qa_index
        }
    except Exception as e:
        logger.error(f"Edit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# JSON FILE MANAGEMENT
# ============================================================================

@app.get("/api/json-files/list")
async def list_json_files():
    """List all available JSON files in the data directory"""
    try:
        import glob

        # Get all JSON files in DATA_DIR (persistent storage)
        search_pattern = os.path.join(DATA_DIR, "*.json")
        json_files = glob.glob(search_pattern)

        # Get just filenames (not full paths)
        json_files = [os.path.basename(f) for f in json_files]

        # Filter out system/config files
        excluded_files = ['package.json', 'package-lock.json', 'tsconfig.json', 'checkpoint_progress.json']
        json_files = [f for f in json_files if f not in excluded_files]

        # Sort alphabetically
        json_files.sort()

        # Get file stats
        file_info = []
        for filename in json_files:
            try:
                # Use full path to read from DATA_DIR
                filepath = get_data_path(filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Count topics and Q&A pairs (supports both formats)
                if isinstance(data, dict) and 'topics' in data:
                    topics_list = data['topics']
                elif isinstance(data, list):
                    topics_list = data
                else:
                    topics_list = []

                topics_count = len(topics_list)
                qa_count = sum(len(topic.get('qa_pairs', [])) for topic in topics_list if isinstance(topic, dict))

                # Compare basename for is_active
                file_info.append({
                    'filename': filename,
                    'topics': topics_count,
                    'qa_pairs': qa_count,
                    'is_active': get_data_path(filename) == current_json_file
                })
            except Exception as e:
                logger.warning(f"Could not read {filename}: {e}")
                continue

        return {
            'files': file_info,
            'current': os.path.basename(current_json_file)
        }
    except Exception as e:
        logger.error(f"Error listing JSON files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/json-files/switch")
async def switch_json_file(request: SwitchFileRequest):
    """Switch to a different JSON file"""
    global current_json_file, chatbot_engine

    try:
        filename = request.filename

        # Get full path in DATA_DIR
        filepath = get_data_path(filename)

        # Validate file exists
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in data directory")

        # Validate it's a valid JSON file with Q&A data (supports both formats)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check for valid format: either array or object with 'topics' key
        is_valid = isinstance(data, list) or (isinstance(data, dict) and 'topics' in data)
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid JSON format: expected array of topics or object with 'topics' key")

        logger.info(f"Switching from '{os.path.basename(current_json_file)}' to '{filename}'")

        # Update current file (store full path)
        current_json_file = filepath

        # Reload chatbot engine with new file (this will rebuild the pickle cache)
        chatbot_engine = JSONChatbotEngine(
            json_path=filepath,
            enable_rephrasing=os.getenv('GROQ_API_KEY') is not None
        )

        # Enable architecture based on USE_V2_ARCHITECTURE flag
        if USE_V2_ARCHITECTURE:
            try:
                logger.info("Enabling V2 parallel-fused architecture...")
                chatbot_engine.enable_v2_architecture()
                if chatbot_engine.v2_enabled:
                    logger.info("✅ V2 architecture enabled")
                else:
                    logger.info("⚠️  V2 architecture not available - using V1")
            except Exception as v2_error:
                logger.warning(f"⚠️  V2 architecture failed: {v2_error} - using V1")
        else:
            logger.info("ℹ️  Using V1 sequential search architecture")

        # Count stats
        topics_count = len(data)
        qa_count = sum(len(topic.get('qa_pairs', [])) for topic in data if isinstance(topic, dict))

        logger.info(f"✅ Switched to '{filename}' - {topics_count} topics, {qa_count} Q&A pairs")

        return {
            'message': f'Successfully switched to {filename}',
            'filename': filename,
            'topics': topics_count,
            'qa_pairs': qa_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching JSON file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/json-files/current")
async def get_current_json_file():
    """Get the currently active JSON file"""
    return {
        'filename': current_json_file,
        'chatbot_loaded': chatbot_engine is not None
    }


@app.post("/api/json-files/upload")
async def upload_json_file(file: UploadFile = File(...)):
    """Upload a new JSON file to the project directory"""
    try:
        logger.info(f"📤 Upload request received - filename: {file.filename}, content_type: {file.content_type}")

        # Validate filename
        filename = file.filename
        if not filename or not filename.endswith('.json'):
            logger.warning(f"❌ Upload rejected - invalid filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be a .json file")

        # Security: prevent path traversal
        filename = os.path.basename(filename)

        # Get full path in DATA_DIR
        filepath = get_data_path(filename)

        # Check if file already exists
        if os.path.exists(filepath):
            raise HTTPException(status_code=409, detail=f"File '{filename}' already exists. Please rename and try again.")

        # Read and validate JSON content
        content = await file.read()
        logger.info(f"📄 File size: {len(content)} bytes")

        try:
            data = json.loads(content.decode('utf-8'))
            logger.info(f"✅ JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError:
            logger.error(f"❌ UTF-8 decode error")
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

        # Validate structure (supports both new and legacy format)
        if isinstance(data, dict) and 'topics' in data:
            topics_list = data['topics']
            logger.info(f"📦 New format detected (tree embedded)")
        elif isinstance(data, list):
            topics_list = data
            logger.info(f"📦 Legacy format detected (array)")
        else:
            logger.error(f"❌ Invalid structure: data is {type(data).__name__}")
            raise HTTPException(status_code=400, detail="JSON must be an array of topics or object with 'topics' key")

        if len(topics_list) == 0:
            logger.error(f"❌ Empty topics array")
            raise HTTPException(status_code=400, detail="JSON file cannot have empty topics")

        logger.info(f"📊 Found {len(topics_list)} topics in uploaded file")

        # Validate each topic has required fields
        for idx, topic in enumerate(topics_list):
            if not isinstance(topic, dict):
                raise HTTPException(status_code=400, detail=f"Topic at index {idx} must be an object")
            if 'qa_pairs' not in topic:
                raise HTTPException(status_code=400, detail=f"Topic at index {idx} missing 'qa_pairs' field")
            if not isinstance(topic['qa_pairs'], list):
                raise HTTPException(status_code=400, detail=f"Topic at index {idx}: 'qa_pairs' must be an array")

        # Save the file to DATA_DIR
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Count stats
        topics_count = len(data)
        qa_count = sum(len(topic.get('qa_pairs', [])) for topic in data if isinstance(topic, dict))

        logger.info(f"✅ Uploaded new file '{filename}' to data directory - {topics_count} topics, {qa_count} Q&A pairs")

        return {
            'message': f'Successfully uploaded {filename}',
            'filename': filename,
            'topics': topics_count,
            'qa_pairs': qa_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading JSON file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class DeleteFileRequest(BaseModel):
    """Delete file request"""
    filename: str


@app.post("/api/json-files/delete")
async def delete_json_file(request: DeleteFileRequest):
    """Delete a JSON file from DATA_DIR (Railway volume)"""
    global current_json_file

    try:
        filename = request.filename

        # Security: prevent path traversal
        filename = os.path.basename(filename)

        # Prevent deleting currently active file
        if get_data_path(filename) == current_json_file:
            raise HTTPException(status_code=400, detail=f"Cannot delete active file '{filename}'. Switch to another file first.")

        # Get full path in DATA_DIR (Railway volume)
        filepath = get_data_path(filename)

        logger.info(f"🗑️ Delete request for: {filepath}")
        logger.info(f"   DATA_DIR: {DATA_DIR}")

        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in data directory")

        # Delete the JSON file
        os.remove(filepath)
        logger.info(f"✅ Deleted: {filepath}")

        # Also delete associated cache files
        deleted_caches = []
        cache_patterns = [
            filepath.replace('.json', '_qa_cache.pkl'),
            filepath.replace('.json', '_metadata_cache.pkl'),
        ]
        for cache_file in cache_patterns:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                deleted_caches.append(os.path.basename(cache_file))
                logger.info(f"🗑️ Deleted cache: {cache_file}")

        return {
            'message': f'Successfully deleted {filename}',
            'filename': filename,
            'deleted_caches': deleted_caches
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CRAWLER FILE MANAGEMENT
# ============================================================================

@app.get("/api/crawler-files/list")
async def list_crawler_files():
    """List all hierarchical crawler JSON output files"""
    try:
        import glob

        # Search in both output directories
        search_patterns = [
            os.path.join(DATA_DIR, "output", "hierarchical_crawl_*.json"),
            os.path.join(DATA_DIR, "hierarchical_crawl_*.json"),
            "output/hierarchical_crawl_*.json",  # Local development
            "hierarchical_crawl_*.json"
        ]

        crawler_files = []
        for pattern in search_patterns:
            found = glob.glob(pattern)
            crawler_files.extend(found)

        # Get just filenames and remove duplicates
        crawler_files = list(set([os.path.basename(f) for f in crawler_files]))

        # Filter out _tree.json files (we want the main files only)
        crawler_files = [f for f in crawler_files if not f.endswith('_tree.json')]

        # Sort by most recent first (based on timestamp in filename)
        crawler_files.sort(reverse=True)

        logger.info(f"📁 Found {len(crawler_files)} crawler output files")

        return {
            'files': crawler_files,
            'count': len(crawler_files)
        }
    except Exception as e:
        logger.error(f"Error listing crawler files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crawler-files/preview")
async def preview_crawler_file(filename: str):
    """Get preview/metadata of a crawler JSON file"""
    try:
        # Try to find the file in various locations
        possible_paths = [
            os.path.join(DATA_DIR, "output", filename),
            os.path.join(DATA_DIR, filename),
            os.path.join("output", filename),
            os.path.join("WorkingFiles", "output", filename),
            filename
        ]

        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

        if not filepath:
            raise HTTPException(status_code=404, detail=f"Crawler file not found: {filename}")

        # Read and extract metadata
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract metadata
        metadata = data.get('crawl_metadata', {})

        return {
            'filename': filename,
            'domain': metadata.get('domain', 'Unknown'),
            'total_elements': metadata.get('total_elements', 0),
            'timestamp': metadata.get('timestamp', ''),
            'crawl_type': metadata.get('crawl_type', ''),
            'statistics': metadata.get('statistics', {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing crawler file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/volume/diagnostics")
async def volume_diagnostics():
    """
    Check Railway volume status and contents
    Useful for debugging persistent storage issues
    """
    try:
        import glob

        # Get volume information
        diagnostics = {
            "data_dir": DATA_DIR,
            "data_dir_exists": os.path.exists(DATA_DIR),
            "data_dir_writable": os.access(DATA_DIR, os.W_OK) if os.path.exists(DATA_DIR) else False,
            "current_json_file": current_json_file,
            "current_json_exists": os.path.exists(current_json_file) if current_json_file else False,
        }

        # List all files in DATA_DIR
        if os.path.exists(DATA_DIR):
            all_files = os.listdir(DATA_DIR)

            # Separate by type
            json_files = [f for f in all_files if f.endswith('.json')]
            db_files = [f for f in all_files if f.endswith('.db')]
            other_files = [f for f in all_files if not (f.endswith('.json') or f.endswith('.db'))]

            # Get file sizes
            file_details = []
            for filename in all_files:
                filepath = os.path.join(DATA_DIR, filename)
                try:
                    size = os.path.getsize(filepath)
                    modified = os.path.getmtime(filepath)
                    from datetime import datetime
                    modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')

                    file_details.append({
                        "name": filename,
                        "size_bytes": size,
                        "size_kb": round(size / 1024, 2),
                        "modified": modified_str
                    })
                except Exception as e:
                    file_details.append({
                        "name": filename,
                        "error": str(e)
                    })

            # Calculate total size
            total_size = sum(f.get("size_bytes", 0) for f in file_details)

            diagnostics.update({
                "total_files": len(all_files),
                "json_files_count": len(json_files),
                "db_files_count": len(db_files),
                "other_files_count": len(other_files),
                "json_files": json_files,
                "db_files": db_files,
                "other_files": other_files,
                "file_details": file_details,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            })
        else:
            diagnostics["error"] = f"DATA_DIR does not exist: {DATA_DIR}"

        return diagnostics

    except Exception as e:
        logger.error(f"Volume diagnostics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TREE VISUALIZATION
# ============================================================================

@app.get("/api/tree/data")
async def get_tree_data():
    """Get hierarchical tree structure for D3.js visualization"""
    try:
        import glob
        
        # Find all tree JSON files - check multiple locations for compatibility
        tree_files = []
        # Production: DATA_DIR root (e.g., /data/*_tree.json)
        tree_files.extend(glob.glob(os.path.join(DATA_DIR, "*_tree.json")))
        # Production: DATA_DIR/output subdirectory
        tree_files.extend(glob.glob(os.path.join(DATA_DIR, "output", "*_tree.json")))
        # Local development: ./output/ directory
        if DATA_DIR != "output":
            tree_files.extend(glob.glob("output/*_tree.json"))
        
        if not tree_files:
            raise HTTPException(status_code=404, detail="No tree data found. Generate data first.")
        
        # Get the most recent tree file
        latest_tree_file = max(tree_files, key=os.path.getctime)
        
        logger.info(f"Loading tree data from: {latest_tree_file}")
        
        # Load and return tree data
        with open(latest_tree_file, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        return tree_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading tree data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tree/list")
async def list_tree_files():
    """List all available tree visualization files"""
    try:
        import glob
        from datetime import datetime

        # Find all tree JSON files - check multiple locations for compatibility
        tree_files = []
        # Production: DATA_DIR root (e.g., /data/*_tree.json)
        tree_files.extend(glob.glob(os.path.join(DATA_DIR, "*_tree.json")))
        # Production: DATA_DIR/output subdirectory
        tree_files.extend(glob.glob(os.path.join(DATA_DIR, "output", "*_tree.json")))
        # Local development: ./output/ directory
        if DATA_DIR != "output":
            tree_files.extend(glob.glob("output/*_tree.json"))
        
        file_info = []
        for filepath in tree_files:
            try:
                # Get file stats
                stat = os.stat(filepath)
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                
                # Load metadata
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                
                file_info.append({
                    'filename': os.path.basename(filepath),
                    'domain': metadata.get('domain', 'Unknown'),
                    'total_nodes': metadata.get('total_nodes', 0),
                    'modified': modified_time.isoformat(),
                    'path': filepath
                })
            except Exception as e:
                logger.warning(f"Could not read tree file {filepath}: {e}")
                continue
        
        # Sort by modified time (newest first)
        file_info.sort(key=lambda x: x['modified'], reverse=True)
        
        return {'files': file_info}
    
    except Exception as e:
        logger.error(f"Error listing tree files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health():
    """System health check"""
    return {
        'status': 'healthy',
        'chatbot': chatbot_engine is not None,
        'editor': qa_modifier is not None,
        'generator': True,
        'generation_running': browser_runner.is_running() if browser_runner else False,
        'current_json_file': current_json_file,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('PORT', 8000))

    print("\n" + "="*60)
    print("🚀 Starting StarshipChatbot Unified Server")
    print("="*60)
    print(f"   Server will run on: http://localhost:{port}")
    print(f"   API docs: http://localhost:{port}/docs")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
