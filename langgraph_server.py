"""
LangGraph FastAPI Server
=========================

FastAPI server that integrates with LangGraph chatbot backend.

Endpoints:
- GET / - Serve HTML viewer with JSON data
- POST /api/chat - Send query, get intelligent response from JSON data
- GET /api/chat/history - Get conversation history
- DELETE /api/chat/history - Clear conversation history
- GET /api/health - Health check

Features:
- Async request handling
- CORS enabled for frontend
- Session management
- Error handling
- Dynamically generates HTML viewer from JSON
- LangGraph agent searches JSON data to answer questions
"""

import logging
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from langgraph_chatbot import (
    QAWorkflowManager,
    build_merge_qa_graph,
    QADataset,  # Class-based data navigation
    SimplifyAgent  # Simplify Q&A pairs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    Manages graph initialization (NO checkpointing - uses JSON file)
    """
    logger.info("="*60)
    logger.info("LANGGRAPH CHATBOT SERVER - STARTUP")
    logger.info("="*60)

    # Build merge QA graph (3-agent, uses MergeQAState)
    # This handles merging 2+ Q&A pairs into 1 using Groq LLM
    merge_graph = build_merge_qa_graph()
    logger.info("✓ Merge QA graph (3-agent) compiled (MergeQAState)")

    # QAWorkflowManager manages merge workflow
    qa_modifier = QAWorkflowManager(
        merge_graph=merge_graph
    )

    # Store in app state
    app.state.qa_modifier = qa_modifier

    # Setup data directory (Railway volume support)
    import os
    DATA_DIR = Path(os.getenv("DATA_DIR", "."))
    JSON_FILE_PATH = DATA_DIR / "browser_agent_test_output.json"

    # Store globally for use in endpoints
    app.state.json_file_path = JSON_FILE_PATH
    app.state.data_dir = DATA_DIR

    logger.info(f"📁 Data directory: {DATA_DIR}")
    logger.info(f"📄 JSON file path: {JSON_FILE_PATH}")

    # Verify JSON file exists
    if JSON_FILE_PATH.exists():
        # Load and show stats
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        logger.info(f"✅ JSON file found: {len(json_data)} topics")
        logger.info("📁 Data source: browser_agent_test_output.json")
    else:
        logger.warning("⚠️  JSON file not found - creating empty dataset")
        # Create empty JSON file with sample structure
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        logger.info("✅ Created empty JSON file")

    logger.info("🚀 Server ready at: http://localhost:8000")
    logger.info("📚 API docs at: http://localhost:8000/docs")
    logger.info("="*60)

    yield  # Server runs here

    # Cleanup on shutdown
    logger.info("="*60)
    logger.info("LANGGRAPH CHATBOT SERVER - SHUTDOWN")
    logger.info("="*60)


# ============================================================================
# INITIALIZE FASTAPI APP WITH LIFESPAN
# ============================================================================

app = FastAPI(
    title="LangGraph Chatbot Server",
    description="FastAPI server with LangGraph + Groq LLM integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class MergeQARequest(BaseModel):
    """Request model for merge Q&A endpoint"""
    user_request: str
    topic_index: int
    selected_qa_indices: List[int]  # Must have 2 or more indices


class SimplifyTopicRequest(BaseModel):
    """Request model for simplify topic endpoint"""
    topic_index: int  # Which topic to simplify (e.g., 0, 1, 2...)


class SimplifyTopicResponse(BaseModel):
    """Response model for simplify topic endpoint"""
    message: str              # Success message like "Simplified 10 Q&A pairs"
    simplified_count: int     # How many Q&A pairs were simplified
    error: Optional[str] = None   # Error message if something goes wrong
    timestamp: str            # When the operation completed


class ModifyResponse(BaseModel):
    """Response model for modify endpoint"""
    modified_data: List[Dict]
    agent_response: str
    error: Optional[str] = None
    timestamp: str




# ============================================================================
# JSON TO HTML CONVERTER
# ============================================================================

# Path to JSON file - will be set dynamically from environment
# This is a placeholder, actual path is set in lifespan context


def generate_html_viewer(json_data: list) -> str:
    """
    Generate HTML viewer with embedded JSON data

    Args:
        json_data: List of topics from JSON file

    Returns:
        Complete HTML string with embedded data
    """
    # Convert JSON to JavaScript format
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q&A Content Viewer - LangGraph Server</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3); overflow: hidden; display: grid; grid-template-columns: 350px 1fr; min-height: 90vh; }}
        .sidebar {{ background: #f8f9fa; border-right: 2px solid #e9ecef; display: flex; flex-direction: column; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 20px; text-align: center; }}
        .header h1 {{ font-size: 1.5em; margin-bottom: 5px; }}
        .header p {{ font-size: 0.9em; opacity: 0.9; }}
        .search-box {{ padding: 20px; border-bottom: 2px solid #e9ecef; }}
        .search-input {{ width: 100%; padding: 12px 15px; border: 2px solid #e9ecef; border-radius: 8px; font-size: 1em; transition: border-color 0.3s ease; }}
        .search-input:focus {{ outline: none; border-color: #667eea; }}
        .topic-list {{ flex: 1; overflow-y: auto; padding: 10px; }}
        .topic-item {{ padding: 15px; margin-bottom: 8px; background: white; border-radius: 8px; transition: all 0.3s ease; border: 2px solid transparent; position: relative; }}
        .topic-item:hover {{ background: #f8f9fa; border-color: #667eea; transform: translateX(5px); }}
        .topic-item.active {{ background: #667eea; color: white; border-color: #667eea; }}
        .topic-name {{ font-weight: 600; font-size: 0.95em; margin-bottom: 5px; cursor: pointer; }}
        .topic-status {{ font-size: 0.8em; opacity: 0.7; }}
        .topic-modify-btn {{ margin-top: 8px; padding: 6px 12px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 600; width: 100%; transition: background 0.3s ease; }}
        .topic-modify-btn:hover {{ background: #218838; }}
        .topic-item.active .topic-modify-btn {{ background: #fff; color: #667eea; }}
        .topic-item.active .topic-modify-btn:hover {{ background: #f0f1ff; }}
        .topic-simplify-btn {{ margin-top: 8px; padding: 6px 12px; background: #007bff; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 600; width: 100%; transition: background 0.3s ease; }}
        .topic-simplify-btn:hover {{ background: #0056b3; }}
        .topic-item.active .topic-simplify-btn {{ background: #fff; color: #007bff; }}
        .topic-item.active .topic-simplify-btn:hover {{ background: #f0f1ff; }}
        .content-area {{ display: flex; flex-direction: column; overflow: hidden; }}
        .content-header {{ background: #f8f9fa; padding: 30px 40px; border-bottom: 2px solid #e9ecef; }}
        .content-title {{ font-size: 2em; color: #212529; margin-bottom: 10px; }}
        .content-meta {{ display: flex; gap: 15px; flex-wrap: wrap; }}
        .meta-badge {{ padding: 6px 12px; background: #667eea; color: white; border-radius: 12px; font-size: 0.85em; }}
        .content-body {{ flex: 1; overflow-y: auto; padding: 40px; }}
        .content-section {{ margin-bottom: 30px; }}
        .section-label {{ font-weight: 600; color: #495057; text-transform: uppercase; font-size: 0.9em; letter-spacing: 0.5px; margin-bottom: 15px; }}
        .section-content {{ background: #f8f9fa; padding: 20px; border-radius: 12px; line-height: 1.8; color: #212529; border-left: 4px solid #667eea; white-space: pre-wrap; }}
        .qa-container {{ background: #f8f9fa; border-radius: 12px; overflow: hidden; }}
        .qa-item {{ border-bottom: 1px solid #e9ecef; }}
        .qa-question {{ padding: 18px 20px; background: white; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; gap: 15px; border-left: 4px solid transparent; }}
        .qa-question:hover {{ background: #f8f9fa; border-left-color: #667eea; }}
        .qa-question.active {{ background: #f0f1ff; border-left-color: #667eea; }}
        .qa-number {{ background: #667eea; color: white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9em; flex-shrink: 0; }}
        .qa-question-text {{ flex: 1; font-weight: 600; color: #212529; }}
        .qa-toggle {{ color: #667eea; font-size: 1.2em; transition: transform 0.3s ease; }}
        .qa-question.active .qa-toggle {{ transform: rotate(180deg); }}
        .qa-answer {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease; background: #ffffff; }}
        .qa-answer.show {{ max-height: 1000px; }}
        .qa-answer-content {{ padding: 20px 20px 20px 67px; line-height: 1.8; color: #495057; }}
        .empty-state {{ display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #6c757d; text-align: center; padding: 40px; }}
        .empty-state-icon {{ font-size: 4em; margin-bottom: 20px; }}
        .qa-checkbox {{ width: 18px; height: 18px; cursor: pointer; margin-right: 10px; }}
        .modify-panel {{ background: #fff3cd; border: 2px solid #ffc107; border-radius: 12px; padding: 20px; margin-bottom: 20px; display: none; }}
        .modify-panel.active {{ display: block; }}
        .modify-panel h3 {{ margin: 0 0 15px 0; color: #856404; font-size: 1.2em; }}
        .modify-panel input {{ width: 100%; padding: 12px; border: 2px solid #ffc107; border-radius: 8px; font-size: 1em; margin-bottom: 10px; }}
        .modify-panel-buttons {{ display: flex; gap: 10px; }}
        .modify-submit-btn {{ flex: 1; padding: 12px; background: #28a745; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }}
        .modify-submit-btn:hover {{ background: #218838; }}
        .modify-submit-btn:disabled {{ background: #6c757d; cursor: not-allowed; }}
        .merge-btn {{ flex: 1; padding: 12px; background: #9b59b6; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }}
        .merge-btn:hover {{ background: #8e44ad; }}
        .merge-btn:disabled {{ background: #6c757d; cursor: not-allowed; }}
        .modify-cancel-btn {{ flex: 1; padding: 12px; background: #6c757d; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }}
        .modify-cancel-btn:hover {{ background: #5a6268; }}
        .selection-info {{ background: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9em; }}
        .unified-qa-box {{ background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 100%); border: 3px solid #667eea; border-radius: 15px; padding: 30px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15); }}
        .unified-questions-header {{ font-size: 1.3em; font-weight: 700; color: #667eea; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }}
        .unified-questions-list {{ list-style: none; counter-reset: question-counter; padding-left: 0; margin-bottom: 30px; }}
        .unified-question-item {{ counter-increment: question-counter; padding: 12px 15px; margin-bottom: 10px; background: white; border-radius: 8px; border-left: 4px solid #667eea; font-size: 1em; line-height: 1.6; color: #212529; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .unified-answer-header {{ font-size: 1.3em; font-weight: 700; color: #28a745; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; border-top: 2px solid #dee2e6; padding-top: 20px; }}
        .unified-answer-content {{ background: white; padding: 20px; border-radius: 10px; line-height: 1.8; color: #212529; font-size: 1.05em; border-left: 4px solid #28a745; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="header">
                <h1>❓ Q&A Training Data</h1>
                <p id="topic-count">0 topics</p>
            </div>
            <div class="search-box">
                <input type="text" id="searchInput" class="search-input" placeholder="🔍 Search topics...">
            </div>
            <div class="search-box" style="border-bottom: none; padding-top: 10px;">
                <input type="text" id="modifyInput" class="search-input" placeholder="✨ Modify request (e.g., 'Simplify all Q&A')">
                <button onclick="sendModifyRequest()" style="width: 100%; margin-top: 10px; padding: 12px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;">Send Request</button>
                <div id="modifyStatus" style="margin-top: 10px; padding: 10px; border-radius: 8px; display: none; font-size: 0.9em;"></div>
            </div>
            <div class="topic-list" id="topicList"></div>
        </div>
        <div class="content-area">
            <div id="contentDisplay">
                <div class="empty-state">
                    <div class="empty-state-icon">❓</div>
                    <h2>Select a Topic</h2>
                    <p>Choose a topic from the left sidebar to view Q&A pairs</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        const DATA = {json_str};

        let currentTopicIndex = null;
        let selectedQAIndices = new Set();

        function renderTopics(topics) {{
            document.getElementById('topicList').innerHTML = topics.map((item, index) => `
                <div class="topic-item" id="topic-${{index}}">
                    <div class="topic-name" onclick="selectTopic(${{index}})">${{escapeHtml(item.topic)}}</div>
                    <div class="topic-status">Status: ${{item.status || 'unknown'}}</div>
                    <button class="topic-simplify-btn" onclick="event.stopPropagation(); simplifyTopic(${{index}})">
                        ✨ Simplify All Q&A
                    </button>
                </div>
            `).join('');
        }}

        function selectTopic(index) {{
            currentTopicIndex = index;
            selectedQAIndices.clear();
            const item = DATA[index];
            document.querySelectorAll('.topic-item').forEach((el, i) => el.classList.toggle('active', i === index));

            const hasQA = item.qa_pairs && item.qa_pairs.length > 0;

            // Check for unified Q&A pairs
            const unifiedSection = hasQA ? detectUnifiedQA(item.qa_pairs) : null;
            // Filter out unified pairs from regular display
            let originalQA = item.qa_pairs || [];
            if (unifiedSection) {{
                // If unified section detected by flag, filter out flagged pairs
                const hasFlaggedPairs = originalQA.some(qa => qa.is_unified === true);
                if (hasFlaggedPairs) {{
                    originalQA = originalQA.filter(qa => qa.is_unified !== true);
                }} else {{
                    // Old data: remove last 10 pairs
                    originalQA = originalQA.slice(0, -10);
                }}
            }}

            document.getElementById('contentDisplay').innerHTML = `
                <div class="content-header">
                    <div class="content-title">${{escapeHtml(item.topic)}}</div>
                    <div class="content-meta">
                        <span class="meta-badge">📊 ${{item.qa_generation_status || 'unknown'}}</span>
                        <span class="meta-badge">🤖 ${{item.qa_model || 'N/A'}}</span>
                        <span class="meta-badge">⏱️ ${{item.qa_generation_time || 0}}s</span>
                        <span class="meta-badge">❓ ${{item.qa_count || 0}} Q&A</span>
                    </div>
                </div>
                <div class="content-body">
                    <div id="modifyPanelContainer"></div>
                    ${{hasQA && originalQA.length > 0 ? `<div class="content-section"><div class="section-label">❓ Questions & Answers</div>${{renderQA(originalQA, index)}}</div>` : ''}}
                    ${{unifiedSection ? renderUnifiedQA(unifiedSection) : ''}}
                    <div class="content-section">
                        <div class="section-label">🌐 Original Content</div>
                        <div class="section-content">${{escapeHtml(item.browser_content || 'No content')}}</div>
                    </div>
                    <div class="content-section">
                        <div class="section-label">🔗 Source</div>
                        <div class="section-content"><a href="${{item.original_url}}" target="_blank">${{escapeHtml(item.semantic_path)}}</a></div>
                    </div>
                </div>
            `;
        }}

        function detectUnifiedQA(qaPairs) {{
            // Method 1: Check for is_unified flag (new SimplifyAgent data)
            const flaggedPairs = qaPairs.filter(qa => qa.is_unified === true);
            if (flaggedPairs.length >= 2) {{
                return {{
                    questions: flaggedPairs.map(qa => qa.question),
                    answer: flaggedPairs[0].answer
                }};
            }}

            // Method 2: Fallback - check if last 10 have same answer (old data)
            if (qaPairs.length >= 10) {{
                const last10 = qaPairs.slice(-10);
                const firstAnswer = last10[0].answer;
                const allSameAnswer = last10.every(qa => qa.answer === firstAnswer);

                if (allSameAnswer) {{
                    return {{
                        questions: last10.map(qa => qa.question),
                        answer: firstAnswer
                    }};
                }}
            }}

            return null;
        }}

        function renderUnifiedQA(unifiedData) {{
            const questionsList = unifiedData.questions.map((q, i) =>
                `<li class="unified-question-item">${{i + 1}}. ${{escapeHtml(q)}}</li>`
            ).join('');

            return `
                <div class="content-section">
                    <div class="section-label">✨ Comprehensive Q&A (Generated)</div>
                    <div class="unified-qa-box">
                        <div class="unified-questions-header">📋 Questions:</div>
                        <ol class="unified-questions-list">
                            ${{questionsList}}
                        </ol>
                        <div class="unified-answer-header">💡 Comprehensive Answer:</div>
                        <div class="unified-answer-content">${{escapeHtml(unifiedData.answer)}}</div>
                    </div>
                </div>
            `;
        }}

        function renderQA(qaPairs, topicIndex) {{
            return `<div class="qa-container">${{qaPairs.map((qa, i) => `
                <div class="qa-item">
                    <div class="qa-question">
                        <input type="checkbox" class="qa-checkbox" id="qa-check-${{i}}" onchange="toggleQASelection(${{i}})" />
                        <div class="qa-number">${{i + 1}}</div>
                        <div class="qa-question-text" onclick="toggleQA(${{i}})" style="cursor: pointer; flex: 1;">${{escapeHtml(qa.question)}}</div>
                        <div class="qa-toggle" onclick="toggleQA(${{i}})" style="cursor: pointer;">▼</div>
                    </div>
                    <div class="qa-answer" id="qa-${{i}}">
                        <div class="qa-answer-content">${{escapeHtml(qa.answer)}}</div>
                    </div>
                </div>
            `).join('')}}</div>`;
        }}

        function toggleQA(index) {{
            const answer = document.getElementById(`qa-${{index}}`);
            const question = answer.previousElementSibling;
            document.querySelectorAll('.qa-answer').forEach((el, i) => {{
                if (i !== index) {{ el.classList.remove('show'); el.previousElementSibling.classList.remove('active'); }}
            }});
            answer.classList.toggle('show');
            question.classList.toggle('active');
        }}

        function escapeHtml(text) {{ const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }}

        function filterTopics(searchTerm) {{
            const filtered = DATA.filter(item => {{
                const search = searchTerm.toLowerCase();
                return !searchTerm || (item.topic && item.topic.toLowerCase().includes(search)) ||
                    (item.browser_content && item.browser_content.toLowerCase().includes(search)) ||
                    (item.qa_pairs && item.qa_pairs.some(qa => qa.question.toLowerCase().includes(search) || qa.answer.toLowerCase().includes(search)));
            }});
            renderTopics(filtered);
            document.getElementById('topic-count').textContent = `${{filtered.length}} topic${{filtered.length !== 1 ? 's' : ''}}`;
        }}

        renderTopics(DATA);
        document.getElementById('topic-count').textContent = `${{DATA.length}} topic${{DATA.length !== 1 ? 's' : ''}}`;
        document.getElementById('searchInput').addEventListener('input', (e) => filterTopics(e.target.value));

        // Restore last modified topic after page reload
        const lastModifiedTopic = localStorage.getItem('lastModifiedTopic');
        if (lastModifiedTopic !== null) {{
            const topicIndex = parseInt(lastModifiedTopic, 10);
            if (topicIndex >= 0 && topicIndex < DATA.length) {{
                selectTopic(topicIndex);
                // Scroll the topic into view
                setTimeout(() => {{
                    const topicElement = document.getElementById(`topic-${{topicIndex}}`);
                    if (topicElement) {{
                        topicElement.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}
                }}, 100);
            }}
            // Clear the saved topic
            localStorage.removeItem('lastModifiedTopic');
        }}

        // Modify request functionality
        async function sendModifyRequest() {{
            const modifyInput = document.getElementById('modifyInput');
            const statusDiv = document.getElementById('modifyStatus');
            const userRequest = modifyInput.value.trim();

            if (!userRequest) {{
                showStatus('Please enter a modification request', 'error');
                return;
            }}

            showStatus('Processing request with LangGraph agents...', 'loading');

            try {{
                const response = await fetch('/api/modify', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ user_request: userRequest }})
                }});

                const result = await response.json();

                if (result.error) {{
                    showStatus(`Error: ${{result.error}}`, 'error');
                }} else {{
                    showStatus(`✅ ${{result.agent_response}}. Reloading page...`, 'success');
                    // Save current topic index before reload (if one is selected)
                    if (currentTopicIndex !== null) {{
                        localStorage.setItem('lastModifiedTopic', currentTopicIndex);
                    }}
                    // Reload page after 2 seconds to show updated data
                    setTimeout(() => location.reload(), 2000);
                }}
            }} catch (error) {{
                showStatus(`Failed: ${{error.message}}`, 'error');
            }}
        }}

        function showStatus(message, type) {{
            const statusDiv = document.getElementById('modifyStatus');
            statusDiv.style.display = 'block';
            statusDiv.textContent = message;
            statusDiv.style.background = type === 'success' ? '#d4edda' :
                                        type === 'error' ? '#f8d7da' :
                                        '#d1ecf1';
            statusDiv.style.color = type === 'success' ? '#155724' :
                                   type === 'error' ? '#721c24' :
                                   '#0c5460';
        }}

        // Allow Enter key to send modify request
        document.getElementById('modifyInput').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') sendModifyRequest();
        }});

        // Simplify topic functionality
        async function simplifyTopic(topicIndex) {{
            if (!confirm(`Simplify all Q&A pairs in this topic to make them more clear and understandable?`)) {{
                return;
            }}

            const statusDiv = document.getElementById('modifyStatus');
            showStatus('Simplifying Q&A pairs with LLM...', 'loading');

            try {{
                const response = await fetch('/api/simplify-topic', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ topic_index: topicIndex }})
                }});

                const result = await response.json();

                if (result.error) {{
                    showStatus(`Error: ${{result.error}}`, 'error');
                }} else {{
                    showStatus(`✅ ${{result.message}}. Reloading page...`, 'success');
                    // Save current topic index before reload
                    localStorage.setItem('lastModifiedTopic', topicIndex);
                    // Reload page after 2 seconds to show updated data
                    setTimeout(() => location.reload(), 2000);
                }}
            }} catch (error) {{
                showStatus(`Failed: ${{error.message}}`, 'error');
            }}
        }}

        // Selective modification functions
        function showModifyPanel(topicIndex) {{
            selectTopic(topicIndex);

            const modifyPanel = `
                <div class="modify-panel active" id="selectiveModifyPanel">
                    <h3>✏️ Modify Selected Q&A Pairs</h3>
                    <div class="selection-info" id="selectionInfo">
                        No questions selected. Select checkboxes below to modify specific Q&A pairs.
                    </div>
                    <input type="text" id="selectiveModifyInput" placeholder="Enter modification request (e.g., 'Simplify to 5th grade level')" />
                    <div class="modify-panel-buttons">
                        <button class="modify-cancel-btn" onclick="hideModifyPanel()">Cancel</button>
                        <button class="merge-btn" id="mergeBtn" onclick="sendMergeRequest()" disabled title="Select 2 or more Q&A pairs to merge (originals kept)">
                            🔗 Merge Q&A Pairs
                        </button>
                    </div>
                </div>
            `;

            document.getElementById('modifyPanelContainer').innerHTML = modifyPanel;
        }}

        function hideModifyPanel() {{
            document.getElementById('modifyPanelContainer').innerHTML = '';
            selectedQAIndices.clear();
            // Uncheck all checkboxes
            document.querySelectorAll('.qa-checkbox').forEach(cb => cb.checked = false);
        }}

        function toggleQASelection(index) {{
            const checkbox = document.getElementById(`qa-check-${{index}}`);
            if (checkbox.checked) {{
                selectedQAIndices.add(index);
            }} else {{
                selectedQAIndices.delete(index);
            }}
            updateSelectionInfo();
        }}

        function updateSelectionInfo() {{
            const infoDiv = document.getElementById('selectionInfo');
            const submitBtn = document.getElementById('selectiveSubmitBtn');
            const mergeBtn = document.getElementById('mergeBtn');

            if (!infoDiv || !submitBtn) return;

            const count = selectedQAIndices.size;
            if (count === 0) {{
                infoDiv.textContent = 'No questions selected. Select checkboxes below to modify specific Q&A pairs.';
                infoDiv.style.background = '#d1ecf1';
                infoDiv.style.color = '#0c5460';
                submitBtn.disabled = true;
                if (mergeBtn) mergeBtn.disabled = true;
            }} else if (count === 1) {{
                infoDiv.textContent = '1 question selected for modification. (Select 2+ to enable merge)';
                infoDiv.style.background = '#d4edda';
                infoDiv.style.color = '#155724';
                submitBtn.disabled = false;
                if (mergeBtn) mergeBtn.disabled = true;
            }} else {{
                infoDiv.textContent = `${{count}} questions selected. You can modify them or merge them into 1 Q&A pair (originals kept).`;
                infoDiv.style.background = '#d1ecf1';
                infoDiv.style.color = '#0c5460';
                submitBtn.disabled = false;
                if (mergeBtn) mergeBtn.disabled = false;
            }}
        }}

        async function sendSelectiveModifyRequest() {{
            const userRequest = document.getElementById('selectiveModifyInput').value.trim();


            // Validate at least 2 indices
            if (selectedQAIndices.size < 2) {{
                alert('Please select at least 2 Q&A pairs to merge');
                return;
            }}

            if (currentTopicIndex === null) {{
                alert('No topic selected');
                return;
            }}

            const count = selectedQAIndices.size;
            // Confirm merge action
            if (!confirm(`Merge these ${{count}} Q&A pairs into 1? Original questions will be kept, and the merged Q&A will be added at the end.`)) {{
                return;
            }}

            // Get selected indices
            const selectedIndices = Array.from(selectedQAIndices);

            // Show loading state
            const mergeBtn = document.getElementById('mergeBtn');
            const submitBtn = document.getElementById('selectiveSubmitBtn');
            mergeBtn.disabled = true;
            submitBtn.disabled = true;
            mergeBtn.textContent = 'Merging...';

            try {{
                const response = await fetch('/api/merge-qa', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        user_request: userRequest,
                        topic_index: currentTopicIndex,
                        selected_qa_indices: selectedIndices
                    }})
                }});

                const result = await response.json();

                if (result.error) {{
                    alert(`Merge Error: ${{result.error}}`);
                    mergeBtn.disabled = false;
                    submitBtn.disabled = false;
                    mergeBtn.textContent = '🔗 Merge Q&A Pairs';
                }} else {{
                    alert(`✅ ${{result.agent_response}}`);
                    // Save current topic index before reload
                    localStorage.setItem('lastModifiedTopic', currentTopicIndex);
                    // Reload page to show updated data
                    location.reload();
                }}
            }} catch (error) {{
                alert(`Merge Failed: ${{error.message}}`);
                mergeBtn.disabled = false;
                submitBtn.disabled = false;
                mergeBtn.textContent = '🔗 Merge Q&A Pairs';
            }}
        }}
    </script>
</body>
</html>"""

    return html


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Root endpoint - Serves HTML viewer with Q&A data

    Uses QADataset class for structured data access
    """
    try:
        # Get JSON file path from app state
        JSON_FILE_PATH = request.app.state.json_file_path

        if not JSON_FILE_PATH.exists():
            logger.error("JSON file not found")
            return HTMLResponse(content="<h1>Error: JSON file not found</h1>", status_code=404)

        logger.info("🔍 HTML Viewer: Loading data with QADataset class...")
        dataset = QADataset.from_json(str(JSON_FILE_PATH))  # ✅ Using class!
        logger.info(f"✅ Loaded {dataset.total_topics()} topics, {dataset.total_qa_pairs()} Q&A pairs")

        # Convert to dict for HTML viewer
        data = dataset.to_dict()

        # Generate HTML viewer
        html_content = generate_html_viewer(data)
        logger.info("✅ Generated HTML viewer successfully")

        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error generating HTML viewer: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)


@app.post("/api/merge-qa", response_model=ModifyResponse)
async def merge_qa_endpoint(merge_request: MergeQARequest, request: Request) -> ModifyResponse:
    """
    Merge 2+ Q&A pairs into 1 using 3-agent workflow (Agent1 → Merge Agent → Agent3)

    Originals are kept intact, merged Q&A is appended at the end.

    Args:
        merge_request: MergeQARequest with 2 or more selected Q&A indices
        request: FastAPI Request object

    Returns:
        ModifyResponse with merged data
    """
    try:
        logger.info("="*60)
        logger.info("🟣 MERGE ENDPOINT: Request received")
        logger.info(f"   Topic Index: {merge_request.topic_index}")
        logger.info(f"   User Request: {merge_request.user_request[:100]}...")
        logger.info(f"   Selected Q&A indices: {merge_request.selected_qa_indices}")
        logger.info(f"   Number of Q&A to merge: {len(merge_request.selected_qa_indices)}")
        logger.info("="*60)

        # Validate input
        if not merge_request.user_request or not merge_request.user_request.strip():
            raise HTTPException(status_code=400, detail="Request cannot be empty")

        # Validate at least 2 indices
        if len(merge_request.selected_qa_indices) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Merge requires at least 2 Q&A pairs, got {len(merge_request.selected_qa_indices)}"
            )

        # Get QA modifier instance from app state
        modifier = request.app.state.qa_modifier

        # Get JSON file path from app state and load using QADataset class
        JSON_FILE_PATH = request.app.state.json_file_path
        logger.info(f"📂 STEP 1: Loading data from JSON file")
        logger.info(f"   File path: {JSON_FILE_PATH}")

        dataset = QADataset.from_json(str(JSON_FILE_PATH))
        logger.info(f"✅ QADataset.from_json() SUCCESS")
        logger.info(f"   Dataset type: {type(dataset)}")
        logger.info(f"   Total topics loaded: {dataset.total_topics()}")
        logger.info(f"   Total Q&A pairs: {dataset.total_qa_pairs()}")

        # Validate topic index
        if merge_request.topic_index < 0 or merge_request.topic_index >= dataset.total_topics():
            raise HTTPException(status_code=400, detail=f"Invalid topic index: {merge_request.topic_index}")

        # Convert to dict for agents (agents still use dict format)
        logger.info(f"📦 STEP 2: Converting QADataset to dict for processing")
        all_data = dataset.to_dict()
        logger.info(f"✅ dataset.to_dict() SUCCESS")
        logger.info(f"   Data type: {type(all_data)}")
        logger.info(f"   Topics in dict: {len(all_data)}")

        # Process merge through LangGraph agents
        logger.info(f"🤖 STEP 3: Calling modifier.merge_qa_pairs()")
        logger.info(f"   Merging indices: {merge_request.selected_qa_indices}")
        result = await modifier.merge_qa_pairs(
            topic_index=merge_request.topic_index,
            selected_qa_indices=merge_request.selected_qa_indices,
            all_data=all_data,
            user_request=merge_request.user_request
        )
        logger.info(f"✅ modifier.merge_qa_pairs() SUCCESS")
        logger.info(f"   Agent response: {result['agent_response']}")

        # SAVE TO JSON FILE using QADataset class
        logger.info(f"💾 STEP 4: Converting dict back to QADataset and saving")
        modified_data = result["modified_data"]
        logger.info(f"   Creating QADataset from modified dict...")
        modified_dataset = QADataset.from_dict(modified_data)
        logger.info(f"✅ QADataset.from_dict() SUCCESS")
        logger.info(f"   Dataset type: {type(modified_dataset)}")
        logger.info(f"   Saving to JSON file...")
        modified_dataset.save_to_json(str(JSON_FILE_PATH))
        logger.info(f"✅ QADataset.save_to_json() SUCCESS")
        logger.info(f"   File saved: {JSON_FILE_PATH}")
        logger.info("="*60)
        logger.info("✅ MERGE ENDPOINT: Complete data flow through QADataset")
        logger.info("="*60)

        return ModifyResponse(
            modified_data=modified_data,
            agent_response=result["agent_response"],
            error=result.get("error"),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Merge Q&A endpoint error: {e}")
        return ModifyResponse(
            modified_data=[],
            agent_response=f"Merge failed: {str(e)}",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )


@app.post("/api/simplify-topic", response_model=SimplifyTopicResponse)
async def simplify_topic_endpoint(simplify_request: SimplifyTopicRequest, request: Request) -> SimplifyTopicResponse:
    """
    Simplify all Q&A pairs in a topic to make them clearer and more understandable
    """
    try:
        logger.info("="*60)
        logger.info(f"🔵 SIMPLIFY ENDPOINT: Request received")
        logger.info(f"   Topic Index: {simplify_request.topic_index}")
        logger.info("="*60)

        # Step 1: Load current data using QADataset class
        JSON_FILE_PATH = request.app.state.json_file_path
        logger.info(f"📂 STEP 1: Loading data from JSON file")
        logger.info(f"   File path: {JSON_FILE_PATH}")

        dataset = QADataset.from_json(str(JSON_FILE_PATH))
        logger.info(f"✅ QADataset.from_json() SUCCESS")
        logger.info(f"   Dataset type: {type(dataset)}")
        logger.info(f"   Total topics loaded: {dataset.total_topics()}")
        logger.info(f"   Total Q&A pairs: {dataset.total_qa_pairs()}")

        # Step 2: Validate topic index
        if simplify_request.topic_index < 0 or simplify_request.topic_index >= dataset.total_topics():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid topic index: {simplify_request.topic_index}"
            )

        # Step 3: Get all data as dict
        logger.info(f"📦 STEP 2: Converting QADataset to dict for processing")
        all_data = dataset.to_dict()
        logger.info(f"✅ dataset.to_dict() SUCCESS")
        logger.info(f"   Data type: {type(all_data)}")
        logger.info(f"   Topics in dict: {len(all_data)}")

        topic = all_data[simplify_request.topic_index]
        qa_pairs = topic.get('qa_pairs', [])
        logger.info(f"✓ Topic '{topic['topic']}' extracted")
        logger.info(f"   Q&A pairs before simplify: {len(qa_pairs)}")

        if len(qa_pairs) == 0:
            raise HTTPException(status_code=400, detail="Topic has no Q&A pairs")

        # Step 4: Call SimplifyAgent to simplify Q&A pairs
        logger.info(f"🤖 STEP 3: Calling SimplifyAgent.simplify()")
        logger.info(f"   Input: {len(qa_pairs)} Q&A pairs")
        simplified_qa_pairs = await SimplifyAgent.simplify(qa_pairs)
        logger.info(f"✅ SimplifyAgent.simplify() SUCCESS")
        logger.info(f"   Output: {len(simplified_qa_pairs)} Q&A pairs")
        logger.info(f"   Added: {len(simplified_qa_pairs) - len(qa_pairs)} new Q&A pairs")

        # Step 5: Update the topic with simplified Q&A pairs
        logger.info(f"📝 STEP 4: Updating topic in dict")
        all_data[simplify_request.topic_index]['qa_pairs'] = simplified_qa_pairs
        all_data[simplify_request.topic_index]['qa_count'] = len(simplified_qa_pairs)
        logger.info(f"✅ Topic updated in dict")

        # Step 6: Save back to JSON file using QADataset class
        logger.info(f"💾 STEP 5: Converting dict back to QADataset and saving")
        logger.info(f"   Creating QADataset from modified dict...")
        modified_dataset = QADataset.from_dict(all_data)
        logger.info(f"✅ QADataset.from_dict() SUCCESS")
        logger.info(f"   Dataset type: {type(modified_dataset)}")
        logger.info(f"   Saving to JSON file...")
        modified_dataset.save_to_json(str(JSON_FILE_PATH))
        logger.info(f"✅ QADataset.save_to_json() SUCCESS")
        logger.info(f"   File saved: {JSON_FILE_PATH}")
        logger.info("="*60)
        logger.info("✅ SIMPLIFY ENDPOINT: Complete data flow through QADataset")
        logger.info("="*60)

        return SimplifyTopicResponse(
            message=f"Successfully simplified {len(simplified_qa_pairs)} Q&A pairs in topic '{topic['topic']}'",
            simplified_count=len(simplified_qa_pairs),
            error=None,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simplify topic endpoint error: {e}")
        return SimplifyTopicResponse(
            message=f"Failed: {str(e)}",
            simplified_count=0,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )


@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint"""
    try:
        # Check if QA modifier is initialized
        modifier = request.app.state.qa_modifier

        return {
            "status": "healthy",
            "service": "LangGraph MergeQA & SimplifyAgent Server",
            "qa_modifier_initialized": modifier is not None,
            "active_features": ["SimplifyAgent", "MergeQA"],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# ============================================================================
# CHECKPOINT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/checkpoints")
async def list_checkpoints(request: Request):
    """Get list of all Q&A data checkpoints from LangGraph"""
    try:
        modifier = request.app.state.qa_modifier
        checkpoints = await modifier.get_history()

        return {
            "checkpoints": checkpoints,
            "total": len(checkpoints),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"List checkpoints error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/checkpoint/{version}/activate")
async def activate_checkpoint_endpoint(version: int, request: Request):
    """Activate specific checkpoint version (rollback) in LangGraph"""
    try:
        modifier = request.app.state.qa_modifier
        data = await modifier.rollback_to_version(version)

        if data is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint version {version} not found")

        return {
            "message": f"Successfully rolled back to checkpoint version {version}",
            "version": version,
            "topics_count": len(data),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activate checkpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status(request: Request):
    """Get system status including LangGraph checkpoint info"""
    try:
        modifier = request.app.state.qa_modifier
        current_state = await modifier.get_current_state()
        history = await modifier.get_history()

        return {
            "service": "LangGraph Chatbot Server",
            "checkpoint_system": "LangGraph AsyncSqliteSaver",
            "database_file": "langgraph_checkpoints.db",
            "current_version": current_state.get("version", 0) if current_state else 0,
            "total_checkpoints": len(history),
            "current_topics": len(current_state.get("modified_data", [])) if current_state else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DATA BACKUP/DOWNLOAD ENDPOINTS
# ============================================================================

@app.get("/api/download-data")
async def download_data(request: Request):
    """
    Download current Q&A data as JSON file

    Useful for:
    - Backing up data from Railway
    - Viewing data locally
    - Migrating data between environments
    """
    try:
        JSON_FILE_PATH = request.app.state.json_file_path

        if not JSON_FILE_PATH.exists():
            raise HTTPException(status_code=404, detail="Data file not found")

        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"📥 Data download requested: {len(data)} topics")

        return JSONResponse(
            content=data,
            headers={
                "Content-Disposition": f"attachment; filename=browser_agent_test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-data")
async def upload_data(request: Request):
    """
    Upload Q&A data to replace current dataset

    IMPORTANT: This will overwrite the existing data!
    Use for restoring backups or migrating data
    """
    try:
        JSON_FILE_PATH = request.app.state.json_file_path

        # Parse request body as JSON
        data = await request.json()

        # Validate data structure
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Data must be a JSON array")

        # Backup current data before overwriting
        if JSON_FILE_PATH.exists():
            backup_path = JSON_FILE_PATH.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Created backup: {backup_path}")

        # Write new data
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"📤 Data uploaded: {len(data)} topics")

        return {
            "message": f"Successfully uploaded {len(data)} topics",
            "topics_count": len(data),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"Upload data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RUN SERVER
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "langgraph_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
