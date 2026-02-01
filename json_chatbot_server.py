"""
JSON Chatbot FastAPI Server
============================

FastAPI server for JSON Q&A chatbot with web interface.

Endpoints:
- GET /                    - Web UI for chatting
- POST /api/chat          - Process user question
- GET /api/topics         - Get all topics
- GET /api/topics/{name}  - Get Q&A pairs for specific topic
- GET /api/health         - Health check
- GET /api/stats          - Get statistics

Usage:
    python json_chatbot_server.py

    Server runs at: http://localhost:8000
"""

import logging
import os
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from json_chatbot_engine import JSONChatbotEngine

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot engine instance
chatbot_engine: Optional[JSONChatbotEngine] = None


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    Initializes chatbot engine on startup
    """
    global chatbot_engine

    logger.info("="*60)
    logger.info("JSON CHATBOT SERVER - STARTUP")
    logger.info("="*60)

    # Get JSON file path from environment or use default
    json_path = os.getenv('JSON_DATA_PATH', 'browser_agent_test_output.json')

    # Check if rephrasing should be enabled
    enable_rephrasing = os.getenv('GROQ_API_KEY') is not None

    logger.info(f"JSON Data Path: {json_path}")
    logger.info(f"Rephrasing: {'Enabled' if enable_rephrasing else 'Disabled'}")

    # Initialize chatbot engine
    try:
        chatbot_engine = JSONChatbotEngine(
            json_path=json_path,
            enable_rephrasing=enable_rephrasing
        )
        logger.info("="*60)
        logger.info("✅ SERVER READY")
        logger.info(f"   Topics: {len(chatbot_engine.dataset.topics)}")
        logger.info(f"   Q&A Pairs: {len(chatbot_engine.dataset.all_qa_pairs)}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot engine: {e}")
        raise

    logger.info("🚀 Server running at: http://localhost:8000")
    logger.info("📚 API docs at: http://localhost:8000/docs")
    logger.info("="*60)

    yield  # Server runs here

    # Cleanup on shutdown
    logger.info("="*60)
    logger.info("JSON CHATBOT SERVER - SHUTDOWN")
    logger.info("="*60)


# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="JSON Q&A Chatbot",
    description="Chatbot powered by CSU_Progress.json with multi-stage similarity search",
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

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str
    matched_by: str
    confidence: float
    source_question: Optional[str] = None
    source_topic: Optional[str] = None
    suggested_questions: Optional[List[str]] = None
    suggested_topics: Optional[List[str]] = None
    pipeline_info: Dict


class TopicInfo(BaseModel):
    """Topic information model"""
    name: str
    url: str
    semantic_path: str
    qa_count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    chatbot_ready: bool
    total_topics: int
    total_qa_pairs: int
    rephrasing_enabled: bool
    timestamp: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve web UI for chatbot
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Q&A Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            max-width: 900px;
            width: 100%;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { font-size: 1em; opacity: 0.9; }
        .chat-area {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 20px;
            padding: 15px 20px;
            border-radius: 12px;
            max-width: 80%;
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
            color: #333;
            border: 1px solid #e0e0e0;
        }
        .message.bot .confidence {
            font-size: 0.85em;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 2px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
        }
        .input-area input:focus {
            outline: none;
            border-color: #667eea;
        }
        .input-area button {
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.3s ease;
        }
        .input-area button:hover {
            background: #5568d3;
        }
        .input-area button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-style: italic;
            padding: 10px;
        }
        .suggested-questions {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }
        .suggested-questions p {
            font-weight: 600;
            margin-bottom: 10px;
            color: #667eea;
        }
        .suggested-questions button {
            display: block;
            width: 100%;
            text-align: left;
            padding: 10px;
            margin-bottom: 8px;
            background: #f0f1ff;
            border: 1px solid #667eea;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        .suggested-questions button:hover {
            background: #e0e1ff;
        }
        .initial-message {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .initial-message h2 {
            color: #667eea;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 JSON Q&A Chatbot</h1>
            <p>Ask me anything about health insurance plans!</p>
        </div>
        <div class="chat-area" id="chatArea">
            <div class="initial-message">
                <h2>👋 Welcome!</h2>
                <p>I'm here to help you with questions about health insurance plans, providers, and more.</p>
                <p style="margin-top: 20px; font-weight: 600;">Try asking:</p>
                <p style="margin-top: 10px;">"What is the Healthy Indiana Plan?"</p>
                <p>"How do I find a doctor?"</p>
                <p>"What is Community Connect?"</p>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Type your question here..." />
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatArea = document.getElementById('chatArea');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        // Allow Enter key to send
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;

            // Clear initial message if present
            const initialMsg = chatArea.querySelector('.initial-message');
            if (initialMsg) initialMsg.remove();

            // Add user message
            addMessage(question, 'user');
            userInput.value = '';
            sendBtn.disabled = true;

            // Show loading
            const loadingId = addMessage('Thinking...', 'loading');

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const result = await response.json();

                // Remove loading
                document.getElementById(loadingId).remove();

                // Add bot response
                addBotMessage(result);

            } catch (error) {
                document.getElementById(loadingId).remove();
                addMessage('Sorry, an error occurred. Please try again.', 'bot');
            }

            sendBtn.disabled = false;
        }

        function addMessage(text, type) {
            const msgId = 'msg-' + Date.now();
            const msgDiv = document.createElement('div');
            msgDiv.id = msgId;
            msgDiv.className = 'message ' + type;
            msgDiv.textContent = text;
            chatArea.appendChild(msgDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
            return msgId;
        }

        function addBotMessage(result) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message bot';

            // Main answer
            const answerP = document.createElement('p');
            answerP.textContent = result.answer;
            msgDiv.appendChild(answerP);

            // Confidence
            const confidenceP = document.createElement('div');
            confidenceP.className = 'confidence';
            confidenceP.textContent = `Matched by: ${result.matched_by} | Confidence: ${(result.confidence * 100).toFixed(1)}%`;
            msgDiv.appendChild(confidenceP);

            // Suggested questions
            if (result.suggested_questions && result.suggested_questions.length > 0) {
                const suggestedDiv = document.createElement('div');
                suggestedDiv.className = 'suggested-questions';

                const suggestedP = document.createElement('p');
                suggestedP.textContent = '💡 Suggested questions:';
                suggestedDiv.appendChild(suggestedP);

                result.suggested_questions.forEach(q => {
                    const btn = document.createElement('button');
                    btn.textContent = q;
                    btn.onclick = () => {
                        userInput.value = q;
                        sendMessage();
                    };
                    suggestedDiv.appendChild(btn);
                });

                msgDiv.appendChild(suggestedDiv);
            }

            chatArea.appendChild(msgDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process user question and return answer

    Args:
        request: ChatRequest with question and session_id

    Returns:
        ChatResponse with answer, confidence, and metadata
    """
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot engine not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Processing chat request: '{request.question}' (session: {request.session_id})")

    try:
        # Process question through engine
        result = chatbot_engine.process_question(
            user_question=request.question,
            session_id=request.session_id
        )

        # Build response
        response = ChatResponse(
            answer=result['answer'],
            matched_by=result['matched_by'],
            confidence=result['confidence'],
            source_question=result['source_qa'].question if result['source_qa'] else None,
            source_topic=result['source_qa'].topic if result['source_qa'] else None,
            suggested_questions=result.get('suggested_questions'),
            suggested_topics=result.get('suggested_topics'),
            pipeline_info=result['pipeline_info']
        )

        logger.info(f"Response: matched_by={response.matched_by}, confidence={response.confidence:.4f}")

        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/api/topics", response_model=List[TopicInfo])
async def get_topics():
    """
    Get all available topics

    Returns:
        List of TopicInfo objects
    """
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot engine not initialized")

    try:
        topics = chatbot_engine.get_all_topics()
        return [TopicInfo(**topic) for topic in topics]

    except Exception as e:
        logger.error(f"Error getting topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting topics: {str(e)}")


@app.get("/api/topics/{topic_name}")
async def get_topic_qa_pairs(topic_name: str):
    """
    Get all Q&A pairs for a specific topic

    Args:
        topic_name: Name of the topic

    Returns:
        Dict with topic info and Q&A pairs
    """
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot engine not initialized")

    try:
        qa_pairs = chatbot_engine.get_topic_qa_pairs(topic_name)

        if qa_pairs is None:
            raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")

        return {
            'topic': topic_name,
            'qa_count': len(qa_pairs),
            'qa_pairs': qa_pairs
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic Q&A pairs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting topic Q&A pairs: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint

    Returns:
        HealthResponse with system status
    """
    return HealthResponse(
        status="healthy" if chatbot_engine else "unhealthy",
        chatbot_ready=chatbot_engine is not None,
        total_topics=len(chatbot_engine.dataset.topics) if chatbot_engine else 0,
        total_qa_pairs=len(chatbot_engine.dataset.all_qa_pairs) if chatbot_engine else 0,
        rephrasing_enabled=chatbot_engine.rephraser is not None if chatbot_engine else False,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/stats")
async def stats():
    """
    Get chatbot statistics

    Returns:
        Dict with detailed statistics
    """
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot engine not initialized")

    try:
        topics = chatbot_engine.get_all_topics()

        return {
            'total_topics': len(topics),
            'total_qa_pairs': len(chatbot_engine.dataset.all_qa_pairs),
            'rephrasing_enabled': chatbot_engine.rephraser is not None,
            'similarity_thresholds': {
                'ideal': chatbot_engine.search_engine.SIMILARITY_THRESHOLD_IDEAL,
                'minimal': chatbot_engine.search_engine.SIMILARITY_THRESHOLD
            },
            'topics': [
                {
                    'name': topic['name'],
                    'qa_count': topic['qa_count']
                }
                for topic in topics
            ]
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv('PORT', 8001))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
