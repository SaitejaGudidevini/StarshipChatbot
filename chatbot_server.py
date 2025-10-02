"""
FastAPI Chatbot Server with Vector DB + Playwright Pipeline
Provides intelligent responses by querying vector database and extracting live content
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path

# Import our custom services
from vector_query_service import VectorQueryService
from content_extractor import ContentExtractor

app = FastAPI(title="Smart Chatbot with Live Content")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize services
vector_service = VectorQueryService()
content_extractor = ContentExtractor()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    timestamp: str

# Chat history storage (in production, use a proper database)
chat_history: List[Dict] = []

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage) -> ChatResponse:
    """
    Main chat endpoint that processes user questions
    """
    try:
        user_query = message.message.strip()
        timestamp = datetime.now().isoformat()
        
        if not user_query:
            return ChatResponse(
                response="Please ask me something!",
                timestamp=timestamp
            )
        
        # Step 1: Query vector database for relevant content
        print(f"🔍 Searching vector DB for: '{user_query}'")
        vector_results = await vector_service.query(user_query, top_k=3)
        
        if not vector_results:
            return ChatResponse(
                response="I couldn't find any relevant information about that topic.",
                timestamp=timestamp
            )
        
        # Step 2: Extract live content for the best match
        best_match = vector_results[0]
        print(f"📄 Best match: {best_match['semantic_path']}")
        
        # Extract live content using Playwright
        live_content = await content_extractor.extract_content(
            url=best_match['url'],
            element_type=best_match['type'].split('/')[0].strip(),
            target_text=best_match['semantic_path'].split('/')[-1]
        )
        
        # Step 3: Build response
        if live_content and live_content != "Content not found":
            response_text = f"Based on the latest information from {best_match['url']}:\n\n{live_content}"
        else:
            # Fallback to vector result if live extraction fails
            response_text = f"Here's what I found about '{user_query}':\n\n{best_match.get('content', 'No content available')}"
        
        # Prepare sources for transparency
        sources = []
        for result in vector_results:
            sources.append({
                "semantic_path": result['semantic_path'],
                "url": result['url'],
                "type": result['type'],
                "similarity": result['similarity']
            })
        
        # Store in chat history
        chat_entry = {
            "user_message": user_query,
            "bot_response": response_text,
            "sources": sources,
            "timestamp": timestamp
        }
        chat_history.append(chat_entry)
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            timestamp=timestamp
        )
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error while processing your question: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/history")
async def get_chat_history():
    """Get chat history"""
    return {"history": chat_history[-10:]}  # Return last 10 messages

@app.delete("/api/history")
async def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smart Chatbot",
        "vector_service": await vector_service.health_check(),
        "content_extractor": await content_extractor.health_check()
    }

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    Path('templates').mkdir(exist_ok=True)
    print("\n" + "="*60)
    print("🤖 SMART CHATBOT SERVER")
    print("="*60)
    print("\nServices starting...")
    print("🔍 Vector Query Service: Initializing...")
    print("🎭 Content Extractor: Initializing...")
    print("\nServer ready at: http://localhost:8002")
    print("="*60)

if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "chatbot_server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )