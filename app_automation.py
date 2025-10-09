"""
Starship Semantic Content Automation
=====================================

Automated content extraction, AI processing, and database storage for semantic paths.

Features:
- Batch processing of semantic paths
- AI-powered content extraction using browser-use
- Groq summarization
- SQLite database storage
- Progress tracking and error handling
- Resume functionality for interrupted runs
"""

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from browser_use import Agent, ChatGoogle
from groq import Groq
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SemanticContentProcessor:
    """Handles automated semantic content processing and storage"""
    
    def __init__(self, data_path: str, db_path: str = "semantic_content.db"):
        self.data_path = Path(data_path)
        self.db_path = db_path
        self.groq_client = None
        self.google_llm = None
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        
        # Initialize AI clients
        self._initialize_ai_clients()
        
        # Initialize database
        self._initialize_database()
        
        # Load semantic data
        self.semantic_data = self._load_semantic_data()
        
    def _initialize_ai_clients(self):
        """Initialize Groq and Google AI clients"""
        # Initialize Groq client for summarization
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info("✅ Groq client initialized for summarization")
            except Exception as e:
                logger.error(f"❌ Groq client initialization failed: {e}")
        else:
            logger.error("❌ GROQ_API_KEY not found in environment")
        
        # Initialize Google LLM for browser-use (copy exact pattern from app_m.py)
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            try:
                self.google_llm = ChatGoogle(
                    model="gemini-2.5-pro",
                    api_key=google_api_key,
                    temperature=0.1
                )
                logger.info("✅ ChatGoogle initialized for browser-use")
            except Exception as e:
                logger.warning(f"⚠️ ChatGoogle initialization failed: {e}")
        else:
            logger.warning("⚠️ GOOGLE_API_KEY not found, AI extraction will be limited")

    def _initialize_database(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semantic_path TEXT UNIQUE NOT NULL,
                original_url TEXT,
                source_type TEXT,
                element_type TEXT,
                topic TEXT,
                raw_content TEXT,
                ai_summary TEXT,
                groq_summary TEXT,
                extraction_method TEXT,
                processing_time_seconds REAL,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create processing stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                total_items INTEGER,
                processed_items INTEGER,
                failed_items INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_duration_seconds REAL,
                status TEXT
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_semantic_path ON semantic_content(semantic_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON semantic_content(status)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")

    def _load_semantic_data(self) -> Dict:
        """Load semantic data from JSON file"""
        try:
            with self.data_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract semantic elements
            if isinstance(data, dict) and 'semantic_elements' in data:
                semantic_elements = data['semantic_elements']
            elif isinstance(data, list):
                # Convert list to dict format
                semantic_elements = {item.get('semantic_path', str(i)): item for i, item in enumerate(data)}
            else:
                semantic_elements = data
            
            logger.info(f"✅ Loaded {len(semantic_elements)} semantic elements")
            return semantic_elements
            
        except Exception as e:
            logger.error(f"❌ Failed to load semantic data: {e}")
            return {}

    def _infer_topic(self, semantic_path: str) -> str:
        """Extract topic from semantic path"""
        # Remove URL prefix and clean up
        topic = semantic_path.split('/')[-1] if '/' in semantic_path else semantic_path
        topic = topic.replace('-', ' ').replace('_', ' ').strip()
        return topic or semantic_path

    async def _extract_content_with_ai(self, entry: Dict) -> Dict[str, str]:
        """Extract content using AI agent with Groq"""
        url = entry.get("original_url", entry.get("semantic_path", ""))
        topic = self._infer_topic(entry.get("semantic_path", ""))
        
        if not self.google_llm:
            return {
                "content": f"AI extraction not available for {topic}",
                "extraction_method": "ai_unavailable",
                "error": "Google LLM not initialized"
            }
        
        try:
            task = f"""
Navigate to {url} and extract comprehensive information about '{topic}'.
Focus on:
1. Main content related to '{topic}'
2. Key details and descriptions
3. Any relevant context or background information
4. Important facts or statistics if present

Provide a detailed but concise summary of the relevant content.
"""
            
            agent = Agent(
                task=task,
                llm=self.google_llm,
                use_cloud=False
            )
            
            start_time = time.time()
            result = await agent.run()
            processing_time = time.time() - start_time
            
            # Convert AgentHistoryList to string if needed
            content_text = str(result) if result else "No content extracted"
            
            return {
                "content": content_text,
                "extraction_method": "ai_agent_google",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"AI extraction failed for {topic}: {e}")
            return {
                "content": f"AI extraction failed for {topic}",
                "extraction_method": "ai_failed",
                "error": str(e)
            }

    async def _summarize_with_groq(self, text: str, topic: str, max_length: int = 150) -> Dict[str, str]:
        """Summarize content using Groq"""
        if not self.groq_client:
            return {
                "summary": str(text)[:max_length] + "...",
                "error": "Groq client not initialized"
            }
        
        try:
            prompt = f"""
Summarize the following content about '{topic}' in approximately {max_length} words.
Focus on key information and main points.

Content:
{text}

Summary:"""
            
            completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=200
            )
            
            summary = completion.choices[0].message.content.strip()
            return {
                "summary": summary,
                "model_used": "llama-3.1-8b-instant",
                "processing_time": "<1s"
            }
            
        except Exception as e:
            logger.error(f"Groq summarization failed for {topic}: {e}")
            fallback_text = str(text)[:max_length] if text else "No content available"
            return {
                "summary": fallback_text + "...",
                "error": f"Groq summarization failed: {str(e)}"
            }

    def _save_to_database(self, semantic_path: str, entry: Dict, content_data: Dict, summary_data: Dict, processing_time: float):
        """Save processed content to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO semantic_content 
                (semantic_path, original_url, source_type, element_type, topic,
                 raw_content, ai_summary, groq_summary, extraction_method,
                 processing_time_seconds, status, error_message, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                semantic_path,
                entry.get("original_url"),
                entry.get("source_type"),
                entry.get("element_type"),
                self._infer_topic(semantic_path),
                content_data.get("content"),
                str(content_data),  # Store full AI extraction info
                str(summary_data),   # Store full Groq summary info
                content_data.get("extraction_method"),
                processing_time,
                "completed" if not content_data.get("error") else "failed",
                content_data.get("error") or summary_data.get("error")
            ))
            
            conn.commit()
            logger.debug(f"✅ Saved to database: {semantic_path}")
            
        except Exception as e:
            logger.error(f"❌ Database save failed for {semantic_path}: {e}")
        finally:
            conn.close()

    def get_processing_status(self) -> Dict:
        """Get current processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM semantic_content WHERE status = "completed"')
        completed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM semantic_content WHERE status = "failed"')
        failed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM semantic_content WHERE status = "pending"')
        pending = cursor.fetchone()[0]
        
        total = len(self.semantic_data)
        
        conn.close()
        
        remaining = total - completed - failed
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "remaining": remaining,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }

    async def process_single_item(self, semantic_path: str, entry: Dict) -> bool:
        """Process a single semantic path item"""
        topic = self._infer_topic(semantic_path)
        logger.info(f"🔄 Processing: {topic}")
        
        start_time = time.time()
        
        try:
            # Extract content with AI
            content_data = await self._extract_content_with_ai(entry)
            
            # Summarize with Groq
            summary_data = await self._summarize_with_groq(
                content_data.get("content", ""), 
                topic
            )
            
            processing_time = time.time() - start_time
            
            # Save to database
            self._save_to_database(semantic_path, entry, content_data, summary_data, processing_time)
            
            if not content_data.get("error") and not summary_data.get("error"):
                logger.info(f"✅ Completed: {topic} ({processing_time:.1f}s)")
                self.processed_count += 1
                return True
            else:
                logger.warning(f"⚠️ Completed with errors: {topic}")
                self.failed_count += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed: {topic} - {e}")
            self.failed_count += 1
            return False

    async def process_all(self, max_concurrent: int = 3, resume: bool = True) -> Dict:
        """Process all semantic paths with concurrency control"""
        logger.info("🚀 Starting automated content processing")
        
        self.start_time = time.time()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get items to process
        items_to_process = []
        
        if resume:
            # Skip already completed items
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT semantic_path FROM semantic_content WHERE status = "completed"')
            completed_paths = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            items_to_process = [
                (path, entry) for path, entry in self.semantic_data.items()
                if path not in completed_paths
            ]
            logger.info(f"📋 Resuming: {len(items_to_process)} remaining items")
        else:
            items_to_process = list(self.semantic_data.items())
            logger.info(f"📋 Processing: {len(items_to_process)} total items")
        
        # Process items with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.process_single_item(item[0], item[1])
        
        # Process in batches to avoid overwhelming the system
        batch_size = max_concurrent * 2
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(items_to_process))
            batch = items_to_process[start_idx:end_idx]
            
            logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} items)")
            
            # Process batch
            tasks = [process_with_semaphore(item) for item in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Progress update
            status = self.get_processing_status()
            logger.info(f"📊 Progress: {status['completed']}/{status['total']} ({status['progress_percent']:.1f}%)")
            
            # Brief pause between batches
            await asyncio.sleep(1)
        
        # Final statistics
        total_time = time.time() - self.start_time
        final_status = self.get_processing_status()
        
        logger.info(f"🎉 Processing completed!")
        logger.info(f"📊 Final stats: {final_status['completed']} completed, {final_status['failed']} failed")
        logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
        
        return {
            "session_id": session_id,
            "total_time": total_time,
            "final_status": final_status
        }

    async def process_batches(self, max_concurrent: int = 3, batch_size: int = 50, max_batches: int = None, resume: bool = True) -> Dict:
        """Process items in batches with user confirmations"""
        logger.info("🚀 Starting batch processing automation")
        
        self.start_time = time.time()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get items to process
        items_to_process = []
        
        if resume:
            # Skip already completed items
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT semantic_path FROM semantic_content WHERE status = "completed"')
            completed_paths = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            items_to_process = [
                (path, entry) for path, entry in self.semantic_data.items()
                if path not in completed_paths
            ]
            logger.info(f"📋 Resuming: {len(items_to_process)} remaining items")
        else:
            items_to_process = list(self.semantic_data.items())
            logger.info(f"📋 Processing: {len(items_to_process)} total items")
        
        if not items_to_process:
            logger.info("✅ No items to process!")
            return {"session_id": session_id, "total_time": 0, "final_status": self.get_processing_status()}
        
        # Process in batches
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        if max_batches:
            total_batches = min(total_batches, max_batches)
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(items_to_process))
            batch_items = items_to_process[start_idx:end_idx]
            
            logger.info(f"\n📦 Batch {batch_num + 1}/{total_batches} ({len(batch_items)} items)")
            
            # Process batch
            batch_start = time.time()
            await self._process_batch_items(batch_items, max_concurrent)
            batch_time = time.time() - batch_start
            
            # Show progress
            status = self.get_processing_status()
            logger.info(f"📊 Batch completed in {batch_time:.1f}s")
            logger.info(f"📈 Progress: {status['completed']}/{status['total']} ({status['progress_percent']:.1f}%)")
            
            # Export after each batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"batch_results_{batch_num+1}_{timestamp}.json"
            self.export_results(export_file)
            logger.info(f"📁 Batch results exported to {export_file}")
            
            # Ask for continuation (except for last batch)
            if batch_num < total_batches - 1:
                response = input(f"\n❓ Continue to batch {batch_num + 2}? (y/n/auto): ").lower()
                if response == 'n':
                    logger.info("🛑 Stopping by user request")
                    break
                elif response == 'auto':
                    logger.info("🔄 Switching to auto mode - will process all remaining batches")
                    # Continue without asking
                    continue
            
            # Brief pause between batches
            await asyncio.sleep(2)
        
        # Final statistics
        total_time = time.time() - self.start_time
        final_status = self.get_processing_status()
        
        logger.info(f"\n🎉 Batch processing completed!")
        logger.info(f"📊 Final stats: {final_status['completed']} completed, {final_status['failed']} failed")
        logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
        
        return {
            "session_id": session_id,
            "total_time": total_time,
            "final_status": final_status
        }

    async def _process_batch_items(self, batch_items, max_concurrent):
        """Process a batch of items with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.process_single_item(item[0], item[1])
        
        # Process batch items
        tasks = [process_with_semaphore(item) for item in batch_items]
        await asyncio.gather(*tasks, return_exceptions=True)

    def export_results(self, output_file: str = "processed_content.json"):
        """Export processed results to JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT semantic_path, topic, raw_content, groq_summary, 
                   extraction_method, processing_time_seconds, status
            FROM semantic_content
            ORDER BY created_at
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "semantic_path": row[0],
                "topic": row[1],
                "summary": row[3],
                "extraction_method": row[4],
                "processing_time": row[5],
                "status": row[6]
            })
        
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Results exported to {output_file}")


def ask_batch_preferences():
    """Ask user for batch processing preferences"""
    print("\n🔧 Batch Processing Options:")
    print("   1. Process specific number of batches (e.g., 2-3 batches)")
    print("   2. Process all items with batch confirmations")
    print("   3. Process all items automatically (full automation)")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == "1":
        batch_size = int(input("Items per batch (recommended: 50-100): ") or "50")
        max_batches = int(input("How many batches to process? "))
        return batch_size, max_batches, False
    elif choice == "2":
        batch_size = int(input("Items per batch (recommended: 50-100): ") or "50")
        return batch_size, None, False
    elif choice == "3":
        return None, None, True
    else:
        print("Invalid choice. Using default settings.")
        return 50, 2, False


async def main():
    """Main automation function with batch processing"""
    # Configuration
    DATA_PATH = "/Users/saitejagudidevini/Documents/Dev/StarshipChatbot/output/hierarchical_crawl_www_mhsindiana_com_20251005_172659_filtered.json"
    DB_PATH = "semantic_content.db"
    MAX_CONCURRENT = 2  # Adjust based on API limits
    
    # Initialize processor
    processor = SemanticContentProcessor(DATA_PATH, DB_PATH)
    
    # Show initial status
    initial_status = processor.get_processing_status()
    logger.info(f"📋 Initial status: {initial_status}")
    
    if initial_status['remaining'] == 0:
        logger.info("✅ All items already processed!")
        processor.export_results("processed_semantic_content.json")
        return
    
    # Get user preferences
    batch_size, max_batches, auto_mode = ask_batch_preferences()
    
    if auto_mode:
        # Full automation - process everything
        logger.info("🚀 Starting full automation...")
        results = await processor.process_all(max_concurrent=MAX_CONCURRENT, resume=True)
    else:
        # Batch processing
        logger.info(f"🔄 Starting batch processing...")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Max batches: {max_batches if max_batches else 'unlimited'}")
        
        results = await processor.process_batches(
            max_concurrent=MAX_CONCURRENT, 
            batch_size=batch_size,
            max_batches=max_batches,
            resume=True
        )
    
    # Export results
    processor.export_results("processed_semantic_content.json")
    
    logger.info("🎉 Automation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())