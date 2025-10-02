"""
Vector Query Service Wrapper
Interfaces with the existing query_chroma.py to provide vector database functionality
"""

import asyncio
import subprocess
import json
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorQueryService:
    """
    Service wrapper for vector database queries
    Uses the existing query_chroma.py functionality
    """
    
    def __init__(self, chroma_script_path: str = "query_chroma.py"):
        self.chroma_script_path = chroma_script_path
        self.initialized = False
        
    async def initialize(self):
        """Initialize the vector service"""
        try:
            # Check if the chroma script exists
            if not os.path.exists(self.chroma_script_path):
                logger.error(f"❌ Chroma script not found: {self.chroma_script_path}")
                self.initialized = False
                return False
                
            self.initialized = True
            logger.info("🔍 Vector Query Service initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector service: {e}")
            self.initialized = False
            return False
    
    async def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar content
        
        Args:
            query_text: The user's question/query
            top_k: Number of top results to return
            
        Returns:
            List of matching results with semantic paths, URLs, types, and similarities
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            if not self.initialized:
                return []
            
            print(f"🔍 Querying vector DB: '{query_text}' (top_k={top_k})")
            
            # Run the query_chroma.py script with the query
            result = await self._run_chroma_query(query_text, top_k)
            
            if not result:
                return []
            
            # Parse and format the results
            formatted_results = self._format_results(result)
            
            print(f"✅ Found {len(formatted_results)} relevant results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            return []
    
    async def _run_chroma_query(self, query_text: str, top_k: int) -> Optional[str]:
        """Run chroma query using the working query_chroma.py functions"""
        try:
            # Import the working functions from query_chroma.py
            import sys
            import os
            sys.path.append(os.getcwd())
            import query_chroma
            import chromadb
            
            # Use the exact same code from query_chroma.py that works
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(name="pytorch_docs")
            
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            # Format output exactly like query_chroma.py does
            output_lines = []
            for meta, distance in zip(results['metadatas'][0], results['distances'][0]):
                output_lines.extend([
                    meta['semantic_path'],
                    f"URL: {meta['original_url']}",
                    f"Type: {meta['source_type']} / {meta['element_type']}",
                    f"Similarity: {1 - distance:.4f}",
                    ""  # Empty line separator
                ])
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error running chroma query: {e}")
            return None
    
    def _format_results(self, raw_output: str) -> List[Dict[str, Any]]:
        """
        Format the raw output from query_chroma.py into structured results
        
        Expected input format:
        https://pytorch.org/Announcements/PyTorch Foundation Welcomes DeepSpeed as a Hosted Project
           URL: https://pytorch.org/blog/category/announcements
           Type: heading / heading
           Similarity: -0.3128
        """
        try:
            results = []
            lines = raw_output.strip().split('\n')
            
            current_result = {}
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    # Empty line - finalize current result if we have one
                    if current_result:
                        results.append(current_result)
                        current_result = {}
                    continue
                
                if line.startswith('http') and 'URL:' not in line and 'Type:' not in line and 'Similarity:' not in line:
                    # This is a semantic path
                    current_result['semantic_path'] = line
                    
                elif line.startswith('URL:'):
                    current_result['url'] = line.replace('URL:', '').strip()
                    
                elif line.startswith('Type:'):
                    current_result['type'] = line.replace('Type:', '').strip()
                    
                elif line.startswith('Similarity:'):
                    similarity_str = line.replace('Similarity:', '').strip()
                    try:
                        current_result['similarity'] = float(similarity_str)
                    except ValueError:
                        current_result['similarity'] = 0.0
            
            # Don't forget the last result
            if current_result:
                results.append(current_result)
            
            # Validate and clean results
            valid_results = []
            for result in results:
                if all(key in result for key in ['semantic_path', 'url', 'type', 'similarity']):
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the vector service"""
        try:
            if not os.path.exists(self.chroma_script_path):
                return {
                    "status": "unhealthy",
                    "error": f"Chroma script not found: {self.chroma_script_path}"
                }
            
            # Try a simple test query
            test_result = await self.query("test", top_k=1)
            
            return {
                "status": "healthy" if test_result is not None else "unhealthy",
                "script_path": self.chroma_script_path,
                "initialized": self.initialized
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Alternative implementation if query_chroma.py doesn't support command line args
class DirectVectorQueryService:
    """
    Alternative implementation that imports query_chroma.py directly
    Use this if the command-line approach doesn't work
    """
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        """Initialize by importing the query module"""
        try:
            # Try to import the query_chroma module
            import query_chroma
            self.query_module = query_chroma
            self.initialized = True
            logger.info("🔍 Direct Vector Query Service initialized")
            return True
            
        except ImportError as e:
            logger.error(f"Could not import query_chroma: {e}")
            self.initialized = False
            return False
    
    async def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query using direct module import"""
        try:
            if not self.initialized:
                await self.initialize()
                
            if not self.initialized:
                return []
            
            # Call the query function from the imported module
            # Note: This assumes query_chroma.py has a function we can call
            # Adjust based on the actual implementation
            results = self.query_module.query_similar(query_text, top_k=top_k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in direct query: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for direct service"""
        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "method": "direct_import",
            "initialized": self.initialized
        }

# Example usage and testing
async def main():
    """Test the vector query service"""
    service = VectorQueryService()
    
    # Test query
    results = await service.query("DeepSpeed", top_k=3)
    
    print("📊 Query Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['semantic_path']}")
        print(f"   URL: {result['url']}")
        print(f"   Type: {result['type']}")
        print(f"   Similarity: {result['similarity']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())