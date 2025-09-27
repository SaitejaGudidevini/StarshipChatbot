"""
JSON Data Visualizer and Categorizer
Web interface to view and categorize crawled data as visible/not visible
Using FastAPI instead of Flask
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
from datetime import datetime
from pathlib import Path
import uvicorn

app = FastAPI(title="Training Data Visualizer")

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

# Global data storage
current_data = None
visible_data = {}
not_visible_data = {}

# Pydantic models for request validation
class CategorizeRequest(BaseModel):
    urls: List[str]
    category: str

class ExportRequest(BaseModel):
    format: Optional[str] = "json"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main visualization page"""
    return templates.TemplateResponse("visualizer.html", {"request": request})

@app.get("/api/load/{filename}")
async def load_json(filename: str):
    """Load a JSON file from output directory"""
    global current_data
    try:
        filepath = f'output/{filename}'
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # Check if this is the new hierarchical flattened format
        if 'semantic_elements' in current_data:
            # New hierarchical flattened format
            formatted_data = []
            for semantic_path, element_data in current_data.get('semantic_elements', {}).items():
                formatted_data.append({
                    'id': semantic_path,  # Use semantic path as unique ID
                    'semantic_path': element_data.get('semantic_path', ''),
                    'element_text': element_data.get('element_text', ''),
                    'element_content': element_data.get('element_content', ''),
                    'element_type': element_data.get('element_type', 'unknown'),
                    'source_type': element_data.get('source_type', 'unknown'),
                    'original_url': element_data.get('original_url', ''),
                    'parent_semantic_path': element_data.get('parent_semantic_path', ''),
                    'depth': element_data.get('depth', 0),
                    'href': element_data.get('href', ''),
                    'button_type': element_data.get('button_type', '')
                })
            
            return JSONResponse({
                'status': 'success',
                'data': formatted_data,
                'metadata': current_data.get('crawl_metadata', {}),
                'total_items': len(formatted_data),
                'format_type': 'hierarchical_flattened'
            })
        
        else:
            # Legacy training data format
            formatted_data = []
            for url, data in current_data.get('training_data', {}).items():
                formatted_data.append({
                    'url': url,
                    'semantic_path': ' → '.join(data.get('semantic_path', [])),
                    'target': data.get('target', ''),
                    'training_phrases': data.get('training_phrases', []),
                    'domain_type': data.get('domain_type', 'unknown'),
                    'meets_golden_image': data.get('meets_golden_image', False),
                    'summary': data.get('summary', '')
                })
            
            return JSONResponse({
                'status': 'success',
                'data': formatted_data,
                'metadata': current_data.get('metadata', {}),
                'total_items': len(formatted_data),
                'format_type': 'legacy_training'
            })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    """List available JSON files in output directory"""
    try:
        output_dir = Path('output')
        if not output_dir.exists():
            return JSONResponse({'files': []})
        
        # Include both legacy and new hierarchical formats
        files = []
        files.extend([f.name for f in output_dir.glob('complete_*.json')])  # Legacy format
        files.extend([f.name for f in output_dir.glob('hierarchical_crawl_*.json')])  # New format
        files.sort(reverse=True)  # Most recent first
        
        return JSONResponse({'files': files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/categorize")
async def categorize(request: CategorizeRequest):
    """Categorize selected items as visible or not visible"""
    global current_data, visible_data, not_visible_data
    
    try:
        selected_ids = request.urls  # Actually IDs for new format
        category = request.category
        
        if not current_data:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        # Handle both new and legacy formats
        if 'semantic_elements' in current_data:
            # New hierarchical flattened format
            data_source = current_data['semantic_elements']
        elif 'training_data' in current_data:
            # Legacy format
            data_source = current_data['training_data']
        else:
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        # Categorize the selected items
        for item_id in selected_ids:
            if item_id in data_source:
                data_item = data_source[item_id]
                
                if category == 'visible':
                    visible_data[item_id] = data_item
                    # Remove from not_visible if it was there
                    not_visible_data.pop(item_id, None)
                else:
                    not_visible_data[item_id] = data_item
                    # Remove from visible if it was there
                    visible_data.pop(item_id, None)
        
        # Save to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save visible data
        if visible_data:
            visible_file = f'output/visible_paths_{timestamp}.json'
            with open(visible_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'category': 'visible',
                    'timestamp': timestamp,
                    'total_items': len(visible_data),
                    'data': visible_data
                }, f, indent=2, ensure_ascii=False)
        
        # Save not visible data
        if not_visible_data:
            not_visible_file = f'output/not_visible_paths_{timestamp}.json'
            with open(not_visible_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'category': 'not_visible',
                    'timestamp': timestamp,
                    'total_items': len(not_visible_data),
                    'data': not_visible_data
                }, f, indent=2, ensure_ascii=False)
        
        return JSONResponse({
            'status': 'success',
            'visible_count': len(visible_data),
            'not_visible_count': len(not_visible_data),
            'message': f'Categorized {len(selected_urls)} items as {category}'
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get current categorization status"""
    return JSONResponse({
        'visible_count': len(visible_data),
        'not_visible_count': len(not_visible_data),
        'total_categorized': len(visible_data) + len(not_visible_data)
    })

@app.post("/api/export")
async def export_final(request: Optional[ExportRequest] = None):
    """Export final categorized data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create final export
    final_export = {
        'export_timestamp': timestamp,
        'visible': {
            'count': len(visible_data),
            'data': visible_data
        },
        'not_visible': {
            'count': len(not_visible_data),
            'data': not_visible_data
        }
    }
    
    export_file = f'output/final_categorized_{timestamp}.json'
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(final_export, f, indent=2, ensure_ascii=False)
    
    return JSONResponse({
        'status': 'success',
        'file': export_file,
        'visible_count': len(visible_data),
        'not_visible_count': len(not_visible_data)
    })

@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup"""
    Path('output').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    print("\n" + "="*60)
    print("JSON DATA VISUALIZER (FastAPI)")
    print("="*60)
    print("\nServer starting...")
    print("Open http://localhost:8000 in your browser")
    print("="*60)

# Additional endpoints for better functionality
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Training Data Visualizer"}

@app.delete("/api/clear")
async def clear_categorization():
    """Clear all categorizations"""
    global visible_data, not_visible_data
    visible_data = {}
    not_visible_data = {}
    return JSONResponse({
        'status': 'success',
        'message': 'All categorizations cleared'
    })

@app.get("/api/data/{category}")
async def get_category_data(category: str):
    """Get data for a specific category"""
    if category == 'visible':
        return JSONResponse({
            'category': 'visible',
            'count': len(visible_data),
            'data': visible_data
        })
    elif category == 'not_visible':
        return JSONResponse({
            'category': 'not_visible',
            'count': len(not_visible_data),
            'data': not_visible_data
        })
    else:
        raise HTTPException(status_code=404, detail="Category not found")

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "visualizer_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )