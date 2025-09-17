"""
JSON Data Visualizer and Categorizer
Web interface to view and categorize crawled data as visible/not visible
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Global data storage
current_data = None
visible_data = {}
not_visible_data = {}

@app.route('/')
def index():
    """Main visualization page"""
    return render_template('visualizer.html')

@app.route('/api/load/<filename>')
def load_json(filename):
    """Load a JSON file from output directory"""
    global current_data
    try:
        filepath = f'output/{filename}'
        with open(filepath, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # Extract just the training data for visualization
        formatted_data = []
        for url, data in current_data.get('training_data', {}).items():
            formatted_data.append({
                'url': url,
                'semantic_path': ' → '.join(data.get('semantic_path', [])),
                'target': data.get('target', ''),
                'training_phrases': data.get('training_phrases', []),
                'domain_type': data.get('domain_type', 'unknown'),
                'meets_golden_image': data.get('meets_golden_image', False)
            })
        
        return jsonify({
            'status': 'success',
            'data': formatted_data,
            'metadata': current_data.get('metadata', {}),
            'total_items': len(formatted_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/files')
def list_files():
    """List available JSON files in output directory"""
    try:
        output_dir = Path('output')
        if not output_dir.exists():
            return jsonify({'files': []})
        
        files = [f.name for f in output_dir.glob('complete_*.json')]
        files.sort(reverse=True)  # Most recent first
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/categorize', methods=['POST'])
def categorize():
    """Categorize selected items as visible or not visible"""
    global current_data, visible_data, not_visible_data
    
    try:
        req_data = request.json
        selected_urls = req_data.get('urls', [])
        category = req_data.get('category', 'visible')
        
        if not current_data or 'training_data' not in current_data:
            return jsonify({'status': 'error', 'message': 'No data loaded'}), 400
        
        # Categorize the selected URLs
        for url in selected_urls:
            if url in current_data['training_data']:
                data_item = current_data['training_data'][url]
                
                if category == 'visible':
                    visible_data[url] = data_item
                    # Remove from not_visible if it was there
                    not_visible_data.pop(url, None)
                else:
                    not_visible_data[url] = data_item
                    # Remove from visible if it was there
                    visible_data.pop(url, None)
        
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
        
        return jsonify({
            'status': 'success',
            'visible_count': len(visible_data),
            'not_visible_count': len(not_visible_data),
            'message': f'Categorized {len(selected_urls)} items as {category}'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get current categorization status"""
    return jsonify({
        'visible_count': len(visible_data),
        'not_visible_count': len(not_visible_data),
        'total_categorized': len(visible_data) + len(not_visible_data)
    })

@app.route('/api/export', methods=['POST'])
def export_final():
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
    
    return jsonify({
        'status': 'success',
        'file': export_file,
        'visible_count': len(visible_data),
        'not_visible_count': len(not_visible_data)
    })

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    Path('output').mkdir(exist_ok=True)
    
    # Create templates directory
    Path('templates').mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("JSON DATA VISUALIZER")
    print("="*60)
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60)
    
    app.run(debug=True, port=5000)