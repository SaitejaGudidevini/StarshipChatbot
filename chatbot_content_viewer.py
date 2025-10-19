import json
import sys
from pathlib import Path

# Default input file (from browser_agent.py output)
DEFAULT_INPUT = '/Users/saiteja/Documents/Dev/StarshipChatbot/browser_agent_test_output.json'

# Allow custom input file via command line argument
input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

# Check if file exists
if not Path(input_file).exists():
    print(f"❌ Error: Input file not found: {input_file}")
    print(f"💡 Usage: python chatbot_content_viewer.py [input_file.json]")
    print(f"💡 Default: {DEFAULT_INPUT}")
    sys.exit(1)

print(f"📂 Reading chatbot content from: {input_file}")

# Read the chatbot content
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to JavaScript variable
json_data = json.dumps(data, indent=2)

# HTML template with embedded data
html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Content Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            display: grid;
            grid-template-columns: 350px 1fr;
            min-height: 90vh;
        }}

        .sidebar {{
            background: #f8f9fa;
            border-right: 2px solid #e9ecef;
            display: flex;
            flex-direction: column;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 1.5em;
            margin-bottom: 5px;
        }}

        .header p {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .search-box {{
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
        }}

        .search-input {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }}

        .search-input:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .topic-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}

        .topic-item {{
            padding: 15px;
            margin-bottom: 8px;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}

        .topic-item:hover {{
            background: #f8f9fa;
            border-color: #667eea;
            transform: translateX(5px);
        }}

        .topic-item.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .topic-item.hidden {{
            display: none;
        }}

        .topic-name {{
            font-weight: 600;
            font-size: 0.95em;
            margin-bottom: 5px;
        }}

        .topic-status {{
            font-size: 0.8em;
            opacity: 0.7;
        }}

        .content-area {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .content-header {{
            background: #f8f9fa;
            padding: 30px 40px;
            border-bottom: 2px solid #e9ecef;
        }}

        .content-title {{
            font-size: 2em;
            color: #212529;
            margin-bottom: 10px;
        }}

        .content-meta {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .meta-badge {{
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border-radius: 12px;
            font-size: 0.85em;
        }}

        .content-body {{
            flex: 1;
            overflow-y: auto;
            padding: 40px;
        }}

        .content-section {{
            margin-bottom: 30px;
        }}

        .section-label {{
            font-weight: 600;
            color: #495057;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section-content {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            line-height: 1.8;
            color: #212529;
            border-left: 4px solid #667eea;
        }}

        .section-content h1,
        .section-content h2,
        .section-content h3 {{
            margin-top: 20px;
            margin-bottom: 10px;
            color: #667eea;
        }}

        .section-content h1 {{
            font-size: 1.5em;
        }}

        .section-content h2 {{
            font-size: 1.3em;
        }}

        .section-content h3 {{
            font-size: 1.1em;
        }}

        .section-content ul,
        .section-content ol {{
            margin-left: 25px;
            margin-top: 10px;
            margin-bottom: 10px;
        }}

        .section-content li {{
            margin-bottom: 8px;
        }}

        .section-content strong {{
            color: #667eea;
        }}

        .section-content a {{
            color: #667eea;
            text-decoration: none;
            border-bottom: 1px solid #667eea;
        }}

        .section-content a:hover {{
            color: #764ba2;
            border-bottom-color: #764ba2;
        }}

        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #6c757d;
            text-align: center;
            padding: 40px;
        }}

        .empty-state-icon {{
            font-size: 4em;
            margin-bottom: 20px;
        }}

        .empty-state h2 {{
            font-size: 1.5em;
            margin-bottom: 10px;
        }}

        .no-results {{
            text-align: center;
            padding: 40px 20px;
            color: #6c757d;
        }}

        /* Scrollbar styling */
        .topic-list::-webkit-scrollbar,
        .content-body::-webkit-scrollbar {{
            width: 8px;
        }}

        .topic-list::-webkit-scrollbar-track,
        .content-body::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}

        .topic-list::-webkit-scrollbar-thumb,
        .content-body::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 4px;
        }}

        .topic-list::-webkit-scrollbar-thumb:hover,
        .content-body::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}

        @media (max-width: 1024px) {{
            .container {{
                grid-template-columns: 1fr;
                grid-template-rows: auto 1fr;
            }}

            .sidebar {{
                border-right: none;
                border-bottom: 2px solid #e9ecef;
                max-height: 50vh;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="header">
                <h1>📚 Chatbot Content</h1>
                <p id="topic-count">0 topics</p>
            </div>
            <div class="search-box">
                <input
                    type="text"
                    id="searchInput"
                    class="search-input"
                    placeholder="🔍 Search topics..."
                >
            </div>
            <div class="topic-list" id="topicList">
                <!-- Topics will be inserted here -->
            </div>
        </div>

        <div class="content-area">
            <div id="contentDisplay">
                <div class="empty-state">
                    <div class="empty-state-icon">📖</div>
                    <h2>Select a Topic</h2>
                    <p>Choose a topic from the left sidebar to view its chatbot summary</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded JSON data
        const DATA = {json_data};

        let selectedIndex = null;

        // Render topic list
        function renderTopics(topics) {{
            const listContainer = document.getElementById('topicList');

            if (topics.length === 0) {{
                listContainer.innerHTML = `
                    <div class="no-results">
                        <h3>No results found</h3>
                        <p>Try adjusting your search</p>
                    </div>
                `;
                return;
            }}

            listContainer.innerHTML = topics.map((item, index) => {{
                const status = item.status || 'unknown';
                return `
                    <div class="topic-item" data-index="${{index}}" onclick="selectTopic(${{index}})">
                        <div class="topic-name">${{escapeHtml(item.topic)}}</div>
                        <div class="topic-status">Status: ${{status}}</div>
                    </div>
                `;
            }}).join('');
        }}

        // Display selected topic content
        function selectTopic(index) {{
            selectedIndex = index;
            const item = DATA[index];

            // Update active state
            document.querySelectorAll('.topic-item').forEach((el, i) => {{
                if (i === index) {{
                    el.classList.add('active');
                }} else {{
                    el.classList.remove('active');
                }}
            }});

            // Render content
            const contentDisplay = document.getElementById('contentDisplay');

            // Convert markdown-style formatting to HTML
            const formattedSummary = formatMarkdown(item.chatbot_summary || 'No summary available');

            // Handle both field naming conventions (browser_content vs browser_use_content)
            const browserContent = item.browser_content || item.browser_use_content || 'No content available';
            const paraphraseTime = item.paraphrase_time || item.paraphrase_time_seconds || 'N/A';

            contentDisplay.innerHTML = `
                <div class="content-header">
                    <div class="content-title">${{escapeHtml(item.topic)}}</div>
                    <div class="content-meta">
                        <span class="meta-badge">📊 Status: ${{item.status || 'unknown'}}</span>
                        <span class="meta-badge">🤖 Model: ${{item.paraphrase_model || 'N/A'}}</span>
                        <span class="meta-badge">⏱️ ${{typeof paraphraseTime === 'number' ? paraphraseTime.toFixed(2) + 's' : paraphraseTime}}</span>
                    </div>
                </div>
                <div class="content-body">
                    <div class="content-section">
                        <div class="section-label">🤖 Chatbot Summary</div>
                        <div class="section-content">
                            ${{formattedSummary}}
                        </div>
                    </div>

                    <div class="content-section">
                        <div class="section-label">🌐 Original Content</div>
                        <div class="section-content">
                            ${{escapeHtml(browserContent)}}
                        </div>
                    </div>

                    <div class="content-section">
                        <div class="section-label">🔗 Source</div>
                        <div class="section-content">
                            <a href="${{item.original_url || item.semantic_path}}" target="_blank">${{escapeHtml(item.semantic_path)}}</a>
                        </div>
                    </div>
                </div>
            `;
        }}

        // Simple markdown-to-HTML converter
        function formatMarkdown(text) {{
            // Convert **bold** to <strong>
            text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');

            // Convert bullet points (lines starting with * or +)
            text = text.replace(/^\\* (.+)$/gm, '<li>$1</li>');
            text = text.replace(/^\\+ (.+)$/gm, '<li>$1</li>');

            // Wrap consecutive list items in <ul>
            text = text.replace(/(<li>.*?<\\/li>\\s*)+/g, '<ul>$&</ul>');

            // Convert double line breaks to paragraph breaks
            text = text.replace(/\\n\\n/g, '</p><p>');

            // Convert single line breaks to <br>
            text = text.replace(/\\n/g, '<br>');

            // Wrap in paragraphs
            text = '<p>' + text + '</p>';

            return text;
        }}

        // Escape HTML
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // Search and filter topics
        function filterTopics(searchTerm) {{
            const filtered = DATA.filter(item => {{
                const searchLower = searchTerm.toLowerCase();
                const browserContent = item.browser_content || item.browser_use_content || '';
                return !searchTerm ||
                    (item.topic && item.topic.toLowerCase().includes(searchLower)) ||
                    (item.chatbot_summary && item.chatbot_summary.toLowerCase().includes(searchLower)) ||
                    (browserContent && browserContent.toLowerCase().includes(searchLower));
            }});

            renderTopics(filtered);

            // Update count
            document.getElementById('topic-count').textContent = `${{filtered.length}} topic${{filtered.length !== 1 ? 's' : ''}}`;
        }}

        // Initialize
        function init() {{
            // Initial render
            renderTopics(DATA);
            document.getElementById('topic-count').textContent = `${{DATA.length}} topic${{DATA.length !== 1 ? 's' : ''}}`;

            // Set up search
            const searchInput = document.getElementById('searchInput');
            searchInput.addEventListener('input', (e) => {{
                filterTopics(e.target.value);
            }});
        }}

        // Start the app
        init();
    </script>
</body>
</html>
'''

# Generate output filename based on input file
input_path = Path(input_file)
output_file = input_path.parent / f"{input_path.stem}_viewer.html"

# Write the HTML file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"\n{'='*60}")
print(f"✅ Chatbot Content Viewer created!")
print(f"{'='*60}")
print(f"📂 Input:  {input_file}")
print(f"📄 Output: {output_file}")
print(f"📖 Topics: {len(data)}")
print(f"\n🌐 Open '{output_file}' in your browser to view the content!")
print(f"{'='*60}")
