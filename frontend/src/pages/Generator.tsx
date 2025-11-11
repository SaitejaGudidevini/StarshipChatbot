import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { GenerationProgress } from '../types';
import { Zap, Play, Square, AlertCircle } from 'lucide-react';

export function Generator() {
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState(10);
  const [useCrawler, setUseCrawler] = useState(false);
  const [maxDepth, setMaxDepth] = useState(2);
  const [maxItems, setMaxItems] = useState<number | null>(null);
  const [threadId, setThreadId] = useState('');
  const [enableCheckpointing, setEnableCheckpointing] = useState(true);
  const [outputFilename, setOutputFilename] = useState('');
  const [jsonFilename, setJsonFilename] = useState('');
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Load available JSON files
  useEffect(() => {
    const loadFiles = async () => {
      try {
        const data = await apiClient.get<{files: Array<{filename: string}>}>('/api/json-files/list');
        setAvailableFiles(data.files.map(f => f.filename));
        if (data.files.length > 0 && !jsonFilename) {
          setJsonFilename(data.files[0].filename); // Select first file by default
        }
      } catch (err) {
        console.error('Failed to load JSON files:', err);
      }
    };
    loadFiles();
  }, []);

  useEffect(() => {
    // Use same origin for SSE connection (works in both dev and production)
    const apiBaseUrl = import.meta.env.VITE_API_URL || window.location.origin;
    const sseUrl = `${apiBaseUrl}/api/generate/stream`;
    console.log('[SSE] Connecting to:', sseUrl);
    const eventSource = new EventSource(sseUrl);

    eventSource.onopen = () => {
      console.log('[SSE] Connection opened');
      setIsConnected(true);
    };

    eventSource.onmessage = (event) => {
      console.log('[SSE] Message received:', event.data);
      const data: GenerationProgress = JSON.parse(event.data);
      console.log('[SSE] Parsed progress:', data);
      setProgress(data);

      if (data.status === 'completed' || data.status === 'error') {
        console.log('[SSE] Generation finished, closing connection');
        eventSource.close();
        setIsConnected(false);
      }
    };

    eventSource.onerror = (error) => {
      console.error('[SSE] Error occurred:', error);
      eventSource.close();
      setIsConnected(false);
    };

    return () => {
      console.log('[SSE] Cleaning up connection');
      eventSource.close();
    };
  }, []);

  const startGeneration = async () => {
    // Validate based on mode
    if (useCrawler && !url.trim()) {
      alert('Please enter a URL');
      return;
    }
    if (!useCrawler) {
      if (!jsonFilename) {
        alert('Please select a JSON file');
        return;
      }
      if (!threadId.trim()) {
        alert('Thread ID is required when processing JSON files (for checkpointing and resume)');
        return;
      }
    }

    try {
      await apiClient.post('/api/generate/start', {
        url: useCrawler ? url : null,
        max_pages: maxPages,
        use_crawler: useCrawler,
        max_depth: maxDepth,
        max_items: maxItems,
        thread_id: threadId || null,
        enable_checkpointing: enableCheckpointing,
        output_filename: outputFilename || null,
        json_filename: !useCrawler ? jsonFilename : null,
      });
    } catch (err) {
      alert('Failed to start generation: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const cancelGeneration = async () => {
    try {
      await apiClient.post('/api/generate/cancel');
    } catch (err) {
      alert('Failed to cancel: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const isProcessing = progress?.status === 'running' || progress?.status === 'initializing' || progress?.status === 'building_graph';
  const percent = progress && progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-slate-900">Generator</h2>
        <p className="text-slate-600 mt-1">Generate Q&A data from websites</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="space-y-4">
          {/* Conditional: Show URL input if using crawler, JSON file selector otherwise */}
          {useCrawler ? (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Website URL
              </label>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                disabled={isProcessing}
                className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-500"
              />
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Select JSON File
              </label>
              <select
                value={jsonFilename}
                onChange={(e) => setJsonFilename(e.target.value)}
                disabled={isProcessing}
                className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-500"
              >
                {availableFiles.length === 0 && (
                  <option value="">No JSON files available</option>
                )}
                {availableFiles.map((filename) => (
                  <option key={filename} value={filename}>
                    {filename}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500 mt-1">
                Process existing JSON file from Settings
              </p>
            </div>
          )}

          {/* Thread ID - Show when checkpointing is enabled OR when processing JSON files */}
          {(enableCheckpointing || !useCrawler) && (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Thread ID {!useCrawler && <span className="text-red-500">*</span>}
                {useCrawler && <span className="text-slate-500 font-normal ml-1">(optional)</span>}
              </label>
              <input
                type="text"
                value={threadId}
                onChange={(e) => setThreadId(e.target.value)}
                placeholder={!useCrawler ? "e.g., pytorch_processing_1" : "Auto-generated (leave empty for new run)"}
                disabled={isProcessing}
                className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-500"
              />
              <p className="text-xs text-slate-500 mt-1">
                {!useCrawler
                  ? "Required for checkpointing. Use the same Thread ID to resume interrupted runs."
                  : "Optional. Provide a custom Thread ID to resume an interrupted crawler run. Leave empty to auto-generate."
                }
              </p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Max Pages
              </label>
              <input
                type="number"
                value={maxPages}
                onChange={(e) => setMaxPages(parseInt(e.target.value) || 10)}
                min="1"
                max="100"
                disabled={isProcessing}
                className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-500"
              />
            </div>

            <div className="space-y-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useCrawler}
                  onChange={(e) => setUseCrawler(e.target.checked)}
                  disabled={isProcessing}
                  className="w-4 h-4 text-blue-500 rounded focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-slate-700">Use Crawler</span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableCheckpointing}
                  onChange={(e) => setEnableCheckpointing(e.target.checked)}
                  disabled={isProcessing}
                  className="w-4 h-4 text-blue-500 rounded focus:ring-blue-500"
                />
                <span className="text-sm font-medium text-slate-700">Enable Checkpointing</span>
                <span className="text-xs text-slate-500">(Auto-save & resume on failure)</span>
              </label>
            </div>
          </div>

          {/* Advanced Options */}
          <div className="border-t border-slate-200 pt-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900"
            >
              <Zap className="w-4 h-4" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced Options
            </button>

            {showAdvanced && (
              <div className="mt-4 space-y-4 bg-slate-50 rounded-lg p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Max Crawl Depth
                      <span className="text-slate-500 font-normal ml-1">(1-5)</span>
                    </label>
                    <input
                      type="number"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(parseInt(e.target.value) || 2)}
                      min="1"
                      max="5"
                      disabled={isProcessing}
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
                    />
                    <p className="text-xs text-slate-500 mt-1">How many levels deep to crawl</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Max Items to Process
                      <span className="text-slate-500 font-normal ml-1">(optional)</span>
                    </label>
                    <input
                      type="number"
                      value={maxItems || ''}
                      onChange={(e) => setMaxItems(e.target.value ? parseInt(e.target.value) : null)}
                      min="1"
                      placeholder="Process all"
                      disabled={isProcessing}
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
                    />
                    <p className="text-xs text-slate-500 mt-1">Leave empty to process all items</p>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Output Filename
                    <span className="text-slate-500 font-normal ml-1">(optional)</span>
                  </label>
                  <input
                    type="text"
                    value={outputFilename}
                    onChange={(e) => setOutputFilename(e.target.value)}
                    placeholder="Auto-generated with timestamp"
                    disabled={isProcessing}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-100"
                  />
                  <p className="text-xs text-slate-500 mt-1">Custom output JSON filename</p>
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            {!isProcessing ? (
              <button
                onClick={startGeneration}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg font-medium hover:from-blue-600 hover:to-cyan-600 transition-all flex items-center gap-2"
              >
                <Play className="w-4 h-4" />
                Start Generation
              </button>
            ) : (
              <button
                onClick={cancelGeneration}
                className="px-6 py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 transition-all flex items-center gap-2"
              >
                <Square className="w-4 h-4" />
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>

      {progress && (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900">Progress</h3>
            <span
              className={`px-3 py-1 rounded-full text-xs font-medium ${
                progress.status === 'running' || progress.status === 'initializing' || progress.status === 'building_graph'
                  ? 'bg-blue-100 text-blue-700'
                  : progress.status === 'completed'
                  ? 'bg-green-100 text-green-700'
                  : progress.status === 'error'
                  ? 'bg-red-100 text-red-700'
                  : 'bg-slate-100 text-slate-600'
              }`}
            >
              {progress.status}
            </span>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-600">
                  Pages: {progress.current} / {progress.total}
                </span>
                <span className="font-medium text-slate-900">{Math.round(percent)}%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300 rounded-full"
                  style={{ width: `${percent}%` }}
                ></div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-50 rounded-lg p-4">
                <p className="text-sm text-slate-600">Q&A Generated</p>
                <p className="text-2xl font-bold text-slate-900 mt-1">{progress.qa_generated}</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-4">
                <p className="text-sm text-slate-600">Elapsed Time</p>
                <p className="text-2xl font-bold text-slate-900 mt-1">{progress.elapsed_seconds}s</p>
              </div>
            </div>

            {progress.current_url && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-xs font-medium text-blue-600 mb-1">Current URL</p>
                <p className="text-sm text-blue-900 break-all">{progress.current_url}</p>
              </div>
            )}

            {progress.error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-900">Error</p>
                  <p className="text-sm text-red-700 mt-1">{progress.error}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {!progress && (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-12 text-center">
          <Zap className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <p className="text-slate-600">No generation in progress</p>
          <p className="text-sm text-slate-500 mt-1">Enter a URL and click Start to begin</p>
        </div>
      )}
    </div>
  );
}
