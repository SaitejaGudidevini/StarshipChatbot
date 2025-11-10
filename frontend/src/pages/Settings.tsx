import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { JsonFileList } from '../types';
import { FileJson, Upload, CheckCircle } from 'lucide-react';

export function Settings() {
  const [fileList, setFileList] = useState<JsonFileList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<JsonFileList>('/api/json-files/list');
      setFileList(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const switchFile = async (filename: string) => {
    if (!window.confirm(`Switch to ${filename}? This will reload the chatbot.`)) return;

    try {
      await apiClient.post('/api/json-files/switch', { filename });
      alert('File switched successfully!');
      await loadFiles();
    } catch (err) {
      alert('Failed to switch file: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      await apiClient.upload('/api/json-files/upload', file);
      alert('File uploaded successfully!');
      await loadFiles();
    } catch (err) {
      alert('Failed to upload file: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-slate-900">Settings</h2>
        <p className="text-slate-600 mt-1">Manage JSON data files</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      )}

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-slate-900">JSON Data Files</h3>
          <label className="px-4 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg font-medium hover:from-blue-600 hover:to-cyan-600 transition-all cursor-pointer flex items-center gap-2">
            <Upload className="w-4 h-4" />
            {uploading ? 'Uploading...' : 'Upload File'}
            <input
              type="file"
              accept=".json"
              onChange={handleUpload}
              disabled={uploading}
              className="hidden"
            />
          </label>
        </div>

        <div className="space-y-3">
          {fileList?.files.map((file, idx) => (
            <div
              key={idx}
              className={`p-4 rounded-lg border-2 transition-all ${
                file.is_active
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                    <FileJson className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-slate-900">{file.filename}</p>
                      {file.is_active && (
                        <span className="flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                          <CheckCircle className="w-3 h-3" />
                          Active
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-600 mt-1">
                      {file.topics} topics • {file.qa_pairs} Q&A pairs
                    </p>
                  </div>
                </div>
                {!file.is_active && (
                  <button
                    onClick={() => switchFile(file.filename)}
                    className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors text-sm font-medium"
                  >
                    Switch to this file
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {fileList && fileList.files.length === 0 && (
          <div className="text-center py-12">
            <FileJson className="w-16 h-16 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-600">No JSON files found</p>
            <p className="text-sm text-slate-500 mt-1">Upload a file to get started</p>
          </div>
        )}
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">System Information</h3>
        <div className="space-y-3">
          <InfoRow label="Current File" value={fileList?.current || 'None'} />
          <InfoRow label="Total Files" value={fileList?.files.length.toString() || '0'} />
          <InfoRow label="API Base URL" value="http://localhost:8000" />
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
      <span className="text-slate-600">{label}</span>
      <span className="font-medium text-slate-900">{value}</span>
    </div>
  );
}
