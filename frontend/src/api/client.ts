// Use same origin as frontend (works for both local dev and Railway production)
// In production: frontend is served from Railway, so API is same origin
// In local dev: can override with VITE_API_URL env var
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export const apiClient = {
  async get<T>(path: string): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`);
    if (!response.ok) throw new Error(`API error: ${response.statusText}`);
    return response.json();
  },

  async post<T>(path: string, data?: unknown): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined,
    });
    if (!response.ok) throw new Error(`API error: ${response.statusText}`);
    return response.json();
  },

  async upload<T>(path: string, file: File): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      // Try to get detailed error message from response body
      let errorDetail = response.statusText;
      try {
        const errorData = await response.json();
        errorDetail = errorData.detail || errorData.message || response.statusText;
      } catch {
        // If response is not JSON, use statusText
      }
      throw new Error(`Upload failed (${response.status}): ${errorDetail}`);
    }
    return response.json();
  },
};
