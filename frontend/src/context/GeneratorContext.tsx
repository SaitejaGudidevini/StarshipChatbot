import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { GenerationProgress, ParallelProgressEvent, WorkerState } from '../types';

interface GeneratorContextType {
  // State
  progress: GenerationProgress | null;
  parallelProgress: ParallelProgressEvent | null;
  workers: Record<string, WorkerState>;
  isParallelMode: boolean;
  isConnected: boolean;

  // Actions
  startSSE: () => void;
  stopSSE: () => void;
  resetProgress: () => void;
}

const GeneratorContext = createContext<GeneratorContextType | null>(null);

export function GeneratorProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [parallelProgress, setParallelProgress] = useState<ParallelProgressEvent | null>(null);
  const [workers, setWorkers] = useState<Record<string, WorkerState>>({});
  const [isParallelMode, setIsParallelMode] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [shouldConnect, setShouldConnect] = useState(false);

  // SSE Connection Effect
  useEffect(() => {
    if (!shouldConnect) {
      return;
    }

    // Small delay to let backend create the tracker
    const connectTimeout = setTimeout(() => {
      const apiBaseUrl = import.meta.env.VITE_API_URL || window.location.origin;
      const sseUrl = `${apiBaseUrl}/api/generate/stream`;
      console.log('[SSE Context] Connecting to:', sseUrl);

      const es = new EventSource(sseUrl);
      setEventSource(es);

      es.onopen = () => {
        console.log('[SSE Context] Connection opened');
        setIsConnected(true);
      };

      es.onmessage = (event) => {
        console.log('[SSE Context] Message received:', event.data);
        const data = JSON.parse(event.data);

        // Check if this is a parallel progress event
        if (data.type && data.workers !== undefined) {
          setIsParallelMode(true);
          setParallelProgress(data as ParallelProgressEvent);
          setWorkers(data.workers);

          // Map to legacy progress format
          setProgress({
            status: data.type === 'batch_completed'
              ? (data.status === 'error' ? 'error' : 'completed')
              : 'running',
            current: data.completed,
            total: data.total,
            qa_generated: data.completed,
            elapsed_seconds: 0,
            error: data.error
          });

          if (data.type === 'batch_completed') {
            console.log('[SSE Context] Batch completed, closing connection');
            es.close();
            setIsConnected(false);
            setShouldConnect(false);
          }
        } else {
          // Legacy progress event
          setIsParallelMode(false);
          setProgress(data as GenerationProgress);

          if (data.status === 'completed' || data.status === 'error') {
            console.log('[SSE Context] Generation finished, closing connection');
            es.close();
            setIsConnected(false);
            setShouldConnect(false);
          }
        }
      };

      es.onerror = (err) => {
        console.error('[SSE Context] Connection error:', err);
        setIsConnected(false);
      };
    }, 1000);

    return () => {
      clearTimeout(connectTimeout);
      if (eventSource) {
        console.log('[SSE Context] Cleaning up connection');
        eventSource.close();
      }
    };
  }, [shouldConnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [eventSource]);

  const startSSE = useCallback(() => {
    console.log('[SSE Context] Starting SSE connection');
    setShouldConnect(true);
  }, []);

  const stopSSE = useCallback(() => {
    console.log('[SSE Context] Stopping SSE connection');
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
    }
    setIsConnected(false);
    setShouldConnect(false);
  }, [eventSource]);

  const resetProgress = useCallback(() => {
    setProgress(null);
    setParallelProgress(null);
    setWorkers({});
    setIsParallelMode(false);
  }, []);

  return (
    <GeneratorContext.Provider
      value={{
        progress,
        parallelProgress,
        workers,
        isParallelMode,
        isConnected,
        startSSE,
        stopSSE,
        resetProgress,
      }}
    >
      {children}
    </GeneratorContext.Provider>
  );
}

export function useGenerator() {
  const context = useContext(GeneratorContext);
  if (!context) {
    throw new Error('useGenerator must be used within a GeneratorProvider');
  }
  return context;
}
