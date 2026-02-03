import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { GenerationProgress, ParallelProgressEvent, WorkerState, LiveTreeNode, NodeState } from '../types';

interface GeneratorContextType {
  // State
  progress: GenerationProgress | null;
  parallelProgress: ParallelProgressEvent | null;
  workers: Record<string, WorkerState>;
  isParallelMode: boolean;
  isConnected: boolean;

  // Live tree state (NEW)
  liveTree: LiveTreeNode | null;
  nodeStates: Record<string, NodeState>;  // semantic_path → {status, workerId}

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

  // Live tree state (NEW)
  const [liveTree, setLiveTree] = useState<LiveTreeNode | null>(null);
  const [nodeStates, setNodeStates] = useState<Record<string, NodeState>>({});

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

        // ─── NEW: Handle tree_init event ───
        if (data.type === 'tree_init') {
          console.log('[SSE Context] Tree init received, nodes:', data.total_nodes);
          setLiveTree(data.tree);
          // Initialize all nodes as "pending"
          const initialStates: Record<string, NodeState> = {};
          const walkTree = (node: LiveTreeNode) => {
            initialStates[node.semantic_path] = { status: 'pending' };
            node.children?.forEach(walkTree);
          };
          walkTree(data.tree);
          setNodeStates(initialStates);
          return;  // tree_init is not a progress event, don't process further
        }

        // ─── NEW: Handle semantic_path in worker_update ───
        if (data.type === 'worker_update' && data.semantic_path) {
          setNodeStates(prev => ({
            ...prev,
            [data.semantic_path]: {
              status: 'processing',
              workerId: data.worker_id,
            },
          }));
        }

        // ─── NEW: Handle semantic_path in item_completed ───
        if (data.type === 'item_completed' && data.semantic_path) {
          setNodeStates(prev => ({
            ...prev,
            [data.semantic_path]: {
              status: data.success ? 'completed' : 'error',
              workerId: data.worker_id,
            },
          }));
        }

        // ─── Existing parallel progress handling ───
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
            // Check if ALL batches are done using overall progress
            const allDone = data.completed >= data.total && data.current_batch >= data.total_batches;
            if (allDone) {
              console.log('[SSE Context] All batches completed, closing connection');
              es.close();
              setIsConnected(false);
              setShouldConnect(false);
            } else {
              console.log(`[SSE Context] Batch ${data.current_batch}/${data.total_batches} completed (${data.completed}/${data.total} items), waiting for next...`);
            }
          }
        } else {
          // Legacy progress event
          setIsParallelMode(false);
          setProgress(data as GenerationProgress);

          if (data.status === 'completed' || data.status === 'error' || data.status === 'cancelled') {
            console.log(`[SSE Context] Generation ${data.status}, closing connection`);
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
    setLiveTree(null);
    setNodeStates({});
  }, []);

  return (
    <GeneratorContext.Provider
      value={{
        progress,
        parallelProgress,
        workers,
        isParallelMode,
        isConnected,
        liveTree,
        nodeStates,
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
