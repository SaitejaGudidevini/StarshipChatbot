import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { GenerationProgress, ParallelProgressEvent, WorkerState, LiveTreeNode, NodeState } from '../types';

// Crawl phase progress (Phase 1: Discovery)
interface CrawlProgress {
  phase: string;
  description: string;
  pagesDiscovered: number;
  pagesProcessed: number;
  currentUrl: string;
  phaseName: string;
}

interface GeneratorContextType {
  // State
  progress: GenerationProgress | null;
  parallelProgress: ParallelProgressEvent | null;
  workers: Record<string, WorkerState>;
  isParallelMode: boolean;
  isConnected: boolean;

  // Crawl phase state (Phase 1: Discovery)
  crawlProgress: CrawlProgress | null;
  currentPhase: 'idle' | 'discovery' | 'qa_generation' | 'complete';

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

  // Crawl phase state (Phase 1: Discovery)
  const [crawlProgress, setCrawlProgress] = useState<CrawlProgress | null>(null);
  const [currentPhase, setCurrentPhase] = useState<'idle' | 'discovery' | 'qa_generation' | 'complete'>('idle');

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

        // ─── Handle phase_start event (Discovery phase begins) ───
        if (data.type === 'phase_start') {
          console.log('[SSE Context] Phase started:', data.phase, data.description);
          setCurrentPhase(data.phase === 'discovery' ? 'discovery' : 'qa_generation');
          setCrawlProgress({
            phase: data.phase,
            description: data.description,
            pagesDiscovered: 0,
            pagesProcessed: 0,
            currentUrl: '',
            phaseName: ''
          });
          // Set initial progress for UI
          setProgress({
            status: 'initializing',
            current: 0,
            total: 0,
            qa_generated: 0,
            elapsed_seconds: 0
          });
          return;
        }

        // ─── Handle crawl_progress event (Discovery phase updates) ───
        if (data.type === 'crawl_progress') {
          console.log('[SSE Context] Crawl progress:', data.pages_discovered, 'discovered,', data.pages_processed, 'processed');
          setCrawlProgress(prev => ({
            ...prev!,
            pagesDiscovered: data.pages_discovered,
            pagesProcessed: data.pages_processed,
            currentUrl: data.current_url || '',
            phaseName: data.phase_name || ''
          }));
          // Update legacy progress for UI compatibility
          setProgress(prev => ({
            ...prev!,
            status: 'building_graph',
            current: data.pages_processed,
            total: data.pages_discovered,
            current_url: data.current_url
          }));
          return;
        }

        // ─── Handle phase_complete event (Discovery phase ends) ───
        if (data.type === 'phase_complete') {
          console.log('[SSE Context] Phase complete:', data.phase, data.summary);
          if (data.phase === 'discovery') {
            // Discovery done, Q&A generation will start soon
            setCrawlProgress(prev => ({
              ...prev!,
              pagesDiscovered: data.summary?.pages_discovered || prev?.pagesDiscovered || 0,
              pagesProcessed: data.summary?.pages_processed || prev?.pagesProcessed || 0,
              phaseName: 'Discovery Complete'
            }));
            setCurrentPhase('qa_generation');
          }
          return;
        }

        // ─── Handle tree_init event ───
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
    setCrawlProgress(null);
    setCurrentPhase('idle');
  }, []);

  return (
    <GeneratorContext.Provider
      value={{
        progress,
        parallelProgress,
        workers,
        isParallelMode,
        isConnected,
        crawlProgress,
        currentPhase,
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
