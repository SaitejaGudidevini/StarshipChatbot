export interface DashboardStats {
  total_topics: number;
  total_qa: number;
  chatbot_status: string;
  editor_status: string;
  generator_status: string;
}

export interface PipelineStage {
  stage_number?: number;
  stage_name?: string;
  stage: string;
  score: number;
  matched_by?: string;
  duration?: number;
  best_match_question?: string;
  best_match_topic?: string;
  threshold_ideal?: number;
  threshold_minimal?: number;
  meets_ideal?: boolean;
  meets_minimal?: boolean;
  top_k_candidates?: number;
  reranker_used?: boolean;
  entity_validation?: boolean;
  rephrased_text?: string;
  semantic_score?: number;  // For comparing semantic vs reranker scores
}

export interface PipelineInfo {
  session_id?: string;
  original_question?: string;
  timestamp?: string;
  stages?: PipelineStage[];
  // V2 Architecture fields
  architecture?: string;
  query_analysis?: {
    intent: string;
    entities: Record<string, any>;
    semantic_query?: string;
  };
  retrieval_details?: {
    retrievers: Record<string, any>;
    rrf_fusion?: any[];
  };
  candidates_evaluated?: number;
  candidates_verified?: number;
  duration?: number;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  text: string;
  confidence?: number;
  topic?: string;
  matched_by?: string;
  pipeline_info?: PipelineInfo;
}

export interface ChatResponse {
  question: string;
  answer: string;
  confidence: number;
  matched_by: string;
  source_topic?: string;
  source_qa_index?: number;
  pipeline_info?: PipelineInfo;
}

export interface Topic {
  name: string;
  qa_count: number;
  has_buckets: boolean;
}

export interface TopicDetail {
  topic: string;
  qa_count: number;
  qa_pairs: QAPair[];
}

export interface QAPair {
  question: string;
  answer: string;
  is_bucketed: boolean;
  bucket_id: string | null;
  is_unified: boolean;
}

export interface GenerationProgress {
  status: 'idle' | 'initializing' | 'building_graph' | 'running' | 'completed' | 'cancelled' | 'error';
  current: number;
  total: number;
  qa_generated: number;
  current_url?: string;
  elapsed_seconds: number;
  error?: string;
}

export interface JsonFile {
  filename: string;
  topics: number;
  qa_pairs: number;
  is_active: boolean;
}

export interface JsonFileList {
  files: JsonFile[];
  current: string;
}

export interface TreeNode {
  title: string;
  url: string;
  semantic_path: string;
  source_type: 'homepage' | 'heading' | 'link';
  depth: number;
  visited: boolean;
  has_content: boolean;
  children: TreeNode[];
}

export interface TreeData {
  metadata: {
    domain: string;
    start_url: string;
    timestamp: string;
    total_nodes: number;
    max_depth: number;
    format: string;
  };
  tree: TreeNode;
}
