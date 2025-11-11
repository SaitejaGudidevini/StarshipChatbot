export interface DashboardStats {
  total_topics: number;
  total_qa: number;
  chatbot_status: string;
  editor_status: string;
  generator_status: string;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  text: string;
  confidence?: number;
  topic?: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  confidence: number;
  matched_by: string;
  source_topic?: string;
  source_qa_index?: number;
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
