// HDC Super-Agent — shared TypeScript types

export type ChatTurnResult = {
  answer: string;
  mode: "generation" | "math" | "analogy" | "similarity" | "research" | "error";
  confidence: number;
  session_id: string | null;
  details: Record<string, unknown>;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  turn?: ChatTurnResult;
  error?: string;
  at: number;
};

export type ModelStats = {
  model: string;
  vocab_size: number;
  training_tokens: number;
  training_docs: number;
  assoc_memory: number;
  last_trained: string | null;
  hdc_memory_records: number;
  convex_connected: boolean;
};

export type PipelineStats = {
  pipeline: {
    documents_trained: number;
    documents_skipped: number;
    tokens_trained: number;
    research_runs: number;
    unique_docs_seen: number;
    last_run: string;
    recent_errors: string[];
  };
  model: {
    vocab_size: number;
    training_tokens: number;
    training_docs: number;
    assoc_memory_size: number;
    dim: number;
    context_size: number;
    last_trained: string;
  };
};

export type HeartbeatStatus = {
  last_run: string | null;
  total_runs: number;
  topic_count: number;
  next_topic_index: number;
  next_topic: string | null;
  topics: string[];
};

export type VocabResponse = {
  vocab_size: number;
  sample: string[];
};

export type AnalogyCandidate = {
  word: string;
  similarity: number;
};

export type AnalogyResult = {
  query: string;
  candidates: AnalogyCandidate[];
  best: string | null;
};

export type SimilarResult = {
  word: string;
  similar: AnalogyCandidate[];
};

export type GenerateResult = {
  seed: string;
  generated: string;
  full_text: string;
};

export type MemoryRecord = {
  task_fp: string;
  solution_preview: string;
  route: string;
  retrieval_count: number;
};

export type MemoryResponse = {
  records: MemoryRecord[];
  stats: {
    total_records: number;
    by_route: Record<string, number>;
    torchhd_available: boolean;
    dim: number;
  };
};

export type HealthCheck = {
  ok: boolean;
  model?: string;
  hdc_dim?: number;
  convex_configured?: boolean;
  detail?: string;
};
