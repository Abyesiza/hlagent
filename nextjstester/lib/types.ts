/** Mirrors `super_agent.app.domain.chat_schemas.ChatTurnResult` JSON. */

export type ChatTurnResult = {
  route: "symbolic" | "neural" | "error";
  intent: string;
  answer: string;
  sympy: Record<string, unknown> | null;
  hdc_similarity: number | null;
  hdc_matched_task: string | null;
  context_snippet: string;
  grounded: boolean;
  session_id: string | null;
  metadata: Record<string, unknown>;
};

export type BlueprintItem = {
  id: string;
  title: string;
  status: "done" | "partial" | "todo";
  notes: string;
};

export type BlueprintSnapshot = {
  version: number;
  items: BlueprintItem[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  turn?: ChatTurnResult;
  error?: string;
  at: number;
};

export type ImproveRequest = {
  instruction: string;
  target_file?: string;
};

export type ImproveResult = {
  ok: boolean;
  target_file: string;
  instruction: string;
  old_code: string;
  new_code: string;
  ast_ok: boolean;
  committed: boolean;
  commit_hash: string | null;
  error: string | null;
  timestamp: string;
};

export type ImprovementHistory = {
  entries: ImproveResult[];
};
