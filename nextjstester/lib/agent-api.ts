import type {
  AnalogyResult,
  ChatTurnResult,
  GenerateResult,
  HealthCheck,
  HeartbeatStatus,
  MemoryResponse,
  ModelStats,
  PipelineStats,
  SimilarResult,
  VocabResponse,
} from "./types";

// ── base URL helpers ──────────────────────────────────────────────────────────

function defaultBase(): string {
  if (typeof process !== "undefined" && process.env.NEXT_PUBLIC_AGENT_API_URL) {
    return process.env.NEXT_PUBLIC_AGENT_API_URL.replace(/\/$/, "");
  }
  return "";
}

export function getAgentBaseUrl(): string {
  if (typeof window === "undefined") return defaultBase();
  try {
    const saved = localStorage.getItem("hlagent-api-base");
    if (saved?.trim()) return saved.trim().replace(/\/$/, "");
  } catch { /* ignore */ }
  return defaultBase();
}

const SESSION_KEY = "hlagent-session-id";

export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();
  try {
    const existing = localStorage.getItem(SESSION_KEY);
    if (existing) return existing;
  } catch { /* ignore */ }
  const id = crypto.randomUUID();
  try { localStorage.setItem(SESSION_KEY, id); } catch { /* ignore */ }
  return id;
}

export function resetSessionId(): string {
  const id = crypto.randomUUID();
  try { localStorage.setItem(SESSION_KEY, id); } catch { /* ignore */ }
  return id;
}

// ── health ────────────────────────────────────────────────────────────────────

export async function fetchHealth(base: string): Promise<HealthCheck> {
  const url = (base || "").trim().replace(/\/$/, "");
  if (!url) {
    return { ok: false, detail: "No API URL set. Enter your backend URL in settings." };
  }
  try {
    const r = await fetch(`${url}/health`, { cache: "no-store", mode: "cors" });
    if (!r.ok) return { ok: false, detail: `HTTP ${r.status} from ${url}/health` };
    const data = await r.json() as Record<string, unknown>;
    return {
      ok: data.status === "ok",
      model: data.model as string,
      hdc_dim: data.hdc_dim as number,
      convex_configured: data.convex_configured as boolean,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : "fetch failed";
    return { ok: false, detail: msg };
  }
}

// ── chat ──────────────────────────────────────────────────────────────────────

export async function postChat(
  base: string,
  message: string,
  sessionId: string | null,
): Promise<ChatTurnResult> {
  const r = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  const data = await r.json() as ChatTurnResult;
  if (!r.ok) throw new Error((data as unknown as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data;
}

// ── model stats ───────────────────────────────────────────────────────────────

export async function fetchModelStats(base: string): Promise<ModelStats | null> {
  try {
    const r = await fetch(`${base}/model/stats`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<ModelStats>;
  } catch { return null; }
}

export async function fetchTrainStatus(base: string): Promise<PipelineStats | null> {
  try {
    const r = await fetch(`${base}/train/status`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<PipelineStats>;
  } catch { return null; }
}

export async function fetchVocab(base: string, limit = 200): Promise<VocabResponse | null> {
  try {
    const r = await fetch(`${base}/model/vocab?limit=${limit}`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<VocabResponse>;
  } catch { return null; }
}

// ── training ──────────────────────────────────────────────────────────────────

export async function trainText(base: string, text: string): Promise<{
  pairs_trained: number;
  vocab_size: number;
  total_tokens: number;
} | null> {
  try {
    const r = await fetch(`${base}/train/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

export async function trainTopic(base: string, topic: string, maxPages = 5): Promise<{
  status: string;
  topic: string;
  message: string;
} | null> {
  try {
    const r = await fetch(`${base}/train/topic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, max_pages: maxPages }),
    });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

// ── generation ────────────────────────────────────────────────────────────────

export async function generate(
  base: string,
  seed: string,
  maxTokens = 60,
  temperature = 0.8,
): Promise<GenerateResult | null> {
  try {
    const r = await fetch(`${base}/model/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed, max_tokens: maxTokens, temperature }),
    });
    if (!r.ok) return null;
    return r.json() as Promise<GenerateResult>;
  } catch { return null; }
}

// ── analogy + similarity ──────────────────────────────────────────────────────

export async function solveAnalogy(
  base: string,
  a: string,
  b: string,
  c: string,
): Promise<AnalogyResult | null> {
  try {
    const r = await fetch(`${base}/analogy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b, c }),
    });
    if (!r.ok) return null;
    return r.json() as Promise<AnalogyResult>;
  } catch { return null; }
}

export async function findSimilar(
  base: string,
  word: string,
  topK = 8,
): Promise<SimilarResult | null> {
  try {
    const r = await fetch(`${base}/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word, top_k: topK }),
    });
    if (!r.ok) return null;
    return r.json() as Promise<SimilarResult>;
  } catch { return null; }
}

// ── memory ────────────────────────────────────────────────────────────────────

export async function fetchMemory(base: string, limit = 30): Promise<MemoryResponse | null> {
  try {
    const r = await fetch(`${base}/memory?limit=${limit}`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<MemoryResponse>;
  } catch { return null; }
}

// ── heartbeat / research ──────────────────────────────────────────────────────

export async function fetchHeartbeatStatus(base: string): Promise<HeartbeatStatus | null> {
  try {
    const r = await fetch(`${base}/heartbeat/status`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<HeartbeatStatus>;
  } catch { return null; }
}

export async function setHeartbeatTopics(base: string, topics: string[]): Promise<boolean> {
  try {
    const r = await fetch(`${base}/heartbeat/topics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topics }),
    });
    return r.ok;
  } catch { return false; }
}

export async function triggerHeartbeat(base: string): Promise<boolean> {
  try {
    const r = await fetch(`${base}/heartbeat/run`, { method: "POST" });
    return r.ok;
  } catch { return false; }
}

export async function researchTopic(base: string, topic: string): Promise<{
  answer: string;
  mode: string;
  details: Record<string, unknown>;
} | null> {
  try {
    const r = await fetch(`${base}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, max_pages: 4 }),
    });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}
