import type { BlueprintSnapshot, ChatTurnResult, FullStackImproveResult, ImprovementHistory, ImproveResult, BlueprintGap } from "./types";

function defaultBase(): string {
  if (typeof process !== "undefined" && process.env.NEXT_PUBLIC_AGENT_API_URL) {
    return process.env.NEXT_PUBLIC_AGENT_API_URL.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
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

export async function fetchHealth(
  base: string,
): Promise<{ status: string; gemini_configured?: boolean } | null> {
  try {
    const r = await fetch(`${base}/health`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

export async function fetchBlueprint(base: string): Promise<BlueprintSnapshot | null> {
  try {
    const r = await fetch(`${base}/api/v1/status`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

export async function fetchNextGaps(
  base: string,
): Promise<{ version: string; gaps: BlueprintGap[] } | null> {
  try {
    const r = await fetch(`${base}/api/v1/gaps`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<{ version: string; gaps: BlueprintGap[] }>;
  } catch { return null; }
}

export async function postChat(
  base: string,
  message: string,
  sessionId: string | null,
  autoImprove = false,
): Promise<ChatTurnResult> {
  const r = await fetch(`${base}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, auto_improve: autoImprove }),
  });
  const data = (await r.json()) as ChatTurnResult;
  if (!r.ok) throw new Error((data as unknown as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data;
}

export async function startAgentJob(
  base: string,
  prompt: string,
  sessionId: string | null,
  autoImprove = false,
): Promise<{ job_id: string }> {
  const r = await fetch(`${base}/api/v1/agent/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, session_id: sessionId, auto_improve: autoImprove }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as { job_id: string };
}

export async function getAgentJob(
  base: string,
  jobId: string,
): Promise<{ result: ChatTurnResult | null; phase?: string; error?: string | null }> {
  const r = await fetch(`${base}/api/v1/agent/jobs/${jobId}`, { cache: "no-store" });
  const data = await r.json();
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return data as { result: ChatTurnResult | null; phase?: string; error?: string | null };
}

export async function fetchCodebaseSnapshot(base: string): Promise<string> {
  try {
    const r = await fetch(`${base}/api/v1/codebase/snapshot`, { cache: "no-store" });
    if (!r.ok) return "";
    const d = (await r.json()) as { content: string };
    return d.content ?? "";
  } catch { return ""; }
}

export async function refreshCodebase(base: string): Promise<{ bytes: number }> {
  const r = await fetch(`${base}/api/v1/codebase/refresh`, { method: "POST" });
  const d = await r.json();
  return d as { bytes: number };
}

export async function requestImprovement(
  base: string,
  instruction: string,
  targetFile?: string,
): Promise<ImproveResult> {
  const r = await fetch(`${base}/api/v1/improve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction, target_file: targetFile || null }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as ImproveResult;
}


export async function requestFullStackImprovement(
  base: string,
  instruction: string,
  targetFile?: string,
): Promise<FullStackImproveResult> {
  const r = await fetch(`${base}/api/v1/improve/fullstack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction, target_file: targetFile || null }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as FullStackImproveResult;
}

export async function fetchImprovements(base: string): Promise<ImprovementHistory> {
  try {
    const r = await fetch(`${base}/api/v1/improve/history`, { cache: "no-store" });
    if (!r.ok) return { entries: [] };
    return r.json() as Promise<ImprovementHistory>;
  } catch { return { entries: [] }; }
}

export async function fetchSicaSummary(base: string): Promise<string> {
  try {
    const r = await fetch(`${base}/api/v1/sica/summary`, { cache: "no-store" });
    if (!r.ok) return "";
    const d = (await r.json()) as { summary: string };
    return d.summary ?? "";
  } catch { return ""; }
}
