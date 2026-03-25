import type { BlueprintSnapshot, ChatTurnResult, FullStackImproveResult, HeartbeatStatus, ImprovementHistory, ImproveResult, BlueprintGap } from "./types";

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

export type HealthCheck = {
  ok: boolean;
  gemini_configured?: boolean;
  convex_configured?: boolean;
  email_notifications_ready?: boolean;
  /** Shown in UI when ok is false (wrong URL, CORS, HTTP error, etc.) */
  detail?: string;
};

/**
 * GET /health on the FastAPI agent. Always returns a result so the UI can explain failures.
 */
export async function fetchHealth(base: string): Promise<HealthCheck> {
  const url = (base || "").trim().replace(/\/$/, "");
  if (!url) {
    return {
      ok: false,
      detail:
        "No API base URL. Set NEXT_PUBLIC_AGENT_API_URL in Vercel (Production) and redeploy, or paste your API URL in the sidebar.",
    };
  }
  try {
    const r = await fetch(`${url}/health`, { cache: "no-store", mode: "cors" });
    if (!r.ok) {
      return {
        ok: false,
        detail: `HTTP ${r.status} from ${url}/health — check the API deployment URL.`,
      };
    }
    const data = (await r.json()) as {
      status?: string;
      gemini_configured?: boolean;
      convex_configured?: boolean;
    };
    const ok = data.status === "ok";
    return {
      ok,
      gemini_configured: data.gemini_configured,
      convex_configured: data.convex_configured,
      detail: ok
        ? undefined
        : `Unexpected health payload: ${JSON.stringify(data)}`,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : "fetch failed";
    return {
      ok: false,
      detail: `${msg}. Common fixes: wrong URL, CORS (redeploy API with SUPER_AGENT_CORS_ORIGINS or default *.vercel.app regex), or calling http from an https page.`,
    };
  }
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
  userLocation?: LocationData | null,
): Promise<ChatTurnResult> {
  const r = await fetch(`${base}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      auto_improve: autoImprove,
      user_location: userLocation ?? null,
    }),
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

export type SicaStatus = {
  benchmark: {
    score: number;
    passed: number;
    failed: number;
    errors: number;
    total: number;
    output?: string;
    timestamp?: string;
  };
  plan: {
    score: number;
    priority_dimension: string;
    gaps: { id: string; title: string; status: string; notes: string }[];
    todos: { id: string; task: string; priority: number; notes: string }[];
  };
  history: { score: number; passed: number; failed: number; timestamp: string }[];
  recent_commits: { hash: string; subject: string; date: string }[];
};

export async function fetchSicaStatus(base: string): Promise<SicaStatus | null> {
  try {
    const r = await fetch(`${base}/api/v1/sica/status`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<SicaStatus>;
  } catch { return null; }
}

export type SicaCycleResult = {
  status: string;
  gap?: { id: string; task: string };
  score_delta?: number;
  reverted?: boolean;
  message?: string;
};

export async function runSicaCycle(base: string, gapId?: string): Promise<{ job_id: string }> {
  const r = await fetch(`${base}/api/v1/sica/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gap_id: gapId ?? null }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as { job_id: string };
}

export type AgentJobSummary = {
  job_id: string;
  job_type: string;
  prompt: string;
  phase: string;
  error?: string | null;
  started_at: string;
  finished_at?: string | null;
  improve_result?: Record<string, unknown> | null;
  turn_route?: string | null;
};

export async function listJobs(base: string, limit = 20): Promise<AgentJobSummary[]> {
  try {
    const r = await fetch(`${base}/api/v1/jobs?limit=${limit}`, { cache: "no-store" });
    if (!r.ok) return [];
    const d = (await r.json()) as { jobs: AgentJobSummary[] };
    return d.jobs ?? [];
  } catch { return []; }
}

export async function getJob(base: string, jobId: string): Promise<AgentJobSummary | null> {
  try {
    const r = await fetch(`${base}/api/v1/jobs/${jobId}`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<AgentJobSummary>;
  } catch { return null; }
}

export type LocationData = {
  city: string;
  region: string;
  country: string;
  country_code: string;
  lat: number | null;
  lon: number | null;
  timezone: string;
  source: "browser" | "ip" | "manual" | "unknown";
};

/**
 * Try browser navigator.geolocation first (precise, requires permission).
 * Falls back to IP geolocation via the agent API.
 */
export async function detectLocation(base: string): Promise<LocationData | null> {
  // 1. Try browser geolocation (requires HTTPS or localhost)
  if (typeof window !== "undefined" && "geolocation" in navigator) {
    try {
      const pos = await new Promise<GeolocationPosition>((resolve, reject) =>
        navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 })
      );
      // Reverse geocode using ip-api with coords isn't available for free,
      // so use the agent's IP endpoint to fill city/country, then override coords
      const ipLoc = await fetchLocationFromServer(base);
      return {
        city: ipLoc?.city ?? "",
        region: ipLoc?.region ?? "",
        country: ipLoc?.country ?? "",
        country_code: ipLoc?.country_code ?? "",
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        timezone: ipLoc?.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
        source: "browser",
      };
    } catch {
      // Permission denied or unavailable — fall through to IP
    }
  }
  // 2. Fall back to IP geolocation
  return fetchLocationFromServer(base);
}

async function fetchLocationFromServer(base: string): Promise<LocationData | null> {
  try {
    const r = await fetch(`${base}/api/v1/location`, { cache: "no-store" });
    if (!r.ok) return null;
    const d = (await r.json()) as { ok: boolean; location?: LocationData };
    if (!d.ok || !d.location) return null;
    return { ...d.location, source: "ip" };
  } catch { return null; }
}

export async function runSandbox(base: string, code: string, timeout = 15): Promise<{
  returncode: number;
  stdout: string;
  stderr: string;
  backend: string;
  timed_out: boolean;
}> {
  const r = await fetch(`${base}/api/v1/sandbox/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, timeout }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data as { detail?: string }).detail ?? `HTTP ${r.status}`);
  return data as { returncode: number; stdout: string; stderr: string; backend: string; timed_out: boolean };
}

export async function fetchHeartbeatStatus(base: string): Promise<HeartbeatStatus | null> {
  try {
    const r = await fetch(`${base}/api/v1/heartbeat/status`, { cache: "no-store" });
    if (!r.ok) return null;
    return r.json() as Promise<HeartbeatStatus>;
  } catch { return null; }
}

export async function setHeartbeatTopics(base: string, topics: string[]): Promise<{ ok: boolean }> {
  try {
    const r = await fetch(`${base}/api/v1/heartbeat/topics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topics }),
    });
    if (!r.ok) return { ok: false };
    return { ok: true };
  } catch { return { ok: false }; }
}

export async function triggerResearch(base: string): Promise<{ status: string }> {
  try {
    const r = await fetch(`${base}/api/v1/research/trigger`, { method: "POST" });
    if (!r.ok) return { status: "error" };
    return r.json() as Promise<{ status: string }>;
  } catch { return { status: "error" }; }
}

export async function fetchResearchMemory(base: string): Promise<string> {
  try {
    const r = await fetch(`${base}/api/v1/memory/research`, { cache: "no-store" });
    if (!r.ok) return "";
    const d = (await r.json()) as { content: string };
    return d.content ?? "";
  } catch { return ""; }
}

export async function clearResearchMemory(base: string): Promise<{ status: string }> {
  try {
    const r = await fetch(`${base}/api/v1/memory/research`, { method: "DELETE" });
    if (!r.ok) return { status: "error" };
    return r.json() as Promise<{ status: string }>;
  } catch { return { status: "error" }; }
}
