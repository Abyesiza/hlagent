"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchBlueprint,
  fetchCodebaseSnapshot,
  fetchHealth,
  fetchImprovements,
  fetchSicaSummary,
  getAgentBaseUrl,
  getAgentJob,
  getOrCreateSessionId,
  postChat,
  refreshCodebase,
  requestFullStackImprovement,
  requestImprovement,
  resetSessionId,
  startAgentJob,
} from "@/lib/agent-api";
import type {
  BlueprintSnapshot,
  ChatMessage,
  ChatTurnResult,
  FullStackImproveResult,
  ImprovementHistory,
  ImproveResult,
} from "@/lib/types";

// ─── storage ─────────────────────────────────────────────────────────────────

const STORAGE_MESSAGES = "hlagent-messages";
const STORAGE_API = "hlagent-api-base";
const STORAGE_AUTO = "hlagent-auto-improve";

const EMPTY_TURN: ChatTurnResult = {
  route: "error", intent: "", answer: "", sympy: null,
  hdc_similarity: null, hdc_matched_task: null, context_snippet: "",
  grounded: false, session_id: null, metadata: {},
};

function ls<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try { const v = localStorage.getItem(key); return v != null ? (JSON.parse(v) as T) : fallback; }
  catch { return fallback; }
}
function lsSet(key: string, v: unknown) {
  try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* quota */ }
}

function loadMessages(): ChatMessage[] {
  const parsed = ls<ChatMessage[]>(STORAGE_MESSAGES, []);
  return Array.isArray(parsed) ? parsed : [];
}

function saveMessages(msgs: ChatMessage[]) { lsSet(STORAGE_MESSAGES, msgs.slice(-200)); }

function fmtTime(at: number) {
  return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function fmtTs(iso: string) {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

// ─── markdown renderer ────────────────────────────────────────────────────────

type Segment = { type: "code"; lang: string; text: string } | { type: "text"; text: string };

function parseSegments(raw: string): Segment[] {
  const segs: Segment[] = [];
  const re = /```(\w*)\n?([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > last) segs.push({ type: "text", text: raw.slice(last, m.index) });
    segs.push({ type: "code", lang: m[1] || "text", text: m[2].trimEnd() });
    last = m.index + m[0].length;
  }
  if (last < raw.length) segs.push({ type: "text", text: raw.slice(last) });
  return segs;
}

function renderInline(text: string): React.ReactNode {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (p.startsWith("*") && p.endsWith("*"))
      return <em key={i}>{p.slice(1, -1)}</em>;
    if (p.startsWith("`") && p.endsWith("`"))
      return <code key={i} className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[11px] text-pink-600 dark:bg-zinc-800 dark:text-pink-400">{p.slice(1, -1)}</code>;
    return p;
  });
}

function renderTextBlock(text: string) {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let listBuf: string[] = [];

  function flushList() {
    if (listBuf.length) {
      nodes.push(
        <ul key={`ul-${nodes.length}`} className="my-1 ml-3 list-none space-y-0.5">
          {listBuf.map((l, i) => (
            <li key={i} className="flex gap-2">
              <span className="mt-0.5 text-zinc-400">•</span>
              <span>{renderInline(l)}</span>
            </li>
          ))}
        </ul>
      );
      listBuf = [];
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) {
      flushList();
      if (nodes.length > 0) nodes.push(<div key={`sp-${i}`} className="h-1.5" />);
      continue;
    }
    if (line.match(/^#{1,3}\s/)) {
      flushList();
      const content = line.replace(/^#{1,3}\s/, "");
      nodes.push(<p key={i} className="mt-2 font-semibold text-zinc-800 dark:text-zinc-100">{renderInline(content)}</p>);
      continue;
    }
    if (line.match(/^[-*]\s/)) {
      listBuf.push(line.slice(2));
      continue;
    }
    if (line.match(/^\d+\.\s/)) {
      listBuf.push(line.replace(/^\d+\.\s/, ""));
      continue;
    }
    flushList();
    nodes.push(<p key={i} className="leading-relaxed">{renderInline(line)}</p>);
  }
  flushList();
  return nodes;
}

function MarkdownMessage({ text }: { text: string }) {
  const segs = parseSegments(text);
  return (
    <div className="text-sm">
      {segs.map((seg, i) =>
        seg.type === "code" ? (
          <pre key={i} className="my-2 overflow-auto rounded-lg bg-zinc-950 px-4 py-3 font-mono text-[11px] leading-relaxed text-emerald-300 dark:bg-zinc-900">
            {seg.text}
          </pre>
        ) : (
          <div key={i}>{renderTextBlock(seg.text)}</div>
        )
      )}
    </div>
  );
}

// ─── improvement card (inline in chat) ───────────────────────────────────────

function InlineImproveCard({ r }: { r: ImproveResult }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`mt-3 rounded-lg border text-xs ${
      r.ok
        ? "border-emerald-300 bg-emerald-50 dark:border-emerald-700 dark:bg-emerald-950/40"
        : "border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-950/40"
    }`}>
      <div className="flex items-center gap-2 px-3 py-2">
        <span className={`font-bold ${r.ok ? "text-emerald-600" : "text-red-500"}`}>
          {r.ok ? "✓ Code changed" : "✗ Improve failed"}
        </span>
        <code className="rounded bg-white/60 px-1.5 py-0.5 font-mono text-[10px] text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300">
          {r.target_file}
        </code>
        {r.committed && (
          <span className="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-[10px] text-blue-600 dark:bg-blue-900 dark:text-blue-300">
            {r.commit_hash?.slice(0, 8) ?? "committed"}
          </span>
        )}
        {r.ast_ok && (
          <span className="text-[10px] text-zinc-400">AST ✓</span>
        )}
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="ml-auto text-[10px] text-zinc-400 underline"
        >
          {open ? "hide diff" : "show diff"}
        </button>
      </div>
      {r.error && (
        <p className="border-t border-red-200 px-3 py-1.5 text-red-600 dark:border-red-800 dark:text-red-300">
          {r.error}
        </p>
      )}
      {open && r.new_code && (
        <pre className="max-h-64 overflow-auto border-t border-emerald-200 bg-zinc-950 px-4 py-3 font-mono text-[10px] leading-relaxed text-emerald-300 dark:border-emerald-800">
          {r.new_code}
        </pre>
      )}
    </div>
  );
}

// ─── full-stack improvement card (inline in chat) ────────────────────────────

function InlineFullStackCard({ r }: { r: FullStackImproveResult }) {
  const layers: { label: string; result: ImproveResult | null | undefined }[] = [
    { label: "Backend", result: r.backend },
    { label: "API client", result: r.frontend_api },
    { label: "UI component", result: r.frontend_ui },
  ];
  return (
    <div className={`mt-3 rounded-lg border text-xs ${
      r.ok
        ? "border-violet-300 bg-violet-50 dark:border-violet-700 dark:bg-violet-950/40"
        : "border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-950/40"
    }`}>
      <div className="flex items-center gap-2 border-b border-violet-200 px-3 py-2 dark:border-violet-800">
        <span className={`font-bold ${r.ok ? "text-violet-600 dark:text-violet-400" : "text-red-500"}`}>
          {r.ok ? "⚡ Full-stack applied" : "✗ Full-stack failed"}
        </span>
        <span className="ml-auto text-[10px] text-zinc-400">{new Date(r.timestamp).toLocaleTimeString()}</span>
      </div>
      <div className="divide-y divide-violet-100 dark:divide-violet-900">
        {layers.map(({ label, result }) => {
          if (!result) return null;
          return (
            <div key={label} className="flex items-start gap-2 px-3 py-2">
              <span className={`shrink-0 w-20 font-semibold ${
                result.ok ? "text-emerald-600 dark:text-emerald-400" : "text-red-500"
              }`}>
                {result.ok ? "✓" : "✗"} {label}
              </span>
              <code className="rounded bg-white/60 px-1.5 py-0.5 font-mono text-[10px] text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300">
                {result.target_file}
              </code>
              {result.committed && (
                <span className="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-[10px] text-blue-600 dark:bg-blue-900 dark:text-blue-300">
                  {result.commit_hash?.slice(0, 8) ?? "committed"}
                </span>
              )}
              {result.error && (
                <span className="text-[10px] text-red-500">{result.error}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}


// ─── sub-components ───────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: "done" | "partial" | "todo" }) {
  const cls =
    status === "done"
      ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-300"
      : status === "partial"
      ? "bg-amber-100 text-amber-700 dark:bg-amber-900/60 dark:text-amber-300"
      : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400";
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      {status}
    </span>
  );
}

function Toggle({ on, onToggle, label }: { on: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={onToggle}
      className="flex items-center gap-2 text-xs"
    >
      <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
        on ? "bg-violet-600" : "bg-zinc-300 dark:bg-zinc-600"
      }`}>
        <span className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
          on ? "translate-x-4" : "translate-x-1"
        }`} />
      </span>
      <span className={on ? "text-violet-700 dark:text-violet-400" : "text-zinc-500"}>{label}</span>
    </button>
  );
}

// ─── suggestion chips ─────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "Tell me about your codebase and architecture",
  "What are you currently able to do?",
  "What is happening in AI research today?",
  "Add a /api/v1/sessions endpoint listing all saved sessions",
  "Improve the HDC memory to include a retrieval count",
  "What is 2 + 2 using SymPy?",
];

// ─── types ────────────────────────────────────────────────────────────────────

type Tab = "chat" | "codebase" | "improve" | "status";
type ThinkingStage = "thinking" | "searching" | "writing code" | "applying";

// ─── main component ───────────────────────────────────────────────────────────

export default function AgentTester() {
  const [baseUrl, setBaseUrl] = useState("");
  const [tab, setTab] = useState<Tab>("chat");

  // health
  const [health, setHealth] = useState<{ status: "ok" | "down" | "checking"; gemini: boolean }>({
    status: "checking", gemini: false,
  });

  // chat
  const [sessionId, setSessionId] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [thinkingStage, setThinkingStage] = useState<ThinkingStage>("thinking");
  const [mode, setMode] = useState<"sync" | "async">("sync");
  const [autoImprove, setAutoImprove] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // codebase
  const [codebase, setCodebase] = useState("");
  const [codebaseLoading, setCodebaseLoading] = useState(false);

  // improve tab
  const [instruction, setInstruction] = useState("");
  const [targetFile, setTargetFile] = useState("");
  const [fullStack, setFullStack] = useState(false);
  const [improving, setImproving] = useState(false);
  const [improveResult, setImproveResult] = useState<ImproveResult | null>(null);
  const [fullStackResult, setFullStackResult] = useState<FullStackImproveResult | null>(null);
  const [improveHistory, setImproveHistory] = useState<ImprovementHistory>({ entries: [] });

  // status
  const [blueprint, setBlueprint] = useState<BlueprintSnapshot | null>(null);
  const [sica, setSica] = useState("");

  // ── init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    setBaseUrl(getAgentBaseUrl());
    setSessionId(getOrCreateSessionId());
    setMessages(loadMessages());
    setAutoImprove(ls(STORAGE_AUTO, false));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  // ── ping ──────────────────────────────────────────────────────────────────
  const ping = useCallback(async () => {
    const b = baseUrl || getAgentBaseUrl();
    setHealth(h => ({ ...h, status: "checking" }));
    const h = await fetchHealth(b);
    setHealth({ status: h?.status === "ok" ? "ok" : "down", gemini: h?.gemini_configured ?? false });
  }, [baseUrl]);

  useEffect(() => {
    if (!baseUrl) return;
    void ping();
    const t = setInterval(() => void ping(), 20_000);
    return () => clearInterval(t);
  }, [baseUrl, ping]);

  // ── tab loaders ───────────────────────────────────────────────────────────
  const loadCodebase = useCallback(async () => {
    setCodebaseLoading(true);
    const txt = await fetchCodebaseSnapshot(baseUrl || getAgentBaseUrl());
    setCodebase(txt);
    setCodebaseLoading(false);
  }, [baseUrl]);

  const loadStatus = useCallback(async () => {
    const b = baseUrl || getAgentBaseUrl();
    const [bp, sc] = await Promise.all([fetchBlueprint(b), fetchSicaSummary(b)]);
    setBlueprint(bp);
    setSica(sc);
  }, [baseUrl]);

  const loadImprovements = useCallback(async () => {
    const hist = await fetchImprovements(baseUrl || getAgentBaseUrl());
    setImproveHistory(hist);
  }, [baseUrl]);

  useEffect(() => {
    if (!baseUrl) return;
    if (tab === "codebase") void loadCodebase();
    if (tab === "status") void loadStatus();
    if (tab === "improve") void loadImprovements();
  }, [tab, baseUrl, loadCodebase, loadStatus, loadImprovements]);

  // ── helpers ───────────────────────────────────────────────────────────────
  const persistBase = (next: string) => {
    const v = next.trim().replace(/\/$/, "");
    setBaseUrl(v);
    try {
      if (v && v !== "http://127.0.0.1:8000") localStorage.setItem(STORAGE_API, v);
      else localStorage.removeItem(STORAGE_API);
    } catch { /* ignore */ }
  };

  const toggleAutoImprove = () => {
    setAutoImprove(prev => { lsSet(STORAGE_AUTO, !prev); return !prev; });
  };

  const newSession = () => {
    const id = resetSessionId();
    setSessionId(id);
    setMessages([]);
    try { localStorage.removeItem(STORAGE_MESSAGES); } catch { /* ignore */ }
  };

  const appendAssistant = (turn: ChatTurnResult, err?: string) => {
    const msg: ChatMessage = {
      id: crypto.randomUUID(), role: "assistant",
      content: err ?? turn.answer, turn, error: err, at: Date.now(),
    };
    setMessages(prev => { const n = [...prev, msg]; saveMessages(n); return n; });
  };

  const appendUser = (text: string) => {
    const msg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text, at: Date.now() };
    setMessages(prev => { const n = [...prev, msg]; saveMessages(n); return n; });
  };

  // ── chat send ─────────────────────────────────────────────────────────────
  const guessStage = (text: string): ThinkingStage => {
    if (/\b(current|today|news|latest|2026)\b/i.test(text)) return "searching";
    if (/\b(add|implement|create|fix|refactor|update|improve)\b/i.test(text) && autoImprove) return "writing code";
    return "thinking";
  };

  const sendSync = async (prefill?: string) => {
    const text = (prefill ?? input).trim();
    if (!text || chatLoading) return;
    const b = baseUrl || getAgentBaseUrl();
    appendUser(text);
    if (!prefill) setInput("");
    setChatLoading(true);
    setThinkingStage(guessStage(text));
    try {
      const turn = await postChat(b, text, sessionId || null, autoImprove);
      appendAssistant(turn);
    } catch (e) {
      appendAssistant({ ...EMPTY_TURN }, e instanceof Error ? e.message : "Request failed");
    } finally { setChatLoading(false); }
  };

  const sendAsync = async (prefill?: string) => {
    const text = (prefill ?? input).trim();
    if (!text || chatLoading) return;
    const b = baseUrl || getAgentBaseUrl();
    appendUser(`[async] ${text}`);
    if (!prefill) setInput("");
    setChatLoading(true);
    setThinkingStage(guessStage(text));
    try {
      const { job_id } = await startAgentJob(b, text, sessionId || null, autoImprove);
      let result: ChatTurnResult | null = null;
      for (let i = 0; i < 300; i++) {
        const job = await getAgentJob(b, job_id);
        result = job.result;
        if (result) break;
        if (i > 10 && i % 5 === 0) setThinkingStage(prev =>
          prev === "thinking" ? "searching" : prev === "searching" ? "writing code" : "thinking"
        );
        await new Promise(r => setTimeout(r, 100));
      }
      if (result) appendAssistant(result);
      else appendAssistant({ ...EMPTY_TURN, answer: "Job did not complete in time." });
    } catch (e) {
      appendAssistant({ ...EMPTY_TURN }, e instanceof Error ? e.message : "Async job failed");
    } finally { setChatLoading(false); }
  };

  const send = (prefill?: string) =>
    void (mode === "sync" ? sendSync(prefill) : sendAsync(prefill));

  // ── improve submit ────────────────────────────────────────────────────────
  const submitImprovement = async () => {
    const instr = instruction.trim();
    if (!instr || improving) return;
    const b = baseUrl || getAgentBaseUrl();
    setImproving(true);
    setImproveResult(null);
    setFullStackResult(null);
    try {
      if (fullStack) {
        const r = await requestFullStackImprovement(b, instr, targetFile.trim() || undefined);
        setFullStackResult(r);
        void loadImprovements();
      } else {
        const r = await requestImprovement(b, instr, targetFile.trim() || undefined);
        setImproveResult(r);
        void loadImprovements();
      }
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Request failed";
      if (fullStack) {
        setFullStackResult({
          ok: false, instruction: instr,
          backend: {
            ok: false, target_file: "?", instruction: instr,
            old_code: "", new_code: "", ast_ok: false, committed: false,
            commit_hash: null, error: errMsg, timestamp: new Date().toISOString(),
          },
          frontend_api: null, frontend_ui: null, timestamp: new Date().toISOString(),
        });
      } else {
        setImproveResult({
          ok: false, target_file: targetFile || "?", instruction: instr,
          old_code: "", new_code: "", ast_ok: false, committed: false,
          commit_hash: null, error: errMsg, timestamp: new Date().toISOString(),
        });
      }
    } finally { setImproving(false); }
  };

  const gaps = useMemo(() => blueprint?.items.filter(i => i.status !== "done") ?? [], [blueprint]);

  // ── tab config ────────────────────────────────────────────────────────────
  const TABS: { id: Tab; label: string; icon: string; badge?: number }[] = [
    { id: "chat",     label: "Chat",     icon: "💬" },
    { id: "improve",  label: "Improve",  icon: "⚡", badge: improveHistory.entries.length || undefined },
    { id: "codebase", label: "Codebase", icon: "🗂" },
    { id: "status",   label: "Status",   icon: "📊", badge: gaps.length || undefined },
  ];

  // ─── render ───────────────────────────────────────────────────────────────
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 lg:flex-row">
      {/* ── sidebar ── */}
      <aside className="flex w-full shrink-0 flex-col gap-5 rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900 lg:w-60">
        {/* connection */}
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">API</p>
          <input
            className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-2.5 py-1.5 font-mono text-[11px] text-zinc-800 focus:outline-none focus:ring-1 focus:ring-violet-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
            value={baseUrl}
            onChange={e => persistBase(e.target.value)}
            placeholder="http://127.0.0.1:8000"
            spellCheck={false}
          />
        </div>

        {/* health pill */}
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${
            health.status === "ok" ? "bg-emerald-500"
            : health.status === "down" ? "bg-red-500"
            : "animate-pulse bg-amber-400"
          }`} />
          <span className="text-xs text-zinc-500">
            {health.status === "ok" ? "Online" : health.status === "down" ? "Offline" : "Checking…"}
          </span>
          <span className={`ml-auto rounded px-1.5 py-0.5 text-[10px] font-semibold ${
            health.gemini
              ? "bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300"
              : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800"
          }`}>
            Gemini {health.gemini ? "✓" : "—"}
          </span>
          <button type="button" onClick={() => void ping()}
            className="text-[11px] text-zinc-400 hover:text-zinc-600">↺</button>
        </div>

        {/* session */}
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Session</p>
          <p className="break-all font-mono text-[10px] text-zinc-400">{sessionId.slice(0, 20)}…</p>
          <p className="mt-0.5 text-[10px] text-zinc-400">{messages.length} messages · persistent</p>
          <button type="button" onClick={newSession}
            className="mt-2 w-full rounded-lg border border-zinc-200 py-1.5 text-[11px] text-zinc-600 transition hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800">
            + New session
          </button>
        </div>

        {/* auto-improve toggle */}
        <div className="rounded-lg border border-violet-200 bg-violet-50 p-3 dark:border-violet-800 dark:bg-violet-950/30">
          <Toggle on={autoImprove} onToggle={toggleAutoImprove} label="Auto-improve" />
          <p className="mt-1.5 text-[10px] leading-relaxed text-zinc-500">
            When ON, code-change requests in chat automatically run the self-improvement pipeline.
          </p>
        </div>

        {/* blueprint mini */}
        {blueprint && (
          <div>
            <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Blueprint</p>
            <div className="flex gap-3 text-[11px]">
              <span className="text-emerald-600">{blueprint.items.filter(i => i.status === "done").length} ✓</span>
              <span className="text-amber-500">{blueprint.items.filter(i => i.status === "partial").length} ~</span>
              <span className="text-zinc-400">{gaps.length} todo</span>
            </div>
          </div>
        )}

        {improveHistory.entries.length > 0 && (
          <div>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Improvements</p>
            <p className="text-[11px] text-zinc-500">
              {improveHistory.entries.filter(e => e.ok).length} applied ·{" "}
              {improveHistory.entries.filter(e => !e.ok).length} failed
            </p>
          </div>
        )}
      </aside>

      {/* ── main ── */}
      <main className="flex min-h-0 flex-1 flex-col">
        {/* tab bar */}
        <div className="mb-4 flex gap-0.5 rounded-xl border border-zinc-200 bg-white p-1 dark:border-zinc-800 dark:bg-zinc-900">
          {TABS.map(t => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`relative flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-all ${
                tab === t.id
                  ? "bg-zinc-900 text-white shadow-sm dark:bg-zinc-100 dark:text-zinc-900"
                  : "text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              <span>{t.icon}</span>
              <span>{t.label}</span>
              {t.badge != null && (
                <span className="absolute right-1.5 top-1 rounded-full bg-violet-500 px-1 text-[8px] font-bold text-white">
                  {t.badge}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* ──────── CHAT TAB ──────── */}
        {tab === "chat" && (
          <div className="flex flex-1 flex-col gap-2">
            {/* toolbar */}
            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-1 rounded-md border border-zinc-200 p-0.5 text-xs dark:border-zinc-700">
                {(["sync", "async"] as const).map(m => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={`rounded px-2 py-1 transition-colors ${
                      mode === m
                        ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                        : "text-zinc-500"
                    }`}
                  >
                    {m === "sync" ? "Sync" : "Async"}
                  </button>
                ))}
              </div>
              <span className="text-[10px] text-zinc-400">
                {mode === "sync" ? "POST /chat — waits for response" : "POST /agent/start — polls job"}
              </span>
              {autoImprove && (
                <span className="ml-auto rounded border border-violet-300 bg-violet-50 px-2 py-0.5 text-[10px] font-medium text-violet-600 dark:border-violet-700 dark:bg-violet-950/40 dark:text-violet-400">
                  ⚡ auto-improve ON
                </span>
              )}
            </div>

            {/* messages */}
            <div className="flex flex-1 flex-col rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
              <div
                className="flex-1 overflow-y-auto px-4 py-4"
                style={{ maxHeight: "min(62vh, 580px)" }}
              >
                {messages.length === 0 && (
                  <div className="py-6">
                    <p className="mb-5 text-sm text-zinc-400">
                      The agent knows its own codebase, remembers this session, searches the web for current
                      events, and — with <strong>Auto-improve ON</strong> — can modify its own code directly from chat.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {SUGGESTIONS.map(s => (
                        <button
                          key={s}
                          type="button"
                          onClick={() => send(s)}
                          className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1.5 text-xs text-zinc-600 transition hover:border-violet-300 hover:bg-violet-50 hover:text-violet-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-violet-950/40"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map(m => {
                  const improveData = m.turn?.metadata?.improve_result as ImproveResult | undefined;
                  const fullStackData = m.turn?.metadata?.fullstack_result as FullStackImproveResult | undefined;
                  return (
                    <div
                      key={m.id}
                      className={`mb-4 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div className={`max-w-[88%] ${m.role === "user" ? "items-end" : "items-start"} flex flex-col`}>
                        <div className={`rounded-2xl px-4 py-3 ${
                          m.role === "user"
                            ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                            : "border border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900"
                        }`}>
                          {m.role === "assistant" && !m.error
                            ? <MarkdownMessage text={m.content} />
                            : <p className="text-sm leading-relaxed">{m.content}</p>
                          }
                          {m.error && (
                            <p className="mt-1 text-xs text-red-500">{m.error}</p>
                          )}
                          {m.turn && m.role === "assistant" && !m.error && (
                            <div className="mt-2.5 flex flex-wrap items-center gap-1.5 border-t border-zinc-100 pt-2 font-mono text-[9px] text-zinc-400 dark:border-zinc-700">
                              <span className={`rounded px-1 py-0.5 ${
                                m.turn.intent === "improve"
                                  ? "bg-violet-100 text-violet-600 dark:bg-violet-900/40 dark:text-violet-400"
                                  : "bg-zinc-100 dark:bg-zinc-800"
                              }`}>
                                {m.turn.intent || m.turn.route}
                              </span>
                              {m.turn.hdc_similarity != null && (
                                <span>hdc={m.turn.hdc_similarity.toFixed(3)}</span>
                              )}
                              {m.turn.grounded && (
                                <span className="rounded bg-blue-100 px-1.5 py-0.5 text-blue-600 dark:bg-blue-900/40 dark:text-blue-300">
                                  🔍 web search
                                </span>
                              )}
                              <span className="ml-auto">{fmtTime(m.at)}</span>
                            </div>
                          )}
                        </div>
                        {/* inline code-change cards */}
                        {fullStackData && <InlineFullStackCard r={fullStackData} />}
                        {!fullStackData && improveData && <InlineImproveCard r={improveData} />}
                      </div>
                    </div>
                  );
                })}

                {chatLoading && (
                  <div className="mb-4 flex justify-start">
                    <div className="flex items-center gap-2.5 rounded-2xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-700 dark:bg-zinc-900">
                      <span className="flex gap-1">
                        {[0, 1, 2].map(i => (
                          <span
                            key={i}
                            className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce"
                            style={{ animationDelay: `${i * 150}ms` }}
                          />
                        ))}
                      </span>
                      <span className="text-xs text-zinc-400">{thinkingStage}…</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* input bar */}
              <div className="border-t border-zinc-100 p-3 dark:border-zinc-800">
                <textarea
                  className="w-full resize-none rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-2.5 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-violet-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-violet-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-violet-600"
                  rows={3}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  placeholder={
                    autoImprove
                      ? "Chat or say 'add X to the API' to modify code automatically…"
                      : "Ask anything — current events, maths, your codebase…"
                  }
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
                  }}
                />
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    disabled={chatLoading || !input.trim()}
                    onClick={() => send()}
                    className="rounded-xl bg-zinc-900 px-5 py-2 text-sm font-semibold text-white transition hover:bg-zinc-700 disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
                  >
                    {chatLoading ? "…" : "Send"}
                  </button>
                  <span className="text-[10px] text-zinc-400">⏎ send · ⇧⏎ newline</span>
                  {messages.length > 0 && (
                    <button
                      type="button"
                      onClick={newSession}
                      className="ml-auto text-[10px] text-zinc-400 underline"
                    >
                      clear session
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ──────── IMPROVE TAB ──────── */}
        {tab === "improve" && (
          <div className="flex flex-col gap-5">
            <div className="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="mb-4 flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-xl dark:bg-violet-900/40">⚡</div>
                <div>
                  <h3 className="font-semibold text-zinc-800 dark:text-zinc-100">Direct code improvement</h3>
                  <p className="mt-0.5 text-xs text-zinc-500">
                    Describe the change. The agent identifies the file, generates the improved version, runs
                    an AST check, writes it, and commits. You can also trigger this from Chat with Auto-improve ON.
                  </p>
                </div>
              </div>

              <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Instruction
              </label>
              <textarea
                className="mb-3 w-full resize-none rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-2.5 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
                rows={4}
                value={instruction}
                onChange={e => setInstruction(e.target.value)}
                placeholder={"Add rate limiting to the /chat endpoint\nImprove the HDC memory to persist across restarts\nAdd a /api/v1/sessions endpoint listing saved sessions"}
              />

              <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Target file <span className="font-normal text-zinc-400">(leave blank — auto-detected)</span>
              </label>
              <input
                className="mb-4 w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-2.5 font-mono text-xs text-zinc-900 placeholder:text-zinc-400 focus:border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
                value={targetFile}
                onChange={e => setTargetFile(e.target.value)}
                placeholder="super_agent/app/api/routes.py"
                spellCheck={false}
              />

              {/* full-stack toggle */}
              <div className="mb-4 flex items-start gap-3 rounded-xl border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-800/50">
                <button
                  type="button"
                  role="switch"
                  aria-checked={fullStack}
                  onClick={() => setFullStack(f => !f)}
                  className="mt-0.5 flex shrink-0 items-center"
                >
                  <span className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    fullStack ? "bg-violet-600" : "bg-zinc-300 dark:bg-zinc-600"
                  }`}>
                    <span className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
                      fullStack ? "translate-x-4" : "translate-x-1"
                    }`} />
                  </span>
                </button>
                <div>
                  <p className={`text-xs font-medium ${fullStack ? "text-violet-700 dark:text-violet-400" : "text-zinc-600 dark:text-zinc-400"}`}>
                    Full-stack mode {fullStack ? "(ON)" : "(OFF)"}
                  </p>
                  <p className="mt-0.5 text-[10px] leading-relaxed text-zinc-400">
                    When ON: also updates <code className="font-mono">agent-api.ts</code> and <code className="font-mono">AgentTester.tsx</code> to expose the new feature in this UI.
                  </p>
                </div>
              </div>

              <button
                type="button"
                disabled={improving || !instruction.trim()}
                onClick={() => void submitImprovement()}
                className="rounded-xl bg-violet-600 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-violet-700 disabled:opacity-40"
              >
                {improving ? (
                  <span className="flex items-center gap-2">
                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    {fullStack ? "Applying full-stack…" : "Applying…"}
                  </span>
                ) : fullStack ? "⚡ Apply (backend + frontend)" : "Apply improvement"}
              </button>
            </div>

            {improveResult && (
              <div className={`rounded-xl border p-5 ${
                improveResult.ok
                  ? "border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/30"
                  : "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/30"
              }`}>
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <span className={`text-sm font-bold ${improveResult.ok ? "text-emerald-600" : "text-red-500"}`}>
                    {improveResult.ok ? "✓ Applied" : "✗ Failed"}
                  </span>
                  <code className="rounded bg-white px-2 py-0.5 text-[11px] text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300">
                    {improveResult.target_file}
                  </code>
                  {improveResult.committed && (
                    <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-medium text-blue-600 dark:bg-blue-900/40 dark:text-blue-300">
                      commit {improveResult.commit_hash?.slice(0, 8) ?? ""}
                    </span>
                  )}
                  {improveResult.ast_ok && (
                    <span className="text-[10px] text-zinc-400">AST ✓</span>
                  )}
                  <span className="ml-auto text-[10px] text-zinc-400">{fmtTs(improveResult.timestamp)}</span>
                </div>
                {improveResult.error && (
                  <p className="mb-3 rounded-lg bg-red-100 px-3 py-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
                    {improveResult.error}
                  </p>
                )}
                {improveResult.new_code && (
                  <details className="mt-1">
                    <summary className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-700">
                      Show new code ({improveResult.new_code.split("\n").length} lines)
                    </summary>
                    <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-zinc-950 px-4 py-3 font-mono text-[10px] text-emerald-300">
                      {improveResult.new_code}
                    </pre>
                  </details>
                )}
              </div>
            )}

            {fullStackResult && (
              <div className={`rounded-xl border p-5 ${
                fullStackResult.ok
                  ? "border-violet-200 bg-violet-50 dark:border-violet-800 dark:bg-violet-950/30"
                  : "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/30"
              }`}>
                <div className="mb-4 flex items-center gap-2">
                  <span className={`text-sm font-bold ${fullStackResult.ok ? "text-violet-600 dark:text-violet-400" : "text-red-500"}`}>
                    {fullStackResult.ok ? "⚡ Full-stack applied" : "✗ Full-stack failed"}
                  </span>
                  <span className="ml-auto text-[10px] text-zinc-400">{fmtTs(fullStackResult.timestamp)}</span>
                </div>
                <div className="space-y-3">
                  {([
                    { key: "backend" as const, label: "Backend", icon: "🐍" },
                    { key: "frontend_api" as const, label: "API client (agent-api.ts)", icon: "🔌" },
                    { key: "frontend_ui" as const, label: "UI component (AgentTester.tsx)", icon: "🖥" },
                  ] as const).map(({ key, label, icon }) => {
                    const sub = fullStackResult[key];
                    if (!sub) return null;
                    return (
                      <div key={key} className={`rounded-lg border p-3 ${
                        sub.ok
                          ? "border-emerald-200 bg-white dark:border-emerald-800 dark:bg-zinc-900"
                          : "border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-950/20"
                      }`}>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-base">{icon}</span>
                          <span className={`text-xs font-semibold ${sub.ok ? "text-emerald-600" : "text-red-500"}`}>
                            {sub.ok ? "✓" : "✗"} {label}
                          </span>
                          <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 dark:bg-zinc-800">
                            {sub.target_file}
                          </code>
                          {sub.committed && (
                            <span className="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-[10px] text-blue-600 dark:bg-blue-900/40 dark:text-blue-300">
                              {sub.commit_hash?.slice(0, 8)}
                            </span>
                          )}
                          {sub.ast_ok && <span className="text-[10px] text-zinc-400">AST ✓</span>}
                        </div>
                        {sub.error && (
                          <p className="mt-2 text-[11px] text-red-600 dark:text-red-400">{sub.error}</p>
                        )}
                        {sub.new_code && sub.ok && (
                          <details className="mt-2">
                            <summary className="cursor-pointer text-[11px] text-zinc-400 hover:text-zinc-600">
                              Show diff ({sub.new_code.split("\n").length} lines)
                            </summary>
                            <pre className="mt-1.5 max-h-48 overflow-auto rounded bg-zinc-950 px-3 py-2 font-mono text-[10px] text-emerald-300">
                              {sub.new_code}
                            </pre>
                          </details>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {improveHistory.entries.length > 0 && (
              <div>
                <h3 className="mb-3 text-sm font-semibold text-zinc-700 dark:text-zinc-200">
                  History ({improveHistory.entries.length})
                </h3>
                <div className="space-y-2">
                  {improveHistory.entries.map((e, i) => (
                    <div key={i} className="rounded-xl border border-zinc-200 bg-white p-3.5 dark:border-zinc-800 dark:bg-zinc-900">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-bold ${e.ok ? "text-emerald-500" : "text-red-500"}`}>
                            {e.ok ? "✓" : "✗"}
                          </span>
                          <span className="text-xs text-zinc-700 dark:text-zinc-200">{e.instruction}</span>
                        </div>
                        <span className="shrink-0 text-[10px] text-zinc-400">{fmtTs(e.timestamp)}</span>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2">
                        <code className="font-mono text-[10px] text-zinc-400">{e.target_file}</code>
                        {e.committed && (
                          <span className="text-[10px] text-blue-500">commit {e.commit_hash?.slice(0, 8)}</span>
                        )}
                        {e.error && <span className="text-[10px] text-red-500">{e.error}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ──────── CODEBASE TAB ──────── */}
        {tab === "codebase" && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
              <span className="text-lg">🗂</span>
              <div>
                <p className="text-sm font-semibold text-zinc-700 dark:text-zinc-200">Live codebase snapshot</p>
                <p className="text-[10px] text-zinc-400">
                  Injected into every system instruction so the agent knows its own code.
                  {codebase && ` · ${(codebase.length / 1024).toFixed(1)} KB`}
                </p>
              </div>
              <button
                type="button"
                disabled={codebaseLoading}
                onClick={async () => {
                  setCodebaseLoading(true);
                  await refreshCodebase(baseUrl || getAgentBaseUrl());
                  await loadCodebase();
                }}
                className="ml-auto rounded-lg border border-zinc-200 px-3 py-1.5 text-xs text-zinc-600 transition hover:bg-zinc-50 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                {codebaseLoading ? "Scanning…" : "↺ Rescan"}
              </button>
            </div>
            <pre className="max-h-[72vh] overflow-auto rounded-xl border border-zinc-200 bg-zinc-950 px-5 py-4 font-mono text-[11px] leading-relaxed text-zinc-200 dark:border-zinc-800">
              {codebaseLoading ? "Scanning…" : codebase || "Click Rescan to generate CODEBASE.md."}
            </pre>
          </div>
        )}

        {/* ──────── STATUS TAB ──────── */}
        {tab === "status" && (
          <div className="flex flex-col gap-5">
            {/* blueprint */}
            <div className="rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-3 border-b border-zinc-100 px-4 py-3 dark:border-zinc-800">
                <span className="text-lg">📊</span>
                <p className="font-semibold text-zinc-700 dark:text-zinc-200">Blueprint</p>
                {blueprint && (
                  <div className="ml-2 flex gap-3 text-xs">
                    <span className="text-emerald-600">{blueprint.items.filter(i => i.status === "done").length} done</span>
                    <span className="text-amber-500">{blueprint.items.filter(i => i.status === "partial").length} partial</span>
                    <span className="text-zinc-400">{gaps.length} todo</span>
                  </div>
                )}
                <button type="button" onClick={() => void loadStatus()}
                  className="ml-auto rounded-lg border border-zinc-200 px-2.5 py-1 text-xs text-zinc-500 dark:border-zinc-700">
                  ↺
                </button>
              </div>
              {blueprint ? (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-zinc-100 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900/50">
                      <th className="px-4 py-2 text-left font-medium text-zinc-500">Feature</th>
                      <th className="px-4 py-2 text-left font-medium text-zinc-500">Status</th>
                      <th className="px-4 py-2 text-left font-medium text-zinc-500">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {blueprint.items.map(item => (
                      <tr key={item.id} className="border-b border-zinc-50 dark:border-zinc-800/50">
                        <td className="px-4 py-2.5 font-medium text-zinc-700 dark:text-zinc-200">{item.title}</td>
                        <td className="px-4 py-2.5"><StatusBadge status={item.status} /></td>
                        <td className="px-4 py-2.5 text-zinc-400">{item.notes}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="px-4 py-6 text-sm text-zinc-400">Click ↺ to load.</p>
              )}
            </div>

            {sica && (
              <div className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
                <p className="mb-2 font-semibold text-zinc-700 dark:text-zinc-200">SICA loop summary</p>
                <pre className="rounded-lg bg-zinc-50 p-3 font-mono text-[11px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                  {sica}
                </pre>
              </div>
            )}

            {/* endpoints */}
            <div className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <p className="mb-3 font-semibold text-zinc-700 dark:text-zinc-200">API Endpoints</p>
              <div className="grid grid-cols-1 gap-1.5 font-mono text-[11px] sm:grid-cols-2">
                {([
                  ["GET",  "/health"],
                  ["GET",  "/api/v1/status"],
                  ["GET",  "/api/v1/gaps"],
                  ["POST", "/api/v1/chat"],
                  ["POST", "/api/v1/agent/start"],
                  ["GET",  "/api/v1/agent/jobs/:id"],
                  ["POST", "/api/v1/sympy/run"],
                  ["POST", "/api/v1/route-intent"],
                  ["GET",  "/api/v1/codebase/snapshot"],
                  ["POST", "/api/v1/codebase/refresh"],
                  ["POST", "/api/v1/improve"],
                  ["POST", "/api/v1/improve/fullstack"],
                  ["GET",  "/api/v1/improve/history"],
                  ["POST", "/api/v1/research/trigger"],
                  ["GET",  "/api/v1/sica/summary"],
                ] as [string, string][]).map(([method, path]) => (
                  <div key={path} className="flex items-center gap-2 rounded-lg bg-zinc-50 px-3 py-1.5 dark:bg-zinc-800/60">
                    <span className={`shrink-0 rounded px-1 text-[9px] font-bold uppercase ${
                      method === "GET"
                        ? "bg-blue-100 text-blue-600 dark:bg-blue-900/40 dark:text-blue-300"
                        : "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-300"
                    }`}>{method}</span>
                    <span className="text-zinc-600 dark:text-zinc-300">{path}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
