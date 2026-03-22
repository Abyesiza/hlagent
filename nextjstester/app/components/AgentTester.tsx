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
  requestImprovement,
  resetSessionId,
  startAgentJob,
} from "@/lib/agent-api";
import type {
  BlueprintSnapshot,
  ChatMessage,
  ChatTurnResult,
  ImprovementHistory,
  ImproveResult,
} from "@/lib/types";

// ─── constants & helpers ────────────────────────────────────────────────────

const STORAGE_MESSAGES = "hlagent-tester-messages";
const STORAGE_API = "hlagent-api-base";

const EMPTY_TURN: ChatTurnResult = {
  route: "error", intent: "", answer: "", sympy: null,
  hdc_similarity: null, hdc_matched_task: null, context_snippet: "",
  grounded: false, session_id: null, metadata: {},
};

function loadMessages(): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_MESSAGES);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatMessage[];
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}

function saveMessages(messages: ChatMessage[]) {
  try { localStorage.setItem(STORAGE_MESSAGES, JSON.stringify(messages.slice(-120))); }
  catch { /* quota / private mode */ }
}

function fmtTs(iso: string) {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

type Tab = "chat" | "codebase" | "improve" | "status";

// ─── sub-components ──────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: "done" | "partial" | "todo" }) {
  const cls =
    status === "done"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
      : status === "partial"
      ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
      : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400";
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}>
      {status}
    </span>
  );
}

function CodeBlock({ code, lang = "python" }: { code: string; lang?: string }) {
  const [open, setOpen] = useState(false);
  const lines = code.split("\n").length;
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="text-[10px] text-blue-600 underline dark:text-blue-400"
      >
        {open ? "Hide" : "Show"} {lines} lines ({lang})
      </button>
      {open && (
        <pre className="mt-1 max-h-72 overflow-auto rounded bg-zinc-950 p-3 text-[11px] leading-relaxed text-emerald-300 dark:bg-zinc-900">
          {code}
        </pre>
      )}
    </div>
  );
}

// ─── main component ──────────────────────────────────────────────────────────

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
  const [mode, setMode] = useState<"sync" | "async">("sync");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // codebase
  const [codebase, setCodebase] = useState("");
  const [codebaseLoading, setCodebaseLoading] = useState(false);

  // improve
  const [instruction, setInstruction] = useState("");
  const [targetFile, setTargetFile] = useState("");
  const [improving, setImproving] = useState(false);
  const [improveResult, setImproveResult] = useState<ImproveResult | null>(null);
  const [improveHistory, setImproveHistory] = useState<ImprovementHistory>({ entries: [] });

  // status
  const [blueprint, setBlueprint] = useState<BlueprintSnapshot | null>(null);
  const [sica, setSica] = useState("");

  // ── init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    setBaseUrl(getAgentBaseUrl());
    setSessionId(getOrCreateSessionId());
    setMessages(loadMessages());
  }, []);

  // scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  // ── ping / refresh health ─────────────────────────────────────────────────
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

  // ── tab data loaders ──────────────────────────────────────────────────────
  const loadCodebase = useCallback(async () => {
    const b = baseUrl || getAgentBaseUrl();
    setCodebaseLoading(true);
    const txt = await fetchCodebaseSnapshot(b);
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
    const b = baseUrl || getAgentBaseUrl();
    const hist = await fetchImprovements(b);
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

  const newSession = () => {
    const id = resetSessionId();
    setSessionId(id);
    setMessages([]);
    try { localStorage.removeItem(STORAGE_MESSAGES); } catch { /* ignore */ }
  };

  const appendAssistant = (turn: ChatTurnResult, err?: string) => {
    const msg: ChatMessage = { id: crypto.randomUUID(), role: "assistant", content: err ?? turn.answer, turn, error: err, at: Date.now() };
    setMessages(prev => { const n = [...prev, msg]; saveMessages(n); return n; });
  };
  const appendUser = (text: string) => {
    const msg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text, at: Date.now() };
    setMessages(prev => { const n = [...prev, msg]; saveMessages(n); return n; });
  };

  // ── chat send ─────────────────────────────────────────────────────────────
  const sendSync = async () => {
    const text = input.trim();
    if (!text || chatLoading) return;
    const b = baseUrl || getAgentBaseUrl();
    appendUser(text);
    setInput("");
    setChatLoading(true);
    try {
      const turn = await postChat(b, text, sessionId || null);
      appendAssistant(turn);
    } catch (e) {
      appendAssistant({ ...EMPTY_TURN }, e instanceof Error ? e.message : "Request failed");
    } finally { setChatLoading(false); }
  };

  const sendAsync = async () => {
    const text = input.trim();
    if (!text || chatLoading) return;
    const b = baseUrl || getAgentBaseUrl();
    appendUser(`[async] ${text}`);
    setInput("");
    setChatLoading(true);
    try {
      const { job_id } = await startAgentJob(b, text, sessionId || null);
      let result: ChatTurnResult | null = null;
      for (let i = 0; i < 200; i++) {
        const job = await getAgentJob(b, job_id);
        result = job.result;
        if (result) break;
        await new Promise(r => setTimeout(r, 50));
      }
      if (result) appendAssistant(result);
      else appendAssistant({ ...EMPTY_TURN, answer: "Job did not complete in time." });
    } catch (e) {
      appendAssistant({ ...EMPTY_TURN }, e instanceof Error ? e.message : "Async job failed");
    } finally { setChatLoading(false); }
  };

  // ── improve submit ────────────────────────────────────────────────────────
  const submitImprovement = async () => {
    const instr = instruction.trim();
    if (!instr || improving) return;
    const b = baseUrl || getAgentBaseUrl();
    setImproving(true);
    setImproveResult(null);
    try {
      const r = await requestImprovement(b, instr, targetFile.trim() || undefined);
      setImproveResult(r);
      void loadImprovements();
    } catch (e) {
      setImproveResult({
        ok: false, target_file: targetFile || "?", instruction: instr,
        old_code: "", new_code: "", ast_ok: false, committed: false,
        commit_hash: null, error: e instanceof Error ? e.message : "Request failed",
        timestamp: new Date().toISOString(),
      });
    } finally { setImproving(false); }
  };

  const gaps = useMemo(() => blueprint?.items.filter(i => i.status !== "done") ?? [], [blueprint]);

  // ── tab bar ───────────────────────────────────────────────────────────────
  const TABS: { id: Tab; label: string; count?: number }[] = [
    { id: "chat", label: "Chat" },
    { id: "improve", label: "Improve" },
    { id: "codebase", label: "Codebase" },
    { id: "status", label: "Status", count: gaps.length || undefined },
  ];

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 md:flex-row">
      {/* ── sidebar ── */}
      <aside className="flex w-full shrink-0 flex-col gap-4 border-b border-zinc-200 pb-4 md:w-64 md:border-b-0 md:border-r md:pb-0 md:pr-4 dark:border-zinc-800">
        <div>
          <h2 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">API URL</h2>
          <input
            className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1.5 font-mono text-xs text-zinc-900 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
            value={baseUrl}
            onChange={e => persistBase(e.target.value)}
            placeholder="http://127.0.0.1:8000"
            spellCheck={false}
          />
        </div>

        {/* health */}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className={`inline-flex h-2 w-2 rounded-full ${
            health.status === "ok" ? "bg-emerald-500"
            : health.status === "down" ? "bg-red-500"
            : "bg-amber-400 animate-pulse"
          }`} />
          <span className="text-xs text-zinc-500">
            {health.status === "ok" ? "Online" : health.status === "down" ? "Offline" : "Checking…"}
          </span>
          {health.status === "ok" && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
              health.gemini
                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
                : "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300"
            }`}>
              Gemini {health.gemini ? "✓" : "✗"}
            </span>
          )}
          <button type="button" onClick={() => void ping()}
            className="ml-auto rounded border border-zinc-200 px-2 py-0.5 text-[10px] text-zinc-500 dark:border-zinc-700">
            ↺
          </button>
        </div>

        {/* session */}
        <div>
          <h2 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Session</h2>
          <p className="mt-1 break-all font-mono text-[10px] text-zinc-400">{sessionId || "—"}</p>
          <p className="mt-0.5 text-[10px] text-zinc-400">
            Saved across reloads · {messages.length} messages
          </p>
          <button type="button" onClick={newSession}
            className="mt-2 rounded border border-zinc-200 px-2 py-1 text-[11px] text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
            New session
          </button>
        </div>

        {/* quick blueprint summary */}
        {blueprint && (
          <div>
            <h2 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Blueprint</h2>
            <div className="mt-1.5 flex gap-2 text-xs">
              <span className="text-emerald-600">{blueprint.items.filter(i => i.status === "done").length} done</span>
              <span className="text-amber-600">{blueprint.items.filter(i => i.status === "partial").length} partial</span>
              <span className="text-zinc-400">{gaps.length} todo</span>
            </div>
          </div>
        )}

        {/* improve history count */}
        {improveHistory.entries.length > 0 && (
          <div>
            <h2 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Improvements</h2>
            <p className="mt-1 text-xs text-zinc-500">{improveHistory.entries.length} applied</p>
          </div>
        )}
      </aside>

      {/* ── main area ── */}
      <main className="flex min-h-0 flex-1 flex-col">
        {/* tab bar */}
        <div className="mb-4 flex gap-1 border-b border-zinc-200 pb-0 dark:border-zinc-800">
          {TABS.map(t => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`relative px-4 py-2 text-sm font-medium transition-colors ${
                tab === t.id
                  ? "border-b-2 border-zinc-900 text-zinc-900 dark:border-zinc-100 dark:text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              }`}
            >
              {t.label}
              {t.count != null && (
                <span className="ml-1.5 rounded-full bg-amber-400 px-1.5 py-0.5 text-[9px] font-bold text-white">
                  {t.count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* ──────── CHAT TAB ──────── */}
        {tab === "chat" && (
          <div className="flex flex-1 flex-col gap-2">
            <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500">
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="radio" name="mode" checked={mode === "sync"} onChange={() => setMode("sync")} />
                <code>POST /chat</code> (sync)
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="radio" name="mode" checked={mode === "async"} onChange={() => setMode("async")} />
                <code>async job</code>
              </label>
            </div>

            <div className="flex flex-1 flex-col rounded-xl border border-zinc-200 bg-zinc-50/60 dark:border-zinc-800 dark:bg-zinc-950/60">
              {/* messages */}
              <div className="flex-1 overflow-y-auto p-4" style={{ maxHeight: "min(60vh, 540px)" }}>
                {messages.length === 0 && (
                  <p className="text-sm text-zinc-400">
                    Ask anything — the agent knows its own codebase, has session memory, and uses{" "}
                    <strong>Google Search</strong> for current events.
                  </p>
                )}
                {messages.map(m => (
                  <div key={m.id} className={`mb-3 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[92%] rounded-2xl px-4 py-2.5 text-sm ${
                      m.role === "user"
                        ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                        : "border border-zinc-200 bg-white text-zinc-900 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
                    }`}>
                      <pre className="whitespace-pre-wrap font-sans leading-relaxed">{m.content}</pre>
                      {m.turn && m.role === "assistant" && !m.error && (
                        <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-zinc-100 pt-2 font-mono text-[9px] text-zinc-400 dark:border-zinc-700">
                          <span>route={m.turn.route}</span>
                          <span>intent={m.turn.intent}</span>
                          {m.turn.hdc_similarity != null && (
                            <span>hdc={m.turn.hdc_similarity.toFixed(3)}</span>
                          )}
                          {m.turn.grounded && (
                            <span className="rounded bg-blue-100 px-1.5 py-0.5 text-blue-700 dark:bg-blue-900 dark:text-blue-300">
                              🔍 search
                            </span>
                          )}
                        </div>
                      )}
                      {m.error && (
                        <p className="mt-1 text-xs text-red-500">{m.error}</p>
                      )}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="mb-3 flex justify-start">
                    <div className="rounded-2xl border border-zinc-200 bg-white px-4 py-2.5 text-sm dark:border-zinc-700 dark:bg-zinc-900">
                      <span className="animate-pulse text-zinc-400">Thinking…</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* input */}
              <div className="border-t border-zinc-200 p-3 dark:border-zinc-800">
                <textarea
                  className="mb-2 w-full resize-none rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-1 focus:ring-zinc-400 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
                  rows={3}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  placeholder="Ask anything — current events, maths, about the codebase…"
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void (mode === "sync" ? sendSync() : sendAsync());
                    }
                  }}
                />
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={chatLoading}
                    onClick={() => void (mode === "sync" ? sendSync() : sendAsync())}
                    className="rounded-lg bg-zinc-900 px-5 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
                  >
                    {chatLoading ? "…" : "Send"}
                  </button>
                  <span className="text-[10px] text-zinc-400">Enter sends · Shift+Enter for newline</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ──────── IMPROVE TAB ──────── */}
        {tab === "improve" && (
          <div className="flex flex-col gap-6">
            <div className="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
              <h3 className="mb-1 text-sm font-semibold text-zinc-800 dark:text-zinc-100">
                Request a code improvement
              </h3>
              <p className="mb-4 text-xs text-zinc-500">
                Describe what you want changed or added. The agent will identify the best file,
                generate the improved code, run an AST check, and commit the change.
              </p>

              <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Instruction <span className="font-normal text-zinc-400">(required)</span>
              </label>
              <textarea
                className="mb-3 w-full resize-none rounded-lg border border-zinc-300 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-1 focus:ring-zinc-400 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
                rows={4}
                value={instruction}
                onChange={e => setInstruction(e.target.value)}
                placeholder="e.g. Add rate limiting to the /chat endpoint&#10;e.g. Improve the HDC memory to persist across restarts&#10;e.g. Add a /api/v1/sessions endpoint listing saved sessions"
              />

              <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Target file <span className="font-normal text-zinc-400">(optional — leave blank for auto-detect)</span>
              </label>
              <input
                className="mb-4 w-full rounded-lg border border-zinc-300 bg-zinc-50 px-3 py-2 font-mono text-xs text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-1 focus:ring-zinc-400 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
                value={targetFile}
                onChange={e => setTargetFile(e.target.value)}
                placeholder="super_agent/app/api/routes.py"
                spellCheck={false}
              />

              <button
                type="button"
                disabled={improving || !instruction.trim()}
                onClick={() => void submitImprovement()}
                className="rounded-lg bg-zinc-900 px-6 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
              >
                {improving ? "Applying improvement…" : "Apply improvement"}
              </button>
            </div>

            {/* result */}
            {improveResult && (
              <div className={`rounded-xl border p-5 ${
                improveResult.ok
                  ? "border-emerald-300 bg-emerald-50 dark:border-emerald-700 dark:bg-emerald-950/30"
                  : "border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-950/30"
              }`}>
                <div className="mb-3 flex items-center gap-3">
                  <span className={`text-lg font-bold ${improveResult.ok ? "text-emerald-600" : "text-red-600"}`}>
                    {improveResult.ok ? "✓ Applied" : "✗ Failed"}
                  </span>
                  {improveResult.committed && (
                    <span className="rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700 dark:bg-blue-900 dark:text-blue-200">
                      committed {improveResult.commit_hash?.slice(0, 8) ?? ""}
                    </span>
                  )}
                  {improveResult.ast_ok && (
                    <span className="rounded bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                      AST ✓
                    </span>
                  )}
                </div>
                <p className="mb-1 font-mono text-xs text-zinc-500">{improveResult.target_file}</p>
                {improveResult.error && (
                  <p className="mb-2 rounded bg-red-100 px-3 py-2 text-xs text-red-700 dark:bg-red-900/40 dark:text-red-300">
                    {improveResult.error}
                  </p>
                )}
                {improveResult.old_code && <CodeBlock code={improveResult.old_code} />}
                {improveResult.new_code && (
                  <div className="mt-2">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-emerald-600">New version:</span>
                    <CodeBlock code={improveResult.new_code} />
                  </div>
                )}
              </div>
            )}

            {/* history */}
            {improveHistory.entries.length > 0 && (
              <div>
                <h3 className="mb-3 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  Improvement history ({improveHistory.entries.length})
                </h3>
                <div className="space-y-2">
                  {improveHistory.entries.map((e, i) => (
                    <div key={i} className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <span className={`mr-2 text-xs font-semibold ${e.ok ? "text-emerald-600" : "text-red-500"}`}>
                            {e.ok ? "✓" : "✗"}
                          </span>
                          <span className="text-xs text-zinc-700 dark:text-zinc-200">{e.instruction}</span>
                        </div>
                        <span className="shrink-0 text-[10px] text-zinc-400">{fmtTs(e.timestamp)}</span>
                      </div>
                      <p className="mt-0.5 font-mono text-[10px] text-zinc-400">{e.target_file}</p>
                      {e.committed && (
                        <span className="text-[10px] text-blue-500">commit {e.commit_hash?.slice(0, 8)}</span>
                      )}
                      {e.error && <p className="mt-1 text-[10px] text-red-500">{e.error}</p>}
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
            <div className="flex items-center gap-3">
              <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Live codebase snapshot</h3>
              <button
                type="button"
                disabled={codebaseLoading}
                onClick={async () => {
                  setCodebaseLoading(true);
                  const b = baseUrl || getAgentBaseUrl();
                  await refreshCodebase(b);
                  await loadCodebase();
                }}
                className="rounded border border-zinc-300 px-2 py-1 text-xs text-zinc-600 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-300"
              >
                {codebaseLoading ? "Refreshing…" : "↺ Refresh scan"}
              </button>
              <span className="text-xs text-zinc-400">
                {codebase ? `${(codebase.length / 1024).toFixed(1)} KB` : ""}
              </span>
            </div>
            <p className="text-xs text-zinc-400">
              This is the exact document injected into the agent's system instruction on every turn.
            </p>
            <pre className="max-h-[70vh] overflow-auto rounded-xl border border-zinc-200 bg-zinc-950 p-5 text-[11px] leading-relaxed text-zinc-200 dark:border-zinc-800">
              {codebaseLoading ? "Scanning…" : codebase || "Click Refresh scan to generate."}
            </pre>
          </div>
        )}

        {/* ──────── STATUS TAB ──────── */}
        {tab === "status" && (
          <div className="flex flex-col gap-6">
            <div className="flex items-center gap-3">
              <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Blueprint status</h3>
              <button type="button" onClick={() => void loadStatus()}
                className="rounded border border-zinc-300 px-2 py-1 text-xs text-zinc-500 dark:border-zinc-700">
                ↺ Refresh
              </button>
            </div>

            {blueprint ? (
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
                      <th className="px-4 py-2 text-left font-semibold text-zinc-500">Feature</th>
                      <th className="px-4 py-2 text-left font-semibold text-zinc-500">Status</th>
                      <th className="px-4 py-2 text-left font-semibold text-zinc-500">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {blueprint.items.map(item => (
                      <tr key={item.id} className="border-b border-zinc-100 dark:border-zinc-800">
                        <td className="px-4 py-2 font-medium text-zinc-700 dark:text-zinc-200">{item.title}</td>
                        <td className="px-4 py-2"><StatusBadge status={item.status} /></td>
                        <td className="px-4 py-2 text-zinc-400">{item.notes}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-zinc-400">Blueprint not loaded. Click Refresh.</p>
            )}

            {sica && (
              <div>
                <h3 className="mb-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">SICA loop summary</h3>
                <pre className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200">
                  {sica}
                </pre>
              </div>
            )}

            <div>
              <h3 className="mb-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">All endpoints</h3>
              <div className="grid grid-cols-1 gap-1 font-mono text-[11px] sm:grid-cols-2">
                {[
                  ["GET", "/health"],
                  ["GET", "/api/v1/status"],
                  ["GET", "/api/v1/gaps"],
                  ["POST", "/api/v1/chat"],
                  ["POST", "/api/v1/agent/start"],
                  ["GET", "/api/v1/agent/jobs/:id"],
                  ["POST", "/api/v1/sympy/run"],
                  ["POST", "/api/v1/route-intent"],
                  ["GET", "/api/v1/codebase/snapshot"],
                  ["POST", "/api/v1/codebase/refresh"],
                  ["POST", "/api/v1/improve"],
                  ["GET", "/api/v1/improve/history"],
                  ["POST", "/api/v1/research/trigger"],
                  ["GET", "/api/v1/sica/summary"],
                ].map(([method, path]) => (
                  <div key={path} className="flex items-center gap-2 rounded bg-zinc-50 px-2 py-1 dark:bg-zinc-900">
                    <span className={`shrink-0 rounded px-1 text-[9px] font-bold uppercase ${
                      method === "GET"
                        ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                        : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
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
