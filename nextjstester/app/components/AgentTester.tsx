"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchHealth,
  fetchModelStats,
  fetchTrainStatus,
  fetchVocab,
  fetchHeartbeatStatus,
  fetchMemory,
  postChat,
  trainText,
  trainTopic,
  generate,
  solveAnalogy,
  findSimilar,
  setHeartbeatTopics,
  triggerHeartbeat,
  researchTopic,
  getAgentBaseUrl,
  getOrCreateSessionId,
  resetSessionId,
} from "@/lib/agent-api";
import type {
  ChatMessage,
  ModelStats,
  PipelineStats,
  HeartbeatStatus,
  VocabResponse,
  MemoryResponse,
  AnalogyResult,
  SimilarResult,
  GenerateResult,
  HealthCheck,
} from "@/lib/types";

// ── storage ───────────────────────────────────────────────────────────────────
const STORAGE_API      = "hlagent-api-base";
const STORAGE_MESSAGES = "hlagent-messages";
const DEFAULT_BASE     = "http://localhost:8000";

function lsRead<T>(key: string, fb: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fb;
  } catch { return fb; }
}

function lsWrite(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

// ── helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function timeAgo(iso: string | null): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

const MODE_COLOR: Record<string, string> = {
  generation:       "var(--cyan)",
  learning:         "var(--amber)",   // still training — amber = "work in progress"
  math:             "var(--green)",
  analogy:          "var(--violet)",
  similarity:       "var(--teal)",
  research:         "var(--blue)",
  error:            "var(--red)",
  chain_of_thought: "var(--blue)",
  busy:             "var(--txt-3)",
};

const MODE_LABEL: Record<string, string> = {
  generation: "generated",
  learning:   "still learning",
  math:       "math",
  analogy:    "analogy",
  similarity: "similar",
  research:   "researched",
  error:      "error",
  chain_of_thought: "chain",
  busy:       "busy",
};

// ── animated counter ──────────────────────────────────────────────────────────
function useAnimatedValue(target: number, duration = 600): number {
  const [display, setDisplay] = useState(target);
  const prev = useRef(target);
  useEffect(() => {
    if (prev.current === target) return;
    const start = prev.current;
    const diff = target - start;
    const t0 = Date.now();
    const tick = () => {
      const elapsed = Date.now() - t0;
      const prog = Math.min(elapsed / duration, 1);
      setDisplay(Math.round(start + diff * prog));
      if (prog < 1) requestAnimationFrame(tick);
      else prev.current = target;
    };
    requestAnimationFrame(tick);
  }, [target, duration]);
  return display;
}

// ── stat card ─────────────────────────────────────────────────────────────────
function StatCard({
  label, value, sub, color, glow,
}: {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
  glow?: boolean;
}) {
  return (
    <div
      style={{
        background: "var(--card)",
        border: `1px solid ${glow ? color ?? "var(--border)" : "var(--border-dim)"}`,
        borderRadius: 10,
        padding: "14px 16px",
        boxShadow: glow ? `0 0 18px ${color ?? "var(--cyan)"}33` : undefined,
        transition: "border-color .3s, box-shadow .3s",
      }}
    >
      <div style={{ fontSize: 11, color: "var(--txt-2)", textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color: color ?? "var(--txt)", fontFamily: "var(--font-mono)", lineHeight: 1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: "var(--txt-3)", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ── confidence bar ────────────────────────────────────────────────────────────
function ConfBar({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ height: 3, background: "var(--elevated)", borderRadius: 2, overflow: "hidden", marginTop: 4 }}>
      <div
        style={{
          height: "100%",
          width: `${Math.min(value * 100, 100)}%`,
          background: color,
          borderRadius: 2,
          transition: "width .4s",
        }}
      />
    </div>
  );
}

// ── similarity pill ───────────────────────────────────────────────────────────
function SimPill({ word, similarity }: { word: string; similarity: number }) {
  const pct = Math.round(similarity * 100);
  const hue = Math.round(120 * similarity);
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "6px 10px",
        background: "var(--elevated)",
        border: "1px solid var(--border-dim)",
        borderRadius: 8,
        fontSize: 13,
        gap: 12,
      }}
    >
      <span style={{ color: "var(--txt)", fontFamily: "var(--font-mono)" }}>{word}</span>
      <span style={{ color: `hsl(${hue},70%,60%)`, fontFamily: "var(--font-mono)", fontSize: 11 }}>
        {pct}%
      </span>
    </div>
  );
}

// ── chat bubble ───────────────────────────────────────────────────────────────
function ChatBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  const mode = msg.turn?.mode ?? "generation";
  const conf = msg.turn?.confidence ?? 0;
  const color = MODE_COLOR[mode] ?? "var(--cyan)";
  return (
    <div
      className="slide-up"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        marginBottom: 12,
      }}
    >
      {!isUser && (
        <div style={{ fontSize: 10, color: "var(--txt-3)", marginBottom: 3, display: "flex", gap: 8, alignItems: "center" }}>
          <span
            style={{
              color,
              background: `${color}1a`,
              border: `1px solid ${color}44`,
              borderRadius: 4,
              padding: "1px 6px",
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: ".06em",
              fontFamily: "var(--font-mono)",
            }}
          >
            {MODE_LABEL[mode] ?? mode}
          </span>
          <span style={{ color: conf < 0.05 ? "var(--amber)" : "var(--txt-3)" }}>
            {conf > 0 ? `conf ${(conf * 100).toFixed(1)}%` : "building…"}
          </span>
        </div>
      )}
      <div
        style={{
          maxWidth: "82%",
          padding: "10px 14px",
          borderRadius: isUser ? "14px 14px 4px 14px" : "4px 14px 14px 14px",
          background: isUser ? "var(--blue)" : "var(--elevated)",
          border: isUser ? "none" : `1px solid var(--border-dim)`,
          color: "var(--txt)",
          fontSize: 14,
          lineHeight: 1.6,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {msg.content}
        {msg.error && <div style={{ color: "var(--red)", fontSize: 12, marginTop: 6 }}>{msg.error}</div>}
      </div>
      {!isUser && conf > 0 && <ConfBar value={conf} color={color} />}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════

type Tab = "chat" | "brain" | "research" | "generate";

export default function AgentTester() {
  // ── SSR-safe mount ─────────────────────────────────────────────────────────
  // All localStorage reads happen in useEffect to avoid hydration mismatch.
  const [mounted, setMounted] = useState(false);

  // ── api base ────────────────────────────────────────────────────────────────
  const [base, setBase] = useState<string>(DEFAULT_BASE);
  const [editingBase, setEditingBase] = useState(false);
  const [tempBase, setTempBase] = useState(DEFAULT_BASE);

  // ── live state ──────────────────────────────────────────────────────────────
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [stats, setStats] = useState<ModelStats | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStats | null>(null);
  const [heartbeat, setHeartbeat] = useState<HeartbeatStatus | null>(null);
  const [sessionId] = useState<string>(() => getOrCreateSessionId());
  const [tab, setTab] = useState<Tab>("chat");

  // ── training feed (live log of what the model is learning) ─────────────────
  const [trainingFeed, setTrainingFeed] = useState<{
    time: string;
    type: "train" | "research" | "heartbeat" | "info";
    topic: string;
    detail: string;
  }[]>([]);

  // ── chat ────────────────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // ── brain ───────────────────────────────────────────────────────────────────
  const [vocab, setVocab] = useState<VocabResponse | null>(null);
  const [vocabFilter, setVocabFilter] = useState("");
  const [analogyA, setAnalogyA] = useState("");
  const [analogyB, setAnalogyB] = useState("");
  const [analogyC, setAnalogyC] = useState("");
  const [analogyResult, setAnalogyResult] = useState<AnalogyResult | null>(null);
  const [analogyLoading, setAnalogyLoading] = useState(false);
  const [similarWord, setSimilarWord] = useState("");
  const [similarResult, setSimilarResult] = useState<SimilarResult | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [memory, setMemory] = useState<MemoryResponse | null>(null);

  // ── research ─────────────────────────────────────────────────────────────────
  const [researchInput, setResearchInput] = useState("");
  const [researchRunning, setResearchRunning] = useState(false);
  const [researchLog, setResearchLog] = useState<string[]>([]);
  const [topicsEdit, setTopicsEdit] = useState("");
  const [topicsEditMode, setTopicsEditMode] = useState(false);
  const [trainTextInput, setTrainTextInput] = useState("");
  const [trainTextLoading, setTrainTextLoading] = useState(false);
  const [trainTextResult, setTrainTextResult] = useState<string | null>(null);

  // ── generate ─────────────────────────────────────────────────────────────────
  const [genSeed, setGenSeed] = useState("");
  const [genTemp, setGenTemp] = useState(0.8);
  const [genTokens, setGenTokens] = useState(50);
  const [genResult, setGenResult] = useState<GenerateResult | null>(null);
  const [genLoading, setGenLoading] = useState(false);

  // animated stats
  const animVocab  = useAnimatedValue(stats?.vocab_size ?? 0);
  const animTokens = useAnimatedValue(stats?.training_tokens ?? 0);
  const animAssoc  = useAnimatedValue(stats?.assoc_memory ?? 0);
  const animDocs   = useAnimatedValue(stats?.training_docs ?? 0);

  // ── hydration-safe localStorage load ────────────────────────────────────────
  useEffect(() => {
    const savedBase = lsRead(STORAGE_API, DEFAULT_BASE);
    const savedMsgs = lsRead<ChatMessage[]>(STORAGE_MESSAGES, []);
    setBase(savedBase);
    setTempBase(savedBase);
    setMessages(savedMsgs);
    setMounted(true);
  }, []);

  // ── polling ─────────────────────────────────────────────────────────────────
  const pollStats = useCallback(async () => {
    if (!base || !mounted) return;
    const [h, s, p, hb] = await Promise.all([
      fetchHealth(base),
      fetchModelStats(base),
      fetchTrainStatus(base),
      fetchHeartbeatStatus(base),
    ]);
    setHealth(h);
    if (s) setStats(s);
    if (p) setPipeline(p);
    if (hb) setHeartbeat(hb);
  }, [base, mounted]);

  const prevRunsRef = useRef<number>(0);

  useEffect(() => { if (mounted) pollStats(); }, [pollStats, mounted]);
  useEffect(() => {
    if (!mounted) return;
    const id = setInterval(async () => {
      await pollStats();
      // Detect when heartbeat fires automatically (total_runs increments)
      setHeartbeat(hb => {
        if (hb && hb.total_runs > prevRunsRef.current) {
          prevRunsRef.current = hb.total_runs;
          if (hb.next_topic) {
            setTrainingFeed(prev => [{
              time: new Date().toLocaleTimeString(),
              type: "heartbeat" as const,
              topic: hb.next_topic!,
              detail: `Heartbeat #${hb.total_runs} completed`,
            }, ...prev.slice(0, 49)]);
          }
        }
        return hb;
      });
    }, 4000);
    return () => clearInterval(id);
  }, [pollStats, mounted]);

  // fetch vocab when brain tab opens
  useEffect(() => {
    if (tab === "brain" && base) {
      fetchVocab(base, 300).then(v => { if (v) setVocab(v); });
      fetchMemory(base, 30).then(m => { if (m) setMemory(m); });
    }
  }, [tab, base]);

  // scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // persist messages (only after mount to avoid SSR issues)
  useEffect(() => {
    if (mounted) lsWrite(STORAGE_MESSAGES, messages.slice(-60));
  }, [messages, mounted]);

  // ── chat ────────────────────────────────────────────────────────────────────
  const sendChat = useCallback(async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: msg, at: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setChatInput("");
    setChatLoading(true);
    try {
      const result = await postChat(base, msg, sessionId);
      const asstMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer,
        turn: result,
        at: Date.now(),
      };
      setMessages(prev => [...prev, asstMsg]);
    } catch (e) {
      const asstMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        error: e instanceof Error ? e.message : "Error",
        at: Date.now(),
      };
      setMessages(prev => [...prev, asstMsg]);
    } finally {
      setChatLoading(false);
      pollStats();
    }
  }, [chatInput, chatLoading, base, sessionId, pollStats]);

  // ── training feed helper ─────────────────────────────────────────────────────
  const addFeedEntry = useCallback((
    type: "train" | "research" | "heartbeat" | "info",
    topic: string,
    detail: string,
  ) => {
    setTrainingFeed(prev => [{
      time: new Date().toLocaleTimeString(),
      type,
      topic,
      detail,
    }, ...prev.slice(0, 49)]);
  }, []);

  // ── analogy ─────────────────────────────────────────────────────────────────
  const runAnalogy = useCallback(async () => {
    if (!analogyA || !analogyB || !analogyC || analogyLoading) return;
    setAnalogyLoading(true);
    const r = await solveAnalogy(base, analogyA, analogyB, analogyC);
    setAnalogyResult(r);
    setAnalogyLoading(false);
  }, [base, analogyA, analogyB, analogyC, analogyLoading]);

  // ── similarity ───────────────────────────────────────────────────────────────
  const runSimilar = useCallback(async () => {
    if (!similarWord || similarLoading) return;
    setSimilarLoading(true);
    const r = await findSimilar(base, similarWord, 10);
    setSimilarResult(r);
    setSimilarLoading(false);
  }, [base, similarWord, similarLoading]);

  // ── research ─────────────────────────────────────────────────────────────────
  const runResearch = useCallback(async (topic: string) => {
    if (!topic.trim() || researchRunning) return;
    setResearchRunning(true);
    addFeedEntry("research", topic, "Scraping Wikipedia + DuckDuckGo…");
    setResearchLog(prev => [`[${new Date().toLocaleTimeString()}] Researching: "${topic}"...`, ...prev]);
    const r = await researchTopic(base, topic.trim());
    if (r) {
      const d = r.details as Record<string, unknown>;
      const detail = `${d.docs_count} docs · ${(d.total_words as number)?.toLocaleString()} words · ${d.pairs_trained} pairs`;
      setResearchLog(prev => [`[${new Date().toLocaleTimeString()}] ✓ ${detail}`, ...prev]);
      addFeedEntry("research", topic, `✓ ${detail}`);
    } else {
      setResearchLog(prev => [`[${new Date().toLocaleTimeString()}] ✗ Failed`, ...prev]);
      addFeedEntry("research", topic, "✗ Failed");
    }
    setResearchRunning(false);
    pollStats();
  }, [base, researchRunning, pollStats, addFeedEntry]);

  const triggerHeartbeatNow = useCallback(async () => {
    const next = heartbeat?.next_topic ?? "next scheduled topic";
    const ok = await triggerHeartbeat(base);
    if (ok) {
      addFeedEntry("heartbeat", next, "Heartbeat triggered — scraping in background");
      setResearchLog(prev => [`[${new Date().toLocaleTimeString()}] ✓ Heartbeat triggered: "${next}"`, ...prev]);
    } else {
      setResearchLog(prev => [`[${new Date().toLocaleTimeString()}] ✗ Heartbeat trigger failed`, ...prev]);
    }
    setTimeout(pollStats, 3000);
  }, [base, heartbeat, pollStats, addFeedEntry]);

  // ── train text ───────────────────────────────────────────────────────────────
  const runTrainText = useCallback(async () => {
    if (!trainTextInput.trim() || trainTextLoading) return;
    setTrainTextLoading(true);
    setTrainTextResult(null);
    const preview = trainTextInput.trim().slice(0, 60) + (trainTextInput.length > 60 ? "…" : "");
    addFeedEntry("train", "manual text", `Training: "${preview}"`);
    const r = await trainText(base, trainTextInput.trim());
    if (r) {
      const result = `✓ ${r.pairs_trained} pairs trained — vocab now ${r.vocab_size.toLocaleString()}`;
      setTrainTextResult(result);
      addFeedEntry("train", "manual text", result);
      setTrainTextInput("");
    } else {
      setTrainTextResult("✗ Training failed");
      addFeedEntry("train", "manual text", "✗ Training failed");
    }
    setTrainTextLoading(false);
    pollStats();
  }, [base, trainTextInput, trainTextLoading, pollStats, addFeedEntry]);

  // ── generate ─────────────────────────────────────────────────────────────────
  const runGenerate = useCallback(async () => {
    if (!genSeed.trim() || genLoading) return;
    setGenLoading(true);
    setGenResult(null);
    const r = await generate(base, genSeed.trim(), genTokens, genTemp);
    setGenResult(r);
    setGenLoading(false);
    if (r?.generated) {
      addFeedEntry("info", "generation", `"${genSeed}" → "${r.generated.slice(0, 60)}"`);
    }
  }, [base, genSeed, genTokens, genTemp, genLoading, addFeedEntry]);

  // ── base URL save ─────────────────────────────────────────────────────────────
  const saveBase = () => {
    const v = tempBase.trim().replace(/\/$/, "");
    setBase(v);
    lsWrite(STORAGE_API, v);
    setEditingBase(false);
    setTimeout(pollStats, 200);
  };


  // ── save topics ───────────────────────────────────────────────────────────────
  const saveTopics = async () => {
    const topics = topicsEdit.split("\n").map(t => t.trim()).filter(Boolean);
    const ok = await setHeartbeatTopics(base, topics);
    if (ok) { setTopicsEditMode(false); pollStats(); }
  };

  // ── status indicator ─────────────────────────────────────────────────────────
  const connected = health?.ok === true;

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  const TAB_ITEMS: { id: Tab; label: string; icon: string }[] = [
    { id: "chat",     label: "Chat",     icon: "◎" },
    { id: "brain",    label: "Brain",    icon: "⬡" },
    { id: "research", label: "Research", icon: "⊞" },
    { id: "generate", label: "Generate", icon: "≋" },
  ];

  // Prevent SSR/client mismatch — render nothing until client hydration completes
  if (!mounted) {
    return (
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center",
        height: "100vh", background: "var(--void)", color: "var(--txt-3)", fontSize: 13,
        fontFamily: "var(--font-mono)", gap: 10,
      }}>
        <div className="neural-dot" style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--cyan)" }} />
        <div className="neural-dot" style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--cyan)", animationDelay: "0.16s" }} />
        <div className="neural-dot" style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--cyan)", animationDelay: "0.32s" }} />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--void)", fontFamily: "var(--font-sans)" }}>

      {/* ── TOP BAR ── */}
      <header style={{
        display: "flex", alignItems: "center", gap: 12, padding: "0 20px",
        height: 52, background: "var(--surface)", borderBottom: "1px solid var(--border-dim)",
        flexShrink: 0,
      }}>
        {/* logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginRight: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: "linear-gradient(135deg, var(--cyan), var(--blue))",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 700, color: "#000",
          }}>H</div>
          <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: "-.01em" }}>HDC Brain</span>
        </div>

        {/* status dot */}
        <div
          className={connected ? "dot-live" : "dot-amber"}
          style={{
            width: 8, height: 8, borderRadius: "50%",
            background: connected ? "var(--green)" : "var(--amber)",
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: 12, color: connected ? "var(--green)" : "var(--amber)" }}>
          {connected ? "connected" : health ? "offline" : "connecting…"}
        </span>

        {/* mini stats */}
        {stats && (
          <div style={{ display: "flex", gap: 16, marginLeft: "auto", fontSize: 12, color: "var(--txt-2)", fontFamily: "var(--font-mono)" }}>
            <span>vocab <span style={{ color: "var(--cyan)" }}>{fmt(stats.vocab_size)}</span></span>
            <span>tokens <span style={{ color: "var(--green)" }}>{fmt(stats.training_tokens)}</span></span>
            <span>assoc <span style={{ color: "var(--violet)" }}>{fmt(stats.assoc_memory)}</span></span>
          </div>
        )}

        {/* api url */}
        <div style={{ marginLeft: stats ? 16 : "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {editingBase ? (
            <>
              <input
                value={tempBase}
                onChange={e => setTempBase(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") saveBase(); if (e.key === "Escape") setEditingBase(false); }}
                autoFocus
                style={{ background: "var(--elevated)", border: "1px solid var(--border)", borderRadius: 6, padding: "3px 8px", fontSize: 12, color: "var(--txt)", width: 240, fontFamily: "var(--font-mono)" }}
              />
              <button onClick={saveBase} style={{ fontSize: 11, color: "var(--cyan)", background: "none", border: "none", cursor: "pointer" }}>Save</button>
              <button onClick={() => setEditingBase(false)} style={{ fontSize: 11, color: "var(--txt-3)", background: "none", border: "none", cursor: "pointer" }}>✕</button>
            </>
          ) : (
            <button
              onClick={() => { setTempBase(base); setEditingBase(true); }}
              style={{ fontSize: 11, color: "var(--txt-3)", background: "none", border: "none", cursor: "pointer", fontFamily: "var(--font-mono)" }}
            >
              {base || "set API URL"}
            </button>
          )}
        </div>
      </header>

      {/* ── BODY ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* ── LEFT SIDEBAR ── */}
        <aside style={{
          width: 200, flexShrink: 0, background: "var(--surface)",
          borderRight: "1px solid var(--border-dim)",
          display: "flex", flexDirection: "column", padding: "16px 12px", gap: 8, overflowY: "auto",
        }}>
          {/* tabs */}
          {TAB_ITEMS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "9px 12px", borderRadius: 8,
                background: tab === t.id ? "var(--elevated)" : "transparent",
                border: tab === t.id ? "1px solid var(--border)" : "1px solid transparent",
                color: tab === t.id ? "var(--txt)" : "var(--txt-2)",
                fontSize: 13, cursor: "pointer", textAlign: "left", fontFamily: "var(--font-sans)",
                transition: "all .15s",
              }}
            >
              <span style={{ fontFamily: "var(--font-mono)", fontSize: 14 }}>{t.icon}</span>
              {t.label}
            </button>
          ))}

          {/* model stats */}
          <div style={{ marginTop: 16, borderTop: "1px solid var(--border-dim)", paddingTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ fontSize: 10, color: "var(--txt-3)", textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 2 }}>Model</div>
            {[
              { label: "Vocab",  val: fmt(animVocab),  color: "var(--cyan)" },
              { label: "Tokens", val: fmt(animTokens), color: "var(--green)" },
              { label: "Assoc",  val: fmt(animAssoc),  color: "var(--violet)" },
              { label: "Docs",   val: fmt(animDocs),   color: "var(--amber)" },
            ].map(row => (
              <div key={row.label} style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                <span style={{ color: "var(--txt-3)" }}>{row.label}</span>
                <span style={{ color: row.color, fontFamily: "var(--font-mono)" }}>{row.val}</span>
              </div>
            ))}
            {stats?.last_trained && (
              <div style={{ fontSize: 10, color: "var(--txt-3)", marginTop: 2 }}>
                trained {timeAgo(stats.last_trained)}
              </div>
            )}
          </div>

          {/* heartbeat */}
          {heartbeat && (
            <div style={{ marginTop: 8, borderTop: "1px solid var(--border-dim)", paddingTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ fontSize: 10, color: "var(--txt-3)", textTransform: "uppercase", letterSpacing: ".08em" }}>Heartbeat</div>
              <div style={{ fontSize: 11, color: "var(--txt-2)" }}>
                {heartbeat.total_runs} runs · {heartbeat.topic_count} topics
              </div>
              <div style={{ fontSize: 10, color: "var(--txt-3)" }}>last: {timeAgo(heartbeat.last_run)}</div>
              {heartbeat.next_topic && (
                <div
                  style={{
                    fontSize: 10, color: "var(--amber)", background: "rgba(255,162,0,.08)",
                    border: "1px solid rgba(255,162,0,.2)", borderRadius: 5, padding: "4px 6px",
                    lineHeight: 1.4,
                  }}
                >
                  Next: {heartbeat.next_topic.slice(0, 50)}…
                </div>
              )}
            </div>
          )}

          {/* clear chat */}
          {tab === "chat" && (
            <button
              onClick={() => setMessages([])}
              style={{
                marginTop: "auto", padding: "7px", borderRadius: 7, fontSize: 12,
                background: "transparent", border: "1px solid var(--border-dim)",
                color: "var(--txt-3)", cursor: "pointer",
              }}
            >
              Clear chat
            </button>
          )}
        </aside>

        {/* ── MAIN CONTENT ── */}
        <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>

          {/* ══════ CHAT TAB ══════ */}
          {tab === "chat" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px" }}>
                {messages.length === 0 && (
                  <div style={{
                    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    height: "100%", gap: 12, color: "var(--txt-3)",
                  }}>
                    <div style={{ fontSize: 48 }}>⬡</div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: "var(--txt-2)" }}>HDC Language Model</div>
                    <div style={{ fontSize: 13, maxWidth: 360, textAlign: "center", lineHeight: 1.6 }}>
                      Chat with your hyperdimensional brain. It learns from everything it researches and gets smarter over time.
                    </div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center", marginTop: 8 }}>
                      {["What is HDC?", "How does bundling work?", "Tell me about one-shot learning"].map(q => (
                        <button
                          key={q}
                          onClick={() => { setChatInput(q); }}
                          style={{
                            padding: "6px 12px", borderRadius: 20, fontSize: 12,
                            background: "var(--elevated)", border: "1px solid var(--border-dim)",
                            color: "var(--txt-2)", cursor: "pointer",
                          }}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {messages.map(m => <ChatBubble key={m.id} msg={m} />)}
                {chatLoading && (
                  <div style={{ display: "flex", gap: 4, padding: "8px 0", paddingLeft: 4 }}>
                    {[0, 1, 2].map(i => (
                      <div key={i} className="neural-dot" style={{
                        width: 7, height: 7, borderRadius: "50%", background: "var(--cyan)",
                        animationDelay: `${i * 0.16}s`,
                      }} />
                    ))}
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              <div style={{
                padding: "12px 20px", background: "var(--surface)", borderTop: "1px solid var(--border-dim)",
                display: "flex", gap: 10,
              }}>
                <input
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                  placeholder="Ask the HDC brain anything…"
                  disabled={chatLoading || !connected}
                  style={{
                    flex: 1, padding: "10px 14px", borderRadius: 10,
                    background: "var(--elevated)", border: "1px solid var(--border-dim)",
                    color: "var(--txt)", fontSize: 14, outline: "none",
                    transition: "border-color .2s",
                  }}
                />
                <button
                  onClick={sendChat}
                  disabled={chatLoading || !chatInput.trim() || !connected}
                  style={{
                    padding: "10px 20px", borderRadius: 10,
                    background: "var(--blue)", border: "none",
                    color: "#fff", fontSize: 14, fontWeight: 600, cursor: "pointer",
                    opacity: (chatLoading || !chatInput.trim() || !connected) ? 0.4 : 1,
                    transition: "opacity .2s",
                  }}
                >
                  Send
                </button>
              </div>
            </div>
          )}

          {/* ══════ BRAIN TAB ══════ */}
          {tab === "brain" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 24, display: "grid", gap: 20, gridTemplateColumns: "1fr 1fr", alignContent: "start" }}>

              {/* top stats row */}
              <div style={{ gridColumn: "1 / -1", display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
                <StatCard label="Vocabulary" value={fmt(animVocab)} sub={`D = ${stats ? pipeline?.model.dim?.toLocaleString() : "10,000"}`} color="var(--cyan)" glow />
                <StatCard label="Training Tokens" value={fmt(animTokens)} sub={`${fmt(animDocs)} docs`} color="var(--green)" glow />
                <StatCard label="Assoc Memory" value={fmt(animAssoc)} sub="bound pairs" color="var(--violet)" />
                <StatCard label="HDC Records" value={fmt(stats?.hdc_memory_records ?? 0)} sub="task→solution" color="var(--amber)" />
              </div>

              {/* analogy solver */}
              <div style={{ background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ color: "var(--violet)" }}>⊛</span> Analogy Solver
                  <span style={{ fontSize: 11, color: "var(--txt-3)", fontWeight: 400 }}>A is to B as C is to ?</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr auto 1fr", gap: 6, alignItems: "center", marginBottom: 12 }}>
                  {[
                    { val: analogyA, set: setAnalogyA, ph: "A" },
                    { val: "→", set: null, ph: "" },
                    { val: analogyB, set: setAnalogyB, ph: "B" },
                    { val: "::", set: null, ph: "" },
                    { val: analogyC, set: setAnalogyC, ph: "C" },
                  ].map((f, i) =>
                    f.set ? (
                      <input
                        key={i}
                        value={f.val}
                        onChange={e => f.set!(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter") runAnalogy(); }}
                        placeholder={f.ph}
                        style={{
                          padding: "8px 10px", borderRadius: 8, fontSize: 13,
                          background: "var(--elevated)", border: "1px solid var(--border-dim)",
                          color: "var(--txt)", outline: "none", textAlign: "center",
                          fontFamily: "var(--font-mono)",
                        }}
                      />
                    ) : (
                      <span key={i} style={{ textAlign: "center", color: "var(--txt-3)", fontFamily: "var(--font-mono)", fontSize: 14 }}>{f.val}</span>
                    )
                  )}
                </div>
                <button
                  onClick={runAnalogy}
                  disabled={analogyLoading || !analogyA || !analogyB || !analogyC}
                  style={{
                    width: "100%", padding: "8px", borderRadius: 8, fontSize: 13,
                    background: "var(--elevated)", border: "1px solid var(--border)",
                    color: "var(--violet)", cursor: "pointer",
                    opacity: analogyLoading ? 0.6 : 1,
                  }}
                >
                  {analogyLoading ? "Solving…" : "Solve Analogy"}
                </button>
                {analogyResult && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, color: "var(--txt-3)", marginBottom: 6, fontFamily: "var(--font-mono)" }}>{analogyResult.query}</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      {analogyResult.candidates.map((c, i) => (
                        <SimPill key={i} word={c.word} similarity={c.similarity} />
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* word similarity */}
              <div style={{ background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ color: "var(--teal)" }}>⊹</span> Semantic Similarity
                </div>
                <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                  <input
                    value={similarWord}
                    onChange={e => setSimilarWord(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") runSimilar(); }}
                    placeholder="Enter a word…"
                    style={{
                      flex: 1, padding: "8px 10px", borderRadius: 8, fontSize: 13,
                      background: "var(--elevated)", border: "1px solid var(--border-dim)",
                      color: "var(--txt)", outline: "none", fontFamily: "var(--font-mono)",
                    }}
                  />
                  <button
                    onClick={runSimilar}
                    disabled={similarLoading || !similarWord}
                    style={{
                      padding: "8px 14px", borderRadius: 8, fontSize: 12,
                      background: "var(--elevated)", border: "1px solid var(--border)",
                      color: "var(--teal)", cursor: "pointer",
                    }}
                  >
                    {similarLoading ? "…" : "Search"}
                  </button>
                </div>
                {similarResult && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {similarResult.similar.length === 0 ? (
                      <div style={{ fontSize: 12, color: "var(--txt-3)" }}>No similar words found. Train more data first.</div>
                    ) : similarResult.similar.map((s, i) => (
                      <SimPill key={i} word={s.word} similarity={s.similarity} />
                    ))}
                  </div>
                )}
              </div>

              {/* vocab browser */}
              <div style={{ gridColumn: "1 / -1", background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ color: "var(--cyan)" }}>⊞</span> Vocabulary Browser
                  </div>
                  <span style={{ fontSize: 11, color: "var(--txt-3)" }}>
                    {vocab?.vocab_size.toLocaleString() ?? "—"} words in memory
                  </span>
                  <input
                    value={vocabFilter}
                    onChange={e => setVocabFilter(e.target.value)}
                    placeholder="filter…"
                    style={{
                      marginLeft: "auto", padding: "4px 10px", borderRadius: 6, fontSize: 12,
                      background: "var(--elevated)", border: "1px solid var(--border-dim)",
                      color: "var(--txt)", outline: "none", width: 140, fontFamily: "var(--font-mono)",
                    }}
                  />
                </div>
                <div style={{
                  display: "flex", flexWrap: "wrap", gap: 6, maxHeight: 180, overflowY: "auto",
                }}>
                  {(vocab?.sample ?? [])
                    .filter(w => !vocabFilter || w.includes(vocabFilter.toLowerCase()))
                    .map(w => (
                      <span
                        key={w}
                        onClick={() => { setSimilarWord(w); setTab("brain"); }}
                        style={{
                          padding: "3px 8px", borderRadius: 5, fontSize: 12,
                          background: "var(--elevated)", border: "1px solid var(--border-dim)",
                          color: "var(--txt-2)", cursor: "pointer", fontFamily: "var(--font-mono)",
                          transition: "border-color .15s",
                        }}
                        onMouseEnter={e => (e.currentTarget.style.borderColor = "var(--cyan)")}
                        onMouseLeave={e => (e.currentTarget.style.borderColor = "var(--border-dim)")}
                      >
                        {w}
                      </span>
                    ))
                  }
                  {vocab === null && (
                    <div style={{ fontSize: 12, color: "var(--txt-3)" }}>Loading vocabulary…</div>
                  )}
                </div>
              </div>

              {/* memory records */}
              {memory && memory.records.length > 0 && (
                <div style={{ gridColumn: "1 / -1", background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--txt-2)", marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ color: "var(--amber)" }}>◈</span> Associative Memory Records
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {memory.records.slice(0, 10).map((r, i) => (
                      <div key={i} style={{
                        display: "flex", gap: 10, alignItems: "center",
                        padding: "6px 10px", background: "var(--elevated)", borderRadius: 7,
                        border: "1px solid var(--border-dim)", fontSize: 12,
                      }}>
                        <span style={{ color: MODE_COLOR[r.route] ?? "var(--txt-2)", fontFamily: "var(--font-mono)", fontSize: 10, flexShrink: 0 }}>
                          {r.route}
                        </span>
                        <span style={{ color: "var(--txt-2)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {r.task_fp}
                        </span>
                        <span style={{ color: "var(--txt-3)", flexShrink: 0, fontSize: 11 }}>
                          ×{r.retrieval_count}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ══════ RESEARCH TAB ══════ */}
          {tab === "research" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 24, display: "grid", gap: 20, gridTemplateColumns: "1fr 1fr", alignContent: "start" }}>

              {/* pipeline stats */}
              <div style={{ gridColumn: "1 / -1", display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
                <StatCard
                  label="Docs Trained"
                  value={fmt(pipeline?.pipeline.documents_trained ?? 0)}
                  sub={`${fmt(pipeline?.pipeline.documents_skipped ?? 0)} skipped`}
                  color="var(--green)"
                  glow={!!pipeline?.pipeline.documents_trained}
                />
                <StatCard
                  label="Research Runs"
                  value={pipeline?.pipeline.research_runs ?? 0}
                  sub={pipeline?.pipeline.last_run ? timeAgo(pipeline.pipeline.last_run) : "—"}
                  color="var(--cyan)"
                />
                <StatCard
                  label="Unique Docs"
                  value={fmt(pipeline?.pipeline.unique_docs_seen ?? 0)}
                  sub="content hashes"
                  color="var(--violet)"
                />
                <StatCard
                  label="Tokens / Run"
                  value={pipeline && pipeline.pipeline.research_runs > 0
                    ? fmt(Math.round((pipeline.pipeline.tokens_trained) / pipeline.pipeline.research_runs))
                    : "—"}
                  sub="avg per research run"
                  color="var(--amber)"
                />
              </div>

              {/* ── TRAINING FEED ── */}
              <div style={{ gridColumn: "1 / -1", background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ color: "var(--green)" }}>▸</span> Live Training Feed
                  </div>
                  {researchRunning && (
                    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      {[0, 1, 2].map(i => (
                        <div key={i} className="neural-dot" style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--cyan)", animationDelay: `${i * 0.16}s` }} />
                      ))}
                      <span style={{ fontSize: 11, color: "var(--cyan)", marginLeft: 4 }}>scraping…</span>
                    </div>
                  )}
                  <div style={{ marginLeft: "auto", fontSize: 11, color: "var(--txt-3)" }}>
                    {pipeline ? (
                      <>dim <span style={{ color: "var(--txt-2)", fontFamily: "var(--font-mono)" }}>{pipeline.model.dim?.toLocaleString()}</span>
                      {" · "}ctx <span style={{ color: "var(--txt-2)", fontFamily: "var(--font-mono)" }}>{pipeline.model.context_size}</span></>
                    ) : null}
                  </div>
                </div>

                {/* vocab growth bar */}
                {stats && stats.vocab_size > 0 && (
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--txt-3)", marginBottom: 5 }}>
                      <span>Vocabulary growth</span>
                      <span style={{ fontFamily: "var(--font-mono)", color: "var(--cyan)" }}>
                        {stats.vocab_size.toLocaleString()} / ~50,000 target
                      </span>
                    </div>
                    <div style={{ height: 6, background: "var(--elevated)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{
                        height: "100%",
                        width: `${Math.min((stats.vocab_size / 50000) * 100, 100)}%`,
                        background: "linear-gradient(90deg, var(--cyan), var(--blue))",
                        borderRadius: 3,
                        transition: "width 1s",
                      }} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--txt-3)", marginTop: 4 }}>
                      <span>0</span>
                      <span>10K</span>
                      <span>25K</span>
                      <span>50K</span>
                    </div>
                  </div>
                )}

                {/* feed entries */}
                <div style={{
                  display: "flex", flexDirection: "column", gap: 4,
                  maxHeight: 220, overflowY: "auto",
                }}>
                  {trainingFeed.length === 0 ? (
                    <div style={{
                      padding: "24px", textAlign: "center", color: "var(--txt-3)",
                      fontSize: 13, background: "var(--elevated)", borderRadius: 8,
                    }}>
                      No training activity yet. Research a topic or train on text below to see the feed.
                    </div>
                  ) : trainingFeed.map((entry, i) => {
                    const colors = {
                      research: "var(--cyan)",
                      heartbeat: "var(--green)",
                      train: "var(--teal)",
                      info: "var(--txt-3)",
                    };
                    const icons = { research: "⊞", heartbeat: "⬡", train: "⊕", info: "◦" };
                    const c = colors[entry.type];
                    return (
                      <div key={i} className={i === 0 ? "slide-up" : undefined} style={{
                        display: "flex", gap: 10, alignItems: "flex-start",
                        padding: "7px 10px", borderRadius: 8,
                        background: i === 0 ? `${c}0d` : "var(--elevated)",
                        border: `1px solid ${i === 0 ? `${c}33` : "var(--border-dim)"}`,
                        fontSize: 12,
                      }}>
                        <span style={{ color: c, fontFamily: "var(--font-mono)", flexShrink: 0, marginTop: 1 }}>
                          {icons[entry.type]}
                        </span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <span style={{ color: c, fontWeight: 600 }}>{entry.topic}</span>
                          {" "}
                          <span style={{ color: "var(--txt-2)" }}>{entry.detail}</span>
                        </div>
                        <span style={{ fontSize: 10, color: "var(--txt-3)", flexShrink: 0, fontFamily: "var(--font-mono)" }}>
                          {entry.time}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* research a topic */}
              <div style={{ background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ color: "var(--cyan)" }}>⊞</span> Research a Topic
                </div>
                <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                  <input
                    value={researchInput}
                    onChange={e => setResearchInput(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") runResearch(researchInput); }}
                    placeholder="e.g. quantum computing overview"
                    disabled={researchRunning}
                    style={{
                      flex: 1, padding: "8px 10px", borderRadius: 8, fontSize: 13,
                      background: "var(--elevated)", border: "1px solid var(--border-dim)",
                      color: "var(--txt)", outline: "none",
                    }}
                  />
                  <button
                    onClick={() => runResearch(researchInput)}
                    disabled={researchRunning || !researchInput.trim()}
                    style={{
                      padding: "8px 16px", borderRadius: 8, fontSize: 12,
                      background: "var(--elevated)", border: "1px solid var(--border)",
                      color: "var(--cyan)", cursor: "pointer",
                      opacity: (researchRunning || !researchInput.trim()) ? 0.5 : 1,
                    }}
                  >
                    {researchRunning ? "Scraping…" : "Research"}
                  </button>
                </div>

                {/* heartbeat trigger */}
                <button
                  onClick={triggerHeartbeatNow}
                  disabled={researchRunning}
                  style={{
                    width: "100%", padding: "8px", borderRadius: 8, fontSize: 12,
                    background: researchRunning ? "var(--elevated)" : "rgba(0,217,126,.1)",
                    border: `1px solid ${researchRunning ? "var(--border-dim)" : "var(--green)"}`,
                    color: researchRunning ? "var(--txt-3)" : "var(--green)", cursor: "pointer",
                    marginBottom: 12,
                  }}
                >
                  ⬡ Trigger Heartbeat Now
                </button>

                {/* activity log */}
                <div style={{ fontSize: 10, color: "var(--txt-3)", marginBottom: 6 }}>Activity Log</div>
                <div style={{
                  background: "var(--void)", borderRadius: 8, padding: "8px 10px",
                  border: "1px solid var(--border-dim)", height: 140, overflowY: "auto",
                  fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.7,
                }}>
                  {researchLog.length === 0
                    ? <span style={{ color: "var(--txt-3)" }}>No activity yet…</span>
                    : researchLog.map((l, i) => (
                        <div key={i} style={{ color: l.includes("✓") ? "var(--green)" : l.includes("✗") ? "var(--red)" : "var(--txt-2)" }}>
                          {l}
                        </div>
                      ))
                  }
                </div>
              </div>

              {/* train on text */}
              <div style={{ background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ color: "var(--teal)" }}>⊕</span> Train on Text
                  <span style={{ fontSize: 11, color: "var(--txt-3)", fontWeight: 400 }}>one-shot learning</span>
                </div>
                <textarea
                  value={trainTextInput}
                  onChange={e => setTrainTextInput(e.target.value)}
                  placeholder="Paste any text to train the model on it instantly…"
                  rows={6}
                  style={{
                    width: "100%", padding: "10px 12px", borderRadius: 8, fontSize: 13,
                    background: "var(--elevated)", border: "1px solid var(--border-dim)",
                    color: "var(--txt)", outline: "none", resize: "vertical",
                    fontFamily: "var(--font-sans)", lineHeight: 1.6,
                    marginBottom: 10,
                  }}
                />
                <button
                  onClick={runTrainText}
                  disabled={trainTextLoading || !trainTextInput.trim()}
                  style={{
                    width: "100%", padding: "9px", borderRadius: 8, fontSize: 13,
                    background: "rgba(0,184,144,.1)", border: "1px solid var(--teal)",
                    color: "var(--teal)", cursor: "pointer",
                    opacity: (trainTextLoading || !trainTextInput.trim()) ? 0.5 : 1,
                  }}
                >
                  {trainTextLoading ? "Training…" : "Train Model"}
                </button>
                {trainTextResult && (
                  <div style={{
                    marginTop: 10, fontSize: 12, padding: "8px 10px", borderRadius: 7,
                    background: trainTextResult.startsWith("✓") ? "rgba(0,217,126,.08)" : "rgba(255,69,102,.08)",
                    border: `1px solid ${trainTextResult.startsWith("✓") ? "var(--green)" : "var(--red)"}`,
                    color: trainTextResult.startsWith("✓") ? "var(--green)" : "var(--red)",
                    fontFamily: "var(--font-mono)",
                  }}>
                    {trainTextResult}
                  </div>
                )}
              </div>

              {/* research topics */}
              <div style={{ gridColumn: "1 / -1", background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 12, padding: 18 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ color: "var(--amber)" }}>◎</span> Heartbeat Topics
                  </div>
                  <span style={{ fontSize: 11, color: "var(--txt-3)" }}>
                    rotated every {heartbeat?.topics?.length ?? 0} topics
                  </span>
                  <button
                    onClick={() => {
                      setTopicsEdit((heartbeat?.topics ?? []).join("\n"));
                      setTopicsEditMode(v => !v);
                    }}
                    style={{
                      marginLeft: "auto", padding: "4px 10px", borderRadius: 6, fontSize: 11,
                      background: "var(--elevated)", border: "1px solid var(--border-dim)",
                      color: "var(--txt-2)", cursor: "pointer",
                    }}
                  >
                    {topicsEditMode ? "Cancel" : "Edit"}
                  </button>
                  {topicsEditMode && (
                    <button
                      onClick={saveTopics}
                      style={{
                        padding: "4px 10px", borderRadius: 6, fontSize: 11,
                        background: "rgba(0,212,255,.1)", border: "1px solid var(--cyan)",
                        color: "var(--cyan)", cursor: "pointer",
                      }}
                    >
                      Save
                    </button>
                  )}
                </div>

                {topicsEditMode ? (
                  <textarea
                    value={topicsEdit}
                    onChange={e => setTopicsEdit(e.target.value)}
                    rows={8}
                    placeholder="One topic per line…"
                    style={{
                      width: "100%", padding: "10px 12px", borderRadius: 8, fontSize: 13,
                      background: "var(--elevated)", border: "1px solid var(--border)",
                      color: "var(--txt)", outline: "none", resize: "vertical",
                      fontFamily: "var(--font-sans)", lineHeight: 1.7,
                    }}
                  />
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {(heartbeat?.topics ?? []).map((t, i) => (
                      <div
                        key={i}
                        style={{
                          display: "flex", alignItems: "center", gap: 10,
                          padding: "8px 12px", borderRadius: 8,
                          background: i === (heartbeat?.next_topic_index ?? -1) ? "rgba(0,212,255,.06)" : "var(--elevated)",
                          border: `1px solid ${i === (heartbeat?.next_topic_index ?? -1) ? "var(--border)" : "var(--border-dim)"}`,
                        }}
                      >
                        <span style={{
                          width: 20, height: 20, borderRadius: 5, flexShrink: 0,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 10, fontFamily: "var(--font-mono)",
                          background: i === (heartbeat?.next_topic_index ?? -1) ? "var(--cyan)" : "var(--card)",
                          color: i === (heartbeat?.next_topic_index ?? -1) ? "#000" : "var(--txt-3)",
                        }}>
                          {i + 1}
                        </span>
                        <span style={{ fontSize: 13, color: i === (heartbeat?.next_topic_index ?? -1) ? "var(--txt)" : "var(--txt-2)" }}>
                          {t}
                        </span>
                        {i === (heartbeat?.next_topic_index ?? -1) && (
                          <span className="blink" style={{ marginLeft: "auto", fontSize: 10, color: "var(--cyan)" }}>NEXT</span>
                        )}
                        <button
                          onClick={() => runResearch(t)}
                          disabled={researchRunning}
                          style={{
                            marginLeft: i === (heartbeat?.next_topic_index ?? -1) ? 0 : "auto",
                            padding: "3px 8px", borderRadius: 5, fontSize: 10,
                            background: "transparent", border: "1px solid var(--border-dim)",
                            color: "var(--txt-3)", cursor: "pointer",
                          }}
                        >
                          Train
                        </button>
                      </div>
                    ))}
                    {(!heartbeat?.topics || heartbeat.topics.length === 0) && (
                      <div style={{ fontSize: 13, color: "var(--txt-3)", padding: 8 }}>
                        No topics set. Click Edit to add research topics.
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══════ GENERATE TAB ══════ */}
          {tab === "generate" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>

              <div style={{ background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 14, padding: 20 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: "var(--txt-2)", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ color: "var(--cyan)" }}>≋</span> Text Generation
                  <span style={{ fontSize: 12, fontWeight: 400, color: "var(--txt-3)" }}>
                    {fmt(stats?.vocab_size ?? 0)} vocab · {fmt(stats?.assoc_memory ?? 0)} associations
                  </span>
                </div>

                <label style={{ display: "block", fontSize: 12, color: "var(--txt-3)", marginBottom: 6 }}>Seed text</label>
                <textarea
                  value={genSeed}
                  onChange={e => setGenSeed(e.target.value)}
                  placeholder="The quick brown fox…"
                  rows={3}
                  style={{
                    width: "100%", padding: "10px 14px", borderRadius: 10, fontSize: 14,
                    background: "var(--elevated)", border: "1px solid var(--border-dim)",
                    color: "var(--txt)", outline: "none", resize: "vertical",
                    fontFamily: "var(--font-sans)", lineHeight: 1.6, marginBottom: 16,
                  }}
                />

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 18 }}>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--txt-3)", marginBottom: 8 }}>
                      <span>Temperature</span>
                      <span style={{ color: "var(--cyan)", fontFamily: "var(--font-mono)" }}>{genTemp.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min={0.1} max={2.0} step={0.05}
                      value={genTemp}
                      onChange={e => setGenTemp(parseFloat(e.target.value))}
                      style={{ width: "100%" }}
                    />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--txt-3)", marginTop: 4 }}>
                      <span>Focused</span>
                      <span>Creative</span>
                    </div>
                  </div>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--txt-3)", marginBottom: 8 }}>
                      <span>Max tokens</span>
                      <span style={{ color: "var(--amber)", fontFamily: "var(--font-mono)" }}>{genTokens}</span>
                    </div>
                    <input
                      type="range" min={10} max={200} step={5}
                      value={genTokens}
                      onChange={e => setGenTokens(parseInt(e.target.value))}
                      style={{ width: "100%" }}
                    />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--txt-3)", marginTop: 4 }}>
                      <span>10</span>
                      <span>200</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={runGenerate}
                  disabled={genLoading || !genSeed.trim() || !connected}
                  style={{
                    width: "100%", padding: "11px", borderRadius: 10, fontSize: 14,
                    background: "linear-gradient(135deg, rgba(0,212,255,.15), rgba(61,130,255,.15))",
                    border: "1px solid var(--border)",
                    color: "var(--cyan)", cursor: "pointer", fontWeight: 600,
                    opacity: (genLoading || !genSeed.trim()) ? 0.5 : 1,
                  }}
                >
                  {genLoading ? "Generating…" : "Generate"}
                </button>
              </div>

              {genResult && (
                <div className="fade-in" style={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14, padding: 20 }}>
                  <div style={{ fontSize: 12, color: "var(--txt-3)", marginBottom: 8 }}>Generated output</div>
                  <div style={{
                    fontSize: 15, lineHeight: 1.8, color: "var(--txt)",
                    background: "var(--elevated)", borderRadius: 10, padding: "14px 16px",
                    border: "1px solid var(--border-dim)", whiteSpace: "pre-wrap",
                  }}>
                    <span style={{ color: "var(--txt-3)" }}>{genResult.seed} </span>
                    <span style={{ color: "var(--cyan)" }}>{genResult.generated || "(no continuation yet — train more data)"}</span>
                  </div>
                </div>
              )}

              {/* suggestions */}
              <div style={{ background: "var(--card)", border: "1px solid var(--border-dim)", borderRadius: 14, padding: 20 }}>
                <div style={{ fontSize: 12, color: "var(--txt-3)", marginBottom: 12 }}>Try these seeds</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {[
                    "hyperdimensional computing uses",
                    "the binding operation connects",
                    "one-shot learning enables",
                    "the associative memory stores",
                    "vector symbolic architectures",
                    "brain inspired computing",
                  ].map(s => (
                    <button
                      key={s}
                      onClick={() => setGenSeed(s)}
                      style={{
                        padding: "6px 12px", borderRadius: 20, fontSize: 12,
                        background: "var(--elevated)", border: "1px solid var(--border-dim)",
                        color: "var(--txt-2)", cursor: "pointer", fontFamily: "var(--font-mono)",
                        transition: "border-color .15s",
                      }}
                      onMouseEnter={e => (e.currentTarget.style.borderColor = "var(--cyan)")}
                      onMouseLeave={e => (e.currentTarget.style.borderColor = "var(--border-dim)")}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
