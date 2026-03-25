"use client";

import {
  clearResearchMemory,
  fetchHeartbeatStatus,
  getAgentBaseUrl,
  setHeartbeatTopics,
  triggerResearch,
} from "@/lib/agent-api";
import { api } from "@/lib/convex-api";
import type { HeartbeatStatus } from "@/lib/types";
import { useQuery } from "convex/react";
import { useEffect, useMemo, useState } from "react";
import { ResearchTabInner } from "./AgentTester";

type Props = { baseUrl: string };

export default function ResearchTabLive({ baseUrl }: Props) {
  const cfg     = useQuery(api.researchConfig.get);
  const memRows = useQuery(api.memory.listAsc, { limit: 500 });
  const taskRows= useQuery(api.tasks.listRecent, { limit: 40 });

  const [intervalSeconds, setIntervalSeconds] = useState(300);
  useEffect(() => {
    const b = baseUrl || getAgentBaseUrl();
    void fetchHeartbeatStatus(b).then((h) => {
      if (h?.interval_seconds) setIntervalSeconds(h.interval_seconds);
    });
  }, [baseUrl]);

  const heartbeat: HeartbeatStatus | null = useMemo(() => {
    if (cfg === undefined) return null;
    const topics = cfg.topics ?? [];
    const idx    = cfg.cursorIndex ?? 0;
    return {
      last_run:          cfg.lastRun ?? null,
      total_runs:        Number(cfg.totalRuns ?? 0),
      topic_count:       topics.length,
      next_topic_index:  topics.length ? idx % topics.length : 0,
      next_topic:        topics.length ? topics[idx % topics.length] ?? null : null,
      topics,
      interval_seconds:  intervalSeconds,
    };
  }, [cfg, intervalSeconds]);

  // Build raw markdown string from Convex rows for the inner tab
  const memory = useMemo(() => {
    if (!memRows?.length) return "";
    return memRows.map((r: { title?: string; body?: string; createdAt?: number }) => {
      const created = r.createdAt;
      const dateStr = typeof created === "number"
        ? new Date(created).toLocaleString(undefined, { timeZone: "UTC" }) + " UTC"
        : "?";
      return `## ${r.title ?? ""}\n*${dateStr}*\n\n${r.body ?? ""}\n`;
    }).join("\n").trim();
  }, [memRows]);

  const [topicsEdit, setTopicsEdit]     = useState("");
  const [topicsSaving, setTopicsSaving] = useState(false);
  const [researchRunning, setResearchRunning] = useState(false);

  useEffect(() => {
    if (cfg !== undefined && cfg.topics) {
      setTopicsEdit(cfg.topics.join("\n"));
    }
  }, [cfg?.updatedAt]); // eslint-disable-line react-hooks/exhaustive-deps

  const isLoading = cfg === undefined || memRows === undefined;

  return (
    <div className="flex flex-col gap-5">
      {/* Convex tasks timeline */}
      {!isLoading && taskRows && taskRows.length > 0 && (
        <div className="rounded-2xl overflow-hidden"
          style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
          <div className="flex items-center gap-2 px-5 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
            <span style={{color:"var(--txt)"}}>⚙</span>
            <p className="font-bold text-sm" style={{color:"var(--txt)"}}>Task log</p>
            <span className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-bold blink"
              style={{background:"var(--teal)22",color:"var(--teal)",border:"1px solid var(--teal)44"}}>
              <span className="h-1.5 w-1.5 rounded-full" style={{background:"var(--teal)"}}/>LIVE
            </span>
          </div>
          <div className="overflow-y-auto" style={{maxHeight:"180px"}}>
            {(taskRows as Record<string, unknown>[]).map(t => {
              const status = String(t.status ?? "");
              const statusColor =
                status === "completed" ? "var(--teal)" :
                status === "running"   ? "var(--cyan)" :
                status === "failed"    ? "var(--red)"  : "var(--txt-3)";
              return (
                <div key={String(t._id)}
                  className="flex items-center gap-3 px-5 py-2.5"
                  style={{borderBottom:"1px solid var(--border-dim)"}}>
                  <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{background:statusColor}}/>
                  <span className="text-xs font-medium" style={{color:"var(--txt)"}}>{String(t.kind ?? "")}</span>
                  <span className="text-[10px]" style={{color:statusColor}}>{status}</span>
                  {t.detail != null && (
                    <span className="text-[10px]" style={{color:"var(--txt-3)"}}>{String(t.detail)}</span>
                  )}
                  {t.error != null && (
                    <span className="text-[10px] ml-1" style={{color:"var(--red)"}}>{String(t.error)}</span>
                  )}
                  {typeof t.updatedAt === "number" && (
                    <span className="ml-auto text-[9px]" style={{color:"var(--txt-3)"}}>
                      {new Date(t.updatedAt as number).toLocaleTimeString()}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="flex flex-col gap-4 py-4">
          {[1,2,3].map(i=>(
            <div key={i} className="skeleton h-20 rounded-2xl"/>
          ))}
        </div>
      ) : (
        <ResearchTabInner
          heartbeat={heartbeat}
          memory={memory}
          memoryLoading={false}
          topicsEdit={topicsEdit}
          setTopicsEdit={setTopicsEdit}
          topicsSaving={topicsSaving}
          researchRunning={researchRunning}
          baseUrl={baseUrl}
          isLive={true}
          onRunResearch={async () => {
            setResearchRunning(true);
            await triggerResearch(baseUrl || getAgentBaseUrl());
            await new Promise(r => setTimeout(r, 1500));
            setResearchRunning(false);
          }}
          onSaveTopics={async () => {
            setTopicsSaving(true);
            const topics = topicsEdit.split("\n").map(t => t.trim()).filter(Boolean);
            await setHeartbeatTopics(baseUrl || getAgentBaseUrl(), topics);
            setTopicsSaving(false);
          }}
          onClearMemory={async () => {
            if (!confirm("Clear all research memory? This cannot be undone.")) return;
            await clearResearchMemory(baseUrl || getAgentBaseUrl());
          }}
          onRefresh={async () => { /* Convex auto-refreshes via useQuery */ }}
        />
      )}
    </div>
  );
}
