"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const _convexUrl = process.env.NEXT_PUBLIC_CONVEX_URL?.trim();
const ResearchTabLive = _convexUrl
  ? dynamic(() => import("./ResearchTabLive"), { ssr: false })
  : null;
const DataBackground = dynamic(() => import("./DataBackground"), { ssr: false });

import {
  clearResearchMemory,
  fetchBlueprint,
  fetchCodebaseSnapshot,
  fetchHealth,
  fetchHeartbeatStatus,
  fetchImprovements,
  fetchResearchMemory,
  fetchSicaSummary,
  getAgentBaseUrl,
  getAgentJob,
  getOrCreateSessionId,
  postChat,
  refreshCodebase,
  requestFullStackImprovement,
  requestImprovement,
  resetSessionId,
  setHeartbeatTopics,
  startAgentJob,
  triggerResearch,
} from "@/lib/agent-api";
import type {
  BlueprintSnapshot,
  ChatMessage,
  ChatTurnResult,
  FullStackImproveResult,
  HeartbeatStatus,
  ImprovementHistory,
  ImproveResult,
} from "@/lib/types";

// ─── storage ─────────────────────────────────────────────────────────────────
const STORAGE_MESSAGES = "hlagent-messages";
const STORAGE_API      = "hlagent-api-base";
const STORAGE_AUTO     = "hlagent-auto-improve";

const EMPTY_TURN: ChatTurnResult = {
  route:"error", intent:"", answer:"", sympy:null,
  hdc_similarity:null, hdc_matched_task:null, context_snippet:"",
  grounded:false, session_id:null, metadata:{},
};

function ls<T>(key:string, fb:T):T {
  if (typeof window==="undefined") return fb;
  try { const v=localStorage.getItem(key); return v!=null?(JSON.parse(v) as T):fb; } catch { return fb; }
}
function lsSet(key:string, v:unknown) {
  try { localStorage.setItem(key,JSON.stringify(v)); } catch {/**/}
}
function loadMessages():ChatMessage[] {
  const p=ls<ChatMessage[]>(STORAGE_MESSAGES,[]);
  return Array.isArray(p)?p:[];
}
function saveMessages(msgs:ChatMessage[]) { lsSet(STORAGE_MESSAGES,msgs.slice(-200)); }
function fmtTime(at:number){ return new Date(at).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}); }
function fmtTs(iso:string){ try{return new Date(iso).toLocaleString();}catch{return iso;} }
function fmtRelative(iso:string|null|undefined):string {
  if(!iso) return "never";
  const diff=Date.now()-new Date(iso).getTime();
  if(diff<60_000) return "just now";
  if(diff<3_600_000) return `${Math.floor(diff/60_000)}m ago`;
  if(diff<86_400_000) return `${Math.floor(diff/3_600_000)}h ago`;
  return new Date(iso).toLocaleDateString();
}

// ─── markdown renderer ────────────────────────────────────────────────────────
type Seg = {type:"code";lang:string;text:string}|{type:"text";text:string};
function parseSegs(raw:string):Seg[] {
  const segs:Seg[]=[];
  const fence=/^```(\w*)\n([\s\S]*?)^```/gm;
  let last=0, m;
  while((m=fence.exec(raw))!==null){
    if(m.index>last) segs.push({type:"text",text:raw.slice(last,m.index)});
    segs.push({type:"code",lang:m[1]||"",text:m[2]});
    last=m.index+m[0].length;
  }
  if(last<raw.length) segs.push({type:"text",text:raw.slice(last)});
  return segs;
}
function renderText(t:string){
  return t.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>")
    .replace(/\*(.+?)\*/g,"<em>$1</em>")
    .replace(/`([^`]+)`/g,'<code class="inline-code">$1</code>');
}
function MarkdownMessage({text}:{text:string}){
  const segs=useMemo(()=>parseSegs(text),[text]);
  return (
    <div className="msg-content text-sm leading-relaxed">
      {segs.map((s,i)=>
        s.type==="code"
          ? <pre key={i} className="my-2 overflow-x-auto rounded-lg p-3 font-mono text-[11px] leading-relaxed" style={{background:"var(--void)",color:"#a8e8b8",border:"1px solid var(--border-dim)"}}><code>{s.text}</code></pre>
          : <span key={i} dangerouslySetInnerHTML={{__html:renderText(s.text).replace(/\n/g,"<br/>")}} />
      )}
    </div>
  );
}

// ─── micro components ──────────────────────────────────────────────────────────

function StatusDot({status}:{status:"ok"|"down"|"checking"}){
  const color=status==="ok"?"var(--green)":status==="down"?"var(--red)":"var(--amber)";
  const cls=status==="ok"?"dot-live":status==="checking"?"dot-amber":"";
  return (
    <span className={`inline-block h-2 w-2 rounded-full shrink-0 ${cls}`}
      style={{background:color}} />
  );
}

function Pill({label,color="blue"}:{label:string;color?:"blue"|"green"|"cyan"|"amber"|"red"|"violet"|"teal"}){
  const map={blue:"var(--blue)",green:"var(--green)",cyan:"var(--cyan)",amber:"var(--amber)",red:"var(--red)",violet:"var(--violet)",teal:"var(--teal)"};
  const c=map[color];
  return (
    <span className="rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wide"
      style={{background:`${c}22`,color:c,border:`1px solid ${c}44`}}>
      {label}
    </span>
  );
}

function NavItem({id,label,icon,active,badge,onClick}:{
  id:string;label:string;icon:string;active:boolean;badge?:number;onClick:()=>void;
}){
  return (
    <button type="button" onClick={onClick}
      className="relative flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-all"
      style={{
        background:active?"var(--elevated)":"transparent",
        color:active?"var(--cyan)":"var(--txt-2)",
        borderLeft:active?"2px solid var(--cyan)":"2px solid transparent",
      }}>
      <span className="text-base leading-none">{icon}</span>
      <span>{label}</span>
      {badge!=null && badge>0 && (
        <span className="ml-auto rounded-full px-1.5 py-0.5 text-[9px] font-bold"
          style={{background:"var(--blue)",color:"#fff"}}>
          {badge}
        </span>
      )}
    </button>
  );
}

function MobileNavItem({id,label,icon,active,badge,onClick}:{
  id:string;label:string;icon:string;active:boolean;badge?:number;onClick:()=>void;
}){
  return (
    <button type="button" onClick={onClick}
      className="relative flex flex-1 flex-col items-center justify-center gap-0.5 py-2 transition-all"
      style={{color:active?"var(--cyan)":"var(--txt-3)"}}>
      <span className="text-lg leading-none">{icon}</span>
      <span className="text-[9px] font-medium">{label}</span>
      {badge!=null && badge>0 && (
        <span className="absolute right-2 top-1 rounded-full px-1 text-[8px] font-bold"
          style={{background:"var(--blue)",color:"#fff"}}>
          {badge}
        </span>
      )}
      {active && (
        <span className="absolute bottom-0 left-1/2 h-0.5 w-4 -translate-x-1/2 rounded-full"
          style={{background:"var(--cyan)"}} />
      )}
    </button>
  );
}

function ThinkingAnim({stage}:{stage:string}){
  return (
    <div className="flex items-center gap-3 rounded-2xl px-4 py-3 slide-up"
      style={{background:"var(--card)",border:"1px solid var(--border-dim)"}}>
      <span className="flex gap-1">
        {[0,1,2].map(i=>(
          <span key={i} className="neural-dot h-1.5 w-1.5 rounded-full"
            style={{background:"var(--cyan)"}} />
        ))}
      </span>
      <span className="text-xs" style={{color:"var(--txt-2)"}}>{stage}…</span>
    </div>
  );
}

function InlineImproveCard({r}:{r:ImproveResult}){
  const changes=r.file_changes?.length
    ? r.file_changes
    : [{file:r.target_file,new_code:r.new_code,error:r.error,committed:r.committed,commit_hash:r.commit_hash,reason:"",ast_ok:r.ast_ok}];
  return (
    <div className="mt-2 rounded-xl overflow-hidden slide-up"
      style={{border:`1px solid ${r.ok?"var(--teal)44":"var(--red)44"}`,background:r.ok?"rgba(0,184,144,0.05)":"rgba(255,69,102,0.05)"}}>
      <div className="flex items-center gap-2 px-3 py-2 text-xs font-semibold"
        style={{borderBottom:"1px solid var(--border-dim)",color:r.ok?"var(--teal)":"var(--red)"}}>
        <span>{r.ok?"⟳ Applied":"✗ Failed"}</span>
        <span className="ml-auto font-mono text-[9px]" style={{color:"var(--txt-3)"}}>
          {fmtTs(r.timestamp)}
        </span>
      </div>
      {changes.map((fc,i)=>(
        <div key={i} className="px-3 py-2 text-[11px]" style={{borderBottom:"1px solid var(--border-dim)"}}>
          <div className="flex flex-wrap items-center gap-2">
            <span style={{color:fc.error?"var(--red)":"var(--teal)"}}>{fc.error?"✗":"✓"}</span>
            <code className="font-mono" style={{color:"var(--txt-2)"}}>{fc.file}</code>
            {fc.committed && <Pill label={(fc.commit_hash as string|null)?.slice(0,8)||"committed"} color="blue"/>}
          </div>
          {fc.error && <p className="mt-1" style={{color:"var(--red)"}}>{fc.error}</p>}
          {fc.new_code && !fc.error && (
            <details className="mt-1.5">
              <summary className="cursor-pointer text-[10px]" style={{color:"var(--txt-3)"}}>
                Show code ({(fc.new_code as string).split("\n").length} lines)
              </summary>
              <pre className="mt-1 max-h-40 overflow-auto rounded-lg p-2 font-mono text-[10px]"
                style={{background:"var(--void)",color:"#a8e8b8"}}>
                {fc.new_code as string}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}

function InlineFullStackCard({r}:{r:FullStackImproveResult}){
  return (
    <div className="mt-2 rounded-xl overflow-hidden slide-up"
      style={{border:`1px solid ${r.ok?"var(--violet)44":"var(--red)44"}`,background:r.ok?"rgba(139,92,246,0.05)":"rgba(255,69,102,0.05)"}}>
      <div className="flex items-center gap-2 px-3 py-2 text-xs font-semibold"
        style={{borderBottom:"1px solid var(--border-dim)",color:r.ok?"var(--violet)":"var(--red)"}}>
        <span>{r.ok?"⚡ Full-stack applied":"✗ Failed"}</span>
        <span className="ml-auto font-mono text-[9px]" style={{color:"var(--txt-3)"}}>{fmtTs(r.timestamp)}</span>
      </div>
      {([
        {key:"backend" as const,label:"Backend",icon:"🐍"},
        {key:"frontend_api" as const,label:"API client",icon:"🔌"},
        {key:"frontend_ui" as const,label:"UI",icon:"🖥"},
      ] as const).map(({key,label,icon})=>{
        const sub=r[key]; if(!sub) return null;
        return (
          <div key={key} className="flex items-center gap-2 px-3 py-2 text-[11px]"
            style={{borderBottom:"1px solid var(--border-dim)"}}>
            <span>{icon}</span>
            <code className="font-mono" style={{color:"var(--txt-2)"}}>{sub.target_file}</code>
            <span className="ml-auto" style={{color:sub.ok?"var(--teal)":"var(--red)"}}>{sub.ok?"✓":"✗"}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── suggestion chips ──────────────────────────────────────────────────────────
const SUGGESTIONS=[
  "What's your current architecture?","What AI news is happening today?",
  "Research the latest on Gemini 2.5 capabilities","What is 2 + 2 using SymPy?",
  "Improve the HDC memory retrieval","Add a /api/v1/sessions list endpoint",
];

type Tab="chat"|"research"|"improve"|"codebase"|"status";
type ThinkingStage="thinking"|"searching"|"writing code"|"applying";

// ─── main component ─────────────────────────────────────────────────────────────
export default function AgentTester(){
  const [baseUrl,setBaseUrl]=useState("");
  const [tab,setTab]=useState<Tab>("chat");

  const [health,setHealth]=useState<{status:"ok"|"down"|"checking";gemini:boolean;convex:boolean;email:boolean;hint?:string}>
    ({status:"checking",gemini:false,convex:false,email:false});

  const [sessionId,setSessionId]=useState("");
  const [input,setInput]=useState("");
  const [messages,setMessages]=useState<ChatMessage[]>([]);
  const [chatLoading,setChatLoading]=useState(false);
  const [thinkingStage,setThinkingStage]=useState<ThinkingStage>("thinking");
  const [mode,setMode]=useState<"sync"|"async">("sync");
  const [autoImprove,setAutoImprove]=useState(false);
  const messagesEndRef=useRef<HTMLDivElement>(null);

  const [codebase,setCodebase]=useState("");
  const [codebaseLoading,setCodebaseLoading]=useState(false);

  const [instruction,setInstruction]=useState("");
  const [targetFile,setTargetFile]=useState("");
  const [fullStack,setFullStack]=useState(false);
  const [improving,setImproving]=useState(false);
  const [improveResult,setImproveResult]=useState<ImproveResult|null>(null);
  const [fullStackResult,setFullStackResult]=useState<FullStackImproveResult|null>(null);
  const [improveHistory,setImproveHistory]=useState<ImprovementHistory>({entries:[]});

  const [blueprint,setBlueprint]=useState<BlueprintSnapshot|null>(null);
  const [sica,setSica]=useState("");

  const [memory,setMemory]=useState("");
  const [memoryLoading,setMemoryLoading]=useState(false);
  const [heartbeat,setHeartbeat]=useState<HeartbeatStatus|null>(null);
  const [topicsEdit,setTopicsEdit]=useState("");
  const [topicsSaving,setTopicsSaving]=useState(false);
  const [researchRunning,setResearchRunning]=useState(false);
  const [sidebarOpen,setSidebarOpen]=useState(false);

  // ── init ────────────────────────────────────────────────────────────────────
  useEffect(()=>{
    setBaseUrl(getAgentBaseUrl());
    setSessionId(getOrCreateSessionId());
    setMessages(loadMessages());
    setAutoImprove(ls(STORAGE_AUTO,false));
  },[]);

  useEffect(()=>{
    messagesEndRef.current?.scrollIntoView({behavior:"smooth"});
  },[messages,chatLoading]);

  // ── ping ────────────────────────────────────────────────────────────────────
  const ping=useCallback(async()=>{
    const b=(baseUrl||getAgentBaseUrl()).trim();
    setHealth(h=>({...h,status:"checking",hint:undefined}));
    const h=await fetchHealth(b);
    setHealth({
      status:h.ok?"ok":"down",
      gemini:h.gemini_configured??false,
      convex:h.convex_configured??false,
      email:h.email_notifications_ready??false,
      hint:h.ok?undefined:h.detail,
    });
  },[baseUrl]);

  useEffect(()=>{
    void ping();
    const t=setInterval(()=>void ping(),20_000);
    return ()=>clearInterval(t);
  },[baseUrl,ping]);

  // ── tab loaders ─────────────────────────────────────────────────────────────
  const loadCodebase=useCallback(async()=>{
    setCodebaseLoading(true);
    const txt=await fetchCodebaseSnapshot(baseUrl||getAgentBaseUrl());
    setCodebase(txt);setCodebaseLoading(false);
  },[baseUrl]);

  const loadStatus=useCallback(async()=>{
    const b=baseUrl||getAgentBaseUrl();
    const [bp,sc]=await Promise.all([fetchBlueprint(b),fetchSicaSummary(b)]);
    setBlueprint(bp);setSica(sc);
  },[baseUrl]);

  const loadImprovements=useCallback(async()=>{
    const hist=await fetchImprovements(baseUrl||getAgentBaseUrl());
    setImproveHistory(hist);
  },[baseUrl]);

  const loadResearch=useCallback(async()=>{
    const b=baseUrl||getAgentBaseUrl();
    setMemoryLoading(true);
    const [mem,hb]=await Promise.all([fetchResearchMemory(b),fetchHeartbeatStatus(b)]);
    setMemory(mem);setHeartbeat(hb);
    if(hb?.topics) setTopicsEdit(hb.topics.join("\n"));
    setMemoryLoading(false);
  },[baseUrl]);

  useEffect(()=>{
    if(!baseUrl) return;
    if(tab==="codebase") void loadCodebase();
    if(tab==="status") void loadStatus();
    if(tab==="improve") void loadImprovements();
    if(tab==="research") void loadResearch();
  },[tab,baseUrl,loadCodebase,loadStatus,loadImprovements,loadResearch]);

  // ── helpers ─────────────────────────────────────────────────────────────────
  const persistBase=(next:string)=>{
    const v=next.trim().replace(/\/$/,"");
    setBaseUrl(v);
    try{ if(v) localStorage.setItem(STORAGE_API,v); else localStorage.removeItem(STORAGE_API); }catch{/**/}
  };

  const newSession=()=>{
    const id=resetSessionId();
    setSessionId(id);setMessages([]);
    try{localStorage.removeItem(STORAGE_MESSAGES);}catch{/**/}
  };

  const emailNote=(turn:ChatTurnResult):string=>{
    const m=turn.metadata;
    if(!m?.user_requested_email) return "";
    if(m.email_notification_sent) return "\n\n---\n*📧 Email copy sent to your inbox.*";
    return "\n\n---\n*📧 Email requested but not sent — check SMTP config.*";
  };

  const appendAssistant=(turn:ChatTurnResult,err?:string)=>{
    const msg:ChatMessage={id:crypto.randomUUID(),role:"assistant",
      content:err??(turn.answer+emailNote(turn)),turn,error:err,at:Date.now()};
    setMessages(prev=>{const n=[...prev,msg];saveMessages(n);return n;});
  };
  const appendUser=(text:string)=>{
    const msg:ChatMessage={id:crypto.randomUUID(),role:"user",content:text,at:Date.now()};
    setMessages(prev=>{const n=[...prev,msg];saveMessages(n);return n;});
  };

  const guessStage=(text:string):ThinkingStage=>{
    if(/\b(research|current|today|news|latest|2026|liverpool|salah|search)\b/i.test(text)) return "searching";
    if(/\b(add|implement|create|fix|refactor|update|improve)\b/i.test(text)&&autoImprove) return "writing code";
    return "thinking";
  };

  // ── send ────────────────────────────────────────────────────────────────────
  const sendSync=async(prefill?:string)=>{
    const text=(prefill??input).trim();
    if(!text||chatLoading) return;
    const b=baseUrl||getAgentBaseUrl();
    appendUser(text);
    if(!prefill) setInput("");
    setChatLoading(true);setThinkingStage(guessStage(text));
    try{ const turn=await postChat(b,text,sessionId||null,autoImprove); appendAssistant(turn); }
    catch(e){ appendAssistant({...EMPTY_TURN},e instanceof Error?e.message:"Request failed"); }
    finally{ setChatLoading(false); }
  };

  const sendAsync=async(prefill?:string)=>{
    const text=(prefill??input).trim();
    if(!text||chatLoading) return;
    const b=baseUrl||getAgentBaseUrl();
    appendUser(`[async] ${text}`);
    if(!prefill) setInput("");
    setChatLoading(true);setThinkingStage(guessStage(text));
    try{
      const {job_id}=await startAgentJob(b,text,sessionId||null,autoImprove);
      let result:ChatTurnResult|null=null;
      for(let i=0;i<300;i++){
        const job=await getAgentJob(b,job_id);result=job.result;
        if(result) break;
        if(i>10&&i%5===0) setThinkingStage(p=>p==="thinking"?"searching":p==="searching"?"writing code":"thinking");
        await new Promise(r=>setTimeout(r,100));
      }
      if(result) appendAssistant(result);
      else appendAssistant({...EMPTY_TURN,answer:"Job did not complete in time."});
    }catch(e){ appendAssistant({...EMPTY_TURN},e instanceof Error?e.message:"Async job failed"); }
    finally{ setChatLoading(false); }
  };

  const send=(prefill?:string)=>void(mode==="sync"?sendSync(prefill):sendAsync(prefill));

  // ── improve ─────────────────────────────────────────────────────────────────
  const submitImprovement=async()=>{
    const instr=instruction.trim();
    if(!instr||improving) return;
    const b=baseUrl||getAgentBaseUrl();
    setImproving(true);setImproveResult(null);setFullStackResult(null);
    try{
      if(fullStack){ const r=await requestFullStackImprovement(b,instr,targetFile.trim()||undefined); setFullStackResult(r); void loadImprovements(); }
      else{ const r=await requestImprovement(b,instr,targetFile.trim()||undefined); setImproveResult(r); void loadImprovements(); }
    }catch(e){
      const errMsg=e instanceof Error?e.message:"Request failed";
      if(fullStack) setFullStackResult({ok:false,instruction:instr,
        backend:{ok:false,target_file:"?",instruction:instr,old_code:"",new_code:"",ast_ok:false,committed:false,commit_hash:null,error:errMsg,timestamp:new Date().toISOString(),file_changes:[]},
        frontend_api:null,frontend_ui:null,timestamp:new Date().toISOString()});
      else setImproveResult({ok:false,target_file:targetFile||"?",instruction:instr,old_code:"",new_code:"",ast_ok:false,committed:false,commit_hash:null,error:errMsg,timestamp:new Date().toISOString(),file_changes:[]});
    }finally{ setImproving(false); }
  };

  const gaps=useMemo(()=>blueprint?.items.filter(i=>i.status!=="done")??[],[blueprint]);
  const effectiveApiBase=useMemo(()=>(baseUrl||getAgentBaseUrl()).trim(),[baseUrl]);

  const TABS:{id:Tab;label:string;icon:string;badge?:number}[]=[
    {id:"chat",    label:"Chat",     icon:"◎"},
    {id:"research",label:"Research", icon:"⬡"},
    {id:"improve", label:"Improve",  icon:"⚡",badge:improveHistory.entries.length||undefined},
    {id:"codebase",label:"Codebase", icon:"◫"},
    {id:"status",  label:"Status",   icon:"◈",badge:gaps.length||undefined},
  ];

  // ─── render ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen overflow-hidden" style={{background:"var(--void)"}}>
      <DataBackground />

      {/* everything above the canvas gets relative z-index */}
      {/* ── SIDEBAR (desktop) ─────────────────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-64 shrink-0 overflow-y-auto"
        style={{background:"var(--surface)",borderRight:"1px solid var(--border-dim)",position:"relative",zIndex:1}}>
        {/* logo */}
        <div className="flex items-center gap-3 px-5 py-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl relative"
            style={{background:"linear-gradient(135deg,var(--blue),var(--cyan))"}}>
            <span className="font-mono text-[13px] font-bold text-black">SA</span>
          </div>
          <div>
            <h1 className="text-sm font-bold" style={{color:"var(--txt)"}}>Super Agent</h1>
            <p className="text-[10px]" style={{color:"var(--txt-3)"}}>Neuro-symbolic · self-evolving</p>
          </div>
        </div>

        {/* status cluster */}
        <div className="mx-4 mb-4 rounded-xl p-3" style={{background:"var(--card)",border:"1px solid var(--border-dim)"}}>
          <div className="flex items-center gap-2 mb-2">
            <StatusDot status={health.status}/>
            <span className="text-xs font-semibold" style={{color:health.status==="ok"?"var(--green)":health.status==="down"?"var(--red)":"var(--amber)"}}>
              {health.status==="ok"?"Online":health.status==="down"?"Offline":"Checking"}
            </span>
            <button onClick={()=>void ping()} className="ml-auto text-xs transition hover:opacity-70" style={{color:"var(--txt-3)"}}>↺</button>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {[
              {k:"gemini",label:"Gemini",v:health.gemini},
              {k:"convex",label:"Convex",v:health.convex},
              {k:"email", label:"Email", v:health.email},
            ].map(({k,label,v})=>(
              <div key={k} className="flex flex-col items-center rounded-lg py-1.5" style={{background:"var(--elevated)"}}>
                <span className="text-[9px] font-semibold uppercase" style={{color:v?"var(--teal)":"var(--txt-3)"}}>{label}</span>
                <span className="text-[10px]" style={{color:v?"var(--green)":"var(--txt-3)"}}>{v?"✓":"—"}</span>
              </div>
            ))}
          </div>
          {health.hint && (
            <p className="mt-2 text-[10px] leading-snug" style={{color:"var(--amber)"}}>{health.hint.slice(0,200)}</p>
          )}
        </div>

        {/* nav */}
        <nav className="flex flex-col gap-0.5 px-3 mb-4">
          {TABS.map(t=>(
            <NavItem key={t.id} id={t.id} label={t.label} icon={t.icon}
              active={tab===t.id} badge={t.badge} onClick={()=>setTab(t.id)}/>
          ))}
        </nav>

        {/* session */}
        <div className="mx-4 mb-4 rounded-xl p-3" style={{background:"var(--card)",border:"1px solid var(--border-dim)"}}>
          <p className="mb-1 text-[9px] font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>Session</p>
          <p className="font-mono text-[10px] truncate" style={{color:"var(--txt-2)"}}>{sessionId.slice(0,24)}…</p>
          <p className="mt-0.5 text-[10px]" style={{color:"var(--txt-3)"}}>{messages.length} messages</p>
          <button onClick={newSession}
            className="mt-2 w-full rounded-lg py-1.5 text-[11px] font-medium transition hover:opacity-80"
            style={{background:"var(--elevated)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}>
            + New session
          </button>
        </div>

        {/* API URL */}
        <div className="mx-4 mb-4">
          <p className="mb-1 text-[9px] font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>API URL</p>
          <input
            className="w-full rounded-lg px-2.5 py-1.5 font-mono text-[10px] outline-none transition"
            style={{background:"var(--elevated)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}
            value={baseUrl} onChange={e=>persistBase(e.target.value)}
            placeholder="http://127.0.0.1:8000" spellCheck={false}/>
        </div>

        {/* auto-improve */}
        <div className="mx-4 mb-4 rounded-xl p-3" style={{background:"var(--card)",border:`1px solid ${autoImprove?"var(--violet)44":"var(--border-dim)"}`}}>
          <div className="flex items-center gap-2">
            <button type="button" role="switch" aria-checked={autoImprove}
              onClick={()=>{setAutoImprove(p=>{lsSet(STORAGE_AUTO,!p);return !p;})}}
              className="relative inline-flex h-5 w-9 items-center rounded-full transition-colors shrink-0"
              style={{background:autoImprove?"var(--violet)":"var(--elevated)"}}>
              <span className="inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform"
                style={{transform:autoImprove?"translateX(18px)":"translateX(2px)"}}/>
            </button>
            <span className="text-[11px] font-semibold" style={{color:autoImprove?"var(--violet)":"var(--txt-2)"}}>Auto-improve</span>
          </div>
          {autoImprove && (
            <p className="mt-1.5 text-[10px] leading-relaxed" style={{color:"var(--txt-3)"}}>
              Code change requests auto-run the SICA pipeline.
            </p>
          )}
        </div>

        {/* stats mini */}
        {heartbeat && tab!=="research" && (
          <div className="mx-4 mb-4">
            <p className="mb-1 text-[9px] font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>Research</p>
            <div className="rounded-xl p-2.5" style={{background:"var(--card)",border:"1px solid var(--border-dim)"}}>
              <div className="flex justify-between text-[10px]">
                <span style={{color:"var(--txt-3)"}}>Runs</span>
                <span style={{color:"var(--cyan)"}}>{heartbeat.total_runs}</span>
              </div>
              <div className="flex justify-between text-[10px]">
                <span style={{color:"var(--txt-3)"}}>Topics</span>
                <span style={{color:"var(--cyan)"}}>{heartbeat.topic_count}</span>
              </div>
              <div className="flex justify-between text-[10px]">
                <span style={{color:"var(--txt-3)"}}>Last</span>
                <span style={{color:"var(--txt-2)"}}>{fmtRelative(heartbeat.last_run)}</span>
              </div>
            </div>
          </div>
        )}

        <div className="flex-1"/>
        <div className="mx-4 mb-4 text-[9px]" style={{color:"var(--txt-3)"}}>
          FastAPI · Gemini · HDC · SymPy · Convex
        </div>
      </aside>

      {/* ── MAIN CONTENT ──────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden" style={{position:"relative",zIndex:1}}>

        {/* mobile header */}
        <header className="flex lg:hidden items-center gap-3 px-4 py-3 shrink-0"
          style={{borderBottom:"1px solid var(--border-dim)",background:"var(--surface)"}}>
          <div className="flex h-8 w-8 items-center justify-center rounded-xl"
            style={{background:"linear-gradient(135deg,var(--blue),var(--cyan))"}}>
            <span className="font-mono text-[11px] font-bold text-black">SA</span>
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-sm font-bold truncate" style={{color:"var(--txt)"}}>Super Agent</h1>
          </div>
          <StatusDot status={health.status}/>
          <span className="text-[9px] font-semibold" style={{color:health.status==="ok"?"var(--green)":"var(--amber)"}}>
            {health.status==="ok"?"Online":"Checking"}
          </span>
        </header>

        {/* tab content */}
        <main className="flex-1 overflow-y-auto pb-20 lg:pb-0">
          <div className="mx-auto max-w-4xl px-4 py-4 lg:px-6 lg:py-6">

            {/* desktop tab strip */}
            <div className="mb-5 hidden lg:flex gap-1 rounded-xl p-1"
              style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
              {TABS.map(t=>(
                <button key={t.id} type="button" onClick={()=>setTab(t.id)}
                  className="relative flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all"
                  style={{
                    background:tab===t.id?"var(--elevated)":"transparent",
                    color:tab===t.id?"var(--cyan)":"var(--txt-3)",
                    border:tab===t.id?"1px solid var(--border)":"1px solid transparent",
                  }}>
                  <span className="text-sm">{t.icon}</span>
                  <span>{t.label}</span>
                  {t.badge!=null && t.badge>0 && (
                    <span className="ml-1 rounded-full px-1.5 py-0.5 text-[9px] font-bold"
                      style={{background:"var(--blue)",color:"#fff"}}>{t.badge}</span>
                  )}
                </button>
              ))}
            </div>

            {/* ──── CHAT TAB ──────────────────────────────────────────────── */}
            {tab==="chat" && (
              <div className="flex flex-col gap-3">
                {/* mode bar */}
                <div className="flex flex-wrap items-center gap-3 rounded-xl px-4 py-2.5"
                  style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
                  <div className="flex items-center gap-0.5 rounded-lg p-0.5"
                    style={{background:"var(--elevated)"}}>
                    {(["sync","async"] as const).map(m=>(
                      <button key={m} type="button" onClick={()=>setMode(m)}
                        className="rounded-md px-3 py-1 text-[11px] font-semibold transition-all"
                        style={{background:mode===m?"var(--card)":"transparent",color:mode===m?"var(--cyan)":"var(--txt-3)",
                          border:mode===m?"1px solid var(--border)":"1px solid transparent"}}>
                        {m==="sync"?"Sync":"Async"}
                      </button>
                    ))}
                  </div>
                  <span className="text-[10px]" style={{color:"var(--txt-3)"}}>
                    {mode==="sync"?"POST /chat — blocks until reply":"POST /agent/start — polls job"}
                  </span>
                  {autoImprove && (
                    <span className="ml-auto rounded-lg px-2 py-0.5 text-[10px] font-semibold"
                      style={{background:"var(--violet)22",color:"var(--violet)",border:"1px solid var(--violet)44"}}>
                      ⚡ auto-improve ON
                    </span>
                  )}
                </div>

                {/* messages */}
                <div className="flex flex-col rounded-2xl overflow-hidden"
                  style={{background:"var(--surface)",border:"1px solid var(--border-dim)",minHeight:"420px"}}>
                  <div className="flex-1 overflow-y-auto px-4 py-4" style={{maxHeight:"min(58vh,580px)"}}>
                    {messages.length===0 && (
                      <div className="py-8 fade-in">
                        <div className="mb-2 flex items-center gap-2">
                          <span className="h-px flex-1" style={{background:"var(--border-dim)"}}/>
                          <span className="text-[10px] font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>
                            Neural Command Interface
                          </span>
                          <span className="h-px flex-1" style={{background:"var(--border-dim)"}}/>
                        </div>
                        <p className="mb-5 text-sm leading-relaxed" style={{color:"var(--txt-3)"}}>
                          The agent knows its own codebase, remembers this session, performs live web search for current events,
                          and can modify its own code when Auto-improve is on.
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {SUGGESTIONS.map(s=>(
                            <button key={s} type="button" onClick={()=>send(s)}
                              className="rounded-xl px-3 py-2 text-xs font-medium transition-all hover:scale-[1.02]"
                              style={{background:"var(--card)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}>
                              {s}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {messages.map(m=>{
                      const improveData=m.turn?.metadata?.improve_result as ImproveResult|undefined;
                      const fsData=m.turn?.metadata?.fullstack_result as FullStackImproveResult|undefined;
                      const isUser=m.role==="user";
                      return (
                        <div key={m.id} className={`mb-4 flex slide-up ${isUser?"justify-end":"justify-start"}`}>
                          <div className={`max-w-[88%] flex flex-col ${isUser?"items-end":"items-start"}`}>
                            <div className="rounded-2xl px-4 py-3" style={isUser
                              ? {background:"linear-gradient(135deg,var(--blue),#1a5fff)",color:"#fff"}
                              : {background:"var(--card)",border:"1px solid var(--border-dim)",color:"var(--txt)"}}>
                              {m.role==="assistant"&&!m.error
                                ? <MarkdownMessage text={m.content}/>
                                : <p className="text-sm leading-relaxed">{m.content}</p>}
                              {m.error && <p className="mt-1 text-xs" style={{color:"var(--red)"}}>{m.error}</p>}
                              {m.turn&&m.role==="assistant"&&!m.error && (
                                <div className="mt-2.5 flex flex-wrap items-center gap-1.5 pt-2 font-mono text-[9px]"
                                  style={{borderTop:"1px solid var(--border-dim)"}}>
                                  <Pill label={m.turn.intent||m.turn.route}
                                    color={m.turn.intent==="improve"?"violet":m.turn.route==="symbolic"?"amber":"blue"}/>
                                  {m.turn.hdc_similarity!=null && (
                                    <span style={{color:"var(--txt-3)"}}>hdc={m.turn.hdc_similarity.toFixed(3)}</span>
                                  )}
                                  {m.turn.grounded && <Pill label="🔍 searched" color="cyan"/>}
                                  {!!m.turn.metadata?.email_notification_sent && <Pill label="📧 emailed" color="teal"/>}
                                  <span className="ml-auto" style={{color:"var(--txt-3)"}}>{fmtTime(m.at)}</span>
                                </div>
                              )}
                            </div>
                            {fsData && <InlineFullStackCard r={fsData}/>}
                            {!fsData && improveData && <InlineImproveCard r={improveData}/>}
                          </div>
                        </div>
                      );
                    })}

                    {chatLoading && (
                      <div className="mb-4 flex justify-start">
                        <ThinkingAnim stage={thinkingStage}/>
                      </div>
                    )}
                    <div ref={messagesEndRef}/>
                  </div>

                  {/* input */}
                  <div className="p-3" style={{borderTop:"1px solid var(--border-dim)"}}>
                    <div className="relative rounded-xl overflow-hidden" style={{border:"1px solid var(--border)"}}>
                      <textarea
                        className="w-full resize-none px-4 py-3 text-sm outline-none"
                        style={{background:"var(--elevated)",color:"var(--txt)",minHeight:"72px"}}
                        rows={3} value={input} onChange={e=>setInput(e.target.value)}
                        placeholder={autoImprove
                          ?"Chat or say 'add X to the API' to auto-modify code…"
                          :"Ask anything — live search, maths, your codebase, email me a summary…"}
                        onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}}
                      />
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <button type="button" disabled={chatLoading||!input.trim()} onClick={()=>send()}
                        className="rounded-xl px-5 py-2 text-sm font-bold transition-all hover:scale-[1.02] disabled:opacity-40"
                        style={{background:"linear-gradient(135deg,var(--blue),var(--cyan))",color:"#000"}}>
                        {chatLoading?"…":"Send ⏎"}
                      </button>
                      <span className="text-[10px]" style={{color:"var(--txt-3)"}}>⇧⏎ newline</span>
                      {messages.length>0 && (
                        <button type="button" onClick={newSession}
                          className="ml-auto text-[10px] transition hover:opacity-70"
                          style={{color:"var(--txt-3)"}}>
                          clear session
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ──── RESEARCH TAB ───────────────────────────────────────────── */}
            {tab==="research" && ResearchTabLive && (
              <ResearchTabLive baseUrl={baseUrl||getAgentBaseUrl()}/>
            )}
            {tab==="research" && !ResearchTabLive && (
              <ResearchTabFallback
                heartbeat={heartbeat} memory={memory} memoryLoading={memoryLoading}
                topicsEdit={topicsEdit} setTopicsEdit={setTopicsEdit}
                topicsSaving={topicsSaving} researchRunning={researchRunning}
                baseUrl={baseUrl||getAgentBaseUrl()}
                onRunResearch={async()=>{
                  setResearchRunning(true);
                  await triggerResearch(baseUrl||getAgentBaseUrl());
                  await new Promise(r=>setTimeout(r,1200));
                  await loadResearch();setResearchRunning(false);
                }}
                onSaveTopics={async()=>{
                  setTopicsSaving(true);
                  const topics=topicsEdit.split("\n").map(t=>t.trim()).filter(Boolean);
                  await setHeartbeatTopics(baseUrl||getAgentBaseUrl(),topics);
                  await loadResearch();setTopicsSaving(false);
                }}
                onClearMemory={async()=>{
                  if(!confirm("Clear all research memory?")) return;
                  await clearResearchMemory(baseUrl||getAgentBaseUrl());
                  setMemory("");
                }}
                onRefresh={loadResearch}
              />
            )}

            {/* ──── IMPROVE TAB ────────────────────────────────────────────── */}
            {tab==="improve" && (
              <div className="flex flex-col gap-5">
                <div className="rounded-2xl p-5" style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
                  <div className="mb-5 flex items-start gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-xl"
                      style={{background:"var(--violet)22",border:"1px solid var(--violet)44"}}>⚡</div>
                    <div>
                      <h3 className="font-bold text-base" style={{color:"var(--txt)"}}>Direct code improvement</h3>
                      <p className="mt-0.5 text-xs" style={{color:"var(--txt-3)"}}>
                        Describe the change. The agent localises affected files, generates code, runs AST validation, writes and commits.
                      </p>
                    </div>
                  </div>
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>Instruction</label>
                  <textarea
                    className="mb-4 w-full resize-none rounded-xl px-4 py-3 text-sm outline-none transition"
                    style={{background:"var(--elevated)",color:"var(--txt)",border:"1px solid var(--border)",minHeight:"96px"}}
                    rows={4} value={instruction} onChange={e=>setInstruction(e.target.value)}
                    placeholder={"Add a /api/v1/sessions list endpoint\nImprove the HDC memory retrieval scoring\nAdd rate limiting to all chat endpoints"}
                  />
                  <label className="mb-1.5 block text-xs font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>Target file (optional)</label>
                  <input
                    className="mb-4 w-full rounded-xl px-4 py-2.5 font-mono text-sm outline-none"
                    style={{background:"var(--elevated)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}
                    value={targetFile} onChange={e=>setTargetFile(e.target.value)}
                    placeholder="super_agent/app/api/routes.py (leave blank to auto-detect)"/>
                  <div className="mb-5 flex items-center gap-3 rounded-xl p-3"
                    style={{background:"var(--elevated)",border:`1px solid ${fullStack?"var(--violet)44":"var(--border-dim)"}`}}>
                    <button type="button" role="switch" aria-checked={fullStack}
                      onClick={()=>setFullStack(f=>!f)}
                      className="relative inline-flex h-5 w-9 items-center rounded-full transition-colors shrink-0"
                      style={{background:fullStack?"var(--violet)":"var(--card)"}}>
                      <span className="inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform"
                        style={{transform:fullStack?"translateX(18px)":"translateX(2px)"}}/>
                    </button>
                    <div>
                      <p className="text-xs font-semibold" style={{color:fullStack?"var(--violet)":"var(--txt-2)"}}>
                        Full-stack mode {fullStack?"(ON)":"(OFF)"}
                      </p>
                      <p className="mt-0.5 text-[10px]" style={{color:"var(--txt-3)"}}>
                        Also updates <code className="font-mono">agent-api.ts</code> and <code className="font-mono">AgentTester.tsx</code>
                      </p>
                    </div>
                  </div>
                  <button type="button" disabled={improving||!instruction.trim()} onClick={()=>void submitImprovement()}
                    className="rounded-xl px-7 py-2.5 text-sm font-bold transition-all hover:scale-[1.02] disabled:opacity-40"
                    style={{background:fullStack?"linear-gradient(135deg,var(--violet),var(--blue))":"linear-gradient(135deg,var(--blue),var(--cyan))",color:"#000"}}>
                    {improving
                      ? <span className="flex items-center gap-2">
                          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-black border-t-transparent"/>
                          {fullStack?"Applying full-stack…":"Applying…"}
                        </span>
                      : fullStack?"⚡ Apply (backend + frontend)":"Apply improvement"}
                  </button>
                </div>

                {improveResult && <ImproveResultCard r={improveResult}/>}
                {fullStackResult && <FullStackResultCard r={fullStackResult}/>}

                {improveHistory.entries.length>0 && (
                  <div>
                    <h3 className="mb-3 text-sm font-bold" style={{color:"var(--txt)"}}>
                      History ({improveHistory.entries.length})
                    </h3>
                    <div className="flex flex-col gap-2">
                      {improveHistory.entries.map((e,i)=>(
                        <div key={i} className="rounded-xl p-3.5"
                          style={{background:"var(--card)",border:`1px solid ${e.ok?"var(--teal)33":"var(--red)33"}`}}>
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span style={{color:e.ok?"var(--teal)":"var(--red)",fontWeight:700}}>{e.ok?"✓":"✗"}</span>
                              <span className="text-sm" style={{color:"var(--txt)"}}>{e.instruction}</span>
                            </div>
                            <span className="shrink-0 text-[10px]" style={{color:"var(--txt-3)"}}>{fmtTs(e.timestamp)}</span>
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-2">
                            <code className="font-mono text-[10px]" style={{color:"var(--txt-2)"}}>{e.target_file}</code>
                            {e.committed && <Pill label={`${e.commit_hash?.slice(0,8)||"committed"}`} color="blue"/>}
                            {e.error && <span className="text-[10px]" style={{color:"var(--red)"}}>{e.error}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ──── CODEBASE TAB ───────────────────────────────────────────── */}
            {tab==="codebase" && (
              <div className="flex flex-col gap-4">
                <div className="flex items-center gap-3 rounded-xl px-4 py-3"
                  style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
                  <span className="text-xl">◫</span>
                  <div>
                    <p className="text-sm font-bold" style={{color:"var(--txt)"}}>Live codebase snapshot</p>
                    <p className="text-[10px]" style={{color:"var(--txt-3)"}}>
                      Injected into every system instruction so the agent knows its own code.
                      {codebase&&` · ${(codebase.length/1024).toFixed(1)} KB`}
                    </p>
                  </div>
                  <button type="button" disabled={codebaseLoading}
                    onClick={async()=>{setCodebaseLoading(true);await refreshCodebase(baseUrl||getAgentBaseUrl());await loadCodebase();}}
                    className="ml-auto rounded-xl px-3 py-1.5 text-xs font-semibold transition hover:opacity-80 disabled:opacity-40"
                    style={{background:"var(--elevated)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}>
                    {codebaseLoading?"Scanning…":"↺ Rescan"}
                  </button>
                </div>
                <pre className="max-h-[72vh] overflow-auto rounded-2xl px-5 py-4 font-mono text-[11px] leading-relaxed"
                  style={{background:"var(--card)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}>
                  {codebaseLoading?"Scanning…":codebase||"Click Rescan to generate CODEBASE.md."}
                </pre>
              </div>
            )}

            {/* ──── STATUS TAB ─────────────────────────────────────────────── */}
            {tab==="status" && (
              <div className="flex flex-col gap-5">
                {/* metrics */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[
                    {label:"Status",value:health.status,color:health.status==="ok"?"var(--green)":"var(--red)"},
                    {label:"Gemini keys",value:health.gemini?"Active":"—",color:health.gemini?"var(--teal)":"var(--txt-3)"},
                    {label:"Research runs",value:heartbeat?heartbeat.total_runs.toString():"—",color:"var(--cyan)"},
                    {label:"Topics tracked",value:heartbeat?heartbeat.topic_count.toString():"—",color:"var(--blue)"},
                  ].map(m=>(
                    <div key={m.label} className="rounded-2xl p-4"
                      style={{background:"var(--card)",border:"1px solid var(--border-dim)"}}>
                      <p className="text-[10px] font-semibold uppercase tracking-widest" style={{color:"var(--txt-3)"}}>{m.label}</p>
                      <p className="mt-1 text-xl font-bold" style={{color:m.color}}>{m.value}</p>
                    </div>
                  ))}
                </div>

                {/* blueprint */}
                <div className="rounded-2xl overflow-hidden"
                  style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
                  <div className="flex items-center gap-3 px-4 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
                    <span className="text-base">◈</span>
                    <p className="font-bold text-sm" style={{color:"var(--txt)"}}>Blueprint</p>
                    {blueprint && (
                      <div className="ml-2 flex gap-3 text-xs">
                        <span style={{color:"var(--teal)"}}>{blueprint.items.filter(i=>i.status==="done").length} done</span>
                        <span style={{color:"var(--amber)"}}>{blueprint.items.filter(i=>i.status==="partial").length} partial</span>
                        <span style={{color:"var(--txt-3)"}}>{gaps.length} todo</span>
                      </div>
                    )}
                    <button type="button" onClick={()=>void loadStatus()}
                      className="ml-auto rounded-lg px-2 py-1 text-xs transition hover:opacity-70"
                      style={{background:"var(--elevated)",color:"var(--txt-3)"}}>↺</button>
                  </div>
                  {blueprint ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr style={{borderBottom:"1px solid var(--border-dim)",background:"var(--elevated)"}}>
                            {["Feature","Status","Notes"].map(h=>(
                              <th key={h} className="px-4 py-2.5 text-left font-semibold uppercase tracking-wider text-[9px]"
                                style={{color:"var(--txt-3)"}}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {blueprint.items.map(item=>(
                            <tr key={item.id} style={{borderBottom:"1px solid var(--border-dim)"}}>
                              <td className="px-4 py-2.5 font-medium" style={{color:"var(--txt)"}}>{item.title}</td>
                              <td className="px-4 py-2.5">
                                <Pill label={item.status}
                                  color={item.status==="done"?"teal":item.status==="partial"?"amber":"red"}/>
                              </td>
                              <td className="px-4 py-2.5" style={{color:"var(--txt-3)"}}>{item.notes}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="px-4 py-6 text-sm" style={{color:"var(--txt-3)"}}>Click ↺ to load.</p>
                  )}
                </div>

                {sica && (
                  <div className="rounded-2xl p-4" style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
                    <p className="mb-2 font-bold text-sm" style={{color:"var(--txt)"}}>SICA loop summary</p>
                    <pre className="rounded-xl p-3 font-mono text-[11px]"
                      style={{background:"var(--card)",color:"var(--txt-2)"}}>
                      {sica}
                    </pre>
                  </div>
                )}

                {/* endpoints */}
                <div className="rounded-2xl p-4" style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
                  <p className="mb-3 font-bold text-sm" style={{color:"var(--txt)"}}>API Endpoints</p>
                  <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                    {([
                      ["GET","/health"],["GET","/api/v1/status"],["GET","/api/v1/gaps"],
                      ["POST","/api/v1/chat"],["POST","/api/v1/agent/start"],
                      ["GET","/api/v1/agent/jobs/:id"],["POST","/api/v1/sympy/run"],
                      ["POST","/api/v1/route-intent"],["GET","/api/v1/codebase/snapshot"],
                      ["POST","/api/v1/codebase/refresh"],["POST","/api/v1/improve"],
                      ["POST","/api/v1/improve/fullstack"],["GET","/api/v1/improve/history"],
                      ["POST","/api/v1/research/trigger"],["GET","/api/v1/heartbeat/status"],
                      ["POST","/api/v1/notify/test"],
                    ] as [string,string][]).map(([method,path])=>(
                      <div key={path} className="flex items-center gap-2 rounded-xl px-3 py-2"
                        style={{background:"var(--card)"}}>
                        <span className="shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase"
                          style={{background:method==="GET"?"var(--blue)22":"var(--teal)22",
                            color:method==="GET"?"var(--blue)":"var(--teal)"}}>
                          {method}
                        </span>
                        <span className="font-mono text-[10px]" style={{color:"var(--txt-2)"}}>{path}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>
      </div>

      {/* ── MOBILE BOTTOM NAV ─────────────────────────────────────────────── */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 flex z-50"
        style={{background:"var(--surface)",borderTop:"1px solid var(--border-dim)"}}>
        {TABS.map(t=>(
          <MobileNavItem key={t.id} id={t.id} label={t.label} icon={t.icon}
            active={tab===t.id} badge={t.badge} onClick={()=>setTab(t.id)}/>
        ))}
      </nav>
    </div>
  );
}

// ─── improve result cards ─────────────────────────────────────────────────────
function ImproveResultCard({r}:{r:ImproveResult}){
  const changes=r.file_changes?.length
    ? r.file_changes
    : [{file:r.target_file,new_code:r.new_code,error:r.error,committed:r.committed,commit_hash:r.commit_hash,reason:"",ast_ok:r.ast_ok}];
  return (
    <div className="rounded-2xl overflow-hidden slide-up"
      style={{border:`1px solid ${r.ok?"var(--teal)55":"var(--red)55"}`,background:"var(--card)"}}>
      <div className="flex items-center gap-3 px-4 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
        <span className="text-sm font-bold" style={{color:r.ok?"var(--teal)":"var(--red)"}}>{r.ok?"✓ Applied":"✗ Failed"}</span>
        <Pill label={`${(r.file_changes?.length||1)} file${(r.file_changes?.length||1)!==1?"s":""}`} color="blue"/>
        <span className="ml-auto text-[10px]" style={{color:"var(--txt-3)"}}>
          {new Date(r.timestamp).toLocaleString()}
        </span>
      </div>
      {changes.map((fc,i)=>(
        <div key={i} className="px-4 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
          <div className="flex flex-wrap items-center gap-2">
            <span style={{color:fc.error?"var(--red)":"var(--teal)"}}>{fc.error?"✗":"✓"}</span>
            <code className="font-mono text-[11px]" style={{color:"var(--txt-2)"}}>{fc.file}</code>
            {fc.committed && <Pill label={(fc.commit_hash as string|null)?.slice(0,8)||"committed"} color="blue"/>}
            {fc.reason && <span className="text-[10px] italic" style={{color:"var(--txt-3)"}}>{fc.reason}</span>}
          </div>
          {fc.error && <p className="mt-1 text-xs" style={{color:"var(--red)"}}>{fc.error}</p>}
          {fc.new_code && !fc.error && (
            <details className="mt-2">
              <summary className="cursor-pointer text-[11px]" style={{color:"var(--txt-3)"}}>
                Show code ({(fc.new_code as string).split("\n").length} lines)
              </summary>
              <pre className="mt-1.5 max-h-56 overflow-auto rounded-xl p-3 font-mono text-[10px]"
                style={{background:"var(--void)",color:"#a8e8b8"}}>
                {fc.new_code as string}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}

function FullStackResultCard({r}:{r:FullStackImproveResult}){
  return (
    <div className="rounded-2xl overflow-hidden slide-up"
      style={{border:`1px solid ${r.ok?"var(--violet)55":"var(--red)55"}`,background:"var(--card)"}}>
      <div className="flex items-center gap-3 px-4 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
        <span className="text-sm font-bold" style={{color:r.ok?"var(--violet)":"var(--red)"}}>{r.ok?"⚡ Full-stack applied":"✗ Failed"}</span>
        <span className="ml-auto text-[10px]" style={{color:"var(--txt-3)"}}>{new Date(r.timestamp).toLocaleString()}</span>
      </div>
      {([
        {key:"backend" as const,label:"Backend",icon:"🐍"},
        {key:"frontend_api" as const,label:"API client (agent-api.ts)",icon:"🔌"},
        {key:"frontend_ui" as const,label:"UI (AgentTester.tsx)",icon:"🖥"},
      ] as const).map(({key,label,icon})=>{
        const sub=r[key]; if(!sub) return null;
        return (
          <div key={key} className="px-4 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
            <div className="flex flex-wrap items-center gap-2">
              <span>{icon}</span>
              <span className="text-xs font-semibold" style={{color:sub.ok?"var(--teal)":"var(--red)"}}>{sub.ok?"✓":"✗"} {label}</span>
              <code className="font-mono text-[10px]" style={{color:"var(--txt-2)"}}>{sub.target_file}</code>
              {sub.committed && <Pill label={sub.commit_hash?.slice(0,8)||"committed"} color="blue"/>}
              {sub.ast_ok && <span className="text-[10px]" style={{color:"var(--txt-3)"}}>AST ✓</span>}
            </div>
            {sub.error && <p className="mt-1.5 text-xs" style={{color:"var(--red)"}}>{sub.error}</p>}
            {sub.new_code && sub.ok && (
              <details className="mt-2">
                <summary className="cursor-pointer text-[11px]" style={{color:"var(--txt-3)"}}>
                  Show diff ({sub.new_code.split("\n").length} lines)
                </summary>
                <pre className="mt-1.5 max-h-48 overflow-auto rounded-xl p-3 font-mono text-[10px]"
                  style={{background:"var(--void)",color:"#a8e8b8"}}>
                  {sub.new_code}
                </pre>
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── fallback research tab (no Convex) ────────────────────────────────────────
function ResearchTabFallback({
  heartbeat,memory,memoryLoading,topicsEdit,setTopicsEdit,
  topicsSaving,researchRunning,baseUrl,
  onRunResearch,onSaveTopics,onClearMemory,onRefresh,
}:{
  heartbeat:HeartbeatStatus|null;memory:string;memoryLoading:boolean;
  topicsEdit:string;setTopicsEdit:(v:string)=>void;topicsSaving:boolean;
  researchRunning:boolean;baseUrl:string;
  onRunResearch:()=>Promise<void>;onSaveTopics:()=>Promise<void>;
  onClearMemory:()=>Promise<void>;onRefresh:()=>Promise<void>;
}){
  return <ResearchTabInner {...{heartbeat,memory,memoryLoading,topicsEdit,setTopicsEdit,topicsSaving,researchRunning,baseUrl,onRunResearch,onSaveTopics,onClearMemory,onRefresh,isLive:false}}/>;
}

type ResearchTabInnerProps={
  heartbeat:HeartbeatStatus|null;memory:string;memoryLoading:boolean;
  topicsEdit:string;setTopicsEdit:(v:string)=>void;topicsSaving:boolean;
  researchRunning:boolean;baseUrl:string;isLive:boolean;
  onRunResearch:()=>Promise<void>;onSaveTopics:()=>Promise<void>;
  onClearMemory:()=>Promise<void>;onRefresh:()=>Promise<void>;
};

export function ResearchTabInner({
  heartbeat,memory,memoryLoading,topicsEdit,setTopicsEdit,
  topicsSaving,researchRunning,isLive,
  onRunResearch,onSaveTopics,onClearMemory,onRefresh,
}:ResearchTabInnerProps){
  const [newTopic,setNewTopic]=useState("");
  const topics=useMemo(()=>topicsEdit.split("\n").map(t=>t.trim()).filter(Boolean),[topicsEdit]);

  const addTopic=()=>{
    const t=newTopic.trim();
    if(!t) return;
    setTopicsEdit([...topics,t].join("\n"));
    setNewTopic("");
  };
  const removeTopic=(i:number)=>{
    const next=topics.filter((_,j)=>j!==i);
    setTopicsEdit(next.join("\n"));
  };

  // parse memory entries from markdown
  const memEntries=useMemo(()=>{
    if(!memory) return [];
    return memory.split(/\n(?=## )/).map(block=>{
      const titleMatch=/^## (.+)/.exec(block);
      const dateMatch=/\*(.+?)\*/.exec(block);
      const body=block.replace(/^## .+\n/,"").replace(/\*.+?\*\n?/,"").trim();
      return {title:titleMatch?.[1]||"Research",date:dateMatch?.[1]||"",body};
    }).filter(e=>e.title!=="Agent Memory");
  },[memory]);

  return (
    <div className="flex flex-col gap-5">
      {/* header card */}
      <div className="rounded-2xl overflow-hidden"
        style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
        <div className="flex items-center gap-3 px-5 py-4" style={{borderBottom:"1px solid var(--border-dim)"}}>
          <div className="flex h-9 w-9 items-center justify-center rounded-xl"
            style={{background:"var(--blue)22",border:"1px solid var(--blue)44"}}>
            <span className="text-lg">⬡</span>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="font-bold" style={{color:"var(--txt)"}}>Proactive Research</h2>
              {isLive && (
                <span className="flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest blink"
                  style={{background:"var(--teal)22",color:"var(--teal)",border:"1px solid var(--teal)44"}}>
                  <span className="h-1.5 w-1.5 rounded-full" style={{background:"var(--teal)"}}/>LIVE
                </span>
              )}
            </div>
            <p className="text-[11px]" style={{color:"var(--txt-3)"}}>
              {heartbeat
                ? `${heartbeat.total_runs} runs completed · ${heartbeat.topic_count} topics · rotates every ${Math.round((heartbeat.interval_seconds||300)/60)}m`
                : "Loading status…"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" onClick={()=>void onRefresh()}
              className="rounded-xl px-3 py-1.5 text-xs font-medium transition hover:opacity-70"
              style={{background:"var(--elevated)",color:"var(--txt-2)",border:"1px solid var(--border-dim)"}}>
              ↺
            </button>
          </div>
        </div>

        {/* stats row */}
        {heartbeat && (
          <div className="grid grid-cols-3 gap-px" style={{background:"var(--border-dim)"}}>
            {[
              {label:"Total runs",value:heartbeat.total_runs.toString(),color:"var(--cyan)"},
              {label:"Last run",value:fmtRelative(heartbeat.last_run),color:"var(--txt-2)"},
              {label:"Up next",value:heartbeat.next_topic||"—",color:"var(--blue)"},
            ].map(s=>(
              <div key={s.label} className="px-4 py-3" style={{background:"var(--card)"}}>
                <p className="text-[9px] font-semibold uppercase tracking-widest mb-0.5" style={{color:"var(--txt-3)"}}>{s.label}</p>
                <p className="text-xs font-semibold truncate" style={{color:s.color}}>{s.value}</p>
              </div>
            ))}
          </div>
        )}

        {/* action buttons */}
        <div className="flex flex-wrap items-center gap-2 px-5 py-4">
          <button type="button" disabled={researchRunning} onClick={()=>void onRunResearch()}
            className="flex items-center gap-2 rounded-xl px-5 py-2 text-xs font-bold transition-all hover:scale-[1.02] disabled:opacity-50"
            style={{background:"linear-gradient(135deg,var(--blue),var(--cyan))",color:"#000"}}>
            {researchRunning
              ? <><span className="h-3 w-3 animate-spin rounded-full border-2 border-black border-t-transparent"/>Researching…</>
              : <>▶ Run research now</>}
          </button>
          {memory && (
            <button type="button" onClick={()=>void onClearMemory()}
              className="ml-auto rounded-xl px-4 py-2 text-xs font-medium transition hover:opacity-80"
              style={{background:"var(--red)18",color:"var(--red)",border:"1px solid var(--red)33"}}>
              🗑 Clear memory
            </button>
          )}
        </div>
      </div>

      {/* topic chips */}
      <div className="rounded-2xl p-5"
        style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
        <div className="mb-4 flex items-center gap-2">
          <h3 className="font-bold text-sm" style={{color:"var(--txt)"}}>Research Topics</h3>
          <span className="rounded-full px-2 py-0.5 text-[9px] font-bold"
            style={{background:"var(--blue)22",color:"var(--blue)"}}>
            {topics.length} / 20
          </span>
        </div>
        <div className="mb-4 flex flex-wrap gap-2">
          {topics.map((t,i)=>(
            <span key={i} className="group flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium"
              style={{background:"var(--elevated)",color:"var(--txt-2)",border:"1px solid var(--border)"}}>
              <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{background:heartbeat?.next_topic===t?"var(--cyan)":"var(--blue)"}}/>
              <span>{t}</span>
              <button type="button" onClick={()=>removeTopic(i)}
                className="opacity-0 group-hover:opacity-100 transition text-xs hover:text-red-400"
                style={{color:"var(--txt-3)",marginLeft:"2px"}}>×</button>
            </span>
          ))}
          {topics.length===0 && (
            <p className="text-xs" style={{color:"var(--txt-3)"}}>No topics yet. Add some below.</p>
          )}
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-xl px-3 py-2 text-sm outline-none"
            style={{background:"var(--elevated)",color:"var(--txt)",border:"1px solid var(--border-dim)"}}
            value={newTopic} onChange={e=>setNewTopic(e.target.value)}
            onKeyDown={e=>{if(e.key==="Enter"){e.preventDefault();addTopic();}}}
            placeholder="Add a topic and press Enter…"/>
          <button type="button" onClick={addTopic} disabled={!newTopic.trim()}
            className="rounded-xl px-4 py-2 text-sm font-semibold transition disabled:opacity-40"
            style={{background:"var(--blue)22",color:"var(--blue)",border:"1px solid var(--blue)44"}}>
            + Add
          </button>
        </div>
        {topics.length>0 && (
          <div className="mt-4 flex items-center justify-between">
            <p className="text-[10px]" style={{color:"var(--txt-3)"}}>
              Changes take effect after saving → synced to Convex
            </p>
            <button type="button" disabled={topicsSaving} onClick={()=>void onSaveTopics()}
              className="rounded-xl px-5 py-2 text-xs font-bold transition-all hover:scale-[1.02] disabled:opacity-40"
              style={{background:"linear-gradient(135deg,var(--teal),var(--blue))",color:"#000"}}>
              {topicsSaving?"Saving…":"Save topics"}
            </button>
          </div>
        )}
      </div>

      {/* memory entries */}
      <div className="rounded-2xl overflow-hidden"
        style={{background:"var(--surface)",border:"1px solid var(--border-dim)"}}>
        <div className="flex items-center gap-2 px-5 py-3" style={{borderBottom:"1px solid var(--border-dim)"}}>
          <span style={{color:"var(--txt)"}}>📝</span>
          <p className="font-bold text-sm" style={{color:"var(--txt)"}}>Research Memory</p>
          {memory && (
            <span className="ml-auto text-[10px]" style={{color:"var(--txt-3)"}}>
              {(memory.length/1024).toFixed(1)} KB · {memEntries.length} entries
            </span>
          )}
        </div>
        {memoryLoading ? (
          <div className="flex flex-col gap-3 p-5">
            {[1,2,3].map(i=>(
              <div key={i} className="skeleton h-16 rounded-xl"/>
            ))}
          </div>
        ) : memEntries.length>0 ? (
          <div className="flex flex-col gap-0">
            {memEntries.map((e,i)=>(
              <MemoryEntryCard key={i} entry={e} index={i}/>
            ))}
          </div>
        ) : (
          <div className="px-5 py-10 text-center">
            <p className="text-2xl mb-2">⬡</p>
            <p className="text-sm" style={{color:"var(--txt-3)"}}>No research yet — click "Run research now"</p>
          </div>
        )}
      </div>
    </div>
  );
}

function MemoryEntryCard({entry,index}:{entry:{title:string;date:string;body:string};index:number}){
  const [expanded,setExpanded]=useState(index===0);
  return (
    <div style={{borderBottom:"1px solid var(--border-dim)"}}>
      <button type="button" onClick={()=>setExpanded(e=>!e)}
        className="w-full flex items-center gap-3 px-5 py-3 text-left transition hover:opacity-80">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg text-[10px] font-bold"
          style={{background:"var(--blue)22",color:"var(--blue)"}}>{index+1}</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold truncate" style={{color:"var(--txt)"}}>{entry.title}</p>
          {entry.date && <p className="text-[10px]" style={{color:"var(--txt-3)"}}>{entry.date}</p>}
        </div>
        <span className="text-xs" style={{color:"var(--txt-3)"}}>{expanded?"▲":"▼"}</span>
      </button>
      {expanded && (
        <div className="px-5 pb-4 fade-in">
          <p className="text-xs leading-relaxed whitespace-pre-wrap" style={{color:"var(--txt-2)"}}>{entry.body}</p>
        </div>
      )}
    </div>
  );
}
