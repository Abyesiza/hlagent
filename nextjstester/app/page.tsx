import AgentTester from "./components/AgentTester";

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-zinc-50 dark:bg-zinc-950">
      <header className="sticky top-0 z-20 border-b border-zinc-200 bg-white/90 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/90">
        <div className="flex items-center gap-4 px-6 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-900 dark:bg-zinc-100">
              <span className="text-[13px] font-bold text-white dark:text-zinc-900">SA</span>
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-none text-zinc-900 dark:text-zinc-50">
                Super Agent
              </h1>
              <p className="mt-0.5 text-[10px] leading-none text-zinc-400">
                Neuro-symbolic · self-evolving · local
              </p>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-3 text-[11px] text-zinc-400">
            <span className="hidden sm:block">FastAPI + Gemini + HDC + SymPy</span>
            <span className="rounded border border-zinc-200 bg-zinc-50 px-2 py-0.5 font-mono dark:border-zinc-800 dark:bg-zinc-900">
              env-driven API URL
            </span>
          </div>
        </div>
      </header>
      <div className="flex flex-1 flex-col px-4 py-5 sm:px-6">
        <AgentTester />
      </div>
    </div>
  );
}
