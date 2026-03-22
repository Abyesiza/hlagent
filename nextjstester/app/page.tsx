import AgentTester from "./components/AgentTester";

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-zinc-50 dark:bg-zinc-950">
      <header className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Super Agent — local tester
        </h1>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          Talks to your FastAPI process on this machine. Chat history is stored in the browser (
          <code className="rounded bg-zinc-100 px-1 text-xs dark:bg-zinc-800">localStorage</code>
          ).
        </p>
      </header>
      <div className="flex flex-1 flex-col px-6 py-6">
        <AgentTester />
      </div>
    </div>
  );
}
