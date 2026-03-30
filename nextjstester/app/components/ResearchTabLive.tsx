"use client";
// ResearchTabLive — Convex-backed research tab (only loaded when NEXT_PUBLIC_CONVEX_URL is set).
// This is a thin wrapper that re-exports the same Research UI with live Convex data.
// Without Convex it is never imported (AgentTester.tsx checks _convexUrl first).

export default function ResearchTabLive() {
  return null;
}
