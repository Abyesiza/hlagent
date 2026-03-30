import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // ── memory entries (research summaries, training logs) ──────────────────
  memoryEntries: defineTable({
    title: v.string(),
    body: v.string(),
    createdAt: v.number(),
  }).index("by_created", ["createdAt"]),

  // ── agent tasks (background job tracking) ───────────────────────────────
  agentTasks: defineTable({
    kind: v.string(),
    status: v.string(),
    detail: v.optional(v.string()),
    error: v.optional(v.string()),
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_created", ["createdAt"]),

  // ── research configuration (heartbeat topics + cursor) ──────────────────
  /** Single logical row (key === "global"): heartbeat topics, cursor, persona. */
  researchConfig: defineTable({
    key: v.string(),
    topics: v.array(v.string()),
    cursorIndex: v.number(),
    lastRun: v.optional(v.string()),
    totalRuns: v.number(),
    persona: v.string(),
    updatedAt: v.number(),
  }).index("by_key", ["key"]),

  // ── HDC model training corpus (scraped document metadata) ───────────────
  trainingCorpus: defineTable({
    topic: v.string(),
    source: v.string(),           // "wikipedia" | "web" | "ddg_snippet"
    url: v.string(),
    title: v.string(),
    wordCount: v.number(),
    contentHash: v.string(),
    scrapedAt: v.string(),
    trainedAt: v.number(),
  })
    .index("by_trained", ["trainedAt"])
    .index("by_topic", ["topic"])
    .index("by_hash", ["contentHash"]),

  // ── HDC model statistics (single row, key === "global") ──────────────────
  hdcModelStats: defineTable({
    key: v.string(),
    vocabSize: v.number(),
    trainingTokens: v.number(),
    trainingDocs: v.number(),
    assocMemorySize: v.number(),
    dim: v.number(),
    contextSize: v.number(),
    lastTrained: v.optional(v.string()),
    createdAt: v.string(),
    updatedAt: v.number(),
  }).index("by_key", ["key"]),

  // ── training run log (each research+train cycle) ─────────────────────────
  trainingRuns: defineTable({
    topic: v.string(),
    docsScraped: v.number(),
    docsTrained: v.number(),
    pairsTrained: v.number(),
    wordsTotal: v.number(),
    elapsedSeconds: v.number(),
    vocabAfter: v.number(),
    createdAt: v.number(),
  }).index("by_created", ["createdAt"]),
});
