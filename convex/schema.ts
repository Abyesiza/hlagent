import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  memoryEntries: defineTable({
    title: v.string(),
    body: v.string(),
    createdAt: v.number(),
  }).index("by_created", ["createdAt"]),

  agentTasks: defineTable({
    kind: v.string(),
    status: v.string(),
    detail: v.optional(v.string()),
    error: v.optional(v.string()),
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_created", ["createdAt"]),

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
});
