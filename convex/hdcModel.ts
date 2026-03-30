import { mutation, query, internalMutation } from "./_generated/server";
import { v } from "convex/values";

// ── HDC Model Stats ──────────────────────────────────────────────────────────

export const getStats = query({
  args: {},
  handler: async (ctx) => {
    const row = await ctx.db
      .query("hdcModelStats")
      .withIndex("by_key", (q) => q.eq("key", "global"))
      .unique();
    return row ?? null;
  },
});

export const upsertStats = mutation({
  args: {
    vocabSize: v.number(),
    trainingTokens: v.number(),
    trainingDocs: v.number(),
    assocMemorySize: v.number(),
    dim: v.number(),
    contextSize: v.number(),
    lastTrained: v.optional(v.string()),
    createdAt: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("hdcModelStats")
      .withIndex("by_key", (q) => q.eq("key", "global"))
      .unique();
    const now = Date.now();
    if (existing) {
      await ctx.db.patch(existing._id, {
        ...args,
        updatedAt: now,
      });
    } else {
      await ctx.db.insert("hdcModelStats", {
        key: "global",
        ...args,
        updatedAt: now,
      });
    }
  },
});

// ── Training Corpus ──────────────────────────────────────────────────────────

export const logTrainedDocument = mutation({
  args: {
    topic: v.string(),
    source: v.string(),
    url: v.string(),
    title: v.string(),
    wordCount: v.number(),
    contentHash: v.string(),
    scrapedAt: v.string(),
  },
  handler: async (ctx, args) => {
    // Skip if already logged (idempotent)
    const existing = await ctx.db
      .query("trainingCorpus")
      .withIndex("by_hash", (q) => q.eq("contentHash", args.contentHash))
      .unique();
    if (existing) return existing._id;

    return await ctx.db.insert("trainingCorpus", {
      ...args,
      trainedAt: Date.now(),
    });
  },
});

export const listCorpus = query({
  args: {
    topic: v.optional(v.string()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const lim = args.limit ?? 50;
    if (args.topic) {
      return await ctx.db
        .query("trainingCorpus")
        .withIndex("by_topic", (q) => q.eq("topic", args.topic!))
        .order("desc")
        .take(lim);
    }
    return await ctx.db
      .query("trainingCorpus")
      .withIndex("by_trained")
      .order("desc")
      .take(lim);
  },
});

export const corpusStats = query({
  args: {},
  handler: async (ctx) => {
    const recent = await ctx.db
      .query("trainingCorpus")
      .withIndex("by_trained")
      .order("desc")
      .take(100);
    const totalDocs = recent.length;
    const totalWords = recent.reduce((acc, d) => acc + d.wordCount, 0);
    const topicCounts: Record<string, number> = {};
    for (const d of recent) {
      topicCounts[d.topic] = (topicCounts[d.topic] ?? 0) + 1;
    }
    return { totalDocs, totalWords, topicCounts };
  },
});

// ── Training Runs ────────────────────────────────────────────────────────────

export const logTrainingRun = mutation({
  args: {
    topic: v.string(),
    docsScraped: v.number(),
    docsTrained: v.number(),
    pairsTrained: v.number(),
    wordsTotal: v.number(),
    elapsedSeconds: v.number(),
    vocabAfter: v.number(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("trainingRuns", {
      ...args,
      createdAt: Date.now(),
    });
  },
});

export const listTrainingRuns = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("trainingRuns")
      .withIndex("by_created")
      .order("desc")
      .take(args.limit ?? 20);
  },
});

// ── Model Weights (Vercel-safe persistence) ───────────────────────────────
// The assoc memory and vocabulary are stored as compact strings to stay
// well under Convex's 8192-element array limit:
//   assocMemoryB64 — base64 of int8-packed ±1 bipolar floats
//   vocabLabels    — newline-delimited word list (hypervectors regenerated deterministically)

export const saveWeights = mutation({
  args: {
    dim: v.number(),
    contextSize: v.number(),
    assocCount: v.number(),
    assocMemoryB64: v.string(),
    vocabLabels: v.string(),
    trainingTokens: v.number(),
    trainingDocs: v.number(),
    lastTrained: v.optional(v.string()),
    createdAt: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("hdcModelWeights")
      .withIndex("by_key", (q) => q.eq("key", "global"))
      .unique();
    const now = Date.now();
    if (existing) {
      await ctx.db.patch(existing._id, { ...args, updatedAt: now });
    } else {
      await ctx.db.insert("hdcModelWeights", {
        key: "global",
        ...args,
        updatedAt: now,
      });
    }
  },
});

export const loadWeights = query({
  args: {},
  handler: async (ctx) => {
    const row = await ctx.db
      .query("hdcModelWeights")
      .withIndex("by_key", (q) => q.eq("key", "global"))
      .unique();
    return row ?? null;
  },
});
