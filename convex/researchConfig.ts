import { mutation, query } from "./_generated/server";
import type { DatabaseReader } from "./_generated/server";
import { v } from "convex/values";

const GLOBAL_KEY = "global" as const;

async function getRow(ctx: { db: DatabaseReader }) {
  return await ctx.db
    .query("researchConfig")
    .withIndex("by_key", (q) => q.eq("key", GLOBAL_KEY))
    .unique();
}

export const get = query({
  args: {},
  handler: async (ctx) => {
    const row = await getRow(ctx);
    if (!row) {
      return {
        initialized: false,
        topics: [] as string[],
        cursorIndex: 0,
        lastRun: null as string | null,
        totalRuns: 0,
        persona: "",
        updatedAt: 0,
      };
    }
    return {
      initialized: true,
      topics: row.topics,
      cursorIndex: row.cursorIndex,
      lastRun: row.lastRun ?? null,
      totalRuns: row.totalRuns,
      persona: row.persona,
      updatedAt: row.updatedAt,
    };
  },
});

export const seed = mutation({
  args: {
    topics: v.array(v.string()),
    cursorIndex: v.number(),
    lastRun: v.optional(v.string()),
    totalRuns: v.number(),
    persona: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await getRow(ctx);
    if (existing) {
      return String(existing._id);
    }
    const now = Date.now();
    const capped = args.topics.slice(0, 20).map((t) => t.trim()).filter(Boolean);
    return String(
      await ctx.db.insert("researchConfig", {
        key: GLOBAL_KEY,
        topics: capped,
        cursorIndex: args.cursorIndex,
        lastRun: args.lastRun,
        totalRuns: args.totalRuns,
        persona: args.persona.trim(),
        updatedAt: now,
      }),
    );
  },
});

export const setTopics = mutation({
  args: { topics: v.array(v.string()) },
  handler: async (ctx, args) => {
    const now = Date.now();
    const capped = args.topics.slice(0, 20).map((t) => t.trim()).filter(Boolean);
    const existing = await getRow(ctx);
    if (existing) {
      await ctx.db.patch(existing._id, { topics: capped, updatedAt: now });
      return;
    }
    await ctx.db.insert("researchConfig", {
      key: GLOBAL_KEY,
      topics: capped,
      cursorIndex: 0,
      totalRuns: 0,
      persona: "",
      updatedAt: now,
    });
  },
});

export const setPersona = mutation({
  args: { persona: v.string() },
  handler: async (ctx, args) => {
    const now = Date.now();
    const existing = await getRow(ctx);
    if (existing) {
      await ctx.db.patch(existing._id, {
        persona: args.persona.trim(),
        updatedAt: now,
      });
      return;
    }
    await ctx.db.insert("researchConfig", {
      key: GLOBAL_KEY,
      topics: [],
      cursorIndex: 0,
      totalRuns: 0,
      persona: args.persona.trim(),
      updatedAt: now,
    });
  },
});

export const appendPersona = mutation({
  args: { block: v.string() },
  handler: async (ctx, args) => {
    const now = Date.now();
    const existing = await getRow(ctx);
    const block = args.block.trim();
    if (!block) {
      return;
    }
    if (existing) {
      const next = existing.persona ? `${existing.persona}\n\n${block}` : block;
      await ctx.db.patch(existing._id, { persona: next, updatedAt: now });
      return;
    }
    await ctx.db.insert("researchConfig", {
      key: GLOBAL_KEY,
      topics: [],
      cursorIndex: 0,
      totalRuns: 0,
      persona: block,
      updatedAt: now,
    });
  },
});

/** After a completed research run (topic rotation or persona-only). */
export const recordRun = mutation({
  args: {
    nextCursorIndex: v.number(),
    lastRunIso: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await getRow(ctx);
    if (!existing) {
      return;
    }
    await ctx.db.patch(existing._id, {
      cursorIndex: args.nextCursorIndex,
      lastRun: args.lastRunIso,
      totalRuns: existing.totalRuns + 1,
      updatedAt: Date.now(),
    });
  },
});
