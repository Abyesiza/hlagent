import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const create = mutation({
  args: {
    kind: v.string(),
    detail: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    return await ctx.db.insert("agentTasks", {
      kind: args.kind,
      status: "queued",
      detail: args.detail,
      createdAt: now,
      updatedAt: now,
    });
  },
});

export const setStatus = mutation({
  args: {
    id: v.id("agentTasks"),
    status: v.string(),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.id, {
      status: args.status,
      updatedAt: Date.now(),
      ...(args.error !== undefined ? { error: args.error } : {}),
    });
  },
});

export const listRecent = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const lim = Math.min(args.limit ?? 50, 200);
    return await ctx.db
      .query("agentTasks")
      .withIndex("by_created", (q) => q.gte("createdAt", 0))
      .order("desc")
      .take(lim);
  },
});
