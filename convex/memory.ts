import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const append = mutation({
  args: {
    title: v.string(),
    body: v.string(),
    createdAt: v.number(),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("memoryEntries", {
      title: args.title,
      body: args.body,
      createdAt: args.createdAt,
    });
  },
});

export const listAsc = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const lim = Math.min(args.limit ?? 500, 2000);
    return await ctx.db
      .query("memoryEntries")
      .withIndex("by_created", (q) => q.gte("createdAt", 0))
      .order("asc")
      .take(lim);
  },
});

export const clearAll = mutation({
  args: {},
  handler: async (ctx) => {
    for (const row of await ctx.db.query("memoryEntries").collect()) {
      await ctx.db.delete(row._id);
    }
  },
});
