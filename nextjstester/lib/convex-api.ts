/**
 * Same Convex deployment as FastAPI (`CONVEX_URL`). Uses `anyApi` so we do not import
 * repo-root `convex/_generated/api.js` (root package.json is `"type": "commonjs"` and breaks Turbopack).
 */
import { anyApi } from "convex/server";

export const api = anyApi;
