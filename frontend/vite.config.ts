import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// API routes the dev server proxies to the FastAPI backend (uvicorn on :8000).
// In production the built bundle is served by FastAPI at the same origin, so these
// relative fetches just work; the proxy only matters for `npm run dev`.
const API_ROUTES = [
  "/new_game",
  "/roll",
  "/legal_moves",
  "/move",
  "/agent_move",
  "/checkpoints",
  "/export_mat",
];

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    // Emit straight into the Python package's static dir (committed, FastAPI-served).
    outDir: fileURLToPath(new URL("../bgrl/web/static", import.meta.url)),
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(API_ROUTES.map((route) => [route, "http://localhost:8000"])),
  },
});
