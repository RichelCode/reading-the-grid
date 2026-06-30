import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// During development the frontend runs on Vite (port 5173) and proxies /api calls to the
// FastAPI backend on port 8000, so the browser always talks to a same-origin /api path.
// In the Docker build there is no dev server: FastAPI serves the built files and the API
// from one origin, so the same /api paths work without a proxy.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
