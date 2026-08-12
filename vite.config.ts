import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND_TARGET = process.env.VITE_BACKEND_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
