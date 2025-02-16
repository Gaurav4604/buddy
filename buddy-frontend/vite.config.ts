// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    hmr: {
      host: "localhost",
      protocol: "ws",
      clientPort: 5173,
    },
  },
});
