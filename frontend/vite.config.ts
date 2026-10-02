import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  base: "/", // absolute asset paths so deep links like /verify load correctly
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
})
