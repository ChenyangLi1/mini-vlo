import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget = env.VITE_BACKEND_TARGET || 'http://127.0.0.1:8000'

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        '/run-pipeline': { target: backendTarget, changeOrigin: true },
        '/get-status': { target: backendTarget, changeOrigin: true },
        '/get-results': { target: backendTarget, changeOrigin: true },
      },
    },
  }
})
