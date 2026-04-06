import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    hmr: {
      path: '/__vite_hmr',
    },
    proxy: {
      '/api/v1/ws': {
        target: process.env.VITE_WS_URL || 'ws://api:8080',
        ws: true,
      },
      '/api': {
        target: process.env.VITE_API_URL || 'http://api:8080',
        changeOrigin: true,
      },
      '/healthz': {
        target: process.env.VITE_API_URL || 'http://api:8080',
        changeOrigin: true,
      },
      '/simulator': {
        target: process.env.VITE_SIM_URL || 'http://simulator:8085',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/simulator/, ''),
      },
    },
  },
})
