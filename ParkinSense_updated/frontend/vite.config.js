import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/predict': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/status': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      }
    }
  }
})
