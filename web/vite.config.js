import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development the Vite server proxies the FastAPI backend; in production
// FastAPI serves the built files from web/dist itself.
const backend = process.env.FITNESSCOACH_API || 'http://127.0.0.1:8001';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/exercise-media': backend,
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1600,
  },
});
