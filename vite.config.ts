import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { defineConfig } from 'vite';

const BACKEND = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: mode !== 'production',
    target: 'es2022',
    rollupOptions: {
      output: {
        // Rollup 4 only accepts the function form of manualChunks.
        manualChunks(id: string): string | undefined {
          if (id.includes('node_modules/lucide-react')) return 'icons';
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return 'react';
          return undefined;
        },
      },
    },
  },
  server: {
    port: 5173,
    // Vite runs standalone in development and proxies API + ops endpoints to
    // the FastAPI backend. In production FastAPI serves the built `dist/`.
    proxy: {
      '/api': BACKEND,
      '/healthz': BACKEND,
      '/readyz': BACKEND,
      '/metrics': BACKEND,
    },
  },
  preview: {
    port: 4173,
  },
}));
