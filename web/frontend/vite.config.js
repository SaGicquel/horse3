import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Permet l'accès depuis d'autres machines
    port: 5173,      // Port par défaut de Vite
    strictPort: false,
  },
  build: {
    // Code splitting optimisé
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor React - chargé en premier, mis en cache longtemps
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // Libs lourdes - lazy loaded
          'vendor-charts': ['recharts'],
          'vendor-motion': ['framer-motion'],
          'vendor-icons': ['lucide-react', '@heroicons/react'],
        },
      },
    },
    // Minification optimale
    minify: 'esbuild',
    target: 'esnext',
    // Taille des chunks
    chunkSizeWarningLimit: 500,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    clearMocks: true,
    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],
    exclude: ['node_modules/**', 'dist/**', 'tests/e2e/**'],
  },
})
