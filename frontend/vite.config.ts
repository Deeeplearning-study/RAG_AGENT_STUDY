import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 6000,
    strictPort: true,
    allowedHosts: ['proxy2.nipa2025.ktcloud.com', 'main1.betta-mackarel.ts.net'],
    proxy: {
      '/api': {
        target: 'http://localhost:8888',
        changeOrigin: true,
      },
    },
  },
});
