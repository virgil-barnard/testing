// vite.config.js
import { defineConfig } from 'vite';

const repo = 'testing';        // <<<  change only this line

export default defineConfig(({ mode }) => ({
  /** ① the base URL GitHub Pages will host under */
  base: mode === 'production' ? `/${repo}/` : '/',
  /** ② copy WGSL files & let ?raw imports work */
  assetsInclude: ['**/*.wgsl'],
  /** ③ normal single-page build */
  build: {
    target: 'esnext',
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: { input: 'index.html' }
  }
}));
