// vite.config.js
import { defineConfig } from 'vite';

const repo = 'https://github.com/virgil-barnard/testing';

export default defineConfig(({ mode }) => ({
  /** ① the base URL GitHub Pages will host under */
  base: mode === 'production' ? `/${repo}/` : '/',
  /** ② copy WGSL files & let ?raw imports work */
  assetsInclude: ['**/*.wgsl'],
  /** ③ normal single-page build */
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: { input: 'index.html' }
  }
}));
