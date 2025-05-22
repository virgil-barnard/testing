// vite.config.js
import { defineConfig } from 'vite';

const repo = 'YOUR_REPO_NAME';        // <<<  change only this line

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
