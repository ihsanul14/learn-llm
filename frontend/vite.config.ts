import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
import devtools from 'solid-devtools/vite';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [devtools(), solidPlugin(), tailwindcss()],
  server: {
    port: 3000,
  },
  // optimizeDeps: {
  //   // Add the problematic package and its dependencies here
  //   include: ['solid-markdown', 'debug'],
  // },
  build: {
    target: 'esnext',
  },
});
