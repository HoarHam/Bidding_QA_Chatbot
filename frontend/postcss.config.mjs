/**
 * Tailwind v4 用 PostCSS 插件接入。
 *
 * v4 是 CSS-first 的：**没有 tailwind.config.js**。
 * 生成那个文件（很多教程和脚手架会这么做）在 v4 下静默失效，
 * 配置要写在 app/globals.css 的 @theme 里。
 */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
