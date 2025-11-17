import type { PluginOption } from "vite";
import vue from "@vitejs/plugin-vue";
import vueJsx from "@vitejs/plugin-vue-jsx";
import VueDevtools from "vite-plugin-vue-devtools";
import progress from "vite-plugin-progress";
import { setupElegantRouter } from "./router";
import { setupUnocss } from "./unocss";
import { setupUnplugin } from "./unplugin";
import { setupHtmlPlugin } from "./html";
import AutoImport from "unplugin-auto-import/vite";
// 优化
import { visualizer } from "rollup-plugin-visualizer";
import viteImagemin from "vite-plugin-imagemin";
import viteCompression from "vite-plugin-compression";

export function setupVitePlugins(viteEnv: Env.ImportMeta, buildTime: string) {
  const plugins: PluginOption = [
    vue({
      script: {
        defineModel: true,
      },
    }),
    vueJsx(),
    AutoImport({
      imports: ["vue", "vue-router", "pinia"], // 自动引入的三方库
      dts: "src/typings/auto-import.d.ts", // 全局自动引入文件存放路径；不配置保存在根目录下；配置为false时将不会生成 auto-imports.d.ts 文件（不影响效果）
      eslintrc: {
        // 项目中使用了 eslint，那么虽然可以正常使用 API 了，但是 eslint 还是会报没有引入的报错。下面的配置可以处理这种情况
        enabled: true, // 会在根目录下自动生成 .eslintrc-auto-import.json 文件
      },
    }),
    VueDevtools(),
    setupElegantRouter(),
    setupUnocss(viteEnv),
    ...setupUnplugin(viteEnv),
    progress(),
    setupHtmlPlugin(buildTime),
    // 调优
    visualizer({ open: false }),
    viteImagemin({
      // 无损压缩配置，无损压缩下图片质量不会变差
      optipng: {
        optimizationLevel: 7,
      },
      // 有损压缩配置，有损压缩下图片质量可能会变差
      pngquant: {
        quality: [0.8, 0.9],
      },
      // svg 优化
      svgo: {
        plugins: [
          {
            name: "removeViewBox",
          },
          {
            name: "removeEmptyAttrs",
            active: false,
          },
        ],
      },
    }),
    viteCompression({
      verbose: true, // 默认即可
      disable: false, // 开启压缩(不禁用)，默认即可
      deleteOriginFile: false, // 删除源文件
      threshold: 5120, // 压缩前最小文件大小
      algorithm: "gzip", // 压缩算法
      ext: ".gz", // 文件类型
    }),
  ];

  return plugins;
}
