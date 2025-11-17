import process from "node:process";
import { URL, fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import { setupVitePlugins } from "./build/plugins";
import { createViteProxy, getBuildTime } from "./build/config";

export default defineConfig((configEnv) => {
  const viteEnv = loadEnv(
    configEnv.mode,
    process.cwd(),
  ) as unknown as Env.ImportMeta;

  const buildTime = getBuildTime();

  const enableProxy = configEnv.command === "serve" && !configEnv.isPreview;

  return {
    base: viteEnv.VITE_BASE_URL,
    resolve: {
      alias: {
        "~": fileURLToPath(new URL("./", import.meta.url)),
        "@": fileURLToPath(new URL("./src", import.meta.url)),
        '@vue-office/excel': '@vue-office/excel' // 根据实际入口文件路径调整
      },
    },
    css: {
      preprocessorOptions: {
        scss: {
          additionalData: `@use "./src/styles/scss/global.scss" as *;`,
        },
      },
    },
    plugins: setupVitePlugins(viteEnv, buildTime),
    define: {
      BUILD_TIME: JSON.stringify(buildTime),
    },
    server: {
      host: "0.0.0.0",
      port: 8080,
      open: true,
      proxy: createViteProxy(viteEnv, enableProxy),
      fs: {
        cachedChecks: false,
      },
    },
    preview: {
      port: 9725,
    },
    build: {
      reportCompressedSize: false,
      sourcemap: viteEnv.VITE_SOURCE_MAP === "Y",
      commonjsOptions: {
        ignoreTryCatch: false,
      },
      rollupOptions: {
        output: {
          chunkFileNames: "js/[name]-[hash].js", // 引入文件名的名称
          entryFileNames: "js/[name]-[hash].js", // 包的入口文件名称
          // assetFileNames: "[ext]/[name]-[hash].[ext]", // 资源文件像字体，图片等
          // 对打包出来的资源文件进行分类，分别放到不同的文件夹内
          assetFileNames(assetsInfo) {
            //  css样式文件
            if (assetsInfo.name?.endsWith(".css")) {
              return "css/[name]-[hash].css";
            }
            //  字体文件
            const fontExts = [".ttf", ".otf", ".woff", ".woff2", ".eot"];
            if (fontExts.some((ext) => assetsInfo.name?.endsWith(ext))) {
              return "font/[name]-[hash].[ext]";
            }

            //  图片文件
            const imgExts = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".icon"];
            if (imgExts.some((ext) => assetsInfo.name?.endsWith(ext))) {
              return "img/[name]-[hash].[ext]";
            }

            //  SVG类型的图片文件
            const imgSvg = [".svg"];
            if (imgSvg.some((ext) => assetsInfo.name?.endsWith(ext))) {
              return "assest/icons/[name].[ext]";
            }

            //  视频文件
            const videoExts = [".mp4", ".avi", ".wmv", ".ram", ".mpg", "mpeg"];
            if (videoExts.some((ext) => assetsInfo.name?.endsWith(ext))) {
              return "video/[name]-[hash].[ext]";
            }
            //  其它文件: 保存在 assets/图片名-哈希值.扩展名
            return "assets/[name]-[hash].[ext]";
          },
          manualChunks(id) {
            if (id.includes("node_modules")) {
              return id
                .toString()
                .split("node_modules/")[1]
                .split("/")[0]
                .toString();
            }
          },
        },
      },
      // 打包环境移除console.log，debugger
      terserOptions: {
        compress: {
          drop_console: true,
          drop_debugger: true,
        },
      },
    },
  };
});
