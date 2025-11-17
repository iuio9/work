import { createApp } from "vue";
import "./plugins/assets";
import {
  setupAppVersionNotification,
  setupDayjs,
  setupIconifyOffline,
  setupLoading,
  setupNProgress,
} from "./plugins";
import { setupStore } from "./store";
import { setupRouter } from "./router";
import { setupI18n } from "./locales";
import App from "./App.vue";

import directive from './directive'
// custom directives
import directives from "@/directives/index";


// vue3 context menu
import "@imengyu/vue3-context-menu/lib/vue3-context-menu.css";
import ContextMenu from "@imengyu/vue3-context-menu";

// mobile 使用
// import 'amfe-flexible';
// import '@/utils/rem.js';

import uploader from "vue-simple-uploader";
import "vue-simple-uploader/dist/style.css";

// 导入 ElementPlus
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

import Lazyload from "vue3-lazyload";

import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import VirtualScroller from 'vue-virtual-scroller'

async function setupApp() {
  setupLoading();

  setupNProgress();

  setupIconifyOffline();

  setupDayjs();

  const app = createApp(App);

  app.use(VirtualScroller)

  // 注册插件
  app.use(Lazyload, {
    loading: "", // 可以指定加载中的图像
    error: "", // 可以指定加载失败的图像
  });

  app.use(directives);

  // 文件上传
  app.use(uploader);

  setupStore(app);

  await setupRouter(app);

  setupI18n(app);

  directive(app)

  setupAppVersionNotification();

  app.use(ContextMenu);

  app.use(ElementPlus); // 挂载 ElementPlus

  app.mount("#app");
}

setupApp();
