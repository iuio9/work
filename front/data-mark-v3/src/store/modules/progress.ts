// src/stores/progress.ts
import { defineStore } from 'pinia';

interface ItemProgress {
  value: number;
  max: number;
}

interface ModuleProgress {
  [itemId: string]: ItemProgress;
}

interface ProgressState {
  modules: Record<string, ModuleProgress>;
}

export const useProgressStore = defineStore('progress', {
  state: (): ProgressState => ({
    // 初始化时可以为空对象，或者预定义一些模块和列表项
    modules: {},
  }),
  getters: {
    // 获取特定模块和列表项的完成百分比
    getItemPercent: (state) => (moduleName: string, itemId: string): number => {
        const module = state.modules[moduleName];
      if (module && module[itemId]) {
        return (module[itemId].value / module[itemId].max) * 100;
      }
      return 0;
    },
    // 获取所有模块及其列表项的状态
    allModules(): Record<string, ModuleProgress> {
      return this.modules;
    },
    // 获取特定模块的所有列表项状态
    getModuleItems(moduleName: string): ModuleProgress | undefined {
      return this.modules[moduleName];
    }
  },
  actions: {
    // 初始化一个新的模块
    initModule(moduleName: string) {
      if (!this.modules[moduleName]) {
        this.modules[moduleName] = {};
      }
    },
    // 初始化指定模块下的新列表项
    initItem(moduleName: string, itemId: string, max: number = 100) {
      this.initModule(moduleName);
      if (!this.modules[moduleName][itemId]) {
        this.modules[moduleName][itemId] = { value: 0, max };
      }
    },
    // 增加指定模块和列表项的进度
    incrementItem(moduleName: string, itemId: string, amount: number = 1) {
      const item = this.modules[moduleName]?.[itemId];
      if (item && item.value + amount <= item.max) {
        item.value += amount;
      }
    },
    // 设置指定模块和列表项的具体进度
    setItemValue(moduleName: string, itemId: string, newValue: number) {
      const item = this.modules[moduleName]?.[itemId];
      if (item && newValue >= 0 && newValue <= item.max) {
        item.value = newValue;
      }
    },
    // 重置指定模块和列表项的进度
    resetItem(moduleName: string, itemId: string) {
      if (this.modules[moduleName] && this.modules[moduleName][itemId]) {
        this.modules[moduleName][itemId].value = 0;
      }
    },
    // 删除模块下的列表项
    removeItem(moduleName: string, itemId: string) {
      if (this.modules[moduleName]) {
        delete this.modules[moduleName][itemId];
      }
    },
    // 删除整个模块
    removeModule(moduleName: string) {
      delete this.modules[moduleName];
    }
  },
});
