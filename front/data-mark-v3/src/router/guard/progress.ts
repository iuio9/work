import type { Router } from "vue-router";

export function createProgressGuard(router: Router) {
  router.beforeEach((_to, _from, next) => {
    window.NProgress?.start?.();
    if (
      _to.name === "data-ano_operation" ||
      _to.name === "data-ano_detail" ||
      _to.name === "data-expansion_add" ||
      _to.name === "data-ano_imgoperate"
    ) {
      if (_from.name === "data-manage_import") {
        _to.meta.activeMenu = "data-manage_maplist";
      } else {
        if(_from.name === "data-manage_map") {
          _to.meta.activeMenu = "data-manage_maplist";
        } else {
          _to.meta.activeMenu = _from.name;
        }
      }
    }
    if (_to.name === "data-expansion_exportres") {
      if (_from.name === "data-expansion_addmap") {
        _to.meta.activeMenu = _from.meta.activeMenu;
      } else {
        _to.meta.activeMenu = _from.name;
      }
    }
    if (_to.name === "data-expansion_addmap") {
      _to.meta.activeMenu = _from.meta.activeMenu;
    }
    if(_to.name === "thirdparty_operate") {
      const { modelId } = _to.query;
      _to.meta.i18nKey = modelId ? 'route.ThirdParty_Edit': 'route.ThirdParty_Add';
    }
    next();
  });
  router.afterEach((_to) => {
    window.NProgress?.done?.();
  });
}
