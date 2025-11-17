package com.qczy.model.response;

import lombok.Data;

/**
 * 路由显示信息
 *
 */
@Data
public class MetaResponse {
    /**
     * 设置该路由在侧边栏和面包屑中展示的名字
     */
    private String title;

    /**
     * 设置该路由的图标，对应路径src/assets/icons/svg
     */
    private String icon;

    /**
     * 排序
     */
    private int order;

    private String i18nKey;

    /**
     * 是否隐藏路由，当设置 true 的时候该路由不会再侧边栏出现
     */
    private boolean hideInMenu;

    private String activeMenu;

    private String localIcon;
    private String href;


    public MetaResponse(String title, String icon,int order,String i18nKey,String activeMenu,String localIcon,boolean hideInMenu,String href) {
        this.title = title;
        this.icon = icon;
        this.order = order;
        this.i18nKey = i18nKey;
        this.hideInMenu = hideInMenu;
        this.activeMenu = activeMenu;
        this.localIcon = localIcon;
        this.href = href;
    }
}

