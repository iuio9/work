package com.qczy.model.response;

import com.qczy.model.entity.ButtonPermissionEntity;
import lombok.Data;
import org.jpedal.parser.shape.S;

import java.util.List;

/**
 * 路由配置信息
 *
 */
@Data
public class RouterResponse {
    /**
     * 路由名字
     */
    private String name;

    /**
     * 路由地址
     */
    private String path;

    /**
     * 是否隐藏路由，当设置 true 的时候该路由不会再侧边栏出现
     */
   // private boolean hideInMenu;

    /**
     * 组件地址
     */
    private String component;

    /**
     * 当你一个路由下面的 children 声明的路由大于1个时，自动会变成嵌套的模式--如组件页面
     */
    //private Boolean alwaysShow;

    /**
     * 其他元素
     */
    private MetaResponse meta;

    /**
     * 子路由
     */
    private List<RouterResponse> children;

/**
 * 排序
 */
    private int order;

    private String i18nKey;

    private String activeMenu;


    private List<ButtonPermissionEntity> buttonPermission;

}
