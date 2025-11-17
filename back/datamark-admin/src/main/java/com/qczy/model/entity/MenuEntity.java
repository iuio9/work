package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.qczy.common.base.BaseEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.Date;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 14:00
 * @Description:
 */
@Data
@TableName("qczy_menu")
public class MenuEntity extends BaseEntity {

    //自增id
    @TableId(type = IdType.AUTO)
    private Integer Id;

    /**
    * 父级菜单id 
    */
    @ApiModelProperty("父级菜单id ")
    private Integer parentId;
    /**
    * 菜单名称
    */
    @NotBlank(message="[菜单名称]不能为空")
    @Size(max= 20,message="编码长度不能超过20")
    @ApiModelProperty("菜单名称")
    @Length(max= 20,message="编码长度不能超过20")
    private String menuName;
    /**
    * 菜单图标
    */
    @ApiModelProperty("菜单图标")
    private String icon;
    /**
    * web页面路径
    */
    @ApiModelProperty("web页面路径")
    private String webPath;
    /**
    * 对应路由里面的组件component
    */
    @ApiModelProperty("对应路由里面的组件component")
    private String component;
    /**
    * 权限类型：0->目录；1->菜单；2->按钮（接口绑定权限）
    */
    @ApiModelProperty("权限类型：0->目录；1->菜单；2->按钮（接口绑定权限）")
    private Integer type;
    /**
    * 排序
    */
    @ApiModelProperty("排序")
    private Integer sort;

    //下级菜单列表
    @TableField(exist = false) //标注不是数据库的字段
    private List<MenuEntity> children;


    @TableField("i18nKey")
    private String i18nKey;

    private String activeMenu;


    private String localIcon;

    private String permissions;

    private Integer hideInMenu;

    private String href;


    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;

}
