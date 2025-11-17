package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.qczy.model.response.DictDataTreeResponse;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.hibernate.validator.constraints.Length;
import org.hibernate.validator.constraints.Range;
import org.springframework.format.annotation.DateTimeFormat;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;
import java.io.Serializable;
import java.util.Date;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 16:49
 * @Description:
 */
@Data
@TableName("qczy_dict_data")
public class DictDataEntity implements Serializable {




    @ApiModelProperty(value = "id")
    private Integer id;

    @ApiModelProperty(value = "字典类型id")
    @NotNull(message = "字典类型id不能为空")
    private Integer typeId;


    /** 父id */
    @ApiModelProperty(value = "父id")
    @NotNull(message = "父类id不能为空")
    private Integer parentId;

    @NotNull
    @ApiModelProperty(value = "排序")
    @NotNull(message = "排序不能为空")
    private Integer dictSort;

    @ApiModelProperty(value = "字典标签")
    @NotBlank(message = "字典标签不能为空")
    private String dictLabel;

    @ApiModelProperty(value = "字典键值")
   /* @NotBlank(message = "字典健值不能为空")*/
    private String dictValue;

  /*  @Range(min = 1, max = 2, message = "状态必须在1-2之间")*/
    @ApiModelProperty(value = "状态（1正常 2停用）")
    private Integer status;

    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    @ApiModelProperty(value = "更新时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Date updateTime;

    @ApiModelProperty(value = "备注")
    private String remark;


    /**
     * 子类
     */
    @TableField(exist = false)
    private List<DictDataEntity> children;



    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
